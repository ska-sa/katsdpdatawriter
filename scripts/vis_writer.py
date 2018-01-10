#!/usr/bin/env python

"""Capture L0 visibilities from a SPEAD stream and write to Ceph object store.

This process lives across multiple observations and hence multiple data sets.
It writes weights, flags and timestamps as well.

The status sensor has the following states:

  - `idle`: data is not being captured
  - `capturing`: data is being captured
  - `ready`: CBF data stream has finished, waiting for capture_done request
  - `finalising`: metadata is being written to file

Objects are stored in chunks split over time and frequency but not baseline.
The chunking is chosen to produce objects with sizes on the order of 2 MB.
Objects have the following naming scheme:

  <obj_base_name>/<capture_block>/<stream>/<dataset>/<idx1>[_<idx2>[_<idx3>]]

  - <obj_base_name>: top-level name (telescope? project? defaults to 'MKAT')
  - <capture_block>: globally unique ID passed to capture_init (observation?)
  - <stream>: name of specific data product (associated with L0 SPEAD stream)
  - <dataset>: 'correlator_data' / 'weights' / 'flags' / etc.
  - <idxN>: chunk start index along N'th dimension

The following useful object parameters are stored in telstate:

  - <stream>_ceph_conf: copy of ceph.conf used to connect to target Ceph cluster
  - <stream>_ceph_pool: the name of the Ceph pool used
  - <capture_block>_<stream>_<dataset>: chunk info dict (dtype, shape, chunks)
"""

from __future__ import print_function, division

import time
import os.path
import os
import threading
import logging
import Queue
import signal
from itertools import product

import numpy as np
import manhole
import spead2
import spead2.recv
import katsdptelstate
import katsdpservices
from katcp import DeviceServer, Sensor
from katcp.kattypes import request, return_reply, Str
from katdal.chunkstore_rados import RadosChunkStore
import dask
import dask.array as da


def generate_chunks(shape, dtype, target_obj_size, dims_to_split=(0, 1)):
    """Generate dask chunk specification from ndarray parameters."""
    dataset_size = np.prod(shape) * np.dtype(dtype).itemsize
    num_chunks = np.ceil(dataset_size / float(target_obj_size))
    chunks = [(s,) for s in shape]
    for dim in dims_to_split:
        if dim >= len(shape):
            continue
        if num_chunks > 0.5 * shape[dim]:
            chunk_sizes = (1,) * shape[dim]
        else:
            items = np.arange(shape[dim])
            chunk_indices = np.array_split(items, num_chunks)
            chunk_sizes = tuple([len(chunk) for chunk in chunk_indices])
        chunks[dim] = chunk_sizes
        num_chunks = np.ceil(num_chunks / len(chunk_sizes))
    return tuple(chunks)


def dsk_from_chunks(chunks, out_name):
    """"Turn chunk spec into slices spec and keys suitable for dask Arrays."""
    keys = list(product([out_name], *[range(len(bds)) for bds in chunks]))
    slices = da.core.slices_from_chunks(chunks)
    return zip(keys, slices)


class VisibilityWriterServer(DeviceServer):
    VERSION_INFO = ("sdp-vis-writer", 0, 1)
    BUILD_INFO = ("sdp-vis-writer", 0, 1, "rc1")

    def __init__(self, logger, l0_endpoints, l0_interface, l0_name,
                 obj_store, obj_base_name, obj_size, telstate, *args, **kwargs):
        super(VisibilityWriterServer, self).__init__(*args, logger=logger, **kwargs)
        self._endpoints = l0_endpoints
        self._interface_address = katsdpservices.get_interface_address(l0_interface)
        self._stream_name = l0_name
        self._obj_store = obj_store
        self._obj_base_name = obj_base_name
        self._obj_size = obj_size
        self._telstate_l0 = telstate
        self._capture_thread = None
        #: Signalled when about to stop the thread. Never waited for, just a thread-safe flag.
        self._stopping = threading.Event()
        self._start_timestamp = None
        self._int_time = None
        self._sync_time = None
        self._rx = None

    def setup_sensors(self):
        self._status_sensor = Sensor.string(
            "status", "The current status of the capture process", "", "idle")
        self.add_sensor(self._status_sensor)
        self._device_status_sensor = Sensor.string(
            "device-status", "Health sensor", "", "ok")
        self.add_sensor(self._device_status_sensor)
        self._input_dumps_sensor = Sensor.integer(
            "input-dumps-total",
            "Number of (possibly partial) input dumps captured in this session.",
            "", default=0)
        self.add_sensor(self._input_dumps_sensor)
        self._input_heaps_sensor = Sensor.integer(
            "input-heaps-total",
            "Number of input heaps captured in this session.", "", default=0)
        self.add_sensor(self._input_heaps_sensor)
        self._input_incomplete_heaps_sensor = Sensor.integer(
            "input-incomplete-heaps-total",
            "Number of incomplete heaps that were dropped.", "", default=0)
        self.add_sensor(self._input_incomplete_heaps_sensor)
        self._input_bytes_sensor = Sensor.integer(
            "input-bytes-total",
            "Number of payload bytes received in this session.", "B", default=0)
        self.add_sensor(self._input_bytes_sensor)

    def _heap_metadata(self, n_chans, n_chans_per_substream, n_bls):
        """Generate chunk metadata for all datasets in heap."""
        target_obj_size = self._obj_size * 2 ** 20
        chunk_info = {}
        dtypes = {'correlator_data': np.complex64, 'flags': np.uint8,
                  'weights': np.uint8, 'weights_channel': np.float32}
        n_substreams = n_chans // n_chans_per_substream
        for dataset, dtype in dtypes.iteritems():
            dtype = np.dtype(dtype)
            shape = [1, n_chans_per_substream, n_bls]
            if dataset == 'weights_channel':
                shape = shape[:-1]
            chunks = list(generate_chunks(shape, dtype, target_obj_size))
            shape[1] = n_chans
            chunks[1] = n_substreams * chunks[1]
            chunk_info[dataset] = {'dtype': dtype, 'shape': tuple(shape),
                                   'chunks': tuple(chunks)}
            num_chunks = np.prod([len(c) for c in chunks])
            chunk_size = np.prod([c[0] for c in chunks]) * dtype.itemsize
            self._logger.info("Splitting dataset %r with shape %s and "
                              "dtype %s into %d chunk(s) of ~%d bytes each",
                              dataset, shape, dtype, num_chunks, chunk_size)
        return chunk_info

    def _write_heap(self, obj_stream_name, chunk_info, vis_data, flags,
                    weights, weights_channel, dump_index, channel0):
        """"Write a single heap to the object store."""
        dask_graph = {}
        output_keys = []
        schedule = dask.threaded.get
        heap = {'correlator_data': vis_data, 'flags': flags,
                'weights': weights, 'weights_channel': weights_channel}
        tfb0 = (dump_index, channel0, 0)
        for dataset, arr in heap.iteritems():
            arr = arr[np.newaxis]
            chunks = list(chunk_info[dataset]['chunks'])
            start_channels = np.r_[0, np.cumsum(chunks[1])].tolist()
            n_chans_per_substream = arr.shape[1]
            start_chunk = start_channels.index(channel0)
            end_chunk = start_channels.index(channel0 + n_chans_per_substream)
            chunks[1] = chunks[1][start_chunk:end_chunk]
            heap_offset = tfb0[:-1] if dataset == 'weights_channel' else tfb0
            offset_slice = lambda s: tuple(slice(s.start + i, s.stop + i)
                                           for (s, i) in zip(s, heap_offset))
            shape_slice = lambda slices: tuple(s.stop - s.start for s in slices)
            array_name = self._obj_store.join(obj_stream_name, dataset)
            dsk = {k + heap_offset:
                   (self._obj_store.put, array_name, offset_slice(s),
                    arr[s].reshape(shape_slice(s)))
                   for k, s in dsk_from_chunks(chunks, dataset)}
            dask_graph.update(dsk)
            output_keys.extend(dsk.keys())
        schedule(dask_graph, output_keys)
        self._logger.info('Wrote %s dump %d, channels starting at %d',
                          obj_stream_name, dump_index, channel0)

    def _write_final(self, obj_stream_name, chunk_info, timestamps):
        """Write final bits after capture is done (timestamps + chunk info)."""
        array_name = self._obj_store.join(obj_stream_name, 'timestamps')
        n_dumps = len(timestamps)
        slices = (slice(0, n_dumps),)
        self._obj_store.put(array_name, slices, timestamps)
        capture_block_id = self._obj_store.split(obj_stream_name)[-2]
        capture_name = self._telstate_l0.SEPARATOR.join((capture_block_id,
                                                         self._stream_name))
        telstate_capture = self._telstate_l0.view(capture_name)
        dask_info = {'dtype': np.dtype(np.float), 'shape': (n_dumps,),
                     'chunks': ((n_dumps,),)}
        telstate_capture.add('timestamps', dask_info, immutable=True)
        for dataset in chunk_info:
            dtype = chunk_info[dataset]['dtype']
            shape = list(chunk_info[dataset]['shape'])
            shape[0] = n_dumps
            chunks = list(chunk_info[dataset]['chunks'])
            chunks[0] = n_dumps * (1,)
            dask_info = {'dtype': dtype, 'shape': tuple(shape),
                         'chunks': tuple(chunks)}
            telstate_capture.add(dataset, dask_info, immutable=True)

    def _do_capture(self, obj_stream_name, chunk_info):
        """Capture a stream from SPEAD and write to object store.

        This is run in a separate thread.

        Parameters
        ----------
        obj_stream_name : string
            Prefix of all object keys associated with captured stream
        chunk_info : dict
            Dict containing dtype / shape / chunks info per dataset in heap
        """
        timestamps = []
        n_dumps = 0
        n_heaps = 0
        n_bytes = 0
        n_incomplete_heaps = 0
        n_stop = 0    # Number of stop heaps received (need one per endpoint)
        self._input_dumps_sensor.set_value(n_dumps)
        self._input_heaps_sensor.set_value(n_heaps)
        self._input_bytes_sensor.set_value(n_bytes)
        self._input_incomplete_heaps_sensor.set_value(n_incomplete_heaps)
        # status to report once the capture stops
        end_status = "complete"

        try:
            self._status_sensor.set_value("wait-data")
            self._logger.info('Waiting for data')
            ig = spead2.ItemGroup()
            first = True
            for heap in self._rx:
                if first:
                    self._logger.info('First heap received')
                    self._status_sensor.set_value("capturing")
                    first = False
                if heap.is_end_of_stream():
                    n_stop += 1
                    if n_stop == len(self._endpoints):
                        self._rx.stop()
                        break
                    else:
                        continue
                if isinstance(heap, spead2.recv.IncompleteHeap):
                    # Don't warn if we've already been asked to stop. There may
                    # be some heaps still in the network at the time we were
                    # asked to stop.
                    if not self._stopping.is_set():
                        self._logger.warning(
                            "dropped incomplete heap %d (%d/%d bytes of payload)",
                            heap.cnt, heap.received_length, heap.heap_length)
                        n_incomplete_heaps += 1
                        self._input_incomplete_heaps_sensor.set_value(n_incomplete_heaps)
                    updated = {}
                else:
                    updated = ig.update(heap)
                if 'timestamp' in updated:
                    vis_data = ig['correlator_data'].value
                    flags = ig['flags'].value
                    weights = ig['weights'].value
                    weights_channel = ig['weights_channel'].value
                    channel0 = ig['frequency'].value
                    timestamp = ig['timestamp'].value
                    try:
                        dump_index = ig['dump_index'].value
                    except KeyError:
                        # Attempt to synthesise dump index from timestamp
                        t0 = timestamps[0] if timestamps else timestamp
                        dump_index = int(round((timestamp - t0) / self._int_time))
                    if not timestamps or timestamp >= timestamps[-1]:
                        if not timestamps or timestamp != timestamps[-1]:
                            # Fill in all missing timestamps since last dump too
                            for n in reversed(range(dump_index + 1 - n_dumps)):
                                timestamps.append(timestamp - n * self._int_time)
                            n_dumps = dump_index + 1
                            self._input_dumps_sensor.set_value(n_dumps)
                        self._write_heap(obj_stream_name, chunk_info, vis_data,
                                         flags, weights, weights_channel,
                                         dump_index, channel0)
                        n_heaps += 1
                        n_bytes += vis_data.nbytes + flags.nbytes
                        n_bytes += weights.nbytes + weights_channel.nbytes
                        self._input_heaps_sensor.set_value(n_heaps)
                        self._input_bytes_sensor.set_value(n_bytes)
                    else:
                        self._logger.warning(
                            'Received timestamp from the past, discarding (%s < %s)',
                            timestamp, timestamps[-1])
        except Exception as err:
            self._logger.exception(err)
            end_status = "error"
        finally:
            self._status_sensor.set_value(end_status)
            self._input_bytes_sensor.set_value(0)
            self._input_heaps_sensor.set_value(0)
            self._input_dumps_sensor.set_value(0)
            # Timestamps in the SPEAD stream are relative to sync_time
            if not timestamps:
                self._logger.warning("Capture block contains no data and hence no timestamps")
            else:
                timestamps = np.array(timestamps) + self._sync_time
                self._write_final(obj_stream_name, chunk_info, timestamps)
                self._logger.info('Wrote %d timestamps', len(timestamps))

    @request(Str(optional=True))
    @return_reply(Str())
    def request_capture_init(self, req, capture_block_id=None):
        """Start listening for L0 data in a separate capture thread."""
        if self._capture_thread is not None:
            self._logger.info("Ignoring capture_init because already capturing")
            return ("fail", "Already capturing")
        timestamp = time.time()
        self._device_status_sensor.set_value("ok")
        self._status_sensor.set_value("wait-metadata")
        self._input_dumps_sensor.set_value(0)
        self._input_bytes_sensor.set_value(0)
        self._start_timestamp = timestamp
        # Set up memory buffers, depending on size of input heaps
        try:
            n_chans = self._telstate_l0['n_chans']
            n_chans_per_substream = self._telstate_l0['n_chans_per_substream']
            n_bls = self._telstate_l0['n_bls']
            # These are needed for timestamps - crash early if not available
            self._int_time = self._telstate_l0['int_time']
            self._sync_time = self._telstate_l0['sync_time']
        except KeyError as error:
            self._logger.error('Missing telescope state key: %s', error)
            return ("fail", "Missing telescope state key: {}".format(error))

        # 10 bytes per visibility: 8 for visibility, 1 for flags, 1 for weights; plus weights_channel
        l0_heap_size = n_bls * n_chans_per_substream * 10 + n_chans_per_substream * 4
        n_substreams = n_chans // n_chans_per_substream
        self._rx = spead2.recv.Stream(spead2.ThreadPool(),
                                      max_heaps=2 * n_substreams,
                                      ring_heaps=2 * n_substreams,
                                      contiguous_only=False)
        memory_pool = spead2.MemoryPool(l0_heap_size, l0_heap_size + 4096,
                                        8 * n_substreams, 8 * n_substreams)
        self._rx.set_memory_pool(memory_pool)
        self._rx.stop_on_stop_item = False
        for endpoint in self._endpoints:
            if self._interface_address is not None:
                self._rx.add_udp_reader(endpoint.host, endpoint.port,
                                        buffer_size=l0_heap_size + 4096,
                                        interface_address=self._interface_address)
            else:
                self._rx.add_udp_reader(endpoint.port, bind_hostname=endpoint.host,
                                        buffer_size=l0_heap_size + 4096)

        self._stopping.clear()
        obj_stream_name = self._obj_store.join(
            self._obj_base_name, capture_block_id, self._stream_name)
        chunk_info = self._heap_metadata(n_chans, n_chans_per_substream, n_bls)
        self._capture_thread = threading.Thread(
            target=self._do_capture, name='capture',
            args=(obj_stream_name, chunk_info))
        self._capture_thread.start()
        self._logger.info("Starting capture to %s", obj_stream_name)
        return ("ok", "Capture initialised to {0}".format(obj_stream_name))

    @request()
    @return_reply(Str())
    def request_capture_done(self, req):
        """Stop capturing, which cleans up the capturing thread."""
        if self._capture_thread is None:
            self._logger.info("Ignoring capture_done because already explicitly stopped")
        return self.capture_done()

    def capture_done(self):
        """Implementation of :meth:`request_capture_done`.

        This is split out to allow it to be called on `KeyboardInterrupt`.
        """
        if self._capture_thread is None:
            return ("fail", "Not capturing")
        self._logger.info("Waiting for capture thread (5s timeout)")
        self._capture_thread.join(timeout=5)
        self._stopping.set()   # Prevents warnings about incomplete heaps as the stop occurs
        if self._capture_thread.isAlive():  # The join timed out
            self._logger.info("Stopping receiver")
            self._rx.stop()
            self._logger.info("Waiting for capture thread")
            self._capture_thread.join()
        self._capture_thread = None
        self._rx = None
        self._logger.info("Joined capture thread")
        self._status_sensor.set_value("finalising")
        self._start_timestamp = None
        self._logger.info("Finalised capture")
        self._status_sensor.set_value("idle")
        return ("ok", "Capture done")


if __name__ == '__main__':
    katsdpservices.setup_logging()
    logger = logging.getLogger("katsdpfilewriter")
    logging.getLogger('spead2').setLevel(logging.WARNING)
    katsdpservices.setup_restart()

    parser = katsdpservices.ArgumentParser()
    parser.add_argument('--l0-spead', default=':7200', metavar='ENDPOINTS',
                        type=katsdptelstate.endpoint.endpoint_list_parser(7200),
                        help='Source port/multicast groups for L0 SPEAD stream. '
                             '[default=%(default)s]')
    parser.add_argument('--l0-interface', metavar='INTERFACE',
                        help='Network interface to subscribe to for L0 stream. '
                             '[default=auto]')
    parser.add_argument('--l0-name', default='sdp_l0', metavar='NAME',
                        help='Name of L0 stream from ingest [default=%(default)s]')
    parser.add_argument('--ceph-conf', default="/etc/ceph/ceph.conf", metavar='CONF',
                        help='Ceph configuration file [default=%(default)s]')
    parser.add_argument('--ceph-pool', default='data_vis', metavar='POOL',
                        help='Name of Ceph pool [default=%(default)s]')
    parser.add_argument('--ceph-keyring',
                        help='Ceph keyring filename (optional)')
    parser.add_argument('--obj-base-name', default='MKAT', metavar='NAME',
                        help='Base name for objects in store [default=%(default)s]')
    parser.add_argument('--obj-size', type=float, default=2.0, metavar='MB',
                        help='Target object size in MB [default=%(default)s]')
    parser.add_argument('-p', '--port', type=int, default=2046, metavar='N',
                        help='KATCP host port [default=%(default)s]')
    parser.add_argument('-a', '--host', default="", metavar='HOST',
                        help='KATCP host address [default=all hosts]')
    parser.set_defaults(telstate='localhost')
    args = parser.parse_args()

    # Connect to object store and save config in telstate
    obj_store = RadosChunkStore.from_config(args.ceph_conf, args.ceph_pool,
                                            args.ceph_keyring)
    telstate_l0 = args.telstate.view(args.l0_name)
    with open(args.ceph_conf, 'r') as ceph_conf:
        telstate_l0.add('ceph_conf', ceph_conf.readlines(), immutable=True)
    telstate_l0.add('ceph_pool', args.ceph_pool, immutable=True)

    restart_queue = Queue.Queue()
    server = VisibilityWriterServer(logger, args.l0_spead,
                                    args.l0_interface, args.l0_name,
                                    obj_store, args.obj_base_name,
                                    args.obj_size, telstate_l0,
                                    host=args.host, port=args.port)
    server.set_restart_queue(restart_queue)
    server.start()
    logger.info("Started visibility writer server.")

    # allow remote debug connections and expose server and args
    manhole.install(oneshot_on='USR1', locals={'server': server, 'args': args})

    def graceful_exit(_signo=None, _stack_frame=None):
        logger.info("Exiting vis_writer on SIGTERM")
        # rely on the interrupt handler around the katcp device server
        # to peform graceful shutdown. this preserves the command
        # line Ctrl-C shutdown.
        os.kill(os.getpid(), signal.SIGINT)

    # mostly needed for Docker use since this process runs as PID 1
    # and does not get passed sigterm unless it has a custom listener
    signal.signal(signal.SIGTERM, graceful_exit)

    try:
        while True:
            try:
                device = restart_queue.get(timeout=0.5)
            except Queue.Empty:
                device = None
            if device is not None:
                logger.info("Stopping")
                device.capture_done()
                device.stop()
                device.join()
                logger.info("Restarting")
                device.start()
                logger.info("Started")
    except KeyboardInterrupt:
        logger.info("Shutting down vis_writer server...")
        logger.info("Activity logging stopped")
        server.capture_done()
        server.stop()
        server.join()
