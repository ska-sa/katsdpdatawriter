#!/usr/bin/env python

"""Capture L0 visibilities from a SPEAD stream and write to a local chunk store.

This process lives across multiple observations and hence multiple data sets.
It writes weights and flags as well.

The status sensor has the following states (with typical transition events):

  - `idle`: ready to start capture
     -> ?capture-init ->
  - `wait-metadata`: waiting for telstate attributes describing L0 stream
    -> start capture thread ->
  - `wait-data`: waiting for first heap of L0 visibilities from SPEAD stream
    -> first SPEAD heap arrives ->
  - `capturing`: SPEAD data is being captured
    -> capture stops ->
  - `finalising`: metadata is being written to telstate
  - `complete`: both data and metadata capture completed
  - `error`: capture failed
    -> ?capture-done ->
  - `idle`: ready to start capture again

Objects are stored in chunks split over time and frequency but not baseline.
The chunking is chosen to produce objects with sizes on the order of 10 MB.
Objects have the following naming scheme:

  <capture_stream>/<array>/<idx1>[_<idx2>[_<idx3>]]

  - <capture_stream>: "file name"/bucket in store i.e. <capture block>_<stream>
  - <capture_block>: unique ID from capture_init (Unix timestamp at init)
  - <stream>: name of specific data product (associated with L0 SPEAD stream)
  - <array>: 'correlator_data' / 'weights' / 'flags' / etc.
  - <idxN>: chunk start index along N'th dimension

The following useful object parameters are stored in telstate:

  - <stream>_s3_endpoint_url: endpoint URL of S3 gateway to Ceph
  - <capture_stream>_chunk_name: data set name in store, i.e. <capture_stream>
  - <capture_stream>_chunk_info: {dtype, shape, chunks} dict per array
"""

from __future__ import print_function, division

import os
import threading
import logging
import Queue
import signal
import subprocess
import time

import numpy as np
import manhole
import dask.array as da
import spead2
import spead2.recv
import katsdptelstate
import katsdpservices
from katcp import DeviceServer, Sensor
from katcp.kattypes import request, return_reply, Str
from katdal.chunkstore_npy import NpyFileChunkStore
import katsdpfilewriter


def _inc_sensor(sensor, delta, status=Sensor.NOMINAL, timestamp=None):
    """Increment sensor value by `delta`."""
    sensor.set_value(sensor.value() + delta, status, timestamp)


def generate_chunks(shape, dtype, target_obj_size, dims_to_split=(0, 1)):
    """Generate dask chunk specification from ndarray parameters."""
    array_size = np.prod(shape) * np.dtype(dtype).itemsize
    num_chunks = np.ceil(array_size / target_obj_size)
    chunks = [(s,) for s in shape]
    for dim in dims_to_split:
        if dim >= len(shape):
            continue
        if num_chunks > 0.5 * shape[dim]:
            chunk_sizes = (1,) * shape[dim]
        else:
            items = np.arange(shape[dim])
            chunk_indices = np.array_split(items, np.ceil(num_chunks))
            chunk_sizes = tuple([len(chunk) for chunk in chunk_indices])
        chunks[dim] = chunk_sizes
        num_chunks /= len(chunk_sizes)
    return tuple(chunks)


class VisibilityWriterServer(DeviceServer):
    VERSION_INFO = ("sdp-vis-writer", 0, 1)
    BUILD_INFO = ('katsdpfilewriter',) + \
        tuple(katsdpfilewriter.__version__.split('.', 1)) + ('',)

    def __init__(self, logger, l0_endpoints, l0_interface, l0_ibv, l0_name, obj_store,
                 obj_size, telstate_l0, *args, **kwargs):
        super(VisibilityWriterServer, self).__init__(*args, logger=logger, **kwargs)
        self._endpoints = l0_endpoints
        self._interface_address = katsdpservices.get_interface_address(l0_interface)
        self._stream_name = l0_name
        self._ibv = l0_ibv
        self._obj_store = obj_store
        self._obj_size = obj_size
        self._telstate_l0 = telstate_l0
        self._capture_thread = None
        # Signalled when about to stop the thread.
        # Never waited for, just a thread-safe flag.
        self._stopping = threading.Event()
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
            "Number of (possibly partial) input dumps captured in this session. "
            "(prometheus: counter)",
            "", default=0)
        self.add_sensor(self._input_dumps_sensor)
        self._input_heaps_sensor = Sensor.integer(
            "input-heaps-total",
            "Number of input heaps captured in this session. (prometheus: counter)",
            "", default=0)
        self.add_sensor(self._input_heaps_sensor)
        self._input_incomplete_heaps_sensor = Sensor.integer(
            "input-incomplete-heaps-total",
            "Number of incomplete heaps that were dropped. (prometheus: counter)",
            "", default=0)
        self.add_sensor(self._input_incomplete_heaps_sensor)
        self._input_bytes_sensor = Sensor.integer(
            "input-bytes-total",
            "Number of payload bytes received in this session. (prometheus: counter)",
            "B", default=0)
        self.add_sensor(self._input_bytes_sensor)
        self._output_bytes_sensor = Sensor.integer(
            "output-bytes-total",
            "Number of payload bytes written to storage in this session. (prometheus: counter)",
            "B", default=0)
        self.add_sensor(self._output_bytes_sensor)
        self._output_chunks_sensor = Sensor.integer(
            "output-chunks-total",
            "Number of chunks written to storage in this session. (prometheus: counter)",
            "B", default=0)
        self.add_sensor(self._output_chunks_sensor)
        self._output_seconds_sensor = Sensor.float(
            "output-seconds-total",
            "Time spent on writing chunks in this session. (prometheus: counter)",
            "B", default=0)
        self.add_sensor(self._output_seconds_sensor)

    def _dump_metadata(self, n_chans, n_chans_per_substream, n_bls):
        """Generate chunk metadata for all arrays in dump."""
        chunk_info = {}
        chunks_per_dump = 0
        dtypes = {'correlator_data': np.complex64, 'flags': np.uint8,
                  'weights': np.uint8, 'weights_channel': np.float32}
        n_substreams = n_chans // n_chans_per_substream
        for array, dtype in dtypes.iteritems():
            dtype = np.dtype(dtype)
            shape = [1, n_chans_per_substream, n_bls]
            if array == 'weights_channel':
                shape = shape[:-1]
            chunks = list(generate_chunks(shape, dtype, self._obj_size))
            shape[1] = n_chans
            chunks[1] = n_substreams * chunks[1]
            chunk_info[array] = {'dtype': dtype, 'shape': tuple(shape),
                                 'chunks': tuple(chunks)}
            num_chunks = np.prod([len(c) for c in chunks])
            chunks_per_dump += num_chunks
            chunk_size = np.prod([c[0] for c in chunks]) * dtype.itemsize
            self._logger.info("Splitting array %r with shape %s and "
                              "dtype %s into %d chunk(s) of ~%d bytes each",
                              array, shape, dtype, num_chunks, chunk_size)
        return chunk_info, chunks_per_dump

    def _add_heap(self, capture_stream_name, chunk_info, heap_arrays,
                  dump_index, channel0):
        """Write a single heap to chunk store."""
        start_time = time.time()
        n_bytes = 0
        n_chunks = 0
        tfb0 = (dump_index, channel0, 0)
        for array, arr in heap_arrays.iteritems():
            # Insert time axis (will be singleton dim as heap is part of 1 dump)
            arr = arr[np.newaxis]
            chunks = list(chunk_info[array]['chunks'])
            start_channels = np.r_[0, np.cumsum(chunks[1])].tolist()
            n_chans_per_substream = arr.shape[1]
            start_chunk = start_channels.index(channel0)
            end_chunk = start_channels.index(channel0 + n_chans_per_substream)
            chunks[1] = chunks[1][start_chunk:end_chunk]
            heap_offset = tfb0[:-1] if array == 'weights_channel' else tfb0
            array_name = self._obj_store.join(capture_stream_name, array)
            for s in da.core.slices_from_chunks(chunks):
                # Adjust slices to start at the heap offset
                offset = tuple(slice(s.start + i, s.stop + i)
                               for (s, i) in zip(s, heap_offset))
                self._obj_store.put_chunk(array_name, offset, arr[s])
                n_chunks += 1
            n_bytes += arr.nbytes
        end_time = time.time()
        elapsed = end_time - start_time
        _inc_sensor(self._output_bytes_sensor, n_bytes)
        _inc_sensor(self._output_chunks_sensor, n_chunks)
        _inc_sensor(self._output_seconds_sensor, elapsed)
        self._logger.debug(
            'Wrote %.3f MB in %.3f s => %.3f MB/s (dump %d, channels starting at %d)',
            n_bytes / 1e6, elapsed, n_bytes / elapsed / 1e6, dump_index, channel0)

    def _write_final(self, capture_stream_name, heap_chunk_info, n_dumps):
        """Write final bits (mostly chunk info) after capture is done."""
        telstate_capture = self._telstate_l0.view(capture_stream_name)
        telstate_capture.add('chunk_name', capture_stream_name, immutable=True)
        full_chunk_info = {}
        for array, info in heap_chunk_info.iteritems():
            shape = list(info['shape'])
            shape[0] = n_dumps
            chunks = list(info['chunks'])
            chunks[0] = n_dumps * (1,)
            full_chunk_info[array] = {'dtype': info['dtype'],
                                      'shape': tuple(shape),
                                      'chunks': tuple(chunks)}
        telstate_capture.add('chunk_info', full_chunk_info, immutable=True)
        # Touch a token file to indicate that no more chunk data is coming
        subprocess.call('sync')
        completion_token = os.path.join(self._obj_store.path,
                                        capture_stream_name, 'complete')
        open(completion_token, 'a').close()

    def _do_capture(self, capture_stream_name, chunk_info):
        """Capture a stream from SPEAD and write to object store.

        This is run in a separate thread.

        Parameters
        ----------
        capture_stream_name : string
            "File name" of captured stream, both in chunk store and telstate
        chunk_info : dict
            Dict containing dtype / shape / chunks info per array in heap
        """
        n_dumps = 0
        n_heaps = 0
        n_bytes = 0
        n_incomplete_heaps = 0
        n_stop = 0    # Number of stop heaps received (need one per endpoint)
        self._input_dumps_sensor.set_value(n_dumps)
        self._input_heaps_sensor.set_value(n_heaps)
        self._input_bytes_sensor.set_value(n_bytes)
        self._input_incomplete_heaps_sensor.set_value(n_incomplete_heaps)
        self._output_bytes_sensor.set_value(0)
        self._output_chunks_sensor.set_value(0)
        self._output_seconds_sensor.set_value(0)
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
                            "dropped incomplete heap %d "
                            "(received %d/%d bytes of payload)",
                            heap.cnt, heap.received_length, heap.heap_length)
                        n_incomplete_heaps += 1
                        self._input_incomplete_heaps_sensor.set_value(n_incomplete_heaps)
                    updated = {}
                else:
                    updated = ig.update(heap)
                if 'timestamp' in updated:
                    vis = ig['correlator_data'].value
                    flags = ig['flags'].value
                    weights = ig['weights'].value
                    weights_channel = ig['weights_channel'].value
                    channel0 = ig['frequency'].value
                    dump_index = int(ig['dump_index'].value)
                    if dump_index >= n_dumps:
                        n_dumps = dump_index + 1
                        self._input_dumps_sensor.set_value(n_dumps)
                    heap_arrays = {'correlator_data': vis, 'flags': flags,
                                   'weights': weights,
                                   'weights_channel': weights_channel}
                    self._add_heap(capture_stream_name, chunk_info, heap_arrays,
                                   dump_index, channel0)
                    heap_nbytes = vis.nbytes + flags.nbytes
                    heap_nbytes += weights.nbytes + weights_channel.nbytes
                    n_heaps += 1
                    n_bytes += heap_nbytes
                    self._input_heaps_sensor.set_value(n_heaps)
                    self._input_bytes_sensor.set_value(n_bytes)
        except Exception as err:
            self._logger.exception(err)
            end_status = "error"
        finally:
            self._input_bytes_sensor.set_value(0)
            self._input_heaps_sensor.set_value(0)
            self._input_dumps_sensor.set_value(0)
            if not n_dumps:
                self._logger.error("Capture block contains no data")
                end_status = "error"
            else:
                self._status_sensor.set_value("finalising")
                self._write_final(capture_stream_name, chunk_info, n_dumps)
                self._logger.info('Wrote %d dumps', n_dumps)
            self._status_sensor.set_value(end_status)

    @request(Str(optional=True))
    @return_reply(Str())
    def request_capture_init(self, req, capture_block_id=None):
        """Start listening for L0 data in a separate capture thread."""
        if self._capture_thread is not None:
            self._logger.info("Ignoring capture_init because already capturing")
            return ("fail", "Already capturing")
        self._device_status_sensor.set_value("ok")
        self._status_sensor.set_value("wait-metadata")
        self._input_dumps_sensor.set_value(0)
        self._input_bytes_sensor.set_value(0)
        # Set up memory buffers, depending on size of input heaps
        try:
            n_chans = self._telstate_l0['n_chans']
            n_chans_per_substream = self._telstate_l0['n_chans_per_substream']
            n_bls = self._telstate_l0['n_bls']
        except KeyError as error:
            self._logger.error('Missing telescope state key: %s', error)
            return ("fail", "Missing telescope state key: {}".format(error))

        # 10 bytes per visibility:
        # 8 for visibility, 1 for flags, 1 for weights; plus weights_channel
        l0_heap_size = n_bls * n_chans_per_substream * 10
        l0_heap_size += n_chans_per_substream * 4
        n_substreams = n_chans // n_chans_per_substream
        chunk_info, n_chunks = self._dump_metadata(n_chans,
                                                   n_chans_per_substream, n_bls)
        # Buffer no more than 2 dumps to keep up with real time
        n_heaps_to_buffer = 2 * n_substreams
        self._rx = spead2.recv.Stream(spead2.ThreadPool(),
                                      max_heaps=2 * n_substreams,
                                      ring_heaps=n_heaps_to_buffer,
                                      contiguous_only=False)
        # We need buffers for max_heaps, ring_heaps, the current heap, and
        # the previous heap that is still in ig, plus 1 for luck.
        n_memory_buffers = n_heaps_to_buffer + 2 * n_substreams + 3
        memory_pool = spead2.MemoryPool(l0_heap_size, l0_heap_size + 4096,
                                        n_memory_buffers, n_memory_buffers)
        self._rx.set_memory_pool(memory_pool)
        self._rx.set_memcpy(spead2.MEMCPY_NONTEMPORAL)
        self._rx.stop_on_stop_item = False
        if self._ibv:
            endpoint_tuples = [(endpoint.host, endpoint.port) for endpoint in self._endpoints]
            self._rx.add_udp_ibv_reader(endpoint_tuples, self._interface_address,
                                        buffer_size=64 * 1024**2)
        else:
            for endpoint in self._endpoints:
                if self._interface_address is not None:
                    self._rx.add_udp_reader(endpoint.host, endpoint.port,
                                            buffer_size=l0_heap_size + 4096,
                                            interface_address=self._interface_address)
                else:
                    self._rx.add_udp_reader(endpoint.port, bind_hostname=endpoint.host,
                                            buffer_size=l0_heap_size + 4096)

        self._stopping.clear()
        sep = self._telstate_l0.SEPARATOR
        capture_stream_name = sep.join((capture_block_id, self._stream_name))
        self._capture_thread = threading.Thread(
            target=self._do_capture, name='capture',
            args=(capture_stream_name, chunk_info))
        self._capture_thread.start()
        self._logger.info("Starting capture to %s", capture_stream_name)
        return ("ok", "Capture initialised to {0}".format(capture_stream_name))

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
    parser.add_argument('--l0-ibv', action='store_true',
                        help='Use ibverbs acceleration to receive L0 stream [default=no]')
    parser.add_argument('--s3-endpoint-url',
                        help='URL of S3 gateway to Ceph cluster')
    parser.add_argument('--npy-path',
                        help='Write NPY files to this directory instead of '
                             'directly to object store')
    parser.add_argument('--obj-size-mb', type=float, default=10., metavar='MB',
                        help='Target object size in MB [default=%(default)s]')
    parser.add_argument('-p', '--port', type=int, default=2046, metavar='N',
                        help='KATCP host port [default=%(default)s]')
    parser.add_argument('-a', '--host', default="", metavar='HOST',
                        help='KATCP host address [default=all hosts]')
    parser.set_defaults(telstate='localhost')
    args = parser.parse_args()
    if not args.npy_path:
        parser.error('--npy-path is required')
    if args.l0_ibv and args.l0_interface is None:
        parser.error('--l0-ibv requires --l0-interface')

    # Connect to object store and save config in telstate
    obj_store = NpyFileChunkStore(args.npy_path)
    telstate_l0 = args.telstate.view(args.l0_name)
    if args.s3_endpoint_url:
        telstate_l0.add('s3_endpoint_url', args.s3_endpoint_url, immutable=True)
    restart_queue = Queue.Queue()
    server = VisibilityWriterServer(logger, args.l0_spead, args.l0_interface, args.l0_ibv,
                                    args.l0_name, obj_store,
                                    args.obj_size_mb * 1e6, telstate_l0,
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
