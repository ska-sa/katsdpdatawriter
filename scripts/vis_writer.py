#!/usr/bin/env python

"""Capture L0 visibilities from a SPEAD stream and write to HDF5 file. When
the file is closed, metadata is also extracted from the telescope state and
written to the file. This process lives across multiple observations and
hence multiple HDF5 files.

The status sensor has the following states:

  - `idle`: data is not being captured
  - `capturing`: data is being captured
  - `ready`: CBF data stream has finished, waiting for capture_done request
  - `finalising`: metadata is being written to file
"""

from __future__ import print_function, division
import spead2
import spead2.recv
import katsdptelstate
import time
import os.path
import os
import sys
import threading
import logging
import Queue
import numpy as np
import signal
import manhole
import netifaces
import concurrent.futures
from katcp import DeviceServer, Sensor
from katcp.kattypes import request, return_reply, Str
import katsdpservices
from katsdpfilewriter import telescope_model, ar1_model, file_writer


#: Bytes free at which a running capture will be stopped
FREE_DISK_THRESHOLD_STOP = 2 * 1024**3
#: Bytes free at which a new capture will be refused
FREE_DISK_THRESHOLD_START = 3 * 1024**3


class FileWriterServer(DeviceServer):
    VERSION_INFO = ("sdp-file-writer", 0, 1)
    BUILD_INFO = ("sdp-file-writer", 0, 1, "rc1")

    def __init__(self, logger, l0_endpoints, l0_interface, l0_name,
                 file_base, telstate, *args, **kwargs):
        super(FileWriterServer, self).__init__(*args, logger=logger, **kwargs)
        self._file_base = file_base
        self._endpoints = l0_endpoints
        self._stream_name = l0_name
        self._interface_address = katsdpservices.get_interface_address(l0_interface)
        self._capture_thread = None
        #: Signalled when about to stop the thread. Never waited for, just a thread-safe flag.
        self._stopping = threading.Event()
        self._telstate_l0 = telstate.view(l0_name)
        self._model = ar1_model.create_model(antenna_mask=self.get_antenna_mask())
        self._file_obj = None
        self._start_timestamp = None
        self._rx = None

    def get_antenna_mask(self):
        """Extract list of antennas from baseline list"""
        antennas = set()
        bls_ordering = self._telstate_l0['bls_ordering']
        for a, b in bls_ordering:
            antennas.add(a[:-1])
            antennas.add(b[:-1])
        return sorted(antennas)

    def setup_sensors(self):
        self._status_sensor = Sensor.string(
                "status", "The current status of the capture process", "", "idle")
        self.add_sensor(self._status_sensor)
        self._device_status_sensor = Sensor.string(
                "device-status", "Health sensor", "", "ok")
        self.add_sensor(self._device_status_sensor)
        self._filename_sensor = Sensor.string(
                "filename", "Final name for file being captured", "")
        self.add_sensor(self._filename_sensor)
        self._input_dumps_sensor = Sensor.integer(
                "input-dumps-total", "Number of (possibly partial) input dumps captured in this session.", "", default=0)
        self.add_sensor(self._input_dumps_sensor)
        self._input_heaps_sensor = Sensor.integer(
                "input-heaps-total", "Number of input heaps captured in this session.", "", default=0)
        self.add_sensor(self._input_heaps_sensor)
        self._input_incomplete_heaps_sensor = Sensor.integer(
                "input-incomplete-heaps-total", "Number of incomplete heaps that were dropped.", "", default=0)
        self.add_sensor(self._input_incomplete_heaps_sensor)
        self._input_bytes_sensor = Sensor.integer(
                "input-bytes-total", "Number of payload bytes received in this session.", "B", default=0)
        self.add_sensor(self._input_bytes_sensor)
        self._disk_free_sensor = Sensor.float(
                "disk-free", "Free disk space in bytes on target device for this file.", "B")
        self.add_sensor(self._disk_free_sensor)

    def _do_capture(self, file_obj):
        """Capture a stream from SPEAD and write to file. This is run in a
        separate thread.

        Parameters
        ----------
        file_obj : :class:`filewriter.File`
            Output file object
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
        loop_time = time.time()
        free_space = file_obj.free_space()
        self._disk_free_sensor.set_value(free_space)
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
                    if not timestamps or timestamp >= timestamps[-1]:
                        if not timestamps or timestamp != timestamps[-1]:
                            timestamps.append(timestamp)
                            n_dumps += 1
                            self._input_dumps_sensor.set_value(n_dumps)
                        time_idx = len(timestamps) - 1
                        file_obj.add_data_heap(vis_data, flags, weights, weights_channel,
                                               time_idx, channel0)
                        n_heaps += 1
                        n_bytes += vis_data.nbytes + flags.nbytes + weights.nbytes \
                            + weights_channel.nbytes
                        self._input_heaps_sensor.set_value(n_heaps)
                        self._input_bytes_sensor.set_value(n_bytes)
                    else:
                        self._logger.warning(
                            'Received timestamp from the past, discarding (%s < %s)',
                            timestamp, timestamps[-1])
                free_space = file_obj.free_space()
                self._disk_free_sensor.set_value(free_space)
                if free_space < FREE_DISK_THRESHOLD_STOP:
                    self._logger.error('Stopping capture because only %d bytes left on disk',
                                       free_space)
                    self._rx.stop()
                    end_status = "disk-full"
                    self._device_status_sensor.set_value("fail", "error")
        except Exception as err:
            self._logger.error(err)
            end_status = "error"
        finally:
            self._status_sensor.set_value(end_status)
            self._input_bytes_sensor.set_value(0)
            self._input_heaps_sensor.set_value(0)
            self._input_dumps_sensor.set_value(0)
            # Timestamps in the SPEAD stream are relative to sync_time
            if not timestamps:
                self._logger.warning("H5 file contains no data and hence no timestamps")
            else:
                timestamps = np.array(timestamps) + self._telstate_l0['sync_time']
                file_obj.set_timestamps(timestamps)
                self._logger.info('Set %d timestamps', len(timestamps))

    @request(Str(optional=True))
    @return_reply(Str())
    def request_capture_init(self, req, capture_block_id=None):
        """Start listening for L0 data and write it to HDF5 file."""
        if self._capture_thread is not None:
            self._logger.info("Ignoring capture_init because already capturing")
            return ("fail", "Already capturing")
        timestamp = time.time()
        self._final_filename = os.path.join(
                self._file_base, "{0}.h5".format(int(timestamp)))
        self._stage_filename = os.path.join(
                self._file_base, "{0}.writing.h5".format(int(timestamp)))
        try:
            stat = os.statvfs(os.path.dirname(self._stage_filename))
        except OSError:
            self._logger.warn("Failed to check free disk space, continuing anyway")
        else:
            free_space = stat.f_bsize * stat.f_bavail
            if free_space < FREE_DISK_THRESHOLD_START:
                self._logger.error("Insufficient disk space to start capture (%d < %d)",
                                  free_space, FREE_DISK_THRESHOLD_START)
                self._device_status_sensor.set_value("fail", "error")
                return ("fail", "Disk too full (only {:.2f} GiB free)".format(free_space / 1024**3))
        self._device_status_sensor.set_value("ok")
        self._filename_sensor.set_value(self._final_filename)
        self._status_sensor.set_value("wait-metadata")
        self._input_dumps_sensor.set_value(0)
        self._input_bytes_sensor.set_value(0)
        self._start_timestamp = timestamp
        # Set up memory buffers, depending on size of input heaps
        try:
            n_chans = self._telstate_l0['n_chans']
            n_chans_per_substream = self._telstate_l0['n_chans_per_substream']
            n_bls = self._telstate_l0['n_bls']
        except KeyError as error:
            self._logger.error('Missing telescope state key: %s', error)
            end_status = 'bad-telstate'
            return ("fail", "Missing telescope state key: {}".format(error))
        self._file_obj = file_writer.File(self._stage_filename, self._stream_name)
        self._file_obj.create_data((n_chans, n_bls))

        # 10 bytes per visibility: 8 for visibility, 1 for flags, 1 for weights; plus weights_channel
        l0_heap_size = n_bls * n_chans_per_substream * 10 + n_chans_per_substream * 4
        n_substreams = n_chans // n_chans_per_substream
        self._rx = spead2.recv.Stream(spead2.ThreadPool(),
                                      max_heaps=2 * n_substreams, ring_heaps=2 * n_substreams,
                                      contiguous_only=False)
        memory_pool = spead2.MemoryPool(l0_heap_size, l0_heap_size+4096, 8 * n_substreams, 8 * n_substreams)
        self._rx.set_memory_pool(memory_pool)
        self._rx.stop_on_stop_item = False
        for endpoint in self._endpoints:
            if self._interface_address is not None:
                self._rx.add_udp_reader(endpoint.host, endpoint.port,
                                        buffer_size=l0_heap_size+4096,
                                        interface_address=self._interface_address)
            else:
                self._rx.add_udp_reader(endpoint.port, bind_hostname=endpoint.host,
                                        buffer_size=l0_heap_size+4096)

        self._stopping.clear()
        self._capture_thread = threading.Thread(
                target=self._do_capture, name='capture',
                args=(self._file_obj,))
        self._capture_thread.start()
        self._logger.info("Starting capture to %s", self._stage_filename)
        return ("ok", "Capture initialised to {0}".format(self._stage_filename))

    @request()
    @return_reply(Str())
    def request_capture_done(self, req):
        """Stop capturing and close the HDF5 file, if it is not already done."""
        if self._capture_thread is None:
            self._logger.info("Ignoring capture_done because already explicitly stopped")
        return self.capture_done()

    def capture_done(self):
        """Implementation of :meth:`request_capture_done`, split out to allow it
        to be called on `KeyboardInterrupt`.
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
        self._file_obj.set_metadata(telescope_model.TelstateModelData(
                self._model, self._telstate_l0.root(), self._start_timestamp))
        self._file_obj.close()
        self._file_obj = None
        self._start_timestamp = None
        self._logger.info("Finalised file")

        # File is now closed, so rename it
        try:
            os.rename(self._stage_filename, self._final_filename)
            result = ("ok", "File renamed to {0}".format(self._final_filename))
        except OSError as e:
            logger.error("Failed to rename output file %s to %s",
                         self._stage_filename, self._final_filename, exc_info=True)
            result = ("fail", "Failed to rename output file from {0} to {1}.".format(
                self._stage_filename, self._final_filename))
        self._status_sensor.set_value("idle")
        return result

def comma_list(type_):
    """Return a function which splits a string on commas and converts each element to
    `type_`."""

    def convert(arg):
        return [type_(x) for x in arg.split(',')]
    return convert

def main():
    katsdpservices.setup_logging()
    logger = logging.getLogger("katsdpfilewriter")
    logging.getLogger('spead2').setLevel(logging.WARNING)
    katsdpservices.setup_restart()

    parser = katsdpservices.ArgumentParser()
    parser.add_argument('--l0-spead', type=katsdptelstate.endpoint.endpoint_list_parser(7200), default=':7200', help='source port/multicast groups for spectral L0 input. [default=%(default)s]', metavar='ENDPOINTS')
    parser.add_argument('--l0-interface', help='interface to subscribe to for L0 data. [default=auto]', metavar='INTERFACE')
    parser.add_argument('--l0-name', default='sdp_l0', help='telstate prefix for L0 metadata. [default=%(default)s]', metavar='NAME')
    parser.add_argument('--file-base', default='.', type=str, help='base directory into which to write HDF5 files. [default=%(default)s]', metavar='DIR')
    parser.add_argument('-p', '--port', dest='port', type=int, default=2046, metavar='N', help='katcp host port. [default=%(default)s]')
    parser.add_argument('-a', '--host', dest='host', type=str, default="", metavar='HOST', help='katcp host address. [default=all hosts]')
    parser.set_defaults(telstate='localhost')
    args = parser.parse_args()
    if not os.access(args.file_base, os.W_OK):
        logger.error('Target directory (%s) is not writable', args.file_base)
        sys.exit(1)

    restart_queue = Queue.Queue()
    server = FileWriterServer(logger, args.l0_spead, args.l0_interface, args.l0_name,
                              args.file_base, args.telstate,
                              host=args.host, port=args.port)
    server.set_restart_queue(restart_queue)
    server.start()
    logger.info("Started file writer server.")


    manhole.install(oneshot_on='USR1', locals={'server':server, 'args':args})
     # allow remote debug connections and expose server and args

    def graceful_exit(_signo=None, _stack_frame=None):
        logger.info("Exiting filewriter on SIGTERM")
        os.kill(os.getpid(), signal.SIGINT)
         # rely on the interrupt handler around the katcp device server
         # to peform graceful shutdown. this preserves the command
         # line Ctrl-C shutdown.

    signal.signal(signal.SIGTERM, graceful_exit)
     # mostly needed for Docker use since this process runs as PID 1
     # and does not get passed sigterm unless it has a custom listener

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
        logger.info("Shutting down file_writer server...")
        logger.info("Activity logging stopped")
        server.capture_done()
        server.stop()
        server.join()

if __name__ == '__main__':
    main()
