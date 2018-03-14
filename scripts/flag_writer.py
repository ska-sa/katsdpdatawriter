#!/usr/bin/env python3

"""Capture L1 flags from the SPEAD stream(s) produced by cal.

We adopt a naive strategy and store the flags for each dump in a single
object. These objects will be later picked up by the trawler process
and inserted into the appropriate bucket in S3 from where they will be
picked up by katdal.

"""

import os
import logging
import signal
import enum
import json
import sys
import asyncio
import time
from collections import defaultdict

import numpy as np
import spead2
import spead2.recv.asyncio
import katsdptelstate
import katsdpservices
from aiokatcp import DeviceServer, Sensor, FailReply
import katsdpflagwriter

# Number of partial flag heaps to cache. Once full, heaps
# will be discarded oldest arrival time first.
FLAG_CACHE_SIZE = 10

class Status(enum.Enum):
    IDLE = 1
    WAIT_DATA = 2
    CAPTURING = 3
    FINISHED = 4


class State(enum.Enum):
    """State of a single capture block"""
    CAPTURING = 1         # capture-init has been called, but not capture-done
    COMPLETE = 2          # capture-done has been called


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, enum.Enum):
            return obj.name
        return json.JSONEncoder.default(self, obj)


class FlagItem():
    def __init__(self, shape_tuple, completed_fragment_count):
        self._flags = np.empty(shape_tuple, dtype=np.int8)
        # Init with no data flag
        self._flags[:] = 0b00001000
        self._fragment_offsets = []
        self._completed_fragment_count = completed_fragment_count
        self.started = time.time()

    def is_complete(self):
        return len(self._fragment_offsets) >= self._completed_fragment_count

    def add_fragment(self, flag_fragment, offset):
        self._flags[offset:offset + flag_fragment.shape[0]] = flag_fragment
        self._fragment_offsets.append(offset // flag_fragment.shape[0])
        return self.is_complete()


def _warn_if_positive(value):
    return Sensor.Status.WARN if value > 0 else Sensor.Status.NOMINAL


class FlagWriterServer(DeviceServer):
    VERSION = "sdp-flag-writer-0.1"
    BUILD_STATE = "katsdpflagwriter-" + katsdpflagwriter.__version__

    def __init__(self, host, port, loop, endpoints, flag_interface, npy_path, telstate, flags_name):
        self._npy_path = npy_path
        self._telstate = telstate
        self._endpoints = endpoints
        self._interface_address = katsdpservices.get_interface_address(flag_interface)
        self._n_streams = len(endpoints)
        self._capture_block_state = {}
         # track the status of each capture block we have seen to date
        self._capture_block_stops = defaultdict(int)
         # track the stops received for each capture block
         # if the number of stops is equal to the number of receiving endpoints
         # then we mark this is done, even if we have not had an explicit capture-done
        self._flags_name = flags_name
        self._flags = {}
        self._flag_fragments = defaultdict(int)
         # index by capture_block_id and dump_index, stores the count of
         # flag fragments for key.

        self._build_state_sensor = Sensor(str, "build-state", "SDP Flag Writer build state.")
        self._status_sensor = Sensor(Status, "status", "The current status of the flag writer process.")
        self._input_heaps_sensor = Sensor(int, "input-heaps-total",
                                          "Number of input heaps captured in this session.")
        self._input_dumps_sensor = Sensor(int, "input-dumps-total",
                                          "Number of complete input dumps captured in this session.")
        self._input_incomplete_sensor = Sensor(int, "input-incomplete-heaps-total",
                                               "Number of heaps dropped due to being incomplete.",
                                               status_func=_warn_if_positive)
        self._input_bytes_sensor = Sensor(int, "input-bytes-total",
                                          "Number of payload bytes received in this session.")
        self._output_objects_sensor = Sensor(int, "output-objects-total",
                                             "Number of objects written to disk in this session.")
        self._input_partial_dumps_sensor = Sensor(int, "input-partial-dumps-total",
                                                  "Number of partial dumps stored (due to age or early done).")
        self._last_dump_timestamp_sensor = Sensor(int, "last-dump-timestamp", "Timestamp of the last dump received.")
        self._capture_block_state_sensor = Sensor(str, "capture-block-state",
                                                  "JSON dict with the state of each capture block seen in this session.",
                                                  default='{}')

        super().__init__(host, port, loop=loop)

        self._build_state_sensor.value = self.BUILD_STATE
        self.sensors.add(self._build_state_sensor)
        self._status_sensor.value = Status.IDLE
        self.sensors.add(self._status_sensor)
        self.sensors.add(self._input_heaps_sensor)
        self.sensors.add(self._input_dumps_sensor)
        self.sensors.add(self._input_incomplete_sensor)
        self.sensors.add(self._input_partial_dumps_sensor)
        self.sensors.add(self._input_bytes_sensor)
        self.sensors.add(self._output_objects_sensor)
        self.sensors.add(self._last_dump_timestamp_sensor)
        self.sensors.add(self._capture_block_state_sensor)
        
        self._rx = spead2.recv.asyncio.Stream(spead2.ThreadPool(),
                                              max_heaps=2 * self._n_streams,
                                              ring_heaps=2 * self._n_streams,
                                              contiguous_only=False)

        n_memory_buffers = 8 * self._n_streams
        try:
            self._n_chans = self._telstate['n_chans']
            self._n_bls = self._telstate['n_bls']
            self._int_time = self._telstate['int_time']
            flag_heap_size = self._telstate['n_chans_per_substream'] * self._n_bls
        except KeyError:
            logger.error("Unable to find flag sizing params (n_bls, n_chans, int_time or n_chans_per_substream) for stream {} in telstate."
                         .format(self._flags_name))
            raise
        memory_pool = spead2.MemoryPool(flag_heap_size, flag_heap_size + 4096,
                                        n_memory_buffers, n_memory_buffers)
        self._rx.set_memory_pool(memory_pool)
        self._rx.stop_on_stop_item = False
        for endpoint in self._endpoints:
            if self._interface_address is not None:
                self._rx.add_udp_reader(endpoint.host, endpoint.port,
                                        buffer_size=flag_heap_size + 4096,
                                        interface_address=self._interface_address)
            else:
                self._rx.add_udp_reader(endpoint.port, bind_hostname=endpoint.host,
                                        buffer_size=flag_heap_size + 4096)

    def _set_capture_block_state(self, capture_block_id, state):
        if state == State.COMPLETE:
            # Remove if present
            self._capture_block_state.pop(capture_block_id, None)
        else:
            self._capture_block_state[capture_block_id] = state
        dumped = json.dumps(self._capture_block_state, sort_keys=True, cls=EnumEncoder)
        self._capture_block_state_sensor.value = dumped

    def _get_capture_block_state(self, capture_block_id):
        return self._capture_block_state.get(capture_block_id, None)

    def _add_flags(self, flags, capture_block_id, dump_index, channel0):
        """Add the flag fragment into an appropriate data structure
        and if a particular cbid / dump_index combination is complete,
        write it to disk.

        We test completion by checking that the len(self._endpoints)
        fragments have arrived.

        Returns
        -------
        bool
            If True then this fragment completed a full dump and an attempt
            was made to write this to disk. If the write failed we log
            an error, but opt to bumble on.
        """
        flag_key = "{}_{}".format(capture_block_id, dump_index)
        if flag_key not in self._flags:
            self._flags[flag_key] = FlagItem((self._n_chans, self._n_bls), len(self._endpoints))

        # Add flag fragment to the FlagItem, and if complete write to disk
        if self._flags[flag_key].add_fragment(flags, channel0):
            self._store_flags(flag_key)
        self._check_cache()

    def _store_flags(self, flag_key):
        (capture_block_id, dump_index) = flag_key.split("_")

        # use ChunkStore compatible chunking scheme, even though our trailing
        # axes will always be 0.
        dump_key = "{}_{}/flags/{:05d}_00000_00000".format(capture_block_id, self._flags_name, int(dump_index))
        flag_filename_temp = os.path.join(self._npy_path, "{}.writing.npy".format(dump_key))
        flag_filename = os.path.join(self._npy_path, "{}.npy".format(dump_key))

        completed_flag_dump = self._flags.pop(flag_key)
        if not completed_flag_dump.is_complete():
            logger.warning("Storing partially complete flag dump %s (Received offsets: %s)", flag_key, completed_flag_dump._fragment_offsets)
            self._input_partial_dumps_sensor.value += 1

        try:
            os.makedirs(os.path.dirname(flag_filename), exist_ok=True)
            np.save(flag_filename_temp, completed_flag_dump._flags)
            os.rename(flag_filename_temp, flag_filename)
            logger.info("Saved flag dump to disk in %s", flag_filename)
            self._output_objects_sensor.value += 1
        except OSError:
            # If we fail to save, log the error, but discard dump and bumble on
            logger.error("Failed to flag dump to %s", flag_filename)

    def _check_cache(self, store_all=False):
        """Check the depth of the flag fragment cache and flush
        and the oldest N dumps that make the cache larger than
        FLAG_CACHE_SIZE.
        """
        to_flush = []
        if store_all:
            to_flush = [k for k in self._flags]
            logger.info("Flushing all flag heaps (%d) to disk.", len(to_flush))
        elif len(self._flags) > FLAG_CACHE_SIZE:
            ordered_keys = sorted([k for k in self._flags], key=lambda k: self._flags[k].started, reverse=True)
            to_flush = ordered_keys[FLAG_CACHE_SIZE:]
            logger.warning("Flushing %d old flag heaps to disk to maintain cache depth.", len(to_flush))

        for flag_key in to_flush:
            self._store_flags(flag_key)

    def stop_spead(self):
        self._rx.stop()

    async def do_capture(self):
        n_dumps = 0
        try:
            self._status_sensor.value = Status.WAIT_DATA
            logger.info("Waiting for data...")
            ig = spead2.ItemGroup()
            first = True
            while True:
                heap = await self._rx.get()
                if first:
                    logger.info("First flag heap received...")
                    self._status_sensor.value = Status.CAPTURING
                    first = False
                if heap.is_end_of_stream():
                    logger.info("Stop packet received")
                if isinstance(heap, spead2.recv.IncompleteHeap):
                    if self._input_incomplete_sensor.value % 100 == 0:
                        logger.warning("dropped incomplete heap %d "
                                       "(received %d/%d bytes of payload)",
                                       heap.cnt, heap.received_length, heap.heap_length)
                    self._input_incomplete_sensor.value += 1
                    updated = {}
                else:
                    updated = ig.update(heap)
                if 'timestamp' in updated:
                    flags = ig['flags'].value
                    channel0 = ig['frequency'].value
                    dump_index = int(ig['dump_index'].value)

                    cbid = ig['capture_block_id'].value
                    cur_state = self._get_capture_block_state(cbid)
                    if cur_state == State.COMPLETE:
                        logger.error("Received flags for CBID %s after capture done. These flags will be *discarded*.", cbid)
                        continue
                    elif not cur_state:
                        logger.warning("Received flags for CBID %s unexpectedly (before an init).", cbid)
                        self._set_capture_block_state(cbid, State.CAPTURING)

                    if dump_index >= n_dumps:
                        n_dumps = dump_index + 1
                        self._input_dumps_sensor.value = n_dumps
                    self._add_flags(flags, cbid, dump_index, channel0)
                    self._input_heaps_sensor.value += 1
                    self._input_bytes_sensor.value += flags.nbytes
        except spead2.Stopped:
            logger.info("SPEAD receiver stopped.")
             # Ctrl-C or halt (stop packets ignored)
        except Exception:
            logger.exception("Error in SPEAD receiver")
        finally:
            self._input_bytes_sensor.value = 0
            self._input_heaps_sensor.value = 0
            self._input_dumps_sensor.value = 0
            self._status_sensor.value = Status.FINISHED

    async def request_capture_init(self, ctx, capture_block_id: str) -> None:
        """Start an observation"""
        if capture_block_id in self._capture_block_state:
            raise FailReply("Capture block ID {} is already active".format(capture_block_id))
        self._set_capture_block_state(capture_block_id, State.CAPTURING)

    def _mark_cbid_complete(self, capture_block_id):
        """Inform other users of the on disk data that we are finished with a
        particular capture_block_id.
        """
        logger.info("Capture block %s flag capture complete.", capture_block_id)
        touch_file = os.path.join(self._npy_path, "{}_{}".format(capture_block_id, self._flags_name),
                                  "{}_complete".format(self._flags_name))
        os.makedirs(os.path.dirname(touch_file), exist_ok=True)
        with open(touch_file, 'a'):
            os.utime(touch_file, None)
        self._set_capture_block_state(capture_block_id, State.COMPLETE)

    async def request_capture_done(self, ctx, capture_block_id: str) -> None:
        """Mark specified capture_block_id as complete and flush flag cache.
        """
        if capture_block_id not in self._capture_block_state:
            raise FailReply("Specified capture block ID {} is unknown.".format(capture_block_id))
        # Allow some time for stragglers to appear before committing all remaining
        # flags to disk.
        await asyncio.sleep(5, loop=self.loop)
        self._check_cache(store_all=True)
        self._mark_cbid_complete(capture_block_id)


def on_shutdown(loop, server):
    loop.remove_signal_handler(signal.SIGINT)
    loop.remove_signal_handler(signal.SIGTERM)
     # in case the exit code below borks, we allow shutdown via traditional means
    server.stop_spead()
    server.halt()


async def run(loop, server):
    await server.start()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, lambda: on_shutdown(loop, server))
    await server.do_capture()
    await server.join()


if __name__ == '__main__':
    katsdpservices.setup_logging()
    logger = logging.getLogger("katsdpflagwriter")
    katsdpservices.setup_restart()

    parser = katsdpservices.ArgumentParser()
    parser.add_argument('--npy-path', default="/var/kat/data", metavar='NPYPATH',
                        help='Root in which to write flag dumps in npy format.')
    parser.add_argument('--flags-spead', default=':7202', metavar='ENDPOINTS',
                        type=katsdptelstate.endpoint.endpoint_list_parser(7202),
                        help='Source port/multicast groups for flags SPEAD streams. '
                             '[default=%(default)s]')
    parser.add_argument('--flags-interface', metavar='INTERFACE',
                        help='Network interface to subscribe to for flag streams. '
                             '[default=auto]')
    parser.add_argument('--flags-name', type=str, default='sdp_l1_flags',
                        help='name for the flags stream. [default=%(default)s]', metavar='NAME')
    parser.add_argument('-p', '--port', type=int, default=2052, metavar='N',
                        help='KATCP host port [default=%(default)s]')
    parser.add_argument('-a', '--host', default="", metavar='HOST',
                        help='KATCP host address [default=all hosts]')

    args = parser.parse_args()

    if not os.path.isdir(args.npy_path):
        logger.error("Specified NPY path, %s, does not exist.", args.npy_path)
        sys.exit(2)

    loop = asyncio.get_event_loop()

    telstate_flags = args.telstate.view(args.flags_name)
    server = FlagWriterServer(args.host, args.port, loop, args.flags_spead,
                              args.flags_interface, args.npy_path,
                              telstate_flags, args.flags_name)
    logger.info("Started flag writer server.")

    loop.run_until_complete(run(loop, server))
    loop.close()
