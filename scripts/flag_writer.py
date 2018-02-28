#!/usr/bin/env python3

"""Capture L1 flags from the SPEAD stream(s) produced by cal.

We adopt a naive strategy and store the flags for each dump in a single
object. These objects will be later picked up by the trawler process
and inserted into the appropriate bucket in S3 from where they will be
picked up by katdal.

"""

import os
import threading
import logging
import signal
import enum
import json
import sys
from collections import defaultdict

import numpy as np
import spead2
import spead2.recv
import katsdptelstate
import katsdpservices
from aiokatcp import DeviceServer, Sensor, FailReply
import katsdpflagwriter

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

def _warn_if_positive(value):
    return katcp.Sensor.WARN if value > 0 else katcp.Sensor.NOMINAL


class FlagWriterServer(DeviceServer):
    VERSION = "sdp-flag-writer-0.1"
    BUILD_STATE = "katsdpflagwriter-" + katsdpflagwriter.__version__

    def __init__(self, host, port, loop, endpoints, npy_path, telstate):
        self._npy_path = npy_path
        self._telstate = telstate
        self._halt = threading.Event()
        self._endpoints = endpoints
        self._capture_block_state = {}
         # track the status of each capture block we have seen to date
        self._capture_block_stops = defaultdict(int)
         # track the stops received for each capture block
         # if the number of stops is equal to the number of receiving endpoints
         # then we mark this is done, even if we have not had an explicit capture-done
        self._flags = {}
        self._flag_fragments = defaultdict(int)
         # index by capture_block_id and dump_index, stores the count of
         # flag fragments for key.

        self._build_state_sensor = Sensor(str, "build-state", "SDP Flag Writer build state.")
        self._status_sensor = Sensor(Status, "status", "The current status of the flag writer process.")
        self._input_heaps_sensor = Sensor(int, "input-heaps-total", "Number of input heaps captured in this session.", default=0)
        self._input_dumps_sensor = Sensor(int, "input-dumps-total", "Number of complete input dumps captured in this session.", default=0)
        self._input_incomplete_sensor = Sensor(int, "input-incomplete-total", "Number of heaps dropped due to being incomplete.", default=0, eval_func=_warn_if_positive)

        self._input_bytes_sensor = Sensor(int, "input-bytes-total", "Number of payload bytes received in this session.", default=0)
        self._output_objects_sensor = Sensor(str, "output-objects-total", "Number of objects written to disk in this session.", default=0)
        self._last_dump_timestamp_sensor = Sensor(int, "last-dump-timestamp", "Timestamp of the last dump received.")
        self._capture_block_state_sensor = Sensor(str, "capture-block-state", "JSON dict with the state of each capture block seen in this session.", default='{}')

        super().__init__(host, port, loop=loop)

        self._build_state_sensor.set_value(self.BUILD_STATE)
        self.sensors.add(self._build_state_sensor)
        self._device_status_sensor.set_value(DeviceStatus.IDLE)
        self.sensors.add(self._device_status_sensor)
        self.sensor.add(self._input_heaps_sensor)
        self.sensor.add(self._input_incomplete_sensor)
        self.sensor.add(self._input_bytes_sensor)
        self.sensor.add(self._last_dump_timestamp_sensor)
        self.sensor.add(self._capture_block_state_sensor)

        self._capture_thread = threading.Thread(target=self._do_capture, name='capture')
        self._capture_thread.start()
        logger.info("Started flag capture thread.")

    def _set_capture_block_state(self, capture_block_id, state):
        if state == State.DEAD:
            # Remove if present
            self._capture_block_state.pop(capture_block_id, None)
        else:
            self._capture_block_state[capture_block_id] = state
        dumped = json.dumps(self._capture_block_state, sort_keys=True, cls=EnumEncoder)
        self._capture_block_state_sensor.set_value(dumped)

    def _get_capture_block_state(self, capture_block_id):
        return self._capture_block_state.get(capture_block_id, None)

    async def write_meta(self, ctx, capture_block_id, streams, lite=True):
        """Implementation of request_write_meta."""
        rate_per_stream = {}
        return rate_per_stream

    def _add_flags(self, flags, capture_block_id, dump_index, channel0):
        """Add the flag fragment into an appropriate data structure
        and if a particular cbid / dump_index combination is complete,
        write it to disk.

        We test completion by checking that the len(self._endpoints)
        fragments have arrived.
        """
        flag_key = "{}_{}".format(capture_block_id, dump_index)
        if flag_key not in self._flags:
            self._flags[flag_key] = np.zeros((len(self._endpoints) * flags.shape[0], flags.shape[1]))

        self._flags[flag_key][channel0:channel0+flag.shape[0]] = flags
        self._flag_fragments[flag_key] += 1

        # Received a complete flag dump - writing to disk
        if self._flag_fragments[flag_key] >= len(self._endpoints):
            flag_filename = os.path.join(self._npy_path, capture_block_id, "{}.flag".format(flag_key))
            os.makedirs(os.path.dirname(flag_filename), exist_ok=True)
            np.save(flag_filename, self._flags.pop(flag_key))
            logger.info("Saved flag array to disk in %s", flag_filename)
            self._flag_fragments.pop(flag_key)
            return True
        return False

    def _do_capture(self, capture_stream_name, chunk_info):
        n_dumps = 0
        try:
            self._status_sensor.set_value(Status.WAIT_DATA)
            logger.info("Waiting for data...")
            ig = spead2.ItemGroup()
            first = True
            for heap in self._rx:
                if self._halt.is_set():
                    logger.info("Requested halt of capture thread, stopping...")
                    break
                if first:
                    logger.info("First flag heap received...")
                    self._status_sensor.set_value(Status.CAPTURING)
                    first = False
                if heap.is_end_of_stream():
                    logger.info("Stop packet received")
                    
                if isinstance(heap, spead2.recv.IncompleteHeap):
                    self._logger.warning("dropped incomplete heap %d "
                            "(received %d/%d bytes of payload)",
                            heap.cnt, heap.received_length, heap.heap_length)
                    self._input_incomplete_.value += 1
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
                        logger.error("Recieved flags for CBID {} after capture done. These flags will be *discarded*.")
                        continue
                    elif not cur_state:
                        logger.warning("Received flags for CBID {} unexpectedly (before an init).")
                        self._set_capture_block_state(cbid, State.CAPTURING)

                    if dump_index >= n_dumps:
                        n_dumps = dump_index + 1
                        self._input_dumps_sensor.value = n_dumps
                    stored = self._add_flags(flags, cbid, dump_index, channel0)
                    if stored:
                        self._output_objects_sensor.value += 1
                    n_heaps += 1
                    self._input_heaps_sensor.value = n_heaps
                    self._input_bytes_sensor.value = flags.nbytes
        except Exception as err:
            self._logger.exception(err)
            end_status = "error"
        finally:
            self._input_bytes_sensor.value = 0
            self._input_heaps_sensor.value = 0
            self._input_dumps_sensor.value = 0
            self._status_sensor.value = State.FINISHED

    async def request_capture_init(self, ctx, capture_block_id: str) -> None:
        """Start an observation"""
        if self._halt:
            raise FailReply("Capture thread is shutting down.")
        if capture_block_id in self._capture_block_state:
            raise FailReply("Capture block ID {} is already active".format(capture_block_id))
        self._set_capture_block_state(capture_block_id, State.CAPTURING)

    def _mark_cbid_complete(self, capture_block_id):
        """Inform other users of the on disk data that we are finished with a
        particular capture_block_id."""
        touch_file = os.path.join(self._npy_path, capture_block_id, "flags_complete")
        os.makedirs(os.path.dirname(touch_file), exist_ok=True)
        with open(touch_file, 'a'):
            os.utime(touch_file, None)
        self._set_capture_block_state(capture_block_id, State.COMPLETE)

    async def request_capture_done(self, ctx, capture_block_id: str) -> None:
        """Notice to mark specified capture_block_id as complete and inform
        downstream services of completion."""
        if not capture_block_id in self._capture_block_state:
            raise FailReply("Specified capture block ID {} is unkown.".format(capture_block_id))
        if self._capture_block_state[capture_block_id] != State.CAPTURING:
            raise FailReply("Specified capture block ID {} is not in state capturing.".format(capture_block_id))
        self._mark_cbid_complete(capture_block_id)


def on_shutdown(loop, server):
    loop.remove_signal_handler(signal.SIGINT)
    loop.remove_signal_handler(signal.SIGTERM)
     # in case the exit code below borks, we allow shutdown via traditional means
    server.halt()


async def run(loop, server):
    await server.start()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, lambda: on_shutdown(loop, server))
    await server.join()


if __name__ == '__main__':
    katsdpservices.setup_logging()
    logger = logging.getLogger("katsdpmetawriter")
    katsdpservices.setup_restart()

    parser = katsdpservices.ArgumentParser()
    parser.add_argument('--rdb-path', default="/var/kat/data", metavar='RDBPATH',
                        help='Root in which to write RDB dumps.')
    parser.add_argument('--store-s3', dest='store_s3', default=False, action='store_true',
                        help='Enable storage of RDB dumps in S3')
    parser.add_argument('--access-key', default="", metavar='ACCESS',
                        help='S3 access key with write permission to the specified bucket. Default is unauthenticated access')
    parser.add_argument('--secret-key', default="", metavar='SECRET',
                        help='S3 secret key for the specified access key. Default is unauthenticated access')
    parser.add_argument('--s3-host', default='localhost', metavar='HOST',
                        help='S3 gateway host address [default=%(default)s]')
    parser.add_argument('--s3-port', default=7480, metavar='PORT',
                        help='S3 gateway port [default=%(default)s]')
    parser.add_argument('-p', '--port', type=int, default=2049, metavar='N',
                        help='KATCP host port [default=%(default)s]')
    parser.add_argument('-a', '--host', default="", metavar='HOST',
                        help='KATCP host address [default=all hosts]')

    args = parser.parse_args()

    if not os.path.exists(args.rdb_path):
        logger.error("Specified RDB path, %s, does not exist.", args.rdb_path)
        sys.exit(2)

    boto_dict = None
    if args.store_s3:
        boto_dict = make_boto_dict(args)
        s3_conn = get_s3_connection(boto_dict, fail_on_boto=True)
        if s3_conn:
            user_id = s3_conn.get_canonical_user_id()
            s3_conn.close()
             # we rebuild the connection each time we want to write a meta-data dump
            logger.info("Successfully tested connection to S3 endpoint as %s.", user_id)
        else:
            logger.warning("S3 endpoint %s:%s not available. Files will only be written locally.", args.s3_host, args.s3_port)
    else:
        logger.info("Running in disk only mode. RDB dumps will not be written to S3")

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(3)

    server = MetaWriterServer(args.host, args.port, loop, executor, boto_dict, args.rdb_path, args.telstate)
    logger.info("Started meta-data writer server.")

    loop.run_until_complete(run(loop, server))
    executor.shutdown()
    loop.close()
