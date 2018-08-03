import os
import logging
import enum
import json
import asyncio
from typing import Any, Dict, Optional

import numpy as np
import spead2
import spead2.recv.asyncio
import katsdpservices
import katdal
import katdal.chunkstore_npy
from katdal.visdatav4 import FLAG_NAMES
from aiokatcp import DeviceServer, Sensor, FailReply

import katsdpdatawriter
from . import spead_write
from .spead_write import RechunkerGroup, Array


logger = logging.getLogger("__name__")


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
    def default(self, obj: Any) -> Any:
        if isinstance(obj, enum.Enum):
            return obj.name
        return json.JSONEncoder.default(self, obj)


class FlagWriterServer(DeviceServer):
    VERSION = "sdp-flag-writer-0.2"
    BUILD_STATE = "katsdpdatawriter-" + katsdpdatawriter.__version__

    def __init__(self, host, port, loop,
                 endpoints, flag_interface, flags_ibv,
                 npy_path, telstate, flags_name):
        super().__init__(host, port, loop=loop)

        self._npy_path = npy_path
        self._telstate_flags = telstate.view(flags_name)
        self._telstate = telstate
        self._endpoints = endpoints
        # track the status of each capture block we have seen to date
        self._capture_block_state = {}
        self._flags_name = flags_name
        # rechunker group for each CBID
        self._flag_streams = {}

        self.sensors.add(Sensor(
            Status, "status", "The current status of the flag writer process."))
        self.sensors.add(Sensor(
            str, "capture-block-state",
            "JSON dict with the state of each capture block seen in this session.",
            default='{}'))

        in_chunks = spead_write.chunks_from_telstate(self._telstate_flags)
        out_chunks = in_chunks   # For now - will change later
        DATA_LOST = 1 << FLAG_NAMES.index('data_lost')
        self._arrays = [Array('flags', in_chunks, out_chunks, DATA_LOST, np.uint8)]
        self._chunk_store = katdal.chunkstore_npy.NpyFileChunkStore(npy_path)

        rx = spead_write.make_receiver(
            self._endpoints, self._arrays,
            katsdpservices.get_interface_address(flag_interface), flags_ibv)
        self._writer = spead_write.SpeadWriter(rx)
        self._writer.first_heap = self._first_heap
        self._writer.rechunker_group = self._rechunker_group

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

    def _get_capture_stream_name(self, capture_block_id):
        return "{}_{}".format(capture_block_id, self._flags_name)

    def stop_spead(self):
        self._writer.stop()

    def _first_heap(self):
        logger.info("First flag heap received...")
        self.sensors['status'].value = Status.CAPTURING

    def _rechunker_group(self, updated: Dict[str, spead2.Item]) -> Optional[RechunkerGroup]:
        cbid = updated['capture_block_id'].value
        cur_state = self._get_capture_block_state(cbid)
        if cur_state == State.COMPLETE:
            logger.error("Received flags for CBID %s after capture done. "
                         "These flags will be *discarded*.", cbid)
            return None
        elif not cur_state:
            logger.warning("Received flags for CBID %s unexpectedly (before an init).", cbid)
            self._set_capture_block_state(cbid, State.CAPTURING)

        if cbid not in self._flag_streams:
            prefix = self._get_capture_stream_name(cbid)
            self._flag_streams[cbid] = RechunkerGroup(
                self._chunk_store, self._writer.sensors, prefix, self._arrays)
        return self._flag_streams[cbid]

    async def do_capture(self):
        try:
            self.sensors['status'].value = Status.WAIT_DATA
            logger.info("Waiting for data...")
            await self._writer.run()
        except Exception:
            logger.exception("Error in SPEAD receiver")
        finally:
            self._writer.clear_input_sensors()
            self.sensors['status'].value = Status.FINISHED

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
        touch_file = os.path.join(self._npy_path, self._get_capture_stream_name(capture_block_id),
                                  "complete")
        os.makedirs(os.path.dirname(touch_file), exist_ok=True)
        with open(touch_file, 'a'):
            os.utime(touch_file, None)
        self._set_capture_block_state(capture_block_id, State.COMPLETE)

    def _write_telstate_meta(self, capture_block_id):
        """Write out chunk information for the specified CBID to telstate."""
        if capture_block_id not in self._flag_streams:
            logger.warning("No flag data received for cbid %s. Flag stream will not be usable.",
                           capture_block_id)
            return
        rechunker_group = self._flag_streams[capture_block_id]
        rechunker_group.close()
        chunk_info = rechunker_group.get_chunk_info()
        capture_stream_name = self._get_capture_stream_name(capture_block_id)
        telstate_capture = self._telstate.view(capture_stream_name)
        telstate_capture.add('chunk_info', chunk_info, immutable=True)
        logger.info("Written chunk information to telstate.")

    async def request_capture_done(self, ctx, capture_block_id: str) -> None:
        """Mark specified capture_block_id as complete and flush flag cache.
        """
        if capture_block_id not in self._capture_block_state:
            raise FailReply("Specified capture block ID {} is unknown.".format(capture_block_id))
        # Allow some time for stragglers to appear
        await asyncio.sleep(5, loop=self.loop)
        self._write_telstate_meta(capture_block_id)
        self._mark_cbid_complete(capture_block_id)
