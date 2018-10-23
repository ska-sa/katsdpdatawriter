import logging
import enum
import json
import asyncio
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import spead2
import spead2.recv.asyncio
import katsdpservices
import katdal
import katdal.chunkstore
from katdal.visdatav4 import FLAG_NAMES
from aiokatcp import DeviceServer, Sensor, SensorSet, FailReply
import katsdptelstate
from katsdptelstate.endpoint import Endpoint

import katsdpdatawriter
from . import spead_write
from .spead_write import RechunkerGroup
from .bounded_executor import BoundedThreadPoolExecutor


logger = logging.getLogger(__name__)


class Status(enum.Enum):
    """Status of the whole process"""
    WAIT_DATA = 1
    CAPTURING = 2
    FINISHED = 3


class State(enum.Enum):
    """State of a single capture block"""
    CAPTURING = 1         # capture-init has been called, but not capture-done
    COMPLETE = 2          # capture-done has been called


class EnumEncoder(json.JSONEncoder):
    """JSON encoder that stringifies enums"""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, enum.Enum):
            return obj.name
        return json.JSONEncoder.default(self, obj)


class FlagWriter(spead_write.SpeadWriter):
    """Glue between :class:`~.SpeadWriter` and :class:`FlagWriterServer`."""
    def __init__(self, sensors: SensorSet, rx: spead2.recv.asyncio.Stream,
                 server: 'FlagWriterServer') -> None:
        super().__init__(sensors, rx)
        self._server = server

    def first_heap(self) -> None:
        logger.info("First flag heap received...")
        self.sensors['status'].value = Status.CAPTURING

    def rechunker_group(self, updated: Dict[str, spead2.Item]) -> Optional[RechunkerGroup]:
        cbid = updated['capture_block_id'].value
        return self._server.rechunker_group(cbid)


class FlagWriterServer(DeviceServer):
    """Top-level device server for flag writer service"""

    VERSION = "sdp-flag-writer-0.2"
    BUILD_STATE = "katsdpdatawriter-" + katsdpdatawriter.__version__

    def __init__(self, host: str, port: int, loop: asyncio.AbstractEventLoop,
                 endpoints: List[Endpoint], flag_interface: Optional[str], flags_ibv: bool,
                 chunk_store: katdal.chunkstore.ChunkStore, chunk_size: float,
                 telstate: katsdptelstate.TelescopeState,
                 input_name: str, output_name: str, rename_src: Mapping[str, str],
                 s3_endpoint_url: Optional[str],
                 max_workers: int) -> None:
        super().__init__(host, port, loop=loop)

        self._chunk_store = chunk_store
        self._telstate = telstate
        # track the status of each capture block we have seen to date
        self._capture_block_state = {}   # type: Dict[str, State]
        self._input_name = input_name
        self._output_name = output_name
        # rechunker group for each CBID
        self._flag_streams = {}          # type: Dict[str, RechunkerGroup]
        self._executor = BoundedThreadPoolExecutor(max_workers=max_workers)

        self.sensors.add(Sensor(
            Status, "status", "The current status of the flag writer process."))
        self.sensors.add(Sensor(
            str, "capture-block-state",
            "JSON dict with the state of each capture block seen in this session.",
            default='{}', initial_status=Sensor.Status.NOMINAL))
        for sensor in spead_write.io_sensors():
            self.sensors.add(sensor)
        self.sensors.add(spead_write.device_status_sensor())

        telstate_input = telstate.view(input_name)
        in_chunks = spead_write.chunks_from_telstate(telstate_input)
        DATA_LOST = 1 << FLAG_NAMES.index('data_lost')
        self._arrays = [
            spead_write.make_array('flags', in_chunks, DATA_LOST, np.uint8, chunk_size)
        ]
        spead_write.write_telstate(telstate, input_name, output_name, rename_src, s3_endpoint_url)

        rx = spead_write.make_receiver(
            endpoints, self._arrays,
            katsdpservices.get_interface_address(flag_interface), flags_ibv)
        self._writer = FlagWriter(self.sensors, rx, self)
        self._capture_task = loop.create_task(self._do_capture())

    def _set_capture_block_state(self, capture_block_id: str, state: State) -> None:
        if state == State.COMPLETE:
            # Remove if present
            self._capture_block_state.pop(capture_block_id, None)
        else:
            self._capture_block_state[capture_block_id] = state
        dumped = json.dumps(self._capture_block_state, sort_keys=True, cls=EnumEncoder)
        self.sensors['capture-block-state'].value = dumped

    def _get_capture_block_state(self, capture_block_id: str) -> Optional[State]:
        return self._capture_block_state.get(capture_block_id, None)

    def _get_capture_stream_name(self, capture_block_id: str) -> str:
        """Get the capture-stream name of the output stream"""
        return "{}_{}".format(capture_block_id, self._output_name)

    def _get_prefix(self, capture_block_id: str) -> str:
        """Get the prefix (aka bucket name) to use with the chunk store"""
        # S3 doesn't allow underscores in bucket names
        return self._get_capture_stream_name(capture_block_id).replace('_', '-')

    def rechunker_group(self, cbid: str) -> Optional[RechunkerGroup]:
        extra = dict(capture_block_id=cbid)
        if not self._get_capture_block_state(cbid):
            logger.error("Received flags for CBID %s outside of init/done. "
                         "These flags will be *discarded*.", cbid, extra=extra)
            return None

        if cbid not in self._flag_streams:
            self._flag_streams[cbid] = RechunkerGroup(
                self._executor, self._chunk_store, self._writer.sensors,
                self._get_prefix(cbid), self._arrays)
        return self._flag_streams[cbid]

    async def _do_capture(self) -> None:
        """Run the entire capture process.

        This runs for the lifetime of the server.
        """
        try:
            spead_write.clear_io_sensors(self.sensors)
            self.sensors['status'].value = Status.WAIT_DATA
            logger.info("Waiting for data...")
            await self._writer.run()
        except Exception:
            logger.exception("Error in SPEAD receiver")
            self.sensors['device-status'].value = spead_write.DeviceStatus.FAIL
        finally:
            spead_write.clear_io_sensors(self.sensors)
            self.sensors['status'].value = Status.FINISHED
            self._executor.shutdown()

    async def request_capture_init(self, ctx, capture_block_id: str) -> None:
        """Start an observation"""
        if capture_block_id in self._capture_block_state:
            raise FailReply("Capture block ID {} is already active".format(capture_block_id))
        self._set_capture_block_state(capture_block_id, State.CAPTURING)

    def _mark_cbid_complete(self, capture_block_id: str) -> None:
        """Inform other users of the on disk data that we are finished with a
        particular capture_block_id.
        """
        extra = dict(capture_block_id=capture_block_id)
        logger.info("Capture block %s flag capture complete.", capture_block_id, extra=extra)
        self._chunk_store.mark_complete(self._get_prefix(capture_block_id))
        self._set_capture_block_state(capture_block_id, State.COMPLETE)

    async def _write_telstate_meta(self, capture_block_id: str) -> None:
        """Write out chunk information for the specified CBID to telstate."""
        extra = dict(capture_block_id=capture_block_id)
        if capture_block_id not in self._flag_streams:
            logger.warning("No flag data received for cbid %s. Flag stream will not be usable.",
                           capture_block_id, extra=extra)
            return
        rechunker_group = self._flag_streams[capture_block_id]
        chunk_info = await rechunker_group.get_chunk_info()
        capture_stream_name = self._get_capture_stream_name(capture_block_id)
        telstate_capture = self._telstate.view(capture_stream_name)
        telstate_capture.add('chunk_info', chunk_info, immutable=True)
        logger.info("Written chunk information to telstate.", extra=extra)

    async def request_capture_done(self, ctx, capture_block_id: str) -> None:
        """Mark specified capture_block_id as complete.

        It flushes the flag cache and writes chunk info into telstate.
        """
        if capture_block_id not in self._capture_block_state:
            raise FailReply("Specified capture block ID {} is unknown.".format(capture_block_id))
        # Allow some time for stragglers to appear
        await asyncio.sleep(5, loop=self.loop)
        await self._write_telstate_meta(capture_block_id)
        self._mark_cbid_complete(capture_block_id)

    async def stop(self, cancel: bool = True) -> None:
        self._writer.stop()
        await self._capture_task
        await super().stop(cancel)
