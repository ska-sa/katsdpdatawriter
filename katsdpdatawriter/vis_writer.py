"""Capture L0 visibilities from a SPEAD stream and write to a local chunk store.

This process lives across multiple observations and hence multiple data sets.
It writes weights and flags as well.

The status sensor has the following states (with typical transition events):

  - `idle`: ready to start capture
     -> ?capture-init ->
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
  - <capture_stream>_chunk_info: {prefix, dtype, shape, chunks} dict per array
"""

import os
import asyncio
import logging
import enum
from typing import List, Tuple, Any, Optional   # noqa: F401

import numpy as np
import aiokatcp
from aiokatcp import DeviceServer, Sensor, FailReply
from katdal.visdatav4 import FLAG_NAMES
import katdal.chunkstore
import katsdptelstate
from katsdptelstate.endpoint import Endpoint
import katsdpservices
import spead2.recv.asyncio

import katsdpdatawriter
from . import spead_write


logger = logging.getLogger(__name__)


class Status(enum.Enum):
    IDLE = 1
    WAIT_DATA = 2
    CAPTURING = 3
    FINALISING = 4
    COMPLETE = 5
    ERROR = 6


# TODO: move this into aiokatcp
class DeviceStatus(enum.Enum):
    """Standard katcp device status"""
    OK = 0
    DEGRADED = 1
    FAIL = 2


def _status_status(value: Status) -> aiokatcp.Sensor.Status:
    if value == Status.ERROR:
        return Sensor.Status.ERROR
    else:
        return Sensor.Status.NOMINAL


def _device_status_status(value: DeviceStatus) -> aiokatcp.Sensor.Status:
    """Sets katcp status for device-status sensor from value"""
    if value == DeviceStatus.OK:
        return Sensor.Status.NOMINAL
    elif value == DeviceStatus.DEGRADED:
        return Sensor.Status.WARN
    else:
        return Sensor.Status.ERROR


def _make_array(name, in_chunks: Tuple[Tuple[int]],
                fill_value: Any, dtype: Any, chunk_size: float) -> spead_write.Array:
    # Shape of a single input chunk
    shape = tuple(c[0] for c in in_chunks)
    # Compute the decomposition of each input chunk
    chunks = katdal.chunkstore.generate_chunks(shape, dtype, chunk_size,
                                               dims_to_split=(0, 1), power_of_two=True)
    # Repeat for each input chunk
    out_chunks = tuple(outc * len(inc) for inc, outc in zip(in_chunks, chunks))
    return spead_write.Array(name, in_chunks, out_chunks, fill_value, dtype)


class VisibilityWriterServer(DeviceServer):
    VERSION = "sdp-vis-writer-0.2"
    BUILD_STATE = "katsdpdatawriter-" + katsdpdatawriter.__version__

    def __init__(self, host: str, port: int, loop: asyncio.AbstractEventLoop,
                 endpoints: List[Endpoint], interface: Optional[str], ibv: bool,
                 chunk_store: katdal.chunkstore.ChunkStore, chunk_size: float,
                 telstate_l0: katsdptelstate.TelescopeState, stream_name: str) -> None:
        super().__init__(host, port, loop=loop)
        self._endpoints = endpoints
        self._interface_address = katsdpservices.get_interface_address(interface)
        self._ibv = ibv
        self._chunk_store = chunk_store
        self._stream_name = stream_name
        self._telstate_l0 = telstate_l0
        self._rx = None    # type: Optional[spead2.recv.asyncio.Stream]

        in_chunks = spead_write.chunks_from_telstate(telstate_l0)
        DATA_LOST = 1 << FLAG_NAMES.index('data_lost')
        self._arrays = [
            _make_array('correlator_data', in_chunks, 0, np.complex64, chunk_size),
            _make_array('flags', in_chunks, DATA_LOST, np.uint8, chunk_size),
            _make_array('weights', in_chunks, 0, np.uint8, chunk_size),
            _make_array('weights_channel', in_chunks[:2], 0, np.float32, chunk_size)
        ]
        self._capture_task = None     # type: Optional[asyncio.Task]
        self._n_substreams = len(in_chunks[1])

        self.sensors.add(Sensor(
            Status, 'status', 'The current status of the capture process',
            default=Status.IDLE, initial_status=Sensor.Status.NOMINAL,
            status_func=_status_status))
        self.sensors.add(Sensor(
            DeviceStatus, 'device-status', 'Health sensor',
            default=DeviceStatus.OK, initial_status=Sensor.Status.NOMINAL,
            status_func=_device_status_status))
        spead_write.add_sensors(self.sensors)

    def _first_heap(self):
        self.sensors['status'].value = Status.CAPTURING

    async def _do_capture(self, capture_stream_name: str, rx: spead2.recv.asyncio.Stream) -> None:
        writer = None
        try:
            rechunker_group = spead_write.RechunkerGroup(
                self._chunk_store, self.sensors, capture_stream_name, self._arrays)
            writer = spead_write.SpeadWriter(self.sensors, rx)
            # mypy doesn't like overriding methods on an instance
            writer.rechunker_group = lambda updated: rechunker_group   # type: ignore
            writer.first_heap = self._first_heap                       # type: ignore
            self.sensors['status'].value = Status.WAIT_DATA

            await writer.run(stops=self._n_substreams)

            self.sensors['status'].value = Status.FINALISING
            view = self._telstate_l0.view(capture_stream_name)
            view.add('chunk_info', rechunker_group.get_chunk_info())
            os.sync()
            touch_file = os.path.join(self._chunk_store.path, capture_stream_name,
                                      'complete')
            os.makedirs(os.path.dirname(touch_file), exist_ok=True)
            with open(touch_file, 'a'):
                os.utime(touch_file, None)
            self.sensors['status'].value = Status.COMPLETE
        except Exception:
            logger.exception('Exception in capture task')
            self.sensors['status'].value = Status.ERROR
            self.sensors['device-status'].value = DeviceStatus.FAIL
        finally:
            spead_write.clear_input_sensors(self.sensors)

    async def request_capture_init(self, ctx, capture_block_id: str = None) -> None:
        """Start listening for L0 data"""
        if self._capture_task is not None:
            logger.info("Ignoring capture_init because already capturing")
            raise FailReply('Already capturing')
        self.sensors['status'].value = Status.WAIT_DATA
        self.sensors['device-status'].value = DeviceStatus.OK
        sep = self._telstate_l0.SEPARATOR
        capture_stream_name = sep.join((capture_block_id, self._stream_name))
        self._rx = spead_write.make_receiver(
            self._endpoints, self._arrays, self._interface_address, self._ibv)
        self._capture_task = self.loop.create_task(self._do_capture(capture_stream_name, self._rx))
        logger.info('Starting capture to %s', capture_stream_name)

    async def capture_done(self) -> None:
        """Implementation of :meth:`request_capture_done`.

        This is split out to allow it to be called on ``SIGINT``.
        """
        if self._capture_task is None:
            return
        capture_task = self._capture_task
        # Give it a chance to stop on its own from stop packets
        try:
            logger.info('Waiting for capture task (5s timeut')
            await asyncio.wait_for(capture_task, timeout=5)
        except asyncio.TimeoutError:
            if self._capture_task is not capture_task:
                return     # Someone else beat us to the cleanup
            logger.info('Stopping receiver and waiting for capture task')
            if self._rx:
                self._rx.stop()
            await capture_task

        if self._capture_task is not capture_task:
            return     # Someone else beat us to the cleanup
        if self._rx:
            self._rx.stop()
        self._capture_task = None
        self.sensors['status'].value = Status.IDLE

    async def request_capture_done(self, ctx) -> None:
        """Stop capturing, which cleans up the capturing task."""
        if self._capture_task is None:
            logger.info("Ignoring capture_done because already explicitly stopped")
            raise FailReply('Not capturing')
        await self.capture_done()

    async def stop(self, cancel=True) -> None:
        await self.capture_done()
        await super().stop(cancel)
