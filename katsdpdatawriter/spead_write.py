"""
Receive heaps from a SPEAD stream and write corresponding data to a chunk store.
"""

import time
import enum
import logging
import os
from typing import Optional, Any, Sequence, Iterable, Dict

import numpy as np
import attr
from aiokatcp import Sensor, SensorSet
import spead2
import spead2.recv.asyncio
import katdal.chunkstore
from katsdptelstate.endpoint import Endpoint

from . import rechunk
from .rechunk import Chunks, Offset


logger = logging.getLogger(__name__)


# TODO: move this into aiokatcp
class DeviceStatus(enum.Enum):
    """Standard katcp device status"""
    OK = 0
    DEGRADED = 1
    FAIL = 2


def _device_status_status(value: DeviceStatus) -> Sensor.Status:
    """Sets katcp status for device-status sensor from value"""
    if value == DeviceStatus.OK:
        return Sensor.Status.NOMINAL
    elif value == DeviceStatus.DEGRADED:
        return Sensor.Status.WARN
    else:
        return Sensor.Status.ERROR


def _warn_if_positive(value: float) -> Sensor.Status:
    return Sensor.Status.WARN if value > 0 else Sensor.Status.NOMINAL


# Just to work around https://github.com/python/mypy/issues/4729
def _dtype_converter(dtype: Any) -> np.dtype:
    return np.dtype(dtype)


def io_sensors() -> Sequence[Sensor]:
    """Create input and output counter sensors."""
    return [
        Sensor(
            int, "input-incomplete-heaps-total",
            "Number of heaps dropped due to being incomplete. (prometheus: counter)",
            status_func=_warn_if_positive),
        Sensor(
            int, "input-too-old-heaps-total",
            "Number of heaps dropped because they are too late. (prometheus: counter)",
            status_func=_warn_if_positive),
        Sensor(
            int, "input-bytes-total",
            "Number of payload bytes received in this session. (prometheus: counter)",
            "B"),
        Sensor(
            int, "input-heaps-total",
            "Number of input heaps captured in this session. (prometheus: counter)"),
        Sensor(
            int, "input-dumps-total",
            "Number of complete input dumps captured in this session. (prometheus: counter)"),
        Sensor(
            int, "output-bytes-total",
            "Number of payload bytes written to chunk store in this session. (prometheus: counter)",
            "B"),
        Sensor(
            int, "output-chunks-total",
            "Number of chunks written to chunk store in this session. (prometheus: counter)"),
        Sensor(
            float, "output-seconds-total",
            "Accumulated time spent writing chunks. (prometheus: counter)",
            "s")
    ]


def device_status_sensor() -> Sensor:
    """Create a sensor to track device status"""
    return Sensor(DeviceStatus, 'device-status', 'Health sensor',
                  default=DeviceStatus.OK, initial_status=Sensor.Status.NOMINAL,
                  status_func=_device_status_status)


def clear_io_sensors(sensors: SensorSet) -> None:
    """Zero the input and output counters in a sensor set"""
    now = time.time()
    for name in ['input-incomplete-heaps-total',
                 'input-too-old-heaps-total',
                 'input-bytes-total',
                 'input-heaps-total',
                 'input-dumps-total',
                 'output-bytes-total',
                 'output-chunks-total',
                 'output-seconds-total']:
        sensor = sensors[name]
        sensor.set_value(sensor.stype(0), timestamp=now)


@attr.s(frozen=True)
class Array:
    """A single array being received over SPEAD. See :class:`.Rechunker` for details."""

    name = attr.ib()         # Excludes the prefix
    in_chunks = attr.ib()
    out_chunks = attr.ib()
    fill_value = attr.ib()
    dtype = attr.ib(converter=_dtype_converter)

    @property
    def substreams(self):
        return int(np.product([len(c) for c in self.in_chunks]))

    @property
    def shape(self):
        return tuple(sum(c) for c in self.in_chunks)

    @property
    def nbytes(self):
        return int(np.product(self.shape)) * self.dtype.itemsize


class ChunkStoreRechunker(rechunk.Rechunker):
    """Rechunker that outputs data to a chunk store.

    The name is used as the array name in the chunk store.
    """
    def __init__(
            self, chunk_store: Any, sensors: SensorSet, name: str,
            in_chunks: Chunks, out_chunks: Chunks,
            fill_value: Any, dtype: Any) -> None:
        super().__init__(name, in_chunks, out_chunks, fill_value, dtype)
        self.chunk_store = chunk_store
        self.chunk_store.create_array(self.name)
        self.sensors = sensors

    def output(self, offset: Offset, value: np.ndarray) -> None:
        start = time.monotonic()
        slices = tuple(slice(ofs, ofs + size) for ofs, size in zip(offset, value.shape))
        self.chunk_store.put_chunk(self.name, slices, value)
        end = time.monotonic()
        self.sensors['output-chunks-total'].value += 1
        self.sensors['output-bytes-total'].value += value.nbytes
        self.sensors['output-seconds-total'].value += end - start

    def out_of_order(self, received: int, seen: int) -> None:
        self.sensors['input-too-old-heaps-total'].value += 1


class RechunkerGroup:
    """Collects a number of rechunkers with common input chunk scheme.

    The arrays need not all have the same shape. However, there must be a
    prefix of the axes on which they all have the same chunking scheme, and
    on the remaining axes there can only be a single chunk. For example, the
    following chunking schemes could co-exist in a group.
    - ((2, 2), (3, 3, 3))
    - ((2, 2), (3, 3, 3), (4,), (3,))
    - ((2, 2), (3, 3, 3), (6,))

    Parameters
    ----------
    chunk_store
        Chunk-store into which output chunks are written.
    sensors
        Sensor set containing an ``input-dumps-total`` sensor, which will
        be updated to reflect the highest dump index seen.
    prefix
        Prefix for naming arrays in the chunk store. It is prepended to the
        names given in `arrays` when storing the chunks.
    arrays
        Descriptions of the incoming arrays.
    """
    def __init__(self, chunk_store: katdal.chunkstore.ChunkStore,
                 sensors: SensorSet, prefix: str,
                 arrays: Sequence[Array]) -> None:
        self.prefix = prefix
        self.arrays = list(arrays)
        self.sensors = sensors
        self._rechunkers = [
            ChunkStoreRechunker(chunk_store, sensors, chunk_store.join(prefix, a.name),
                                a.in_chunks, a.out_chunks,
                                a.fill_value, a.dtype) for a in arrays]

    def add(self, offset_prefix: Offset, values: Iterable[np.ndarray]) -> None:
        """Add a value per array for rechunking.

        For each array passed to the constructor, there must be corresponding
        element in `values`. Each such value has an offset given by
        `offset_prefix` plus enough 0's to match the dimensionality.
        """
        dump_index = offset_prefix[0]
        if dump_index >= self.sensors['input-dumps-total'].value:
            self.sensors['input-dumps-total'].value = dump_index + 1
        for rechunker, value in zip(self._rechunkers, values):
            offset = offset_prefix + (0,) * (value.ndim - len(offset_prefix))
            rechunker.add(offset, value)

    def get_chunk_info(self) -> Dict[str, Dict[str, Any]]:
        """Get the chunk information to place into telstate to describe the arrays.

        This closes the rechunkers (flushing partial output chunks), so no
        further calls to :meth:`add` should be made.
        """
        for rechunker in self._rechunkers:
            rechunker.close()
        return {array.name: rechunker.get_chunk_info(self.prefix)
                for array, rechunker in zip(self.arrays, self._rechunkers)}


class SpeadWriter:
    """Base class to receive data over SPEAD and write it to a chunk store.

    It supports multiplexing between instances of :class:`RechunkerGroup` based
    on contents of the SPEAD heaps. This is implemented by subclassing and
    overriding :meth:`rechunker_group`.

    Parameters
    ----------
    sensors
        Server sensors including all those returned by :meth:`io_sensors`.
        These are updated as heaps are received.
    rx
        SPEAD receiver. It should be set up with :attr:`stop_on_stop_item` set
        to false. :meth:`make_receiver` returns a suitable receiver with
        optimised memory pool allocations.
    """
    def __init__(self, sensors: SensorSet, rx: spead2.recv.asyncio.Stream) -> None:
        self.sensors = sensors
        self.rx = rx

    async def run(self, stops: int = None) -> None:
        """Run the receiver.

        Parameters
        ----------
        stops
            If specified, this method will stop once it has seen `stops` stop
            items. Otherwise, it will run until cancelled or :meth:`stop` is
            called.
        """
        first = True
        n_stop = 0
        ig = spead2.ItemGroup()
        async for heap in self.rx:
            if first:
                self.first_heap()
                first = False
            updated = {}   # type: Dict[str, spead2.Item]
            if heap.is_end_of_stream():
                n_stop += 1
                if stops is not None and n_stop == stops:
                    self.rx.stop()
                    break
                else:
                    updated = {}
            elif isinstance(heap, spead2.recv.IncompleteHeap):
                self.sensors['input-incomplete-heaps-total'].value += 1
            else:
                try:
                    updated = ig.update(heap)
                except Exception:
                    logger.exception('Invalid heap')

            if 'timestamp' in updated:
                channel0 = int(updated['frequency'].value)
                dump_index = int(updated['dump_index'].value)
                group = self.rechunker_group(updated)
                # Check if subclass decided the heap was good
                if group is not None:
                    # Get values and add time dimension
                    values = [ig[array.name].value[np.newaxis, ...] for array in group.arrays]
                    nbytes = sum(value.nbytes for value in values)
                    group.add((dump_index, channel0), values)
                    self.sensors['input-heaps-total'].value += 1
                    self.sensors['input-bytes-total'].value += nbytes

    def stop(self) -> None:
        """Gracefully stop :meth:`run`."""
        self.rx.stop()

    def first_heap(self):
        """Callback to notify about the first heap being received.

        The default does nothing, but may be overridden
        """
        pass    # pragma: no cover

    def rechunker_group(self, updated: Dict[str, spead2.Item]) -> Optional[RechunkerGroup]:
        """Obtain the rechunker group associated with a particular heap.

        This must be implemented in derived classes.
        """
        raise NotImplementedError    # pragma: no cover

    def write_complete_marker(self, chunk_store: katdal.chunkstore.ChunkStore,
                              capture_stream_name: str) -> None:
        touch_file = os.path.join(chunk_store.path, capture_stream_name, "complete")
        os.makedirs(os.path.dirname(touch_file), exist_ok=True)
        with open(touch_file, 'a'):
            os.utime(touch_file, None)


def chunks_from_telstate(telstate):
    """Determine input chunking scheme for visibility data from telescope state.

    The provided `telstate` must be a view of the appropriate stream.

    Raises
    ------
    KeyError
        if any of the necessary telescope state keys are missing.
    """
    try:
        n_chans = telstate['n_chans']
        n_bls = telstate['n_bls']
        n_chans_per_substream = telstate['n_chans_per_substream']
    except KeyError:
        logger.error("Unable to find sizing params (n_bls, n_chans, "
                     "or n_chans_per_substream) in telstate.")
        raise

    n_substreams = n_chans // n_chans_per_substream
    return ((1,), (n_chans_per_substream,) * n_substreams, (n_bls,))


def make_receiver(endpoints: Sequence[Endpoint],
                  arrays: Sequence[Array],
                  interface_address: Optional[str],
                  ibv: bool,
                  max_heaps_per_substream: int = 2,
                  ring_heaps_per_substream: int = 8):
    """Generate a SPEAD receiver suitable for :class:`SpeadWriter`.

    Parameters
    ----------
    endpoints
        Multicast UDP endpoints to subscribe to
    arrays
        Arrays that will arrive in each heap
    interface_address
        If given, IP address of a local interface to bind to
    ibv
        If true, use ibverbs acceleration (see SPEAD documentation)
    max_heaps_per_substream
        Number of simultaneously incomplete SPEAD heaps allowed per substream
    ring_heaps_per_substream
        Number of complete heaps allowed in the SPEAD ringbuffer, per substream
    """
    n_substreams = arrays[0].substreams

    max_heaps = max_heaps_per_substream * n_substreams
    ring_heaps = ring_heaps_per_substream * n_substreams
    rx = spead2.recv.asyncio.Stream(spead2.ThreadPool(),
                                    max_heaps=max_heaps,
                                    ring_heaps=ring_heaps,
                                    contiguous_only=False)
    n_memory_buffers = max_heaps + ring_heaps + 2
    heap_size = sum(a.nbytes for a in arrays)
    memory_pool = spead2.MemoryPool(heap_size, heap_size + 4096,
                                    n_memory_buffers, n_memory_buffers)
    rx.set_memory_pool(memory_pool)
    rx.set_memcpy(spead2.MEMCPY_NONTEMPORAL)
    rx.stop_on_stop_item = False
    if ibv:
        endpoint_tuples = [(endpoint.host, endpoint.port) for endpoint in endpoints]
        rx.add_udp_ibv_reader(endpoint_tuples, interface_address,
                              buffer_size=64 * 1024**2)
    else:
        for endpoint in endpoints:
            if interface_address is not None:
                rx.add_udp_reader(endpoint.host, endpoint.port,
                                  buffer_size=heap_size + 4096,
                                  interface_address=interface_address)
            else:
                rx.add_udp_reader(endpoint.port, bind_hostname=endpoint.host,
                                  buffer_size=heap_size + 4096)
    return rx
