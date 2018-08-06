"""
Receive heaps from a SPEAD stream and write corresponding data to a chunk store.
"""

import time
import logging
from typing import Optional, Any, Sequence, Iterable, Dict

import numpy as np
import attr
from aiokatcp import Sensor, SensorSet
import spead2
import spead2.recv.asyncio

from . import rechunk
from .rechunk import Chunks, Offset


logger = logging.getLogger(__name__)


def _warn_if_positive(value: float) -> Sensor.Status:
    return Sensor.Status.WARN if value > 0 else Sensor.Status.NOMINAL


# Just to work around https://github.com/python/mypy/issues/4729
def _dtype_converter(dtype: Any) -> np.dtype:
    return np.dtype(dtype)


def add_sensors(sensors: SensorSet) -> None:
    """Add input and output counters to a server's sensors."""
    sensors.add(Sensor(
        int, "input-incomplete-heaps-total",
        "Number of heaps dropped due to being incomplete. (prometheus: counter)",
        status_func=_warn_if_positive))
    sensors.add(Sensor(
        int, "input-bytes-total",
        "Number of payload bytes received in this session. (prometheus: counter)",
        "B"))
    sensors.add(Sensor(
        int, "input-heaps-total",
        "Number of input heaps captured in this session. (prometheus: counter)"))
    sensors.add(Sensor(
        int, "input-dumps-total",
        "Number of complete input dumps captured in this session. (prometheus: counter)"))

    sensors.add(Sensor(
        int, "output-bytes-total",
        "Number of payload bytes written to chunk store in this session. (prometheus: counter)",
        "B"))
    sensors.add(Sensor(
        int, "output-chunks-total",
        "Number of chunks written to chunk store in this session. (prometheus: counter)"))
    sensors.add(Sensor(
        float, "output-seconds-total",
        "Accumulated time spent writing flag dumps. (prometheus: counter)",
        "s"))


def clear_io_sensors(sensors: SensorSet) -> None:
    """Zero the input and output counters in a sensor set"""
    now = time.time()
    for name in ['input-incomplete-heaps-total',
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
        self.sensors = sensors

    def output(self, offset: Offset, value: np.ndarray) -> None:
        start = time.monotonic()
        slices = tuple(slice(ofs, ofs + size) for ofs, size in zip(offset, value.shape))
        self.chunk_store.put_chunk(self.name, slices, value)
        end = time.monotonic()
        self.sensors['output-chunks-total'].value += 1
        self.sensors['output-bytes-total'].value += value.nbytes
        self.sensors['output-seconds-total'].value += end - start


class RechunkerGroup:
    """Collects a number of rechunkers with common input chunk scheme"""
    def __init__(self, chunk_store: Any,
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
        dump_index = offset_prefix[0]
        if dump_index >= self.sensors['input-dumps-total'].value:
            self.sensors['input-dumps-total'].value = dump_index + 1
        for rechunker, value in zip(self._rechunkers, values):
            offset = offset_prefix + (0,) * (value.ndim - len(offset_prefix))
            rechunker.add(offset, value)

    def close(self) -> None:
        for rechunker in self._rechunkers:
            rechunker.close()

    def get_chunk_info(self) -> Dict[str, Dict[str, Any]]:
        return {array.name: rechunker.get_chunk_info(self.prefix)
                for array, rechunker in zip(self.arrays, self._rechunkers)}


class SpeadWriter:
    def __init__(self, sensors: SensorSet, rx: spead2.recv.asyncio.Stream) -> None:
        self.sensors = sensors
        self.rx = rx

    async def run(self, stops: int = None) -> None:
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
        self.rx.stop()

    def first_heap(self):
        """Callback to notify about the first heap being received.

        The default does nothing, but may be overridden
        """
        pass

    def rechunker_group(self, updated: Dict[str, spead2.Item]) -> Optional[RechunkerGroup]:
        """Obtain the rechunker group associated with a particular heap.

        This must be implemented in derived classes.
        """
        raise NotImplementedError


def chunks_from_telstate(telstate):
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


def make_receiver(endpoints, arrays, interface_address, ibv,
                  max_heaps_per_substream=2, ring_heaps_per_substream=8):
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
