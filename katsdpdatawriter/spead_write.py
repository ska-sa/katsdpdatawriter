"""
Receive heaps from a SPEAD stream and write corresponding data to a chunk store.
"""

import copy
import argparse
import os.path
import time
import enum
import logging
import concurrent.futures
import asyncio
import socket
import functools
from collections import Counter
from typing import (Optional, Any, Sequence, Iterable,           # noqa: F401
                    Mapping, MutableMapping, Set, Dict, Tuple)

import numpy as np
import attr
import aiomonitor
from aiokatcp import Sensor, SensorSet
import spead2
import spead2.recv.asyncio
import katdal.chunkstore
import katdal.chunkstore_npy
import katdal.chunkstore_s3
import katsdptelstate
from katsdptelstate.endpoint import Endpoint

from . import rechunk
from .rechunk import Chunks, Offset
from .queue_space import QueueSpace


logger = logging.getLogger(__name__)


@attr.s
class ChunkParams:
    max_size = attr.ib()                    # type: int     # maximum size in bytes
    max_channels = attr.ib(default=None)    # type: Optional[int]
    max_dumps = attr.ib(default=None)       # type: Optional[int]

    @staticmethod
    def from_args(args: argparse.Namespace):
        """Create from command-line arguments (see :func:`add_common_args`)"""
        if args.buffer_dumps < args.obj_max_dumps:
            logger.warning('Decreasing --obj-max-dumps to match --buffer-dumps (%d)',
                           args.buffer_dumps)
        max_dumps = min(args.obj_max_dumps, args.buffer_dumps)
        return ChunkParams(args.obj_size_mb * 1e6, args.obj_max_channels, max_dumps)


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
            int, "input-missing-heaps-total",
            "Number of gaps in the heaps seen. (prometheus: counter)",
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
            "s"),
        Sensor(
            float, "output-seconds",
            "Time spent on the last chunk write.",
            "s"),
        Sensor(
            int, "active-chunks",
            "Number of chunks currently being written. (prometheus: gauge)"),
        Sensor(
            float, "queued-bytes",
            "Number of bytes that have been received but not yet written. (prometheus: gauge)"),
        Sensor(
            float, "max-queued-bytes",
            "Maximum value of queued-bytes sensor for this capture block. (prometheus: gauge)")
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
                 'input-missing-heaps-total',
                 'input-bytes-total',
                 'input-heaps-total',
                 'input-dumps-total',
                 'output-bytes-total',
                 'output-chunks-total',
                 'output-seconds-total',
                 'active-chunks',
                 'queued-bytes',
                 'max-queued-bytes']:
        sensor = sensors[name]
        sensor.set_value(sensor.stype(0), timestamp=now)


@attr.s(frozen=True)
class Array:
    """A single array being received over SPEAD. See :class:`.Rechunker` for details."""

    name = attr.ib()         # type: str # Excludes the prefix
    in_chunks = attr.ib()    # type: Tuple[Tuple[int, ...], ...]
    out_chunks = attr.ib()   # type: Tuple[Tuple[int, ...], ...]
    fill_value = attr.ib()   # type: Any
    dtype = attr.ib(converter=_dtype_converter)    # type: np.dtype

    @property
    def substreams(self):
        return int(np.product([len(c) for c in self.in_chunks]))

    @property
    def shape(self):
        """Shape of a single input dump"""
        return tuple(sum(c) for c in self.in_chunks)

    @property
    def nbytes(self):
        """Number of bytes in a single input dump"""
        return int(np.product(self.shape)) * self.dtype.itemsize


def make_array(name, in_chunks: Tuple[Tuple[int, ...], ...],
               fill_value: Any, dtype: Any, chunk_params: ChunkParams) -> Array:
    """Create an :class:`Array` with computed output chunk scheme.

    The output chunks are determined by splitting the input chunks along axes 0
    and 1 (time and frequency in typical use) to produce chunks subject to the
    constraints of `chunk_params`.
    """
    # Shape of a single input chunk
    assert in_chunks[0] == (1,)     # Only one chunk in time, with one dump
    shape = tuple(c[0] for c in in_chunks)
    if chunk_params.max_channels is not None:
        max_dim_elements = {1: chunk_params.max_channels}
    else:
        max_dim_elements = {}
    # Compute the decomposition of each input chunk
    chunks = katdal.chunkstore.generate_chunks(
        shape, dtype, chunk_params.max_size,
        dims_to_split=(0, 1), power_of_two=True,
        max_dim_elements=max_dim_elements)  # type: Tuple[Tuple[int, ...], ...]
    # Accumulate in time to make up the chunk size
    chunk_size = np.dtype(dtype).itemsize * np.prod([c[0] for c in chunks])
    n_time = 1
    while (chunk_size * 2 <= chunk_params.max_size
           and (chunk_params.max_dumps is None or n_time * 2 <= chunk_params.max_dumps)):
        n_time *= 2
        chunk_size *= 2
    # the ignore is to suppress see https://github.com/python/mypy/issues/6337
    chunks = ((n_time,),) + chunks[1:]   # type: ignore
    logger.info('Using chunks of shape %s (%.3fMB) for %s',
                tuple(c[0] for c in chunks), chunk_size / 1e6, name)
    # Repeat for each input chunk
    out_chunks = tuple(outc * len(inc) for inc, outc in zip(in_chunks, chunks))
    return Array(name, in_chunks, out_chunks, fill_value, dtype)


class ChunkStoreRechunker(rechunk.Rechunker):
    """Rechunker that outputs data to a chunk store via an executor.

    The name is used as the array name in the chunk store.

    .. note::

       The :meth:`output` coroutine will return as soon as it has posted the
       chunk to the executor. It only blocks to acquire from the
       `executor_queue_space`.
    """
    def __init__(
            self,
            executor: concurrent.futures.Executor,
            executor_queue_space: QueueSpace,
            chunk_store: katdal.chunkstore.ChunkStore,
            sensors: SensorSet, name: str,
            in_chunks: Chunks, out_chunks: Chunks,
            fill_value: Any, dtype: Any) -> None:
        super().__init__(name, in_chunks, out_chunks, fill_value, dtype)
        self.executor = executor
        self.executor_queue_space = executor_queue_space
        self.chunk_store = chunk_store
        self.chunk_store.create_array(self.name)
        self.sensors = sensors
        self._futures = set()    # type: Set[asyncio.Future[float]]
        self._loop = asyncio.get_event_loop()

    def _put_chunk(self, slices: Tuple[slice, ...], value: np.ndarray) -> float:
        """Put a chunk into the chunk store and return statistics.

        This is run in a separate thread, using an executor.
        """
        def increment_active_chunks():
            self.sensors['active-chunks'].value += 1

        def decrement_active_chunks():
            self.sensors['active-chunks'].value -= 1

        start = time.monotonic()
        self._loop.call_soon_threadsafe(increment_active_chunks)
        try:
            self.chunk_store.put_chunk(self.name, slices, value)
        finally:
            self._loop.call_soon_threadsafe(decrement_active_chunks)
        end = time.monotonic()
        return end - start

    def _update_stats(self, nbytes: int, future: 'asyncio.Future[float]') -> None:
        """Done callback for a future running :meth:`_put_chunk`.

        This is run on the event loop, so can safely update sensors. It also
        logs any errors.
        """
        self._futures.remove(future)
        self.executor_queue_space.release(nbytes)
        self.sensors['queued-bytes'].value -= nbytes
        try:
            elapsed = future.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception('Failed to write a chunk to %s', self.name)
            self.sensors['device-status'].value = DeviceStatus.FAIL
        else:
            self.sensors['output-chunks-total'].value += 1
            self.sensors['output-bytes-total'].value += nbytes
            self.sensors['output-seconds-total'].value += elapsed
            self.sensors['output-seconds'].value = elapsed

    async def output(self, offset: Offset, value: np.ndarray) -> None:
        slices = tuple(slice(ofs, ofs + size) for ofs, size in zip(offset, value.shape))
        await self.executor_queue_space.acquire(value.nbytes)
        future = asyncio.ensure_future(
            self._loop.run_in_executor(self.executor, self._put_chunk, slices, value))
        self._futures.add(future)
        future.add_done_callback(functools.partial(self._update_stats, value.nbytes))

    def out_of_order(self, received: int, seen: int) -> None:
        self.sensors['input-too-old-heaps-total'].value += 1

    async def close(self) -> None:
        """Close and wait for all asynchronous writes to complete."""
        await super().close()
        # asyncio.wait is implemented by adding a done callback to each
        # future. Done callbacks are run in order of addition, so when
        # wait returns, we are guaranteed that the done callbacks have
        # run.
        if self._futures:
            await asyncio.wait(self._futures)


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
    executor
        Executor used for asynchronous writes to the chunk store.
    executor_queue_space
        :class:`QueueSpace` bounding the number of bytes that can be in flight
        within `executor`.
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
    def __init__(self,
                 executor: concurrent.futures.Executor,
                 executor_queue_space: QueueSpace,
                 chunk_store: katdal.chunkstore.ChunkStore,
                 sensors: SensorSet, prefix: str,
                 arrays: Sequence[Array]) -> None:
        self.prefix = prefix
        self.arrays = list(arrays)
        self.sensors = sensors
        self._expected = Counter()    # type: MutableMapping[Offset, int]
        self._seen = Counter()        # type: MutableMapping[Offset, int]
        self._rechunkers = [
            ChunkStoreRechunker(executor, executor_queue_space,
                                chunk_store, sensors,
                                chunk_store.join(prefix, a.name),
                                a.in_chunks, a.out_chunks,
                                a.fill_value, a.dtype) for a in arrays]

    async def add(self, offset_prefix: Offset, values: Iterable[np.ndarray]) -> None:
        """Add a value per array for rechunking.

        For each array passed to the constructor, there must be corresponding
        element in `values`. Each such value has an offset given by
        `offset_prefix` plus enough 0's to match the dimensionality.
        """
        dump_index = offset_prefix[0]
        if dump_index >= self.sensors['input-dumps-total'].value:
            self.sensors['input-dumps-total'].value = dump_index + 1

        nbytes = sum(value.nbytes for value in values)
        self.sensors['input-heaps-total'].value += 1
        self.sensors['input-bytes-total'].value += nbytes
        queued_bytes = self.sensors['queued-bytes'].value + nbytes
        self.sensors['queued-bytes'].value = queued_bytes
        if queued_bytes > self.sensors['max-queued-bytes'].value:
            self.sensors['max-queued-bytes'].value = queued_bytes

        # Update our idea of how many heaps we've missed out on, assuming heaps
        # for each substream arrive in order.
        substream = offset_prefix[1:]
        old_missing = self._expected[substream] - self._seen[substream]
        if dump_index >= self._expected[substream]:
            self._expected[substream] = dump_index + 1
        self._seen[substream] += 1
        new_missing = self._expected[substream] - self._seen[substream]
        self.sensors['input-missing-heaps-total'].value += new_missing - old_missing

        for rechunker, value in zip(self._rechunkers, values):
            offset = offset_prefix + (0,) * (value.ndim - len(offset_prefix))
            await rechunker.add(offset, value)

    async def get_chunk_info(self) -> Dict[str, Dict[str, Any]]:
        """Get the chunk information to place into telstate to describe the arrays.

        This closes the rechunkers (flushing partial output chunks), so no
        further calls to :meth:`add` should be made.
        """
        for rechunker in self._rechunkers:
            await rechunker.close()
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
                    await group.add((dump_index, channel0), values)

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


def write_telstate(telstate: katsdptelstate.TelescopeState,
                   input_name: str, output_name: str, rename_src: Mapping[str, str],
                   s3_endpoint_url: Optional[str]) -> None:
    """Write telstate information about output stream."""
    telstate_out = telstate.view(output_name)
    if output_name != input_name:
        telstate_out.add('inherit', input_name, immutable=True)
        if rename_src:
            telstate_in = telstate.view(input_name)
            src_streams_in = telstate_in['src_streams']
            src_streams_out = [rename_src.get(stream, stream) for stream in src_streams_in]
            telstate_out.add('src_streams', src_streams_out, immutable=True)
    if s3_endpoint_url is not None:
        telstate_out.add('s3_endpoint_url', s3_endpoint_url, immutable=True)


def make_receiver(endpoints: Sequence[Endpoint],
                  arrays: Sequence[Array],
                  interface_address: Optional[str],
                  ibv: bool,
                  max_heaps_per_substream: int = 2,
                  ring_heaps_per_substream: int = 2) -> spead2.recv.asyncio.Stream:
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
    heap_size = sum(a.nbytes // a.substreams for a in arrays)
    memory_pool = spead2.MemoryPool(heap_size, heap_size + 4096,
                                    n_memory_buffers, n_memory_buffers)
    rx.set_memory_pool(memory_pool)
    rx.set_memcpy(spead2.MEMCPY_NONTEMPORAL)
    rx.stop_on_stop_item = False
    if ibv:
        # The main scripts check this; the assert keeps mypy happy
        assert interface_address is not None, "Interface address is required when using ibverbs"
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


class _DictAction(argparse.Action):
    """Argparse action that takes argument of form KEY:VALUE and updates a dict with it.

    The input value is expected to be a 2-tuple, so the type must be one that
    generates such a tuple.
    """
    def __init__(self, option_strings, dest, nargs=None, const=None, default=None,
                 type=None, choices=None, required=False, help=None, metavar=None):
        # This code is somewhat cargo-culted from _AppendAction in the argparse
        # source.
        if nargs == 0:
            raise ValueError('nargs for dict action must be > 0')
        if const is not None:
            raise ValueError('const is not supported for dict action')
        super().__init__(
                option_strings=option_strings,
                dest=dest,
                nargs=nargs,
                const=const,
                default=default,
                type=type,
                choices=choices,
                required=required,
                help=help,
                metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        d = getattr(namespace, self.dest, None)
        if d is None:
            d = {}
        else:
            d = copy.copy(d)
        d.update([values])
        setattr(namespace, self.dest, d)


def _split_colon(value):
    "Splits a KEY:VALUE string into its two parts"""
    parts = value.split(':')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('Expected exactly one colon in {!r}'.format(value))
    return parts


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Inject command-line arguments that are common to the writers"""
    group = parser.add_argument_group('Chunk store options')
    group.add_argument('--npy-path', metavar='PATH',
                       help='Write NPY files to this directory instead of '
                            'directly to object store')
    group.add_argument('--s3-endpoint-url', metavar='URL',
                       help='URL of S3 gateway to Ceph cluster')
    group.add_argument('--s3-access-key', metavar='KEY',
                       help='Access key for S3')
    group.add_argument('--s3-secret-key', metavar='KEY',
                       help='Secret key for S3')
    group.add_argument('--s3-expire', type=int, metavar='DAYS',
                       help='Days after which to expire the data')
    group.add_argument('--direct-write', action='store_true',
                       help='Use O_DIRECT for writing to .npy files')

    group = parser.add_argument_group('Instrumentation options')
    group.add_argument('--no-aiomonitor', dest='aiomonitor', action='store_false',
                       help='Disable aiomonitor debugging server')
    group.add_argument('--aiomonitor-port', type=int,
                       default=aiomonitor.MONITOR_PORT, metavar='PORT',
                       help='Port for aiomonitor [%(default)s]')
    group.add_argument('--aioconsole-port', type=int,
                       default=aiomonitor.CONSOLE_PORT, metavar='PORT',
                       help='Port for aioconsole [%(default)s]')
    group.add_argument('--dashboard-port', type=int, metavar='PORT',
                       help='Port for dashboard [disabled]')
    group.add_argument('--external-hostname', default=socket.getfqdn(), metavar='HOSTNAME',
                       help='Hostname through which the dashboard will be accessed [%(default)s]')
    group.add_argument('--dashboard-allow-websocket-origin', action='append', metavar='ORIGIN',
                       help='Origin at which the dashboard may be accessed'
                       ' (may be repeated) [auto]')

    parser.add_argument('--new-name', metavar='NAME',
                        help='Name for the output stream')
    parser.add_argument('--rename-src', metavar='OLD-NAME:NEW-NAME',
                        type=_split_colon, action=_DictAction,
                        help='Rewrite src_streams for new name (repeat for each rename)')
    parser.add_argument('--obj-size-mb', type=float, default=20., metavar='MB',
                        help='Target object size in MB [%(default)s]')
    parser.add_argument('--obj-max-channels', type=int, metavar='CHANNELS',
                        help='Maximum number of channels per object [no limit]')
    parser.add_argument('--obj-max-dumps', type=int, metavar='DUMPS', default=16,
                        help='Maximum number of dumps per object [%(default)s]')
    parser.add_argument('--workers', type=int, default=50,
                        help='Threads to use for writing chunks [%(default)s]')
    parser.add_argument('--buffer-dumps', type=int, default=20, metavar='DUMPS',
                        help='Number of full dumps to buffer in write queue')
    parser.add_argument('-p', '--port', type=int, metavar='N',
                        help='KATCP host port [%(default)s]')
    parser.add_argument('-a', '--host', default="", metavar='HOST',
                        help='KATCP host address [all hosts]')


def chunk_store_from_args(parser: argparse.ArgumentParser,
                          args: argparse.Namespace) -> katdal.chunkstore.ChunkStore:
    """Create a chunk store from user-provided arguments.

    This checks that a consistent set of the arguments created by
    :meth:`add_common_arguments` was given by the user. If not, it calls
    ``parser.error`` (which terminates the process). Otherwise, it returns a
    new chunk store (any exceptions from the chunk store constructor are passed
    through.
    """
    if not args.npy_path:
        for arg_name in ['s3_endpoint_url', 's3_access_key', 's3_secret_key']:
            if not getattr(args, arg_name):
                parser.error('--{} is required if --npy-path is not given'
                             .format(arg_name.replace('_', '-')))
                # Real parser.error kills the process, but the unit tests mock
                # it and so we want to ensure that we don't carry on.
    else:
        if not os.path.isdir(args.npy_path):
            parser.error("Specified --npy-path ({}) does not exist.".format(args.npy_path))

    if args.npy_path:
        chunk_store = katdal.chunkstore_npy.NpyFileChunkStore(
            args.npy_path, direct_write=args.direct_write)
    else:
        chunk_store = katdal.chunkstore_s3.S3ChunkStore.from_url(
            args.s3_endpoint_url,
            credentials=(args.s3_access_key, args.s3_secret_key),
            expiry_days=args.s3_expire or 0)
    return chunk_store
