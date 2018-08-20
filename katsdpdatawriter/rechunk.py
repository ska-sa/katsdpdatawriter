import logging
import itertools
from typing import Tuple, Dict, Any, Optional

import numpy as np


logger = logging.getLogger(__name__)


Offset = Tuple[int, ...]
Shape = Tuple[int, ...]
Chunks1D = Tuple[int, ...]
Chunks = Tuple[Chunks1D, ...]
Slices = Tuple[slice, ...]


def _offset_to_size_1d(chunks: Chunks1D) -> Dict[int, int]:
    """Maps offset of start of each chunk to the size of that chunk

    Parameters
    ----------
    chunks : tuple of int
        Chunk sizes

    Returns
    -------
    dict
    """
    out = {}
    cur = 0
    for c in chunks:
        if c <= 0:
            raise ValueError('Chunk sizes must be positive')
        out[cur] = c
        cur += c
    return out


def _offset_to_size(chunks: Chunks) -> Tuple[Dict[int, int], ...]:
    """Multi-dimensional version of :func:`_offset_to_size_1d`."""
    return tuple(_offset_to_size_1d(c) for c in chunks)


def _split_chunks_1d(in_chunks: Chunks1D, out_chunks: Chunks1D) -> Dict[int, Slices]:
    """
    Divide output chunks into groups that align to the input chunks.

    For each input chunk, a tuple of slices is generated to index within that
    input chunk. The result is a dictionary indexed by offset.

    >>> _split_chunks_1d((4, 6, 2), (1, 3, 2, 2, 2, 2))
    {
        0: (slice(0, 1), slice(1, 4)),
        4: (slice(0, 2), slice(2, 4), slice(4, 6)),
        10: (slice(0, 2),)
    }

    Raises
    ------
    ValueError
        if an output chunk spans multiple input chunks
    ValueError
        if ``sum(in_chunks) != sum(out_chunks)``
    """
    out = {}
    pos = 0
    if sum(in_chunks) != sum(out_chunks):
        raise ValueError('chunks imply different shapes')
    offset = 0
    for c in in_chunks:
        slices = []
        cur = 0
        while cur < c:
            oc = out_chunks[pos]
            pos += 1
            slices.append(slice(cur, cur + oc))
            cur += oc
        if cur > c:
            raise ValueError('input and output chunks do not align')
        out[offset] = tuple(slices)
        offset += c
    return out


def _split_chunks(in_chunks: Chunks, out_chunks: Chunks) -> Tuple[Dict[int, Slices], ...]:
    """Multi-dimensional version of :meth:`_split_chunks_1d`."""
    if len(in_chunks) != len(out_chunks):
        raise ValueError('in_chunks and out_chunks have different length')
    return tuple(_split_chunks_1d(*item) for item in zip(in_chunks, out_chunks))


class Rechunker:
    """
    Takes a stream of chunks and generates output with same data but new
    chunking scheme.

    This is similar in concept to dask's rechunk, but in a streaming fashion
    (with the assumption that time is on the first axis). It is more limited
    through: non-time axes can only be split, not re-combined. The time axis
    must be size-1 chunks on input, but can be larger on output (accumulation
    in time).

    Incoming chunks whose coordinates differ only in the time axis must be
    received in order (out-of-order chunks will be discarded). Chunks with
    different non-time coordinates are handled completely independently. This
    does not apply when no time accumulation is being done, in which case
    chunks can arrive in any order.

    Memory usage depends on whether accumulation-in-time is being done. If
    so, it stores data internally (enough for one complete output dump). If
    not, there is no internal data storage, and memory usage only scales with
    the metadata (number of chunks etc).

    Do not instantiate this class directly. Instead, subclass it and implement
    :meth:`output`.

    Parameters
    ----------
    name : str
        Name of this array (purely for logging)
    in_chunks : tuple of tuple of int
        Chunking scheme of the input. The first element must be ``(1,)``
    out_chunks : tuple of tuple of int
        Chunking scheme of the output. The first element must be a 1-tuple,
        with the value indicating the size of each chunk (except possibly
        the last) in time.
    fill_value
        Value to store where no input is received for some of the input
        chunks that combine to form an output chunk.
    dtype : numpy dtype
        Data type of the array

    Raises
    ------
    ValueError
        if the restrictions on the input and output chunks are not met
    """

    class _Item:
        """Intermediate chunk under construction.

        An intermediate chunk has the output chunk size in the time axis and
        the input chunk size in other axes.
        """
        def __init__(self, offset: Offset, initial_value: np.ndarray) -> None:
            self.offset = offset
            self.value = initial_value

        def add(self, offset: Offset, value: np.ndarray) -> None:
            """Add a new input chunk."""
            assert offset[1:] == self.offset[1:]
            if value.shape[1:] != self.value.shape[1:] or value.shape[0] != 1:
                raise ValueError('value has wrong shape')
            rel = offset[0] - self.offset[0]
            self.value[rel:rel+1] = value

    def __init__(self, name: str,
                 in_chunks: Chunks,
                 out_chunks: Chunks,
                 fill_value: Any, dtype: Any) -> None:
        if in_chunks[0] != (1,):
            raise ValueError('in_chunks does not start with (1,)')
        if len(out_chunks[0]) != 1:
            raise ValueError('out_chunks does not start with a singleton')

        self.name = name
        self.in_chunks = in_chunks
        self.out_chunks = out_chunks
        self.fill_value = fill_value
        self.dtype = np.dtype(dtype)
        self._items = {}   # type: Dict[Tuple[int, ...], Rechunker._Item]  # Indexed by offset[1:]
        self._sizes = _offset_to_size(in_chunks[1:])
        self._split_chunks = _split_chunks(in_chunks[1:], out_chunks[1:])
        self._time_accum = out_chunks[0][0]
        self._n_dumps = 0

    def out_of_order(self, received: int, seen: int) -> None:
        """Report a chunk received from the past.

        This can be overridden to change the reporting channel.
        """
        logger.warning(
            "Received old chunk for array %s (%d < %d)",
            self.name, received, seen)         # pragma: nocover

    def _item_shape(self, offset: Offset) -> Shape:
        """Expected shape for the :class:`Item` holding the input chunk starting at `offset`."""
        sizes = tuple(s[ofs] for ofs, s in zip(offset[1:], self._sizes))
        return (self._time_accum,) + sizes

    def _flush(self, item: _Item) -> None:
        """Send `item` to :meth:`output`."""
        slices = tuple(s[ofs] for ofs, s in zip(item.offset[1:], self._split_chunks))
        for idx in itertools.product(*slices):
            full_idx = np.index_exp[0:len(item.value)] + idx
            offset = tuple(s.start + offset for s, offset in zip(full_idx, item.offset))
            self.output(offset, item.value[full_idx])

    def _get_item(self, offset: Offset) -> Optional[_Item]:
        """Get the item that should hold the input chunk starting at `offset`.

        It returns ``None`` if the offset is too far in the past to be captured.
        """
        key = offset[1:]
        # Round down to the start of the accumulation
        item_offset = (offset[0] // self._time_accum * self._time_accum,) + key
        item = self._items.get(key)
        if item is None or item.offset[0] < item_offset[0]:
            if item is not None:
                self._flush(item)
                item.value = None   # Allow GC to reclaim memory now
            shape = self._item_shape(offset)
            initial_value = np.full(shape, self.fill_value, self.dtype)
            item = self._Item(item_offset, initial_value)
            self._items[key] = item
        elif item.offset[0] > item_offset[0]:
            self.out_of_order(offset[0], item.offset[0])
            item = None
        return item

    def add(self, offset: Offset, value: np.ndarray) -> None:
        """Add a new incoming chunk.

        Parameters
        ----------
        offset : tuple of int
            Start coordinates of the chunk. It must be aligned to be configured
            chunking scheme.
        values : array-like
            Values of the chunk.

        Raises
        ------
        ValueError
            if `offset` has the wrong number of dimensions
        ValueError
            if `value` has the wrong shape for `offset`
        KeyError
            if `offset` does not match the input chunking scheme
        """
        if len(offset) != len(self.in_chunks):
            raise ValueError('wrong number of dimensions')
        if self._time_accum > 1:
            item = self._get_item(offset)
            if item is not None:
                item.add(offset, value)
        else:
            shape = self._item_shape(offset)
            value = np.require(value, dtype=self.dtype)
            if value.shape != shape:
                raise ValueError('value has wrong shape')
            item = self._Item(offset, value)
            self._flush(item)
        self._n_dumps = max(self._n_dumps, offset[0] + 1)

    def close(self) -> None:
        """Flush out any partially buffered items"""
        for item in self._items.values():
            # Truncate to last seen dump
            times = self._n_dumps - item.offset[0]
            if times < item.value.shape[0]:
                item.value = item.value[:times]
            self._flush(item)
        self._items.clear()

    def _get_shape(self) -> Shape:
        return (self._n_dumps,) + tuple(sum(c) for c in self.out_chunks[1:])

    def _get_chunks(self) -> Chunks:
        c = self.out_chunks[0][0]
        full = self._n_dumps // c
        last = self._n_dumps % c
        if last > 0:
            time_chunks = (c,) * full + (last,)
        else:
            time_chunks = (c,) * full
        return (time_chunks,) + self.out_chunks[1:]

    def get_chunk_info(self, prefix: str) -> Dict[str, Any]:
        """Get chunk info to be placed into telstate to describe the output.

        Parameters
        ----------
        prefix : str
            The array name prefix to retrieve the chunks from the chunk store
        """
        return {
            'prefix': prefix,
            'dtype': self.dtype,
            'shape': self._get_shape(),
            'chunks': self._get_chunks()
        }

    def output(self, offset: Offset, value: np.ndarray) -> None:
        """Called with each output chunk."""
        raise NotImplementedError      # pragma: nocover
