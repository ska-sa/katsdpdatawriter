from unittest import mock

import numpy as np
from nose.tools import assert_equal, assert_raises

from .. import rechunk


def test_offset_to_size_1d():
    out = rechunk._offset_to_size_1d((1, 5, 7, 4, 2))
    assert_equal(out, {0: 1, 1: 5, 6: 7, 13: 4, 17: 2})


def test_split_chunks_1d():
    out = rechunk._split_chunks_1d((4, 6, 2), (1, 3, 2, 2, 2, 2))
    assert_equal(
        out,
        {
            0: (slice(0, 1), slice(1, 4)),
            4: (slice(0, 2), slice(2, 4), slice(4, 6)),
            10: (slice(0, 2),)
        })


def test_split_chunks_1d_out_chunks_too_short():
    with assert_raises(ValueError):
        rechunk._split_chunks_1d((4, 6, 2), (1, 3, 2, 2, 2, 1))


def test_split_chunks_1d_out_chunks_too_long():
    with assert_raises(ValueError):
        rechunk._split_chunks_1d((4, 6, 2), (1, 3, 2, 2, 2, 2, 4))


def test_split_chunks_1d_misaligned():
    with assert_raises(ValueError):
        # out_chunks not aligned
        rechunk._split_chunks_1d((4, 6, 2), (1, 4, 1, 2, 2, 2))


class MockRechunker(rechunk.Rechunker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def output(self, offset, value):
        self.calls.append((offset, value.copy()))


class BaseTestRechunker:
    def setup(self):
        self.data = np.arange(64).reshape(4, 8, 2).astype(np.uint8)
        self.expected = np.full_like(self.data, 253, np.uint8)

    def send_chunk(self, offset):
        idx = tuple(slice(ofs, ofs + size) for ofs, size in zip(offset, (1, 4, 2)))
        value = self.data[idx]
        self.r.add(offset, value)
        self.expected[idx] = self.data[idx]

    def check_values(self):
        # Checks that the calls contain the expected values given the
        # data send. Does NOT check that the offsets and sizes correspond
        # correctly to chunks.
        for call in self.r.calls:
            idx = tuple(slice(ofs, ofs + size) for ofs, size in zip(call[0], call[1].shape))
            expected = self.expected[idx]
            np.testing.assert_array_equal(expected, call[1])

    def test_add_bad_offset(self):
        with assert_raises(KeyError):
            self.r.add((0, 2, 0), np.zeros((1, 2, 2), np.uint8))
        with assert_raises(ValueError):
            self.r.add((0, 0), np.zeros((1, 2, 2), np.uint8))

    def test_add_bad_shape(self):
        with assert_raises(ValueError):
            self.r.add((0, 0, 0), np.zeros((1, 2, 2), np.uint8))
        with assert_raises(ValueError):
            self.r.add((0, 0, 0), np.zeros((2, 4, 2), np.uint8))


class TestRechunker(BaseTestRechunker):
    def setup(self):
        self.r = MockRechunker('flags',
                               ((1,), (4, 4), (2,)),
                               ((2,), (2, 2, 4), (2,)),
                               253, np.uint8)
        super().setup()

    def test_end_partial(self, reorder=False):
        if reorder:
            for i in range(3):
                self.send_chunk((i, 0, 0))
            for i in range(3):
                self.send_chunk((i, 4, 0))
        else:
            for i in range(3):
                self.send_chunk((i, 0, 0))
                self.send_chunk((i, 4, 0))
        self.r.close()
        offsets = [call[0] for call in self.r.calls]
        shapes = [call[1].shape for call in self.r.calls]
        assert_equal(
            offsets,
            [(0, 0, 0), (0, 2, 0), (0, 4, 0),
             (2, 0, 0), (2, 2, 0), (2, 4, 0)])
        assert_equal(
            shapes,
            [(2, 2, 2), (2, 2, 2), (2, 4, 2),
             (1, 2, 2), (1, 2, 2), (1, 4, 2)])
        self.check_values()
        assert_equal(
            self.r.get_chunk_info('flags'),
            {
                'prefix': 'flags',
                'chunks': ((2, 1), (2, 2, 4), (2,)),
                'shape': (3, 8, 2),
                'dtype': np.uint8
            })

    def test_end_full(self):
        for i in range(4):
            self.send_chunk((i, 0, 0))
            self.send_chunk((i, 4, 0))
        self.r.close()
        offsets = [call[0] for call in self.r.calls]
        shapes = [call[1].shape for call in self.r.calls]
        assert_equal(
            offsets,
            [(0, 0, 0), (0, 2, 0), (0, 4, 0),
             (2, 0, 0), (2, 2, 0), (2, 4, 0)])
        assert_equal(
            shapes,
            [(2, 2, 2), (2, 2, 2), (2, 4, 2),
             (2, 2, 2), (2, 2, 2), (2, 4, 2)])
        self.check_values()
        assert_equal(
            self.r.get_chunk_info('flags'),
            {
                'prefix': 'flags',
                'chunks': ((2, 2), (2, 2, 4), (2,)),
                'shape': (4, 8, 2),
                'dtype': np.uint8
            })

    def test_reorder(self):
        self.test_end_partial(reorder=True)

    def test_out_of_order(self):
        with mock.patch.object(self.r, 'out_of_order'):
            self.send_chunk((2, 0, 0))
            self.send_chunk((0, 0, 0))
            self.r.out_of_order.assert_called_with(0, 2)

    def test_missing(self):
        self.send_chunk((1, 0, 0))
        self.send_chunk((2, 4, 0))
        self.r.close()

        offsets = [call[0] for call in self.r.calls]
        shapes = [call[1].shape for call in self.r.calls]
        assert_equal(
            offsets,
            [(0, 0, 0), (0, 2, 0), (2, 4, 0)])
        assert_equal(
            shapes,
            [(2, 2, 2), (2, 2, 2), (1, 4, 2)])
        self.check_values()
        assert_equal(
            self.r.get_chunk_info('flags'),
            {
                'prefix': 'flags',
                'chunks': ((2, 1), (2, 2, 4), (2,)),
                'shape': (3, 8, 2),
                'dtype': np.uint8
            })

    def test_bad_in_chunks(self):
        with assert_raises(ValueError):
            # in_chunks does not start with (1,)
            MockRechunker('foo', ((2,), (4, 4)), ((2,), (4, 4)), 253, np.uint8)
        with assert_raises(ValueError):
            # zero-sized chunks
            MockRechunker('foo', ((1,), (4, 4, 0)), ((2,), (4, 4)), 253, np.uint8)

    def test_bad_out_chunks(self):
        with assert_raises(ValueError):
            # does not start with singleton
            MockRechunker('foo', ((1,), (4, 4)), ((2, 2), (4, 4)), 253, np.uint8)

    def test_mismatched_chunks(self):
        with assert_raises(ValueError):
            # Dimensions don't match
            MockRechunker('foo', ((1,), (4, 4)), ((2,), (4, 4), (2,)), 253, np.uint8)
        with assert_raises(ValueError):
            # Lengths don't match
            MockRechunker('foo', ((1,), (4, 4)), ((2,), (4, 4, 1)), 253, np.uint8)
        with assert_raises(ValueError):
            # Chunks don't align
            MockRechunker('foo', ((1,), (4, 4)), ((2,), (3, 5)), 253, np.uint8)


class TestRechunkerNoAccum(BaseTestRechunker):
    def setup(self):
        self.r = MockRechunker('flags',
                               ((1,), (4, 4), (2,)),
                               ((1,), (2, 2, 4), (2,)),
                               253, np.uint8)
        super().setup()

    def test(self):
        for i in range(2):
            self.send_chunk((i, 0, 0))
            self.send_chunk((i, 4, 0))
        self.r.close()
        offsets = [call[0] for call in self.r.calls]
        shapes = [call[1].shape for call in self.r.calls]
        assert_equal(
            offsets,
            [(0, 0, 0), (0, 2, 0), (0, 4, 0),
             (1, 0, 0), (1, 2, 0), (1, 4, 0)])
        assert_equal(
            shapes,
            [(1, 2, 2), (1, 2, 2), (1, 4, 2),
             (1, 2, 2), (1, 2, 2), (1, 4, 2)])
        self.check_values()
        assert_equal(
            self.r.get_chunk_info('flags'),
            {
                'prefix': 'flags',
                'chunks': ((1, 1), (2, 2, 4), (2,)),
                'shape': (2, 8, 2),
                'dtype': np.uint8
            })
