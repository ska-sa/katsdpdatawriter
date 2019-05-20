import argparse
from unittest import mock
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from nose.tools import assert_equal, assert_count_equal, assert_is_instance, assert_raises
import asynctest
from aiokatcp import SensorSet
from katdal.chunkstore import ChunkStore
import katsdpservices

from ..spead_write import (Array, RechunkerGroup, io_sensors,
                           add_common_args, chunk_store_from_args)
from ..rechunk import Offset
from ..queue_space import QueueSpace


class TestArray:
    def setup(self) -> None:
        self.array = Array(
            'foo',
            in_chunks=((1,), (4, 4, 4), (2, 2)),
            out_chunks=((2,), (2, 2, 4, 4), (2, 2)),
            fill_value=253,
            dtype=np.float32)

    def test_dtype(self) -> None:
        # Check that the converter converted a dtype-like to a real dtype
        assert_equal(self.array.dtype, np.dtype(np.float32))
        assert_is_instance(self.array.dtype, np.dtype)

    def test_substreams(self) -> None:
        assert_equal(self.array.substreams, 6)

    def test_shape(self) -> None:
        assert_equal(self.array.shape, (1, 12, 4))

    def test_nbytes(self) -> None:
        assert_equal(self.array.nbytes, 192)


def _join(*args: str) -> str:
    return '/'.join(args)


class TestRechunkerGroup(asynctest.TestCase):
    def setUp(self) -> None:
        self.chunk_store = mock.create_autospec(spec=ChunkStore, spec_set=True, instance=True)
        self.chunk_store.join = _join

        self.sensors = SensorSet(set())
        for sensor in io_sensors():
            self.sensors.add(sensor)

        self.arrays = [
            Array('weights',
                  ((1,), (4, 4), (2,)),
                  ((1,), (2, 2, 2, 2), (2,)),
                  0, np.uint8),
            Array('weights_channel',
                  ((1,), (4, 4)),
                  ((2,), (2, 2, 2, 2)),
                  0, np.float32)
        ]

        self.weights = np.arange(32).reshape(2, 8, 2).astype(np.uint8)
        self.weights_channel = np.arange(16).reshape(2, 8).astype(np.float32)

        self.executor = ThreadPoolExecutor(4)
        self.executor_queue_space = QueueSpace(5 * sum(array.nbytes for array in self.arrays))
        self.r = RechunkerGroup(self.executor, self.executor_queue_space,
                                self.chunk_store, self.sensors, 'prefix', self.arrays)

    def tearDown(self):
        self.executor.shutdown(wait=True)

    async def add_chunks(self, offset: Offset) -> None:
        slices = np.s_[offset[0]:offset[0]+1, offset[1]:offset[1]+4, :]
        weights = self.weights[slices]
        weights_channel = self.weights_channel[slices[:2]]
        await self.r.add(offset, [weights, weights_channel])

    async def test(self) -> None:
        for i in range(0, 8, 4):
            for j in range(2):
                await self.add_chunks((j, i))
        chunk_info = await self.r.get_chunk_info()

        expected_calls = []
        for i in range(0, 8, 4):
            for j in range(2):
                for k in range(i, i + 4, 2):
                    expected_calls.append(mock.call(
                        'prefix/weights', np.s_[j:j+1, k:k+2, 0:2], mock.ANY))
        for i in range(0, 8, 2):
            expected_calls.append(mock.call(
                'prefix/weights_channel', np.s_[0:2, i:i+2], mock.ANY))
        assert_count_equal(expected_calls, self.chunk_store.put_chunk.mock_calls)
        # Check the array values. assert_count_equal doesn't work well for this
        # because of how equality operators are implemented in numpy.
        for call in self.chunk_store.put_chunk.mock_calls:
            name, slices, value = call[1]
            if name == 'prefix/weights':
                np.testing.assert_array_equal(self.weights[slices], value)
            else:
                np.testing.assert_array_equal(self.weights_channel[slices], value)

        assert_equal(
            chunk_info,
            {
                'weights': {
                    'prefix': 'prefix',
                    'chunks': ((1, 1), (2, 2, 2, 2), (2,)),
                    'shape': (2, 8, 2),
                    'dtype': '|u1'
                },
                'weights_channel': {
                    'prefix': 'prefix',
                    'chunks': ((2,), (2, 2, 2, 2)),
                    'shape': (2, 8),
                    'dtype': '<f4'      # TODO: assumes little-endian hardware
                }
            })


# SpeadWriter gets exercised via its derived classes


class BadArguments(Exception):
    """Exception used in mock when replacing ArgumentParser.Error"""


@mock.patch.object(katsdpservices.ArgumentParser, 'error', side_effect=BadArguments)
class TestChunkStoreFromArgs:
    """Test both :meth:`.add_common_args` and :meth:`.chunk_store_from_args`"""
    def setup(self) -> None:
        self.parser = katsdpservices.ArgumentParser()
        add_common_args(self.parser)

    def test_missing_args(self, error):
        with assert_raises(BadArguments):
            chunk_store_from_args(self.parser, self.parser.parse_args([]))
        error.assert_called_with('--s3-endpoint-url is required if --npy-path is not given')
        with assert_raises(BadArguments):
            chunk_store_from_args(self.parser, self.parser.parse_args(
                ['--s3-endpoint-url', 'http://invalid/', '--s3-access-key', 'ACCESS']))
        error.assert_called_with('--s3-secret-key is required if --npy-path is not given')

    def test_missing_path(self, error):
        with assert_raises(BadArguments):
            chunk_store_from_args(self.parser, self.parser.parse_args(
                ['--npy-path=/doesnotexist']))
        error.assert_called_with('Specified --npy-path (/doesnotexist) does not exist.')

    def test_npy(self, error):
        with mock.patch('katdal.chunkstore_npy.NpyFileChunkStore') as m:
            chunk_store_from_args(self.parser, self.parser.parse_args(
                ['--npy-path=/']))
        m.assert_called_with('/', direct_write=False)

    def test_npy_direct_write(self, error):
        with mock.patch('katdal.chunkstore_npy.NpyFileChunkStore') as m:
            chunk_store_from_args(self.parser, self.parser.parse_args(
                ['--npy-path=/', '--direct-write']))
        m.assert_called_with('/', direct_write=True)

    def test_s3(self, error):
        with mock.patch('katdal.chunkstore_s3.S3ChunkStore.from_url') as m:
            chunk_store_from_args(self.parser, self.parser.parse_args(
                ['--s3-endpoint-url=https://s3.invalid',
                 '--s3-secret-key=S3CR3T', '--s3-access-key', 'ACCESS']))
        m.assert_called_with('https://s3.invalid', credentials=('ACCESS', 'S3CR3T'), expiry_days=0)

    def test_s3_expire(self, error):
        with mock.patch('katdal.chunkstore_s3.S3ChunkStore.from_url') as m:
            chunk_store_from_args(self.parser, self.parser.parse_args(
                ['--s3-endpoint-url=https://s3.invalid',
                 '--s3-secret-key=S3CR3T', '--s3-access-key', 'ACCESS',
                 '--s3-expiry-days=7']))
        m.assert_called_with('https://s3.invalid', credentials=('ACCESS', 'S3CR3T'), expiry_days=7)

    def test_rename_src(self, error):
        args = self.parser.parse_args([
            '--rename-src=foo:bar', '--rename-src', 'x:y',
            '--new-name', 'xyz'])
        assert_equal(args.rename_src, {'foo': 'bar', 'x': 'y'})

    def test_rename_src_bad_colons(self, error):
        with assert_raises(BadArguments):
            self.parser.parse_args(['--rename-src=foo:bar:baz', '--new-name', 'xyz'])
