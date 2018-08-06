"""Tests for :mod:`katsdpdatawriter.flag_writer`."""

import os.path
import tempfile
import shutil
import asyncio

import numpy as np
from nose.tools import (assert_equal, assert_in, assert_true,
                        assert_regex, assert_raises_regex, assert_logs)

import aiokatcp
import spead2
import spead2.send.asyncio
from katdal.chunkstore_npy import NpyFileChunkStore

from ..flag_writer import FlagWriterServer, Status
from .test_writer import BaseTestWriterServer


class TestFlagWriterServer(BaseTestWriterServer):
    async def setup_server(self) -> FlagWriterServer:
        server = FlagWriterServer(
            host='127.0.0.1', port=0, loop=self.loop, endpoints=self.endpoints,
            flag_interface='lo', flags_ibv=False, npy_path=self.npy_path,
            telstate=self.telstate, flags_name='sdp_l1_flags')
        await server.start()
        self.addCleanup(server.stop)
        return server

    def setup_ig(self) -> spead2.send.ItemGroup:
        self.cbid = '1234567890'
        n_chans_per_substream = self.telstate['n_chans_per_substream']
        n_bls = self.telstate['n_bls']
        flags = np.random.randint(0, 256, (n_chans_per_substream, n_bls), np.uint8)

        ig = spead2.send.ItemGroup()
        # This is copied and adapted from katsdpcal
        ig.add_item(id=None, name='flags', description="Flags for visibilities",
                    shape=(self.telstate['n_chans_per_substream'], self.telstate['n_bls']),
                    dtype=None, format=[('u', 8)], value=flags)
        ig.add_item(id=None, name='timestamp', description="Seconds since sync time",
                    shape=(), dtype=None, format=[('f', 64)], value=100.0)
        ig.add_item(id=None, name='dump_index', description='Index in time',
                    shape=(), dtype=None, format=[('u', 64)], value=0)
        ig.add_item(id=0x4103, name='frequency',
                    description="Channel index of first channel in the heap",
                    shape=(), dtype=np.uint32, value=0)
        ig.add_item(id=None, name='capture_block_id', description='SDP capture block ID',
                    shape=(None,), dtype=None, format=[('c', 8)], value=self.cbid)
        return ig

    async def stop_server(self) -> None:
        for queue in self.inproc_queues.values():
            queue.stop()
        await self.server.stop()
        await self.capture_task

    async def setUp(self) -> None:
        self.npy_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.npy_path)
        self.telstate = self.setup_telstate()
        self.setup_spead()
        self.server = await self.setup_server()
        self.client = await self.setup_client(self.server)
        self.ig = self.setup_ig()
        self.capture_task = self.loop.create_task(self.server.do_capture())

    async def test_capture(self) -> None:
        n_chans = self.telstate['n_chans']
        n_chans_per_substream = self.telstate['n_chans_per_substream']
        n_bls = self.telstate['n_bls']
        self.assert_sensor_equals('status', Status.WAIT_DATA)
        self.assert_sensor_equals('capture-block-state', '{}')

        await self.client.request('capture-init', self.cbid)
        self.assert_sensor_equals('capture-block-state', '{"%s": "CAPTURING"}' % self.cbid)

        await self.tx[0].async_send_heap(self.ig.get_heap())
        await asyncio.sleep(0.5)  # Give time for the heap to arrive
        self.assert_sensor_equals('status', Status.CAPTURING)

        await self.client.request('capture-done', self.cbid)
        self.assert_sensor_equals('status', Status.CAPTURING)  # Should still be capturing
        self.assert_sensor_equals('capture-block-state', '{}')
        await self.stop_server()
        capture_stream = self.cbid + '_sdp_l1_flags'
        assert_true(os.path.exists(
            os.path.join(self.npy_path, capture_stream, 'complete')))
        store = NpyFileChunkStore(self.npy_path)
        chunk = store.get_chunk(
            store.join(capture_stream, 'flags'),
            np.s_[0:1, 0:n_chans_per_substream, 0:n_bls], np.uint8)
        np.testing.assert_array_equal(self.ig['flags'].value[np.newaxis], chunk)

        view = self.telstate.view(capture_stream)
        chunk_info = view['chunk_info']
        n_substreams = n_chans // n_chans_per_substream
        assert_equal(
            chunk_info,
            {
                'flags': {
                    'prefix': capture_stream,
                    'shape': (1, n_chans, n_bls),
                    'chunks': ((1,), (n_chans_per_substream,) * n_substreams, (n_bls,)),
                    'dtype': np.dtype(np.uint8)
                }
            })
        assert_in('chunk_info', view)

    async def test_double_init(self) -> None:
        await self.client.request('capture-init', self.cbid)
        with assert_raises_regex(aiokatcp.FailReply, 'already active'):
            await self.client.request('capture-init', self.cbid)
        self.assert_sensor_equals('capture-block-state', '{"%s": "CAPTURING"}' % self.cbid)

    async def test_done_without_init(self) -> None:
        with assert_raises_regex(aiokatcp.FailReply, 'unknown'):
            await self.client.request('capture-done', self.cbid)

    async def test_no_data(self) -> None:
        self.assert_sensor_equals('capture-block-state', '{}')
        await self.client.request('capture-init', self.cbid)
        self.assert_sensor_equals('capture-block-state', '{"%s": "CAPTURING"}' % self.cbid)
        with assert_logs('katsdpdatawriter.flag_writer', 'WARNING'):
            await self.client.request('capture-done', self.cbid)
        self.assert_sensor_equals('capture-block-state', '{}')

    async def test_data_after_done(self) -> None:
        await self.client.request('capture-init', self.cbid)
        await self.client.request('capture-done', self.cbid)
        with assert_logs('katsdpdatawriter.flag_writer', 'WARNING') as cm:
            await self.tx[0].async_send_heap(self.ig.get_heap())
            await asyncio.sleep(0.5)
        assert_regex(cm.output[0], 'outside of init/done')
