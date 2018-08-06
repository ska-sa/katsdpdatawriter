"""Tests for :mod:`katsdpdatawriter.vis_writer`."""

import tempfile
import shutil
import asyncio

import numpy as np
import katdal.chunkstore_npy
import spead2.send.asyncio
from aiokatcp import FailReply
from nose.tools import assert_raises_regex

from ..vis_writer import VisibilityWriterServer, Status
from .test_writer import BaseTestWriterServer


class TestVisWriterServer(BaseTestWriterServer):
    async def setup_server(self) -> VisibilityWriterServer:
        server = VisibilityWriterServer(
            host='127.0.0.1', port=0, loop=self.loop, endpoints=self.endpoints,
            interface='lo', ibv=False, chunk_store=self.chunk_store, chunk_size=10000,
            telstate_l0=self.telstate, stream_name='sdp_l0')
        await server.start()
        self.addCleanup(server.stop)
        return server

    def setup_ig(self) -> spead2.send.ItemGroup:
        n_chans_per_substream = self.telstate['n_chans_per_substream']
        n_bls = self.telstate['n_bls']
        shape = (n_chans_per_substream, n_bls)
        vis = np.zeros(shape, np.complex64)
        flags = np.random.randint(0, 256, shape, np.uint8)
        weights = np.random.randint(0, 256, shape, np.uint8)
        weights_channel = np.random.random(n_chans_per_substream).astype(np.float32)
        # Adapted from katsdpingest/sender.py
        ig = spead2.send.ItemGroup()
        ig.add_item(id=None, name='correlator_data',
                    description="Visibilities",
                    shape=(n_chans_per_substream, n_bls), dtype=np.complex64,
                    value=vis)
        ig.add_item(id=None, name='flags',
                    description="Flags for visibilities",
                    shape=(n_chans_per_substream, n_bls), dtype=np.uint8,
                    value=flags)
        ig.add_item(id=None, name='weights',
                    description="Detailed weights, to be scaled by weights_channel",
                    shape=(n_chans_per_substream, n_bls), dtype=np.uint8,
                    value=weights)
        ig.add_item(id=None, name='weights_channel',
                    description="Coarse (per-channel) weights",
                    shape=(n_chans_per_substream,), dtype=np.float32,
                    value=weights_channel)
        ig.add_item(id=None, name='timestamp',
                    description="Seconds since CBF sync time",
                    shape=(), dtype=None, format=[('f', 64)],
                    value=100.0)
        ig.add_item(id=None, name='dump_index',
                    description='Index in time',
                    shape=(), dtype=None, format=[('u', 64)],
                    value=1)
        ig.add_item(id=0x4103, name='frequency',
                    description="Channel index of first channel in the heap",
                    shape=(), dtype=np.uint32,
                    value=0)
        return ig

    async def setUp(self) -> None:
        npy_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, npy_path)
        self.chunk_store = katdal.chunkstore_npy.NpyFileChunkStore(npy_path)
        self.telstate = self.setup_telstate()
        self.setup_spead()
        self.server = await self.setup_server()
        self.client = await self.setup_client(self.server)
        self.ig = self.setup_ig()

    async def test_capture(self) -> None:
        cbid = '1234567890'
        self.assert_sensor_equals('status', Status.IDLE)
        await self.client.request('capture-init', cbid)
        self.assert_sensor_equals('status', Status.WAIT_DATA)
        for tx in self.tx:
            await tx.async_send_heap(self.ig.get_start())
        await self.tx[0].async_send_heap(self.ig.get_heap())
        await asyncio.sleep(0.5)
        self.assert_sensor_equals('status', Status.CAPTURING)
        self.assert_sensor_equals('input-heaps-total', 1)
        for tx in self.tx:
            await tx.async_send_heap(self.ig.get_end())
        await asyncio.sleep(0.5)
        self.assert_sensor_equals('status', Status.COMPLETE)
        await self.client.request('capture-done')
        self.assert_sensor_equals('status', Status.IDLE)

    async def test_missing_stop_item(self) -> None:
        cbid = '1234567890'
        await self.client.request('capture-init', cbid)
        for tx in self.tx:
            await tx.async_send_heap(self.ig.get_start())
        await self.tx[0].async_send_heap(self.ig.get_heap())
        for tx in self.tx[:-1]:
            await tx.async_send_heap(self.ig.get_end())
        await self.client.request('capture-done')

    async def test_double_init(self) -> None:
        await self.client.request('capture-init', '1234567890')
        with assert_raises_regex(FailReply, '(?i)already capturing'):
            await self.client.request('capture-init', '9876543210')

    async def test_done_without_init(self) -> None:
        with assert_raises_regex(FailReply, '(?i)not capturing'):
            await self.client.request('capture-done')
