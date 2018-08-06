"""Tests for :mod:`katsdpdatawriter.flag_writer`."""

import os.path
import tempfile
import shutil
from unittest import mock
import asyncio

import numpy as np
import asynctest
from nose.tools import (assert_equal, assert_in, assert_true,
                        assert_regex, assert_raises_regex, assert_logs)

import aiokatcp
import spead2
import spead2.send.asyncio
import katsdptelstate
from katsdptelstate.endpoint import Endpoint
from katdal.chunkstore_npy import NpyFileChunkStore

from ..flag_writer import FlagWriterServer, Status


class TestFlagWriterServer(asynctest.TestCase):
    @classmethod
    def setup_telstate(cls) -> katsdptelstate.TelescopeState:
        telstate = katsdptelstate.TelescopeState()
        ants = ['m000', 'm001']
        n_ants = len(ants)
        telstate.add('n_chans', 4096, immutable=True)
        telstate.add('n_chans_per_substream', 1024, immutable=True)
        telstate.add('n_bls', n_ants * (n_ants + 1) * 2, immutable=True)
        return telstate

    def setup_spead(self) -> None:
        def add_udp_reader(stream, host, port, *args, **kwargs):
            queue = self.inproc_queues[Endpoint(host, port)]
            stream.add_inproc_reader(queue)

        self.endpoints = [Endpoint('239.102.254.{}'.format(i), 7148) for i in range(4)]
        self.inproc_queues = {endpoint: spead2.InprocQueue() for endpoint in self.endpoints}
        tx_pool = spead2.ThreadPool()
        self.tx = [spead2.send.asyncio.InprocStream(tx_pool, self.inproc_queues[endpoint])
                   for endpoint in self.endpoints]
        patcher = mock.patch('spead2.recv.asyncio.Stream.add_udp_reader', add_udp_reader)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def setup_server(self) -> FlagWriterServer:
        server = FlagWriterServer(
            host='127.0.0.1', port=0, loop=self.loop, endpoints=self.endpoints,
            flag_interface='lo', flags_ibv=False, npy_path=self.npy_path,
            telstate=self.telstate, flags_name='sdp_l1_flags')
        await server.start()
        self.addCleanup(server.stop)
        return server

    async def setup_client(self, server) -> aiokatcp.Client:
        port = server.server.sockets[0].getsockname()[1]
        client = await aiokatcp.Client.connect('localhost', port)
        self.addCleanup(client.wait_closed)
        self.addCleanup(client.close)
        return client

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

    def assert_sensor_equals(self, name, value, status=frozenset([aiokatcp.Sensor.Status.NOMINAL])):
        assert_equal(self.server.sensors[name].value, value)
        assert_in(self.server.sensors[name].status, status)

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
