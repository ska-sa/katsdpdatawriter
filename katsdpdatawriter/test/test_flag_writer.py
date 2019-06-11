"""Tests for :mod:`katsdpdatawriter.flag_writer`."""

import tempfile
import shutil
from unittest import mock
from typing import Dict, Any

import numpy as np
from nose.tools import (assert_equal, assert_true,
                        assert_regex, assert_raises_regex, assert_logs)

import aiokatcp
from aiokatcp import Sensor
import spead2
import spead2.send.asyncio
import katdal.chunkstore
from katdal.chunkstore_npy import NpyFileChunkStore

from ..flag_writer import FlagWriterServer, Status
from ..spead_write import DeviceStatus, ChunkParams
from .test_writer import BaseTestWriterServer


class TestFlagWriterServer(BaseTestWriterServer):
    async def setup_server(self, **arg_overrides) -> FlagWriterServer:
        args = dict(
            host='127.0.0.1', port=0, loop=self.loop, endpoints=self.endpoints,
            flag_interface='lo', flags_ibv=False,
            chunk_store=self.chunk_store, chunk_params=self.chunk_params,
            telstate=self.telstate.root(),
            input_name='sdp_l1_flags', output_name='sdp_l1_flags', rename_src={},
            s3_endpoint_url=None, max_workers=4, buffer_dumps=2)
        args.update(arg_overrides)
        server = FlagWriterServer(**args)
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

    async def setUp(self) -> None:
        self.npy_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.npy_path)
        self.chunk_store = NpyFileChunkStore(self.npy_path)
        self.telstate = self.setup_telstate('sdp_l1_flags')
        self.telstate['src_streams'] = ['sdp_l0']
        self.chunk_channels = 128
        self.chunk_params = ChunkParams(self.telstate['n_bls'] * self.chunk_channels,
                                        self.chunk_channels)
        self.setup_sleep()
        self.setup_spead()
        self.server = await self.setup_server()
        self.client = await self.setup_client(self.server)
        self.ig = self.setup_ig()

    def _check_chunk_info(self, output_name: str = 'sdp_l1_flags') -> Dict[str, Any]:
        n_chans = self.telstate['n_chans']
        n_bls = self.telstate['n_bls']
        capture_stream = '{}_{}'.format(self.cbid, output_name)

        view = self.telstate.root().view(capture_stream)
        chunk_info = view['chunk_info']
        n_chunks = n_chans // self.chunk_channels
        assert_equal(
            chunk_info,
            {
                'flags': {
                    'prefix': capture_stream.replace('_', '-'),
                    'shape': (1, n_chans, n_bls),
                    'chunks': ((1,), (self.chunk_channels,) * n_chunks, (n_bls,)),
                    'dtype': np.dtype(np.uint8)
                }
            })
        return chunk_info['flags']

    async def test_capture(self, output_name: str = 'sdp_l1_flags') -> None:
        n_chans_per_substream = self.telstate['n_chans_per_substream']
        self.assert_sensor_equals('status', Status.WAIT_DATA)
        self.assert_sensor_equals('capture-block-state', '{}')

        await self.client.request('capture-init', self.cbid)
        self.assert_sensor_equals('capture-block-state', '{"%s": "CAPTURING"}' % self.cbid)

        await self.send_heap(self.tx[0], self.ig.get_heap())
        self.assert_sensor_equals('status', Status.CAPTURING)

        await self.client.request('capture-done', self.cbid)
        self.assert_sensor_equals('status', Status.CAPTURING)  # Should still be capturing
        self.assert_sensor_equals('capture-block-state', '{}')
        await self.stop_server()
        capture_stream = '{}_{}'.format(self.cbid, output_name)
        prefix = capture_stream.replace('_', '-')
        assert_true(self.chunk_store.is_complete(prefix))

        # Validate the data written
        chunk_info = self._check_chunk_info(output_name)
        data = self.chunk_store.get_dask_array(
            self.chunk_store.join(chunk_info['prefix'], 'flags'),
            chunk_info['chunks'], chunk_info['dtype']).compute()
        n_chans_per_substream = self.telstate['n_chans_per_substream']
        np.testing.assert_array_equal(self.ig['flags'].value[np.newaxis],
                                      data[:, :n_chans_per_substream, :])
        np.testing.assert_equal(0, data[:, n_chans_per_substream:, :])

    async def test_new_name(self) -> None:
        # Replace client and server with different args
        output_name = 'sdp_l1_flags_new'
        rename_src = {'sdp_l0': 'sdp_l0_new'}
        s3_endpoint_url = 'http://new.invalid/'
        await self.server.stop()
        self.server = await self.setup_server(output_name=output_name,
                                              rename_src=rename_src,
                                              s3_endpoint_url=s3_endpoint_url)
        self.client = await self.setup_client(self.server)
        await self.test_capture(output_name)
        telstate_output = self.telstate.root().view(output_name)
        assert_equal(telstate_output['inherit'], 'sdp_l1_flags')
        assert_equal(telstate_output['s3_endpoint_url'], s3_endpoint_url)
        assert_equal(telstate_output['src_streams'], ['sdp_l0_new'])

    async def test_failed_write(self) -> None:
        with mock.patch.object(NpyFileChunkStore, 'put_chunk',
                               side_effect=katdal.chunkstore.StoreUnavailable):
            await self.client.request('capture-init', self.cbid)
            await self.send_heap(self.tx[0], self.ig.get_heap())
            await self.client.request('capture-done', self.cbid)
        self._check_chunk_info()
        self.assert_sensor_equals('device-status', DeviceStatus.FAIL, {Sensor.Status.ERROR})

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
            await self.send_heap(self.tx[0], self.ig.get_heap())
        assert_regex(cm.output[0], 'outside of init/done')
