"""Base functionality for :mod:`test_vis_writer` and :mod:`test_flag_writer`"""

from unittest import mock

import asynctest

import katsdptelstate
from katsdptelstate.endpoint import Endpoint
import aiokatcp
import spead2
from nose.tools import assert_equal, assert_in


class BaseTestWriterServer(asynctest.TestCase):
    @classmethod
    def setup_telstate(cls) -> katsdptelstate.TelescopeState:
        telstate = katsdptelstate.TelescopeState()
        n_ants = 3
        telstate.add('n_chans', 4096, immutable=True)
        telstate.add('n_chans_per_substream', 1024, immutable=True)
        telstate.add('n_bls', n_ants * (n_ants + 1) * 2, immutable=True)
        return telstate

    def setup_spead(self) -> None:
        def add_udp_reader(stream, host: str, port: int, *args, **kwargs) -> None:
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

    async def setup_client(self, server: aiokatcp.DeviceServer) -> aiokatcp.Client:
        assert server.server is not None, "Server has not been started"
        # mypy doesn't know about asyncio.base_events.Server, which has the 'sockets' member
        port = server.server.sockets[0].getsockname()[1]    # type: ignore
        client = await aiokatcp.Client.connect('localhost', port)
        self.addCleanup(client.wait_closed)
        self.addCleanup(client.close)
        return client

    def assert_sensor_equals(self, name, value, status=frozenset([aiokatcp.Sensor.Status.NOMINAL])):
        assert_equal(self.server.sensors[name].value, value)
        assert_in(self.server.sensors[name].status, status)
