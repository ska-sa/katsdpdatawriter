"""Base functionality for :mod:`test_vis_writer` and :mod:`test_flag_writer`"""

from unittest import mock
import asyncio

import asynctest

import katsdptelstate
from katsdptelstate.endpoint import Endpoint
import aiokatcp
import spead2
import spead2.recv.asyncio
from nose.tools import assert_equal, assert_in


class BaseTestWriterServer(asynctest.TestCase):
    @classmethod
    def setup_telstate(cls, namespace: str) -> katsdptelstate.TelescopeState:
        telstate = katsdptelstate.TelescopeState().view(namespace)
        n_ants = 3
        telstate.add('n_chans', 4096, immutable=True)
        telstate.add('n_chans_per_substream', 1024, immutable=True)
        telstate.add('n_bls', n_ants * (n_ants + 1) * 2, immutable=True)
        return telstate

    def setup_sleep(self) -> None:
        """Patch loop.call_later so that delayed callbacks run immediately.

        This speeds up the tests where the code under test has a 5s timeout.
        """
        def call_later(delay, callback, *args):
            return self.loop.call_soon(callback, *args)

        patcher = mock.patch.object(self.loop, 'call_later', call_later)
        patcher.start()
        self.addCleanup(patcher.stop)

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

        async def get(stream, loop=None):
            heap = await orig_get(stream, loop)
            self.received_heaps.release()
            return heap

        self.received_heaps = asyncio.Semaphore(value=0, loop=self.loop)
        orig_get = spead2.recv.asyncio.Stream.get
        patcher = mock.patch('spead2.recv.asyncio.Stream.get', get)
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

    async def send_heap(self, tx, heap):
        """Send a heap and wait for it to be received.

        .. note:: This only works if all heaps are sent through this interface.
        """
        assert self.received_heaps.locked()
        await tx.async_send_heap(heap)
        # The above just waits until it's been transmitted into the inproc
        # queue, but we want to wait until it's come out the other end.
        await self.received_heaps.acquire()
