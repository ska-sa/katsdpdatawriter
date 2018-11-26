import asynctest
from nose.tools import assert_true, assert_false

from ..queue_space import QueueSpace


class TestQueueSpace(asynctest.TestCase):
    def setUp(self):
        self.qs = QueueSpace(value=100, loop=self.loop)

    async def test_immediate(self):
        result = await self.qs.acquire(100)
        assert_true(result)

    async def test_block(self):
        task = self.loop.create_task(self.qs.acquire(200))
        await asynctest.exhaust_callbacks(self.loop)
        assert_false(task.done())
        self.qs.release(120)
        await asynctest.exhaust_callbacks(self.loop)
        assert_true(task.done())
        assert_true(await task)

    async def test_cancel(self):
        task1 = self.loop.create_task(self.qs.acquire(200))
        task2 = self.loop.create_task(self.qs.acquire(100))
        await asynctest.exhaust_callbacks(self.loop)
        assert_false(task1.done())
        task1.cancel()
        await asynctest.exhaust_callbacks(self.loop)
        assert_true(task2.done())
        assert_true(await task2)

    async def test_release_multiple(self):
        task1 = self.loop.create_task(self.qs.acquire(200))
        task2 = self.loop.create_task(self.qs.acquire(100))
        await asynctest.exhaust_callbacks(self.loop)
        assert_false(task1.done())
        self.qs.release(200)
        await asynctest.exhaust_callbacks(self.loop)
        assert_true(task1.done())
        assert_true(task2.done())
        assert_true(await task1)
        assert_true(await task2)
