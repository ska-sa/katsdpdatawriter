import asyncio
from collections import deque


class QueueSpace:
    """Manage space in a queue.

    This is logically similar to a semaphore, but allows the user to specify
    how much to acquire and release, rather than 1. It is first-come,
    first-served, so a large acquire will block the queue until there is
    space, even if there are later smaller acquires that could have been
    satisfied.
    """
    def __init__(self, value: int = 0, *, loop: asyncio.AbstractEventLoop = None) -> None:
        self._loop = loop if loop is not None else asyncio.get_event_loop()
        self._value = value
        self._waiters = deque()      # type: deque

    async def acquire(self, value: int) -> bool:
        if value <= self._value:
            self._value -= value
            return True
        future = self._loop.create_future()
        future.add_done_callback(self._cancel_handler)
        self._waiters.append((future, value))
        await future
        return True

    def _wakeup(self):
        while self._waiters:
            if self._waiters[0][0].done():
                # Can happen if it was cancelled
                self._waiters.popleft()
            elif self._waiters[0][1] <= self._value:
                future, req = self._waiters.popleft()
                self._value -= req
                future.set_result(None)
            else:
                break

    def _cancel_handler(self, future):
        if future.cancelled():
            self._wakeup()  # Give next requester a chance

    def release(self, value: int) -> None:
        self._value += value
        self._wakeup()

    def locked(self, value: int) -> bool:
        return value > self._value
