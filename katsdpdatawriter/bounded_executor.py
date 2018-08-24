"""Concurrent executor with bounded work queue"""

import threading
import concurrent.futures
import functools
from typing import Callable, Any


class BoundedThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    """Thread pool executor with bounded queue depth.

    If items are submitted to the executor faster than it can process them,
    :meth:`submit` will block rather than allowing unbounded queue growth.

    `max_queued` is the maximum number of items that have been submitted but
    not yet started executing. It must be at least 1, as otherwise it is
    impossible to submit anything. Note that executing tasks do not count
    towards the limit.
    """
    def __init__(self, max_workers: int = None, max_queued: int = 1) -> None:
        if max_queued < 1:
            raise ValueError('max_queued must be at least 1')
        super().__init__(max_workers)
        self._queue_sem = threading.BoundedSemaphore(max_queued)

    def _check_cancel(self, future: concurrent.futures.Future) -> None:
        if future.cancelled():
            self._queue_sem.release()

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> concurrent.futures.Future:
        @functools.wraps(fn)
        def wrapper(*wrapper_args: Any, **wrapper_kwargs: Any) -> Any:
            self._queue_sem.release()
            return fn(*wrapper_args, **wrapper_kwargs)

        self._queue_sem.acquire()
        future = super().submit(wrapper, *args, **kwargs)
        # If the future is cancelled before it gets a chance to execute, it
        # won't reach the release call in wrapper. Handle that in a callback.
        future.add_done_callback(self._check_cancel)
        return future
