"""Tests for :class:`.BoundedThreadPoolExecutor`."""

import time
import threading

from nose.tools import assert_raises, assert_true

from ..bounded_executor import BoundedThreadPoolExecutor


class TestBoundedExecutor:
    def test_max_queued_zero(self):
        with assert_raises(ValueError):
            BoundedThreadPoolExecutor(max_queued=0)

    def test_block(self):
        def wakeup():
            time.sleep(0.1)
            event.set()

        with BoundedThreadPoolExecutor(max_workers=3, max_queued=2) as executor:
            event = threading.Event()
            # Fill the maximum possible number of slots
            futures = []
            for i in range(5):
                futures.append(executor.submit(event.wait))
            # Schedule the event to be set in the future
            event_thread = threading.Thread(target=wakeup)
            event_thread.start()
            # Push another task, which should block until event is set
            futures.append(executor.submit(event.wait))
            # We should only get here once event is set.
            assert_true(event.is_set())
            for future in futures:
                future.result()

    def test_cancel(self):
        def task():
            sem.release()
            event.wait()

        with BoundedThreadPoolExecutor(max_workers=3, max_queued=2) as executor:
            # Event which we set to allow tasks to complete
            event = threading.Event()
            # Event that tasks set when starting so that we can wait for a
            # certain number of tasks to start
            sem = threading.Semaphore(0)
            # Start some tasks and wait until they're running
            futures = []
            for i in range(3):
                futures.append(executor.submit(task))
                sem.acquire()
            # Start lots more tasks, cancelling before it can start
            for i in range(20):
                future = executor.submit(event.wait)
                future.cancel()
            event.set()
            for future in futures:
                future.result()
