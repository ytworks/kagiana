"""Tests for the forcibly-abortable worker pool."""

from __future__ import annotations

import os
import time

import pytest

from kagiana.pool import KillableWorkerPool, WorkerInitError
from kagiana.schema import Status


def _init_double() -> object:
    return lambda value: value * 2


def _init_stall_on_zero() -> object:
    def task(value: int) -> int:
        if value == 0:
            # A tight C-free spin, standing in for a non-interruptible
            # native computation: no sleep, no Python-level cancellation point
            # that a thread-based timeout could use.
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                pass
        return value * 10

    return task


def _init_raise_on_three() -> object:
    def task(value: int) -> int:
        if value == 3:
            raise ValueError("bad molecule")
        return value

    return task


def _init_broken() -> object:
    raise RuntimeError("cannot build worker state")


def test_results_come_back_in_input_order() -> None:
    with KillableWorkerPool(4, _init_double) as pool:
        results = list(pool.map_ordered(range(50)))
    assert [r.index for r in results] == list(range(50))
    assert [r.value for r in results] == [i * 2 for i in range(50)]
    assert all(r.status == Status.OK for r in results)


def test_timeout_kills_the_worker_and_the_run_continues() -> None:
    started = time.monotonic()
    with KillableWorkerPool(2, _init_stall_on_zero, timeout=1.0) as pool:
        results = list(pool.map_ordered(range(5)))
    elapsed = time.monotonic() - started

    assert results[0].status == Status.TIMEOUT
    assert results[0].value is None
    assert "1s" in results[0].detail
    assert [r.status for r in results[1:]] == [Status.OK] * 4
    assert [r.value for r in results[1:]] == [10, 20, 30, 40]
    # The stalling task would have run for 30s had it not been killed.
    assert elapsed < 20
    assert pool.kills == 1


def test_task_exception_is_reported_per_row() -> None:
    with KillableWorkerPool(2, _init_raise_on_three) as pool:
        results = list(pool.map_ordered(range(6)))
    assert results[3].status == Status.ERROR
    assert "bad molecule" in results[3].detail
    assert [r.status for i, r in enumerate(results) if i != 3] == [Status.OK] * 5


def test_worker_init_failure_is_fatal() -> None:
    with pytest.raises(WorkerInitError, match="cannot build worker state"):
        with KillableWorkerPool(1, _init_broken) as pool:
            list(pool.map_ordered(range(3)))


def test_empty_input_is_handled() -> None:
    with KillableWorkerPool(2, _init_double) as pool:
        assert list(pool.map_ordered([])) == []


def _init_sleep_25ms() -> object:
    def task(value: int) -> int:
        time.sleep(0.025)
        return value

    return task


def _init_exit_before_ready() -> object:
    # Dying here, before the worker reports itself ready, is what a missing
    # shared library or an OOM kill during setup looks like.
    os._exit(7)


def test_a_task_that_finishes_after_its_deadline_is_not_a_success() -> None:
    """A deadline that is only checked between polls is not a deadline.

    The work here outlives a 5 ms limit but completes well inside the parent's
    poll interval, so the result arrives before any deadline sweep. It must
    still be reported as a timeout.
    """
    with KillableWorkerPool(1, _init_sleep_25ms, timeout=0.005) as pool:
        results = list(pool.map_ordered(range(3)))

    assert [r.status for r in results] == [Status.TIMEOUT] * 3
    assert all(r.value is None for r in results)
    assert all(r.elapsed_s > 0.005 for r in results)


def test_a_generous_deadline_still_accepts_the_result() -> None:
    with KillableWorkerPool(2, _init_sleep_25ms, timeout=5.0) as pool:
        results = list(pool.map_ordered(range(4)))
    assert [r.status for r in results] == [Status.OK] * 4
    assert [r.value for r in results] == [0, 1, 2, 3]


def test_a_worker_dying_before_ready_fails_after_a_bounded_number_of_tries() -> None:
    """Respawning a worker that cannot start would loop for ever."""
    with pytest.raises(WorkerInitError) as caught:
        with KillableWorkerPool(1, _init_exit_before_ready, timeout=1.0) as pool:
            list(pool.map_ordered(range(2)))
    message = str(caught.value)
    assert "before it finished starting up" in message
    # The exit code is the only clue to why it died, so it has to be reported.
    assert "7" in message
