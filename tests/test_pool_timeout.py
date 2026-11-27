"""Tests for the forcibly-abortable worker pool."""

from __future__ import annotations

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
