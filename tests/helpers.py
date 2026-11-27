"""Top-level helpers for tests that start worker processes.

A worker started with the spawn method imports the module its initializer came
from, so these cannot be defined inside a test function.
"""

from __future__ import annotations

from collections.abc import Callable


def init_instant() -> Callable[[int], int]:
    return lambda value: value * 2


def init_quick() -> Callable[[object], object]:
    return lambda value: value


def init_late_then_report() -> Callable[[object], object]:
    """A task whose work, and whose progress report, land after the deadline."""
    import time

    from kagiana.pool import report_progress

    def task(value: object) -> object:
        time.sleep(0.14)
        report_progress("reported after the deadline")
        return value

    return task
