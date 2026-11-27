"""Top-level helpers for tests that start worker processes.

A worker started with the spawn method imports the module its initializer came
from, so these cannot be defined inside a test function.
"""

from __future__ import annotations

from typing import Callable


def init_instant() -> Callable[[int], int]:
    return lambda value: value * 2
