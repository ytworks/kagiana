"""Progress reporting that stays out of the way of machine consumers.

Data goes to the output CSV, progress and warnings go to stderr, so a caller
can parse stdout or the result files without filtering noise.  Progress is
suppressed when stderr is not a terminal, which is the normal case when the
tool is driven by another program.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def want_progress(quiet: bool) -> bool:
    if quiet:
        return False
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


@contextmanager
def progress_bar(total: int, label: str, quiet: bool) -> Iterator[Any]:
    """A tqdm bar on a terminal, and a no-op object otherwise."""
    if not want_progress(quiet):
        class _Silent:
            def update(self, _n: int = 1) -> None:
                return None

        yield _Silent()
        return

    from tqdm import tqdm

    with tqdm(total=total, desc=label, unit="mol", file=sys.stderr) as bar:
        yield bar


def note(message: str, quiet: bool = False) -> None:
    """Write an informational line to stderr."""
    if not quiet:
        print(message, file=sys.stderr)
