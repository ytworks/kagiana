"""A sidecar log for rows that did not score cleanly.

The result CSV says *that* a row failed, in its ``status`` column, but not why.
Putting the reason in a JSON Lines file next to the results keeps the CSV
schema stable while still making a failure diagnosable: one object per row,
with the chunk and row it came from, the SMILES string and the message.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

from .schema import Status


def errors_path(out_path: str | Path) -> Path:
    path = Path(out_path)
    return path.parent / f"{path.stem}_errors.jsonl"


class ErrorLog:
    """Append-only JSON Lines log, created only once something goes wrong."""

    def __init__(self, out_path: str | Path) -> None:
        self._path = errors_path(out_path)
        self._handle: TextIO | None = None
        self._count = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def count(self) -> int:
        return self._count

    @property
    def used(self) -> bool:
        return self._count > 0

    def record(
        self,
        chunk: int,
        row: int,
        smiles: str,
        status: str,
        detail: str,
    ) -> None:
        # A molecule that simply lacks the required features screened fine; it
        # is a result, not a failure.
        if status in (Status.OK, Status.NO_FEATURE):
            return
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(self._path, "w", encoding="utf-8")
        entry: dict[str, Any] = {
            "chunk": chunk,
            "row": row,
            "SMILES": smiles,
            "status": status,
            "detail": detail,
        }
        self._handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._count += 1

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "ErrorLog":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
