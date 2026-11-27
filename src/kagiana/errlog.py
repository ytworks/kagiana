"""A sidecar log for rows that did not score cleanly.

The result CSV says *that* a row failed, in its ``status`` column, but not why.
Putting the reason in a JSON Lines file next to the results keeps the CSV
schema stable while still making a failure diagnosable: one object per row,
with the chunk and row it came from, the SMILES string and the message.

A resumed run has to add to this log rather than start it over, or the
diagnostics for chunks finished earlier would be thrown away by the first
failure of the new run.  Entries are therefore held per chunk: a chunk that is
being screened again replaces its own entries, and a chunk skipped because it
was already finished keeps the ones it wrote last time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .schema import Status

#: Upper bound on retained entries.  Failures are normally rare, but a badly
#: formed library could produce one per row, and the log must not become the
#: reason a run exhausts memory.
MAX_RETAINED_ENTRIES = 200_000


def errors_path(out_path: str | Path) -> Path:
    path = Path(out_path)
    return path.parent / f"{path.stem}_errors.jsonl"


class ErrorLog:
    """Per-chunk JSON Lines log, written out once the run has finished."""

    def __init__(self, out_path: str | Path, resume: bool = False) -> None:
        self._path = errors_path(out_path)
        self._entries: list[dict[str, Any]] = []
        self._kept_chunks: set[int] = set()
        self._recomputed_chunks: set[int] = set()
        self._truncated = False
        self._existing: list[dict[str, Any]] = []
        if resume:
            self._existing = self._load_existing()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def count(self) -> int:
        return len(self._retained()) + len(self._entries)

    @property
    def used(self) -> bool:
        return self.count > 0

    @property
    def truncated(self) -> bool:
        """Whether retention was capped and some entries were dropped."""
        return self._truncated

    def _load_existing(self) -> list[dict[str, Any]]:
        if not self._path.is_file():
            return []
        entries: list[dict[str, Any]] = []
        with open(self._path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # A log cut off mid-write is not a reason to fail the run.
                    continue
                if isinstance(entry, dict):
                    entries.append(entry)
        return entries

    def keep_chunk(self, chunk: int) -> None:
        """Keep what an earlier run recorded for a chunk being skipped."""
        self._kept_chunks.add(chunk)

    def recompute_chunk(self, chunk: int) -> None:
        """Drop what an earlier run recorded for a chunk being screened again."""
        self._recomputed_chunks.add(chunk)

    def _retained(self) -> list[dict[str, Any]]:
        if not self._existing:
            return []
        return [
            entry
            for entry in self._existing
            if entry.get("chunk") in self._kept_chunks
            and entry.get("chunk") not in self._recomputed_chunks
        ]

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
        if self.count >= MAX_RETAINED_ENTRIES:
            self._truncated = True
            return
        self._entries.append(
            {
                "chunk": chunk,
                "row": row,
                "SMILES": smiles,
                "status": status,
                "detail": detail,
            }
        )

    def close(self) -> None:
        """Write the log, or remove a stale one when there is nothing to say."""
        entries = self._retained() + self._entries
        if not entries:
            # Nothing failed this time; an older log would be misleading.
            if self._path.is_file() and self._existing:
                self._path.unlink(missing_ok=True)
            return

        entries.sort(key=lambda entry: (entry.get("chunk", 0), entry.get("row", 0)))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".part")
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def __enter__(self) -> "ErrorLog":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
