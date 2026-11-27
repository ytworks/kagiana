"""A sidecar log for rows that did not score cleanly.

The result CSV says *that* a row failed, in its ``status`` column, but not why.
Putting the reason in a JSON Lines file next to the results keeps the CSV
schema stable while still making a failure diagnosable: one object per row,
with the chunk and row it came from, the SMILES string and the message.

Diagnostics are kept per chunk, in the run's state directory, and are written
before that chunk's results are: a chunk counts as finished only once both are
on disk, so a resumed run never skips a chunk whose reasons were lost. A chunk
that failed nothing still leaves its (empty) file, because "this generation
screened the chunk and found nothing wrong" and "this chunk was never screened"
have to be told apart.

One file is open at a time and each is renamed into place only when complete,
so a run over many chunks neither exhausts file descriptors nor leaves a
half-written piece that a later run would read as finished.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TextIO

from .runstate import RunState
from .schema import Status


def errors_path(out_path: str | Path) -> Path:
    path = Path(out_path)
    return path.parent / f"{path.stem}_errors.jsonl"


class ErrorLog:
    """Per-chunk JSON Lines diagnostics, published once the run has finished."""

    def __init__(self, out_path: str | Path, state: RunState) -> None:
        self._path = errors_path(out_path)
        self._state = state
        self._chunk: int | None = None
        self._handle: TextIO | None = None
        self._temporary: Path | None = None
        self._written = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def count(self) -> int:
        return self._written

    @property
    def used(self) -> bool:
        return self._path.is_file()

    # -- per-chunk lifecycle ----------------------------------------------

    def start_chunk(self, chunk: int) -> None:
        """Begin collecting diagnostics for a chunk about to be screened."""
        self.finish_chunk()
        self._chunk = chunk
        target = self._state.chunk_error_path(chunk)
        # Anything an earlier generation wrote for this chunk is about to be
        # replaced along with the chunk's results.
        target.unlink(missing_ok=True)
        self._temporary = target.with_suffix(".jsonl.part")
        self._temporary.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self._temporary, "w", encoding="utf-8")

    def finish_chunk(self) -> None:
        """Make the current chunk's diagnostics durable before moving on."""
        if self._chunk is None:
            return
        chunk, self._chunk = self._chunk, None
        handle, self._handle = self._handle, None
        temporary, self._temporary = self._temporary, None
        if handle is None or temporary is None:
            return
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        # The final name appears only now, so a later run cannot mistake a
        # half-written piece for a finished one.
        os.replace(temporary, self._state.chunk_error_path(chunk))

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
        if self._chunk != chunk:
            self.start_chunk(chunk)
        assert self._handle is not None
        entry = {
            "chunk": chunk,
            "row": row,
            "SMILES": smiles,
            "status": status,
            "detail": detail,
        }
        self._handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._written += 1

    # -- publishing -------------------------------------------------------

    def publish(self) -> int:
        """Combine the finished chunks' diagnostics into the log beside the results.

        Returns how many entries the log holds. A run that found nothing wrong
        removes an older log rather than leaving one that points at molecules
        the current results do not mention.
        """
        self.finish_chunk()

        pieces = [
            path
            for path in (
                self._state.chunk_error_path(chunk)
                for chunk in sorted(self._state.completed)
            )
            if path.is_file()
        ]

        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".part")
        total = 0
        try:
            with open(temporary, "w", encoding="utf-8") as destination:
                for piece in pieces:
                    with open(piece, encoding="utf-8") as source:
                        for line in source:
                            if not line.strip():
                                continue
                            destination.write(line)
                            total += 1
                destination.flush()
                os.fsync(destination.fileno())
            if total == 0:
                temporary.unlink(missing_ok=True)
                self._path.unlink(missing_ok=True)
            else:
                os.replace(temporary, self._path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

        self._written = total
        return total

    def close(self) -> int:
        return self.publish()

    def __enter__(self) -> ErrorLog:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def read_entries(path: str | Path) -> list[dict[str, Any]]:
    """Read a published diagnostic log, for tests and for callers that report it."""
    entries: list[dict[str, Any]] = []
    target = Path(path)
    if not target.is_file():
        return entries
    with open(target, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    return entries
