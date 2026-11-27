"""A sidecar log for rows that did not score cleanly.

The result CSV says *that* a row failed, in its ``status`` column, but not why.
Putting the reason in a JSON Lines file next to the results keeps the CSV
schema stable while still making a failure diagnosable: one object per row,
with the chunk and row it came from, the SMILES string and the message.

Two properties follow from how a run can end.

Diagnostics are durable per chunk.  A chunk's reasons are written to their own
file as soon as that chunk's results are, because from that moment a resumed
run is entitled to skip it — anything still buffered in memory would be lost.
That also means the log holds no bound on how much it remembers: every failed
row keeps its reason, however many there are.

The log reflects the run that produced the results beside it.  A re-run that
screens a chunk again replaces that chunk's reasons, and a run that ends with
nothing to report removes an older log rather than leaving one that points at
molecules the current results do not mention.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

from .schema import Status


def _remove_all(parts: list[Path]) -> None:
    for part in parts:
        part.unlink(missing_ok=True)


def errors_path(out_path: str | Path) -> Path:
    path = Path(out_path)
    return path.parent / f"{path.stem}_errors.jsonl"


def _chunk_path(out_path: str | Path, chunk: int) -> Path:
    path = Path(out_path)
    return path.parent / f".{path.stem}_errors_{chunk}.jsonl.part"


class ErrorLog:
    """Per-chunk JSON Lines diagnostics, combined once the run has finished."""

    def __init__(self, out_path: str | Path, resume: bool = False) -> None:
        self._out_path = Path(out_path)
        self._path = errors_path(out_path)
        self._resume = resume
        self._chunk: int | None = None
        self._handle: Any = None
        self._written = 0
        self._kept_chunks: list[int] = []
        self._finished_chunks: list[int] = []
        if resume:
            self._split_existing_log()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def count(self) -> int:
        return self._written

    @property
    def used(self) -> bool:
        return self._path.is_file()

    def _split_existing_log(self) -> None:
        """Turn a log from an earlier run back into its per-chunk pieces.

        A resumed run keeps the diagnostics of chunks it skips, and the only
        record of those is the log the earlier run left behind.  Splitting it
        by chunk in one streaming pass means the rest of this class works on
        per-chunk files whatever the earlier run did, with no bound on how
        many entries are remembered.

        A piece that is already on disk is left alone: it comes from a run that
        was interrupted after that chunk finished but before its diagnostics
        were folded into the log, so it is the more complete record.
        """
        if not self._path.is_file():
            return
        handles: dict[int, Any] = {}
        try:
            with open(self._path, encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    try:
                        chunk = int(json.loads(line).get("chunk", 0))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        # A log truncated mid-write is not a reason to fail.
                        continue
                    if chunk not in handles:
                        target = _chunk_path(self._out_path, chunk)
                        if target.is_file():
                            handles[chunk] = None
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        handles[chunk] = open(target, "w", encoding="utf-8")
                    handle = handles[chunk]
                    if handle is not None:
                        handle.write(line)
        finally:
            for handle in handles.values():
                if handle is not None:
                    handle.close()

    # -- per-chunk lifecycle ----------------------------------------------

    def start_chunk(self, chunk: int) -> None:
        """Begin collecting diagnostics for a chunk about to be screened."""
        self.finish_chunk()
        self._chunk = chunk
        # A previous run's reasons for this chunk no longer describe it.
        _chunk_path(self._out_path, chunk).unlink(missing_ok=True)

    def keep_chunk(self, chunk: int) -> None:
        """Keep what an earlier run recorded for a chunk being skipped."""
        self.finish_chunk()
        if _chunk_path(self._out_path, chunk).is_file():
            self._kept_chunks.append(chunk)

    def finish_chunk(self) -> None:
        """Make the current chunk's diagnostics durable before moving on."""
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None
        if self._chunk is not None:
            self._finished_chunks.append(self._chunk)
            self._chunk = None

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
        if self._handle is None:
            target = _chunk_path(self._out_path, chunk)
            target.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(target, "w", encoding="utf-8")
        entry = {
            "chunk": chunk,
            "row": row,
            "SMILES": smiles,
            "status": status,
            "detail": detail,
        }
        self._handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._written += 1

    # -- combining --------------------------------------------------------

    def _parts(self) -> list[Path]:
        chunks = sorted(set(self._finished_chunks) | set(self._kept_chunks))
        return [
            path
            for path in (_chunk_path(self._out_path, chunk) for chunk in chunks)
            if path.is_file()
        ]

    def _iter_entries(self) -> Iterator[str]:
        for part in self._parts():
            with open(part, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield line

    def close(self) -> None:
        """Combine the per-chunk files into the log beside the results."""
        self.finish_chunk()
        parts = self._parts()

        if not parts:
            # Nothing failed in the run that produced the current results, so
            # an older log would point at molecules those results do not
            # mention.
            self._path.unlink(missing_ok=True)
            _remove_all(parts)
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".part")
        total = 0
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                for line in self._iter_entries():
                    handle.write(line)
                    total += 1
                handle.flush()
                os.fsync(handle.fileno())
            if total == 0:
                temporary.unlink(missing_ok=True)
                self._path.unlink(missing_ok=True)
            else:
                os.replace(temporary, self._path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        self._written = total
        _remove_all(parts)

    def __enter__(self) -> "ErrorLog":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
