"""What a run has actually finished, recorded beside its results.

``--resume`` cannot work from the presence of output files alone.  A file under
its final name is complete, but it says nothing about *which* run produced it:
screening a different library into the same output directory leaves a mixture,
and resuming would happily keep half of the old results and half of the new
ones.

So a run records what it finished, and under which inputs.  The inputs are
reduced to a generation key; a resumed run skips only the chunks this
generation recorded as finished, and a run whose inputs differ starts over
rather than inheriting someone else's results.

The record lives in a hidden directory next to the output file, alongside the
per-chunk diagnostics, because those have to be kept in step with it: a chunk
counts as finished only once both are on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_STATE_FILE = "state.json"
_VERSION = 1


def state_directory(out_path: str | Path) -> Path:
    path = Path(out_path)
    return path.parent / f".{path.stem}_state"


def _file_identity(path: str | Path | None) -> Any:
    """How a file is recognised again: its path, size and modification time."""
    if path is None:
        return None
    target = Path(path)
    try:
        stat = target.stat()
    except OSError:
        return {"path": str(target), "missing": True}
    return {
        "path": str(target.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def generation_key(**inputs: Any) -> str:
    """A stable name for one combination of inputs and settings.

    Values that are paths are reduced to the file's identity, so editing a
    library in place starts a new generation even though its name is unchanged.
    """
    material: dict[str, Any] = {}
    for name, value in sorted(inputs.items()):
        if isinstance(value, Path):
            material[name] = _file_identity(value)
        else:
            material[name] = value
    encoded = json.dumps(material, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


@dataclass
class RunState:
    """The chunks and outputs one generation of a run has finished."""

    directory: Path
    key: str
    completed: set[int] = field(default_factory=set)
    finalized: bool = False

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def open(cls, out_path: str | Path, key: str, resume: bool) -> "RunState":
        """Load the record for this generation, or start a fresh one.

        A run that is not resuming starts over by definition, and so does one
        whose inputs no longer match what the record describes.  In both cases
        the old record and its diagnostics go, because they describe results
        this run is about to replace.
        """
        directory = state_directory(out_path)
        stored = cls._read(directory)
        if not resume or stored is None or stored.get("key") != key:
            cls._discard(directory)
            directory.mkdir(parents=True, exist_ok=True)
            state = cls(directory=directory, key=key)
            state._write()
            return state
        return cls(
            directory=directory,
            key=key,
            completed={int(index) for index in stored.get("completed", [])},
            finalized=bool(stored.get("finalized", False)),
        )

    @staticmethod
    def _read(directory: Path) -> dict[str, Any] | None:
        path = directory / _STATE_FILE
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict) or data.get("version") != _VERSION:
            return None
        return data

    @staticmethod
    def _discard(directory: Path) -> None:
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)

    def _write(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _VERSION,
            "key": self.key,
            "completed": sorted(self.completed),
            "finalized": self.finalized,
        }
        target = self.directory / _STATE_FILE
        temporary = target.with_suffix(".json.part")
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    # -- queries and updates ----------------------------------------------

    def is_chunk_complete(self, chunk: int) -> bool:
        return chunk in self.completed

    def mark_chunk_complete(self, chunk: int) -> None:
        """Record a chunk as finished, once all of its outputs are on disk."""
        self.completed.add(chunk)
        self._write()

    def mark_finalized(self) -> None:
        """Record a whole-file run as finished, once both outputs are on disk."""
        self.finalized = True
        self._write()

    def chunk_error_path(self, chunk: int) -> Path:
        """Where this generation keeps a chunk's diagnostics."""
        return self.directory / f"errors_{chunk}.jsonl"
