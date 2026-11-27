"""Library CSV reading, output naming and resume support.

Two reading modes exist and are deliberately kept distinct, because they serve
different library sizes:

* whole-file mode reads the library in one go and writes one pair of outputs;
* chunked mode reads a fixed number of rows at a time and writes a numbered
  pair of outputs per chunk, so a run over a very large library keeps memory
  flat and leaves partial results on disk if it is interrupted.

Two properties of this module matter for correctness rather than convenience.

Library cells are read as text.  Inferring types would rewrite the caller's
own data: an identifier of ``00123`` would come back as ``123`` and no longer
match the library it came from, and a note of ``NA`` would come back empty.
Columns this tool does not score are carried through exactly as they arrived.

Result files are replaced atomically.  A file that appears under its final name
is therefore complete, which is what lets ``--resume`` tell finished work from
work that was interrupted halfway through writing.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Iterator

import pandas as pd

from .schema import InputError, require_smiles_column

#: Read options for every library read.  ``dtype=str`` keeps identifiers and
#: annotations exactly as written, and disabling NA detection stops cell values
#: such as "NA", "NULL", "None" and "nan" from being turned into blanks.
_LIBRARY_READ_OPTIONS: dict[str, object] = {
    "dtype": str,
    "keep_default_na": False,
    "na_filter": False,
}


def _stem_and_suffix(out_path: str | Path) -> tuple[Path, str, str]:
    path = Path(out_path)
    suffix = path.suffix or ".csv"
    return path.parent, path.stem, suffix


def numbered_path(out_path: str | Path, index: int) -> Path:
    """``R.csv`` -> ``R_0.csv``: all rows of one chunk."""
    parent, stem, suffix = _stem_and_suffix(out_path)
    return parent / f"{stem}_{index}{suffix}"


def screened_path(out_path: str | Path) -> Path:
    """``R.csv`` -> ``R_screened.csv``: the rows that passed."""
    parent, stem, suffix = _stem_and_suffix(out_path)
    return parent / f"{stem}_screened{suffix}"


def screened_numbered_path(out_path: str | Path, index: int) -> Path:
    """``R.csv`` -> ``R_screened_0.csv``: the rows of one chunk that passed."""
    parent, stem, suffix = _stem_and_suffix(out_path)
    return parent / f"{stem}_screened_{index}{suffix}"


def screened_all_path(out_path: str | Path) -> Path:
    """``R.csv`` -> ``R_screened_all.csv``: every passing row of a chunked run."""
    parent, stem, suffix = _stem_and_suffix(out_path)
    return parent / f"{stem}_screened_all{suffix}"


def all_rows_path(out_path: str | Path) -> Path:
    """``R.csv`` -> ``R_all.csv``: every row with its scores."""
    parent, stem, suffix = _stem_and_suffix(out_path)
    return parent / f"{stem}_all{suffix}"


def read_chunks(
    library_path: str | Path, chunksize: int | None
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield ``(chunk_index, frame)`` pairs from a library CSV.

    ``chunksize`` of ``None`` reads the whole file as a single chunk.  Cells
    come back as text, so the columns this tool passes through are unchanged.
    """
    path = Path(library_path)
    if not path.is_file():
        raise InputError(f"library file not found: {path}")

    try:
        if chunksize is None:
            frame = pd.read_csv(path, **_LIBRARY_READ_OPTIONS)
            require_smiles_column(frame.columns, str(path))
            yield 0, frame
            return

        if chunksize < 1:
            raise InputError(f"chunksize must be >= 1, got {chunksize}")
        reader = pd.read_csv(path, chunksize=chunksize, **_LIBRARY_READ_OPTIONS)
        seen = 0
        for index, frame in enumerate(reader):
            if index == 0:
                require_smiles_column(frame.columns, str(path))
            seen += 1
            yield index, frame
        if seen == 0:
            # A header with no records: yield the empty frame so that callers
            # still produce their outputs, with the columns a caller expects.
            frame = pd.read_csv(path, **_LIBRARY_READ_OPTIONS)
            require_smiles_column(frame.columns, str(path))
            yield 0, frame
    except pd.errors.EmptyDataError as exc:
        raise InputError(f"{path}: the CSV holds no data") from exc
    except pd.errors.ParserError as exc:
        raise InputError(f"{path}: could not be parsed as CSV: {exc}") from exc


def count_records(csv_path: str | Path) -> int:
    """Number of data records in a CSV.

    Counts CSV records rather than lines, so a field holding a newline is one
    record and not two.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def iter_records(csv_path: str | Path) -> Iterator[tuple[int, list[str]]]:
    """Yield ``(record_number, fields)`` for each data record, header excluded.

    The record number counts data records from 1, which is what a reader of the
    file would call the row, and the header is returned as record 0.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        for number, fields in enumerate(csv.reader(handle)):
            yield number, fields


def write_frame(frame: pd.DataFrame, path: str | Path, index: bool) -> Path:
    """Write a frame so that it only appears under its final name when complete.

    The frame goes to a temporary file in the same directory and is then moved
    into place, which is atomic within a filesystem.  A run interrupted while
    writing therefore leaves no half-written file for ``--resume`` to mistake
    for finished work.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        dir=target.parent,
        prefix=f".{target.stem}-",
        suffix=f"{target.suffix}.part",
        delete=False,
        newline="",
        encoding="utf-8",
    )
    temporary = Path(handle.name)
    try:
        with handle:
            frame.to_csv(handle, index=index)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def append_frame(
    frame: pd.DataFrame, path: str | Path, index: bool, header: bool
) -> Path:
    """Append a frame to a growing file, writing the header only once.

    Used where output is produced incrementally and the file is meant to grow
    as the run proceeds; such a file is not atomic by nature, so ``--resume``
    never treats one as evidence of finished work.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=index, mode="a" if not header else "w", header=header)
    return target


def concat_csv_body(source: str | Path, destination_handle: object, header: bool) -> int:
    """Copy one CSV into an open handle, keeping at most one header line.

    Returns the number of data records copied.  Combining finished per-chunk
    files this way keeps the peak memory of an aggregation at one chunk instead
    of the whole run.
    """
    copied = 0
    with open(source, newline="", encoding="utf-8") as reader:
        for position, line in enumerate(reader):
            if position == 0:
                if header:
                    destination_handle.write(line)  # type: ignore[attr-defined]
                continue
            destination_handle.write(line)  # type: ignore[attr-defined]
            copied += 1
    return copied


def read_status_counts(path: str | Path) -> dict[str, int]:
    """Tally the ``status`` column of a result file this tool wrote.

    Used when ``--resume`` skips a finished chunk: the run summary has to
    account for failures recorded earlier instead of reporting none.
    """
    try:
        frame = pd.read_csv(path, usecols=["status"], dtype=str, keep_default_na=False)
    except (ValueError, pd.errors.EmptyDataError):
        return {}
    counts: dict[str, int] = {}
    for status in frame["status"]:
        counts[status] = counts.get(status, 0) + 1
    return counts


def smiles_values(frame: pd.DataFrame, column: str) -> list[str]:
    """Library SMILES as strings, matching how they are handed to RDKit."""
    return [str(value) for value in frame[column].tolist()]
