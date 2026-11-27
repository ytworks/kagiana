"""Library CSV reading, output naming and resume support.

Two reading modes exist and are deliberately kept distinct, because they serve
different library sizes:

* whole-file mode reads the library in one go and writes one pair of outputs;
* chunked mode reads a fixed number of rows at a time and writes a numbered
  pair of outputs per chunk, so a run over a very large library keeps memory
  flat and leaves partial results on disk if it is interrupted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from .schema import InputError, require_smiles_column


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

    ``chunksize`` of ``None`` reads the whole file as a single chunk.
    """
    path = Path(library_path)
    if not path.is_file():
        raise InputError(f"library file not found: {path}")

    try:
        if chunksize is None:
            frame = pd.read_csv(path)
            require_smiles_column(frame.columns, str(path))
            yield 0, frame
            return

        if chunksize < 1:
            raise InputError(f"chunksize must be >= 1, got {chunksize}")
        reader = pd.read_csv(path, chunksize=chunksize)
        for index, frame in enumerate(reader):
            if index == 0:
                require_smiles_column(frame.columns, str(path))
            yield index, frame
    except pd.errors.EmptyDataError as exc:
        raise InputError(f"{path}: the CSV holds no data") from exc
    except pd.errors.ParserError as exc:
        raise InputError(f"{path}: could not be parsed as CSV: {exc}") from exc


def count_rows(library_path: str | Path) -> int:
    """Number of data rows in a CSV, read without loading it into memory."""
    with open(library_path, "rb") as handle:
        total = sum(1 for _ in handle)
    return max(0, total - 1)


def write_frame(frame: pd.DataFrame, path: str | Path, index: bool) -> Path:
    """Write a frame, creating the parent directory when needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=index)
    return target


def smiles_values(frame: pd.DataFrame, column: str) -> list[str]:
    """Library SMILES as strings, matching how they are handed to RDKit."""
    return [str(value) for value in frame[column].tolist()]
