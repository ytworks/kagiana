"""Tests for library reading, output naming and input validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kagiana import csv_io
from kagiana.schema import InputError


def test_output_names_follow_the_convention() -> None:
    out = "/data/results/run.csv"
    assert csv_io.numbered_path(out, 7) == Path("/data/results/run_7.csv")
    assert csv_io.screened_path(out) == Path("/data/results/run_screened.csv")
    assert csv_io.screened_numbered_path(out, 2) == Path("/data/results/run_screened_2.csv")
    assert csv_io.screened_all_path(out) == Path("/data/results/run_screened_all.csv")
    assert csv_io.all_rows_path(out) == Path("/data/results/run_all.csv")


def test_a_path_containing_csv_in_a_directory_is_not_mangled() -> None:
    # Naive string replacement of ".csv" would corrupt this path.
    out = "/data/csv.exports/run.csv"
    assert csv_io.screened_path(out) == Path("/data/csv.exports/run_screened.csv")


def test_whole_file_mode_yields_one_chunk(library: Path) -> None:
    chunks = list(csv_io.read_chunks(library, None))
    assert len(chunks) == 1
    index, frame = chunks[0]
    assert index == 0
    assert len(frame) == 10


def test_chunked_mode_splits_the_library(library: Path) -> None:
    chunks = list(csv_io.read_chunks(library, 4))
    assert [len(frame) for _i, frame in chunks] == [4, 4, 2]
    assert [index for index, _frame in chunks] == [0, 1, 2]


def test_record_count_matches_the_frame(library: Path) -> None:
    assert csv_io.count_records(library) == 10


def test_records_are_counted_not_lines(tmp_path: Path) -> None:
    """A field holding a newline is one record, not two."""
    path = tmp_path / "quoted.csv"
    path.write_text('SMILES,note\nCCO,"first line\nsecond line"\n', encoding="utf-8")
    assert csv_io.count_records(path) == 1


def test_a_missing_library_is_an_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="not found"):
        list(csv_io.read_chunks(tmp_path / "absent.csv", None))


def test_a_library_without_a_smiles_column_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "wrong.csv"
    path.write_text("smiles,id\nCCO,1\n", encoding="utf-8")
    with pytest.raises(InputError, match="SMILES"):
        list(csv_io.read_chunks(path, None))


def test_smiles_are_read_as_strings(library: Path) -> None:
    _index, frame = next(csv_io.read_chunks(library, None))
    values = csv_io.smiles_values(frame, "SMILES")
    assert all(isinstance(value, str) for value in values)
    assert values[0] == "NCCc1ccc(O)cc1"


def test_write_frame_creates_missing_directories(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeper" / "out.csv"
    csv_io.write_frame(pd.DataFrame({"a": [1]}), target, index=False)
    assert target.read_text(encoding="utf-8").splitlines() == ["a", "1"]
