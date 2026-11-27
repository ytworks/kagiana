"""Regressions for the faults found in review.

Each test reproduces the condition from the review comment it belongs to, so a
future change that reintroduces the fault fails here rather than in a run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kagiana import csv_io
from kagiana.backends.cpu3d import init_worker as cpu3d_init_worker
from kagiana.cli import EXIT_INPUT_ERROR, EXIT_OK, main
from kagiana.molecules import mol_from_smiles
from kagiana.schema import Status


# --- library cells are passed through as written ---------------------------


@pytest.fixture
def awkward_library(tmp_path: Path) -> Path:
    """A library whose cells type inference would quietly rewrite."""
    path = tmp_path / "awkward.csv"
    path.write_text(
        "ID,SMILES,vendor_note\n"
        "00123,NCCc1ccc(O)cc1,NA\n"
        "123,NCCc1ccc(OC)cc1,NULL\n"
        "00456,c1ccccc1,None\n"
        "00789,,missing structure\n",
        encoding="utf-8",
    )
    return path


def test_identifiers_keep_their_leading_zeros(awkward_library: Path) -> None:
    _index, frame = next(csv_io.read_chunks(awkward_library, None))
    assert list(frame["ID"]) == ["00123", "123", "00456", "00789"]
    # Losing the zeros would make the first two rows indistinguishable.
    assert frame["ID"].iloc[0] != frame["ID"].iloc[1]


def test_annotation_cells_are_not_turned_into_blanks(awkward_library: Path) -> None:
    _index, frame = next(csv_io.read_chunks(awkward_library, None))
    assert list(frame["vendor_note"])[:3] == ["NA", "NULL", "None"]


def test_chunked_reading_passes_cells_through_identically(
    awkward_library: Path,
) -> None:
    whole = next(csv_io.read_chunks(awkward_library, None))[1]
    chunked = pd.concat(
        [frame for _i, frame in csv_io.read_chunks(awkward_library, 2)],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(whole, chunked)


def test_a_blank_smiles_cell_is_not_a_molecule() -> None:
    """RDKit accepts the empty string and returns an atomless molecule."""
    assert mol_from_smiles("") is None
    assert mol_from_smiles("   ") is None
    assert mol_from_smiles(None) is None
    assert mol_from_smiles("CCO") is not None


def test_a_blank_smiles_row_is_reported_not_scored(
    tmp_path: Path, awkward_library: Path, query: Path, fdef: Path, params: Path
) -> None:
    out = tmp_path / "r.csv"
    assert main(
        [
            "pharm2d",
            "--library", str(awkward_library), "--query", str(query),
            "--fdef", str(fdef), "--params", str(params),
            "--out", str(out), "--quiet",
        ]
    ) == EXIT_OK

    all_rows = pd.read_csv(tmp_path / "r_all.csv", dtype=str, keep_default_na=False)
    blank = all_rows[all_rows["ID"] == "00789"].iloc[0]
    assert blank["status"] == Status.PARSE_ERROR
    assert blank["tanimoto"] == ""
    # The identifiers survive the round trip through the result file.
    assert list(all_rows["ID"]) == ["00123", "123", "00456", "00789"]
    assert list(all_rows["vendor_note"])[:3] == ["NA", "NULL", "None"]


# --- a misspelt feature family is an input error ---------------------------


@pytest.fixture
def misspelt_query(tmp_path: Path, pharmacophore: Path) -> Path:
    text = pharmacophore.read_text(encoding="utf-8").replace("Acceptor", "Acceptorr")
    path = tmp_path / "typo.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_misspelt_feature_family_stops_the_screen(
    tmp_path: Path, library: Path, fdef: Path, misspelt_query: Path
) -> None:
    """Otherwise it screens cleanly and reports no hits, which looks like a
    library in which nothing matched."""
    out = tmp_path / "r.csv"
    assert main(
        [
            "pharm3d",
            "--library", str(library), "--pharmacophore", str(misspelt_query),
            "--fdef", str(fdef), "--out", str(out), "--quiet",
        ]
    ) == EXIT_INPUT_ERROR
    assert not out.exists()


def test_validate_and_screening_agree_about_a_misspelt_family(
    tmp_path: Path, library: Path, fdef: Path, misspelt_query: Path
) -> None:
    validate_code = main(
        ["validate", "--library", str(library), "--fdef", str(fdef),
         "--pharmacophore", str(misspelt_query), "--json"]
    )
    screen_code = main(
        ["pharm3d", "--library", str(library), "--pharmacophore", str(misspelt_query),
         "--fdef", str(fdef), "--out", str(tmp_path / "r.csv"), "--quiet"]
    )
    assert validate_code == screen_code == EXIT_INPUT_ERROR


# --- the pose ceiling is a ceiling ----------------------------------------


def test_the_pose_count_never_exceeds_the_requested_ceiling(
    fdef: Path, pharmacophore: Path
) -> None:
    task = cpu3d_init_worker(str(fdef), str(pharmacophore), 1.5, 0.5, 1, 1)
    for smiles in ("NCCc1cc(O)cc(O)c1", "NCCc1ccc(O)cc1", "NCCc1cc(O)ccc1O"):
        score = task(smiles)
        assert score is not None
        assert score.num_match <= 1, f"{smiles} produced {score.num_match} poses"


@pytest.mark.parametrize(("conformers", "max_poses"), [(1, 1), (2, 3), (10, 2)])
def test_the_ceiling_scales_with_the_conformer_count(
    fdef: Path, pharmacophore: Path, conformers: int, max_poses: int
) -> None:
    task = cpu3d_init_worker(
        str(fdef), str(pharmacophore), 1.5, 0.5, conformers, max_poses
    )
    score = task("NCCc1cc(O)cc(O)c1")
    assert score is not None
    assert score.num_match <= conformers * max_poses


# --- resume only skips work that really finished ---------------------------


def _pharm3d_args(library: Path, pharmacophore: Path, fdef: Path, out: Path) -> list[str]:
    return [
        "pharm3d",
        "--library", str(library), "--pharmacophore", str(pharmacophore),
        "--fdef", str(fdef), "--out", str(out), "--quiet",
    ]


def test_resume_rebuilds_a_missing_hit_list(
    tmp_path: Path, library: Path, fdef: Path, pharmacophore: Path
) -> None:
    """An interruption between the two outputs must not look like success."""
    out = tmp_path / "r.csv"
    argv = _pharm3d_args(library, pharmacophore, fdef, out)
    assert main(argv) == EXIT_OK
    expected = pd.read_csv(csv_io.screened_path(out), index_col=0)

    # Stand in for a crash after the full file was written but before the
    # hit list was.
    csv_io.screened_path(out).unlink()
    assert main([*argv, "--resume"]) == EXIT_OK
    assert csv_io.screened_path(out).is_file()
    rebuilt = pd.read_csv(csv_io.screened_path(out), index_col=0)
    pd.testing.assert_frame_equal(
        expected.drop(columns=["elapsed_s"]), rebuilt.drop(columns=["elapsed_s"])
    )


def test_a_half_written_output_is_never_left_behind(
    tmp_path: Path, library: Path, fdef: Path, pharmacophore: Path
) -> None:
    """Outputs are replaced atomically, so no partial file carries a final name."""
    out = tmp_path / "r.csv"
    assert main(_pharm3d_args(library, pharmacophore, fdef, out)) == EXIT_OK
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".part"]
    assert leftovers == []
    # Every file present parses completely.
    for path in tmp_path.glob("r*.csv"):
        pd.read_csv(path)


def test_resume_keeps_earlier_diagnostics_and_counts(
    tmp_path: Path, awkward_library: Path, fdef: Path, pharmacophore: Path
) -> None:
    out = tmp_path / "big.csv"
    argv = [
        "pharm3d",
        "--library", str(awkward_library), "--pharmacophore", str(pharmacophore),
        "--fdef", str(fdef), "--out", str(out),
        "--large", "--chunksize", "2", "--quiet",
    ]
    summary_path = tmp_path / "s.json"
    assert main([*argv, "--json-summary", str(summary_path)]) == EXIT_OK
    first = json.loads(summary_path.read_text(encoding="utf-8"))
    assert first["n_parse_error"] == 1

    errors = (tmp_path / "big_errors.jsonl").read_text(encoding="utf-8")
    assert "00789" not in errors  # the row index, not the identifier
    assert json.loads(errors.splitlines()[0])["status"] == Status.PARSE_ERROR

    # Resuming must neither lose the recorded failure nor its count.
    resumed_path = tmp_path / "s2.json"
    assert main([*argv, "--resume", "--json-summary", str(resumed_path)]) == EXIT_OK
    resumed = json.loads(resumed_path.read_text(encoding="utf-8"))
    assert resumed["n_total"] == first["n_total"]
    assert resumed["n_parse_error"] == first["n_parse_error"]
    assert resumed["n_screened"] == first["n_screened"]
    assert (tmp_path / "big_errors.jsonl").read_text(encoding="utf-8") == errors


def test_the_combined_hit_list_matches_an_uninterrupted_run(
    tmp_path: Path, library: Path, fdef: Path, pharmacophore: Path
) -> None:
    common = [
        "pharm3d",
        "--library", str(library), "--pharmacophore", str(pharmacophore),
        "--fdef", str(fdef), "--quiet",
    ]
    assert main([*common, "--out", str(tmp_path / "whole.csv")]) == EXIT_OK
    assert main(
        [*common, "--out", str(tmp_path / "chunked.csv"), "--large", "--chunksize", "3"]
    ) == EXIT_OK

    whole = pd.read_csv(csv_io.screened_path(tmp_path / "whole.csv"), index_col=0)
    combined = pd.read_csv(csv_io.screened_all_path(tmp_path / "chunked.csv"), index_col=0)
    assert len(combined) == len(whole)
    assert list(combined["ID"]) == list(whole["ID"])


# --- validate mirrors what a real run would reject -------------------------


def test_validate_rejects_a_ragged_record(tmp_path: Path, fdef: Path) -> None:
    path = tmp_path / "ragged.csv"
    path.write_text("ID,SMILES\n1,CCO\n2,CCC,extra\n", encoding="utf-8")
    assert main(["validate", "--library", str(path), "--fdef", str(fdef), "--json"]) == (
        EXIT_INPUT_ERROR
    )


def test_validate_survives_a_header_only_library(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("ID,SMILES\n", encoding="utf-8")
    code = main(["validate", "--library", str(path), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert report["ok"] is True
    assert report["checks"][0]["detail"]["records"] == 0


def test_validate_counts_records_not_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "quoted.csv"
    path.write_text('SMILES,note\nCCO,"one\ntwo"\n', encoding="utf-8")
    assert main(["validate", "--library", str(path), "--json"]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["checks"][0]["detail"]["records"] == 1


def test_an_unreadable_reference_molecule_is_an_input_error(
    tmp_path: Path, library: Path, fdef: Path, params: Path
) -> None:
    bad_query = tmp_path / "q.csv"
    bad_query.write_text("SMILES\nnot-a-smiles\n", encoding="utf-8")

    validate_code = main(
        ["validate", "--library", str(library), "--fdef", str(fdef),
         "--params", str(params), "--query", str(bad_query), "--json"]
    )
    screen_code = main(
        ["pharm2d", "--library", str(library), "--query", str(bad_query),
         "--fdef", str(fdef), "--params", str(params),
         "--out", str(tmp_path / "r.csv"), "--quiet"]
    )
    # Both commands must call this what it is, rather than one of them failing
    # later inside a worker.
    assert validate_code == screen_code == EXIT_INPUT_ERROR


def test_a_non_finite_coordinate_is_rejected(tmp_path: Path) -> None:
    from kagiana.pharmacophore3d import read_feature_points
    from kagiana.schema import InputError

    for bad in ("nan", "inf", "-inf"):
        path = tmp_path / f"{bad}.csv"
        path.write_text(
            f"feature_type,X,Y,Z\nDonor,{bad},0,0\nAcceptor,3,0,0\n", encoding="utf-8"
        )
        with pytest.raises(InputError, match="finite"):
            read_feature_points(path)
