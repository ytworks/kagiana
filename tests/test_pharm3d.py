"""Tests for 3D screening: pharmacophore bounds, output modes, timeouts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kagiana.backends.cpu3d import Partial3D, init_worker, screen_molecule
from kagiana.cli import main
from kagiana.features import load_feature_factory
from kagiana.pharmacophore3d import make_pharmacophore, read_feature_points
from kagiana.schema import InputError, Stage, Status


def test_feature_points_are_read(pharmacophore: Path) -> None:
    points = read_feature_points(pharmacophore)
    assert [point.feature_type for point in points] == ["Donor", "Acceptor", "Aromatic"]


def test_distance_bounds_bracket_the_reference_distances(pharmacophore: Path) -> None:
    from rdkit import Geometry

    points = read_feature_points(pharmacophore)
    limit_upper, limit_lower = 1.5, 0.5
    query = make_pharmacophore(pharmacophore, limit_upper, limit_lower)

    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            a = Geometry.Point3D(points[i].x, points[i].y, points[i].z)
            b = Geometry.Point3D(points[j].x, points[j].y, points[j].z)
            distance = Geometry.Point3D.Distance(a, b)
            assert query.getLowerBound(i, j) == pytest.approx(
                max(0.0, distance - limit_lower)
            )
            assert query.getUpperBound(i, j) == pytest.approx(distance + limit_upper)


def test_a_lower_bound_never_goes_negative(tmp_path: Path) -> None:
    path = tmp_path / "close.csv"
    path.write_text(
        "feature_type,X,Y,Z\nDonor,0,0,0\nAcceptor,0.1,0,0\n", encoding="utf-8"
    )
    query = make_pharmacophore(path, 1.5, 0.5)
    assert query.getLowerBound(0, 1) == pytest.approx(0.0)


def test_malformed_pharmacophore_files_are_rejected(tmp_path: Path) -> None:
    missing_column = tmp_path / "a.csv"
    missing_column.write_text("feature_type,X,Y\nDonor,0,0\n", encoding="utf-8")
    with pytest.raises(InputError, match="missing column"):
        read_feature_points(missing_column)

    bad_number = tmp_path / "b.csv"
    bad_number.write_text(
        "feature_type,X,Y,Z\nDonor,0,0,0\nAcceptor,x,0,0\n", encoding="utf-8"
    )
    with pytest.raises(InputError, match="not a number"):
        read_feature_points(bad_number)

    too_few = tmp_path / "c.csv"
    too_few.write_text("feature_type,X,Y,Z\nDonor,0,0,0\n", encoding="utf-8")
    with pytest.raises(InputError, match="at least 2"):
        read_feature_points(too_few)


def test_the_molecule_the_query_came_from_embeds_poses(
    fdef: Path, pharmacophore: Path
) -> None:
    """The query was derived from this molecule, so it must satisfy it."""
    factory = load_feature_factory(fdef)
    query = make_pharmacophore(pharmacophore, 1.5, 0.5)
    score = screen_molecule("NCCc1ccc(O)cc1", factory, query, 10, 200)
    assert score is not None
    assert score.match is True
    assert score.num_match > 0


def test_a_molecule_lacking_the_features_does_not_match(
    fdef: Path, pharmacophore: Path
) -> None:
    factory = load_feature_factory(fdef)
    query = make_pharmacophore(pharmacophore, 1.5, 0.5)
    score = screen_molecule("c1ccccc1", factory, query, 10, 200)
    assert score is not None
    assert score.match is False
    assert score.num_match == 0


def test_an_unparsable_smiles_returns_nothing(fdef: Path, pharmacophore: Path) -> None:
    factory = load_feature_factory(fdef)
    query = make_pharmacophore(pharmacophore, 1.5, 0.5)
    assert screen_molecule("not-a-smiles", factory, query, 10, 200) is None


def test_the_pose_ceiling_scales_with_the_conformer_count(
    fdef: Path, pharmacophore: Path
) -> None:
    factory = load_feature_factory(fdef)
    query = make_pharmacophore(pharmacophore, 1.5, 0.5)
    tight = screen_molecule("NCCc1ccc(O)cc1", factory, query, 10, 1)
    assert tight is not None
    # Generation stops once the ceiling is passed, so the count stays bounded.
    assert tight.num_match <= 20


def test_an_interrupted_assignment_counts_as_one_pose() -> None:
    assert Partial3D(True, 7, in_assignment=True).num_match() == 8
    assert Partial3D(True, 7, in_assignment=False).num_match() == 7
    assert Partial3D(False, 0, in_assignment=False).num_match() == 0


def test_worker_init_builds_a_usable_task(fdef: Path, pharmacophore: Path) -> None:
    task = init_worker(str(fdef), str(pharmacophore), 1.5, 0.5, 10, 20)
    assert task("NCCc1ccc(O)cc1").num_match > 0
    assert task("broken-smiles") is None


def test_whole_file_mode_writes_two_indexed_files(
    tmp_path: Path, library: Path, fdef: Path, pharmacophore: Path
) -> None:
    out = tmp_path / "r.csv"
    summary_path = tmp_path / "s.json"
    assert main(
        [
            "pharm3d",
            "--library", str(library),
            "--pharmacophore", str(pharmacophore),
            "--fdef", str(fdef),
            "--out", str(out),
            "--jobs", "2", "--quiet",
            "--json-summary", str(summary_path),
        ]
    ) == 0

    scored = pd.read_csv(out, index_col=0)
    screened = pd.read_csv(tmp_path / "r_screened.csv", index_col=0)

    assert list(scored.columns) == [
        "ID", "SMILES", "vendor_note",
        "match", "numMatch", "status", "elapsed_s", "stage", "best_rmsd",
    ]
    assert len(scored) == 10
    assert (screened["numMatch"] > 0).all()
    assert len(screened) == (scored["numMatch"] > 0).sum()
    # The screened rows keep their position in the library.
    assert list(screened.index) == list(scored[scored["numMatch"] > 0].index)
    assert set(scored["stage"]) == {Stage.CPU}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_total"] == 10
    assert summary["n_screened"] == len(screened)
    assert summary["n_chunks"] == 1


def test_large_mode_writes_numbered_chunks_and_a_combined_file(
    tmp_path: Path, library: Path, fdef: Path, pharmacophore: Path
) -> None:
    out = tmp_path / "big.csv"
    assert main(
        [
            "pharm3d",
            "--library", str(library),
            "--pharmacophore", str(pharmacophore),
            "--fdef", str(fdef),
            "--out", str(out),
            "--large", "--chunksize", "4",
            "--jobs", "2", "--quiet",
        ]
    ) == 0

    # Three chunks of 4 + 4 + 2 rows.
    assert [len(pd.read_csv(tmp_path / f"big_{i}.csv", index_col=0)) for i in range(3)] == [4, 4, 2]
    for i in range(3):
        assert (tmp_path / f"big_screened_{i}.csv").is_file()
    # The per-chunk file is not written for a whole-file run and vice versa.
    assert not out.exists()

    combined = pd.read_csv(tmp_path / "big_screened_all.csv", index_col=0)
    per_chunk = sum(
        len(pd.read_csv(tmp_path / f"big_screened_{i}.csv", index_col=0)) for i in range(3)
    )
    assert len(combined) == per_chunk
    assert (combined["numMatch"] > 0).all()


def test_the_two_modes_agree_on_the_scores(
    tmp_path: Path, library: Path, fdef: Path, pharmacophore: Path
) -> None:
    common = [
        "--library", str(library),
        "--pharmacophore", str(pharmacophore),
        "--fdef", str(fdef),
        "--jobs", "2", "--quiet",
    ]
    assert main(["pharm3d", *common, "--out", str(tmp_path / "whole.csv")]) == 0
    assert main(
        ["pharm3d", *common, "--out", str(tmp_path / "chunked.csv"), "--large", "--chunksize", "3"]
    ) == 0

    whole = pd.read_csv(tmp_path / "whole.csv", index_col=0)
    chunked = pd.concat(
        [pd.read_csv(tmp_path / f"chunked_{i}.csv", index_col=0) for i in range(4)],
        ignore_index=True,
    )
    assert list(whole["numMatch"]) == list(chunked["numMatch"])
    assert list(whole["match"]) == list(chunked["match"])


def test_an_impossible_deadline_times_rows_out_without_failing_the_run(
    tmp_path: Path, library: Path, fdef: Path, pharmacophore: Path
) -> None:
    """A deadline nothing can meet must yield timeout rows, not a crash."""
    out = tmp_path / "r.csv"
    summary_path = tmp_path / "s.json"
    assert main(
        [
            "pharm3d",
            "--library", str(library),
            "--pharmacophore", str(pharmacophore),
            "--fdef", str(fdef),
            "--out", str(out),
            "--timeout", "0.001",
            "--jobs", "2", "--quiet",
            "--json-summary", str(summary_path),
        ]
    ) == 0

    scored = pd.read_csv(out, index_col=0)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_timeout"] > 0
    assert len(scored) == 10
    timed_out = scored[scored["status"] == Status.TIMEOUT]
    assert len(timed_out) == summary["n_timeout"]
    # A timed-out row still reports a count rather than an empty cell.
    assert timed_out["numMatch"].notna().all()

    entries = [
        json.loads(line)
        for line in (tmp_path / "r_errors.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(entry["status"] == Status.TIMEOUT for entry in entries)


def test_resume_skips_chunks_that_are_already_written(
    tmp_path: Path, library: Path, fdef: Path, pharmacophore: Path
) -> None:
    out = tmp_path / "big.csv"
    argv = [
        "pharm3d",
        "--library", str(library),
        "--pharmacophore", str(pharmacophore),
        "--fdef", str(fdef),
        "--out", str(out),
        "--large", "--chunksize", "4",
        "--quiet",
    ]
    assert main(argv) == 0
    first = (tmp_path / "big_0.csv").read_text(encoding="utf-8")

    # Make the first chunk's output recognisable, then resume: it must survive.
    marker = first.replace("dopamine-like", "MARKER")
    (tmp_path / "big_0.csv").write_text(marker, encoding="utf-8")
    assert main([*argv, "--resume"]) == 0
    assert "MARKER" in (tmp_path / "big_0.csv").read_text(encoding="utf-8")
