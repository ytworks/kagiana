"""Regressions for the faults found in review.

Each test reproduces the condition from the review comment it belongs to, so a
future change that reintroduces the fault fails here rather than in a run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from kagiana import csv_io
from kagiana.backends.cpu3d import init_worker as cpu3d_init_worker
from kagiana.cli import EXIT_INPUT_ERROR, EXIT_OK, main
from kagiana.molecules import mol_from_smiles
from kagiana.schema import Status

from .helpers import init_quick

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


# --- second review round ---------------------------------------------------


def test_a_task_that_finished_in_time_survives_a_slow_consumer() -> None:
    """The deadline is about how long the work took, not when it was read.

    A caller that pauses between results would otherwise turn finished tasks
    into timeouts, because the parent only noticed them after the deadline.
    """
    import time

    from kagiana.pool import KillableWorkerPool

    from .helpers import init_instant

    with KillableWorkerPool(1, init_instant, timeout=0.05) as pool:
        collected = []
        for position, result in enumerate(pool.map_ordered(range(3))):
            collected.append((result.status, result.value))
            if position == 0:
                time.sleep(0.15)

    assert collected == [(Status.OK, 0), (Status.OK, 2), (Status.OK, 4)]


def test_an_unsatisfiable_query_is_settled_without_searching() -> None:
    """More query points than distinct positions means no assignment exists.

    Walking the tree to discover that costs hundreds of thousands of steps, in
    the parent process, where the per-molecule deadline cannot reach it.
    """
    import time

    import numpy as np

    from kagiana.geometry import best_assignment_rmsd

    query = np.random.default_rng(0).normal(size=(9, 3))
    candidates = [
        [np.array([float(i), 0.0, 0.0]) for i in range(8)] for _ in range(9)
    ]
    started = time.monotonic()
    best, exhaustive = best_assignment_rmsd(query, candidates, budget=1)
    assert best == float("inf")
    assert exhaustive is True
    assert time.monotonic() - started < 0.05


def test_a_wide_search_is_bounded_by_the_node_budget() -> None:
    import numpy as np

    from kagiana.geometry import best_assignment_rmsd

    query = np.random.default_rng(1).normal(size=(6, 3))
    wide = [
        [np.array([float(i), float(j), 0.0]) for i in range(40)] for j in range(6)
    ]
    best, exhaustive = best_assignment_rmsd(query, wide)
    # The search was cut short, so it has ruled nothing out.
    assert exhaustive is False
    assert np.isfinite(best)


def test_a_rejected_row_reports_the_match_a_cpu_run_would(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A feature definition can be sensitive to explicit hydrogens.

    The prefilter has to add them to embed conformers; deciding the match
    column on that molecule would disagree with a CPU run of the same library.
    """
    from rdkit.Chem import AllChem

    from kagiana.backends import gpu3d
    from kagiana.backends.cpu3d import screen_molecule
    from kagiana.features import load_feature_factory
    from kagiana.pharmacophore3d import make_pharmacophore

    fdef = tmp_path / "carbon4.fdef"
    fdef.write_text(
        "DefineFeature FourConnectedCarbon [#6;D4]\n"
        "  Family Carbon4\n"
        "  Weights 1\n"
        "EndFeature\n",
        encoding="utf-8",
    )
    query = tmp_path / "q.csv"
    query.write_text(
        "feature_type,X,Y,Z\nCarbon4,0,0,0\nCarbon4,10,0,0\n", encoding="utf-8"
    )

    cpu = screen_molecule(
        "CC", load_feature_factory(fdef), make_pharmacophore(query, 1.5, 0.5), 10, 200
    )
    assert cpu is not None
    # Without explicit hydrogens an ethane carbon is not four-connected.
    assert cpu.match is False

    def fake_modules() -> tuple[Any, Any, Any, Any]:
        return None, None, None, (lambda **_kw: None)

    def fake_ensemble(mols: list[Any], confs: int, hardware: Any) -> None:
        for mol in mols:
            AllChem.EmbedMultipleConfs(mol, numConfs=confs, randomSeed=1)

    monkeypatch.setattr(gpu3d, "_require_gpu_modules", fake_modules)
    monkeypatch.setattr(gpu3d, "_embed_ensemble", fake_ensemble)

    decided, _rmsd = gpu3d.prefilter_chunk(
        ["CC"],
        fdef_path=str(fdef),
        pharmacophore_path=str(query),
        limit_upper=1.5,
        limit_lower=0.5,
        prefilter_conformers=5,
        prefilter_rmsd=0.0,
        quiet=True,
    )
    if 0 in decided:
        assert decided[0]["match"] == cpu.match


def test_diagnostics_survive_an_interrupted_chunked_run(
    tmp_path: Path, fdef: Path, pharmacophore: Path
) -> None:
    """A chunk's reasons must be on disk before that chunk becomes skippable."""
    library = tmp_path / "lib.csv"
    rows = ["ID,SMILES", "A,not-a-smiles"] + [
        f"C{i},NCCc1ccc(O)cc1" for i in range(6)
    ]
    library.write_text("\n".join(rows) + "\n", encoding="utf-8")

    out = tmp_path / "r.csv"
    argv = [
        "pharm3d",
        "--library", str(library), "--pharmacophore", str(pharmacophore),
        "--fdef", str(fdef), "--out", str(out),
        "--large", "--chunksize", "2", "--quiet",
    ]
    assert main(argv) == EXIT_OK
    errors = (tmp_path / "r_errors.jsonl").read_text(encoding="utf-8")
    assert "not-a-smiles" in errors
    # The reason is recorded, not just the status.
    assert json.loads(errors.splitlines()[0])["detail"]

    # Stand in for an interruption partway through: later chunks are gone.
    for index in (2, 3):
        csv_io.numbered_path(out, index).unlink(missing_ok=True)
        csv_io.screened_numbered_path(out, index).unlink(missing_ok=True)

    summary_path = tmp_path / "s.json"
    assert main([*argv, "--resume", "--json-summary", str(summary_path)]) == EXIT_OK
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_parse_error"] == 1
    assert "not-a-smiles" in (tmp_path / "r_errors.jsonl").read_text(encoding="utf-8")


def test_a_clean_rerun_clears_an_older_diagnostic_log(
    tmp_path: Path, fdef: Path, pharmacophore: Path
) -> None:
    """A log beside the results must describe the run that produced them."""
    broken = tmp_path / "broken.csv"
    broken.write_text("ID,SMILES\n1,not-a-smiles\n", encoding="utf-8")
    clean = tmp_path / "clean.csv"
    clean.write_text("ID,SMILES\n1,CCO\n", encoding="utf-8")

    out = tmp_path / "r.csv"
    base = ["--pharmacophore", str(pharmacophore), "--fdef", str(fdef),
            "--out", str(out), "--quiet"]
    assert main(["pharm3d", "--library", str(broken), *base]) == EXIT_OK
    assert (tmp_path / "r_errors.jsonl").is_file()

    assert main(["pharm3d", "--library", str(clean), *base]) == EXIT_OK
    assert not (tmp_path / "r_errors.jsonl").exists()


def test_an_unclosed_quote_is_reported_by_validate(tmp_path: Path) -> None:
    path = tmp_path / "unclosed.csv"
    path.write_text('ID,SMILES\n1,"CCO', encoding="utf-8")
    assert main(["validate", "--library", str(path), "--json"]) == EXIT_INPUT_ERROR


@pytest.mark.parametrize("mode", ["pharm2d", "pharm3d"])
def test_a_library_with_no_records_yields_an_empty_result(
    tmp_path: Path, mode: str, query: Path, fdef: Path, params: Path,
    pharmacophore: Path,
) -> None:
    """Accepting a header-only library means producing a readable result."""
    library = tmp_path / "empty.csv"
    library.write_text("ID,SMILES\n", encoding="utf-8")
    assert main(["validate", "--library", str(library), "--json"]) == EXIT_OK

    out = tmp_path / "r.csv"
    if mode == "pharm2d":
        argv = ["pharm2d", "--library", str(library), "--query", str(query),
                "--fdef", str(fdef), "--params", str(params),
                "--out", str(out), "--quiet"]
        expected = ["ID", "SMILES", "tanimoto", "dice", "simpson", "status", "elapsed_s"]
        result = csv_io.all_rows_path(out)
        index_col = None
    else:
        argv = ["pharm3d", "--library", str(library),
                "--pharmacophore", str(pharmacophore), "--fdef", str(fdef),
                "--out", str(out), "--quiet"]
        expected = ["ID", "SMILES", "match", "numMatch", "status", "elapsed_s",
                    "stage", "best_rmsd"]
        result = out
        index_col = 0

    assert main(argv) == EXIT_OK
    frame = pd.read_csv(result, index_col=index_col)
    assert list(frame.columns) == expected
    assert len(frame) == 0


def test_a_reference_that_cannot_be_fingerprinted_is_an_input_error(
    tmp_path: Path, library: Path, fdef: Path, params: Path
) -> None:
    """RDKit parses this molecule happily and then cannot sign it."""
    query_path = tmp_path / "q.csv"
    query_path.write_text("SMILES\nOc1ccc(O)cc1\n", encoding="utf-8")

    validate_code = main(
        ["validate", "--library", str(library), "--fdef", str(fdef),
         "--params", str(params), "--query", str(query_path), "--json"]
    )
    screen_code = main(
        ["pharm2d", "--library", str(library), "--query", str(query_path),
         "--fdef", str(fdef), "--params", str(params),
         "--out", str(tmp_path / "r.csv"), "--quiet"]
    )
    assert validate_code == screen_code == EXIT_INPUT_ERROR


def test_hits_are_counted_as_records_not_lines(
    tmp_path: Path, fdef: Path, pharmacophore: Path
) -> None:
    """A hit whose own cells contain a newline is still one hit."""
    library = tmp_path / "lib.csv"
    library.write_text(
        'ID,SMILES,note\n1,NCCc1ccc(O)cc1,"first\nsecond"\n', encoding="utf-8"
    )
    out = tmp_path / "r.csv"
    summary_path = tmp_path / "s.json"
    assert main(
        ["pharm3d", "--library", str(library), "--pharmacophore", str(pharmacophore),
         "--fdef", str(fdef), "--out", str(out), "--large", "--chunksize", "10",
         "--quiet", "--json-summary", str(summary_path)]
    ) == EXIT_OK

    combined = pd.read_csv(csv_io.screened_all_path(out), index_col=0)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(combined) == 1
    assert summary["n_screened"] == len(combined)
    assert combined["note"].iloc[0] == "first\nsecond"


# --- third review round ----------------------------------------------------


def _write(path: Path, *rows: str) -> Path:
    path.write_text("\n".join(("ID,SMILES", *rows)) + "\n", encoding="utf-8")
    return path


def test_resume_does_not_mix_results_from_a_different_library(
    tmp_path: Path, fdef: Path, pharmacophore: Path
) -> None:
    """Output files alone cannot say which run produced them.

    Screening a second library into the same place, then resuming, must not
    keep whichever half of the earlier results happens to still be on disk.
    """
    old = _write(tmp_path / "old.csv", "old0,NCCc1ccc(O)cc1", "old1,NCCc1ccc(OC)cc1")
    new = _write(tmp_path / "new.csv", "new0,c1ccccc1", "new1,C", "new2,CCO")
    out = tmp_path / "r.csv"
    base = ["--pharmacophore", str(pharmacophore), "--fdef", str(fdef), "--quiet"]

    assert main(["pharm3d", "--library", str(old), *base, "--out", str(out)]) == EXIT_OK
    assert list(pd.read_csv(csv_io.screened_path(out), index_col=0)["ID"]) == [
        "old0", "old1"
    ]

    # Stand in for an interruption between the two outputs: the full file has
    # been replaced with the new library's results, the hit list has not.
    staging = tmp_path / "staged.csv"
    assert main(["pharm3d", "--library", str(new), *base, "--out", str(staging)]) == EXIT_OK
    out.write_text(staging.read_text(encoding="utf-8"), encoding="utf-8")

    assert main(
        ["pharm3d", "--library", str(new), *base, "--out", str(out), "--resume"]
    ) == EXIT_OK
    full = pd.read_csv(out, index_col=0)
    hits = pd.read_csv(csv_io.screened_path(out), index_col=0)
    assert list(full["ID"]) == ["new0", "new1", "new2"]
    # Nothing from the earlier library may survive in either file.
    assert not set(hits["ID"]) & {"old0", "old1"}


def test_resume_does_not_keep_a_stale_chunk_from_a_different_library(
    tmp_path: Path, fdef: Path, pharmacophore: Path
) -> None:
    old = _write(tmp_path / "old.csv", "old0,NCCc1ccc(O)cc1", "old1,NCCc1ccc(OC)cc1")
    new = _write(tmp_path / "new.csv", "new0,c1ccccc1", "new1,C", "new2,CCO")
    out = tmp_path / "r.csv"
    base = [
        "--pharmacophore", str(pharmacophore), "--fdef", str(fdef),
        "--large", "--chunksize", "1", "--quiet",
    ]

    assert main(["pharm3d", "--library", str(old), *base, "--out", str(out)]) == EXIT_OK
    # Stand in for an interruption after the first chunk of the new run: chunk
    # 0 is the new library's, chunks 1 and 2 are missing or stale.
    assert main(
        ["pharm3d", "--library", str(new), *base, "--out", str(out), "--resume"]
    ) == EXIT_OK

    produced = [
        pd.read_csv(csv_io.numbered_path(out, index), index_col=0)["ID"].iloc[0]
        for index in range(3)
    ]
    assert produced == ["new0", "new1", "new2"]


def test_a_finished_run_publishes_its_diagnostics_when_resumed(
    tmp_path: Path, fdef: Path, pharmacophore: Path
) -> None:
    """Interrupting between the results and the log must not hide the reasons."""
    library = _write(tmp_path / "lib.csv", "1,not-a-smiles", "2,NCCc1ccc(O)cc1")
    out = tmp_path / "r.csv"
    argv = [
        "pharm3d", "--library", str(library),
        "--pharmacophore", str(pharmacophore), "--fdef", str(fdef),
        "--out", str(out), "--quiet",
    ]
    assert main(argv) == EXIT_OK
    log = tmp_path / "r_errors.jsonl"
    assert log.is_file()

    # Stand in for a crash after both CSVs were replaced but before the log was
    # published.
    log.unlink()
    summary_path = tmp_path / "s.json"
    assert main([*argv, "--resume", "--json-summary", str(summary_path)]) == EXIT_OK
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_parse_error"] == 1
    assert log.is_file()
    assert str(log) in summary["outputs"]


def test_a_chunk_that_failed_nothing_is_recorded_as_screened(
    tmp_path: Path, fdef: Path, pharmacophore: Path
) -> None:
    """"Screened, nothing wrong" and "never screened" must not look alike.

    Otherwise a resumed run would fall back on an older log and resurrect
    failures the current results do not have.
    """
    broken = _write(tmp_path / "broken.csv", "1,not-a-smiles", "2,NCCc1ccc(O)cc1")
    clean = _write(tmp_path / "clean.csv", "1,CCO", "2,NCCc1ccc(O)cc1")
    out = tmp_path / "r.csv"
    base = [
        "--pharmacophore", str(pharmacophore), "--fdef", str(fdef),
        "--large", "--chunksize", "1", "--quiet",
    ]

    assert main(["pharm3d", "--library", str(broken), *base, "--out", str(out)]) == EXIT_OK
    assert (tmp_path / "r_errors.jsonl").is_file()

    summary_path = tmp_path / "s.json"
    assert main(
        ["pharm3d", "--library", str(clean), *base, "--out", str(out),
         "--json-summary", str(summary_path)]
    ) == EXIT_OK
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_parse_error"] == 0
    assert not (tmp_path / "r_errors.jsonl").exists()


def test_many_chunks_with_diagnostics_do_not_exhaust_file_descriptors(
    tmp_path: Path, fdef: Path, pharmacophore: Path
) -> None:
    """One chunk's diagnostics are open at a time, whatever the chunk count."""
    import resource

    rows = [f"{i},not-a-smiles-{i}" for i in range(80)]
    library = _write(tmp_path / "lib.csv", *rows)
    out = tmp_path / "r.csv"

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(256, hard), hard))
    try:
        assert main(
            ["pharm3d", "--library", str(library),
             "--pharmacophore", str(pharmacophore), "--fdef", str(fdef),
             "--out", str(out), "--large", "--chunksize", "1", "--jobs", "1",
             "--quiet"]
        ) == EXIT_OK
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

    from kagiana.errlog import read_entries

    assert len(read_entries(tmp_path / "r_errors.jsonl")) == 80


def test_progress_reported_after_the_deadline_is_not_counted(
    fdef: Path, pharmacophore: Path
) -> None:
    """A timed-out row reports the work done inside the deadline, no more.

    The worker keeps generating poses until it is killed, and the parent may
    read its reports much later; neither may inflate the count. The bound is
    what could have been reported within the deadline: with 10 ms, at most the
    report made before the first assignment, which credits one pose for the
    assignment it interrupted. Before progress reports carried a time, a run
    whose result was read 300 ms late came back with 199.

    Comparing two runs' counts to each other would be a race rather than a
    test, since whether any report lands inside 10 ms depends on the machine.
    The rule itself is pinned by
    :func:`test_progress_from_after_the_deadline_does_not_reach_the_row`.
    """
    import time

    from kagiana.backends.cpu3d import init_worker
    from kagiana.pool import KillableWorkerPool

    for stall in (0.0, 0.3):
        with KillableWorkerPool(
            1, init_worker, (str(fdef), str(pharmacophore), 1.5, 0.5, 200, 20),
            timeout=0.01,
        ) as pool:
            for result in pool.map_ordered(["NCCc1ccc(O)cc1"]):
                if stall:
                    time.sleep(stall)
                assert result.status == Status.TIMEOUT
                counted = result.partial.num_match() if result.partial else 0
                assert counted <= 1, f"stall={stall} reported {counted} poses"


def test_the_fit_uses_the_features_a_cpu_run_would_see(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit hydrogens are needed to embed, but not to identify features.

    Measuring the fit on features found after adding them can reject a molecule
    the constrained pass would score.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from kagiana.backends import gpu3d
    from kagiana.backends.cpu3d import screen_molecule
    from kagiana.features import load_feature_factory
    from kagiana.pharmacophore3d import make_pharmacophore

    fdef = tmp_path / "tips.fdef"
    fdef.write_text(
        "DefineFeature TerminalOrOxyCarbon [#6;D1,$([#6]-[#8])]\n"
        "  Family CarbonTip\n  Weights 1\nEndFeature\n\n"
        "DefineFeature AnyNitrogen [#7]\n  Family Nitro\n  Weights 1\nEndFeature\n",
        encoding="utf-8",
    )
    query = tmp_path / "q.csv"
    query.write_text(
        "feature_type,X,Y,Z\nCarbonTip,0,0,0\nNitro,2.5,0,0\n", encoding="utf-8"
    )

    smiles = "CC(N)c1ccc(CO)cc1"
    cpu = screen_molecule(
        smiles, load_feature_factory(fdef), make_pharmacophore(query, 1.5, 0.5), 10, 200
    )
    assert cpu is not None and cpu.num_match > 0

    bare = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(bare)
    AllChem.EmbedMultipleConfs(mol, numConfs=20, randomSeed=42)
    best, exhaustive = gpu3d._best_rmsd_for_mol(
        mol,
        bare,
        load_feature_factory(fdef),
        ["CarbonTip", "Nitro"],
        np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]),
    )
    assert exhaustive is True
    # The default threshold is 1.5; a fit measured on the wrong representation
    # came out at 1.879 and the molecule was discarded.
    assert best < 1.5


def test_a_column_name_containing_a_newline_survives_aggregation(
    tmp_path: Path, fdef: Path, pharmacophore: Path
) -> None:
    """Skipping one physical line of a header would write the rest out as data."""
    import csv as csv_module

    library = tmp_path / "lib.csv"
    with open(library, "w", newline="", encoding="utf-8") as handle:
        writer = csv_module.writer(handle)
        writer.writerow(["ID", "SMILES", "vendor\nnote"])
        writer.writerow(["1", "NCCc1ccc(O)cc1", "a"])
        writer.writerow(["2", "NCCc1ccc(OC)cc1", "b"])

    out = tmp_path / "r.csv"
    summary_path = tmp_path / "s.json"
    assert main(
        ["pharm3d", "--library", str(library), "--pharmacophore", str(pharmacophore),
         "--fdef", str(fdef), "--out", str(out), "--large", "--chunksize", "1",
         "--quiet", "--json-summary", str(summary_path)]
    ) == EXIT_OK

    with open(csv_io.screened_all_path(out), newline="", encoding="utf-8") as handle:
        records = list(csv_module.reader(handle))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(records) - 1 == summary["n_screened"] == 2
    assert "vendor\nnote" in records[0]


# --- fourth review round ---------------------------------------------------


def test_the_deadline_is_judged_on_the_moment_the_work_finished() -> None:
    """The rule itself, independent of how a delay came about.

    A worker can be slow to pick up its task as well as slow to do it, and the
    parent can be slow to read the answer. Only one of those is the task
    overrunning, so the verdict comes from the time the worker reports against
    the deadline the parent set — not from an elapsed time, which misses the
    wait before the work began, and not from the parent's clock when it reads.
    """
    from kagiana.pool import _IDLE, KillableWorkerPool, _Slot

    pool = KillableWorkerPool(1, init_quick, timeout=0.05)
    dispatched_at = 100.0

    def slot_at(finished_at: float) -> tuple[_Slot, object]:
        class _Pipe:
            """Hands over one already-sent message and nothing more."""

            def __init__(self, message: object) -> None:
                self._message = message

            def recv(self) -> object:
                return self._message

        slot = _Slot(
            conn=_Pipe(("ok", (0, "value", finished_at))),  # type: ignore[arg-type]
            process=None,
            state="busy",
            index=0,
            deadline=dispatched_at + 0.05,
        )
        slot.dispatched_at = dispatched_at
        return slot, slot.conn

    # Finished inside the deadline: a success, whenever it is read.
    slot, _ = slot_at(dispatched_at + 0.01)
    pool._slots = [slot]
    in_time = pool._receive(0, slot)
    assert in_time is not None
    assert in_time.status == Status.OK
    assert in_time.value == "value"

    # Began late and so finished past the deadline: a timeout, even though the
    # work itself took no time at all.
    slot, _ = slot_at(dispatched_at + 0.14)
    pool._slots = [slot]
    overran = pool._receive(0, slot)
    assert overran is not None
    assert overran.status == Status.TIMEOUT
    assert overran.value is None
    assert slot.state == _IDLE


def test_progress_from_after_the_deadline_does_not_reach_the_row() -> None:
    """Same rule for a partial count: the report carries when it was made."""
    from kagiana.pool import KillableWorkerPool, _Slot

    pool = KillableWorkerPool(1, init_quick, timeout=0.05)
    dispatched_at = 100.0

    class _Pipe:
        def __init__(self, message: object) -> None:
            self._message = message

        def recv(self) -> object:
            return self._message

    def deliver(reported_at: float, value: str) -> _Slot:
        slot = _Slot(
            conn=_Pipe(("progress", (0, reported_at, value))),  # type: ignore[arg-type]
            process=None,
            state="busy",
            index=0,
            deadline=dispatched_at + 0.05,
        )
        slot.dispatched_at = dispatched_at
        pool._slots = [slot]
        assert pool._receive(0, slot) is None
        return slot

    assert deliver(dispatched_at + 0.01, "in time").progress == "in time"
    assert deliver(dispatched_at + 0.14, "too late").progress is None


def test_a_task_that_finished_in_time_survives_a_slow_consumer_end_to_end() -> None:
    """The same two cases through real worker processes."""
    import time

    from kagiana.pool import KillableWorkerPool

    from .helpers import init_late_then_report

    with KillableWorkerPool(1, init_late_then_report, timeout=0.05) as pool:
        results = pool.map_ordered(["C"])
        time.sleep(0.4)
        late = next(results)
    assert late.status == Status.TIMEOUT
    assert late.partial is None

    with KillableWorkerPool(1, init_quick, timeout=0.05) as pool:
        results = pool.map_ordered(["C"])
        time.sleep(0.4)
        prompt = next(results)
    assert prompt.status == Status.OK


def test_a_colliding_state_directory_is_reported_not_deleted(
    tmp_path: Path, library: Path, fdef: Path, pharmacophore: Path
) -> None:
    """The state directory's name comes from --out, so it can collide."""
    from kagiana.runstate import RunState, generation_key, state_directory

    out = tmp_path / "data.csv"
    theirs = state_directory(out)
    theirs.mkdir()
    keepsake = theirs / "important.txt"
    keepsake.write_text("keep me", encoding="utf-8")

    with pytest.raises(Exception) as caught:
        RunState.open(out, generation_key(mode="test"), resume=True)
    assert "not created by this tool" in str(caught.value)
    assert keepsake.read_text(encoding="utf-8") == "keep me"

    # The screening command reports it as an input problem rather than crashing.
    assert main(
        ["pharm3d", "--library", str(library), "--pharmacophore", str(pharmacophore),
         "--fdef", str(fdef), "--out", str(out), "--quiet"]
    ) == EXIT_INPUT_ERROR
    assert keepsake.read_text(encoding="utf-8") == "keep me"


def test_pharm2d_resume_does_not_reuse_another_librarys_results(
    tmp_path: Path, query: Path, fdef: Path, params: Path
) -> None:
    old = _write(tmp_path / "old.csv", "old0,NCCc1ccc(O)cc1", "old1,NCCc1ccc(OC)cc1")
    new = _write(tmp_path / "new.csv", "new0,CC(=O)Oc1ccccc1C(=O)O", "new1,C")
    out = tmp_path / "r.csv"
    base = ["--query", str(query), "--fdef", str(fdef), "--params", str(params),
            "--quiet"]

    assert main(["pharm2d", "--library", str(old), *base, "--out", str(out)]) == EXIT_OK
    assert set(pd.read_csv(out)["ID"]) == {"old0", "old1"}

    assert main(
        ["pharm2d", "--library", str(new), *base, "--out", str(out), "--resume"]
    ) == EXIT_OK
    assert list(pd.read_csv(csv_io.all_rows_path(out))["ID"]) == ["new0", "new1"]
    survivors = pd.read_csv(out)
    if len(survivors):
        assert not set(survivors["ID"]) & {"old0", "old1"}


def test_pharm2d_resume_republishes_diagnostics_and_counts(
    tmp_path: Path, query: Path, fdef: Path, params: Path
) -> None:
    library = _write(
        tmp_path / "lib.csv", "1,not-a-smiles", "2,also-broken", "3,NCCc1ccc(O)cc1"
    )
    out = tmp_path / "r.csv"
    argv = ["pharm2d", "--library", str(library), "--query", str(query),
            "--fdef", str(fdef), "--params", str(params), "--out", str(out), "--quiet"]
    assert main(argv) == EXIT_OK

    # Stand in for a crash after the results were replaced but before the log
    # was published.
    (tmp_path / "r_errors.jsonl").unlink()
    summary_path = tmp_path / "s.json"
    assert main([*argv, "--resume", "--json-summary", str(summary_path)]) == EXIT_OK

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_total"] == 3
    assert summary["n_parse_error"] == 2
    assert (tmp_path / "r_errors.jsonl").is_file()


@pytest.mark.parametrize(
    "field", ["prefilter_rmsd", "prefilter_conformers", "backend"]
)
def test_settings_that_change_a_row_change_the_resume_key(field: str) -> None:
    """Resuming must not reuse results the current settings would not produce."""
    from kagiana.runstate import generation_key

    settings: dict[str, Any] = {
        "mode": "pharm3d",
        "backend": "gpu",
        "library": Path("tests/data/library.csv"),
        "pharmacophore": Path("tests/data/pharmacophore.csv"),
        "fdef": Path("tests/data/test.fdef"),
        "large": False,
        "chunksize": None,
        "conformers": 10,
        "max_poses": 20,
        "limit_upper": 1.5,
        "limit_lower": 0.5,
        "timeout": 10.0,
        "prefilter_rmsd": 0.0,
        "prefilter_conformers": 20,
    }
    baseline = generation_key(**settings)
    assert generation_key(**settings) == baseline

    changed = dict(settings)
    changed[field] = {"prefilter_rmsd": 100.0, "prefilter_conformers": 50, "backend": "cpu"}[field]
    assert generation_key(**changed) != baseline


def test_the_fit_respects_the_weights_in_a_feature_definition() -> None:
    """A feature's point is where its definition puts it, not its atoms' mean.

    RDKit's zinc binder is weighted 1,0 and sits on the sulfur; averaging the
    sulfur and the carbon moves the point and rejects a perfect match.
    """
    import os

    import numpy as np
    from rdkit import Chem, RDConfig
    from rdkit.Chem import AllChem

    from kagiana.backends import gpu3d
    from kagiana.features import load_feature_factory

    factory = load_feature_factory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
    bare = Chem.MolFromSmiles("SCc1ccccc1")
    mol = Chem.AddHs(bare)
    AllChem.EmbedMultipleConfs(mol, numConfs=1, randomSeed=7)

    # Take the query from this very conformer, so a perfect fit exists.
    posed = gpu3d._posed_copy(bare)
    gpu3d._copy_positions(mol.GetConformer(), posed, bare.GetNumAtoms())
    points: dict[str, Any] = {}
    for feature in factory.GetFeaturesForMol(posed):
        if feature.GetFamily() in ("ZnBinder", "Aromatic"):
            points.setdefault(feature.GetFamily(), feature.GetPos())
    assert set(points) == {"ZnBinder", "Aromatic"}

    types = list(points)
    coords = np.array([[points[t].x, points[t].y, points[t].z] for t in types])
    best, exhaustive = gpu3d._best_rmsd_for_mol(mol, bare, factory, types, coords)
    assert exhaustive is True
    # Averaging the atoms gave 0.308, which a 0.1 threshold discards.
    assert best < 0.01


# --- fifth review round ----------------------------------------------------


def test_a_state_directory_from_an_earlier_version_is_adopted(
    tmp_path: Path, library: Path, fdef: Path, pharmacophore: Path
) -> None:
    """Versions before the ownership marker wrote none; their runs still resume."""
    from kagiana.runstate import RunState, generation_key, state_directory

    out = tmp_path / "r.csv"
    directory = state_directory(out)
    directory.mkdir()
    key = generation_key(mode="test")
    (directory / "state.json").write_text(
        json.dumps({"version": 1, "key": key, "completed": [0, 1], "finalized": True}),
        encoding="utf-8",
    )

    state = RunState.open(out, key, resume=True)
    assert state.completed == {0, 1}
    assert state.finalized is True


def test_only_the_files_this_tool_creates_are_removed(tmp_path: Path) -> None:
    """A caller may keep something of their own in the state directory."""
    from kagiana.runstate import RunState, generation_key, state_directory

    out = tmp_path / "r.csv"
    directory = state_directory(out)
    directory.mkdir()
    (directory / ".kagiana-run-state").write_text("", encoding="utf-8")
    mine = directory / "errors_manual.jsonl"
    mine.write_text("mine", encoding="utf-8")
    ours = directory / "errors_0.jsonl"
    ours.write_text("ours", encoding="utf-8")

    # A fresh generation clears what this tool wrote, and nothing else.
    RunState.open(out, generation_key(mode="fresh"), resume=False)
    assert mine.read_text(encoding="utf-8") == "mine"
    assert not ours.exists()


def test_dropping_no_all_rows_does_not_adopt_an_older_full_file(
    tmp_path: Path, query: Path, fdef: Path, params: Path
) -> None:
    """Which outputs a run owes is part of what makes a run the same run."""
    old = _write(tmp_path / "old.csv", "old0,not-a-smiles", "old1,also-broken")
    new = _write(tmp_path / "new.csv", "new0,NCCc1ccc(O)cc1")
    out = tmp_path / "r.csv"
    base = ["pharm2d", "--query", str(query), "--fdef", str(fdef),
            "--params", str(params), "--out", str(out), "--quiet"]

    assert main([*base, "--library", str(old)]) == EXIT_OK
    assert list(pd.read_csv(csv_io.all_rows_path(out))["ID"]) == ["old0", "old1"]

    # The new library is screened without the full-row file, then resumed with
    # it: the older file must not be passed off as this run's.
    assert main([*base, "--library", str(new), "--no-all-rows"]) == EXIT_OK
    assert main([*base, "--library", str(new), "--resume"]) == EXIT_OK
    assert list(pd.read_csv(csv_io.all_rows_path(out))["ID"]) == ["new0"]


def test_resume_repeats_the_counts_without_the_full_row_file(
    tmp_path: Path, query: Path, fdef: Path, params: Path
) -> None:
    """The tallies come from what the run recorded, not from re-reading files.

    With --no-all-rows there is no file to re-read, and even with one the chunk
    count is not recoverable from it.
    """
    library = _write(tmp_path / "lib.csv", "1,NCCc1ccc(O)cc1", "2,not-a-smiles")
    out = tmp_path / "r.csv"
    argv = ["pharm2d", "--library", str(library), "--query", str(query),
            "--fdef", str(fdef), "--params", str(params), "--out", str(out),
            "--no-all-rows", "--quiet"]

    first_path = tmp_path / "first.json"
    assert main([*argv, "--json-summary", str(first_path)]) == EXIT_OK
    second_path = tmp_path / "second.json"
    assert main([*argv, "--resume", "--json-summary", str(second_path)]) == EXIT_OK

    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    for field in ("n_total", "n_screened", "n_parse_error", "n_error", "n_chunks"):
        assert second[field] == first[field], field


def test_resume_repeats_the_chunk_count_in_the_default_mode(
    tmp_path: Path, fdef: Path, pharmacophore: Path
) -> None:
    library = _write(tmp_path / "lib.csv", "1,NCCc1ccc(O)cc1", "2,c1ccccc1")
    out = tmp_path / "r.csv"
    argv = ["pharm3d", "--library", str(library),
            "--pharmacophore", str(pharmacophore), "--fdef", str(fdef),
            "--out", str(out), "--quiet"]

    first_path = tmp_path / "first.json"
    assert main([*argv, "--json-summary", str(first_path)]) == EXIT_OK
    second_path = tmp_path / "second.json"
    assert main([*argv, "--resume", "--json-summary", str(second_path)]) == EXIT_OK

    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    assert second["n_chunks"] == first["n_chunks"] == 1
    assert second["n_total"] == first["n_total"]


# --- sixth review round ----------------------------------------------------


def test_another_application_state_file_is_not_adopted(tmp_path: Path) -> None:
    """`version: 1` is a thing many programs write.

    The directory's name comes from --out, so a collision can be with another
    tool's own record; adopting it on the version alone overwrote it.
    """
    from kagiana.runstate import RunState, generation_key, state_directory
    from kagiana.schema import InputError

    out = tmp_path / "other.csv"
    directory = state_directory(out)
    directory.mkdir()
    theirs = {"version": 1, "application": "inventory", "cursor": 17}
    state_file = directory / "state.json"
    state_file.write_text(json.dumps(theirs), encoding="utf-8")

    with pytest.raises(InputError, match="not created by this tool"):
        RunState.open(out, generation_key(mode="test"), resume=True)
    assert json.loads(state_file.read_text(encoding="utf-8")) == theirs
    assert not (directory / ".kagiana-run-state").exists()


@pytest.mark.parametrize(
    "stored",
    [
        {"version": 1, "key": "not-a-key", "completed": [], "finalized": False},
        {"version": 1, "key": "0" * 32, "completed": "all", "finalized": False},
        {"version": 1, "key": "0" * 32, "completed": [0], "finalized": "yes"},
        {"version": 1, "key": "0" * 32, "completed": [-1], "finalized": False},
        {"version": 1, "key": "0" * 32, "completed": [0], "finalized": False, "summary": []},
    ],
)
def test_a_state_file_of_the_wrong_shape_is_not_ours(
    tmp_path: Path, stored: dict[str, Any]
) -> None:
    from kagiana.runstate import RunState, generation_key, state_directory
    from kagiana.schema import InputError

    out = tmp_path / "r.csv"
    directory = state_directory(out)
    directory.mkdir()
    (directory / "state.json").write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(InputError):
        RunState.open(out, generation_key(mode="test"), resume=True)


def test_a_record_without_tallies_recovers_them_from_the_results(
    tmp_path: Path, fdef: Path, pharmacophore: Path
) -> None:
    """Earlier versions recorded a finished run without its counts.

    Reporting such a run as having screened nothing would be worse than
    re-reading the files it wrote, which still hold the answer.
    """
    library = _write(tmp_path / "lib.csv", "1,NCCc1ccc(O)cc1", "2,not-a-smiles")
    out = tmp_path / "r.csv"
    argv = ["pharm3d", "--library", str(library),
            "--pharmacophore", str(pharmacophore), "--fdef", str(fdef),
            "--out", str(out), "--quiet"]

    first_path = tmp_path / "first.json"
    assert main([*argv, "--json-summary", str(first_path)]) == EXIT_OK
    first = json.loads(first_path.read_text(encoding="utf-8"))
    assert first["n_parse_error"] == 1

    # Strip the tallies, leaving the record as an earlier version wrote it.
    from kagiana.runstate import state_directory

    state_file = state_directory(out) / "state.json"
    stored = json.loads(state_file.read_text(encoding="utf-8"))
    del stored["summary"]
    state_file.write_text(json.dumps(stored), encoding="utf-8")

    second_path = tmp_path / "second.json"
    assert main([*argv, "--resume", "--json-summary", str(second_path)]) == EXIT_OK
    second = json.loads(second_path.read_text(encoding="utf-8"))
    for field in ("n_total", "n_screened", "n_parse_error", "n_chunks"):
        assert second[field] == first[field], field

    # Recovered once, then stored, so the next resume needs no re-reading.
    assert json.loads(state_file.read_text(encoding="utf-8"))["summary"]
