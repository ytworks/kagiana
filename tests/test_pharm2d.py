"""Tests for 2D screening: score definitions, cutoffs and output files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kagiana.backends.cpu2d import similarity_scores
from kagiana.backends.gpu2d import (
    expected_scores_match_cpu,
    intersections_numpy,
    pack_uint8,
)
from kagiana.cli import main
from kagiana.query2d import build_sig_factory, fingerprint, load_params
from kagiana.schema import InputError, Status


def test_params_are_read_in_full(params: Path) -> None:
    loaded = load_params(params)
    assert loaded.n_maxpoint == 3
    assert loaded.bins == ((0.0, 2.0), (2.0, 5.0), (5.0, 8.0))
    assert loaded.skip_feature == ("NegIonizable",)
    assert loaded.chunksize == 4
    assert loaded.tanimoto_cutoff == 0.3


def test_a_missing_params_key_is_an_input_error(tmp_path: Path) -> None:
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"n_maxpoint": 3}), encoding="utf-8")
    with pytest.raises(InputError, match="missing required key"):
        load_params(path)


def test_cli_overrides_beat_the_params_file(params: Path) -> None:
    loaded = load_params(params).with_overrides(tanimoto_cutoff=0.9, chunksize=11)
    assert loaded.tanimoto_cutoff == 0.9
    assert loaded.chunksize == 11
    # Untouched values stay as the file had them.
    assert loaded.dice_cutoff == 0.3


def test_identical_molecules_score_one(fdef: Path, params: Path) -> None:
    sig_factory = build_sig_factory(fdef, load_params(params))
    fp = fingerprint("NCCc1ccc(O)cc1", sig_factory)
    scores = similarity_scores(fp, fp)
    assert scores.tanimoto == pytest.approx(1.0)
    assert scores.dice == pytest.approx(1.0)
    assert scores.simpson == pytest.approx(1.0)


def test_scores_follow_their_definitions(fdef: Path, params: Path) -> None:
    sig_factory = build_sig_factory(fdef, load_params(params))
    left = fingerprint("NCCc1ccc(O)cc1", sig_factory)
    right = fingerprint("NCCc1ccc(OC)cc1", sig_factory)

    a = left.GetNumOnBits()
    b = right.GetNumOnBits()
    i = (left & right).GetNumOnBits()
    scores = similarity_scores(left, right)

    assert scores.tanimoto == pytest.approx(i / (a + b - i))
    assert scores.dice == pytest.approx(2 * i / (a + b))
    assert scores.simpson == pytest.approx(i / min(a, b))


def test_a_featureless_molecule_scores_zero_rather_than_dividing_by_zero(
    fdef: Path, params: Path
) -> None:
    sig_factory = build_sig_factory(fdef, load_params(params))
    reference = fingerprint("NCCc1ccc(O)cc1", sig_factory)
    methane = fingerprint("C", sig_factory)
    assert methane.GetNumOnBits() == 0
    scores = similarity_scores(reference, methane)
    assert (scores.tanimoto, scores.dice, scores.simpson) == (0.0, 0.0, 0.0)


def test_gpu_packing_reproduces_rdkit_intersections(fdef: Path, params: Path) -> None:
    """The packed-bit path the GPU uses must agree with RDKit bit-for-bit."""
    sig_factory = build_sig_factory(fdef, load_params(params))
    reference = fingerprint("NCCc1ccc(O)cc1", sig_factory)
    others = ["NCCc1ccc(OC)cc1", "NCCc1ccccc1", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    fingerprints = [fingerprint(smiles, sig_factory) for smiles in others]

    n_bits = reference.GetNumBits()
    packed_reference = pack_uint8([tuple(reference.GetOnBits())], n_bits)
    packed_library = pack_uint8(
        [tuple(fp.GetOnBits()) for fp in fingerprints], n_bits
    )
    intersections = intersections_numpy(packed_reference, packed_library)

    for offset, fp in enumerate(fingerprints):
        expected = (reference & fp).GetNumOnBits()
        assert int(intersections[offset]) == expected

        cpu = similarity_scores(reference, fp)
        gpu = expected_scores_match_cpu(
            reference.GetNumOnBits(), fp.GetNumOnBits(), int(intersections[offset])
        )
        assert gpu == pytest.approx((cpu.tanimoto, cpu.dice, cpu.simpson))


def test_screening_writes_the_expected_files(
    tmp_path: Path, library: Path, query: Path, fdef: Path, params: Path
) -> None:
    out = tmp_path / "r.csv"
    summary_path = tmp_path / "summary.json"
    code = main(
        [
            "pharm2d",
            "--library", str(library),
            "--query", str(query),
            "--fdef", str(fdef),
            "--params", str(params),
            "--out", str(out),
            "--jobs", "2",
            "--quiet",
            "--json-summary", str(summary_path),
        ]
    )
    assert code == 0

    screened = pd.read_csv(out)
    all_rows = pd.read_csv(tmp_path / "r_all.csv")

    # Input columns survive, score and diagnostic columns are appended in order.
    assert list(all_rows.columns) == [
        "ID", "SMILES", "vendor_note",
        "tanimoto", "dice", "simpson", "status", "elapsed_s",
    ]
    # No index column was written.
    assert "Unnamed: 0" not in all_rows.columns
    assert len(all_rows) == 10

    # Every surviving row clears every cutoff.
    assert (screened["tanimoto"] >= 0.3).all()
    assert (screened["dice"] >= 0.3).all()
    assert (screened["simpson"] >= 0.3).all()
    assert set(screened["ID"]) <= set(all_rows["ID"])

    # The reference molecule is in the library and must score a perfect match.
    reference_row = all_rows[all_rows["SMILES"] == "NCCc1ccc(O)cc1"].iloc[0]
    assert reference_row["tanimoto"] == pytest.approx(1.0)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["mode"] == "pharm2d"
    assert summary["backend"] == "cpu"
    assert summary["n_total"] == 10
    assert summary["n_screened"] == len(screened)
    # The library holds a deliberately broken SMILES string.
    assert summary["n_parse_error"] == 1
    assert summary["n_chunks"] == 3


def test_unparsable_rows_are_flagged_not_fatal(
    tmp_path: Path, library: Path, query: Path, fdef: Path, params: Path
) -> None:
    out = tmp_path / "r.csv"
    assert main(
        [
            "pharm2d",
            "--library", str(library), "--query", str(query),
            "--fdef", str(fdef), "--params", str(params),
            "--out", str(out), "--quiet",
        ]
    ) == 0

    all_rows = pd.read_csv(tmp_path / "r_all.csv")
    broken = all_rows[all_rows["SMILES"] == "this-is-not-a-smiles"].iloc[0]
    assert broken["status"] == Status.PARSE_ERROR
    assert np.isnan(broken["tanimoto"])

    # The reason is recorded next to the results.
    entries = [
        json.loads(line)
        for line in (tmp_path / "r_errors.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(entry["SMILES"] == "this-is-not-a-smiles" for entry in entries)
    assert all({"chunk", "row", "SMILES", "status", "detail"} <= entry.keys() for entry in entries)


def test_no_all_rows_suppresses_the_extra_file(
    tmp_path: Path, library: Path, query: Path, fdef: Path, params: Path
) -> None:
    out = tmp_path / "r.csv"
    assert main(
        [
            "pharm2d",
            "--library", str(library), "--query", str(query),
            "--fdef", str(fdef), "--params", str(params),
            "--out", str(out), "--quiet", "--no-all-rows",
        ]
    ) == 0
    assert out.is_file()
    assert not (tmp_path / "r_all.csv").exists()


def test_a_stricter_cutoff_keeps_fewer_rows(
    tmp_path: Path, library: Path, query: Path, fdef: Path, params: Path
) -> None:
    def run(cutoff: str, name: str) -> int:
        out = tmp_path / name
        assert main(
            [
                "pharm2d",
                "--library", str(library), "--query", str(query),
                "--fdef", str(fdef), "--params", str(params),
                "--out", str(out), "--quiet", "--tanimoto-cutoff", cutoff,
            ]
        ) == 0
        return len(pd.read_csv(out))

    assert run("0.0", "loose.csv") > run("0.99", "tight.csv")
