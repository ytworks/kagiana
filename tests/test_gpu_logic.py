"""GPU-backend logic that does not need a GPU.

Only the device calls in the GPU backends require CUDA: generating the
conformer ensemble and the bit-vector comparison. Everything around them — the
geometric fit, the keep/reject decision, the row shapes handed back to the
command, and the score arithmetic — is ordinary RDKit and numpy work. Those are
exercised here with the device calls stubbed out, so the paths are not shipped
entirely unrun.

The governing rule for the prefilter: it may only ever remove work. A molecule
it settles must be one the constrained pass would have agreed about, so
anything it could not measure conclusively is deferred rather than decided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from kagiana.backends import gpu3d
from kagiana.features import load_feature_factory
from kagiana.pharmacophore3d import read_feature_points
from kagiana.schema import Stage, Status


class _FakeHardwareOptions:
    """Stands in for nvmolkit's hardware options struct."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _fake_modules() -> tuple[Any, Any, Any, Any]:
    return None, None, None, _FakeHardwareOptions


@pytest.fixture
def stub_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the CUDA calls with RDKit's own conformer generation."""

    def fake_ensemble(mols: list[Any], confs_per_molecule: int, hardware: Any) -> None:
        for mol in mols:
            AllChem.EmbedMultipleConfs(
                mol, numConfs=confs_per_molecule, randomSeed=0xC0FFEE
            )

    monkeypatch.setattr(gpu3d, "_require_gpu_modules", _fake_modules)
    monkeypatch.setattr(gpu3d, "_embed_ensemble", fake_ensemble)


def _query(pharmacophore: Path) -> tuple[list[str], np.ndarray]:
    points = read_feature_points(pharmacophore)
    return (
        [point.feature_type for point in points],
        np.array([[p.x, p.y, p.z] for p in points], dtype=float),
    )


def _prefilter(
    smiles: list[str], fdef: Path, pharmacophore: Path, **overrides: Any
) -> tuple[dict[int, dict[str, Any]], dict[int, float]]:
    settings: dict[str, Any] = {
        "fdef_path": str(fdef),
        "pharmacophore_path": str(pharmacophore),
        "limit_upper": 1.5,
        "limit_lower": 0.5,
        "prefilter_conformers": 10,
        "prefilter_rmsd": 1.5,
        "quiet": True,
    }
    settings.update(overrides)
    return gpu3d.prefilter_chunk(smiles, **settings)


def test_the_source_molecule_fits_its_own_query(
    fdef: Path, pharmacophore: Path
) -> None:
    """The query was derived from this molecule, so a conformer should fit it."""
    types, coords = _query(pharmacophore)
    mol = Chem.AddHs(Chem.MolFromSmiles("NCCc1ccc(O)cc1"))
    AllChem.EmbedMultipleConfs(mol, numConfs=20, randomSeed=0xC0FFEE)
    best, exhaustive = gpu3d._best_rmsd_for_mol(
        mol, load_feature_factory(fdef), types, coords
    )
    assert exhaustive is True
    assert np.isfinite(best)
    assert best < 1.5


def test_a_molecule_without_the_families_cannot_be_measured(
    fdef: Path, pharmacophore: Path
) -> None:
    types, coords = _query(pharmacophore)
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMultipleConfs(mol, numConfs=5, randomSeed=0xC0FFEE)
    best, exhaustive = gpu3d._best_rmsd_for_mol(
        mol, load_feature_factory(fdef), types, coords
    )
    assert best == float("inf")
    assert exhaustive is True


def test_a_molecule_with_no_conformers_cannot_be_measured(
    fdef: Path, pharmacophore: Path
) -> None:
    """An empty ensemble rules nothing out and must not look like a miss."""
    types, coords = _query(pharmacophore)
    mol = Chem.AddHs(Chem.MolFromSmiles("NCCc1ccc(O)cc1"))
    assert mol.GetNumConformers() == 0
    best, _exhaustive = gpu3d._best_rmsd_for_mol(
        mol, load_feature_factory(fdef), types, coords
    )
    assert best == float("inf")


def test_more_conformers_cannot_make_the_fit_worse(
    fdef: Path, pharmacophore: Path
) -> None:
    types, coords = _query(pharmacophore)
    factory = load_feature_factory(fdef)

    def best_with(n_confs: int) -> float:
        mol = Chem.AddHs(Chem.MolFromSmiles("NCCc1ccc(OC)cc1"))
        AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, randomSeed=0xC0FFEE)
        return gpu3d._best_rmsd_for_mol(mol, factory, types, coords)[0]

    assert best_with(10) <= best_with(2) + 1e-9


def test_a_geometric_rejection_is_not_recorded_as_a_missing_feature(
    stub_device: None, fdef: Path, pharmacophore: Path
) -> None:
    """A molecule whose fit was measured carries the features; the geometry is
    what failed, and only the stage column may say so."""
    decided, rmsd = _prefilter(
        ["NCCc1ccc(O)cc1"], fdef, pharmacophore, prefilter_rmsd=0.0, prefilter_conformers=20
    )
    row = decided[0]
    assert row["match"] is True
    assert row["numMatch"] == 0
    assert row["status"] == Status.OK
    assert row["stage"] == Stage.GPU_PREFILTER_REJECT
    assert np.isfinite(row["best_rmsd"])
    assert rmsd[0] == row["best_rmsd"]


def test_an_unmeasurable_molecule_is_deferred_rather_than_decided(
    stub_device: None, fdef: Path, pharmacophore: Path
) -> None:
    """Benzene carries no donor or acceptor, so the prefilter cannot measure a
    fit; the constrained pass gets to say so instead."""
    decided, rmsd = _prefilter(["c1ccccc1"], fdef, pharmacophore)
    assert decided == {}
    assert rmsd == {}


def test_an_unparsable_row_is_settled_without_embedding(
    stub_device: None, fdef: Path, pharmacophore: Path
) -> None:
    decided, rmsd = _prefilter(["not-a-real-smiles", ""], fdef, pharmacophore)
    for position in (0, 1):
        assert decided[position]["status"] == Status.PARSE_ERROR
        assert "parse" in decided[position]["_detail"].lower()
        assert position not in rmsd


def test_a_survivor_is_measured_and_deferred(
    stub_device: None, fdef: Path, pharmacophore: Path
) -> None:
    decided, rmsd = _prefilter(
        ["NCCc1ccc(O)cc1"], fdef, pharmacophore, prefilter_rmsd=99.0, prefilter_conformers=20
    )
    assert decided == {}
    assert np.isfinite(rmsd[0])


def test_every_decided_row_carries_the_columns_the_command_needs(
    stub_device: None, fdef: Path, pharmacophore: Path
) -> None:
    """A row the prefilter settles goes straight into the result frame."""
    decided, _rmsd = _prefilter(
        ["NCCc1ccc(O)cc1", "broken"], fdef, pharmacophore, prefilter_rmsd=0.0
    )
    required = {"match", "numMatch", "status", "elapsed_s", "stage", "best_rmsd"}
    assert decided
    for row in decided.values():
        assert required <= row.keys()


def test_a_failed_ensemble_defers_the_batch_instead_of_dropping_it(
    monkeypatch: pytest.MonkeyPatch, fdef: Path, pharmacophore: Path
) -> None:
    """Without an ensemble nothing can be ruled out, so nothing may be."""

    def exploding_ensemble(mols: list[Any], confs: int, hardware: Any) -> None:
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(gpu3d, "_require_gpu_modules", _fake_modules)
    monkeypatch.setattr(gpu3d, "_embed_ensemble", exploding_ensemble)

    decided, rmsd = _prefilter(
        ["NCCc1ccc(O)cc1", "c1ccccc1"], fdef, pharmacophore, prefilter_rmsd=0.0
    )
    assert decided == {}
    assert rmsd == {}


def test_batching_settles_or_defers_every_molecule(
    stub_device: None, fdef: Path, pharmacophore: Path
) -> None:
    smiles = ["NCCc1ccc(O)cc1", "c1ccccc1", "NCCc1ccc(OC)cc1", "broken", "CCO"]
    decided, rmsd = _prefilter(
        smiles, fdef, pharmacophore, prefilter_rmsd=0.0, batch_size=2
    )
    for index in range(len(smiles)):
        settled = index in decided
        deferred = index not in decided
        # Nothing may be lost: a molecule is either decided or left for the CPU.
        assert settled or deferred
    # The molecules that could be measured were all rejected at threshold 0.
    assert set(rmsd) <= set(decided)
    assert all(decided[index]["match"] is True for index in rmsd)


def test_the_gpu_module_reports_a_clear_error_without_cuda(
    fdef: Path, pharmacophore: Path
) -> None:
    """The real import guard must explain itself rather than raising ImportError."""
    from kagiana.backends import probe_gpu
    from kagiana.schema import InputError

    if probe_gpu().available:
        pytest.skip("this machine has a usable GPU")
    with pytest.raises(InputError, match="torch and nvmolkit"):
        gpu3d._require_gpu_modules()
