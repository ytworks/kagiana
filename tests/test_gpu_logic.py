"""GPU-backend logic that does not need a GPU.

Only the device calls in the GPU backends require CUDA: generating the
conformer ensemble and the bit-vector comparison. Everything around them — the
geometric fit, the keep/reject decision, the row shapes handed back to the
command, and the score arithmetic — is ordinary RDKit and numpy work. Those are
exercised here with the device calls stubbed out, so the paths are not shipped
entirely unrun.
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


@pytest.fixture
def stub_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the CUDA calls with RDKit's own conformer generation."""

    def fake_modules() -> tuple[Any, Any, Any, Any]:
        return None, None, None, _FakeHardwareOptions

    def fake_ensemble(mols: list[Any], confs_per_molecule: int, hardware: Any) -> None:
        for mol in mols:
            AllChem.EmbedMultipleConfs(
                mol, numConfs=confs_per_molecule, randomSeed=0xC0FFEE
            )

    monkeypatch.setattr(gpu3d, "_require_gpu_modules", fake_modules)
    monkeypatch.setattr(gpu3d, "_embed_ensemble", fake_ensemble)


def _query(pharmacophore: Path) -> tuple[list[str], np.ndarray]:
    points = read_feature_points(pharmacophore)
    return (
        [point.feature_type for point in points],
        np.array([[p.x, p.y, p.z] for p in points], dtype=float),
    )


def test_the_source_molecule_fits_its_own_query(
    fdef: Path, pharmacophore: Path
) -> None:
    """The query was derived from this molecule, so a conformer should fit it."""
    types, coords = _query(pharmacophore)
    mol = Chem.AddHs(Chem.MolFromSmiles("NCCc1ccc(O)cc1"))
    AllChem.EmbedMultipleConfs(mol, numConfs=20, randomSeed=0xC0FFEE)
    best = gpu3d._best_rmsd_for_mol(mol, load_feature_factory(fdef), types, coords)
    assert np.isfinite(best)
    assert best < 1.5


def test_a_molecule_without_the_families_cannot_be_fitted(
    fdef: Path, pharmacophore: Path
) -> None:
    types, coords = _query(pharmacophore)
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMultipleConfs(mol, numConfs=5, randomSeed=0xC0FFEE)
    best = gpu3d._best_rmsd_for_mol(mol, load_feature_factory(fdef), types, coords)
    assert best == float("inf")


def test_more_conformers_cannot_make_the_fit_worse(
    fdef: Path, pharmacophore: Path
) -> None:
    types, coords = _query(pharmacophore)
    factory = load_feature_factory(fdef)

    def best_with(n_confs: int) -> float:
        mol = Chem.AddHs(Chem.MolFromSmiles("NCCc1ccc(OC)cc1"))
        AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, randomSeed=0xC0FFEE)
        return gpu3d._best_rmsd_for_mol(mol, factory, types, coords)

    assert best_with(10) <= best_with(2) + 1e-9


def test_prefilter_decides_rejects_and_defers_survivors(
    stub_device: None, fdef: Path, pharmacophore: Path
) -> None:
    smiles = [
        "NCCc1ccc(O)cc1",      # the molecule the query came from: must survive
        "c1ccccc1",            # no donor or acceptor: must be rejected
        "not-a-real-smiles",   # unparsable: decided immediately
    ]
    decided, rmsd = gpu3d.prefilter_chunk(
        smiles,
        fdef_path=str(fdef),
        pharmacophore_path=str(pharmacophore),
        limit_upper=1.5,
        limit_lower=0.5,
        prefilter_conformers=20,
        prefilter_rmsd=1.5,
        quiet=True,
    )

    # The survivor is left for the constrained pass rather than being decided.
    assert 0 not in decided
    assert np.isfinite(rmsd[0])

    assert decided[1]["status"] == Status.NO_FEATURE
    assert decided[1]["stage"] == Stage.GPU_PREFILTER_REJECT
    assert decided[1]["numMatch"] == 0
    assert decided[1]["match"] is False

    assert decided[2]["status"] == Status.PARSE_ERROR
    assert "parse" in decided[2]["_detail"].lower()
    # An unparsable molecule is never embedded, so it has no fit.
    assert 2 not in rmsd


def test_every_decided_row_carries_the_columns_the_command_needs(
    stub_device: None, fdef: Path, pharmacophore: Path
) -> None:
    """A row the prefilter settles goes straight into the result frame."""
    decided, _rmsd = gpu3d.prefilter_chunk(
        ["c1ccccc1", "broken"],
        fdef_path=str(fdef),
        pharmacophore_path=str(pharmacophore),
        limit_upper=1.5,
        limit_lower=0.5,
        prefilter_conformers=5,
        prefilter_rmsd=1.5,
        quiet=True,
    )
    required = {"match", "numMatch", "status", "elapsed_s", "stage", "best_rmsd"}
    for row in decided.values():
        assert required <= row.keys()


def test_a_generous_threshold_keeps_everything_measurable(
    stub_device: None, fdef: Path, pharmacophore: Path
) -> None:
    smiles = ["NCCc1ccc(O)cc1", "NCCc1ccc(OC)cc1", "OCCNc1ccc(O)cc1"]
    decided, rmsd = gpu3d.prefilter_chunk(
        smiles,
        fdef_path=str(fdef),
        pharmacophore_path=str(pharmacophore),
        limit_upper=1.5,
        limit_lower=0.5,
        prefilter_conformers=10,
        prefilter_rmsd=99.0,
        quiet=True,
    )
    assert decided == {}
    assert len(rmsd) == len(smiles)


def test_a_zero_threshold_rejects_everything(
    stub_device: None, fdef: Path, pharmacophore: Path
) -> None:
    smiles = ["NCCc1ccc(O)cc1", "NCCc1ccc(OC)cc1"]
    decided, _rmsd = gpu3d.prefilter_chunk(
        smiles,
        fdef_path=str(fdef),
        pharmacophore_path=str(pharmacophore),
        limit_upper=1.5,
        limit_lower=0.5,
        prefilter_conformers=5,
        prefilter_rmsd=0.0,
        quiet=True,
    )
    assert set(decided) == {0, 1}


def test_a_failed_ensemble_defers_the_batch_instead_of_dropping_it(
    monkeypatch: pytest.MonkeyPatch, fdef: Path, pharmacophore: Path
) -> None:
    """Without an ensemble nothing can be ruled out, so nothing may be."""

    def fake_modules() -> tuple[Any, Any, Any, Any]:
        return None, None, None, _FakeHardwareOptions

    def exploding_ensemble(mols: list[Any], confs: int, hardware: Any) -> None:
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(gpu3d, "_require_gpu_modules", fake_modules)
    monkeypatch.setattr(gpu3d, "_embed_ensemble", exploding_ensemble)

    decided, rmsd = gpu3d.prefilter_chunk(
        ["NCCc1ccc(O)cc1", "c1ccccc1"],
        fdef_path=str(fdef),
        pharmacophore_path=str(pharmacophore),
        limit_upper=1.5,
        limit_lower=0.5,
        prefilter_conformers=5,
        prefilter_rmsd=1.5,
        quiet=True,
    )
    # Both molecules fall through to the constrained pass.
    assert decided == {}
    assert rmsd == {}


def test_batching_covers_every_molecule(
    stub_device: None, fdef: Path, pharmacophore: Path
) -> None:
    smiles = ["NCCc1ccc(O)cc1", "c1ccccc1", "NCCc1ccc(OC)cc1", "C", "CCO"]
    decided, rmsd = gpu3d.prefilter_chunk(
        smiles,
        fdef_path=str(fdef),
        pharmacophore_path=str(pharmacophore),
        limit_upper=1.5,
        limit_lower=0.5,
        prefilter_conformers=5,
        prefilter_rmsd=1.5,
        quiet=True,
        batch_size=2,
    )
    # Every molecule is either decided here or measured and deferred.
    assert all(index in decided or index in rmsd for index in range(len(smiles)))


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
