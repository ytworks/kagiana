"""3D screening with a GPU conformer prefilter in front of the CPU pass.

Constrained embedding — generating conformers that are forced to satisfy the
pharmacophore — cannot be moved to the GPU: the GPU embedder takes no bounds
matrix and no coordinate map, which is exactly how the constraint is expressed.
What the GPU does well is produce a large unconstrained conformer ensemble
quickly.

So this backend screens in two stages.  First the GPU builds an ensemble per
molecule and the best rigid superposition of its features onto the query is
measured; molecules that cannot come close are rejected outright.  The
survivors then go through the same constrained-embedding pass the CPU backend
uses, so the reported scores keep their meaning.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..features import load_feature_factory
from ..geometry import best_assignment_rmsd, group_positions_by_family
from ..pharmacophore3d import read_feature_points
from ..progress import note, progress_bar
from ..schema import InputError, Stage, Status

NAN = float("nan")

#: Molecules handed to the GPU embedder at a time.
DEFAULT_BATCH_SIZE = 2000


def _require_gpu_modules() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from nvmolkit.embedMolecules import EmbedMolecules
        from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
        from nvmolkit.types import HardwareOptions
    except Exception as exc:  # noqa: BLE001
        raise InputError(
            "the GPU 3D backend needs torch and nvmolkit with a CUDA device: " f"{exc}"
        ) from exc
    return torch, EmbedMolecules, MMFFOptimizeMoleculesConfs, HardwareOptions


def _embed_ensemble(
    mols: list[Any],
    confs_per_molecule: int,
    hardware: Any | None,
) -> None:
    """Generate and relax a conformer ensemble for a batch, in place."""
    from rdkit.Chem.rdDistGeom import ETKDGv3

    _torch, embed_molecules, optimize_confs, _hw = _require_gpu_modules()
    params = ETKDGv3()
    embed_molecules(
        mols, params, confsPerMolecule=confs_per_molecule, hardwareOptions=hardware
    )
    optimize_confs(mols, hardwareOptions=hardware)


def _best_rmsd_for_mol(
    mol: Any,
    feature_factory: Any,
    query_types: list[str],
    query_coords: np.ndarray,
) -> float:
    """Best fit of any conformer of a molecule onto the query points."""
    best = float("inf")
    for conformer in mol.GetConformers():
        features = feature_factory.GetFeaturesForMol(mol, confId=conformer.GetId())
        grouped = group_positions_by_family(features)
        candidates = [grouped.get(family, []) for family in query_types]
        if any(len(group) == 0 for group in candidates):
            # This molecule cannot present the required families at all.
            return float("inf")
        best = min(best, best_assignment_rmsd(query_coords, candidates))
    return best


def prefilter_chunk(
    smiles: list[str],
    fdef_path: str,
    pharmacophore_path: str,
    limit_upper: float,
    limit_lower: float,
    prefilter_conformers: int,
    prefilter_rmsd: float,
    quiet: bool,
    gpu_ids: list[int] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[dict[int, dict[str, Any]], dict[int, float]]:
    """Decide which molecules are worth the constrained-embedding pass.

    Returns the rows already settled (unparsable molecules and rejections)
    keyed by position, plus the best fit RMSD of every molecule that was
    measured, so a surviving row can report it too.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    _torch, _embed, _optimize, hardware_options = _require_gpu_modules()

    feature_factory = load_feature_factory(fdef_path)
    points = read_feature_points(pharmacophore_path)
    query_types = [point.feature_type for point in points]
    query_coords = np.array([[p.x, p.y, p.z] for p in points], dtype=float)

    hardware = hardware_options(gpuIds=gpu_ids or [])

    decided: dict[int, dict[str, Any]] = {}
    rmsd_by_position: dict[int, float] = {}

    positions: list[int] = []
    mols: list[Any] = []
    for position, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            decided[position] = {
                "match": False,
                "numMatch": 0,
                "status": Status.PARSE_ERROR,
                "elapsed_s": 0.0,
                "stage": Stage.GPU_PREFILTER_REJECT,
                "best_rmsd": NAN,
                "_detail": "RDKit could not parse the SMILES string",
            }
            continue
        positions.append(position)
        mols.append(Chem.AddHs(mol))

    with progress_bar(len(mols), "pharm3d gpu prefilter", quiet) as bar:
        for start in range(0, len(mols), batch_size):
            batch = mols[start : start + batch_size]
            batch_positions = positions[start : start + batch_size]
            try:
                _embed_ensemble(batch, prefilter_conformers, hardware)
            except Exception as exc:  # noqa: BLE001
                # Without an ensemble nothing can be ruled out, so the whole
                # batch goes to the constrained pass rather than being dropped.
                note(
                    f"gpu 3D: ensemble generation failed for a batch of "
                    f"{len(batch)} molecules, screening them on the CPU instead "
                    f"({type(exc).__name__}: {exc})",
                    quiet,
                )
                bar.update(len(batch))
                continue

            for mol, position in zip(batch, batch_positions):
                best = _best_rmsd_for_mol(
                    mol, feature_factory, query_types, query_coords
                )
                rmsd_by_position[position] = best
                if not np.isfinite(best) or best > prefilter_rmsd:
                    decided[position] = {
                        "match": False,
                        "numMatch": 0,
                        "status": Status.NO_FEATURE,
                        "elapsed_s": 0.0,
                        "stage": Stage.GPU_PREFILTER_REJECT,
                        "best_rmsd": best if np.isfinite(best) else NAN,
                        "_detail": "",
                    }
                bar.update(1)

    kept = len(smiles) - len(decided)
    note(
        f"gpu 3D prefilter: {kept} of {len(smiles)} molecules kept for "
        f"constrained embedding (RMSD <= {prefilter_rmsd:g})",
        quiet,
    )
    return decided, rmsd_by_position
