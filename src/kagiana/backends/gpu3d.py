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
from ..molecules import mol_from_smiles
from ..pharmacophore3d import make_pharmacophore, read_feature_points
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
) -> tuple[float, bool]:
    """Best fit of any conformer of a molecule onto the query points.

    Returns the smallest RMSD found and whether every conformer was searched
    exhaustively.  A molecule with no conformers, or one whose conformers never
    present all the required families, comes back as infinity; the caller sends
    those to the constrained pass rather than deciding them here.
    """
    best = float("inf")
    exhaustive = True
    measured = False
    for conformer in mol.GetConformers():
        features = feature_factory.GetFeaturesForMol(mol, confId=conformer.GetId())
        grouped = group_positions_by_family(features)
        candidates = [grouped.get(family, []) for family in query_types]
        if any(len(group) == 0 for group in candidates):
            # This conformer cannot present the required families.
            continue
        rmsd, complete = best_assignment_rmsd(query_coords, candidates)
        exhaustive = exhaustive and complete
        measured = True
        best = min(best, rmsd)
    if not measured:
        return float("inf"), True
    return best, exhaustive


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

    from rdkit.Chem.Pharm3D import EmbedLib

    feature_factory = load_feature_factory(fdef_path)
    points = read_feature_points(pharmacophore_path)
    # The same query object the constrained pass builds, so that the match
    # column can be settled with the constrained pass's own predicate.
    pharmacophore = make_pharmacophore(pharmacophore_path, limit_upper, limit_lower)
    query_types = [point.feature_type for point in points]
    query_coords = np.array([[p.x, p.y, p.z] for p in points], dtype=float)

    hardware = hardware_options(gpuIds=gpu_ids or [])

    decided: dict[int, dict[str, Any]] = {}
    rmsd_by_position: dict[int, float] = {}

    # Unparsable rows need no conformers, so they are settled before anything
    # is embedded; the rest are kept as SMILES and turned into molecules one
    # batch at a time.
    pending: list[tuple[int, str]] = []
    for position, smi in enumerate(smiles):
        if mol_from_smiles(smi) is None:
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
        pending.append((position, smi))

    with progress_bar(len(pending), "pharm3d gpu prefilter", quiet) as bar:
        for start_index in range(0, len(pending), batch_size):
            slice_ = pending[start_index : start_index + batch_size]
            # Molecules are built per batch and dropped at the end of it: a
            # conformer ensemble is large, and holding the whole chunk's
            # ensembles would grow host memory with the chunk width.
            batch_positions = [position for position, _ in slice_]
            # Two representations of each molecule, and the difference matters.
            # Embedding and feature geometry need explicit hydrogens, but a
            # feature definition can be sensitive to them (a carbon is D1
            # without them and D4 with them), so whether a molecule carries the
            # required features is decided on the same representation the
            # constrained pass uses.
            base = [mol_from_smiles(smi) for _, smi in slice_]
            batch = [Chem.AddHs(mol) for mol in base]
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
                batch.clear()
                continue

            for mol, bare, position in zip(batch, base, batch_positions):
                best, exhaustive = _best_rmsd_for_mol(
                    mol, feature_factory, query_types, query_coords
                )
                if np.isfinite(best):
                    rmsd_by_position[position] = best
                # Only a measured fit that is exhaustive and clearly too far
                # off is settled here.  A molecule whose ensemble produced no
                # usable conformer, or whose search hit the assignment budget,
                # has not been ruled out, so it goes to the constrained pass
                # instead of being recorded as a miss.
                if exhaustive and np.isfinite(best) and best > prefilter_rmsd:
                    # Ask the constrained pass's own question, on its own
                    # molecule: does this carry the required features at all?
                    # Anything else would disagree with a CPU run of the same
                    # library for feature definitions that count hydrogens.
                    try:
                        carries_features = bool(
                            EmbedLib.MatchPharmacophoreToMol(
                                bare, feature_factory, pharmacophore
                            )[0]
                        )
                    except Exception:  # noqa: BLE001
                        # Undecidable here; let the constrained pass answer.
                        bar.update(1)
                        continue
                    decided[position] = {
                        # The geometry is what failed, and the stage column is
                        # what records that.
                        "match": carries_features,
                        "numMatch": 0,
                        "status": Status.OK if carries_features else Status.NO_FEATURE,
                        "elapsed_s": 0.0,
                        "stage": Stage.GPU_PREFILTER_REJECT,
                        "best_rmsd": best,
                        "_detail": "",
                    }
                bar.update(1)

            batch.clear()

    kept = len(smiles) - len(decided)
    note(
        f"gpu 3D prefilter: {kept} of {len(smiles)} molecules kept for "
        f"constrained embedding (RMSD <= {prefilter_rmsd:g})",
        quiet,
    )
    return decided, rmsd_by_position
