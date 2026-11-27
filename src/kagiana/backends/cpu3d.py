"""3D pharmacophore screening on the CPU.

For each molecule: check that it carries the required features at all, derive
its distance-bounds matrix, enumerate the feature-to-atom assignments whose
distances fit the query, and try to embed a conformer under each assignment.
The score is the number of poses successfully embedded.

Pose generation is capped, and progress is reported to the parent process
before every assignment so that a molecule killed on its deadline still
accounts for the work already done plus the assignment that was interrupted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rdkit import RDLogger
from rdkit.Chem import AllChem, rdDistGeom
from rdkit.Chem.Pharm3D import EmbedLib

from ..features import load_feature_factory
from ..molecules import mol_from_smiles
from ..pharmacophore3d import make_pharmacophore
from ..pool import report_progress


@dataclass(frozen=True)
class Score3D:
    match: bool
    num_match: int


@dataclass(frozen=True)
class Partial3D:
    """What a molecule had achieved when its deadline expired."""

    match: bool
    poses: int
    in_assignment: bool

    def num_match(self) -> int:
        # The interrupted assignment is credited with a single pose, matching
        # how a stalled embedding is accounted for when it is abandoned.
        return self.poses + (1 if self.in_assignment else 0)


def screen_molecule(
    smiles: str,
    feature_factory: Any,
    pharmacophore: Any,
    n_conformers: int,
    max_generate_poses: int,
) -> Score3D | None:
    """Screen one SMILES string, or return ``None`` if RDKit cannot parse it."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)

    # Does the molecule carry the required feature types in the first place?
    match, feature_matches = EmbedLib.MatchPharmacophoreToMol(
        mol, feature_factory, pharmacophore
    )
    report_progress(Partial3D(bool(match), 0, False))
    if not match:
        return Score3D(False, 0)

    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    assignments = EmbedLib.GetAllPharmacophoreMatches(
        feature_matches, bounds, pharmacophore
    )

    poses = 0
    for assignment in assignments:
        remaining = max_generate_poses - poses
        if remaining <= 0:
            # The ceiling has been reached exactly; generating more would
            # report a count the caller did not ask for.
            break
        atom_ids = [feature.GetAtomIds() for feature in assignment]
        report_progress(Partial3D(True, poses, True))
        try:
            _bounds, embeds, _n_fail = EmbedLib.EmbedPharmacophore(
                mol,
                atom_ids,
                pharmacophore,
                # Never ask for more poses than the remaining budget, so that
                # an assignment which embeds only part of what it was asked
                # for still cannot push the total over the ceiling.
                count=min(n_conformers, remaining),
                silent=1,
            )
        except Exception:  # noqa: BLE001 - a single assignment may be unembeddable
            continue
        poses += len(embeds)
        report_progress(Partial3D(True, poses, False))

    return Score3D(True, poses)


def init_worker(
    fdef_path: str,
    pharmacophore_path: str,
    limit_upper: float,
    limit_lower: float,
    n_conformers: int,
    max_poses: int,
) -> Callable[[str], Score3D | None]:
    """Build the per-worker state and return the task callable."""
    RDLogger.DisableLog("rdApp.*")
    feature_factory = load_feature_factory(fdef_path)
    pharmacophore = make_pharmacophore(pharmacophore_path, limit_upper, limit_lower)
    # The cap scales with the per-assignment conformer count, so that asking for
    # more conformers does not silently tighten the ceiling.
    max_generate_poses = max_poses * n_conformers

    def task(smiles: str) -> Score3D | None:
        return screen_molecule(
            smiles, feature_factory, pharmacophore, n_conformers, max_generate_poses
        )

    return task
