"""2D pharmacophore screening on the CPU.

Each library molecule is turned into a 2D pharmacophore fingerprint and
compared with the reference molecule's fingerprint.  Three similarity scores
are derived from a single intersection count so that they always agree with one
another.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rdkit import RDLogger

from ..query2d import Params2D, build_sig_factory, fingerprint, reference_fingerprint


@dataclass(frozen=True)
class Score2D:
    tanimoto: float
    dice: float
    simpson: float


def similarity_scores(reference_fp: Any, query_fp: Any) -> Score2D:
    """Tanimoto, Dice and Simpson scores between two fingerprints.

    All three come from the same on-bit counts: ``a`` and ``b`` are the number
    of set bits on each side and ``i`` their intersection.  Zero denominators
    (a molecule with no pharmacophore features at all) score 0 rather than
    raising.
    """
    a = reference_fp.GetNumOnBits()
    b = query_fp.GetNumOnBits()
    i = (reference_fp & query_fp).GetNumOnBits()

    union = a + b - i
    total = a + b
    smaller = min(a, b)
    return Score2D(
        tanimoto=i / union if union else 0.0,
        dice=2 * i / total if total else 0.0,
        simpson=i / smaller if smaller else 0.0,
    )


def score_smiles(smiles: str, sig_factory: Any, reference_fp: Any) -> Score2D | None:
    """Score one SMILES string, or ``None`` when RDKit cannot parse it."""
    query_fp = fingerprint(smiles, sig_factory)
    if query_fp is None:
        return None
    return similarity_scores(reference_fp, query_fp)


def init_worker(
    fdef_path: str, params: Params2D, reference_smiles: str
) -> Callable[[str], Score2D | None]:
    """Build the per-worker state and return the task callable."""
    # Unparsable SMILES are expected in a screening library and are reported
    # per row, so RDKit's own complaints would only be noise on stderr.
    RDLogger.DisableLog("rdApp.*")
    sig_factory = build_sig_factory(fdef_path, params)
    reference_fp = reference_fingerprint(reference_smiles, sig_factory)

    def task(smiles: str) -> Score2D | None:
        return score_smiles(smiles, sig_factory, reference_fp)

    return task
