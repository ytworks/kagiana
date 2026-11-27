"""Geometric fitting of conformer features onto a 3D pharmacophore.

Used by the GPU prefilter: once a conformer ensemble exists, how well a
molecule can present the required features is measured by superposing its
candidate feature points onto the query points and taking the RMSD of the best
assignment.  This is a cheap, rigid-body screen, not a substitute for
constrained embedding; it exists to decide which molecules deserve the
expensive pass.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable, Sequence

import numpy as np

#: Ceiling on assignments tried per conformer, so that a molecule with many
#: candidate features per query point cannot dominate the run.
DEFAULT_COMBINATION_BUDGET = 2000


def kabsch_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    """RMSD between two point sets after optimal rigid-body superposition.

    Both arrays are ``(n, 3)`` and paired row by row.  Translation is removed
    by centring and rotation by the Kabsch construction, with the reflection
    case corrected so the result is a proper rotation.
    """
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("both point sets must have the same (n, 3) shape")

    mobile_c = mobile - mobile.mean(axis=0)
    target_c = target - target.mean(axis=0)

    covariance = mobile_c.T @ target_c
    u, _s, vt = np.linalg.svd(covariance)
    # A negative determinant would mean a reflection rather than a rotation.
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, sign if sign != 0 else 1.0])
    rotation = vt.T @ correction @ u.T

    rotated = (rotation @ mobile_c.T).T
    deviation = rotated - target_c
    return float(np.sqrt((deviation**2).sum() / len(mobile)))


def _limit_candidates(
    candidates: Sequence[Sequence[np.ndarray]], budget: int
) -> list[list[np.ndarray]]:
    """Trim per-point candidate lists until the assignment count fits a budget."""
    trimmed = [list(group) for group in candidates]
    while trimmed:
        total = 1
        for group in trimmed:
            total *= max(1, len(group))
        if total <= budget:
            return trimmed
        widest = max(range(len(trimmed)), key=lambda i: len(trimmed[i]))
        if len(trimmed[widest]) <= 1:
            return trimmed
        trimmed[widest] = trimmed[widest][:-1]
    return trimmed


def best_assignment_rmsd(
    query_points: np.ndarray,
    candidates: Sequence[Sequence[np.ndarray]],
    budget: int = DEFAULT_COMBINATION_BUDGET,
) -> float:
    """Smallest RMSD over all ways of assigning candidate points to query points.

    ``candidates[i]`` holds the positions in the conformer that could play the
    role of query point ``i`` (its feature family).  A point with no candidate
    makes the molecule unfittable, reported as infinity.
    """
    if len(candidates) != len(query_points):
        raise ValueError("one candidate group is required per query point")
    if any(len(group) == 0 for group in candidates):
        return float("inf")

    best = float("inf")
    for combination in product(*_limit_candidates(candidates, budget)):
        # Two query points must not be served by the same position.
        stacked = np.vstack(combination)
        if len(np.unique(stacked, axis=0)) < len(stacked):
            continue
        best = min(best, kabsch_rmsd(stacked, query_points))
    return best


def group_positions_by_family(
    features: Iterable[object],
) -> dict[str, list[np.ndarray]]:
    """Index conformer feature positions by feature family."""
    grouped: dict[str, list[np.ndarray]] = {}
    for feature in features:
        position = feature.GetPos()  # type: ignore[attr-defined]
        grouped.setdefault(feature.GetFamily(), []).append(  # type: ignore[attr-defined]
            np.array([position.x, position.y, position.z], dtype=float)
        )
    return grouped
