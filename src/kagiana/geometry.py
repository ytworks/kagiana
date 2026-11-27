"""Geometric fitting of conformer features onto a 3D pharmacophore.

Used by the GPU prefilter: once a conformer ensemble exists, how well a
molecule can present the required features is measured by superposing its
candidate feature points onto the query points and taking the RMSD of the best
assignment.  This is a cheap, rigid-body screen, not a substitute for
constrained embedding; it exists to decide which molecules deserve the
expensive pass.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

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


def _distinct_assignments(
    candidates: Sequence[Sequence[np.ndarray]], budget: int
) -> Iterator[list[np.ndarray]]:
    """Enumerate assignments that give each query point a distinct position.

    A query may name the same feature family more than once, in which case the
    same candidate positions are offered for several points.  Only assignments
    that use a position once are chemically meaningful, so they are built by
    backtracking over the positions still free rather than by filtering a full
    cartesian product afterwards: with six points of one family there are 720
    valid assignments among 46,656 combinations, and trimming the candidate
    lists to fit the product would throw away every valid one.

    The budget counts assignments actually produced.  Stopping early is
    reported by the caller rather than silently returning the best of a partial
    search, because a search that could not finish has not ruled anything out.
    """
    n_points = len(candidates)
    produced = 0
    chosen: list[np.ndarray] = []
    used: list[tuple[float, float, float]] = []

    def walk(point: int) -> Iterator[list[np.ndarray]]:
        nonlocal produced
        if produced >= budget:
            return
        if point == n_points:
            produced += 1
            yield list(chosen)
            return
        for position in candidates[point]:
            key = (float(position[0]), float(position[1]), float(position[2]))
            if key in used:
                continue
            chosen.append(position)
            used.append(key)
            yield from walk(point + 1)
            chosen.pop()
            used.pop()
            if produced >= budget:
                return

    yield from walk(0)


def best_assignment_rmsd(
    query_points: np.ndarray,
    candidates: Sequence[Sequence[np.ndarray]],
    budget: int = DEFAULT_COMBINATION_BUDGET,
) -> tuple[float, bool]:
    """Best RMSD over the assignments of candidate points to query points.

    ``candidates[i]`` holds the positions in the conformer that could play the
    role of query point ``i`` (its feature family).  Returns the smallest RMSD
    found and whether the search was exhaustive.  A non-exhaustive search means
    the budget ran out: the answer is an upper bound, and the caller must not
    reject a molecule on it.  A query point with no candidate at all is
    conclusive, and reports infinity with the search complete.
    """
    if len(candidates) != len(query_points):
        raise ValueError("one candidate group is required per query point")
    if any(len(group) == 0 for group in candidates):
        return float("inf"), True
    if budget < 1:
        return float("inf"), False

    best = float("inf")
    count = 0
    for assignment in _distinct_assignments(candidates, budget):
        count += 1
        best = min(best, kabsch_rmsd(np.vstack(assignment), query_points))

    # The search is exhaustive unless it was cut off at the budget.
    return best, count < budget


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
