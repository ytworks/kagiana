"""Geometric fitting of conformer features onto a 3D pharmacophore.

Used by the GPU prefilter: once a conformer ensemble exists, how well a
molecule can present the required features is measured by superposing its
candidate feature points onto the query points and taking the RMSD of the best
assignment.  This is a cheap, rigid-body screen, not a substitute for
constrained embedding; it exists to decide which molecules deserve the
expensive pass.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

#: Ceiling on assignments tried per conformer, so that a molecule with many
#: candidate features per query point cannot dominate the run.
DEFAULT_COMBINATION_BUDGET = 2000

#: Ceiling on partial assignments examined per conformer.  Completed
#: assignments are not the only cost: a query whose points cannot all be
#: satisfied at once produces none of them while still walking an enormous
#: tree, and that walk happens in the parent process where the per-molecule
#: deadline does not reach it.
DEFAULT_NODE_BUDGET = 50_000


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


#: Counters from the most recent search, read by the caller to tell a complete
#: search from one that ran out of budget.
_LAST_SEARCH: dict[str, int] = {"produced": 0, "visited": 0}


def _position_key(position: np.ndarray) -> tuple[float, float, float]:
    return (float(position[0]), float(position[1]), float(position[2]))


def _too_few_positions(candidates: Sequence[Sequence[np.ndarray]]) -> bool:
    """Whether no assignment can exist because positions would have to repeat.

    Each query point needs a position of its own, so a query with more points
    than there are distinct positions on offer has no valid assignment at all.
    Settling that by counting avoids searching a tree that cannot contain an
    answer, which is where an unsatisfiable query would otherwise spend its
    time.
    """
    distinct: set[tuple[float, float, float]] = set()
    for group in candidates:
        for position in group:
            distinct.add(_position_key(position))
            if len(distinct) >= len(candidates):
                return False
    return len(distinct) < len(candidates)


def _distinct_assignments(
    candidates: Sequence[Sequence[np.ndarray]],
    budget: int,
    node_budget: int,
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
    visited = 0
    chosen: list[np.ndarray] = []
    used: set[tuple[float, float, float]] = set()

    def walk(point: int) -> Iterator[list[np.ndarray]]:
        nonlocal produced, visited
        if produced >= budget or visited >= node_budget:
            return
        if point == n_points:
            produced += 1
            yield list(chosen)
            return
        for position in candidates[point]:
            key = _position_key(position)
            if key in used:
                continue
            visited += 1
            if visited >= node_budget:
                return
            chosen.append(position)
            used.add(key)
            yield from walk(point + 1)
            chosen.pop()
            used.discard(key)
            if produced >= budget:
                return

    yield from walk(0)
    # Report exhaustion through the counters the caller can see.
    _LAST_SEARCH["produced"] = produced
    _LAST_SEARCH["visited"] = visited


def best_assignment_rmsd(
    query_points: np.ndarray,
    candidates: Sequence[Sequence[np.ndarray]],
    budget: int = DEFAULT_COMBINATION_BUDGET,
    node_budget: int = DEFAULT_NODE_BUDGET,
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
    if _too_few_positions(candidates):
        # Settled without searching: the points cannot all be given a position
        # of their own.
        return float("inf"), True
    if budget < 1 or node_budget < 1:
        return float("inf"), False

    _LAST_SEARCH["produced"] = 0
    _LAST_SEARCH["visited"] = 0
    best = float("inf")
    count = 0
    for assignment in _distinct_assignments(candidates, budget, node_budget):
        count += 1
        best = min(best, kabsch_rmsd(np.vstack(assignment), query_points))

    # A search that hit either ceiling has not ruled anything out.
    exhausted = count >= budget or _LAST_SEARCH["visited"] >= node_budget
    return best, not exhausted
