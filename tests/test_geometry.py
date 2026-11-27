"""Tests for the geometric fitting helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from kagiana.geometry import best_assignment_rmsd, kabsch_rmsd


def _triangle() -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])


def test_identical_point_sets_fit_exactly() -> None:
    points = _triangle()
    assert kabsch_rmsd(points, points.copy()) == pytest.approx(0.0, abs=1e-9)


def test_rigid_motion_is_removed() -> None:
    points = _triangle()
    angle = 0.7
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = (rotation @ points.T).T + np.array([10.0, -5.0, 2.0])
    assert kabsch_rmsd(moved, points) == pytest.approx(0.0, abs=1e-9)


def test_reflection_is_not_treated_as_a_fit() -> None:
    # A mirrored, non-planar set cannot be superposed by rotation alone.
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mirrored = points * np.array([1.0, 1.0, -1.0])
    assert kabsch_rmsd(mirrored, points) > 0.1


def test_distortion_raises_the_rmsd() -> None:
    points = _triangle()
    distorted = points + np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert kabsch_rmsd(distorted, points) > 0.1


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError):
        kabsch_rmsd(np.zeros((3, 3)), np.zeros((4, 3)))


def test_best_assignment_picks_the_right_pairing() -> None:
    query = _triangle()
    # The candidates are the query points themselves, offered in a scrambled
    # order per position; the best assignment must recover a perfect fit.
    candidates = [
        [query[1], query[0]],
        [query[2], query[1]],
        [query[0], query[2]],
    ]
    best, exhaustive = best_assignment_rmsd(query, candidates)
    assert best == pytest.approx(0.0, abs=1e-9)
    assert exhaustive is True


def test_missing_candidate_is_conclusive() -> None:
    query = _triangle()
    best, exhaustive = best_assignment_rmsd(query, [[query[0]], [], [query[2]]])
    assert best == float("inf")
    # Nothing can be assigned, so there is nothing left to search.
    assert exhaustive is True


def test_a_repeated_family_keeps_its_valid_assignments() -> None:
    """A query naming one family several times must still find an exact fit.

    Trimming candidate lists to fit a cartesian product would discard every
    valid assignment here: six points of one family have 46,656 combinations
    but only 720 that use each position once.
    """
    query = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0],
         [1.0, 1.0, 5.0], [2.0, -3.0, 1.0], [-4.0, 1.0, 2.0]]
    )
    candidates = [[query[i] for i in range(len(query))] for _ in range(len(query))]
    best, exhaustive = best_assignment_rmsd(query, candidates)
    assert best == pytest.approx(0.0, abs=1e-9)
    assert exhaustive is True


def test_a_position_is_never_used_twice() -> None:
    query = _triangle()
    # Only two distinct positions are on offer for three query points.
    candidates = [[query[0], query[1]], [query[0], query[1]], [query[0], query[1]]]
    best, exhaustive = best_assignment_rmsd(query, candidates)
    assert best == float("inf")
    assert exhaustive is True


def test_an_exhausted_budget_is_reported_as_inconclusive() -> None:
    """A search cut short has ruled nothing out, and must say so."""
    query = _triangle()
    wide = [[np.array([float(i), float(j), 0.0]) for i in range(30)] for j in range(3)]
    best, exhaustive = best_assignment_rmsd(query, wide, budget=50)
    assert math.isfinite(best)
    assert exhaustive is False
