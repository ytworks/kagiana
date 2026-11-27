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
    assert best_assignment_rmsd(query, candidates) == pytest.approx(0.0, abs=1e-9)


def test_missing_candidate_makes_a_molecule_unfittable() -> None:
    query = _triangle()
    assert best_assignment_rmsd(query, [[query[0]], [], [query[2]]]) == float("inf")


def test_budget_limits_the_search_without_failing() -> None:
    query = _triangle()
    wide = [[np.array([float(i), 0.0, 0.0]) for i in range(30)] for _ in range(3)]
    result = best_assignment_rmsd(query, wide, budget=50)
    assert math.isfinite(result)
