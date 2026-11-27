"""Shared fixtures: paths to the small CSV / fdef inputs used by the tests."""

from __future__ import annotations

from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture
def library() -> Path:
    return DATA / "library.csv"


@pytest.fixture
def query() -> Path:
    return DATA / "query.csv"


@pytest.fixture
def fdef() -> Path:
    return DATA / "test.fdef"


@pytest.fixture
def params() -> Path:
    return DATA / "params.json"


@pytest.fixture
def pharmacophore() -> Path:
    return DATA / "pharmacophore.csv"
