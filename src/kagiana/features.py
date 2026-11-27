"""Chemical-feature factory loading, cached per process.

Building a feature factory from an fdef file is expensive relative to scoring a
single molecule, and the resulting object does not survive pickling well, so
worker processes build it once from a path and keep it in this cache.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from rdkit.Chem import ChemicalFeatures

from .schema import InputError


@lru_cache(maxsize=4)
def _build(path_str: str) -> Any:
    return ChemicalFeatures.BuildFeatureFactory(path_str)


def load_feature_factory(fdef_path: str | Path) -> Any:
    """Return a feature factory for ``fdef_path``, reusing a cached instance."""
    path = Path(fdef_path)
    if not path.is_file():
        raise InputError(f"fdef file not found: {path}")
    try:
        return _build(str(path))
    except InputError:
        raise
    except Exception as exc:  # malformed fdef
        raise InputError(f"failed to read fdef file {path}: {exc}") from exc


def feature_families(fdef_path: str | Path) -> list[str]:
    """List the feature families defined by an fdef file."""
    factory = load_feature_factory(fdef_path)
    return list(factory.GetFeatureFamilies())
