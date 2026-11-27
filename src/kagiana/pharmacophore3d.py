"""3D pharmacophore queries built from a feature-point CSV.

The CSV holds one feature point per row with the columns ``feature_type``,
``X``, ``Y``, ``Z``.  Pairwise distance bounds are derived from the reference
coordinates: for every pair the lower bound is ``max(0, d - limit_lower)`` and
the upper bound is ``d + limit_upper``.  All feature points must be satisfied
for a molecule to match; there is no partial-match mode.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Geometry
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm3D import Pharmacophore

from .features import feature_families
from .schema import FEATURE_COORD_COLUMNS, FEATURE_TYPE_COLUMN, InputError

DEFAULT_LIMIT_UPPER = 1.5
DEFAULT_LIMIT_LOWER = 0.5


@dataclass(frozen=True)
class FeaturePoint:
    feature_type: str
    x: float
    y: float
    z: float


def read_feature_points(pharmacophore_path: str | Path) -> list[FeaturePoint]:
    """Read and validate the feature-point CSV."""
    path = Path(pharmacophore_path)
    if not path.is_file():
        raise InputError(f"pharmacophore file not found: {path}")

    points: list[FeaturePoint] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        required = (FEATURE_TYPE_COLUMN, *FEATURE_COORD_COLUMNS)
        missing = [name for name in required if name not in header]
        if missing:
            raise InputError(
                f"{path}: pharmacophore CSV is missing column(s) "
                f"{', '.join(missing)} (columns: {', '.join(header)})"
            )
        for lineno, row in enumerate(reader, start=2):
            feature_type = (row.get(FEATURE_TYPE_COLUMN) or "").strip()
            if not feature_type:
                raise InputError(f"{path}:{lineno}: empty {FEATURE_TYPE_COLUMN}")
            coords = []
            for name in FEATURE_COORD_COLUMNS:
                raw = row.get(name)
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    raise InputError(
                        f"{path}:{lineno}: column {name} is not a number: {raw!r}"
                    ) from None
                if not math.isfinite(value):
                    raise InputError(
                        f"{path}:{lineno}: column {name} must be a finite "
                        f"coordinate, got {raw!r}"
                    )
                coords.append(value)
            points.append(FeaturePoint(feature_type, *coords))

    if len(points) < 2:
        raise InputError(
            f"{path}: at least 2 feature points are required, found {len(points)}"
        )
    return points


def require_known_feature_types(
    points: list[FeaturePoint], fdef_path: str | Path
) -> None:
    """Fail when a query asks for a feature family the fdef does not define.

    A misspelt family matches nothing, which would otherwise look exactly like
    a library in which no molecule carries the required features: a clean run
    reporting zero hits.  Both the pre-flight check and the screening commands
    go through here so that such a query is an input error either way.
    """
    known = set(feature_families(fdef_path))
    unknown = sorted({point.feature_type for point in points} - known)
    if unknown:
        raise InputError(
            "feature type(s) not defined by the fdef file "
            f"{fdef_path}: {', '.join(unknown)} "
            f"(defined: {', '.join(sorted(known))})"
        )


def make_pharmacophore(
    pharmacophore_path: str | Path,
    limit_upper: float = DEFAULT_LIMIT_UPPER,
    limit_lower: float = DEFAULT_LIMIT_LOWER,
) -> Any:
    """Build an RDKit pharmacophore with distance bounds from a feature CSV."""
    if limit_upper < 0 or limit_lower < 0:
        raise InputError("--limit-upper and --limit-lower must be non-negative")

    points = read_feature_points(pharmacophore_path)
    features = [
        ChemicalFeatures.FreeChemicalFeature(
            point.feature_type, Geometry.Point3D(point.x, point.y, point.z)
        )
        for point in points
    ]

    pharmacophore = Pharmacophore.Pharmacophore(features)
    for i in range(len(features) - 1):
        pos_a = features[i].GetPos()
        for j in range(i + 1, len(features)):
            pos_b = features[j].GetPos()
            distance = Geometry.Point3D.Distance(pos_a, pos_b)
            pharmacophore.setLowerBound(i, j, max(0.0, distance - limit_lower))
            pharmacophore.setUpperBound(i, j, distance + limit_upper)
    return pharmacophore


def reference_coordinates(points: list[FeaturePoint]) -> list[tuple[float, float, float]]:
    """Feature-point coordinates, used by the GPU prefilter's geometric fit."""
    return [(p.x, p.y, p.z) for p in points]
