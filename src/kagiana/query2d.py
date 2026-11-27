"""2D pharmacophore query setup: params JSON, signature factory, reference FP.

The params JSON is the single source of truth for the signature parameters and
the score cutoffs; no value here has a hard-coded fallback.  Only the cutoffs
and the chunk width can be overridden from the command line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rdkit.Chem import AllChem
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory

from .features import load_feature_factory
from .molecules import mol_from_smiles
from .schema import (
    MIN_POINT_COUNT_2D,
    PARAMS_KEYS_2D,
    SMILES_COLUMN,
    InputError,
    require_smiles_column,
)


@dataclass(frozen=True)
class Params2D:
    """Picklable 2D configuration, safe to hand to worker processes."""

    n_maxpoint: int
    bins: tuple[tuple[float, float], ...]
    skip_feature: tuple[str, ...]
    chunksize: int
    tanimoto_cutoff: float
    simpson_cutoff: float
    dice_cutoff: float

    def with_overrides(
        self,
        tanimoto_cutoff: float | None = None,
        simpson_cutoff: float | None = None,
        dice_cutoff: float | None = None,
        chunksize: int | None = None,
    ) -> Params2D:
        changes: dict[str, Any] = {}
        if tanimoto_cutoff is not None:
            changes["tanimoto_cutoff"] = float(tanimoto_cutoff)
        if simpson_cutoff is not None:
            changes["simpson_cutoff"] = float(simpson_cutoff)
        if dice_cutoff is not None:
            changes["dice_cutoff"] = float(dice_cutoff)
        if chunksize is not None:
            changes["chunksize"] = int(chunksize)
        return replace(self, **changes) if changes else self


def load_params(params_path: str | Path) -> Params2D:
    """Read and validate the params JSON."""
    path = Path(params_path)
    if not path.is_file():
        raise InputError(f"params file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise InputError(f"{path}: expected a JSON object at the top level")

    missing = [key for key in PARAMS_KEYS_2D if key not in data]
    if missing:
        raise InputError(f"{path}: missing required key(s): {', '.join(missing)}")

    try:
        n_maxpoint = int(data["n_maxpoint"])
        chunksize = int(data["chunksize"])
    except (TypeError, ValueError) as exc:
        raise InputError(f"{path}: n_maxpoint and chunksize must be integers: {exc}") from exc

    if n_maxpoint < MIN_POINT_COUNT_2D:
        raise InputError(
            f"{path}: n_maxpoint must be >= {MIN_POINT_COUNT_2D}, got {n_maxpoint}"
        )
    if chunksize < 1:
        raise InputError(f"{path}: chunksize must be >= 1, got {chunksize}")

    raw_bins = data["bins"]
    if not isinstance(raw_bins, (list, tuple)) or not raw_bins:
        raise InputError(f"{path}: bins must be a non-empty list of [lower, upper] pairs")
    bins: list[tuple[float, float]] = []
    for index, entry in enumerate(raw_bins):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise InputError(f"{path}: bins[{index}] must be a [lower, upper] pair")
        try:
            lower, upper = float(entry[0]), float(entry[1])
        except (TypeError, ValueError) as exc:
            raise InputError(f"{path}: bins[{index}] must hold numbers: {exc}") from exc
        if lower >= upper:
            raise InputError(f"{path}: bins[{index}] requires lower < upper, got {entry}")
        bins.append((lower, upper))

    raw_skip = data["skip_feature"]
    if raw_skip is None:
        skip_feature: tuple[str, ...] = ()
    elif isinstance(raw_skip, str):
        skip_feature = (raw_skip,)
    elif isinstance(raw_skip, (list, tuple)):
        skip_feature = tuple(str(item) for item in raw_skip)
    else:
        raise InputError(f"{path}: skip_feature must be a list of feature family names")

    cutoffs: dict[str, float] = {}
    for key in ("tanimoto_cutoff", "simpson_cutoff", "dice_cutoff"):
        try:
            cutoffs[key] = float(data[key])
        except (TypeError, ValueError) as exc:
            raise InputError(f"{path}: {key} must be a number: {exc}") from exc

    return Params2D(
        n_maxpoint=n_maxpoint,
        bins=tuple(bins),
        skip_feature=skip_feature,
        chunksize=chunksize,
        **cutoffs,
    )


def build_sig_factory(fdef_path: str | Path, params: Params2D) -> SigFactory:
    """Build the 2D signature factory used for every fingerprint in a run."""
    factory = load_feature_factory(fdef_path)
    sig_factory = SigFactory(
        factory, minPointCount=MIN_POINT_COUNT_2D, maxPointCount=params.n_maxpoint
    )
    sig_factory.SetBins([tuple(b) for b in params.bins])
    sig_factory.skipFeats = list(params.skip_feature)
    sig_factory.Init()
    return sig_factory


def fingerprint(smiles: str, sig_factory: SigFactory) -> Any:
    """2D pharmacophore fingerprint of a SMILES string, or ``None`` if unparsable."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    return Generate.Gen2DFingerprint(mol, sig_factory)


def read_reference_smiles(query_path: str | Path) -> str:
    """Return the reference SMILES: the first row of the query CSV."""
    import pandas as pd

    path = Path(query_path)
    if not path.is_file():
        raise InputError(f"query file not found: {path}")
    frame = pd.read_csv(path)
    require_smiles_column(frame.columns, str(path))
    values = frame[SMILES_COLUMN].values.tolist()
    if not values:
        raise InputError(f"{path}: query CSV holds no rows")
    smiles = str(values[0])
    if not smiles or smiles.lower() == "nan":
        raise InputError(f"{path}: the first {SMILES_COLUMN} value is empty")
    return smiles


def reference_fingerprint(smiles: str, sig_factory: SigFactory) -> Any:
    """Fingerprint of the reference molecule, as an input error when it fails.

    Generating a signature can fail on a molecule RDKit parses without
    complaint, so this is reported the same way an unreadable file is: the
    reference molecule is input, and the run cannot start without it.
    """
    try:
        reference = fingerprint(smiles, sig_factory)
    except InputError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise InputError(
            f"no fingerprint could be generated for the reference molecule "
            f"{smiles!r}: {type(exc).__name__}: {exc}"
        ) from exc
    if reference is None:
        raise InputError(f"reference SMILES could not be parsed: {smiles!r}")
    return reference
