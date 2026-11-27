"""Column contracts, per-row status vocabulary and the machine-readable run summary.

The CSV contract is deliberately rigid: a library file must carry a ``SMILES``
column, every other input column is passed through untouched, and the score
columns are appended in a fixed order.  Agents can therefore rely on the shape
of the output without inspecting the tool's configuration.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

SMILES_COLUMN = "SMILES"

# Pharmacophore feature-point CSV (3D queries).
FEATURE_TYPE_COLUMN = "feature_type"
FEATURE_COORD_COLUMNS = ("X", "Y", "Z")

# Score columns appended by each screening mode.
SCORE_COLUMNS_2D = ("tanimoto", "dice", "simpson")
SCORE_COLUMNS_3D = ("match", "numMatch")

# Diagnostic columns appended after the score columns.
DIAGNOSTIC_COLUMNS_2D = ("status", "elapsed_s")
DIAGNOSTIC_COLUMNS_3D = ("status", "elapsed_s", "stage", "best_rmsd")

# Required keys of the 2D params JSON.  There are no hard-coded fallbacks: the
# params file is the single source of truth for these values.
PARAMS_KEYS_2D = (
    "n_maxpoint",
    "bins",
    "skip_feature",
    "chunksize",
    "tanimoto_cutoff",
    "simpson_cutoff",
    "dice_cutoff",
)

# 2D signature factories always start at pairs of features.
MIN_POINT_COUNT_2D = 2


class Status:
    """Per-row outcome, reported in the ``status`` column."""

    OK = "ok"
    TIMEOUT = "timeout"
    PARSE_ERROR = "parse_error"
    NO_FEATURE = "no_feature"
    ERROR = "error"


class Stage:
    """Which path produced a 3D row, reported in the ``stage`` column."""

    CPU = "cpu"
    GPU_PREFILTER_REJECT = "gpu_prefilter_reject"
    GPU_PREFILTER_PASS = "gpu_prefilter_pass"


class InputError(Exception):
    """Raised for malformed input; the CLI turns this into exit code 2."""


@dataclasses.dataclass
class RunSummary:
    """Machine-readable summary of a screening run."""

    mode: str
    backend: str
    library: str
    out: str
    n_total: int = 0
    n_screened: int = 0
    n_timeout: int = 0
    n_parse_error: int = 0
    n_error: int = 0
    n_chunks: int = 0
    elapsed_s: float = 0.0
    outputs: list[str] = dataclasses.field(default_factory=list)
    notes: list[str] = dataclasses.field(default_factory=list)

    def count_status(self, status: str) -> None:
        if status == Status.TIMEOUT:
            self.n_timeout += 1
        elif status == Status.PARSE_ERROR:
            self.n_parse_error += 1
        elif status == Status.ERROR:
            self.n_error += 1

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        target.write_text(text + "\n", encoding="utf-8")


def require_smiles_column(columns: Any, source: str) -> None:
    """Fail fast when a library or query CSV lacks the ``SMILES`` column."""
    if SMILES_COLUMN not in list(columns):
        raise InputError(
            f"{source}: required column {SMILES_COLUMN!r} not found "
            f"(columns: {', '.join(map(str, columns))})"
        )
