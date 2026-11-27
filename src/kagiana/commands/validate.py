"""Check input files without screening anything.

Useful as a cheap pre-flight step: it reports the problems the screening
commands would raise, all at once, instead of failing on the first one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .. import csv_io
from ..features import feature_families
from ..pharmacophore3d import read_feature_points
from ..query2d import build_sig_factory, load_params, read_reference_smiles
from ..schema import SMILES_COLUMN, InputError


def collect(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {"ok": True, "checks": [], "errors": []}

    def record(name: str, detail: Any) -> None:
        report["checks"].append({"check": name, "ok": True, "detail": detail})

    def fail(name: str, message: str) -> None:
        report["ok"] = False
        report["checks"].append({"check": name, "ok": False, "detail": message})
        report["errors"].append(message)

    try:
        rows = csv_io.count_rows(args.library)
        _, frame = next(csv_io.read_chunks(args.library, 1))
        record(
            "library",
            {
                "path": str(args.library),
                "rows": rows,
                "columns": list(frame.columns),
                "first_smiles": str(frame[SMILES_COLUMN].iloc[0]),
            },
        )
    except (InputError, StopIteration) as exc:
        fail("library", f"{exc}")

    families: list[str] = []
    if args.fdef:
        try:
            families = feature_families(args.fdef)
            record("fdef", {"path": str(args.fdef), "families": families})
        except InputError as exc:
            fail("fdef", f"{exc}")

    if args.pharmacophore:
        try:
            points = read_feature_points(args.pharmacophore)
            types = [point.feature_type for point in points]
            record(
                "pharmacophore",
                {"path": str(args.pharmacophore), "n_points": len(points), "feature_types": types},
            )
            if families:
                unknown = sorted({t for t in types if t not in families})
                if unknown:
                    fail(
                        "pharmacophore/fdef",
                        "feature type(s) not defined by the fdef file: "
                        + ", ".join(unknown),
                    )
        except InputError as exc:
            fail("pharmacophore", f"{exc}")

    if args.params:
        try:
            params = load_params(args.params)
            record(
                "params",
                {
                    "path": str(args.params),
                    "n_maxpoint": params.n_maxpoint,
                    "bins": [list(b) for b in params.bins],
                    "skip_feature": list(params.skip_feature),
                    "chunksize": params.chunksize,
                    "cutoffs": {
                        "tanimoto": params.tanimoto_cutoff,
                        "dice": params.dice_cutoff,
                        "simpson": params.simpson_cutoff,
                    },
                },
            )
            if args.fdef:
                try:
                    build_sig_factory(args.fdef, params)
                    record("sig_factory", "built successfully")
                except Exception as exc:  # noqa: BLE001
                    fail("sig_factory", f"could not build the signature factory: {exc}")
        except InputError as exc:
            fail("params", f"{exc}")

    if args.query:
        try:
            smiles = read_reference_smiles(args.query)
            record("query", {"path": str(args.query), "reference_smiles": smiles})
        except InputError as exc:
            fail("query", f"{exc}")

    return report


def run(args: argparse.Namespace) -> int:
    report = collect(args)
    if args.json:
        json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        for check in report["checks"]:
            mark = "ok  " if check["ok"] else "FAIL"
            print(f"[{mark}] {check['check']}: {check['detail']}")
        print("valid" if report["ok"] else "invalid")
    return 0 if report["ok"] else 2
