"""Check input files without screening anything.

A pre-flight step is only useful if passing it means the real run will not fail
on its inputs.  So every check here mirrors what a screening command does: the
library is walked as CSV records to the end, feature families are matched
against the fdef with the same function the screening commands use, and the
reference molecule is parsed and fingerprinted rather than merely read.

Problems are collected instead of raised one at a time, so a caller sees
everything wrong with its inputs in one pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .. import csv_io
from ..features import feature_families
from ..molecules import mol_from_smiles
from ..pharmacophore3d import read_feature_points, require_known_feature_types
from ..query2d import build_sig_factory, fingerprint, load_params, read_reference_smiles
from ..schema import SMILES_COLUMN, InputError


def _check_library(args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    """Walk the library as CSV records, reporting the first structural fault."""
    header: list[str] = []
    records = 0
    for number, fields in csv_io.iter_records(args.library):
        if number == 0:
            header = fields
            if SMILES_COLUMN not in header:
                return {}, (
                    f"{args.library}: required column {SMILES_COLUMN!r} not found "
                    f"(columns: {', '.join(header)})"
                )
            continue
        if not fields:
            # A trailing blank line is not a record.
            continue
        if len(fields) != len(header):
            return {}, (
                f"{args.library}: record {number} has {len(fields)} field(s) but "
                f"the header has {len(header)}; the screening commands would "
                f"fail to parse this file"
            )
        records += 1

    if not header:
        return {}, f"{args.library}: the CSV holds no data"

    smiles_index = header.index(SMILES_COLUMN)
    first_smiles: str | None = None
    for number, fields in csv_io.iter_records(args.library):
        if number == 1 and len(fields) == len(header):
            first_smiles = fields[smiles_index]
            break

    detail: dict[str, Any] = {
        "path": str(args.library),
        "records": records,
        "columns": header,
    }
    if first_smiles is not None:
        detail["first_smiles"] = first_smiles
    else:
        # A header with no records is valid input; it simply screens nothing.
        detail["note"] = "the library holds a header but no records"
    return detail, None


def collect(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {"ok": True, "checks": [], "errors": []}

    def record(name: str, detail: Any) -> None:
        report["checks"].append({"check": name, "ok": True, "detail": detail})

    def fail(name: str, message: str) -> None:
        report["ok"] = False
        report["checks"].append({"check": name, "ok": False, "detail": message})
        report["errors"].append(message)

    try:
        detail, problem = _check_library(args)
        if problem:
            fail("library", problem)
        else:
            record("library", detail)
    except (InputError, OSError) as exc:
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
            record(
                "pharmacophore",
                {
                    "path": str(args.pharmacophore),
                    "n_points": len(points),
                    "feature_types": [point.feature_type for point in points],
                },
            )
            if args.fdef and families:
                try:
                    # The same check the screening command runs, so that
                    # passing here means that command will not reject the query.
                    require_known_feature_types(points, args.fdef)
                    record("pharmacophore/fdef", "every feature type is defined")
                except InputError as exc:
                    fail("pharmacophore/fdef", f"{exc}")
        except InputError as exc:
            fail("pharmacophore", f"{exc}")

    params = None
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
        except InputError as exc:
            fail("params", f"{exc}")

    sig_factory = None
    if params is not None and args.fdef:
        try:
            sig_factory = build_sig_factory(args.fdef, params)
            record("sig_factory", "built successfully")
        except Exception as exc:  # noqa: BLE001
            fail("sig_factory", f"could not build the signature factory: {exc}")

    if args.query:
        try:
            smiles = read_reference_smiles(args.query)
            # Parsing matters: an unreadable reference molecule would otherwise
            # only surface once the screening run had already started.
            if mol_from_smiles(smiles) is None:
                fail(
                    "query",
                    f"{args.query}: the reference SMILES could not be parsed: {smiles!r}",
                )
            elif sig_factory is not None:
                if fingerprint(smiles, sig_factory) is None:
                    fail(
                        "query",
                        f"{args.query}: no fingerprint could be generated for the "
                        f"reference molecule {smiles!r}",
                    )
                else:
                    record(
                        "query",
                        {
                            "path": str(args.query),
                            "reference_smiles": smiles,
                            "fingerprint": "generated successfully",
                        },
                    )
            else:
                record(
                    "query", {"path": str(args.query), "reference_smiles": smiles}
                )
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
