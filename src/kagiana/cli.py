"""Command-line entry point.

The interface is built for scripted and agent-driven use: no prompts, no
interactive fallbacks, data on the output files, progress and warnings on
stderr, and a machine-readable summary on request.

Exit codes
    0   the run finished
    1   a runtime failure
    2   bad input (missing file, malformed CSV or params, unusable backend)
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import __version__
from .backends import AUTO, BACKEND_CHOICES
from .commands import backends_cmd, pharm2d, pharm3d, validate
from .pharmacophore3d import DEFAULT_LIMIT_LOWER, DEFAULT_LIMIT_UPPER
from .pool import WorkerInitError, default_jobs
from .progress import note
from .schema import InputError, RunSummary

EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_INPUT_ERROR = 2

# Defaults for 3D screening.  These are the settled working values; the help
# text shows them so that a caller can see what it gets without reading docs.
DEFAULT_CONFORMERS = 10
DEFAULT_MAX_POSES = 20
DEFAULT_TIMEOUT = 10.0
DEFAULT_LARGE_CHUNKSIZE = 1_000_000
DEFAULT_PREFILTER_RMSD = 1.5
DEFAULT_PREFILTER_CONFORMERS = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kagiana",
        description=(
            "Pharmacophore screening over a SMILES library. "
            "Scores go to CSV, progress to stderr; every row carries a status."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"kagiana {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_pharm2d(subparsers)
    _add_pharm3d(subparsers)

    backends_parser = subparsers.add_parser(
        "backends",
        help="report which backends this machine can run",
        description=(
            "Report RDKit, CPU and GPU availability. The GPU path needs CUDA, "
            "torch and nvmolkit; when any of those is missing the reasons are listed."
        ),
    )
    backends_parser.add_argument(
        "--json", action="store_true", help="emit the report as JSON on stdout"
    )
    backends_parser.set_defaults(handler=backends_cmd.run, returns_code=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="check input files without screening",
        description=(
            "Report every problem the screening commands would raise, in one pass. "
            "Exits 2 when any check fails."
        ),
    )
    validate_parser.add_argument("--library", required=True, help="library CSV (needs a SMILES column)")
    validate_parser.add_argument("--fdef", help="RDKit fdef feature-definition file")
    validate_parser.add_argument("--pharmacophore", help="3D feature-point CSV")
    validate_parser.add_argument("--params", help="2D params JSON")
    validate_parser.add_argument("--query", help="2D reference-molecule CSV")
    validate_parser.add_argument(
        "--json", action="store_true", help="emit the report as JSON on stdout"
    )
    validate_parser.set_defaults(handler=validate.run, returns_code=True)

    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=f"worker processes (default: all usable cores, {default_jobs()} here)",
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default=AUTO,
        help="auto falls back to cpu when no GPU is usable; gpu fails instead",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip work whose output files already exist",
    )
    parser.add_argument(
        "--json-summary",
        metavar="PATH",
        help="write a machine-readable run summary to this path",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress progress and notes on stderr"
    )


def _add_pharm2d(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "pharm2d",
        help="2D pharmacophore fingerprint screening",
        description=(
            "Score every library molecule against one reference molecule using 2D "
            "pharmacophore fingerprints, and keep the rows that clear all three "
            "cutoffs. The library is always read in chunks of the width given in "
            "the params JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--library", required=True, help="library CSV (needs a SMILES column)")
    parser.add_argument(
        "--query",
        required=True,
        help="reference-molecule CSV; the first SMILES row is used",
    )
    parser.add_argument("--fdef", required=True, help="RDKit fdef feature-definition file")
    parser.add_argument("--out", required=True, help="output CSV of the rows that passed")
    parser.add_argument(
        "--params",
        required=True,
        help="params JSON: n_maxpoint, bins, skip_feature, chunksize and the three cutoffs",
    )
    parser.add_argument(
        "--tanimoto-cutoff", type=float, default=None, help="override the params JSON value"
    )
    parser.add_argument(
        "--dice-cutoff", type=float, default=None, help="override the params JSON value"
    )
    parser.add_argument(
        "--simpson-cutoff", type=float, default=None, help="override the params JSON value"
    )
    parser.add_argument(
        "--chunksize", type=int, default=None, help="override the params JSON value"
    )
    parser.add_argument(
        "--no-all-rows",
        action="store_true",
        help="skip the <out>_all.csv file holding every row with its scores",
    )
    _add_common(parser)
    parser.set_defaults(handler=pharm2d.run, returns_code=False)


def _add_pharm3d(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "pharm3d",
        help="3D pharmacophore screening by constrained embedding",
        description=(
            "Count the poses each library molecule can adopt that satisfy a 3D "
            "pharmacophore. Molecules are screened under a per-molecule deadline; "
            "one that overruns is recorded as a timeout and the run continues."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--library", required=True, help="library CSV (needs a SMILES column)")
    parser.add_argument(
        "--pharmacophore",
        required=True,
        help="feature-point CSV with columns feature_type, X, Y, Z",
    )
    parser.add_argument("--fdef", required=True, help="RDKit fdef feature-definition file")
    parser.add_argument("--out", required=True, help="output CSV of every row with its scores")
    parser.add_argument(
        "--large",
        action="store_true",
        help="read the library in chunks and write numbered outputs per chunk",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_LARGE_CHUNKSIZE,
        help="rows per chunk, used with --large",
    )
    parser.add_argument(
        "--conformers",
        type=int,
        default=DEFAULT_CONFORMERS,
        help="conformers requested per feature assignment",
    )
    parser.add_argument(
        "--max-poses",
        type=int,
        default=DEFAULT_MAX_POSES,
        help="pose ceiling per molecule, multiplied by --conformers",
    )
    parser.add_argument(
        "--limit-upper",
        type=float,
        default=DEFAULT_LIMIT_UPPER,
        help="added to each reference distance to form the upper bound",
    )
    parser.add_argument(
        "--limit-lower",
        type=float,
        default=DEFAULT_LIMIT_LOWER,
        help="subtracted from each reference distance to form the lower bound",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="seconds a single molecule may take before its worker is killed",
    )
    parser.add_argument(
        "--prefilter-rmsd",
        type=float,
        default=DEFAULT_PREFILTER_RMSD,
        help="GPU backend: keep molecules whose best conformer fit is within this RMSD",
    )
    parser.add_argument(
        "--prefilter-conformers",
        type=int,
        default=DEFAULT_PREFILTER_CONFORMERS,
        help="GPU backend: conformers generated per molecule for the prefilter",
    )
    _add_common(parser)
    parser.set_defaults(handler=pharm3d.run, returns_code=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if getattr(args, "returns_code", False):
            return int(args.handler(args))

        summary: RunSummary = args.handler(args)
    except InputError as exc:
        print(f"kagiana: input error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except WorkerInitError as exc:
        print(f"kagiana: worker setup failed: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    except KeyboardInterrupt:
        print("kagiana: interrupted", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        print(f"kagiana: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR

    if args.json_summary:
        summary.write(args.json_summary)
    note(
        f"{summary.mode}: {summary.n_total} molecules, {summary.n_screened} passed, "
        f"{summary.n_timeout} timed out, {summary.n_parse_error} unparsable "
        f"in {summary.elapsed_s:.1f}s ({summary.backend} backend)",
        args.quiet,
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
