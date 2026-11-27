"""2D pharmacophore screening.

The library is always read in chunks, using the width from the params JSON, and
the rows that clear all three cutoffs are collected and written as one file.
The per-row scores for every molecule are also kept, which makes it possible to
re-filter without screening again.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import monotonic
from typing import Any

import pandas as pd

from .. import csv_io
from ..backends import GPU, resolve_backend
from ..backends.cpu2d import init_worker as cpu_init_worker
from ..errlog import ErrorLog
from ..pool import KillableWorkerPool, default_jobs
from ..progress import note, progress_bar
from ..query2d import (
    build_sig_factory,
    load_params,
    read_reference_smiles,
    reference_fingerprint,
)
from ..runstate import RunState, generation_key
from ..schema import (
    DIAGNOSTIC_COLUMNS_2D,
    SCORE_COLUMNS_2D,
    SMILES_COLUMN,
    InputError,
    RunSummary,
    Status,
)

NAN = float("nan")


def run(args: argparse.Namespace) -> RunSummary:
    started = monotonic()

    params = load_params(args.params).with_overrides(
        tanimoto_cutoff=args.tanimoto_cutoff,
        simpson_cutoff=args.simpson_cutoff,
        dice_cutoff=args.dice_cutoff,
        chunksize=args.chunksize,
    )
    reference_smiles = read_reference_smiles(args.query)
    # Build the signature factory and the reference fingerprint here too: a bad
    # fdef, an unknown feature family or an unreadable reference molecule should
    # be an input error before any worker is started, not a worker that fails
    # to initialise once the run is under way.
    reference_fingerprint(reference_smiles, build_sig_factory(args.fdef, params))

    backend, probe, notes = resolve_backend(args.backend)
    if backend == GPU and not probe.available:
        raise InputError(
            "--backend gpu was requested but the GPU path is unavailable: "
            + "; ".join(probe.reasons)
        )

    summary = RunSummary(
        mode="pharm2d",
        backend=backend,
        library=str(args.library),
        out=str(args.out),
    )
    summary.notes.extend(notes)
    for message in notes:
        note(message, args.quiet)

    jobs = args.jobs or default_jobs()
    all_rows_target = csv_io.all_rows_path(args.out)
    write_all_rows = not args.no_all_rows

    screened_frames: list[pd.DataFrame] = []
    wrote_all_rows_header = False
    state = RunState.open(
        args.out,
        generation_key(
            mode="pharm2d",
            library=Path(args.library),
            query=Path(args.query),
            fdef=Path(args.fdef),
            params=Path(args.params),
            chunksize=params.chunksize,
            write_all_rows=not args.no_all_rows,
            cutoffs=(
                params.tanimoto_cutoff,
                params.dice_cutoff,
                params.simpson_cutoff,
            ),
        ),
        resume=args.resume,
    )
    error_log = ErrorLog(args.out, state)

    # Skip the run only when this generation recorded it as finished and the
    # outputs it owes are there. The output file existing is not enough: it may
    # hold the results of a different library screened into the same place.
    required = [Path(args.out)] + ([all_rows_target] if write_all_rows else [])
    if (
        args.resume
        and state.finalized
        and all(target.is_file() for target in required)
    ):
        message = f"resume: {args.out} is already complete, nothing to do"
        summary.notes.append(message)
        note(message, args.quiet)
        summary.outputs.extend(str(target) for target in required)
        _restore_summary(summary, state.summary)
        # Diagnostics collected before the interruption are published here too,
        # or a finished run would leave its reasons unreadable.
        if error_log.publish():
            summary.outputs.append(str(error_log.path))
        summary.elapsed_s = monotonic() - started
        return summary

    for chunk_index, frame in csv_io.read_chunks(args.library, params.chunksize):
        summary.n_chunks += 1
        smiles = csv_io.smiles_values(frame, SMILES_COLUMN)
        error_log.start_chunk(chunk_index)

        if backend == GPU:
            from ..backends import gpu2d

            rows = gpu2d.score_chunk(
                smiles,
                fdef_path=str(args.fdef),
                params=params,
                reference_smiles=reference_smiles,
                jobs=jobs,
                quiet=args.quiet,
            )
        else:
            rows = _score_chunk_cpu(
                smiles,
                fdef_path=str(args.fdef),
                params=params,
                reference_smiles=reference_smiles,
                jobs=jobs,
                chunk_index=chunk_index,
                quiet=args.quiet,
            )

        scored = _attach_columns(frame, rows)
        summary.n_total += len(scored)
        for position, row in enumerate(rows):
            summary.count_status(row["status"])
            error_log.record(
                chunk=chunk_index,
                row=position,
                smiles=smiles[position],
                status=row["status"],
                detail=str(row.get("_detail", "")),
            )

        if write_all_rows:
            mode = "a" if wrote_all_rows_header else "w"
            all_rows_target.parent.mkdir(parents=True, exist_ok=True)
            scored.to_csv(
                all_rows_target,
                index=False,
                mode=mode,
                header=not wrote_all_rows_header,
            )
            wrote_all_rows_header = True

        error_log.finish_chunk()
        state.mark_chunk_complete(chunk_index)
        screened_frames.append(_apply_cutoffs(scored, params))

    screened = (
        pd.concat(screened_frames, ignore_index=True)
        if screened_frames
        else pd.DataFrame()
    )
    summary.n_screened = len(screened)
    csv_io.write_frame(screened, args.out, index=False)
    summary.outputs.append(str(args.out))
    if write_all_rows and wrote_all_rows_header:
        summary.outputs.append(str(all_rows_target))
    state.mark_finalized(_summary_tallies(summary))
    if error_log.publish():
        summary.outputs.append(str(error_log.path))
        summary.notes.append(
            f"{error_log.count} row(s) did not score cleanly; see {error_log.path}"
        )

    summary.elapsed_s = monotonic() - started
    return summary


def _score_chunk_cpu(
    smiles: list[str],
    fdef_path: str,
    params: Any,
    reference_smiles: str,
    jobs: int,
    chunk_index: int,
    quiet: bool,
) -> list[dict[str, Any]]:
    """Fingerprint and score a chunk across worker processes."""
    rows: list[dict[str, Any]] = []
    label = f"pharm2d chunk {chunk_index}"
    with progress_bar(len(smiles), label, quiet) as bar, KillableWorkerPool(
        n_workers=min(jobs, max(1, len(smiles))),
        initializer=cpu_init_worker,
        initargs=(fdef_path, params, reference_smiles),
        timeout=None,
    ) as pool:
        for result in pool.map_ordered(smiles):
            rows.append(_row_from_result(result))
            bar.update(1)
    return rows


#: The tallies a finished run records, so that resuming can repeat them.
_TALLIES = (
    "n_total",
    "n_screened",
    "n_timeout",
    "n_parse_error",
    "n_error",
    "n_chunks",
)


def _summary_tallies(summary: RunSummary) -> dict[str, int]:
    return {name: int(getattr(summary, name)) for name in _TALLIES}


def _restore_summary(summary: RunSummary, stored: dict[str, Any]) -> None:
    """Put back the counts a finished run reported."""
    for name in _TALLIES:
        value = stored.get(name)
        if isinstance(value, int):
            setattr(summary, name, value)


def _row_from_result(result: Any) -> dict[str, Any]:
    if result.status == Status.OK and result.value is None:
        # RDKit could not build a molecule from this SMILES string.
        return {
            "tanimoto": NAN,
            "dice": NAN,
            "simpson": NAN,
            "status": Status.PARSE_ERROR,
            "elapsed_s": result.elapsed_s,
            "_detail": "RDKit could not parse the SMILES string",
        }
    if result.status != Status.OK:
        return {
            "tanimoto": NAN,
            "dice": NAN,
            "simpson": NAN,
            "status": result.status,
            "elapsed_s": result.elapsed_s,
            "_detail": result.detail,
        }
    score = result.value
    return {
        "tanimoto": score.tanimoto,
        "dice": score.dice,
        "simpson": score.simpson,
        "status": Status.OK,
        "elapsed_s": result.elapsed_s,
        "_detail": "",
    }


def _attach_columns(frame: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    scored = frame.copy()
    if not rows:
        # A library with a header and no records still gets a result file, and
        # it has to carry the columns a caller expects to read.
        for column in (*SCORE_COLUMNS_2D, *DIAGNOSTIC_COLUMNS_2D):
            scored[column] = pd.Series(dtype="object")
        return scored
    extra = pd.DataFrame(rows, index=scored.index)
    for column in (*SCORE_COLUMNS_2D, *DIAGNOSTIC_COLUMNS_2D):
        scored[column] = extra[column]
    return scored


def _apply_cutoffs(frame: pd.DataFrame, params: Any) -> pd.DataFrame:
    """Keep the rows that clear every cutoff (an AND of the three scores)."""
    keep = frame["tanimoto"] >= params.tanimoto_cutoff
    keep &= frame["dice"] >= params.dice_cutoff
    keep &= frame["simpson"] >= params.simpson_cutoff
    return frame[keep]
