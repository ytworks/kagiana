"""3D pharmacophore screening.

Two reading modes, matching how large the library is:

* whole-file mode loads the library at once and writes ``R.csv`` with every row
  plus ``R_screened.csv`` with the hits;
* chunked mode (``--large``) reads a fixed number of rows at a time and writes
  ``R_<i>.csv`` and ``R_screened_<i>.csv`` as it goes, then concatenates the
  hits into ``R_screened_all.csv``.

Every molecule runs under a deadline.  When it expires the worker holding it is
killed and the row is marked ``timeout``, keeping the poses it had already
embedded plus one for the assignment that was interrupted.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import monotonic
from typing import Any

import pandas as pd

from .. import csv_io
from ..errlog import ErrorLog
from ..backends import GPU, resolve_backend
from ..backends.cpu3d import Partial3D, init_worker as cpu_init_worker
from ..pool import KillableWorkerPool, default_jobs
from ..progress import note, progress_bar
from ..schema import (
    DIAGNOSTIC_COLUMNS_3D,
    SCORE_COLUMNS_3D,
    SMILES_COLUMN,
    InputError,
    RunSummary,
    Stage,
    Status,
)

NAN = float("nan")


def run(args: argparse.Namespace) -> RunSummary:
    started = monotonic()

    # Validate the query before starting workers.
    from ..pharmacophore3d import make_pharmacophore

    make_pharmacophore(args.pharmacophore, args.limit_upper, args.limit_lower)

    if args.conformers < 1:
        raise InputError(f"--conformers must be >= 1, got {args.conformers}")
    if args.max_poses < 1:
        raise InputError(f"--max-poses must be >= 1, got {args.max_poses}")
    if args.timeout is not None and args.timeout <= 0:
        raise InputError(f"--timeout must be > 0, got {args.timeout}")

    backend, probe, notes = resolve_backend(args.backend)
    if backend == GPU and not probe.available:
        raise InputError(
            "--backend gpu was requested but the GPU path is unavailable: "
            + "; ".join(probe.reasons)
        )

    summary = RunSummary(
        mode="pharm3d",
        backend=backend,
        library=str(args.library),
        out=str(args.out),
    )
    summary.notes.extend(notes)
    for message in notes:
        note(message, args.quiet)

    jobs = args.jobs or default_jobs()
    chunksize = args.chunksize if args.large else None

    if not args.large and args.resume and Path(args.out).is_file():
        message = f"resume: {args.out} already exists, nothing to do"
        summary.notes.append(message)
        note(message, args.quiet)
        summary.outputs.append(str(args.out))
        summary.elapsed_s = monotonic() - started
        return summary

    screened_frames: list[pd.DataFrame] = []
    error_log = ErrorLog(args.out)

    for chunk_index, frame in csv_io.read_chunks(args.library, chunksize):
        summary.n_chunks += 1

        if args.large:
            chunk_target = csv_io.numbered_path(args.out, chunk_index)
            screened_target = csv_io.screened_numbered_path(args.out, chunk_index)
            if args.resume and chunk_target.is_file() and screened_target.is_file():
                message = f"resume: chunk {chunk_index} already written, skipping"
                summary.notes.append(message)
                note(message, args.quiet)
                done = pd.read_csv(screened_target, index_col=0)
                screened_frames.append(done)
                summary.n_total += len(pd.read_csv(chunk_target, index_col=0))
                summary.outputs.extend([str(chunk_target), str(screened_target)])
                continue
        else:
            chunk_target = Path(args.out)
            screened_target = csv_io.screened_path(args.out)

        smiles = csv_io.smiles_values(frame, SMILES_COLUMN)
        rows = _screen_chunk(
            smiles,
            args=args,
            backend=backend,
            jobs=jobs,
            chunk_index=chunk_index,
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

        csv_io.write_frame(scored, chunk_target, index=True)
        hits = scored[scored["numMatch"] > 0]
        csv_io.write_frame(hits, screened_target, index=True)
        summary.outputs.extend([str(chunk_target), str(screened_target)])
        screened_frames.append(hits)

    error_log.close()
    all_hits = (
        pd.concat(screened_frames, ignore_index=True)
        if screened_frames
        else pd.DataFrame()
    )
    summary.n_screened = len(all_hits)

    if args.large:
        target = csv_io.screened_all_path(args.out)
        csv_io.write_frame(all_hits, target, index=True)
        summary.outputs.append(str(target))

    if error_log.used:
        summary.outputs.append(str(error_log.path))
        summary.notes.append(
            f"{error_log.count} row(s) did not score cleanly; see {error_log.path}"
        )

    summary.elapsed_s = monotonic() - started
    return summary


def _screen_chunk(
    smiles: list[str],
    args: argparse.Namespace,
    backend: str,
    jobs: int,
    chunk_index: int,
) -> list[dict[str, Any]]:
    """Screen one chunk, optionally using the GPU prefilter first."""
    prefilter: dict[int, dict[str, Any]] = {}
    prefilter_rmsd: dict[int, float] = {}
    if backend == GPU:
        from ..backends import gpu3d

        prefilter, prefilter_rmsd = gpu3d.prefilter_chunk(
            smiles,
            fdef_path=str(args.fdef),
            pharmacophore_path=str(args.pharmacophore),
            limit_upper=args.limit_upper,
            limit_lower=args.limit_lower,
            prefilter_conformers=args.prefilter_conformers,
            prefilter_rmsd=args.prefilter_rmsd,
            quiet=args.quiet,
        )

    # Only molecules that survived the prefilter need the constrained-embedding
    # pass; the rest are already decided.
    todo = [i for i in range(len(smiles)) if i not in prefilter]
    rows: list[dict[str, Any]] = [dict() for _ in smiles]
    for index, decided in prefilter.items():
        rows[index] = decided

    if todo:
        label = f"pharm3d chunk {chunk_index}"
        stage = Stage.GPU_PREFILTER_PASS if backend == GPU else Stage.CPU
        with progress_bar(len(todo), label, args.quiet) as bar:
            with KillableWorkerPool(
                n_workers=min(jobs, max(1, len(todo))),
                initializer=cpu_init_worker,
                initargs=(
                    str(args.fdef),
                    str(args.pharmacophore),
                    args.limit_upper,
                    args.limit_lower,
                    args.conformers,
                    args.max_poses,
                ),
                timeout=args.timeout,
            ) as pool:
                for result in pool.map_ordered([smiles[i] for i in todo]):
                    position = todo[result.index]
                    row = _row_from_result(result, stage)
                    # Carry the prefilter's fit through to the surviving rows.
                    row["best_rmsd"] = prefilter_rmsd.get(position, NAN)
                    rows[position] = row
                    bar.update(1)
    return rows


def _row_from_result(result: Any, stage: str) -> dict[str, Any]:
    if result.status == Status.TIMEOUT:
        partial: Partial3D | None = result.partial
        return {
            "match": bool(partial.match) if partial else False,
            "numMatch": partial.num_match() if partial else 0,
            "status": Status.TIMEOUT,
            "elapsed_s": result.elapsed_s,
            "stage": stage,
            "best_rmsd": NAN,
            "_detail": result.detail,
        }
    if result.status == Status.OK and result.value is None:
        return {
            "match": False,
            "numMatch": 0,
            "status": Status.PARSE_ERROR,
            "elapsed_s": result.elapsed_s,
            "stage": stage,
            "best_rmsd": NAN,
            "_detail": result.detail,
        }
    if result.status != Status.OK:
        return {
            "match": False,
            "numMatch": 0,
            "status": result.status,
            "elapsed_s": result.elapsed_s,
            "stage": stage,
            "best_rmsd": NAN,
            "_detail": result.detail,
        }
    score = result.value
    return {
        "match": score.match,
        "numMatch": score.num_match,
        "status": Status.OK if score.match else Status.NO_FEATURE,
        "elapsed_s": result.elapsed_s,
        "stage": stage,
        "best_rmsd": NAN,
        "_detail": "",
    }


def _attach_columns(frame: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    scored = frame.copy()
    extra = pd.DataFrame(rows, index=scored.index)
    for column in (*SCORE_COLUMNS_3D, *DIAGNOSTIC_COLUMNS_3D):
        scored[column] = extra[column]
    return scored
