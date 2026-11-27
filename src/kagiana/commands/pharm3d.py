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
import csv
import os
from pathlib import Path
from time import monotonic
from typing import Any

import pandas as pd

from .. import csv_io
from ..backends import GPU, resolve_backend
from ..backends.cpu3d import Partial3D
from ..backends.cpu3d import init_worker as cpu_init_worker
from ..errlog import ErrorLog
from ..pool import KillableWorkerPool, default_jobs
from ..progress import note, progress_bar
from ..runstate import RunState, generation_key
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

    # Validate the query before starting workers: a misspelt feature family
    # would otherwise match nothing and come back as a clean run with no hits.
    from ..pharmacophore3d import (
        make_pharmacophore,
        read_feature_points,
        require_known_feature_types,
    )

    require_known_feature_types(read_feature_points(args.pharmacophore), args.fdef)
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

    # What counts as "the same run": the inputs and the settings that change
    # what the results would be.  Resuming into an output directory whose
    # results came from anything else starts over rather than mixing the two.
    key = generation_key(
        mode="pharm3d",
        # The backend that will actually run, not the one that was asked for:
        # the prefilter settings below only bear on results it produced.
        backend=backend,
        prefilter_rmsd=args.prefilter_rmsd,
        prefilter_conformers=args.prefilter_conformers,
        library=Path(args.library),
        pharmacophore=Path(args.pharmacophore),
        fdef=Path(args.fdef),
        large=bool(args.large),
        chunksize=chunksize,
        conformers=args.conformers,
        max_poses=args.max_poses,
        limit_upper=args.limit_upper,
        limit_lower=args.limit_lower,
        timeout=args.timeout,
    )
    state = RunState.open(args.out, key, resume=args.resume)
    error_log = ErrorLog(args.out, state)

    whole_file_outputs = (Path(args.out), csv_io.screened_path(args.out))
    if (
        not args.large
        and args.resume
        and state.finalized
        and all(target.is_file() for target in whole_file_outputs)
    ):
        message = f"resume: {args.out} is already complete, nothing to do"
        summary.notes.append(message)
        note(message, args.quiet)
        summary.outputs.extend(str(target) for target in whole_file_outputs)
        _restore_summary(summary, state.summary)
        # Diagnostics recorded before the interruption are published here too,
        # or a finished run would leave its reasons unreadable.
        if error_log.publish():
            summary.outputs.append(str(error_log.path))
        summary.elapsed_s = monotonic() - started
        return summary

    # Chunk files finished in this generation, in the order they were written:
    # the combined file is built by streaming these rather than by holding
    # every hit of the run in memory.
    hit_files: list[Path] = []
    # Hits are counted per chunk as they are written. Counting the lines of the
    # combined file instead would miscount a row whose own cells contain a
    # newline.
    hit_total = 0

    for chunk_index, frame in csv_io.read_chunks(args.library, chunksize):
        summary.n_chunks += 1

        if args.large:
            chunk_target = csv_io.numbered_path(args.out, chunk_index)
            screened_target = csv_io.screened_numbered_path(args.out, chunk_index)
            # Only a chunk this generation recorded as finished may be skipped.
            # Its files existing is not enough: they may be left over from a
            # different library screened into the same place.
            if (
                args.resume
                and state.is_chunk_complete(chunk_index)
                and chunk_target.is_file()
                and screened_target.is_file()
            ):
                message = f"resume: chunk {chunk_index} already written, skipping"
                summary.notes.append(message)
                note(message, args.quiet)
                summary.n_total += csv_io.count_records(chunk_target)
                # Failures recorded earlier still belong in this run's summary.
                for status, count in csv_io.read_status_counts(chunk_target).items():
                    for _ in range(count):
                        summary.count_status(status)
                hit_files.append(screened_target)
                hit_total += csv_io.count_records(screened_target)
                summary.outputs.extend([str(chunk_target), str(screened_target)])
                continue
        else:
            chunk_target = Path(args.out)
            screened_target = csv_io.screened_path(args.out)

        error_log.start_chunk(chunk_index)
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

        # Make this chunk's diagnostics durable before its results appear, and
        # record the chunk as finished only once both outputs are on disk.
        error_log.finish_chunk()
        hits = scored[scored["numMatch"] > 0]
        csv_io.write_frame(scored, chunk_target, index=True)
        csv_io.write_frame(hits, screened_target, index=True)
        state.mark_chunk_complete(chunk_index)
        summary.outputs.extend([str(chunk_target), str(screened_target)])
        hit_total += len(hits)
        hit_files.append(screened_target)

    summary.n_screened = hit_total

    if args.large:
        # Stream the finished per-chunk files into the combined one so that
        # peak memory stays at one chunk however many hits the run found.
        target = csv_io.screened_all_path(args.out)
        _combine_hits(hit_files, target)
        summary.outputs.append(str(target))

    if error_log.publish():
        summary.outputs.append(str(error_log.path))
        summary.notes.append(
            f"{error_log.count} row(s) did not score cleanly; see {error_log.path}"
        )

    # Recorded last: the run is finished once everything it owes is on disk,
    # and the tallies travel with that record so a resume can repeat them.
    state.mark_finalized(_summary_tallies(summary))

    summary.elapsed_s = monotonic() - started
    return summary


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


def _combine_hits(hit_files: list[Path], target: Path) -> None:
    """Concatenate finished per-chunk hit files into one, keeping one header."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for position, source in enumerate(hit_files):
                csv_io.concat_csv_records(source, writer, write_header=position == 0)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
    rows: list[dict[str, Any]] = [{} for _ in smiles]
    for index, decided in prefilter.items():
        rows[index] = decided

    if todo:
        label = f"pharm3d chunk {chunk_index}"
        stage = Stage.GPU_PREFILTER_PASS if backend == GPU else Stage.CPU
        with progress_bar(len(todo), label, args.quiet) as bar, KillableWorkerPool(
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
            "_detail": "RDKit could not build a molecule from the SMILES string",
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
    if not rows:
        # A library with a header and no records still gets a result file, and
        # it has to carry the columns a caller expects to read.
        for column in (*SCORE_COLUMNS_3D, *DIAGNOSTIC_COLUMNS_3D):
            scored[column] = pd.Series(dtype="object")
        return scored
    extra = pd.DataFrame(rows, index=scored.index)
    for column in (*SCORE_COLUMNS_3D, *DIAGNOSTIC_COLUMNS_3D):
        scored[column] = extra[column]
    return scored
