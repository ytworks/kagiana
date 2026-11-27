# Timeouts, resuming and diagnostics

These three are one design, not three. A molecule that cannot be screened has to be abandoned
without ending the run; abandoning it has to be recorded; and a run that is interrupted has to be
able to tell what it finished from what it did not. Each of the three has been got wrong at least
once here, and the mistakes were not obvious, so this is what the current design is and why.

## Why the timeout needs a separate process

The expensive part of 3D screening happens inside RDKit's C++ layer. A distance-geometry embedding
that fails to converge does not return to Python, so no Python-level mechanism can stop it:
`concurrent.futures` cannot cancel a running task, and a thread cannot be interrupted while it is
inside the extension.

So each worker is a process, it is given one molecule at a time, and the parent kills it when the
deadline passes. The run continues on a replacement worker.

## What the deadline is measured against

The deadline is set when the task is dispatched: `dispatched_at + timeout`. Everything is judged
against that one value, on the monotonic clock the parent and the worker share.

Four things went wrong here in turn, and all four are now covered by tests:

| What was measured | What it got wrong |
|---|---|
| Wall clock at the moment the parent read the result | A task that overran was accepted as a success if its result arrived between two deadline sweeps |
| Wall clock at the moment the parent read the result | A task that finished in time became a `timeout` when the caller paused between results |
| The worker's elapsed time from the start of the task | A worker that began work late was accepted, because the wait before starting was not counted |
| Nothing: progress reports carried no time at all | Poses generated after the deadline were added to the timed-out row |

The result and every progress report now carry the worker's own `monotonic()` reading, and the
parent compares those against the deadline it set. A report from after the deadline is dropped.

## What a timed-out row says

The row keeps the work that was done inside the deadline: the poses already embedded, plus one for
the assignment that was interrupted. The worker reports its progress before each assignment, so
the parent still has that count after killing the worker.

## Why `--resume` cannot trust the filesystem

Every result file is written to a temporary name and moved into place, so a file under its final
name is complete. That is necessary but not sufficient: it says nothing about *which* run produced
it. Screening a second library into the same output directory leaves a mixture, and resuming on
file existence alone kept whichever half was still on disk — in one reproduction, chunk results
`new0, old1, new2`.

So a run records what it finished, in `.<out>_state/`:

- a generation key, derived from the inputs and from every setting that changes what a row says
  (the library's identity, the query, the fdef, the chunking, the conformer and pose limits, the
  distance margins, the deadline, the backend that will actually run, and the prefilter settings);
- the chunks it completed;
- whether a whole-file run finished both of its outputs;
- the diagnostics of each chunk.

`--resume` skips only what that record covers, and a run whose key differs starts over. The
directory's name is derived from the output path, so it can collide with a directory of the
caller's own: one is touched only when it carries the marker this tool writes, and a collision is
reported as an input problem rather than deleted.

## The order that makes a chunk resumable

Within a chunk, the order of writes is the invariant:

1. screen the molecules;
2. write the chunk's diagnostics and close them;
3. write the chunk's two result CSVs, each atomically;
4. record the chunk as complete.

An interruption between any two of these leaves the chunk un-recorded, so it is screened again.
The one order that cannot work is recording completion before the diagnostics are durable: the
chunk would then be skipped and its reasons lost.

## Why a chunk that failed nothing still leaves a file

"This chunk was screened and nothing was wrong" and "this chunk was never screened" have to be
distinguishable. If they are not, a resumed run falls back on an older log and resurrects failures
the current results do not have. So every completed chunk leaves its diagnostics file, empty when
there is nothing to report.

One file is open at a time, and each is renamed into place when complete: a run over many chunks
would otherwise exhaust file descriptors, and an interruption mid-write would leave a partial file
that a later run reads as finished.

## Where the deadline does not reach

Two pieces of work run in the parent process, outside the per-molecule deadline:

- **The GPU prefilter.** Conformers are generated for a whole batch in one device call. A
  per-molecule interruption point inside it would mean giving up the batching that makes it worth
  using. The CLI help and the READMEs say so.
- **The assignment search** in `geometry.py`. It is bounded instead: a query with more points than
  there are distinct candidate positions is settled by counting rather than searched, and the
  number of partial assignments examined is capped. A search that hits either ceiling reports
  itself as inconclusive, and the caller sends that molecule to the CPU rather than rejecting it.
