# kagiana

kagiana screens a SMILES library against a pharmacophore, in two modes: 2D fingerprint similarity
to one reference molecule, and 3D matching by constrained embedding. The command line is built for
programs rather than people: non-interactive, self-describing, every row carrying a status.
Package: `src/kagiana/`.

## Session start

ALWAYS read `PROGRESS.md` first. It is the cross-session memory: current state, what has been
measured, what is deliberately refused, and what has already been settled in review.

Then read `docs/reliability.md` before changing the worker pool, the resume logic or the error log.
Those three are coupled, and most of the review findings on this branch were in them rather than
in the chemistry.

## Where things live

- `src/kagiana/pool.py` — the worker pool that can abort a task by force. The core of the timeout
  guarantee; read `docs/reliability.md` before touching it.
- `src/kagiana/runstate.py` — what a run has finished, and under which inputs. The basis of
  `--resume`.
- `src/kagiana/errlog.py` — per-chunk diagnostics, kept in step with the state.
- `src/kagiana/csv_io.py` — library reading, atomic writes, output naming. The CSV contract.
- `src/kagiana/molecules.py` — the single place a library cell becomes a molecule.
- `src/kagiana/query2d.py`, `src/kagiana/pharmacophore3d.py` — the two query types.
- `src/kagiana/features.py` — fdef loading, cached per process.
- `src/kagiana/geometry.py` — Kabsch RMSD and the assignment search used by the GPU prefilter.
- `src/kagiana/backends/` — `cpu2d.py`, `cpu3d.py`, `gpu2d.py`, `gpu3d.py`, and the capability
  probe in `__init__.py`.
- `src/kagiana/commands/` — one module per subcommand.

## Rules specific to this project

1. **A deadline is a deadline.** A molecule that overruns `--timeout` is reported as `timeout`,
   whether the delay was in the work, in the worker starting up, or in the parent reading the
   result. Progress reported after the deadline does not count toward the row. Both are judged on
   one clock, against the deadline set at dispatch. Four review rounds found four different ways
   to get this wrong; `docs/reliability.md` lists them.

2. **`--resume` trusts the record, never the filesystem.** A file under its final name is
   complete, but says nothing about which run produced it. Only what the current generation key
   recorded as finished may be skipped. Any setting that changes what a row says belongs in that
   key.

3. **The prefilter may only remove work.** A molecule the GPU prefilter settles must be one the
   constrained pass would have agreed about. Anything it cannot measure conclusively — no usable
   conformer, a missing feature family, an assignment search that hit its budget — goes to the
   CPU rather than being decided. Features are identified *and placed* on the molecule the
   constrained pass asks about, because explicit hydrogens change both.

4. **Library cells are the caller's data.** They are read as text and passed through unchanged. An
   identifier of `00123` must not come back as `123`, and a note of `NA` must not come back empty.
   Never route the library through type inference.

5. **Never substitute a molecule.** A cell that does not describe a molecule — unparsable, or
   blank, which RDKit accepts as an atomless molecule — is reported with a reason, not screened as
   something else.

6. **One bad molecule must not end a run.** Failures are per row: a status in the CSV and a reason
   in the JSON Lines log beside it. The run's exit code reports the run, not the rows.

7. **Never report a number you have not measured.** Every figure in the READMEs came from a run
   that is reproducible by the command beside it. The scaling table is one measurement on one
   machine and says so. Do not fill a gap with an estimate.

8. **Say what is unverified.** The GPU path has never run: no CUDA here, and nvMolKit ships
   x86_64 Linux wheels only. The READMEs carry a verification-status section, and it stays honest.
   Tests that need a device are marked `gpu` and skip themselves.

9. **Both READMEs change together.** `scripts/check_readme_pair.py` compares the heading outline,
   the commands in code blocks, the links and the numbers in tables. CI runs it.

## Test policy

- `pytest` runs everything; `-m 'not gpu'` is the default outcome anyway, since the GPU tests skip
  without a device.
- No test may touch the network.
- A test asserting current behaviour is a contract. If a change makes one fail, decide whether the
  behaviour or the test is wrong, and say which in the commit message.
- Reproduce a bug before fixing it, and leave the reproduction as a test. `tests/test_review_fixes.py`
  is one test per review finding, named for the condition rather than the fix.
- Worker initializers used by tests live in `tests/helpers.py`: the spawn start method imports the
  module they come from, so they cannot be defined inside a test function.

## Style

Immutable data where it costs nothing, small focused modules, explicit error handling, no silently
swallowed exceptions, named constants rather than magic numbers. Docstrings say *why*, not what the
signature already says. `ruff` with a 120-column line length.
