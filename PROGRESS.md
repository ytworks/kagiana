# Progress

Cross-session memory for kagiana: what exists, what is measured, what is not done, and what to
pick up next. Read this first, then `CLAUDE.md`.

Last updated: 2026-11-28. Branch `feat/pharmacophore-screening-cli`, draft PR #1.
140 tests pass locally and in CI on Ubuntu and macOS for Python 3.11 to 3.13; 3 skip without a
CUDA device. The GPU path has now been run on real hardware.

## State

Both screening modes work and are exercised on real data. The CLI is `kagiana` with four
subcommands.

| Part | State |
|---|---|
| `pharm2d` | Works. 2D pharmacophore fingerprints against one reference molecule; Tanimoto, Dice and Simpson from one intersection count |
| `pharm3d` | Works. Feature match, distance bounds, assignment enumeration, constrained embedding; scored by poses embedded |
| CPU parallelism | Works and measured. Worker processes, results in input order |
| Timeouts | Works. Worker killed on its deadline; verified on real data (Suramin at 10 s) |
| `--resume` | Works. Driven by a recorded generation key, not by file existence |
| GPU 2D | Run. Scores identical to the CPU backend, largest difference 0 |
| GPU 3D | Run. Same hits as the CPU backend; `match` and `numMatch` identical on every row |
| Docker CPU | Built and run, including inside the container |
| Docker GPU | Built and run on a Blackwell card |

## What is measured

50,000 synthetic molecules, 12 cores, identical results at every job count:

| Mode | `--jobs 1` | `--jobs 4` | `--jobs 12` |
|---|---|---|---|
| `pharm3d` | 24.8 | 8.0 | 4.0 |
| `pharm2d` | 4.3 | 1.9 | 1.3 |

Seconds. Reproduce with `kagiana pharm3d --library <50k rows> --large --chunksize 5000 --jobs N`.

Peak resident memory in `--large` mode follows the chunk width, not the hit count: 100 MB at
`--chunksize 2000` and 122 MB at `--chunksize 25000`, for a library with 14,539 hits.

On one NVIDIA RTX PRO 6000 Blackwell (compute capability 12.0, driver 595.99.02), with nvMolKit
0.7.0, torch 2.11.0+cu128 and RDKit 2026.03.6, through `docker/Dockerfile.gpu`:

| Check | Result |
|---|---|
| `backends` reports the device | Yes — the published wheels work on a GPU newer than their CUDA |
| 2D scores against the CPU backend | Identical, largest difference 0 |
| 3D hits against the CPU backend | The same four molecules |
| 3D `match` and `numMatch` per row | Identical on every row |
| Hits lost to the prefilter | 0; it settled only the unparsable row |

Wall time on ten molecules was 0.1 s on the CPU against 1.8 s (2D) and 2.1 s (3D) on the GPU: CUDA
start-up is the whole cost at that size. How the GPU path behaves on a library large enough to pay
for itself is still unmeasured.

On the GPR17 antagonist set (44 molecules), 2D similarity to SAR-2 ranks all 12 selective
meta-substituted compounds above every non-selective one; ROC AUC 0.931 for selective against
non-selective. The separation is driven by the linker class and by meta/para, not by selectivity
as such: the carboxamide meta/para pair has *identical* fingerprints, so 2D cannot tell them
apart at all with those bins.

## Decisions worth not relitigating

- **The timeout kills a process.** Nothing else can stop work inside RDKit's C++ layer. See
  `docs/reliability.md`.
- **GPU 3D cannot use constrained embedding.** The GPU embedder takes no bounds matrix and no
  coordinate map, which is how a pharmacophore constraint is expressed. Hence the two-stage
  design, and hence the rule that the prefilter may only remove work.
- **GPU 2D offloads only the similarity matrix.** Building a 2D pharmacophore fingerprint is RDKit
  graph work. The gain is bounded by that, and the README says so.
- **`--timeout` does not cover the GPU prefilter.** Documented rather than fixed; a per-molecule
  interruption point inside a batched device call would defeat the batching.
- **Library cells are read as text.** Settled after a review finding: type inference turned
  `00123` into `123` and `NA` into a blank.
- **`None` in a pass-through column.** Raised as a defect, then withdrawn: `None` and a blank both
  mean "not applicable" here, and the proposed fix introduced a worse hazard (a blank SMILES cell
  becoming RDKit's atomless molecule). The blank-cell guard was kept; the rest was not.

## Things the code refuses to do, on purpose

- Guess at a molecule. A cell that does not describe one is reported.
- Fall back to the CPU when `--backend gpu` was asked for. Exit code 2 instead; nvMolKit has no
  CPU fallback of its own, and a run that asked for a device should not quietly become a CPU run.
- Accept a feature family the fdef does not define. A misspelt family matches nothing, which looks
  exactly like a library in which nothing matched.
- Accept a non-finite feature coordinate.
- Delete a state directory it cannot show it created.
- Report a pose count above `--max-poses × --conformers`.

## CI notes

- `lint` runs `ruff` and `scripts/check_readme_pair.py`.
- `test` runs on Ubuntu and macOS for Python 3.11, 3.12 and 3.13. Both platforms matter: the start
  method differs (fork on Linux, spawn on macOS) and the worker pool is the part most sensitive
  to it.
- `docker` builds the CPU image and smoke-tests it. The GPU image is not built: it needs
  `linux/amd64` and its wheels are large.

## What to pick up next

1. **Measure the GPU path at a size where it could pay.** The agreement runs were ten molecules,
   where CUDA start-up is the whole cost. The open question is the crossover: at what library size
   does each mode beat the CPU, and for 2D whether fingerprint generation dominates so thoroughly
   that the device never does. That decides whether the 2D GPU path is worth keeping.
2. **Run `pytest -m gpu` on a device.** The three skipped tests have not run anywhere; the
   agreement above was checked through the CLI instead.
3. **Decide the GPU image's base.** It builds on `nvidia/cuda:12.9.1-runtime-ubuntu24.04` and
   worked on a compute-capability 12.0 card, which is newer than the toolkit the wheels were built
   with. That is worth a note if a newer nvMolKit changes its pinning.
