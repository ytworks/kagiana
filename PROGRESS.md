# Progress

Cross-session memory for kagiana: what exists, what is measured, what is not done, and what to
pick up next. Read this first, then `CLAUDE.md`.

Last updated: 2026-11-28. Branch `feat/pharmacophore-screening-cli`, draft PR #1.
126 tests pass locally; 3 skip without a CUDA device.

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
| GPU 2D | Written, never run. Fingerprints stay on the CPU; the similarity matrix goes to the device |
| GPU 3D | Written, never run. Conformer ensemble as a prefilter, survivors to the CPU pass |
| Docker CPU | Built and run, including inside the container |
| Docker GPU | Written, never built |

## What is measured

50,000 synthetic molecules, 12 cores, identical results at every job count:

| Mode | `--jobs 1` | `--jobs 4` | `--jobs 12` |
|---|---|---|---|
| `pharm3d` | 24.8 | 8.0 | 4.0 |
| `pharm2d` | 4.3 | 1.9 | 1.3 |

Seconds. Reproduce with `kagiana pharm3d --library <50k rows> --large --chunksize 5000 --jobs N`.

Peak resident memory in `--large` mode follows the chunk width, not the hit count: 100 MB at
`--chunksize 2000` and 122 MB at `--chunksize 25000`, for a library with 14,539 hits.

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

1. **Run the GPU path.** It has never executed. Needs a Linux x86_64 host with an NVIDIA GPU
   (compute capability 7.0+, driver 560.28+). The steps are in the READMEs' verification section;
   `uv run pytest -m gpu` compares GPU scores against the CPU backend.
2. **Build the GPU image.** `docker build --platform linux/amd64 -f docker/Dockerfile.gpu`.
   Dependency resolution for that platform was checked (nvmolkit 0.7.0 pins `rdkit==2026.3.6`,
   torch 2.15.0) but nothing was built.
3. **Decide whether the 2D GPU path is worth keeping.** If fingerprint generation dominates at
   realistic library sizes, the device buys little; that is a measurement, not a guess.
