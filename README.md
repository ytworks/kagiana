# kagiana

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)

**English** | [日本語](README.ja.md)

kagiana screens a SMILES library against a pharmacophore, in two modes: 2D fingerprint similarity
to one reference molecule, and 3D matching by constrained embedding. The command line is built for
programs rather than people: non-interactive, self-describing, with stable exit codes.

Give it a CSV with a SMILES column and it appends score columns, leaving every other column exactly
as it found it. Screening runs across CPU cores, no single molecule can stall a run, and a CUDA
path is available for the heavy parts.

> **Status: version 0.1, in development.** Both the CPU and the GPU paths have been run; see
> [Verification status](#verification-status) for what was measured on which hardware.

- [Features](#features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Input files](#input-files)
- [Output files](#output-files)
- [CLI reference](#cli-reference)
- [Parallelism and timeouts](#parallelism-and-timeouts)
- [Resuming](#resuming)
- [GPU acceleration](#gpu-acceleration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Verification status](#verification-status)
- [License](#license)

## Features

- **Two screening modes.** 2D pharmacophore fingerprints, and 3D matching by constrained
  embedding.
- **A deadline per molecule.** A molecule whose embedding fails to converge is abandoned and the
  run continues.
- **Multicore.** Molecules are screened across worker processes, with results in input order and
  identical whatever the job count.
- **Resumable.** An interrupted run picks up from what it recorded as finished, not from whichever
  files happen to exist.
- **Built for programs.** Data in CSV, progress on stderr, a status on every row, a reason for
  every failure, a machine-readable run summary and meaningful exit codes.
- **Your columns are yours.** Cells this tool does not score are passed through as written.

## How it works

**2D.** The reference molecule and each library molecule become 2D pharmacophore signatures. Three
similarity scores come from a single intersection count, so they always agree with one another; a
row is kept when it clears all three cutoffs.

**3D.** For each molecule: check that it carries the required feature types at all, derive its
distance-bounds matrix, enumerate the feature-to-atom assignments whose distances fit the query,
and try to embed a conformer under each. The score is the number of poses embedded.

Every molecule is screened in a worker process under a deadline. The heavy part of 3D screening
happens inside RDKit's C++ layer, where a Python-level interrupt never arrives, so the parent
kills the worker outright when its deadline passes and starts a replacement.

## Requirements

- Python 3.11 or newer
- RDKit, pandas, numpy, tqdm (installed with the package)
- For the CUDA path: Linux x86_64, an NVIDIA GPU of compute capability 7.0 or newer, driver 560.28
  or newer, and a CUDA build of PyTorch

## Installation

```bash
uv venv --python 3.11
uv pip install -e .
```

For the test suite, install the dev extra instead:

```bash
uv pip install -e ".[dev]"
```

For the CUDA path, on a machine that has one:

```bash
uv pip install --torch-backend=cu128 -e ".[gpu]"
```

Check what the machine can do:

```bash
kagiana backends --json
```

### Docker

```bash
docker build -f docker/Dockerfile.cpu -t kagiana:cpu .
docker run --rm \
    -v "$PWD/data:/work/data" -v "$PWD/results:/work/results" \
    kagiana:cpu pharm3d \
        --library /work/data/library.csv \
        --pharmacophore /work/data/pharmacophore.csv \
        --fdef /work/data/features.fdef \
        --out /work/results/pharm3d.csv
```

The GPU image needs an x86_64 host with the NVIDIA Container Toolkit:

```bash
docker build --platform linux/amd64 -f docker/Dockerfile.gpu -t kagiana:gpu .
docker run --rm --gpus all kagiana:gpu backends
```

Or with Compose, which mounts `./data` and `./results`:

```bash
docker compose -f docker/docker-compose.yml --profile cpu run --rm cpu backends
docker compose -f docker/docker-compose.yml --profile gpu run --rm gpu backends
```

## Quickstart

```bash
# Check the inputs first; nothing is screened
kagiana validate --library data/library.csv --fdef settings/features.fdef \
    --pharmacophore data/pharmacophore.csv --json

# 2D: score a library against one reference molecule
kagiana pharm2d \
    --library data/library.csv \
    --query data/reference.csv \
    --fdef settings/features.fdef \
    --params data/params.json \
    --out results/pharm2d.csv

# 3D: count the poses each molecule can adopt that satisfy the query
kagiana pharm3d \
    --library data/library.csv \
    --pharmacophore data/pharmacophore.csv \
    --fdef settings/features.fdef \
    --out results/pharm3d.csv \
    --jobs 8
```

## Input files

### Library CSV

A `SMILES` column is required. Every other column is carried through to the output untouched: cells
are read as text, so an identifier of `00123` does not come back as `123` and a note of `NA` does
not come back empty.

```csv
ID,SMILES,vendor_note
LIB-001,NCCc1ccc(O)cc1,dopamine-like
LIB-002,NCCc1ccccc1,phenethylamine
```

### Reference molecule CSV

For `pharm2d --query`: a `SMILES` column, of which **the first row** is the reference molecule.
Later rows are ignored.

### Feature-point CSV

For `pharm3d --pharmacophore`: one feature point per row, with the columns `feature_type`, `X`,
`Y`, `Z`. `feature_type` must be a feature family the fdef file defines; a misspelt one is an input
error rather than a run that reports no hits.

```csv
feature_type,X,Y,Z
Donor,-3.606,0.408,0.856
Acceptor,4.101,0.237,-0.398
Aromatic,1.412,-0.215,-0.031
```

Pairwise distance bounds come from these coordinates: for every pair at reference distance `d`, the
lower bound is `max(0, d - limit_lower)` and the upper bound is `d + limit_upper`. **All** feature
points must be satisfied.

### params JSON

For `pharm2d --params`. All seven keys are required; there are no hidden fallbacks.

```json
{
  "n_maxpoint": 3,
  "bins": [[0, 2], [2, 5], [5, 8]],
  "skip_feature": ["ZnBinder"],
  "chunksize": 100000,
  "tanimoto_cutoff": 0.6,
  "simpson_cutoff": 0.6,
  "dice_cutoff": 0.6
}
```

`n_maxpoint` is the largest feature-point count in a signature, and the smallest is always 2.
`bins` are topological distance bins. `skip_feature` names families to leave out. `chunksize` is
how many library rows are read at a time. A row is kept only when it clears **all three** cutoffs.

The three cutoffs and `chunksize` can be overridden on the command line; the signature parameters
cannot, so a run's fingerprints always match its params file.

## Output files

### Columns

Score columns are appended to the input columns, followed by diagnostic columns.

| Mode | Score columns | Diagnostic columns |
| --- | --- | --- |
| `pharm2d` | `tanimoto`, `dice`, `simpson` | `status`, `elapsed_s` |
| `pharm3d` | `match`, `numMatch` | `status`, `elapsed_s`, `stage`, `best_rmsd` |

`match` says whether the molecule carries the required feature types at all; `numMatch` is how many
satisfying poses were embedded. `stage` records which path produced the row, and `best_rmsd` the
GPU prefilter's best rigid fit.

### Row status

| Value | Meaning |
| --- | --- |
| `ok` | Scored normally |
| `no_feature` | Screened fine; the molecule does not carry the required features |
| `parse_error` | The cell does not describe a molecule: unparsable, or blank |
| `timeout` | The molecule exceeded `--timeout` and its worker was killed |
| `error` | Scoring raised; the reason is in the diagnostic log |

Rows that did not score cleanly are also written to `<out>_errors.jsonl`, one JSON object per row
with its chunk, row, SMILES and the reason. The result CSV keeps its schema; the explanation lives
beside it.

### File naming

`pharm2d` always reads the library in chunks of `chunksize`.

| File | Contents |
| --- | --- |
| `<out>.csv` | The rows that cleared all three cutoffs |
| `<out>_all.csv` | Every row with its scores (suppress with `--no-all-rows`) |

`pharm3d` in whole-file mode, the default, reads the library in one go.

| File | Contents |
| --- | --- |
| `<out>.csv` | Every row with its scores |
| `<out>_screened.csv` | The rows with `numMatch > 0` |

`pharm3d --large` reads the library in chunks of `--chunksize` and writes as it goes, so an
interrupted run leaves usable partial results.

| File | Contents |
| --- | --- |
| `<out>_<i>.csv` | Every row of chunk `i` |
| `<out>_screened_<i>.csv` | The hits of chunk `i` |
| `<out>_screened_all.csv` | Every hit of the run |

## CLI reference

```text
kagiana pharm2d   --library L.csv --query Q.csv --fdef F.fdef --out R.csv --params params.json
                  [--tanimoto-cutoff F] [--dice-cutoff F] [--simpson-cutoff F] [--chunksize N]
                  [--jobs N] [--backend auto|cpu|gpu] [--no-all-rows]
                  [--resume] [--json-summary S.json] [--quiet]

kagiana pharm3d   --library L.csv --pharmacophore P.csv --fdef F.fdef --out R.csv
                  [--large] [--chunksize N]
                  [--conformers N] [--max-poses N] [--limit-upper F] [--limit-lower F]
                  [--timeout SEC] [--jobs N]
                  [--backend auto|cpu|gpu] [--prefilter-rmsd F] [--prefilter-conformers N]
                  [--resume] [--json-summary S.json] [--quiet]

kagiana backends  [--json]
kagiana validate  --library L.csv [--fdef F.fdef] [--pharmacophore P.csv]
                  [--params P.json] [--query Q.csv] [--json]
```

Data goes to the output CSV and progress to stderr, so a caller can parse the results without
filtering noise. `--json-summary` writes the counts, the backend used and the files produced.

### Defaults

`pharm3d` ships the settled working values.

| Option | Default | Meaning |
| --- | --- | --- |
| `--conformers` | 10 | Conformers requested per feature assignment |
| `--max-poses` | 20 | Multiplied by `--conformers` for the per-molecule ceiling |
| `--limit-upper` | 1.5 | Ångström added to each reference distance |
| `--limit-lower` | 0.5 | Ångström subtracted from each reference distance |
| `--timeout` | 10 | Seconds a molecule may take in the constrained-embedding stage |
| `--chunksize` | 1000000 | Rows per chunk, with `--large` |

`--jobs` defaults to every usable core and `--backend` to `auto`. No input or output path has a
default: a missing one is an input error rather than a write to somewhere unexpected.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | The run finished |
| 1 | A runtime failure |
| 2 | Bad input: a missing file, a malformed CSV or params file, a failed `validate`, or an unusable backend that was explicitly requested |

## Parallelism and timeouts

Molecules are screened across worker processes. The heavy part of 3D screening happens inside
RDKit's C++ layer, where a Python-level interrupt never arrives, so a molecule whose
distance-geometry embedding fails to converge cannot be stopped from inside the worker. Instead the
parent tracks when each molecule was dispatched and **kills the worker outright** when `--timeout`
passes, then starts a replacement. The run continues, and the row is marked `timeout`, keeping the
poses already embedded plus one for the assignment that was interrupted.

A molecule that finishes after its deadline is reported as a timeout too, whether the delay was in
the work, in the worker starting up, or in the parent reading the result.

**What the limit covers.** `--timeout` applies to the constrained-embedding stage. The GPU
prefilter is not covered: it generates conformers for a whole batch in one device call, and a
per-molecule interruption point inside that call would mean giving up the batching that makes it
worth using. [docs/reliability.md](docs/reliability.md) explains the design and the ways it has
been got wrong.

### Measured scaling

50,000 molecules on a 12-core machine, in seconds, with results identical at every job count:

| Mode | `--jobs 1` | `--jobs 4` | `--jobs 12` |
| --- | --- | --- | --- |
| `pharm3d` | 24.8 | 8.0 | 4.0 |
| `pharm2d` | 4.3 | 1.9 | 1.3 |

Workers are started per chunk, and on platforms that spawn rather than fork each one re-imports
RDKit. For a library of a few thousand molecules that startup can cost more than the screening
itself, so `--jobs 1` is sometimes the faster choice there; the gain shows up once a chunk holds
tens of thousands of molecules.

## Resuming

`--resume` skips only the work *this* combination of inputs and settings recorded as finished.
Output files on their own cannot say which run produced them, so a run keeps a small record in a
hidden `.<out>_state/` directory beside its results: the inputs it ran on, the chunks it completed,
and the diagnostics of each.

Screening a different library into the same output directory therefore starts over instead of
inheriting half of the earlier results, and a chunk counts as finished only once its diagnostics
and all of its CSVs are on disk. Every result file is written to a temporary name and moved into
place, so a file under its final name is complete.

## GPU acceleration

The CUDA path uses [nvMolKit](https://github.com/NVIDIA-BioNeMo/nvMolKit). It pins the RDKit
version it was built against.

Verified on an NVIDIA RTX PRO 6000 Blackwell (compute capability 12.0) with nvMolKit 0.7.0 and
torch 2.11.0+cu128: the 2D scores match the CPU backend exactly, and the 3D path finds the same
hits. The GPU is slower on a ten-molecule library, where CUDA start-up dominates; it is meant for
libraries large enough for that to disappear.

`--backend auto` uses the GPU when one is usable and falls back to the CPU with a note on stderr
otherwise. `--backend gpu` fails with exit code 2 instead of falling back, because nvMolKit has no
CPU fallback of its own and a run that asked for a device should not quietly become a CPU run.

What each mode offloads:

- **2D** — building a pharmacophore fingerprint is RDKit graph work and stays on the CPU workers;
  the GPU takes the all-against-one bit comparison. The speed-up is therefore bounded by
  fingerprint generation and matters most on large libraries.
- **3D** — constrained embedding cannot move to the GPU: the GPU embedder takes no bounds matrix
  and no coordinate map, which is exactly how a pharmacophore constraint is expressed. So the GPU
  path screens in two stages. It first builds an unconstrained conformer ensemble on the device and
  measures the best rigid superposition of each molecule's features onto the query points;
  molecules that cannot come close are rejected. Survivors then go through the same
  constrained-embedding pass the CPU backend uses, so `match` and `numMatch` keep their meaning.

The prefilter only ever removes work. It settles a molecule only when it measured a fit for every
query feature and that fit is clearly too far off; such a row reports `match=True`, `numMatch=0`
and `stage=gpu_prefilter_reject`, because the molecule does carry the features and it is the
geometry that failed. Anything it could not measure — no usable conformer, a missing feature
family, or an assignment search that hit its budget — is passed to the constrained stage rather
than decided. `--prefilter-rmsd` controls how aggressive the first stage is.

## Troubleshooting

**Many `pharm2d` rows come back as `error` with "distance bin not found".** This is a limitation of
RDKit's 2D signature factory rather than a problem with the library. A feature-to-feature distance
is measured between feature *atom sets*, which can produce distance triples that a triangle
inequality forbids, and the factory prunes exactly those bin combinations when it is built. Widening
the bins, reducing `n_maxpoint` to 2, or skipping multi-atom feature families via `skip_feature` all
reduce it. The run itself completes either way: the affected rows are flagged and explained in
`<out>_errors.jsonl`.

**`--backend gpu` exits with code 2.** The GPU path is unavailable; run `kagiana backends --json`
to see the reasons. Use `--backend auto` to screen on the CPU instead.

**A state directory collision is reported.** The hidden record `--resume` depends on takes its name
from the output path. If a directory of yours is already there, kagiana reports it instead of
deleting it; move it aside or choose another `--out`.

## Development

```bash
uv pip install -e ".[dev]"
uv run pytest
uv run ruff check .
python scripts/check_readme_pair.py
```

[PROGRESS.md](PROGRESS.md) is the cross-session record of what is done, what is measured and what
to pick up next. [CLAUDE.md](CLAUDE.md) holds the rules this project is built to.

Both READMEs change together: `scripts/check_readme_pair.py` compares the heading outline, the
commands in code blocks, the links and the numbers in tables, and CI runs it.

## Verification status

The CPU paths are covered by the test suite, run in CI on Ubuntu and macOS for Python 3.11 to 3.13,
and were exercised on real inputs, including inside the CPU container image.

The GPU paths have now been run, on one NVIDIA RTX PRO 6000 Blackwell (compute capability 12.0,
driver 595.99.02) with nvMolKit 0.7.0, torch 2.11.0+cu128 and RDKit 2026.03.6. The GPU image built
and `backends` reported the device, which also answers whether nvMolKit's published wheels work on
a GPU generation newer than the CUDA they were built with: they do.

| Check | Result |
| --- | --- |
| 2D scores against the CPU backend | Identical; largest difference 0 |
| 3D hits against the CPU backend | The same four molecules |
| 3D `match` and `numMatch` per row | Identical on every row |
| Hits lost to the prefilter | 0 |

The prefilter settled only the unparsable row, before any conformer was generated, and deferred
every molecule it could not measure — which is the invariant it is built to.

Reproduce with:

```bash
docker build -f docker/Dockerfile.gpu -t kagiana:gpu .
docker run --rm --gpus "device=0" kagiana:gpu backends --json
uv run pytest -m gpu
```

Still unmeasured: how the GPU path behaves on a library large enough for the device to pay for
itself. The runs above were ten molecules, where CUDA start-up is the whole cost.

## License

Apache-2.0. See [LICENSE](LICENSE).
