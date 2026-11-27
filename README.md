# kagiana

Pharmacophore screening over a SMILES library, in two flavours:

- **`pharm2d`** — 2D pharmacophore fingerprints compared against one reference
  molecule (Tanimoto, Dice, Simpson).
- **`pharm3d`** — 3D pharmacophore matching by constrained embedding, scored by
  the number of poses a molecule can adopt that satisfy the query.

It is built to be driven by programs as much as by people: no prompts, data in
CSV, progress and warnings on stderr, a status for every row, a machine-readable
run summary, and meaningful exit codes. Screening runs across CPU cores, no
molecule can stall a run, and a CUDA path is available for the heavy parts.

## Install

Python 3.11 or newer.

```bash
uv venv --python 3.11
uv pip install -e .            # add ".[dev]" for the test suite
```

CUDA path (Linux x86_64 with an NVIDIA GPU; see [GPU acceleration](#gpu-acceleration)):

```bash
uv pip install --torch-backend=cu128 ".[gpu]"
```

Check what the machine can do:

```bash
kagiana backends --json
```

## Quick start

```bash
# 2D: score a library against one reference molecule
kagiana pharm2d \
    --library data/library.csv \
    --query data/reference.csv \
    --fdef settings/features.fdef \
    --params data/params.json \
    --out results/pharm2d.csv

# 3D: count satisfying poses per molecule
kagiana pharm3d \
    --library data/library.csv \
    --pharmacophore data/pharmacophore.csv \
    --fdef settings/features.fdef \
    --out results/pharm3d.csv \
    --jobs 8

# Check the inputs first, without screening anything
kagiana validate --library data/library.csv --fdef settings/features.fdef \
    --pharmacophore data/pharmacophore.csv --json
```

## Input files

### Library CSV (both modes)

A `SMILES` column is required; every other column is carried through to the
output untouched.

```csv
ID,SMILES,vendor_note
LIB-001,NCCc1ccc(O)cc1,dopamine-like
LIB-002,NCCc1ccccc1,phenethylamine
```

### Reference molecule CSV (`pharm2d --query`)

A `SMILES` column, of which **the first row** is the reference molecule. Later
rows are ignored.

### Feature-point CSV (`pharm3d --pharmacophore`)

One feature point per row, with the columns `feature_type`, `X`, `Y`, `Z`.
`feature_type` must be a feature family the fdef file defines.

```csv
feature_type,X,Y,Z
Donor,-3.606,0.408,0.856
Acceptor,4.101,0.237,-0.398
Aromatic,1.412,-0.215,-0.031
```

Pairwise distance bounds come from these coordinates: for every pair at
reference distance `d`, the lower bound is `max(0, d - limit_lower)` and the
upper bound is `d + limit_upper`. **All** feature points must be satisfied.

### params JSON (`pharm2d --params`)

All seven keys are required; there are no hidden fallbacks.

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

| Key | Meaning |
| --- | --- |
| `n_maxpoint` | Largest feature-point count in a signature (the smallest is always 2) |
| `bins` | Topological distance bins, as `[lower, upper]` pairs |
| `skip_feature` | Feature families to leave out of the signature |
| `chunksize` | Library rows read at a time |
| `tanimoto_cutoff`, `simpson_cutoff`, `dice_cutoff` | A row is kept only when it clears **all three** |

The three cutoffs and `chunksize` can be overridden on the command line; the
signature parameters cannot, so a run's fingerprints always match its params
file.

## Output files

Score columns are appended to the input columns, followed by diagnostic
columns:

| Mode | Score columns | Diagnostic columns |
| --- | --- | --- |
| `pharm2d` | `tanimoto`, `dice`, `simpson` | `status`, `elapsed_s` |
| `pharm3d` | `match`, `numMatch` | `status`, `elapsed_s`, `stage`, `best_rmsd` |

`match` says whether the molecule carries the required feature types at all;
`numMatch` is how many satisfying poses were embedded. `stage` records which
path produced the row, and `best_rmsd` the GPU prefilter's best rigid fit
(empty on the CPU path).

### `status` values

| Value | Meaning |
| --- | --- |
| `ok` | Scored normally |
| `no_feature` | Screened fine; the molecule does not carry the required features |
| `parse_error` | RDKit could not build a molecule from the SMILES string |
| `timeout` | The molecule exceeded `--timeout` and its worker was killed |
| `error` | Scoring raised; the reason is in the error log |

Rows that did not score cleanly are also written to `<out>_errors.jsonl`, one
JSON object per row with its chunk, row, SMILES and the reason. The result CSV
keeps its schema; the explanation lives next to it.

### File naming

**`pharm2d`** always reads the library in chunks of `chunksize`:

| File | Contents |
| --- | --- |
| `<out>.csv` | The rows that cleared all three cutoffs (no index column) |
| `<out>_all.csv` | Every row with its scores (suppress with `--no-all-rows`) |

**`pharm3d`**, whole-file mode (the default) reads the library in one go:

| File | Contents |
| --- | --- |
| `<out>.csv` | Every row with its scores |
| `<out>_screened.csv` | The rows with `numMatch > 0` |

**`pharm3d --large`** reads the library in chunks of `--chunksize` and writes as
it goes, so an interrupted run leaves usable partial results:

| File | Contents |
| --- | --- |
| `<out>_<i>.csv` | Every row of chunk `i` |
| `<out>_screened_<i>.csv` | The hits of chunk `i` |
| `<out>_screened_all.csv` | Every hit of the run |

`--resume` skips only the work *this* combination of inputs and settings
recorded as finished. Output files on their own cannot say which run produced
them, so a run keeps a small record in a hidden `.<out>_state/` directory
beside its results: the inputs it ran on, the chunks it completed, and the
diagnostics of each. Screening a different library into the same output
directory therefore starts over instead of inheriting half of the earlier
results, and a chunk counts as finished only once its diagnostics and both of
its CSVs are on disk.

## Defaults

`pharm3d` ships the settled working values:

| Option | Default |
| --- | --- |
| `--conformers` | `10` conformers requested per feature assignment |
| `--max-poses` | `20`, multiplied by `--conformers` for the per-molecule ceiling (so 200) |
| `--limit-upper` | `1.5` Å added to each reference distance |
| `--limit-lower` | `0.5` Å subtracted from each reference distance |
| `--timeout` | `10` s per molecule |
| `--chunksize` | `1000000` rows, with `--large` |
| `--jobs` | every usable core |
| `--backend` | `auto` |

No input or output path has a default: a missing one is an input error rather
than a write to somewhere unexpected.

## Parallelism and timeouts

Molecules are screened across worker processes. The expensive part of 3D
screening happens inside RDKit's C++ layer, where a Python-level interrupt
never arrives, so a molecule whose distance-geometry embedding fails to
converge cannot be stopped from inside the worker. Instead the parent tracks
when each molecule was dispatched and **kills the worker outright** when
`--timeout` passes, then starts a replacement. The run continues, and the row
is marked `timeout`, keeping the poses already embedded plus one for the
assignment that was interrupted. A molecule that finishes after its deadline is
reported as a timeout too, rather than as a success that happened to be late.

**What the limit covers.** `--timeout` applies to the constrained-embedding
stage, which is where screening can fail to converge. The GPU prefilter is not
covered: it generates conformers for a whole batch in one device call, and a
per-molecule interruption point inside that call would mean giving up the
batching that makes it worth using. A prefilter batch therefore runs to
completion even under a short `--timeout`.

### Measured scaling

A 50,000-molecule library on a 12-core machine, with results identical at every
job count:

| Mode | `--jobs 1` | `--jobs 4` | `--jobs 12` |
| --- | --- | --- | --- |
| `pharm3d` | 24.8 s | 8.0 s | 4.0 s |
| `pharm2d` | 4.3 s | 1.9 s | 1.3 s |

Workers are started per chunk, and on platforms that spawn rather than fork
each one re-imports RDKit. For a library of a few thousand molecules that
startup can cost more than the screening itself, so `--jobs 1` is sometimes the
faster choice there; the gain shows up once a chunk holds tens of thousands of
molecules.

## GPU acceleration

The CUDA path uses [nvMolKit](https://github.com/NVIDIA-BioNeMo/nvMolKit).
Requirements: Linux x86_64, an NVIDIA GPU of compute capability 7.0 or newer,
driver 560.28 or newer (CUDA 12.6+), and a CUDA build of PyTorch. nvMolKit pins
the RDKit version it was built against (`rdkit==2026.3.6` for nvmolkit 0.7.0).

`--backend auto` uses the GPU when one is usable and falls back to the CPU with
a note on stderr otherwise. `--backend gpu` fails with exit code 2 instead of
falling back — nvMolKit has no CPU fallback of its own, so a run that asked for
the GPU should not quietly become a CPU run.

What each mode actually offloads:

- **2D** — building a pharmacophore fingerprint is RDKit graph work and stays on
  the CPU workers; the GPU takes the all-against-one bit comparison. The
  speed-up is therefore bounded by fingerprint generation and matters most on
  large libraries.
- **3D** — constrained embedding cannot move to the GPU: the GPU embedder takes
  no bounds matrix and no coordinate map, which is exactly how a pharmacophore
  constraint is expressed. So the GPU path screens in two stages. It first
  builds an unconstrained conformer ensemble (ETKDG + MMFF on the device) and
  measures the best rigid superposition of each molecule's features onto the
  query points; molecules that cannot come close are rejected. Survivors then go
  through the same constrained-embedding pass the CPU backend uses, so
  `match` and `numMatch` keep their meaning. `--prefilter-rmsd` controls how
  aggressive the first stage is — raise it if you suspect it is discarding hits.

  The prefilter only ever removes work. It settles a molecule only when it
  measured a fit for every query feature and that fit is clearly too far off;
  such a row reports `match=True`, `numMatch=0` and
  `stage=gpu_prefilter_reject`, because the molecule does carry the features
  and it is the geometry that failed. Anything it could not measure — no usable
  conformer, a missing feature family, or an assignment search that hit its
  budget — is passed to the constrained stage rather than decided.

## Docker

```bash
# CPU
docker build -f docker/Dockerfile.cpu -t kagiana:cpu .
docker run --rm \
    -v "$PWD/data:/work/data" -v "$PWD/results:/work/results" \
    kagiana:cpu pharm3d \
        --library /work/data/library.csv \
        --pharmacophore /work/data/pharmacophore.csv \
        --fdef /work/data/features.fdef \
        --out /work/results/pharm3d.csv

# GPU (x86_64 host with the NVIDIA Container Toolkit)
docker build --platform linux/amd64 -f docker/Dockerfile.gpu -t kagiana:gpu .
docker run --rm --gpus all kagiana:gpu backends
```

Or with Compose, which mounts `./data` and `./results`:

```bash
docker compose -f docker/docker-compose.yml --profile cpu run --rm cpu backends
docker compose -f docker/docker-compose.yml --profile gpu run --rm gpu backends
```

## Troubleshooting

**Many `pharm2d` rows come back as `error` with "distance bin not found".**
This is a limitation of RDKit's 2D signature factory rather than a problem with
the library. A feature-to-feature distance is measured between feature *atom
sets*, which can produce distance triples that a triangle inequality forbids,
and the factory prunes exactly those bin combinations when it is built. Widening
the bins, reducing `n_maxpoint` to `2`, or skipping multi-atom feature families
(aromatic rings, lumped hydrophobes) via `skip_feature` all reduce it. The run
itself completes either way: the affected rows are flagged and explained in
`<out>_errors.jsonl` rather than aborting the screen.

**`--backend gpu` exits with code 2.** The GPU path is unavailable; run
`kagiana backends --json` to see the reasons. Use `--backend auto` to screen on
the CPU instead.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | The run finished |
| `1` | A runtime failure |
| `2` | Bad input: a missing file, a malformed CSV or params file, a failed `validate`, or an unusable backend that was explicitly requested |

## Tests

```bash
uv run pytest           # GPU tests skip themselves without a CUDA device
uv run pytest -m gpu    # only the GPU tests
```

## Verification status

The CPU paths are covered by the test suite and were exercised on real inputs,
including inside the CPU container image.

**The GPU paths have not been run.** They were developed on macOS arm64, which
cannot host CUDA, and nvMolKit publishes x86_64 Linux wheels only. The
dependency resolution for `linux/amd64` was checked, and the CUDA base image
tag was confirmed to exist, but neither the GPU image build nor
`--backend gpu` itself has been executed. To verify on suitable hardware:

```bash
docker build --platform linux/amd64 -f docker/Dockerfile.gpu -t kagiana:gpu .
docker run --rm --gpus all kagiana:gpu backends --json     # expect "available": true
docker run --rm --gpus all -v "$PWD/tests/data:/work/data:ro" -v "$PWD/results:/work/results" \
    kagiana:gpu pharm2d --library /work/data/library.csv --query /work/data/query.csv \
        --fdef /work/data/test.fdef --params /work/data/params.json \
        --out /work/results/gpu2d.csv --backend gpu
uv run pytest -m gpu    # compares GPU scores against the CPU backend
```

## Licence

See [LICENSE](LICENSE).
