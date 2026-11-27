"""GPU-path tests.

These are skipped unless the machine actually has CUDA, torch and nvmolkit, so
the suite stays runnable where the GPU extension cannot be installed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kagiana.backends import probe_gpu

pytestmark = pytest.mark.gpu

gpu_required = pytest.mark.skipif(
    not probe_gpu().available, reason="no usable CUDA GPU with torch and nvmolkit"
)


@gpu_required
def test_the_probe_reports_a_device() -> None:
    probe = probe_gpu()
    assert probe.device_count >= 1
    assert probe.torch_version
    assert probe.nvmolkit_version


@gpu_required
def test_gpu_2d_scores_agree_with_the_cpu_backend(
    tmp_path: Path, library: Path, query: Path, fdef: Path, params: Path
) -> None:
    from kagiana.cli import main

    common = [
        "--library", str(library), "--query", str(query),
        "--fdef", str(fdef), "--params", str(params),
        "--tanimoto-cutoff", "0.0", "--dice-cutoff", "0.0",
        "--simpson-cutoff", "0.0", "--quiet",
    ]
    assert main(["pharm2d", *common, "--out", str(tmp_path / "cpu.csv"), "--backend", "cpu"]) == 0
    assert main(["pharm2d", *common, "--out", str(tmp_path / "gpu.csv"), "--backend", "gpu"]) == 0

    cpu = pd.read_csv(tmp_path / "cpu_all.csv")
    gpu = pd.read_csv(tmp_path / "gpu_all.csv")
    assert list(cpu["status"]) == list(gpu["status"])
    for column in ("tanimoto", "dice", "simpson"):
        pd.testing.assert_series_equal(
            cpu[column], gpu[column], check_exact=False, atol=1e-9
        )


@gpu_required
def test_gpu_3d_keeps_the_hits_the_cpu_backend_finds(
    tmp_path: Path, library: Path, fdef: Path, pharmacophore: Path
) -> None:
    """The prefilter may only discard molecules the CPU pass would reject."""
    from kagiana.cli import main

    common = [
        "--library", str(library),
        "--pharmacophore", str(pharmacophore),
        "--fdef", str(fdef),
        "--quiet",
    ]
    assert main(["pharm3d", *common, "--out", str(tmp_path / "cpu.csv"), "--backend", "cpu"]) == 0
    assert main(
        ["pharm3d", *common, "--out", str(tmp_path / "gpu.csv"), "--backend", "gpu",
         "--prefilter-rmsd", "2.5"]
    ) == 0

    cpu = pd.read_csv(tmp_path / "cpu.csv", index_col=0)
    gpu = pd.read_csv(tmp_path / "gpu.csv", index_col=0)
    cpu_hits = set(cpu[cpu["numMatch"] > 0]["ID"])
    gpu_hits = set(gpu[gpu["numMatch"] > 0]["ID"])
    assert gpu_hits <= cpu_hits
    assert gpu["best_rmsd"].notna().any()
