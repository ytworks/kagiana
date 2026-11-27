"""Screening backends and the runtime capability probe."""

from __future__ import annotations

import importlib
import platform
from dataclasses import dataclass, field
from typing import Any

CPU = "cpu"
GPU = "gpu"
AUTO = "auto"
BACKEND_CHOICES = (AUTO, CPU, GPU)


@dataclass
class GpuProbe:
    """What the GPU path needs, and whether this machine has it."""

    available: bool = False
    reasons: list[str] = field(default_factory=list)
    torch_version: str | None = None
    nvmolkit_version: str | None = None
    cuda_version: str | None = None
    device_count: int = 0
    device_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reasons": self.reasons,
            "torch_version": self.torch_version,
            "nvmolkit_version": self.nvmolkit_version,
            "cuda_version": self.cuda_version,
            "device_count": self.device_count,
            "device_names": self.device_names,
        }


def probe_gpu() -> GpuProbe:
    """Report GPU readiness without raising.

    The GPU extension has no CPU fallback of its own, so the CLI has to decide
    up front whether the GPU path can run at all.
    """
    probe = GpuProbe()

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # noqa: BLE001 - any import problem is a reason
        probe.reasons.append(f"torch is not importable: {exc}")
        torch = None
    else:
        probe.torch_version = getattr(torch, "__version__", None)
        probe.cuda_version = getattr(getattr(torch, "version", None), "cuda", None)

    try:
        nvmolkit = importlib.import_module("nvmolkit")
    except Exception as exc:  # noqa: BLE001
        probe.reasons.append(f"nvmolkit is not importable: {exc}")
    else:
        probe.nvmolkit_version = getattr(nvmolkit, "__version__", "unknown")

    if torch is not None:
        try:
            if torch.cuda.is_available():
                probe.device_count = int(torch.cuda.device_count())
                probe.device_names = [
                    str(torch.cuda.get_device_name(i)) for i in range(probe.device_count)
                ]
            else:
                probe.reasons.append("no CUDA device is visible to torch")
        except Exception as exc:  # noqa: BLE001
            probe.reasons.append(f"querying CUDA devices failed: {exc}")

    if platform.system() != "Linux":
        probe.reasons.append(
            f"the GPU extension ships Linux builds only; this host is {platform.system()}"
        )

    probe.available = not probe.reasons and probe.device_count > 0
    return probe


def resolve_backend(requested: str) -> tuple[str, GpuProbe, list[str]]:
    """Turn ``--backend`` into the backend that will actually run.

    Returns the chosen backend, the probe, and notes worth showing the user.
    Requesting ``gpu`` explicitly on a machine without one is an input error,
    reported by the caller; ``auto`` falls back to the CPU path instead.
    """
    notes: list[str] = []
    if requested == CPU:
        return CPU, GpuProbe(reasons=["not probed: --backend cpu"]), notes

    probe = probe_gpu()
    if requested == GPU:
        return GPU, probe, notes
    if probe.available:
        notes.append(
            f"backend auto: using the GPU path ({probe.device_count} device(s))"
        )
        return GPU, probe, notes
    notes.append(
        "backend auto: falling back to the CPU path ("
        + "; ".join(probe.reasons)
        + ")"
    )
    return CPU, probe, notes
