"""Report which screening backends this machine can run."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import Any

from .. import __version__
from ..backends import probe_gpu
from ..pool import default_jobs, preferred_start_method


def collect() -> dict[str, Any]:
    import rdkit

    probe = probe_gpu()
    return {
        "kagiana_version": __version__,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "rdkit_version": rdkit.__version__,
        "cpu": {
            "available": True,
            "default_jobs": default_jobs(),
            "start_method": preferred_start_method(),
        },
        "gpu": probe.to_dict(),
    }


def run(args: argparse.Namespace) -> int:
    info = collect()
    if args.json:
        json.dump(info, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print(f"kagiana {info['kagiana_version']} on {info['platform']}")
    print(f"  python  {info['python_version']}")
    print(f"  rdkit   {info['rdkit_version']}")
    cpu = info["cpu"]
    print(
        f"  cpu     available (default --jobs {cpu['default_jobs']}, "
        f"start method {cpu['start_method']})"
    )
    gpu = info["gpu"]
    if gpu["available"]:
        names = ", ".join(gpu["device_names"]) or "unnamed device"
        print(f"  gpu     available ({gpu['device_count']} device(s): {names})")
        print(f"          torch {gpu['torch_version']}, cuda {gpu['cuda_version']}")
        print(f"          nvmolkit {gpu['nvmolkit_version']}")
    else:
        print("  gpu     unavailable")
        for reason in gpu["reasons"]:
            print(f"          - {reason}")
    return 0
