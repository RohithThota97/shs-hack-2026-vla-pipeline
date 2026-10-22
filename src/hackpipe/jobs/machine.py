"""What this machine is: CPU, RAM, free disk, GPU (from nvidia-smi). Shown next to every job."""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

from hackpipe.common.config import REPO_ROOT
from hackpipe.common.writelock import machine_id


def gpus() -> list[dict[str, Any]]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    rows = []
    for line in out.stdout.strip().splitlines():
        name, total, used, util = [x.strip() for x in line.split(",")]
        rows.append({"name": name, "memory_total_mib": int(total), "memory_used_mib": int(used),
                     "utilization_pct": int(util)})
    return rows


def ram_gib() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def info() -> dict[str, Any]:
    disk = shutil.disk_usage(REPO_ROOT)
    g = gpus()
    return {"machine": machine_id(), "cpus": os.cpu_count(), "ram_gib": ram_gib(),
            "disk_free_gib": round(disk.free / 2**30, 1), "gpus": g,
            "label": f"{machine_id()} · " + (f"{g[0]['name']} {g[0]['memory_total_mib'] // 1024} GB" if g
                                             else "CPU only")}
