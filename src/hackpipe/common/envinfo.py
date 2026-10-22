"""Capture code version and training environment for run records."""
from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any

from hackpipe.common.config import REPO_ROOT


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=REPO_ROOT)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_info() -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def gpu_info() -> str | None:
    return _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])


def pip_freeze() -> list[str]:
    out = _run([sys.executable, "-m", "pip", "freeze"])
    return out.splitlines() if out else []


def environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": gpu_info(),
        "git": git_info(),
        "packages": pip_freeze(),
    }
