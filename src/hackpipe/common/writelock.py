"""Registry writer lock — one machine writes the registry at a time.

The registry is a SQLite file that may sit on a synced Google Drive folder used by the laptop and by Colab.
Two machines writing it at once can corrupt it, so a machine claims the writer role (a JSON file next to the
registry) and keeps it until it releases it or another machine takes over on purpose. No timeouts: a training run
on Colab can last hours without touching the registry.
"""
from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

from hackpipe.common.config import project_config, resolve
from hackpipe.common.ids import now_iso


class WriterLockError(RuntimeError):
    pass


def machine_id() -> str:
    """'colab' on any Colab session (hostnames change per session), else the host name; HACKPIPE_MACHINE wins."""
    if os.environ.get("HACKPIPE_MACHINE"):
        return os.environ["HACKPIPE_MACHINE"]
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") is not None:
        return "colab"
    return platform.node() or "unknown"


def lock_path() -> Path:
    reg = resolve(project_config()["project"]["registry_path"])
    return reg.with_name(reg.name + ".writer.json")


def holder() -> dict[str, Any] | None:
    p = lock_path()
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (OSError, ValueError):
        return None


def acquire(force: bool = False, who: str | None = None) -> dict[str, Any]:
    """Claim (or keep) the writer role for this machine. Raises if another machine holds it, unless force."""
    me = who or machine_id()
    cur = holder()
    if cur and cur.get("machine") != me and not force:
        raise WriterLockError(f"The registry writer is '{cur['machine']}' (since {cur.get('since')}). "
                              "Release it there, or take over on purpose (UI button / --force).")
    rec = cur if cur and cur.get("machine") == me else {"machine": me, "since": now_iso()}
    if cur and cur.get("machine") != me:
        rec["took_over_from"] = cur.get("machine")
    p = lock_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2))
    return rec


def release(who: str | None = None) -> bool:
    cur = holder()
    if cur and cur.get("machine") == (who or machine_id()):
        lock_path().unlink(missing_ok=True)
        return True
    return False
