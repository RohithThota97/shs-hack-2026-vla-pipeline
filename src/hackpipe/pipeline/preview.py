"""What the console shows before running the pipeline on a folder: what's in it, what's new, expected time."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hackpipe.common.config import REPO_ROOT, source_key
from hackpipe.common.lerobot_io import DatasetFolder
from hackpipe.common.registry import Registry
from hackpipe.common.writelock import machine_id
from hackpipe.jobs.estimates import estimate
from hackpipe.pipeline.sources import SEARCH_DIRS, find_roots



def candidate_folders(base: Path = REPO_ROOT) -> list[str]:
    """LeRobot dataset folders (meta/info.json) anywhere below data/incoming and data/raw (nested ones too,
    e.g. data/incoming/fr3_duo_rand500/fr3_duo_rand500)."""
    return sorted({source_key(r) for d in SEARCH_DIRS for r in find_roots(base / d)})


def new_folders(registry: Registry | None = None, base: Path = REPO_ROOT, folders: list[str] | None = None,
                errors: list[str] | None = None) -> list[dict[str, Any]]:
    """Folders with episodes the registry hasn't seen (or hasn't checked yet) — what "Add all new" runs on.
    Folders that can't be read are skipped and, if `errors` is given, listed there."""
    reg = registry or Registry()
    out = []
    for f in candidate_folders(base) if folders is None else folders:
        try:
            p = preview(f, reg)
        except (FileNotFoundError, ValueError, KeyError, OSError) as e:
            if errors is not None:
                errors.append(f"{f}: {type(e).__name__}: {e}")
            continue
        if p["new"] or p["pending_qc"]:
            out.append(p)
    return out


def preview(root: str, registry: Registry | None = None) -> dict[str, Any]:
    reg = registry or Registry()
    ds = DatasetFolder(root)
    src = source_key(root)
    n = int(ds.info.get("total_episodes", 0))
    eps = ds.episodes_meta()
    lengths = ({int(r["episode_index"]): int(r["length"]) for _, r in eps.iterrows()} if len(eps)
               else {i: int(ds.info.get("total_frames", 0)) // max(n, 1) for i in range(n)})
    registered = {d["episode_index"]: d for d in reg.demos(source=src)}
    new = [i for i in lengths if i not in registered]
    pending = [i for i, d in registered.items() if d["status"] == "imported"]
    cams = len(ds.video_keys) or 1
    decode_units = sum(lengths[i] for i in new + pending) * cams
    est = estimate("auto", decode_units, reg, machine_id()) if decode_units else {"seconds": 0, "basis": "nothing to check"}
    return {
        "source": src, "episodes": n, "frames": int(ds.info.get("total_frames", 0)), "fps": ds.fps,
        "cameras": [k.rsplit(".", 1)[-1] for k in ds.video_keys], "tasks": sorted(ds.tasks().values()),
        "robot_type": ds.robot_type, "codebase_version": ds.info.get("codebase_version"),
        "new": len(new), "registered": len(registered), "pending_qc": len(pending),
        "work_units": decode_units, "estimate": est, "needs": "cpu",
    }
