"""What the console shows before running the pipeline on a folder: what's in it, what's new, expected time."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hackpipe.common.config import REPO_ROOT, source_key
from hackpipe.common.lerobot_io import DatasetFolder
from hackpipe.common.registry import Registry
from hackpipe.common.writelock import machine_id
from hackpipe.jobs.estimates import estimate

SEARCH_DIRS = ("data/incoming", "data/raw")


def candidate_folders(base: Path = REPO_ROOT) -> list[str]:
    """LeRobot dataset folders (meta/info.json) directly or one level below data/incoming and data/raw."""
    out = []
    for d in SEARCH_DIRS:
        top = base / d
        if not top.is_dir():
            continue
        for child in sorted(top.iterdir()):
            for cand in [child, *(sorted(child.iterdir()) if child.is_dir() else [])]:
                if (cand / "meta" / "info.json").exists():
                    out.append(source_key(cand))
    return sorted(set(out))


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
