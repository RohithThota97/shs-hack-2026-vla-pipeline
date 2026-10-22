"""Training input for multi-source versions: merge the sources into one LeRobot folder with
`lerobot-edit-dataset --operation.type merge`, map (source, episode) → merged episode index, and verify the
mapping by content hash before recording it in the manifest. Runs where LeRobot is installed (Colab)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from hackpipe.common.config import dataset_config, resolve
from hackpipe.common.ids import demo_id, now_iso
from hackpipe.common.lerobot_io import DatasetFolder
from hackpipe.common.registry import Registry
from hackpipe.dataset.versioning import load_manifest


class MaterializeError(RuntimeError):
    pass


def plan(version: str, registry: Registry | None = None) -> dict[str, Any]:
    """The merge command and the episode mapping (merge concatenates the sources in this order)."""
    reg = registry or Registry()
    m = load_manifest(version, reg)
    sources = m.get("sources") or [m["source"]]
    out = Path("data") / "processed" / version
    offsets, total = {}, 0
    for s in sources:
        offsets[s] = total
        total += int(DatasetFolder(resolve(s)).info["total_episodes"])
    splits = {k: sorted(offsets[s] + i for s, i in m["episodes"][k]) for k in ("train", "val")}
    cmd = ["lerobot-edit-dataset", "--new_repo_id", f"local/{version}", "--new_root", str(resolve(out)),
           "--operation.type", "merge",
           "--operation.repo_ids", json.dumps([f"local/{Path(s).name}" for s in sources]),
           "--operation.roots", json.dumps([str(resolve(s)) for s in sources])]
    return {"version": version, "root": out.as_posix(), "repo_id": f"local/{version}", "sources": sources,
            "offsets": offsets, "total_episodes": total, "splits": splits, "command": cmd}


def verify(p: dict[str, Any], manifest: dict[str, Any], registry: Registry) -> int:
    """Every mapped train/val episode in the merged folder must equal its source episode (content hash)."""
    from hackpipe.ingest.importer import _episode_checksum
    cols = dataset_config()["columns"]
    merged = DatasetFolder(resolve(p["root"]))
    if int(merged.info["total_episodes"]) != p["total_episodes"]:
        raise MaterializeError(f"merged folder has {merged.info['total_episodes']} episodes, "
                               f"expected {p['total_episodes']}")
    df = merged.frames()
    hash_cols = [cols["frame"], cols["timestamp"], cols["state"], cols["action"]]
    n = 0
    for split in ("train", "val"):
        for src, idx in manifest["episodes"][split]:
            ep = df[df[cols["episode"]] == p["offsets"][src] + idx].sort_values(cols["frame"])
            want = registry.get_demo(demo_id(src, idx))["meta"]["content_sha256"]
            if _episode_checksum(ep, hash_cols) != want:
                raise MaterializeError(f"merged episode {p['offsets'][src] + idx} is not {src}:{idx} — the merge "
                                       "did not concatenate the sources in order")
            n += 1
    return n


def materialize(version: str, registry: Registry | None = None, run: bool = True) -> dict[str, Any]:
    reg = registry or Registry()
    manifest = load_manifest(version, reg)
    if not manifest.get("needs_materialize"):
        return {"version": version, "needed": False, "reason": "single source: train directly from it"}
    p = plan(version, reg)
    if not run:
        return p | {"needed": True, "ran": False}
    if resolve(p["root"]).exists():
        raise MaterializeError(f"{p['root']} already exists; remove it to rebuild")
    proc = subprocess.run(p["command"])
    if proc.returncode != 0:
        raise MaterializeError(f"lerobot-edit-dataset exited with {proc.returncode}")
    checked = verify(p, manifest, reg)
    path = Path(reg.get_dataset(version)["manifest_path"])
    stored = json.loads(path.read_text())
    stored["materialized"] = {k: p[k] for k in ("root", "repo_id", "offsets", "total_episodes", "splits")} | {
        "verified_episodes": checked, "created_at": now_iso()}
    path.write_text(json.dumps(stored, indent=2))
    return p | {"needed": True, "ran": True, "verified_episodes": checked}
