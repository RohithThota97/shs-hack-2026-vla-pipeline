"""R1 — Import episodes from a dataset folder into the registry as demo records.

Each episode becomes one demo record with a deterministic UUID, so re-importing never duplicates.
`watch()` polls a folder and imports new episodes as they appear.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hackpipe.common.config import dataset_config, project_config
from hackpipe.common.ids import demo_id, now_iso, sha256_bytes
from hackpipe.common.lerobot_io import DatasetFolder
from hackpipe.common.registry import Registry


def _episode_checksum(ep: pd.DataFrame, cols: list[str]) -> str:
    """Content hash of one episode's tabular data (episodes share parquet files in v3)."""
    parts = []
    for c in cols:
        if c in ep.columns:
            vals = ep[c].to_numpy()
            if vals.dtype == object:
                vals = np.stack([np.asarray(v, dtype=np.float64).ravel() for v in vals])
            parts.append(np.ascontiguousarray(vals).tobytes())
    return sha256_bytes(b"".join(parts))


def build_demo_records(root: str | Path, source: str | None = None) -> list[dict[str, Any]]:
    ds = DatasetFolder(root)
    cfg = dataset_config()["columns"]
    proj = project_config()
    df = ds.frames()
    tasks = ds.tasks()
    source = source or str(Path(root).resolve())
    ep_col, ts_col = cfg["episode"], cfg["timestamp"]
    hash_cols = [cfg["frame"], ts_col, cfg["state"], cfg["action"]]
    records = []
    for ep_idx, ep in df.groupby(ep_col, sort=True):
        ep = ep.sort_values(cfg["frame"]) if cfg["frame"] in ep.columns else ep
        task_text = None
        if cfg["task_index"] in ep.columns and len(ep):
            task_text = tasks.get(int(ep[cfg["task_index"]].iloc[0]))
        ts = ep[ts_col].to_numpy(dtype=float) if ts_col in ep.columns else np.array([])
        records.append({
            "demo_id": demo_id(source, int(ep_idx)),
            "source": source,
            "episode_index": int(ep_idx),
            "meta": {
                "n_frames": int(len(ep)),
                "fps": ds.fps,
                "duration_s": float(ts[-1] - ts[0]) if ts.size > 1 else 0.0,
                "task": task_text,
                "robot": {"type": ds.robot_type or proj["robot"]["type"],
                          "embodiment": proj["robot"].get("embodiment")},
                "cameras": {k: ds.features[k].get("shape") for k in ds.video_keys},
                "state_dim": ds.feature_dim(cfg["state"]),
                "action_dim": ds.feature_dim(cfg["action"]),
                "operator": proj["demo_defaults"]["operator"],
                "environment": proj["demo_defaults"]["environment"],
                "content_sha256": _episode_checksum(ep, hash_cols),
                "codebase_version": ds.info.get("codebase_version"),
                "imported_at": now_iso(),
            },
        })
    return records


def ingest(root: str | Path, registry: Registry | None = None) -> int:
    reg = registry or Registry()
    records = build_demo_records(root)
    for r in records:
        reg.upsert_demo(r["demo_id"], r["source"], r["episode_index"], r["meta"])
    return len(records)


def watch(root: str | Path, interval_s: int = 60, registry: Registry | None = None) -> None:
    """Poll a dataset folder and import new episodes. Stop with Ctrl+C."""
    reg = registry or Registry()
    seen = {d["demo_id"] for d in reg.demos(source=str(Path(root).resolve()))}
    print(f"Watching {root} every {interval_s}s ({len(seen)} demos already registered)")
    while True:
        try:
            records = build_demo_records(root)
        except FileNotFoundError as e:
            print(f"waiting: {e}")
            records = []
        new = [r for r in records if r["demo_id"] not in seen]
        for r in new:
            reg.upsert_demo(r["demo_id"], r["source"], r["episode_index"], r["meta"])
            seen.add(r["demo_id"])
        if new:
            print(f"{now_iso()} imported {len(new)} new demos")
        time.sleep(interval_s)
