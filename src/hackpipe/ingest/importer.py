"""R1 — Import episodes from a dataset folder into the registry as demo records.

Each episode becomes one demo record with a deterministic UUID, so re-importing never duplicates.
Per demo: source, recording span, robot config, operator, task, environment (incl. per-episode seed and
lighting from meta/fr3_episodes_extra.jsonl when present), cameras, SHA-256 of every file holding the episode,
import time and pipeline version.
`watch()` polls a folder, imports new episodes as they appear and re-runs the quality report (R2).

Incremental: a demo already registered with unchanged files is skipped; a changed one goes back to "imported"
(re-checked by QC); a demo whose content and file checksums equal a demo from another folder is a DUPLICATE and
never enters a dataset version. File hashes are cached by (path, size, mtime) so re-imports don't rehash GBs.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import hackpipe
from hackpipe.common.config import dataset_config, project_config, resolve, source_key
from hackpipe.common.envinfo import git_info
from hackpipe.common.ids import demo_id, now_iso, sha256_bytes, sha256_file
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


def _episode_files(ds: DatasetFolder, row: pd.Series | None) -> list[Path]:
    """Data file + one video file per camera that hold this episode (v3 packs many episodes per file)."""
    if row is None:
        return ds.data_files() + [f for k in ds.video_keys for f in ds.video_files(k)]
    files = []
    if "data/chunk_index" in row:
        files.append(ds.episode_data_file(row["data/chunk_index"], row["data/file_index"]))
    for k in ds.video_keys:
        if f"videos/{k}/chunk_index" in row:
            files.append(ds.episode_video_file(k, row[f"videos/{k}/chunk_index"], row[f"videos/{k}/file_index"]))
    return files


def _lighting(extra: dict[str, Any]) -> dict[str, Any] | None:
    light = ((extra.get("randomization") or {}).get("applied") or {}).get("light")
    if not light:
        return None
    return {"color_temperature_K": (light.get("rect") or {}).get("color_temperature_K"),
            "rect_intensity": (light.get("rect") or {}).get("intensity_to"),
            "dome_intensity": (light.get("dome") or {}).get("intensity_to"),
            "rect_tilt_deg": light.get("rect_tilt_deg")}


def _outcome(extra: dict[str, Any]) -> dict[str, Any]:
    fm = extra.get("final_metrics") or {}
    return {"status": extra.get("status"), "success": extra.get("success"), "failure": extra.get("failure"),
            "at_rest": fm.get("at_rest"), "footprint_margin_mm": fm.get("footprint_margin_mm"),
            "tilt_deg": fm.get("tilt_deg"),
            "placement_offset_mm": (fm.get("placement_offset_mm") or {}).get("horizontal")}


class HashCache:
    """sha256 per file, reused while (size, mtime) are unchanged. Stored next to the registry."""

    def __init__(self, path: Path | None = None):
        reg = resolve(project_config()["project"]["registry_path"])
        self.path = path or reg.with_name("file_hash_cache.json")
        try:
            self.data: dict[str, Any] = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.data = {}
        self.hits = self.misses = 0

    def sha(self, f: Path) -> str:
        st = f.stat()
        key, sig = str(f.resolve()), [st.st_size, st.st_mtime_ns]
        hit = self.data.get(key)
        if hit and hit["sig"] == sig:
            self.hits += 1
            return hit["sha256"]
        self.misses += 1
        digest = sha256_file(f)
        self.data[key] = {"sig": sig, "sha256": digest}
        return digest

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data))
        os.replace(tmp, self.path)


def duplicate_key(episode_index: int, meta: dict[str, Any]) -> str:
    """Same episode of the same files: episode index + content + checksums of the files holding it.
    (In LeRobot v3 several episodes share files, and here even their content is identical.)"""
    files = meta.get("files_sha256") or {}
    return "|".join([str(episode_index), str(meta.get("content_sha256")), *sorted(str(v) for v in files.values())])


def build_demo_records(root: str | Path, source: str | None = None,
                       cache: HashCache | None = None) -> list[dict[str, Any]]:
    ds = DatasetFolder(root)
    cfg = dataset_config()["columns"]
    proj = project_config()
    df = ds.frames()
    tasks = ds.tasks()
    extras = ds.episode_extras()
    ep_meta = ds.episodes_meta()
    ep_rows = {int(r["episode_index"]): r for _, r in ep_meta.iterrows()} if len(ep_meta) else {}
    source = source or source_key(root)
    ep_col, ts_col = cfg["episode"], cfg["timestamp"]
    hash_cols = [cfg["frame"], ts_col, cfg["state"], cfg["action"]]
    cache = cache or HashCache()
    pipeline = {"hackpipe": hackpipe.__version__, "git_commit": git_info().get("commit")}
    cameras = {k: {"shape": ds.features[k].get("shape"),
                   "fps": (ds.features[k].get("info") or {}).get("video.fps", ds.fps),
                   "codec": (ds.features[k].get("info") or {}).get("video.codec")} for k in ds.video_keys}
    records = []
    for ep_idx, ep in df.groupby(ep_col, sort=True):
        ep_idx = int(ep_idx)
        ep = ep.sort_values(cfg["frame"]) if cfg["frame"] in ep.columns else ep
        task_text = None
        if cfg["task_index"] in ep.columns and len(ep):
            task_text = tasks.get(int(ep[cfg["task_index"]].iloc[0]))
        ts = ep[ts_col].to_numpy(dtype=float) if ts_col in ep.columns else np.array([])
        files = {}
        for f in _episode_files(ds, ep_rows.get(ep_idx)):
            if f.exists():
                files[f.relative_to(ds.root).as_posix()] = cache.sha(f)
            else:
                files[f.relative_to(ds.root).as_posix()] = None   # missing → R2 flags it
        extra = extras.get(ep_idx, {})
        environment = dict(proj["demo_defaults"]["environment"])
        if extra:
            environment.update({"variation": extra.get("name"), "seed": extra.get("seed"),
                                "lighting": _lighting(extra)})
            stage_root = (extra.get("stack_lock") or {}).get("root")
            if stage_root:
                environment["layout_id"] = Path(stage_root).name.split(".")[0]
        records.append({
            "demo_id": demo_id(source, ep_idx),
            "source": source,
            "episode_index": ep_idx,
            "meta": {
                "n_frames": int(len(ep)),
                "fps": ds.fps,
                "recording": {"start_s": float(ts[0]) if ts.size else None,
                              "end_s": float(ts[-1]) if ts.size else None,
                              "wallclock_start": None},   # not in the provided data
                "duration_s": float(ts[-1] - ts[0]) if ts.size > 1 else 0.0,
                "task": task_text,
                "instruction": task_text,
                "robot": {"type": ds.robot_type or proj["robot"]["type"],
                          "embodiment": proj["robot"].get("embodiment"),
                          "end_effector": proj["robot"].get("end_effector"),
                          "controller": proj["robot"].get("controller")},
                "cameras": cameras,
                "state_dim": ds.feature_dim(cfg["state"]),
                "action_dim": ds.feature_dim(cfg["action"]),
                "operator": proj["demo_defaults"]["operator"],
                "environment": environment,
                "outcome": _outcome(extra) if extra else None,
                "content_sha256": _episode_checksum(ep, hash_cols),
                "trajectory_sha256": _episode_checksum(ep, [cfg["state"], cfg["action"]]),
                "files_sha256": files,
                "codebase_version": ds.info.get("codebase_version"),
                "pipeline": pipeline,
                "imported_at": now_iso(),
            },
        })
    return records


def import_demos(root: str | Path, registry: Registry | None = None) -> dict[str, Any]:
    """Register the folder's episodes. Returns counts of new / changed / unchanged / duplicate demos."""
    reg = registry or Registry()
    cache = HashCache()
    records = build_demo_records(root, cache=cache)
    cache.save()
    known = {duplicate_key(d["episode_index"], d["meta"]): d for d in reg.demos() if d["status"] != "DUPLICATE"}
    out: dict[str, Any] = {"source": source_key(root), "episodes": len(records), "new": [], "changed": [],
                           "unchanged": [], "duplicate": [], "hash_cache": {"hits": cache.hits,
                                                                            "misses": cache.misses}}
    for r in records:
        existing = reg.get_demo(r["demo_id"])
        if existing:
            same = (existing["meta"].get("content_sha256") == r["meta"]["content_sha256"]
                    and existing["meta"].get("files_sha256") == r["meta"]["files_sha256"])
            if same:
                out["unchanged"].append(r["episode_index"])
                continue
            reg.upsert_demo(r["demo_id"], r["source"], r["episode_index"], r["meta"], status="imported")
            out["changed"].append(r["episode_index"])
            continue
        twin = known.get(duplicate_key(r["episode_index"], r["meta"]))
        if twin and twin["source"] != r["source"]:
            reg.upsert_demo(r["demo_id"], r["source"], r["episode_index"],
                            {**r["meta"], "duplicate_of": twin["demo_id"]}, status="DUPLICATE")
            out["duplicate"].append(r["episode_index"])
            continue
        reg.upsert_demo(r["demo_id"], r["source"], r["episode_index"], r["meta"])
        out["new"].append(r["episode_index"])
    return out


def ingest(root: str | Path, registry: Registry | None = None) -> int:
    """Import a folder; returns the number of episodes in it (see import_demos for the breakdown)."""
    return import_demos(root, registry)["episodes"]


def watch(root: str | Path, interval_s: int = 60, registry: Registry | None = None,
          max_polls: int | None = None) -> None:
    """Poll a dataset folder, import new episodes and refresh the QC report. Stop with Ctrl+C."""
    from hackpipe.quality.report import run_qc
    reg = registry or Registry()
    seen = {d["demo_id"] for d in reg.demos(source=source_key(root))}
    print(f"Watching {root} every {interval_s}s ({len(seen)} demos already registered)")
    polls = 0
    while max_polls is None or polls < max_polls:
        polls += 1
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
            report = run_qc(root, reg)
            print(f"{now_iso()} imported {len(new)} new demos · QC {report['status_counts']}")
        if max_polls is None or polls < max_polls:
            time.sleep(interval_s)
