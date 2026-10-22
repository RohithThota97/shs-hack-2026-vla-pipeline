"""R2 — Per-episode quality checks → VALID / REVIEW / INVALID.

Hard failures (INVALID): missing state/action, NaN or inf values, wrong dimensions, joint-limit violations,
non-increasing timestamps.
Soft failures (REVIEW): missing frames (timestamp gaps), frame_index gaps, timestamp drift vs fps,
too short / too long, final state not reached.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hackpipe.common.config import dataset_config
from hackpipe.common.lerobot_io import DatasetFolder

VALID, REVIEW, INVALID = "VALID", "REVIEW", "INVALID"


@dataclass
class EpisodeCheck:
    episode_index: int
    status: str = VALID
    hard: list[str] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> "EpisodeCheck":
        self.status = INVALID if self.hard else (REVIEW if self.soft else VALID)
        return self


def _stack(series: pd.Series) -> np.ndarray | None:
    try:
        return np.stack([np.asarray(v, dtype=np.float64).ravel() for v in series])
    except (ValueError, TypeError):
        return None


def check_episode(ep: pd.DataFrame, fps: float, expected_dims: dict[str, int | None],
                  cfg: dict[str, Any] | None = None) -> EpisodeCheck:
    cfg = cfg or dataset_config()
    cols, q = cfg["columns"], cfg["quality"]
    ep_idx = int(ep[cols["episode"]].iloc[0])
    res = EpisodeCheck(ep_idx)
    if cols["frame"] in ep.columns:
        ep = ep.sort_values(cols["frame"])
    n = len(ep)
    res.stats["n_frames"] = n

    # Length
    if n < q["min_episode_frames"]:
        res.soft.append(f"too_short ({n} < {q['min_episode_frames']} frames)")
    if n > q["max_episode_frames"]:
        res.soft.append(f"too_long ({n} > {q['max_episode_frames']} frames)")

    # Frame index continuity
    if cols["frame"] in ep.columns and n:
        fi = ep[cols["frame"]].to_numpy()
        expected = np.arange(fi[0], fi[0] + n)
        if fi[0] != 0 or not np.array_equal(fi, expected):
            res.soft.append("frame_index_gaps")

    # Timestamps: increasing, gaps (missing frames), drift
    if cols["timestamp"] in ep.columns and n > 1:
        ts = ep[cols["timestamp"]].to_numpy(dtype=float)
        dt = np.diff(ts)
        if np.any(dt <= 0):
            res.hard.append("timestamps_not_increasing")
        period = 1.0 / fps
        gaps = int(np.sum(dt > q["gap_factor"] * period))
        res.stats["timestamp_gaps"] = gaps
        if gaps:
            res.soft.append(f"missing_frames ({gaps} gaps)")
        tol = q["timestamp_tolerance_s"] if q["timestamp_tolerance_s"] is not None else 0.5 * period
        if cols["frame"] in ep.columns:
            drift = np.abs((ts - ts[0]) - (ep[cols["frame"]].to_numpy() - ep[cols["frame"]].iloc[0]) * period)
            res.stats["max_timestamp_drift_s"] = float(drift.max())
            if drift.max() > tol and not gaps:
                res.soft.append(f"timestamp_drift ({drift.max():.3f}s > {tol:.3f}s)")
    elif cols["timestamp"] not in ep.columns:
        res.hard.append("missing_timestamps")

    # State and action
    for key_name in ("state", "action"):
        col = cols[key_name]
        if col not in ep.columns:
            res.hard.append(f"missing_{key_name}")
            continue
        arr = _stack(ep[col])
        if arr is None:
            res.hard.append(f"{key_name}_not_numeric_or_ragged")
            continue
        if not np.all(np.isfinite(arr)):
            res.hard.append(f"{key_name}_nan_or_inf")
        exp = expected_dims.get(key_name)
        if exp is not None and arr.shape[1] != exp:
            res.hard.append(f"{key_name}_dim {arr.shape[1]} != {exp}")
        limits = (q.get("joint_limits") or {}).get(col)
        if limits and np.all(np.isfinite(arr)):
            lo, hi = np.asarray(limits["min"]), np.asarray(limits["max"])
            if arr.shape[1] == lo.size and (np.any(arr < lo) or np.any(arr > hi)):
                res.hard.append(f"{key_name}_outside_joint_limits")

    # Final state reached (only if the dataset has a success column)
    succ = cols.get("success")
    if succ and succ in ep.columns and n:
        if not bool(ep[succ].iloc[-1]):
            res.soft.append("final_state_not_reached")
    return res.finalize()


def check_dataset(root: str | Path, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or dataset_config()
    ds = DatasetFolder(root)
    cols = cfg["columns"]
    df = ds.frames()
    dims = {"state": ds.feature_dim(cols["state"]), "action": ds.feature_dim(cols["action"])}
    episodes = [check_episode(ep, ds.fps, dims, cfg) for _, ep in df.groupby(cols["episode"], sort=True)]

    dataset_issues = []
    if "total_episodes" in ds.info and ds.info["total_episodes"] != len(episodes):
        dataset_issues.append(f"info.json total_episodes={ds.info['total_episodes']} but found {len(episodes)}")
    if "total_frames" in ds.info and ds.info["total_frames"] != len(df):
        dataset_issues.append(f"info.json total_frames={ds.info['total_frames']} but found {len(df)}")
    videos = check_videos(ds) if cfg["quality"].get("check_videos") else {"skipped": "disabled in config"}
    if videos.get("missing_keys"):
        dataset_issues.append(f"no video files for: {videos['missing_keys']}")
    if videos.get("corrupted"):
        dataset_issues.append(f"corrupted video files: {len(videos['corrupted'])}")

    return {"root": str(Path(root).resolve()), "fps": ds.fps, "n_episodes": len(episodes),
            "n_frames": int(len(df)), "episodes": episodes, "dataset_issues": dataset_issues, "videos": videos}


def check_videos(ds: DatasetFolder) -> dict[str, Any]:
    """Each video key has files, and each file decodes start to end (needs PyAV)."""
    out: dict[str, Any] = {"missing_keys": [], "corrupted": [], "checked_files": 0}
    for key in ds.video_keys:
        if not ds.video_files(key):
            out["missing_keys"].append(key)
    try:
        import av  # type: ignore
    except ImportError:
        out["decode_check"] = "skipped (PyAV not installed: pip install -e '.[video]')"
        return out
    for key in ds.video_keys:
        for f in ds.video_files(key):
            out["checked_files"] += 1
            try:
                with av.open(str(f)) as container:
                    for _ in container.decode(video=0):
                        pass
            except Exception as e:  # noqa: BLE001 — any decode error means corrupted
                out["corrupted"].append({"file": str(f), "error": str(e)[:200]})
    out["decode_check"] = "done"
    return out
