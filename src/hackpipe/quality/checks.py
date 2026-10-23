"""R2 — Per-episode quality checks → VALID / REVIEW / INVALID.

Hard failures (INVALID): missing state/action/timestamps, missing or undecodable video for a camera, NaN or inf
values, wrong dimensions (vs data and robot config), joint-limit violations, non-increasing timestamps.
Soft failures (REVIEW): missing frames (timestamp gaps; decoded frames per camera vs duration × fps), frame_index
gaps, timestamp drift vs fps, camera/state desync, too short / too long, final state not reached (success flag,
or the organisers' per-episode outcome: status, object at rest, inside the tray footprint, tilt).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hackpipe.common.config import dataset_config, project_config
from hackpipe.common.ids import sha256_bytes
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
                  cfg: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> EpisodeCheck:
    cfg = cfg or dataset_config()
    cols, q = cfg["columns"], cfg["quality"]
    ep_idx = int(ep[cols["episode"]].iloc[0])
    res = EpisodeCheck(ep_idx)
    if cols["frame"] in ep.columns:
        ep = ep.sort_values(cols["frame"])
    n = len(ep)
    res.stats["n_frames"] = n

    # Length, in seconds (frames / fps) so one limit fits 30 fps and 6 fps recordings; *_frames still accepted
    lo, hi = q.get("min_episode_seconds"), q.get("max_episode_seconds")
    if not fps or fps <= 0:
        res.soft.append("no_fps (length not checked)")
        lo = hi = None
        secs = 0.0
    else:
        secs = n / fps
        res.stats["seconds"] = round(secs, 2)
        if lo is None and q.get("min_episode_frames") is not None:
            lo = q["min_episode_frames"] / fps
        if hi is None and q.get("max_episode_frames") is not None:
            hi = q["max_episode_frames"] / fps
    if lo is not None and secs < lo:
        res.soft.append(f"too_short ({secs:.1f} s < {lo:g} s, {n} frames at {fps:g} fps)")
    if hi is not None and secs > hi:
        res.soft.append(f"too_long ({secs:.1f} s > {hi:g} s, {n} frames at {fps:g} fps)")

    # Frame index continuity
    if cols["frame"] in ep.columns and n:
        fi = ep[cols["frame"]].to_numpy()
        expected = np.arange(fi[0], fi[0] + n)
        if fi[0] != 0 or not np.array_equal(fi, expected):
            res.soft.append("frame_index_gaps")

    # Timestamps: increasing, gaps (missing frames), drift (need fps)
    if cols["timestamp"] in ep.columns and n > 1 and fps and fps > 0:
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
    if extra:
        res.soft += check_final_state(extra, q.get("final_state") or {})
    return res.finalize()


def check_final_state(extra: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    """Simulation end state from the per-episode outcome record (meta/fr3_episodes_extra.jsonl)."""
    out = []
    if extra.get("success") is False or (extra.get("status") not in (None, "PASS")):
        out.append(f"final_state_not_reached (status={extra.get('status')}, failure={extra.get('failure')})")
    fm = extra.get("final_metrics") or {}
    if rules.get("require_at_rest", True) and fm.get("at_rest") is False:
        out.append("object_not_at_rest")
    margin, min_margin = fm.get("footprint_margin_mm"), rules.get("min_footprint_margin_mm", 0.0)
    if margin is not None and margin < min_margin:
        out.append(f"object_outside_target_zone (margin {margin:.1f} mm < {min_margin} mm)")
    tilt, max_tilt = fm.get("tilt_deg"), rules.get("max_tilt_deg")
    if tilt is not None and max_tilt is not None and tilt > max_tilt:
        out.append(f"object_tilted ({tilt:.1f} deg > {max_tilt} deg)")
    return out


def check_dataset(root: str | Path, cfg: dict[str, Any] | None = None,
                  only_episodes: set[int] | None = None) -> dict[str, Any]:
    """only_episodes: check just these episodes (incremental QC); dataset-level checks always run."""
    cfg = cfg or dataset_config()
    ds = DatasetFolder(root)
    cols, q = cfg["columns"], cfg["quality"]
    df = ds.frames()
    robot = project_config().get("robot") or {}
    dims = {"state": ds.feature_dim(cols["state"]), "action": ds.feature_dim(cols["action"])}
    dataset_issues = []
    for name in ("state", "action"):
        want = robot.get(f"{name}_dim")
        if want is not None and dims[name] is not None and dims[name] != want:
            dataset_issues.append(f"{name} dim in info.json {dims[name]} != robot config {want}")
            dims[name] = want                      # every episode then fails the dimension check
    extras = ds.episode_extras()
    n_episodes = int(df[cols["episode"]].nunique())
    episodes = [check_episode(ep, ds.fps, dims, cfg, extras.get(int(i)))
                for i, ep in df.groupby(cols["episode"], sort=True)
                if only_episodes is None or int(i) in only_episodes]

    if "total_episodes" in ds.info and ds.info["total_episodes"] != n_episodes:
        dataset_issues.append(f"info.json total_episodes={ds.info['total_episodes']} but found {n_episodes}")
    if "total_frames" in ds.info and ds.info["total_frames"] != len(df):
        dataset_issues.append(f"info.json total_frames={ds.info['total_frames']} but found {len(df)}")
    ep_meta = ds.episodes_meta()
    files = None
    if only_episodes is not None and len(ep_meta):
        rows = ep_meta[ep_meta["episode_index"].isin(only_episodes)]
        files = {ds.episode_video_file(k, r[f"videos/{k}/chunk_index"], r[f"videos/{k}/file_index"])
                 for _, r in rows.iterrows() for k in ds.video_keys if f"videos/{k}/chunk_index" in r}
    if not episodes:
        videos, frame_times = {"skipped": "no episodes to check"}, {}
    elif q.get("check_videos"):
        videos, frame_times = check_videos(ds, files)
    else:
        videos, frame_times = {"skipped": "disabled in config"}, {}
    if videos.get("missing_keys"):
        dataset_issues.append(f"no video files for: {videos['missing_keys']}")
    if videos.get("corrupted"):
        dataset_issues.append(f"corrupted video files: {len(videos['corrupted'])}")

    if len(ep_meta) and ds.video_keys:        # file presence + sync need no decoding; frame counts do
        corrupted = {Path(c["file"]) for c in videos.get("corrupted", [])}
        tol_s = q["timestamp_tolerance_s"] if q["timestamp_tolerance_s"] is not None else 0.5 / ds.fps
        by_ep = {e.episode_index: e for e in episodes}
        for _, row in ep_meta.iterrows():
            e = by_ep.get(int(row["episode_index"]))
            if e is not None:
                check_episode_videos(e, row, ds, frame_times, corrupted, tol_s)
                e.finalize()

    return {"root": str(Path(root).resolve()), "fps": ds.fps, "n_episodes": n_episodes,
            "n_frames": int(len(df)), "episodes": episodes, "dataset_issues": dataset_issues, "videos": videos,
            "redundancy": redundancy(df, cols)}


def redundancy(df: pd.DataFrame, cols: dict[str, Any]) -> dict[str, Any]:
    """Episodes whose state + action trajectories are byte-identical (only images may differ)."""
    groups: dict[str, list[int]] = {}
    for i, ep in df.groupby(cols["episode"], sort=True):
        ep = ep.sort_values(cols["frame"]) if cols["frame"] in ep.columns else ep
        parts = [_stack(ep[c]) for c in (cols["state"], cols["action"]) if c in ep.columns]
        key = sha256_bytes(b"".join(np.ascontiguousarray(p).tobytes() for p in parts if p is not None))
        groups.setdefault(key, []).append(int(i))
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    return {"distinct_trajectories": len(groups), "largest_identical_group": sizes[0] if sizes else 0,
            "identical_groups": [v for v in groups.values() if len(v) > 1][:20]}


def check_episode_videos(e: EpisodeCheck, row: pd.Series, ds: DatasetFolder,
                         frame_times: dict[Path, np.ndarray], corrupted: set[Path], tol_s: float) -> None:
    """Per camera: file present and decodable, decoded frames ≈ duration × fps, video span aligned with states."""
    n = e.stats.get("n_frames", 0)
    for key in ds.video_keys:
        name = key.rsplit(".", 1)[-1]
        if f"videos/{key}/chunk_index" not in row:
            e.hard.append(f"missing_video:{name} (not in episode metadata)")
            continue
        f = ds.episode_video_file(key, row[f"videos/{key}/chunk_index"], row[f"videos/{key}/file_index"])
        if not f.exists():
            e.hard.append(f"missing_video:{name}")
            continue
        if f in corrupted:
            e.hard.append(f"video_corrupted:{name}")
            continue
        t0, t1 = float(row[f"videos/{key}/from_timestamp"]), float(row[f"videos/{key}/to_timestamp"])
        if ds.fps and ds.fps > 0 and abs((t1 - t0) - n / ds.fps) > tol_s:      # no fps: flagged as no_fps
            e.soft.append(f"camera_state_desync:{name} (video {t1 - t0:.3f}s vs states {n / ds.fps:.3f}s)")
        times = frame_times.get(f)
        if times is not None and ds.fps and ds.fps > 0:
            half = 0.5 / ds.fps
            decoded = int(np.sum((times >= t0 - half) & (times < t1 - half)))
            e.stats[f"video_frames:{name}"] = decoded
            if decoded != n:
                e.soft.append(f"missing_frames:{name} ({decoded} decoded vs {n} expected)")


def check_videos(ds: DatasetFolder, files: set[Path] | None = None) -> tuple[dict[str, Any], dict[Path, np.ndarray]]:
    """Each video key has files, and each file decodes start to end (needs PyAV). Also returns frame times.
    files: decode only these (incremental QC); None = all."""
    out: dict[str, Any] = {"missing_keys": [], "corrupted": [], "checked_files": 0, "decoded_frames": 0}
    times: dict[Path, np.ndarray] = {}
    for key in ds.video_keys:
        if not ds.video_files(key):
            out["missing_keys"].append(key)
    try:
        import av  # type: ignore
    except ImportError:
        out["decode_check"] = "skipped (PyAV not installed: pip install -e '.[video]')"
        out["skipped"] = True
        return out, times
    for key in ds.video_keys:
        for f in ds.video_files(key):
            if files is not None and f not in files:
                continue
            out["checked_files"] += 1
            t = []
            try:
                with av.open(str(f)) as container:
                    for frame in container.decode(video=0):
                        t.append(float(frame.time) if frame.time is not None else np.nan)
                times[f] = np.asarray(t)
                out["decoded_frames"] += len(t)
            except Exception as e:  # noqa: BLE001 — any decode error means corrupted
                out["corrupted"].append({"file": str(f), "error": str(e)[:200]})
    out["decode_check"] = "done"
    return out, times
