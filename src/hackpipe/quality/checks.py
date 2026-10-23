"""R2 dataset and episode quality checks."""
from __future__ import annotations

import hashlib
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
        return np.stack([np.asarray(value, dtype=np.float64).ravel() for value in series])
    except (ValueError, TypeError):
        return None


def numeric_trajectory_hash(ep: pd.DataFrame, cfg: dict[str, Any] | None = None) -> str | None:
    """Hash exact numeric state/action samples; this says nothing about visual identity."""
    cfg = cfg or dataset_config()
    cols = cfg["columns"]
    if cols["frame"] in ep:
        ep = ep.sort_values(cols["frame"])
    arrays = []
    for key in ("state", "action"):
        column = cols[key]
        if column not in ep:
            return None
        arr = _stack(ep[column])
        if arr is None or not np.all(np.isfinite(arr)):
            return None
        arrays.append(arr.astype("<f8", copy=False))
    digest = hashlib.sha256()
    for key, arr in zip(("state", "action"), arrays):
        digest.update(key.encode("utf-8"))
        digest.update(np.asarray(arr.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


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

    if n < q["min_episode_frames"]:
        res.soft.append(f"too_short ({n} < {q['min_episode_frames']} frames)")
    if n > q["max_episode_frames"]:
        res.soft.append(f"too_long ({n} > {q['max_episode_frames']} frames)")

    if cols["frame"] in ep.columns and n:
        fi = ep[cols["frame"]].to_numpy()
        try:
            expected = np.arange(fi[0], fi[0] + n)
            if fi[0] != 0 or not np.array_equal(fi, expected):
                res.soft.append("frame_index_gaps")
        except (TypeError, ValueError):
            res.hard.append("frame_index_invalid")

    if cols["timestamp"] not in ep.columns:
        res.hard.append("missing_timestamps")
    elif n:
        try:
            ts = ep[cols["timestamp"]].to_numpy(dtype=float)
            if not np.all(np.isfinite(ts)):
                res.hard.append("timestamps_not_finite")
            elif n > 1:
                dt = np.diff(ts)
                if np.any(dt <= 0):
                    res.hard.append("timestamps_not_increasing")
                if not np.isfinite(fps) or fps <= 0:
                    res.hard.append("invalid_dataset_fps")
                else:
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
        except (TypeError, ValueError):
            res.hard.append("timestamps_not_numeric")

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

    success_col = cols.get("success")
    if success_col:
        if success_col not in ep.columns or not n:
            res.stats["success_label"] = "UNKNOWN"
            res.soft.append("success_label_unavailable")
        else:
            values = ep[success_col]
            allowed = values.dropna().isin([True, False, 0, 1]).all() and values.notna().all()
            if not allowed:
                res.stats["success_label"] = "UNKNOWN"
                res.hard.append("success_label_invalid")
            else:
                final = bool(values.iloc[-1])
                res.stats["success_label"] = final
                if not final:
                    res.soft.append("final_state_not_reached")
    else:
        res.stats["success_label"] = "UNKNOWN"
    return res.finalize()


def check_videos(ds: DatasetFolder) -> dict[str, Any]:
    """Validate referenced camera files and decode when PyAV is available."""
    out: dict[str, Any] = {"declared_keys": list(ds.video_keys), "missing_keys": [],
                           "missing_files": [], "corrupted": [],
                           "checked_files": 0, "timelines": {}}
    for key in ds.video_keys:
        if not ds.video_files(key):
            out["missing_keys"].append(key)
    metadata = ds.episode_metadata()
    referenced: dict[str, set[str]] = {key: set() for key in ds.video_keys}
    if "episode_index" in metadata.columns:
        for row in metadata.to_dict(orient="records"):
            for key in ds.video_keys:
                try:
                    path = ds.video_path_for_episode(row, key)
                    referenced[key].add(str(path))
                    if not path.is_file():
                        out["missing_files"].append({"episode_index": int(row["episode_index"]),
                                                      "key": key, "file": str(path)})
                except (KeyError, TypeError, ValueError) as exc:
                    out["missing_files"].append({"episode_index": row.get("episode_index"),
                                                  "key": key, "error": str(exc)})
    try:
        import av  # type: ignore
    except ImportError:
        out["decode_check"] = "skipped (PyAV unavailable)"
        out["verification_source"] = "skipped"
        return out
    for key, paths in referenced.items():
        for path_text in sorted(paths):
            path = Path(path_text)
            if not path.is_file():
                continue
            out["checked_files"] += 1
            try:
                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    pts, shapes = [], set()
                    for frame in container.decode(video=0):
                        if frame.time is not None:
                            pts.append(float(frame.time))
                        shapes.add((int(frame.height), int(frame.width), 3))
                    if not pts:
                        raise ValueError("video contains no timestamped frames")
                    if not np.all(np.isfinite(pts)) or np.any(np.diff(pts) <= 0):
                        raise ValueError("video frame timestamps are non-finite or non-increasing")
                    out["timelines"][path_text] = {"timestamps": pts, "shapes": sorted(shapes),
                                                       "fps": float(stream.average_rate) if stream.average_rate else None}
            except Exception as exc:  # decoder-specific errors are recorded, not hidden
                out["corrupted"].append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"[:300]})
    out["decode_check"] = "done"
    out["verification_source"] = "PyAV full decode"
    out["full_decode"] = True
    return out


def check_dataset(root: str | Path, cfg: dict[str, Any] | None = None,
                  camera_report: str | Path | None = None) -> dict[str, Any]:
    cfg = cfg or dataset_config()
    ds = DatasetFolder(root)
    cols = cfg["columns"]
    try:
        df = ds.frames()
    except Exception as exc:
        try:
            fps = ds.fps
        except (KeyError, TypeError, ValueError):
            fps = float("nan")
        problem = f"cannot read dataset data parquet: {type(exc).__name__}: {exc}"
        videos = check_videos(ds) if cfg["quality"].get("check_videos") else {
            "declared_keys": list(ds.video_keys), "missing_keys": [], "missing_files": [],
            "corrupted": [], "checked_files": 0,
            "decode_check": "skipped (disabled in config)", "verification_source": "skipped"}
        return {"root": str(Path(root).resolve()), "fps": fps, "n_episodes": 0, "n_frames": 0,
                "episodes": [], "dataset_issues": [problem], "blocking_issues": [problem],
                "readiness": {"status": "BLOCKED", "ready_for_training": False,
                              "blocking_issues": [problem]},
                "potential_duplicate_trajectories": [], "videos": videos}
    structure = ds.validate_structure(df, cols["task_index"])
    try:
        fps = ds.fps
    except (KeyError, TypeError, ValueError):
        fps = float("nan")
    dims = {"state": ds.feature_dim(cols["state"]), "action": ds.feature_dim(cols["action"])}
    episodes = ([check_episode(ep, fps, dims, cfg) for _, ep in df.groupby(cols["episode"], sort=True)]
                if cols["episode"] in df.columns else [])
    if cols["episode"] not in df.columns and not any("episode_index" in problem for problem in structure["blocking_issues"]):
        structure["blocking_issues"].append(f"data parquet is missing configured episode column {cols['episode']!r}")
    by_id = {episode.episode_index: episode for episode in episodes}
    for episode_index, issues in structure["episode_issues"].items():
        if episode_index in by_id:
            by_id[episode_index].hard.extend(f"structure: {issue}" for issue in issues)
            by_id[episode_index].finalize()

    videos = check_videos(ds) if cfg["quality"].get("check_videos") else {
        "declared_keys": list(ds.video_keys), "missing_keys": [], "missing_files": [],
        "corrupted": [], "checked_files": 0,
        "timelines": {}, "decode_check": "skipped (disabled in config)", "verification_source": "skipped"}
    if (camera_report is not None and cfg["quality"].get("check_videos") and
            videos.get("verification_source") == "skipped"):
        from hackpipe.quality.camera_evidence import CameraEvidenceError, validate_ffmpeg_report
        try:
            videos = validate_ffmpeg_report(ds, camera_report)
        except CameraEvidenceError as exc:
            videos["evidence_error"] = str(exc)
            if exc.blocking:
                videos["corrupted"].append({"file": None, "error": str(exc)})
    dataset_issues = list(structure["blocking_issues"])
    if videos.get("missing_keys"):
        dataset_issues.append(f"no video files for declared cameras: {videos['missing_keys']}")
    if videos.get("missing_files"):
        dataset_issues.append(f"referenced camera files missing: {len(videos['missing_files'])}")
    if videos.get("corrupted"):
        dataset_issues.append(f"corrupted video files: {len(videos['corrupted'])}")

    meta = structure["episode_metadata"]
    if videos.get("decode_check") == "done" and not videos.get("corrupted"):
        for row in meta.to_dict(orient="records"):
            try:
                episode_index = int(row["episode_index"])
                episode = by_id.get(episode_index)
                if episode is None:
                    continue
                for key in ds.video_keys:
                    path = str(ds.video_path_for_episode(row, key))
                    timeline = videos["timelines"].get(path)
                    if timeline is None:
                        continue
                    start, end = float(row[f"videos/{key}/from_timestamp"]), float(row[f"videos/{key}/to_timestamp"])
                    selected_pts = [timestamp for timestamp in timeline["timestamps"]
                                    if start - 1e-6 <= timestamp < end - 1e-6]
                    n_video = len(selected_pts)
                    expected_n = episode.stats["n_frames"]
                    if n_video != expected_n:
                        episode.soft.append(f"camera_frame_count_mismatch:{key} ({n_video} != {expected_n})")
                    elif cols["timestamp"] in df.columns and selected_pts:
                        frame_rows = df[df[cols["episode"]] == episode_index].sort_values(cols["frame"])
                        source_ts = frame_rows[cols["timestamp"]].to_numpy(dtype=float)
                        expected_pts = start + source_ts
                        alignment_error = np.abs(np.asarray(selected_pts) - expected_pts)
                        tolerance = 0.5 / ds.fps if ds.fps > 0 else 0.0
                        if (not np.all(np.isfinite(expected_pts)) or
                                float(alignment_error.max()) > tolerance):
                            episode.soft.append(f"camera_timestamp_mismatch:{key} "
                                                f"(max {float(alignment_error.max()):.4f}s > {tolerance:.4f}s)")
                    feature_shape = tuple(ds.features[key]["shape"])
                    if timeline["shapes"] and any(tuple(shape) != feature_shape for shape in timeline["shapes"]):
                        episode.hard.append(f"camera_shape_mismatch:{key}")
                    camera_fps = timeline.get("fps")
                    if camera_fps and abs(camera_fps - ds.fps) > max(0.1, ds.fps * 0.01):
                        episode.soft.append(f"camera_fps_mismatch:{key} ({camera_fps} != {ds.fps})")
                episode.finalize()
            except (KeyError, TypeError, ValueError) as exc:
                if int(row.get("episode_index", -1)) in by_id:
                    by_id[int(row["episode_index"])].soft.append(f"camera_timing_unverified: {exc}")
                    by_id[int(row["episode_index"])].finalize()

    trajectory_groups: dict[str, list[int]] = {}
    if cols["episode"] in df.columns:
        for episode_index, ep in df.groupby(cols["episode"], sort=True):
            digest = numeric_trajectory_hash(ep, cfg)
            if digest:
                trajectory_groups.setdefault(digest, []).append(int(episode_index))
    duplicates = [{"episode_indices": ids, "numeric_sha256": digest,
                   "interpretation": "numeric state/action match only; camera images were not compared"}
                  for digest, ids in trajectory_groups.items() if len(ids) > 1]
    for group in duplicates:
        for episode_index in group["episode_indices"]:
            if episode_index in by_id:
                by_id[episode_index].soft.append("potential_duplicate_numeric_trajectory; visual identity unverified")
                by_id[episode_index].finalize()

    blockers = list(dict.fromkeys(dataset_issues))
    if blockers:
        reason = "; ".join(blockers)
        for episode in episodes:
            episode.hard.append(f"dataset_blocked: {reason}")
            episode.finalize()
    if blockers:
        readiness = "BLOCKED"
    elif videos.get("decode_check", "").startswith("skipped") and ds.video_keys:
        readiness = "REVIEW"
    elif any(ep.status != VALID for ep in episodes):
        readiness = "REVIEW"
    else:
        readiness = "READY"
    videos.pop("timelines", None)  # internal frame timing data is not part of the report
    return {"root": str(Path(root).resolve()), "fps": fps, "n_episodes": len(episodes),
            "n_frames": int(len(df)), "episodes": episodes, "dataset_issues": dataset_issues,
            "blocking_issues": blockers, "readiness": {"status": readiness,
            "ready_for_training": readiness == "READY", "blocking_issues": blockers},
            "potential_duplicate_trajectories": duplicates, "videos": videos}
