"""Read a LeRobotDataset v3 folder without importing LeRobot.

Layout (LeRobot v3): meta/info.json (schema, fps, path templates), meta/stats.json, meta/tasks.parquet,
meta/episodes/*/*.parquet, data/*/*.parquet (many episodes per file), videos/<key>/*/*.mp4.
If the organisers' dataset uses another format, add a converter in hackpipe.ingest and keep this API.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


class DatasetFolder:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"{info_path} not found — is this a LeRobotDataset v3 folder?")
        try:
            self.info: dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid LeRobot metadata at {info_path}: {exc}") from exc
        if not isinstance(self.info, dict):
            raise ValueError(f"Invalid LeRobot metadata at {info_path}: expected a JSON object")
        self._frames: pd.DataFrame | None = None
        self._episode_metadata: pd.DataFrame | None = None

    # ---------- schema ----------
    @property
    def fps(self) -> float:
        return float(self.info["fps"])

    @property
    def features(self) -> dict[str, dict]:
        features = self.info.get("features", {})
        return features if isinstance(features, dict) else {}

    @property
    def video_keys(self) -> list[str]:
        return [k for k, v in self.features.items() if isinstance(v, dict) and v.get("dtype") == "video"]

    @property
    def robot_type(self) -> str | None:
        return self.info.get("robot_type")

    def feature_dim(self, key: str) -> int | None:
        feature = self.features.get(key)
        shape = feature.get("shape") if isinstance(feature, dict) else None
        if not shape:
            return None
        try:
            dim = 1
            for size in shape:
                dim *= int(size)
            return dim if dim > 0 else None
        except (TypeError, ValueError, OverflowError):
            return None

    # ---------- data ----------
    def data_files(self) -> list[Path]:
        return sorted((self.root / "data").rglob("*.parquet"))

    def frames(self) -> pd.DataFrame:
        if self._frames is None:
            files = self.data_files()
            if not files:
                raise FileNotFoundError(f"No parquet files under {self.root / 'data'}")
            self._frames = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        return self._frames

    def tasks(self) -> dict[int, str]:
        """task_index → task text (meta/tasks.parquet; legacy meta/tasks.jsonl)."""
        p = self.root / "meta" / "tasks.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if "task_index" in df.columns:
                names = df.index if df.index.name == "task" or "task" not in df.columns else df["task"]
                return {int(i): str(t) for t, i in zip(names, df["task_index"])}
        legacy = self.root / "meta" / "tasks.jsonl"
        if legacy.exists():
            out = {}
            for line in legacy.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    out[int(rec["task_index"])] = str(rec["task"])
            return out
        return {}

    def video_files(self, key: str) -> list[Path]:
        return sorted((self.root / "videos" / key).rglob("*.mp4"))

    def episode_metadata(self) -> pd.DataFrame:
        """Load LeRobot v3 per-episode references from ``meta/episodes``."""
        if self._episode_metadata is None:
            paths = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
            if not paths:
                self._episode_metadata = pd.DataFrame()
            else:
                self._episode_metadata = pd.concat(
                    [pd.read_parquet(path) for path in paths], ignore_index=True
                )
        return self._episode_metadata

    @staticmethod
    def _inside_root(root: Path, relative_path: str) -> Path:
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"dataset reference escapes the root: {relative_path}") from exc
        return candidate

    def data_path_for_episode(self, episode: dict[str, Any]) -> Path:
        chunk = int(episode["data/chunk_index"])
        file_index = int(episode["data/file_index"])
        template = self.info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
        return self._inside_root(self.root, template.format(chunk_index=chunk, file_index=file_index))

    def video_path_for_episode(self, episode: dict[str, Any], key: str) -> Path:
        prefix = f"videos/{key}"
        chunk = int(episode[f"{prefix}/chunk_index"])
        file_index = int(episode[f"{prefix}/file_index"])
        template = self.info.get(
            "video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
        relative = template.format(video_key=key, chunk_index=chunk, file_index=file_index)
        return self._inside_root(self.root, relative)

    def validate_structure(self, frames: pd.DataFrame | None = None,
                           task_index_column: str = "task_index") -> dict[str, Any]:
        """Return dataset-level blockers and episode-scoped structural errors."""
        blockers: list[str] = []
        episode_issues: dict[int, list[str]] = {}

        def episode_issue(episode_index: int, message: str) -> None:
            episode_issues.setdefault(episode_index, []).append(message)

        try:
            fps = float(self.info["fps"])
            if not math.isfinite(fps) or fps <= 0:
                blockers.append("info.json fps must be a finite positive number")
        except (KeyError, TypeError, ValueError):
            blockers.append("info.json is missing a valid fps")
        for count_key in ("total_episodes", "total_frames"):
            if count_key not in self.info:
                blockers.append(f"info.json is missing required {count_key}")

        features = self.info.get("features")
        if not isinstance(features, dict):
            blockers.append("info.json features must be an object")
            features = {}
        for key in ("observation.state", "action"):
            feature = features.get(key)
            if not isinstance(feature, dict) or not feature.get("shape"):
                blockers.append(f"info.json is missing a valid {key} feature")
            elif (not isinstance(feature["shape"], list) or not feature["shape"] or
                  any(not isinstance(value, int) or value <= 0 for value in feature["shape"])):
                blockers.append(f"info.json {key} feature has an invalid shape")
        for key in self.video_keys:
            camera_feature = features.get(key)
            shape = camera_feature.get("shape") if isinstance(camera_feature, dict) else None
            if not isinstance(shape, list) or len(shape) != 3 or shape[-1] != 3 or any(
                not isinstance(value, int) or value <= 0 for value in shape
            ):
                blockers.append(f"info.json camera feature {key!r} has an invalid HWC RGB shape")
        if self.video_keys and not self.info.get("video_path"):
            blockers.append("info.json is missing video_path for declared camera features")

        try:
            metadata = self.episode_metadata()
        except Exception as exc:  # malformed parquet should become an explicit blocker
            metadata = pd.DataFrame()
            blockers.append(f"cannot read meta/episodes parquet: {type(exc).__name__}: {exc}")
        required_episode_columns = {"episode_index", "length", "data/chunk_index", "data/file_index"}
        missing_columns = sorted(required_episode_columns - set(metadata.columns))
        if metadata.empty:
            blockers.append("no readable episode metadata under meta/episodes")
        elif missing_columns:
            blockers.append(f"episode metadata is missing required columns: {missing_columns}")

        if not metadata.empty and not missing_columns:
            ids: list[int] = []
            for row in metadata.to_dict(orient="records"):
                try:
                    episode_index = int(row["episode_index"])
                    ids.append(episode_index)
                    if int(row["length"]) <= 0:
                        episode_issue(episode_index, "episode metadata length must be positive")
                    try:
                        data_path = self.data_path_for_episode(row)
                        if not data_path.is_file():
                            episode_issue(episode_index, f"referenced data file is missing: {data_path}")
                    except (KeyError, TypeError, ValueError) as exc:
                        episode_issue(episode_index, f"invalid data file reference: {exc}")
                    for key in self.video_keys:
                        prefix = f"videos/{key}"
                        required = [f"{prefix}/{field}" for field in
                                    ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
                        absent = [name for name in required if name not in row]
                        if absent:
                            episode_issue(episode_index, f"camera {key!r} metadata is missing {absent}")
                            continue
                        try:
                            start, end = float(row[required[2]]), float(row[required[3]])
                            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                                raise ValueError("camera segment timestamps must be finite and increasing")
                            video_path = self.video_path_for_episode(row, key)
                            if not video_path.is_file():
                                episode_issue(episode_index, f"referenced camera file is missing: {video_path}")
                        except (KeyError, TypeError, ValueError) as exc:
                            episode_issue(episode_index, f"invalid camera {key!r} reference: {exc}")
                except (KeyError, TypeError, ValueError) as exc:
                    blockers.append(f"malformed episode metadata row: {exc}")
            if len(ids) != len(set(ids)):
                blockers.append("meta/episodes contains duplicate episode_index values")
            declared_count = self.info.get("total_episodes")
            if declared_count is not None:
                try:
                    if int(declared_count) != len(ids):
                        blockers.append(f"info.json total_episodes={declared_count} but meta/episodes has {len(ids)}")
                except (TypeError, ValueError):
                    blockers.append(f"info.json total_episodes is invalid: {declared_count!r}")

        if frames is not None:
            episode_column = "episode_index"
            if episode_column not in frames:
                blockers.append("data parquet is missing episode_index")
            else:
                try:
                    frame_ids = set(int(value) for value in frames[episode_column].dropna().unique())
                except (TypeError, ValueError, OverflowError):
                    frame_ids = set()
                    blockers.append("data contains invalid episode_index values")
                meta_ids = set(ids) if not metadata.empty and not missing_columns else set()
                for episode_index in sorted(meta_ids - frame_ids):
                    episode_issue(episode_index, "episode metadata has no data rows")
                for episode_index in sorted(frame_ids - meta_ids):
                    episode_issue(episode_index, "data rows have no matching meta/episodes entry")
                if not metadata.empty and not missing_columns:
                    length_by_id = {}
                    for row in metadata.to_dict(orient="records"):
                        try:
                            length_by_id[int(row["episode_index"])] = int(row["length"])
                        except (KeyError, TypeError, ValueError, OverflowError):
                            continue
                    actual_counts = frames.groupby(episode_column).size().to_dict()
                    for episode_index in sorted(meta_ids & frame_ids):
                        if int(actual_counts.get(episode_index, 0)) != length_by_id.get(episode_index):
                            episode_issue(episode_index,
                                          f"episode row count {actual_counts.get(episode_index, 0)} "
                                          f"does not match metadata length {length_by_id.get(episode_index)}")
                declared_frames = self.info.get("total_frames")
                if declared_frames is not None:
                    try:
                        if int(declared_frames) != len(frames):
                            blockers.append(f"info.json total_frames={declared_frames} but data has {len(frames)}")
                    except (TypeError, ValueError):
                        blockers.append(f"info.json total_frames is invalid: {declared_frames!r}")
                if task_index_column in frames:
                    try:
                        task_ids = set(self.tasks())
                        unknown = sorted(set(int(value) for value in frames[task_index_column].dropna().unique()) - task_ids)
                        if unknown:
                            blockers.append(f"data rows reference unknown task_index values: {unknown}")
                    except Exception as exc:
                        blockers.append(f"cannot read task metadata: {type(exc).__name__}: {exc}")

        return {"blocking_issues": blockers, "episode_issues": episode_issues,
                "episode_metadata": metadata}
