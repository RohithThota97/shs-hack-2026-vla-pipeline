"""Read a LeRobotDataset v3 folder without importing LeRobot.

Layout (LeRobot v3): meta/info.json (schema, fps, path templates), meta/stats.json, meta/tasks.parquet,
meta/episodes/*/*.parquet, data/*/*.parquet (many episodes per file), videos/<key>/*/*.mp4.
If the organisers' dataset uses another format, add a converter in hackpipe.ingest and keep this API.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class DatasetFolder:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"{info_path} not found — is this a LeRobotDataset v3 folder?")
        self.info: dict[str, Any] = json.loads(info_path.read_text())
        self._frames: pd.DataFrame | None = None

    # ---------- schema ----------
    @property
    def fps(self) -> float:
        return float(self.info["fps"])

    @property
    def features(self) -> dict[str, dict]:
        return self.info.get("features", {})

    @property
    def video_keys(self) -> list[str]:
        return [k for k, v in self.features.items() if v.get("dtype") == "video"]

    @property
    def robot_type(self) -> str | None:
        return self.info.get("robot_type")

    def feature_dim(self, key: str) -> int | None:
        shape = self.features.get(key, {}).get("shape")
        if not shape:
            return None
        dim = 1
        for s in shape:
            dim *= int(s)
        return dim

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

    # ---------- per-episode metadata ----------
    def episodes_meta(self) -> pd.DataFrame:
        """meta/episodes/*/*.parquet without the per-episode stats columns; empty if absent."""
        files = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        return df[[c for c in df.columns if not c.startswith("stats/")]]

    def episode_extras(self) -> dict[int, dict[str, Any]]:
        """Optional per-episode records (organiser format: meta/fr3_episodes_extra.jsonl) keyed by episode."""
        p = self.root / "meta" / "fr3_episodes_extra.jsonl"
        if not p.exists():
            return {}
        out = {}
        for i, line in enumerate(l for l in p.read_text().splitlines() if l.strip()):
            rec = json.loads(line)
            out[int(rec.get("episode_index", i))] = rec
        return out

    def episode_video_file(self, key: str, chunk_index: int, file_index: int) -> Path:
        template = self.info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
        return self.root / template.format(video_key=key, chunk_index=int(chunk_index), file_index=int(file_index))

    def episode_data_file(self, chunk_index: int, file_index: int) -> Path:
        template = self.info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
        return self.root / template.format(chunk_index=int(chunk_index), file_index=int(file_index))

    def stats_path(self) -> Path:
        return self.root / "meta" / "stats.json"
