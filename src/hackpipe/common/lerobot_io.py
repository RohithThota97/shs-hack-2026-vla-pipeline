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
