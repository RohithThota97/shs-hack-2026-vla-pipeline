"""Fixtures: an isolated registry/outputs folder and a small synthetic LeRobot v3 dataset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FPS = 10
DIM = 4


def make_dataset(root: Path, n_episodes: int = 10, frames: int = 30, broken: dict | None = None) -> Path:
    """broken: {episode_index: "nan" | "gap" | "short" | "nonmono"}"""
    broken = broken or {}
    rows = []
    index = 0
    for ep in range(n_episodes):
        n = 5 if broken.get(ep) == "short" else frames
        for f in range(n):
            ts = f / FPS
            if broken.get(ep) == "gap" and f >= n // 2:
                ts += 3 / FPS                       # 3 missing frames
            if broken.get(ep) == "nonmono" and f == n // 2:
                ts = 0.0
            action = np.full(DIM, 0.1 * f, dtype=np.float32)
            if broken.get(ep) == "nan" and f == 3:
                action[0] = np.nan
            rows.append({"timestamp": np.float32(ts), "frame_index": f, "episode_index": ep, "index": index,
                         "task_index": ep % 2, "observation.state": np.full(DIM, 0.05 * f, dtype=np.float32),
                         "action": action})
            index += 1
    df = pd.DataFrame(rows)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    df.to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    tasks = pd.DataFrame({"task_index": [0, 1]}, index=pd.Index(["pick red box", "pick blue box"], name="task"))
    tasks.to_parquet(root / "meta" / "tasks.parquet")
    info = {"codebase_version": "v3.0", "robot_type": "synthetic", "fps": FPS,
            "total_episodes": n_episodes, "total_frames": len(df),
            "features": {"observation.state": {"dtype": "float32", "shape": [DIM]},
                         "action": {"dtype": "float32", "shape": [DIM]},
                         "timestamp": {"dtype": "float32", "shape": [1]},
                         "frame_index": {"dtype": "int64", "shape": [1]},
                         "episode_index": {"dtype": "int64", "shape": [1]},
                         "index": {"dtype": "int64", "shape": [1]},
                         "task_index": {"dtype": "int64", "shape": [1]}}}
    (root / "meta" / "info.json").write_text(json.dumps(info))
    return root


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("HACKPIPE_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("HACKPIPE_REGISTRY", str(tmp_path / "registry.db"))
    from hackpipe.common.registry import Registry
    reg = Registry()
    yield tmp_path, reg
    reg.close()


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(tmp_path / "ds", broken={2: "nan", 4: "gap", 6: "short", 8: "nonmono"})
