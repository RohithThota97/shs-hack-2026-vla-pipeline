"""Fixtures: an isolated registry/outputs folder and a small synthetic LeRobot v3 dataset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_CONFIGS = Path(__file__).resolve().parents[1] / "configs"

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


def write_test_configs(cfg_dir: Path, **dataset_overrides) -> Path:
    """Repo configs adapted to the synthetic data (4-dim, short episodes, no held-out source)."""
    cfg_dir.mkdir(parents=True, exist_ok=True)
    project = yaml.safe_load((REPO_CONFIGS / "project.yaml").read_text())
    project["robot"].update({"state_dim": None, "action_dim": None})
    project["dataset"].update({"root": None, "repo_id": "local/synthetic", "revision": None})
    project["hub"]["model_repo_id"] = None
    ds = yaml.safe_load((REPO_CONFIGS / "dataset.yaml").read_text())
    ds["columns"]["success"] = None
    ds["quality"].update({"min_episode_frames": 10, "max_episode_frames": 100000, "joint_limits": None})
    ds["splits"].update({"train": 0.8, "val": 0.1, "test": 0.1, "test_sources": [], "leakage_keys": []})
    ds["preprocessing"]["rename_map"] = {}
    for k, v in dataset_overrides.items():
        section, key = k.split("__")
        ds[section][key] = v
    (cfg_dir / "project.yaml").write_text(yaml.safe_dump(project))
    (cfg_dir / "dataset.yaml").write_text(yaml.safe_dump(ds))
    (cfg_dir / "gates.yaml").write_text((REPO_CONFIGS / "gates.yaml").read_text())
    return cfg_dir


VIDEO_KEY = "observation.images.front"


def write_video(path: Path, frame_pts: list[int], fps: int = FPS, corrupt: bool = False) -> None:
    """Tiny MPEG-4 video whose frames sit at the given pts (in frames), so gaps = missing frames."""
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    if corrupt:
        path.write_bytes(b"not a video" * 100)
        return
    from fractions import Fraction
    with av.open(str(path), "w") as c:
        st = c.add_stream("mpeg4", rate=fps)
        st.width, st.height, st.pix_fmt = 32, 32, "yuv420p"
        st.codec_context.time_base = Fraction(1, fps)
        for p in frame_pts:
            fr = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24")
            fr.pts, fr.time_base = p, Fraction(1, fps)
            for pkt in st.encode(fr):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)


def add_fr3_extras(root: Path, n_episodes: int, frames: int, seeds: list[int] | None = None,
                   video: str | None = "ok", outcome: dict | None = None) -> Path:
    """Add v3 episode metadata, organiser-style per-episode records, stats.json and (optionally) one camera.
    video: "ok" | "short:<ep>" (drop 3 frames of that episode) | "corrupt" | "missing" | None (no camera)
    outcome: {episode_index: {"status": ..., "footprint_margin_mm": ...}} overrides."""
    seeds = seeds or [7000 + i for i in range(n_episodes)]
    outcome = outcome or {}
    rows, extras = [], []
    for ep in range(n_episodes):
        row = {"episode_index": ep, "length": frames, "data/chunk_index": 0, "data/file_index": 0}
        if video:
            row |= {f"videos/{VIDEO_KEY}/chunk_index": 0, f"videos/{VIDEO_KEY}/file_index": 0,
                    f"videos/{VIDEO_KEY}/from_timestamp": ep * frames / FPS,
                    f"videos/{VIDEO_KEY}/to_timestamp": (ep + 1) * frames / FPS}
        rows.append(row)
        o = outcome.get(ep, {})
        extras.append({"episode_index": ep, "name": f"syn_s{seeds[ep]}", "seed": seeds[ep],
                       "status": o.get("status", "PASS"), "success": o.get("status", "PASS") == "PASS",
                       "failure": None,
                       "randomization": {"applied": {"light": {
                           "rect": {"intensity_to": 3000.0 + ep, "color_temperature_K": 5000.0 + 10 * ep},
                           "dome": {"intensity_to": 500.0}, "rect_tilt_deg": [0.0, 0.0]}}},
                       "final_metrics": {"at_rest": True, "footprint_margin_mm": o.get("footprint_margin_mm", 50.0),
                                         "tilt_deg": 0.3, "placement_offset_mm": {"horizontal": 0.4}},
                       "stack_lock": {"root": "work/task_stage/syn_root_v1.usda"}})
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    (root / "meta" / "fr3_episodes_extra.jsonl").write_text("\n".join(json.dumps(e) for e in extras) + "\n")
    (root / "meta" / "stats.json").write_text(json.dumps({"action": {"mean": [0.0] * DIM}}))
    if video:
        info = json.loads((root / "meta" / "info.json").read_text())
        info["features"][VIDEO_KEY] = {"dtype": "video", "shape": [32, 32, 3], "info": {"video.fps": FPS}}
        (root / "meta" / "info.json").write_text(json.dumps(info))
        pts = list(range(n_episodes * frames))
        if video.startswith("short:"):
            k = int(video.split(":")[1])
            pts = [p for p in pts if not (k * frames + 5 <= p < k * frames + 8)]
        if video != "missing":
            write_video(root / "videos" / VIDEO_KEY / "chunk-000" / "file-000.mp4", pts,
                        corrupt=(video == "corrupt"))
    return root


@pytest.fixture(autouse=True)
def test_configs(tmp_path, monkeypatch):
    """Every test runs on synthetic-data configs, never on the real configs/."""
    monkeypatch.setenv("HACKPIPE_CONFIG_DIR", str(write_test_configs(tmp_path / "configs")))
    return tmp_path / "configs"


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
