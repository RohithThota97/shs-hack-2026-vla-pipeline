"""Isaac Sim recordings (`scripts/isaac/record_demos.py`) → a LeRobot v3 folder like the organisers' datasets.

A recording folder holds one sub-folder per episode:
    meta.json        name, mode (render | dart | grid), source + episode_index it was made from, seed, light, success,
                     stage, final box pose / tilt, noise settings
    state.npy        (T, 16) float32 — measured state at frame k (physics tick 2k)
    action.npy       (T, 16) float32 — target for the next 2 ticks (the demo's action, also under noise)
    cams/<camera>/000000.jpg …   one image per frame per camera (224 × 224)

Written like the source dataset (`template`): same features, fps and path templates, AV1 videos with one file per
episode, meta/episodes, tasks, stats.json, info.json, plus organiser-style `fr3_episodes_extra.jsonl` with
`derived_from` (the episode it was rendered from), so a new dataset version keeps each copy in its original's split.
Failed episodes (box not on the tray) are left out unless keep_failed.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hackpipe.common.config import dataset_config, resolve, source_key
from hackpipe.common.lerobot_io import DatasetFolder

QUANTILES = (1, 10, 50, 90, 99)


def _episodes(rec_dir: Path, keep_failed: bool) -> list[tuple[Path, dict[str, Any]]]:
    out = []
    for d in sorted(p for p in rec_dir.iterdir() if (p / "meta.json").is_file()):
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("success") or keep_failed:
            out.append((d, meta))
    return out


def _stats(x: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, np.float64)
    st = {"min": x.min(0), "max": x.max(0), "mean": x.mean(0), "std": x.std(0), "count": [len(x)]}
    st.update({f"q{q:02d}": np.percentile(x, q, axis=0) for q in QUANTILES})
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in st.items()}


def _image_stats(frames: list[np.ndarray]) -> dict[str, Any]:
    """Per-channel stats of sampled frames in 0..1, shaped [3, 1, 1] like LeRobot's."""
    px = np.concatenate([f.reshape(-1, 3) for f in frames]).astype(np.float64) / 255
    st = _stats(px)
    return {k: ([[[v]] for v in vals] if k != "count" else [len(frames)]) for k, vals in st.items()}


def _encode(frames_dir: Path, out: Path, fps: int, n: int) -> list[np.ndarray]:
    """jpg frames → AV1 mp4 (same codec / GOP as the organisers' videos). Returns a few frames for stats."""
    import os

    import av
    from PIL import Image
    os.environ.setdefault("SVT_LOG", "1")          # SVT-AV1: errors only, not its banner per file
    out.parent.mkdir(parents=True, exist_ok=True)
    sample = []
    with av.open(str(out), "w") as c:
        st = c.add_stream("libsvtav1", rate=fps)
        st.width = st.height = 224
        st.pix_fmt = "yuv420p"
        st.options = {"crf": "30", "preset": "12", "g": "2"}
        for k in range(n):
            img = np.asarray(Image.open(frames_dir / f"{k:06d}.jpg").convert("RGB"))
            if img.shape[:2] != (224, 224):
                raise ValueError(f"{frames_dir}/{k:06d}.jpg is {img.shape[:2]}, expected 224 x 224")
            if k % max(1, n // 8) == 0:
                sample.append(img)
            for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
    return sample


def convert(rec_dir: str | Path, out_root: str | Path, template: str | Path | None = None,
            keep_failed: bool = False, task: str | None = None) -> dict[str, Any]:
    rec_dir, out_root = Path(rec_dir), resolve(out_root)
    from hackpipe.common.config import project_config
    tpl = DatasetFolder(resolve(template or project_config()["dataset"]["root"]))
    cols = dataset_config()["columns"]
    eps = _episodes(rec_dir, keep_failed)
    if not eps:
        raise RuntimeError(f"no {'recorded' if keep_failed else 'successful'} episodes in {rec_dir}")
    if out_root.exists():
        raise FileExistsError(f"{out_root} exists — pick another name or remove it")
    fps = int(tpl.fps)
    cams = tpl.video_keys
    task = task or next(iter(tpl.tasks().values()))
    tmp = out_root.with_name(out_root.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    rows, ep_rows, extras, samples = [], [], [], {c: [] for c in cams}
    index = 0
    for e, (d, meta) in enumerate(eps):
        S, A = np.load(d / "state.npy").astype(np.float32), np.load(d / "action.npy").astype(np.float32)
        n = len(S)
        if len(A) != n or S.shape[1] != 16:
            raise ValueError(f"{d}: state {S.shape} / action {A.shape} don't match")
        for k in range(n):
            rows.append({cols["state"]: S[k], cols["action"]: A[k], "success": bool(meta.get("success")),
                         "raw_state_row": 2 * k, "raw_action_row": 2 * k + 2,
                         cols["timestamp"]: np.float32(k / fps), cols["frame"]: k, cols["episode"]: e,
                         "index": index + k, cols["task_index"]: 0})
        row = {"episode_index": e, "tasks": [task], "length": n, "data/chunk_index": 0, "data/file_index": 0,
               "dataset_from_index": index, "dataset_to_index": index + n,
               "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0}
        for c in cams:
            samples[c] += _encode(d / "cams" / c.split(".")[-1], tmp / tpl.info["video_path"].format(
                video_key=c, chunk_index=0, file_index=e), fps, n)
            row |= {f"videos/{c}/chunk_index": 0, f"videos/{c}/file_index": e,
                    f"videos/{c}/from_timestamp": 0.0, f"videos/{c}/to_timestamp": n / fps}
        ep_rows.append(row)
        src = meta.get("source")
        extras.append({"episode_index": e, "name": f"{meta.get('mode', 'sim')}_{meta.get('name')}",
                       "seed": meta.get("seed"), "status": "PASS" if meta.get("success") else "FAIL",
                       "success": bool(meta.get("success")), "failure": None if meta.get("success") else "sim",
                       "randomization": {"applied": {"light": {   # only what was actually rendered
                           k: v for k, v in (meta.get("light") or {}).items()
                           if k != "rect_tilt_deg" or meta.get("light_tilt_applied")}}},
                       "final_box_pose": meta.get("final_box_pose"),
                       "final_metrics": {"at_rest": bool(meta.get("success")), "tilt_deg": meta.get("box_tilt_deg")},
                       "derived_from": {"source": source_key(resolve(src)) if src else None,
                                        "episode_index": meta.get("source_episode")},
                       "sim": {k: meta.get(k) for k in ("mode", "stage", "noise", "grid", "recorder", "usd")}})
        index += n
    (tmp / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(tmp / tpl.info["data_path"].format(chunk_index=0, file_index=0))
    (tmp / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pd.DataFrame(ep_rows).to_parquet(tmp / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index([task], name="task")).to_parquet(tmp / "meta" / "tasks.parquet")
    stats = {cols["state"]: _stats(np.stack(df[cols["state"]])), cols["action"]: _stats(np.stack(df[cols["action"]]))}
    for k in (cols["timestamp"], cols["frame"], cols["episode"], "index", cols["task_index"]):
        stats[k] = _stats(df[[k]].to_numpy())
    stats.update({c: _image_stats(samples[c]) for c in cams})
    (tmp / "meta" / "stats.json").write_text(json.dumps(stats))
    info = {**tpl.info, "total_episodes": len(eps), "total_frames": index, "total_tasks": 1,
            "splits": {"train": f"0:{len(eps)}"}}
    (tmp / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    (tmp / "meta" / "fr3_episodes_extra.jsonl").write_text("".join(json.dumps(x) + "\n" for x in extras))
    conv = tpl.root / "meta" / "fr3_conversion.json"
    if conv.exists():
        c = json.loads(conv.read_text())
        c["converter"] = {"name": "hackpipe.ingest.sim_recording", "from": str(rec_dir)}
        (tmp / "meta" / "fr3_conversion.json").write_text(json.dumps(c, indent=1))
    out_root.parent.mkdir(parents=True, exist_ok=True)
    tmp.rename(out_root)
    return {"root": str(out_root), "episodes": len(eps), "frames": index,
            "left_out_failed": sum(1 for p in rec_dir.iterdir() if (p / "meta.json").is_file()) - len(eps)}
