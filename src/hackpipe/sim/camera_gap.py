"""How different the evaluation scene looks from the training renders (diagnosis step A3).

`record_demos.py --mode snapshot` saves the 3 cameras at reset for some training episodes, each lit like that
episode: snapshots/<episode_index>/<camera>.png. Here each is compared with frame 0 of the same episode in the
training data (same robot pose, same lighting), as mean absolute pixel difference (0–255).

For scale, the same number is computed between frame 0 of two *training* episodes (only the lighting differs):
a sim gap far above that means the cameras see a different-looking scene than in training (missing textures,
other camera parameters) — then re-rendered training data (record_demos.py --mode render) is needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hackpipe.common.config import resolve
from hackpipe.common.lerobot_io import DatasetFolder

CAMS = ("external_camera", "left_wrist_camera", "right_wrist_camera")


def _mad(a: np.ndarray, b: np.ndarray) -> float:
    return round(float(np.abs(a.astype(float) - b.astype(float)).mean()), 2)


def camera_gap(snapshots: str | Path, source: str | Path, out_png: str | Path | None = None) -> dict[str, Any]:
    from PIL import Image

    from hackpipe.dataset.subtasks import Videos
    ds = DatasetFolder(resolve(source))
    v = Videos(ds)
    snap = Path(snapshots)
    eps = sorted(int(p.name) for p in snap.iterdir() if p.is_dir() and p.name.isdigit())
    if not eps:
        raise FileNotFoundError(f"no snapshots/<episode>/ folders in {snap}")
    rows, sheet = [], []
    for ep in eps:
        train = [v.frames(ep, c, [0])[0] for c in CAMS]
        sim = [np.asarray(Image.open(snap / str(ep) / f"{c}.png").convert("RGB")) for c in CAMS]
        rows.append({"episode": ep, **{c: _mad(s, t) for c, s, t in zip(CAMS, sim, train)}})
        sheet.append(np.concatenate([np.concatenate(train, 1), np.concatenate(sim, 1)], 0))
    others = sorted(set(int(i) for i in ds.episode_extras()) - set(eps)) or eps[1:]   # any other episode
    a, b = eps[0], (others[0] if others else eps[0])
    baseline = {c: _mad(v.frames(a, c, [0])[0], v.frames(b, c, [0])[0]) for c in CAMS} if a != b else None
    mean = {c: round(float(np.mean([r[c] for r in rows])), 2) for c in CAMS}
    verdict = {c: (None if not baseline else
                   "like training" if mean[c] <= 2 * max(baseline[c], 1.0) else "differs from training")
               for c in CAMS}
    if out_png:
        Image.fromarray(np.concatenate(sheet, 0)).save(out_png)       # per episode: training row, sim row
    return {"episodes": rows, "mean_abs_diff": mean, "between_training_episodes": baseline,
            "baseline_pair": [a, b], "verdict": verdict, "sheet": str(out_png) if out_png else None}
