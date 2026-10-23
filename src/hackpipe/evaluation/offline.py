"""Offline evaluation (no simulator): what a trained policy predicts on held-out demo frames vs what the demo did.

For every `every`-th frame on the policy's rate grid of the test (or val) episodes, the policy gets that frame's
cameras + state and an instruction — the complete task, and the frame's sub-task — and predicts an action chunk.
Reported per instruction kind and sub-task:
- `first_arm_rad`: arm-joint error of the first predicted action (the one executed next in closed loop)
- `chunk_arm_rad`: mean arm-joint error over the whole chunk (valid steps only)
- `gripper`: mean gripper error (0 = open … 1 = closed scale)
- `hold_still_arm_rad`: the same error for a do-nothing policy that repeats the current state, as scale

Low numbers here are necessary, not sufficient: closed-loop success needs the sim (configs/eval/sim_*.yaml).
The test split is the frozen held-out source; use split="val" while tuning.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hackpipe.common.config import project_config, resolve
from hackpipe.common.ids import now_iso
from hackpipe.common.registry import Registry

ARM = [*range(7), *range(8, 15)]
GRIP = [7, 15]


def model_rate(model: dict[str, Any], fps: float) -> tuple[float, int | None]:
    """(policy rate in Hz, chunk length or None = from the checkpoint) from the model's run."""
    out = Path(model["meta"].get("output_dir") or "")
    cfg_file = resolve(out) / "train_config.json"
    if model["meta"].get("trainer") == "curriculum" and cfg_file.exists():
        a = json.loads(cfg_file.read_text()).get("args") or {}
        return float(a.get("rate_hz", 6)), int(a.get("chunk", 20))
    return float(fps), None


def model_sources(model: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    """Folder names the model was trained on: the curriculum's train_sources, else the version's trainable ones."""
    from hackpipe.dataset.versioning import trainable_sources
    from hackpipe.training.curriculum import source_names
    cfg_file = resolve(model["meta"].get("output_dir") or ".") / "train_config.json"
    if model["meta"].get("trainer") == "curriculum" and cfg_file.exists():
        names = source_names((json.loads(cfg_file.read_text()).get("args") or {}).get("train_sources"))
        if names:
            return names
        return sorted({Path(s).name for s in (manifest.get("subtasks") or {}).get("sources", {})})
    return [Path(s).name for s in trainable_sources(manifest)[0]]


def eval_tables(manifest: dict[str, Any], split: str, rate_hz: float,
                sources: list[str] | None = None) -> list[dict[str, Any]]:
    """Frame tables of the split's episodes with sub-task labels (labelled on the fly — the test source has none).
    sources: folder names to include (None = all of the split)."""
    from hackpipe.common.config import project_config as pc
    from hackpipe.dataset.subtasks import Joints, kind_of, label_source, subtasks_config
    from hackpipe.training.curriculum import build_table
    cfg = subtasks_config()
    j = Joints(cfg)
    eps: dict[str, list[int]] = defaultdict(list)
    for s, i in manifest["episodes"][split]:
        if sources is None or Path(s).name in sources:
            eps[s].append(int(i))
    if not eps:
        raise RuntimeError(f"no {split} episodes of {sources} in {manifest.get('version')}")
    tables = []
    for src, e in sorted(eps.items()):
        kind = kind_of(src, cfg) or kind_of(pc()["dataset"]["root"], cfg)
        if kind is None:
            raise RuntimeError(f"no sub-task rules for {src}: add it to configs/subtasks.yaml → sources")
        lab = label_source(src, kind, sorted(e), cfg, progress=None)
        t = build_table(src, sorted(e), {**lab, "flagged": []},
                        {"rate_hz": rate_hz, "gripper_lag_fix": False}, [j.l_g, j.r_g])
        tables.append(t)
    return tables


def score(tables: list[dict[str, Any]], predict: Callable[[dict[str, Any], np.ndarray, str], np.ndarray],
          chunk: int, every: int = 3, batch: int = 8, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """predict(table, rows, mode) → (len(rows), chunk, 16) actions; mode "full" or "sub". Pure numpy otherwise."""
    from hackpipe.training.curriculum import chunk_for
    acc: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    n_pred, t0 = 0, time.time()
    for t in tables:
        rows = t["rows"][::max(1, every)]
        for mode, m in (("full", 1), ("sub", 0)):
            for b in range(0, len(rows), batch):
                rs = rows[b:b + batch]
                pred = np.asarray(predict(t, rs, mode), np.float64)
                if pred.ndim != 3 or pred.shape[1] < chunk:
                    raise ValueError(f"the policy returned {pred.shape[1:2]} actions per prediction, expected "
                                     f"{chunk}: set the chunk to the checkpoint's chunk_size")
                for k, r in enumerate(rs):
                    idx, pad, _ = chunk_for(t, r, m, chunk, "mask")
                    valid = ~pad
                    gt = t["A"][idx].astype(np.float64)
                    p = pred[k][:len(idx)]
                    key = (mode, f"{t['name']} · {t['sub_text'][r]}")
                    for kk in (key, (mode, f"{t['name']} · all")):
                        acc[kk]["first_arm_rad"].append(float(np.abs(p[0, ARM] - gt[0, ARM]).mean()))
                        acc[kk]["chunk_arm_rad"].append(float(np.abs(p[valid][:, ARM] - gt[valid][:, ARM]).mean()))
                        acc[kk]["gripper"].append(float(np.abs(p[valid][:, GRIP] - gt[valid][:, GRIP]).mean()))
                        acc[kk]["hold_still_arm_rad"].append(
                            float(np.abs(t["S"][r][ARM].astype(np.float64) - gt[valid][:, ARM]).mean()))
                n_pred += len(rs)
                if progress and (b // batch) % 25 == 0:
                    progress(f"{t['name']} {mode}: {b + len(rs)}/{len(rows)} frames, {time.time() - t0:.0f}s")
    rows_out = [{"instruction": mode, "subtask": key, "n": len(v["first_arm_rad"]),
                 **{m: round(float(np.mean(x)), 4) for m, x in v.items()}}
                for (mode, key), v in sorted(acc.items())]
    return {"rows": rows_out, "predictions": n_pred, "seconds": round(time.time() - t0, 1)}


def lerobot_predictor(checkpoint: Path, tables: list[dict[str, Any]], device: str | None = None):
    """predict() for score(): the checkpoint (LeRobot format, with its pre/post processors) on dataset frames."""
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(str(checkpoint), config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(cfg, str(checkpoint),
                                         preprocessor_overrides={"device_processor": {"device": device}})
    lr = {t["name"]: LeRobotDataset(f"local/{t['name']}", root=resolve(t["source"]), video_backend="pyav")
          for t in tables}
    cams = [k for k in cfg.input_features if "image" in k]

    @torch.no_grad()
    def predict(t: dict[str, Any], rows: np.ndarray, mode: str) -> np.ndarray:
        items = [lr[t["name"]][int(t["abs_index"][r])] for r in rows]
        missing = [c for c in cams if c not in items[0]]
        if missing:
            raise KeyError(f"the policy expects cameras {missing}, {t['name']} has "
                           f"{[k for k in items[0] if 'image' in k]}")
        batch = {c: torch.stack([it[c] for it in items]) for c in cams}
        batch["observation.state"] = torch.from_numpy(np.stack([t["S"][r] for r in rows]))
        batch["task"] = [str(t["full_text"][r] if mode == "full" else t["sub_text"][r]) for r in rows]
        policy.reset()
        out = post(policy.predict_action_chunk(pre(batch)))
        return out.float().cpu().numpy()

    return predict, getattr(cfg, "chunk_size", None)


def offline_eval(model_version: str, registry: Registry | None = None, split: str = "test", every: int = 3,
                 batch: int = 8, device: str | None = None, progress: Callable[[str], None] | None = print
                 ) -> dict[str, Any]:
    from hackpipe.common.lerobot_io import DatasetFolder
    from hackpipe.dataset.versioning import load_manifest
    reg = registry or Registry()
    model = reg.get_model(model_version)
    if model is None:
        raise KeyError(f"Unknown model {model_version}")
    ckpt = resolve(model["checkpoint_dir"] or "")
    if not (ckpt / "config.json").exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt} (config.json missing)")
    m = load_manifest(model["dataset_version"], reg)
    # val: only what the model was trained on; test: the held-out sources (same task, never trained on)
    names = (model_sources(model, m) if split == "val"
             else sorted({Path(s).name for s, _ in m["episodes"]["test"]}))
    srcs = sorted({s for s, _ in m["episodes"][split] if Path(s).name in names})
    if not srcs:
        raise RuntimeError(f"{model['dataset_version']} has no {split} episodes of {names}")
    rate, chunk = model_rate(model, DatasetFolder(resolve(srcs[0])).fps)
    tables = eval_tables(m, split, rate, names)
    predict, ck_chunk = lerobot_predictor(ckpt, tables, device)
    res = score(tables, predict, int(chunk or ck_chunk or 50), every, batch, progress)
    res.update({"model": model_version, "checkpoint": str(ckpt), "dataset_version": model["dataset_version"],
                "split": split, "rate_hz": rate, "every": every, "episodes": {t["source"]: len(set(t["ep"][t["rows"]]))
                                                                            for t in tables},
                "created_at": now_iso()})
    run_dir = resolve(project_config()["project"]["outputs_dir"]) / "runs" / model_version
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"offline_eval_{split}.json"
    out.write_text(json.dumps(res, indent=1))
    res["file"] = str(out)
    return res
