"""Offline evaluation (no simulator): what a trained policy predicts on held-out demo frames vs what the demo did.

For every `every`-th frame on the policy's rate grid of the test (or val) episodes, the policy gets that frame's
cameras + state and an instruction — the complete task, and the frame's sub-task — and predicts an action chunk.
Reported per instruction kind and sub-task:
- `first_arm_rad`: arm-joint error of the first predicted action (the one executed next in closed loop)
- `chunk_arm_rad`: mean arm-joint error over the whole chunk (valid steps only)
- `gripper`: mean gripper error (0 = open … 1 = closed scale)
- `hold_still_arm_rad`: the same error for a do-nothing policy that repeats the current state, as scale

Executed stream (complete-task instruction): the first `every` actions of each prediction, as a policy that
re-plans every `every` steps would send them (teacher-forced: it always sees the recorded frame). From it:
- `executed`: arm error mean / p50 / p95 / p99, per sub-task error + DTW shape distance
- `gripper_events`: every crossing of 0.5 in the demo's gripper command — caught within ±1 s? timing error
- `smoothness`: largest joint jump per step, within a prediction vs at re-plan, next to the demo's own jumps
- `horizon_arm_rad`: error by how far ahead the step is (1 … chunk)
- `robustness`: the same chunk error with darker / brighter images, noisy joint readings, reworded or no
  instruction (first `robust_eps` episodes per source) — as a ratio to the unchanged baseline
- `latency_ms`: one prediction, batch 1 (the re-plan cost on the robot)
Metrics after the team's eval_offline.py (AI-project-work); gates in configs/gates.yaml → offline.

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
ROBUSTNESS = {"images_darker_x0.6": {"img_scale": 0.6}, "images_brighter_x1.4": {"img_scale": 1.4},
              "state_noise_0.01rad": {"state_noise": 0.01}, "state_noise_0.05rad": {"state_noise": 0.05},
              "reworded_instruction": {"task": "reworded"}, "no_instruction": {"task": ""}}


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


def _r(x: float | None, n: int = 4) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Dynamic-time-warping distance between two joint trajectories, per step of the warping path (rad)."""
    n, m = len(a), len(b)
    if not n or not m:
        return float("nan")
    cost = np.abs(a[:, None, :] - b[None, :, :]).mean(-1)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    steps = np.zeros((n + 1, m + 1), np.int64)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            prev = min((acc[i - 1, j - 1], steps[i - 1, j - 1]), (acc[i - 1, j], steps[i - 1, j]),
                       (acc[i, j - 1], steps[i, j - 1]))
            acc[i, j], steps[i, j] = cost[i - 1, j - 1] + prev[0], prev[1] + 1
    return float(acc[n, m] / steps[n, m])


def gripper_events(a: np.ndarray, g: int) -> list[tuple[int, str]]:
    """(step, "up" | "down") wherever the gripper command crosses 0.5."""
    hi = a[:, g] > 0.5
    return [(int(i), "up" if hi[i] else "down") for i in np.flatnonzero(np.diff(hi.astype(int))) + 1]


def stream_metrics(streams: list[dict[str, Any]], hz: float) -> dict[str, Any]:
    """Executed-stream metrics over episodes: {pred, gt (n, 16), sub (n,), replan (n,) bool, name}."""
    if not streams:
        return {}
    E = np.concatenate([np.abs(x["pred"] - x["gt"]) for x in streams])
    subs = np.concatenate([x["sub"] for x in streams])
    names = np.concatenate([np.full(len(x["sub"]), x["name"], object) for x in streams])
    per_sub = []
    for key in sorted({(n, t) for n, t in zip(names, subs)}):
        m = (names == key[0]) & (subs == key[1])
        d = [dtw(x["pred"][x["sub"] == key[1]][:, ARM], x["gt"][x["sub"] == key[1]][:, ARM])
             for x in streams if x["name"] == key[0]]
        per_sub.append({"subtask": f"{key[0]} · {key[1]}", "n": int(m.sum()), "arm_rad": _r(E[m][:, ARM].mean()),
                        "gripper": _r(E[m][:, GRIP].mean()), "dtw_rad": _r(np.nanmean(d) if d else None)})
    events = []
    tol = int(round(hz))                                  # ±1 s
    for x in streams:
        for g, side in ((GRIP[0], "left"), (GRIP[1], "right")):
            pred = gripper_events(x["pred"], g)
            for t, kind in gripper_events(x["gt"], g):
                cand = [p for p, k in pred if k == kind and abs(p - t) <= tol]
                best = min(cand, key=lambda p: abs(p - t)) if cand else None
                events.append({"source": x["name"], "ep": x["ep"], "gripper": side, "kind": kind,
                               "t_demo_s": round(t / hz, 2), "hit": best is not None,
                               "timing_error_s": round((best - t) / hz, 2) if best is not None else None})
    jumps_in, jumps_re, demo = [], [], []
    for x in streams:
        j = np.abs(np.diff(x["pred"][:, ARM], axis=0)).max(1)
        jumps_re += j[x["replan"][1:]].tolist()
        jumps_in += j[~x["replan"][1:]].tolist()
        demo += np.abs(np.diff(x["gt"][:, ARM], axis=0)).max(1).tolist()
    pct = lambda v, q: _r(np.percentile(v, q)) if len(v) else None       # noqa: E731
    hits = [e for e in events if e["hit"]]
    return {
        "executed": {"arm_mean": _r(E[:, ARM].mean()), "arm_p50": pct(E[:, ARM], 50), "arm_p95": pct(E[:, ARM], 95),
                     "arm_p99": pct(E[:, ARM], 99), "left_arm": _r(E[:, 0:7].mean()), "right_arm": _r(E[:, 8:15].mean()),
                     "gripper_mean": _r(E[:, GRIP].mean()), "steps": int(len(E)), "per_subtask": per_sub},
        "gripper_events": {"n": len(events), "hit_rate": _r(len(hits) / len(events), 3) if events else None,
                           "mean_abs_timing_error_s": _r(np.mean([abs(e["timing_error_s"]) for e in hits]), 3)
                           if hits else None, "events": events},
        "smoothness": {"within_prediction_p95": pct(jumps_in, 95), "at_replan_p95": pct(jumps_re, 95),
                       "demo_p95": pct(demo, 95)},
    }


def score(tables: list[dict[str, Any]], predict: Callable[..., np.ndarray], chunk: int, every: int = 3,
          batch: int = 8, progress: Callable[[str], None] | None = None, hz: float = 6.0,
          robust: dict[str, dict[str, Any]] | None = None, robust_eps: int = 3) -> dict[str, Any]:
    """predict(table, rows, mode[, variant=…]) → (len(rows), chunk, 16) actions; mode "full" or "sub"; variant only
    for robustness (img_scale, state_noise, task). Pure numpy otherwise."""
    from hackpipe.training.curriculum import chunk_for
    if every > chunk:
        raise ValueError(f"every={every} is longer than the chunk ({chunk}): the executed stream would have gaps")
    acc: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    hor_sum, hor_n = np.zeros(chunk), np.zeros(chunk)
    streams: dict[tuple[str, int], dict[str, list]] = {}
    n_pred, t0 = 0, time.time()

    def call(t, rs, mode, variant=None):
        pred = np.asarray(predict(t, rs, mode, variant=variant) if variant is not None else predict(t, rs, mode),
                          np.float64)
        if pred.ndim != 3 or pred.shape[1] < chunk:
            raise ValueError(f"the policy returned {pred.shape[1:2]} actions per prediction, expected "
                             f"{chunk}: set the chunk to the checkpoint's chunk_size")
        return pred

    for t in tables:
        rows = t["rows"][::max(1, every)]
        for mode, m in (("full", 1), ("sub", 0)):
            for b in range(0, len(rows), batch):
                rs = rows[b:b + batch]
                pred = call(t, rs, mode)
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
                    if mode == "full":
                        err = np.abs(p - gt)[:, ARM].mean(1)
                        hor_sum[:len(err)][valid] += err[valid]
                        hor_n[:len(err)][valid] += 1
                        ex = valid.copy()
                        ex[every:] = False
                        st = streams.setdefault((t["name"], int(t["ep"][r])), {"pred": [], "gt": [], "sub": [],
                                                                               "replan": []})
                        n_ex = int(ex.sum())
                        st["pred"].append(p[ex])
                        st["gt"].append(gt[ex])
                        st["sub"] += [str(t["sub_text"][i]) for i in idx[ex]]
                        st["replan"] += [True] + [False] * (n_ex - 1)
                n_pred += len(rs)
                if progress and (b // batch) % 25 == 0:
                    progress(f"{t['name']} {mode}: {b + len(rs)}/{len(rows)} frames, {time.time() - t0:.0f}s")
    rows_out = [{"instruction": mode, "subtask": key, "n": len(v["first_arm_rad"]),
                 **{m: round(float(np.mean(x)), 4) for m, x in v.items()}}
                for (mode, key), v in sorted(acc.items())]
    out: dict[str, Any] = {"rows": rows_out, "predictions": n_pred,
                           "horizon_arm_rad": [_r(x) for x in hor_sum / np.maximum(hor_n, 1)]}
    out.update(stream_metrics([{"name": n, "ep": e, "pred": np.concatenate(v["pred"]), "gt": np.concatenate(v["gt"]),
                                "sub": np.array(v["sub"], object), "replan": np.array(v["replan"], bool)}
                               for (n, e), v in sorted(streams.items()) if v["pred"]], hz))
    if robust:
        res: dict[str, list[float]] = defaultdict(list)
        for t in tables:
            eps = sorted(set(t["ep"][t["rows"]].tolist()))[:robust_eps]
            rows = t["rows"][::max(1, every)]
            rows = rows[np.isin(t["ep"][rows], eps)]
            for name, var in {"baseline": {}, **robust}.items():
                for b in range(0, len(rows), batch):
                    rs = rows[b:b + batch]
                    pred = call(t, rs, "full", var)
                    for k, r in enumerate(rs):
                        idx, pad, _ = chunk_for(t, r, 1, chunk, "mask")
                        p, gt = pred[k][:len(idx)], t["A"][idx].astype(np.float64)
                        res[name].append(float(np.abs(p[~pad][:, ARM] - gt[~pad][:, ARM]).mean()))
                    n_pred += len(rs)
                if progress:
                    progress(f"{t['name']} robustness {name}: done, {time.time() - t0:.0f}s")
        base = float(np.mean(res["baseline"]))
        out["robustness"] = {"episodes_per_source": robust_eps, "baseline_chunk_arm_rad": _r(base),
                             "variants": {k: {"chunk_arm_rad": _r(np.mean(v)),
                                              "ratio": _r(np.mean(v) / base, 3) if base > 0 else None}
                                          for k, v in res.items() if k != "baseline"}}
    out.update(predictions=n_pred, seconds=round(time.time() - t0, 1))
    return out


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

    from hackpipe.training.curriculum import reword
    rng = np.random.default_rng(0)

    @torch.no_grad()
    def predict(t: dict[str, Any], rows: np.ndarray, mode: str, variant: dict[str, Any] | None = None) -> np.ndarray:
        v = variant or {}
        items = [lr[t["name"]][int(t["abs_index"][r])] for r in rows]
        missing = [c for c in cams if c not in items[0]]
        if missing:
            raise KeyError(f"the policy expects cameras {missing}, {t['name']} has "
                           f"{[k for k in items[0] if 'image' in k]}")
        batch = {c: torch.stack([it[c] for it in items]) for c in cams}
        if v.get("img_scale"):                                          # darker / brighter cameras
            batch = {c: (x * float(v["img_scale"])).clamp(0, 1) for c, x in batch.items()}
        state = np.stack([t["S"][r] for r in rows]).astype(np.float32)
        if v.get("state_noise"):                                        # noisy joint readings (arms only)
            state[:, ARM] += rng.normal(0, float(v["state_noise"]), (len(rows), len(ARM))).astype(np.float32)
        batch["observation.state"] = torch.from_numpy(state)
        texts = [str(t["full_text"][r] if mode == "full" else t["sub_text"][r]) for r in rows]
        if "task" in v:
            texts = [reword(x, 1) for x in texts] if v["task"] == "reworded" else [v["task"]] * len(rows)
        batch["task"] = texts
        policy.reset()
        out = post(policy.predict_action_chunk(pre(batch)))
        return out.float().cpu().numpy()

    return predict, getattr(cfg, "chunk_size", None)


def latency(predict: Callable[..., np.ndarray], table: dict[str, Any], rate_hz: float, n: int = 6) -> dict[str, Any]:
    """One prediction with batch 1 (what a re-plan costs on the robot); the first call is warm-up."""
    ms = []
    for r in table["rows"][:: max(1, len(table["rows"]) // n)][: n + 1]:
        t0 = time.perf_counter()
        predict(table, np.array([r]), "full")
        ms.append((time.perf_counter() - t0) * 1000)
    ms = ms[1:] or ms
    return {"replan_median": round(float(np.median(ms)), 1), "replan_max": round(float(np.max(ms)), 1),
            "budget_ms": round(1000 / rate_hz, 1), "calls": len(ms)}


def pending(reg: Registry, split: str = "test") -> list[str]:
    """Finished models (not timing runs) with a checkpoint here and no offline eval of this split yet."""
    out_dir = resolve(project_config()["project"]["outputs_dir"]) / "runs"
    return [m["version"] for m in reg.models()
            if m["meta"].get("purpose") != "timing" and m["status"] in ("trained", "evaluated", "deployed")
            and (resolve(m["checkpoint_dir"] or ".") / "config.json").is_file()
            and not (out_dir / m["version"] / f"offline_eval_{split}.json").is_file()]


def offline_eval(model_version: str, registry: Registry | None = None, split: str = "test", every: int = 3,
                 batch: int = 8, device: str | None = None, progress: Callable[[str], None] | None = print,
                 robustness: bool = True, robust_eps: int = 3) -> dict[str, Any]:
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
    res = score(tables, predict, int(chunk or ck_chunk or 50), every, batch, progress, hz=rate,
                robust=ROBUSTNESS if robustness else None, robust_eps=robust_eps)
    res["latency_ms"] = latency(predict, tables[0], rate)
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
