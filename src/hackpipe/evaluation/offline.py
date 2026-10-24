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
- `gripper_events`: every grasp / release in the demo's gripper command (crossing 0.25) — caught within ±1 s?
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
    for d in range(2, n + m + 1):             # cells on one anti-diagonal depend only on the two before it
        i = np.arange(max(1, d - m), min(n, d - 1) + 1)
        j = d - i
        cand = np.stack([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        cs = np.stack([steps[i - 1, j - 1], steps[i - 1, j], steps[i, j - 1]])
        k = cand.argmin(0)
        acc[i, j] = cost[i - 1, j - 1] + cand[k, np.arange(len(i))]
        steps[i, j] = cs[k, np.arange(len(i))] + 1
    return float(acc[n, m] / steps[n, m])


GRIP_CLOSED_BELOW = 0.25   # FR3 Duo gripper command: 1.0 start, 0.48 open (pre-grasp), 0.0 closed on the box


def gripper_events(a: np.ndarray, g: int) -> list[tuple[int, str]]:
    """(step, "close" | "open") wherever the gripper command crosses GRIP_CLOSED_BELOW: the grasps and releases (the
    1.0 → 0.48 pre-shape at the start is not an event)."""
    closed = a[:, g] < GRIP_CLOSED_BELOW
    return [(int(i), "close" if closed[i] else "open") for i in np.flatnonzero(np.diff(closed.astype(int))) + 1]


def stream_metrics(streams: list[dict[str, Any]], hz: float) -> dict[str, Any]:
    """Executed-stream metrics over episodes: {pred, gt, still (n, 16), sub (n,), replan (n,) bool, name}. `still` =
    the state at the prediction's start (doing nothing), so model and doing nothing are judged on the same steps."""
    if not streams:
        return {}
    E = np.concatenate([np.abs(x["pred"] - x["gt"]) for x in streams])
    H = np.concatenate([np.abs(x["still"] - x["gt"]) for x in streams])
    subs = np.concatenate([x["sub"] for x in streams])
    names = np.concatenate([np.full(len(x["sub"]), x["name"], object) for x in streams])
    per_sub = []
    for key in sorted({(n, t) for n, t in zip(names, subs)}):
        m = (names == key[0]) & (subs == key[1])
        d = [dtw(x["pred"][x["sub"] == key[1]][:, ARM], x["gt"][x["sub"] == key[1]][:, ARM])
             for x in streams if x["name"] == key[0]]
        per_sub.append({"subtask": f"{key[0]} · {key[1]}", "n": int(m.sum()), "arm_rad": _r(E[m][:, ARM].mean()),
                        "hold_still_rad": _r(H[m][:, ARM].mean()),
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
        "executed": {"arm_mean": _r(E[:, ARM].mean()), "hold_still_arm_mean": _r(H[:, ARM].mean()),
                     "arm_p50": pct(E[:, ARM], 50), "arm_p95": pct(E[:, ARM], 95),
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
    full_err: dict[tuple[str, int], float] = {}        # chunk error per row, complete task: the robustness baseline
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
                        full_err[(t["name"], int(r))] = float(np.abs(p[valid][:, ARM] - gt[valid][:, ARM]).mean())
                    if mode == "full":
                        err = np.abs(p - gt)[:, ARM].mean(1)
                        hor_sum[:len(err)][valid] += err[valid]
                        hor_n[:len(err)][valid] += 1
                        ex = valid.copy()
                        ex[every:] = False
                        st = streams.setdefault((t["name"], int(t["ep"][r])), {"pred": [], "gt": [], "sub": [],
                                                                               "replan": [], "still": []})
                        n_ex = int(ex.sum())
                        st["pred"].append(p[ex])
                        st["gt"].append(gt[ex])
                        st["still"].append(np.repeat(t["S"][r][None].astype(np.float64), n_ex, 0))
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
    episodes = [{"name": n, "ep": e, "pred": np.concatenate(v["pred"]), "gt": np.concatenate(v["gt"]),
                 "still": np.concatenate(v["still"]), "sub": np.array(v["sub"], object),
                 "replan": np.array(v["replan"], bool)} for (n, e), v in sorted(streams.items()) if v["pred"]]
    out.update(stream_metrics(episodes, hz))
    out["_streams"] = episodes                       # trajectories, saved next to the result (not in the JSON)
    if robust:
        res: dict[str, list[float]] = defaultdict(list)
        for t in tables:
            eps = sorted(set(t["ep"][t["rows"]].tolist()))[:robust_eps]
            rows = t["rows"][::max(1, every)]
            rows = rows[np.isin(t["ep"][rows], eps)]
            res["baseline"] += [full_err[(t["name"], int(r))] for r in rows]   # already predicted above
            for name, var in robust.items():
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
        t0 = time.perf_counter()                       # the model only (frame decoding above is not robot time)
        out = post(policy.predict_action_chunk(pre(batch))).float().cpu().numpy()
        predict.infer_ms = (time.perf_counter() - t0) * 1000
        return out

    return predict, getattr(cfg, "chunk_size", None)


def latency(predict: Callable[..., np.ndarray], table: dict[str, Any], rate_hz: float, n: int = 6) -> dict[str, Any]:
    """One prediction with batch 1 (what a re-plan costs on the robot: the model, not the frame decoding); the first
    call is warm-up."""
    ms = []
    for r in table["rows"][:: max(1, len(table["rows"]) // n)][: n + 1]:
        t0 = time.perf_counter()
        predict(table, np.array([r]), "full")
        ms.append(getattr(predict, "infer_ms", None) or (time.perf_counter() - t0) * 1000)
    ms = ms[1:] or ms
    return {"replan_median": round(float(np.median(ms)), 1), "replan_max": round(float(np.max(ms)), 1),
            "budget_ms": round(1000 / rate_hz, 1), "calls": len(ms)}


def result_path(model_version: str, split: str = "test", step: int | None = None) -> Path:
    """outputs/runs/<model>/offline_eval_<split>.json (the model's current checkpoint) or …_<split>_<step>.json (one
    saved checkpoint, e.g. tested during training) — the one place that names the files."""
    name = f"offline_eval_{split}.json" if step is None else f"offline_eval_{split}_{int(step):06d}.json"
    return resolve(project_config()["project"]["outputs_dir"]) / "runs" / model_version / name


def load_result(model_version: str, split: str = "test", step: int | None = None) -> dict[str, Any] | None:
    f = result_path(model_version, split, step)
    return json.loads(f.read_text()) if f.is_file() else None


def trajectories(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The executed streams saved next to a result: per episode name, ep, pred, gt, still (n, 16), sub, replan."""
    if not result or not result.get("file"):
        return []
    f = Path(result["file"]).with_suffix(".npz")
    if not f.is_file():
        return []
    z = np.load(f, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return [{**m, **{k: z[f"{i}_{k}"] for k in ("pred", "gt", "still", "replan")},
             "sub": z[f"{i}_sub"].astype(str)} for i, m in enumerate(meta)]


def step_of(checkpoint: str | Path | None) -> int | None:
    """…/checkpoints/<step>/pretrained_model → step (None for `last` or other folders)."""
    p = Path(str(checkpoint or ""))
    return int(p.parent.name) if p.name == "pretrained_model" and p.parent.name.isdigit() else None


def result_for(model: dict[str, Any], split: str = "test") -> dict[str, Any] | None:
    """The test of the checkpoint the model points at now: the main result if it matches, else that step's test
    (made during training), else the main result (stale — is_current() says so)."""
    main = load_result(model["version"], split)
    if is_current(main, model):
        return main
    step = step_of(model["checkpoint_dir"])
    by_step = load_result(model["version"], split, step) if step is not None else None
    return by_step if is_current(by_step, model) else main


def pending(reg: Registry, split: str = "test") -> list[str]:
    """Finished models (not timing runs) with a checkpoint here, no current offline eval of this split (none, one
    from before the executed-stream metrics, or one of another checkpoint) and none running."""
    from hackpipe.training.ranking import FINISHED
    return [m["version"] for m in reg.models()
            if m["meta"].get("purpose") != "timing" and m["status"] in FINISHED
            and (resolve(m["checkpoint_dir"] or ".") / "config.json").is_file()
            and not is_current(load_result(m["version"], split), m) and not is_running(m["version"], split)]


def offline_eval(model_version: str, registry: Registry | None = None, split: str = "test", every: int = 3,
                 batch: int = 8, device: str | None = None, progress: Callable[[str], None] | None = print,
                 robustness: bool = True, robust_eps: int = 3, step: int | None = None) -> dict[str, Any]:
    """Runs once per model at a time: a marker with this process ID is kept while it runs. step: test that saved
    checkpoint (checkpoints/<step>/pretrained_model, e.g. while the run still trains) instead of the model's current
    one; the result goes to its own file and does not change the model."""
    import os
    if is_running(model_version, split, step):
        raise RuntimeError(f"an offline eval of {model_version}" + (f" step {step}" if step is not None else "")
                           + " is already running")
    marker = running_marker(model_version, split, step)
    marker.parent.mkdir(parents=True, exist_ok=True)
    import socket
    marker.write_text(f"{os.getpid()} {socket.gethostname()} {now_iso()}")
    try:
        return _offline_eval(model_version, registry, split, every, batch, device, progress, robustness, robust_eps,
                             step)
    finally:
        marker.unlink(missing_ok=True)


def _offline_eval(model_version: str, registry: Registry | None, split: str, every: int, batch: int,
                  device: str | None, progress: Callable[[str], None] | None, robustness: bool,
                  robust_eps: int, step: int | None = None) -> dict[str, Any]:
    from hackpipe.common.lerobot_io import DatasetFolder
    from hackpipe.dataset.versioning import load_manifest
    reg = registry or Registry()
    model = reg.get_model(model_version)
    if model is None:
        raise KeyError(f"Unknown model {model_version}")
    ckpt = resolve(model["checkpoint_dir"] or "")
    if step is not None:
        ckpt = resolve(model["meta"].get("output_dir") or ".") / "checkpoints" / f"{int(step):06d}" / "pretrained_model"
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
    from hackpipe.common.config import gates_config
    gated = set((gates_config().get("offline") or {}).get("block", {}).get("robustness_variants") or ROBUSTNESS)
    variants, skipped = {k: v for k, v in ROBUSTNESS.items() if k in gated}, {}   # only what the checks use
    from hackpipe.training.curriculum import reword
    texts = {str(x) for t in tables for x in t["full_text"][t["rows"]]}
    if "reworded_instruction" in variants and not any(reword(x, 1) != x for x in texts):   # = the baseline
        variants.pop("reworded_instruction")
        skipped["reworded_instruction"] = "no paraphrase pattern matches the test instructions"
    res = score(tables, predict, int(chunk or ck_chunk or 50), every, batch, progress, hz=rate,
                robust=variants if robustness else None, robust_eps=robust_eps)
    if robustness:
        res["robustness"]["skipped"] = skipped
    res["latency_ms"] = latency(predict, tables[0], rate)
    streams = res.pop("_streams", [])
    res.update({"model": model_version, "checkpoint": str(ckpt), "step": step if step is not None else step_of(ckpt),
                "dataset_version": model["dataset_version"],
                "split": split, "rate_hz": rate, "every": every, "episodes": {t["source"]: len(set(t["ep"][t["rows"]]))
                                                                            for t in tables},
                "created_at": now_iso()})
    out = result_path(model_version, split, step)
    out.parent.mkdir(parents=True, exist_ok=True)
    res["file"] = str(out)
    save_trajectories(out.with_suffix(".npz"), streams)
    out.write_text(json.dumps(res, indent=1))
    if step is None and step_of(ckpt) is not None:     # also findable by step (Evaluate → checkpoints table)
        sp = result_path(model_version, split, step_of(ckpt))
        save_trajectories(sp.with_suffix(".npz"), streams)
        sp.write_text(json.dumps({**res, "file": str(sp)}, indent=1))
    return res


def save_trajectories(path: Path, streams: list[dict[str, Any]]) -> None:
    arrays: dict[str, Any] = {"meta": np.array(json.dumps([{"name": x["name"], "ep": int(x["ep"])} for x in streams]))}
    for i, x in enumerate(streams):
        arrays.update({f"{i}_pred": x["pred"].astype(np.float32), f"{i}_gt": x["gt"].astype(np.float32),
                       f"{i}_still": x["still"].astype(np.float32), f"{i}_replan": x["replan"],
                       f"{i}_sub": np.asarray(x["sub"], dtype="U")})
    np.savez_compressed(path, **arrays)


def running_marker(model_version: str, split: str = "test", step: int | None = None) -> Path:
    return result_path(model_version, split, step).with_suffix(".running")


def is_running(model_version: str, split: str = "test", step: int | None = None) -> bool:
    """An offline eval of this model is in progress: a marker written on this machine whose process still runs (a
    marker that came with a state import from another machine, or whose process ended, doesn't count)."""
    import os
    import socket
    f = running_marker(model_version, split, step)
    try:
        parts = f.read_text().split()
        if len(parts) > 2 and parts[1] != socket.gethostname():
            return False
        os.kill(int(parts[0]), 0)
        return True
    except (OSError, ValueError, IndexError):
        return False


def is_current(res: dict[str, Any] | None, model: dict[str, Any] | None = None) -> bool:
    """The eval has the metrics the gates need (evals made before the executed-stream metrics don't) and, given the
    model, tested the checkpoint the model points at now (a new best checkpoint needs a new test)."""
    if not (res and (res.get("executed") or {}).get("arm_mean") is not None):
        return False
    if model is None or not res.get("checkpoint"):
        return True
    return resolve(res["checkpoint"]).resolve() == resolve(model["checkpoint_dir"] or ".").resolve()
