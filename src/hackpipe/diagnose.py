"""Diagnostic tools (`hackpipe diagnose <tool>`), from our team repo (github.com/yugalrk/Robotics--VLA-training:
tools/, tools_show_io.py, tools_ablation.py). Each answers one question about a model or a sim run:

  probe-frames   export recorded frames (state, 3 cameras, the next 50 recorded actions) to an .npz — the shared
                 input of the probes below (default: held-out-style frames of handover episodes 0 and 5)
  io             one SmolVLA inference end to end: the inputs it gets and the action chunk it returns, next to
                 what the demonstrator did (inputs.png + io_example.json)
  ablation       how random are the outputs (5 noise seeds), and how much do the cameras / the state matter
                 (black cameras, images or state from another moment, vs. just holding the pose)
  server         send probe frames to a running policy server (`hackpipe serve …`) and compare what it serves
                 with the recorded actions — checks the server's input mapping, normalisation and prompt
  pi05           score input-setup variants of an openpi pi0.5 checkpoint (normalisation, state in the prompt,
                 camera order, prompt, horizon) against the recorded actions; plus the checkpoint's training step
  grasp          when does the left gripper close in sim runs, and how close is the arm to the demo's grasp pose
  episode        one Isaac Sim episode log vs the demo: arm motion, grippers, fingertips-to-box distance

The sim-run tools read the Isaac Sim scripts' step logs (one JSON row per logged step: "step" (30 Hz index),
"state", "action", optional "box", "left_tips") and the reference file (`hackpipe sim reference`).
The camera domain gap (training vs sim cameras) is sim/camera_gap.py.
"""
from __future__ import annotations

import io as _io
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hackpipe.deploy.server import CAMERAS, TASK

ARM = [*range(7), *range(8, 15)]
NAMES = [f"L_j{i}" for i in range(1, 8)] + ["L_grip"] + [f"R_j{i}" for i in range(1, 8)] + ["R_grip"]
SHORT = {CAMERAS[0]: "ext", CAMERAS[1]: "lw", CAMERAS[2]: "rw"}


# ---------------------------------------------------------------- probe frames
def export_probe_frames(source: str | Path, out: str | Path, episodes: tuple[int, ...] = (0, 5),
                        frames: tuple[int, ...] = (0, 200, 330, 600, 900, 1200), horizon: int = 50) -> dict[str, Any]:
    """state (n, 16), actions (n, horizon, 16: the recorded 30 fps actions from that frame, edge-padded),
    ext / lw / rw (n, H, W, 3) uint8, where (n, 2) = (episode, frame)."""
    from hackpipe.common.lerobot_io import DatasetFolder
    from hackpipe.dataset.subtasks import Videos
    ds = DatasetFolder(Path(source))
    df, videos = ds.frames(), Videos(ds)
    res: dict[str, list[Any]] = {k: [] for k in ("state", "actions", "ext", "lw", "rw", "where")}
    for ep in episodes:
        d = df[df["episode_index"] == ep].sort_values("frame_index")
        S, A = np.stack(d["observation.state"]).astype(np.float32), np.stack(d["action"]).astype(np.float32)
        ok = [t for t in frames if t < len(S)]
        imgs = {c: videos.frames(ep, c.split(".")[-1], ok) for c in CAMERAS}
        for i, t in enumerate(ok):
            res["state"].append(S[t])
            res["actions"].append(np.pad(A[t:t + horizon], ((0, max(0, t + horizon - len(A))), (0, 0)), mode="edge"))
            for c in CAMERAS:
                res[SHORT[c]].append(imgs[c][i])
            res["where"].append((ep, t))
    arrays = {k: np.array(v) for k, v in res.items()}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    return {"out": str(out), **{k: list(v.shape) for k, v in arrays.items()}}


def load_probe(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def probe_obs(F: dict[str, np.ndarray], i: int, images: list[np.ndarray] | None = None,
              state: np.ndarray | None = None) -> dict[str, np.ndarray]:
    imgs = images if images is not None else [F["ext"][i], F["lw"][i], F["rw"][i]]
    return {"observation.state": (F["state"][i] if state is None else state).astype(np.float32),
            **dict(zip(CAMERAS, imgs))}


# ---------------------------------------------------------------- a LeRobot checkpoint's chunk
def chunk_fn(checkpoint: str, device: str = "auto") -> Callable[..., np.ndarray]:
    """(obs, task, seed) → the full predicted action chunk (chunk, 16), through the serving code's input path."""
    import torch

    from hackpipe.deploy.server import LeRobotPolicy
    p = LeRobotPolicy(checkpoint, device)

    def predict(obs: dict[str, np.ndarray], task: str = TASK, seed: int = 0) -> np.ndarray:
        torch.manual_seed(seed)
        p.reset()
        frame = p._prepare({k: np.asarray(obs[p.source_of[k]]) for k in p.input_features}, p.device, task=task)
        with torch.inference_mode():
            return p.post(p.policy.predict_action_chunk(p.pre(frame))).squeeze(0).float().cpu().numpy()
    return predict


def demo_at_rate(F: dict[str, np.ndarray], i: int, stride: int, n: int) -> np.ndarray:
    """The recorded actions at the policy's spacing (stride frames of 30 fps per action), first n."""
    return F["actions"][i][::stride][:n]


def io_example(predict: Callable[..., np.ndarray], F: dict[str, np.ndarray], i: int, out_dir: str | Path,
               stride: int = 5) -> dict[str, Any]:
    from PIL import Image, ImageDraw
    obs = probe_obs(F, i)
    chunk = predict(obs, TASK, 0)
    demo = demo_at_rate(F, i, stride, len(chunk))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    strip = Image.new("RGB", (3 * 224, 244), "white")
    d = ImageDraw.Draw(strip)
    for k, (c, label) in enumerate(zip(CAMERAS, ["head (external)", "left wrist", "right wrist"])):
        strip.paste(Image.fromarray(obs[c]).resize((224, 224)), (k * 224, 20))
        d.text((k * 224 + 4, 4), label, fill="black")
    strip.save(out / "inputs.png")
    ep, t = map(int, F["where"][i])
    n = min(len(chunk), len(demo))
    res = {"episode": ep, "frame": t, "time_s": t / 30, "instruction": TASK,
           "input_state": dict(zip(NAMES, obs["observation.state"].round(4).tolist())), "images": "inputs.png",
           "output_chunk": chunk.round(4).tolist(), "demo_actions": demo.round(4).tolist(), "columns": NAMES,
           "arm_error_per_step_rad": np.abs(chunk[:n][:, ARM] - demo[:n][:, ARM]).mean(1).round(4).tolist(),
           "right_arm_max_change_rad": round(float(np.abs(chunk[:, 8:15] - obs["observation.state"][8:15]).max()), 4)}
    (out / "io_example.json").write_text(json.dumps(res, indent=1))
    return res


def ablation(predict: Callable[..., np.ndarray], F: dict[str, np.ndarray], n_actions: int = 10, stride: int = 5,
             seeds: int = 5) -> dict[str, Any]:
    """Error of the first n_actions (arm joints, rad) vs the demonstrator under input ablations, and the spread
    between noise seeds for the same input."""
    n = len(F["state"])
    other = (np.arange(n) + 2) % n                                   # another phase of the task
    demo = [demo_at_rate(F, i, stride, n_actions) for i in range(n)]
    black = np.zeros_like(F["ext"][0])
    err = lambda i, a: float(np.abs(a[:n_actions][:, ARM] - demo[i][:, ARM]).mean())  # noqa: E731
    normal = [[predict(probe_obs(F, i), TASK, s)[:n_actions] for s in range(seeds)] for i in range(n)]
    e_normal = float(np.mean([err(i, normal[i][s]) for i in range(n) for s in range(seeds)]))

    def cond(make: Callable[[int], dict[str, np.ndarray]]) -> float:
        return float(np.mean([err(i, predict(make(i), TASK, 0)) for i in range(n)]))
    imgs = lambda i: [F["ext"][i], F["lw"][i], F["rw"][i]]  # noqa: E731
    rows = {
        "normal (all inputs correct)": e_normal,
        "all 3 cameras black": cond(lambda i: probe_obs(F, i, [black] * 3)),
        "images from a different moment": cond(lambda i: probe_obs(F, i, imgs(other[i]))),
        "head camera black, wrists correct": cond(lambda i: probe_obs(F, i, [black, F["lw"][i], F["rw"][i]])),
        "wrist cameras black, head correct": cond(lambda i: probe_obs(F, i, [F["ext"][i], black, black])),
        "correct images, state of a different moment": cond(lambda i: probe_obs(F, i, state=F["state"][other[i]])),
        "baseline: just hold the current pose": float(np.mean([np.abs(F["state"][i][ARM] - demo[i][:, ARM]).mean()
                                                                for i in range(n)])),
    }
    return {"seed_spread_first_action_rad": round(float(np.mean([np.stack(normal[i])[:, 0, ARM].std(0).mean()
                                                                for i in range(n)])), 4),
            "seed_spread_rad": round(float(np.mean([np.stack(normal[i])[:, :, ARM].std(0).mean()
                                                    for i in range(n)])), 4),
            "error_rad": {k: round(v, 4) for k, v in rows.items()},
            "x_normal": {k: round(v / e_normal, 2) if e_normal else None for k, v in rows.items()}}


# ---------------------------------------------------------------- a running server
def server_check(url: str, F: dict[str, np.ndarray], calls: int | None = None) -> dict[str, Any]:
    """For every probe frame: /reset, then `calls` /act with that frame's observation (the server serves its chunk
    one action per call), compared with the recorded actions at the server's rate (/info policy_hz)."""
    from hackpipe.deploy.serving import http_json
    info = http_json(url + "/info")
    stride = max(1, int(round(30 / float(info.get("policy_hz") or 30))))
    calls = calls or int(info.get("n_action_steps") or 1)
    rows = []
    for i in range(len(F["state"])):
        obs = probe_obs(F, i)
        buf = _io.BytesIO()
        np.savez(buf, **obs, task=np.array(TASK))
        http_json(url + "/reset", b"")
        got = [http_json(url + "/act", buf.getvalue(), timeout=300) for _ in range(calls)]   # first call: inference
        served = np.array([g["action"] for g in got])
        ref = demo_at_rate(F, i, stride, calls)
        m = min(len(served), len(ref))
        rows.append({"where": [int(x) for x in F["where"][i]],
                     "arm_error_rad": round(float(np.abs(served[:m][:, ARM] - ref[:m][:, ARM]).mean()), 4),
                     "hold_pose_rad": round(float(np.abs(obs["observation.state"][ARM] - ref[:m][:, ARM]).mean()), 4),
                     "gripper_error": round(float(np.abs(served[:m][:, [7, 15]] - ref[:m][:, [7, 15]]).mean()), 3),
                     "first_call_ms": round(float(got[0]["latency_ms"]), 1)})
    return {"server": {k: info.get(k) for k in ("policy_type", "checkpoint", "policy_hz", "n_action_steps")},
            "calls_per_frame": calls, "frames": rows,
            "mean_arm_error_rad": round(float(np.mean([r["arm_error_rad"] for r in rows])), 4),
            "mean_hold_pose_rad": round(float(np.mean([r["hold_pose_rad"] for r in rows])), 4)}


# ---------------------------------------------------------------- pi0.5 (openpi)
def score_variant(infer: Callable[[dict[str, Any], str], np.ndarray], F: dict[str, np.ndarray], n: int,
                  prompt: str = TASK) -> dict[str, float]:
    """Arm error of the first n predicted actions (30 fps) read as absolute targets and as deltas from the state,
    and the hold-pose baseline. infer(obs, prompt) → (horizon, 16)."""
    ab, de, ho = [], [], []
    for i in range(len(F["state"])):
        a = np.asarray(infer(probe_obs(F, i), prompt))[:n]
        ref = F["actions"][i][:n]
        ab.append(np.abs(a[:, ARM] - ref[:, ARM]).mean())
        de.append(np.abs((F["state"][i] + a)[:, ARM] - ref[:, ARM]).mean())
        ho.append(np.abs(F["state"][i][ARM] - ref[:, ARM]).mean())
    return {"absolute": round(float(np.mean(ab)), 4), "as_delta": round(float(np.mean(de)), 4),
            "hold_pose": round(float(np.mean(ho)), 4)}


def pi05_variants(checkpoint: str, F: dict[str, np.ndarray], horizons: tuple[int, ...] = (50,)) -> dict[str, Any]:
    """Which input setup was the checkpoint trained with? The variant with the lowest error is the one. Each variant
    is the server's own pipeline (deploy/server.py openpi_pipeline) with one setting changed."""
    import jax

    from hackpipe.deploy.server import openpi_load, openpi_pipeline
    out: dict[str, Any] = {}
    for H in horizons:
        cfg, model, norm = openpi_load(checkpoint, H)
        noise = np.asarray(jax.random.normal(jax.random.key(0), (H, cfg.action_dim)), np.float32)

        def make(**kw: Any) -> Callable[[dict[str, Any], str], np.ndarray]:
            pol = openpi_pipeline(cfg, model, norm, **kw)
            return lambda obs, prompt: pol.infer({**obs, "prompt": prompt}, noise=noise)["actions"]
        n = min(H, 25)
        variants = {"as served (quantile, state in prompt, ext/left/right)": (make(), TASK),
                    "z-score normalisation": (make(quantile=False), TASK),
                    "no state in prompt": (make(discrete_state=False), TASK),
                    "wrists swapped": (make(order=(CAMERAS[0], CAMERAS[2], CAMERAS[1])), TASK),
                    "lowercase prompt": (make(), TASK.lower()), "empty prompt": (make(), "")}
        out[f"horizon {H} (first {n})"] = {k: score_variant(f, F, n, p) for k, (f, p) in variants.items()}
    return out


def pi05_step(checkpoint: str) -> int | None:
    """The training step stored in an openpi checkpoint's train_state (None if it can't be read)."""
    import tensorstore as ts
    spec = {"driver": "zarr", "kvstore": {"driver": "ocdbt", "base": f"file://{Path(checkpoint).resolve() / 'train_state'}",
                                          "path": "step"}}
    try:
        return int(np.asarray(ts.open(spec).result().read().result()))
    except Exception:  # noqa: BLE001 — older layouts: report unknown rather than fail the probe
        return None


# ---------------------------------------------------------------- sim runs
def read_steps(path: str | Path) -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    out = {"step": np.array([r.get("step", i) for i, r in enumerate(rows)]),
           "state": np.array([r["state"] for r in rows], float), "action": np.array([r["action"] for r in rows], float)}
    if rows and "box" in rows[0]:
        out["box"] = np.array([r["box"] for r in rows], float)
    if rows and "left_tips" in rows[0]:
        out["left_tips"] = np.array([r["left_tips"] for r in rows], float)
    return out


def first_close(cmd: np.ndarray, below: float = 0.2) -> int | None:
    i = np.flatnonzero(np.asarray(cmd) < below)
    return int(i[0]) if len(i) else None


def grasp_timing(ref: dict[str, Any], run: dict[str, np.ndarray], fps: float = 30.0) -> dict[str, Any]:
    """When the left gripper closes in a run vs the demo, and how far (sum over the 7 joints, rad) the arm is from
    the demo's grasp pose then, from the demo's pose at the same time, and at its closest before closing."""
    D, DA = np.array(ref["demo_states"]), np.array(ref["demo_actions"])
    arm = list(range(7))
    dg = first_close(DA[:, 7])
    S, A = run["state"], run["action"]
    step = run.get("step", np.arange(len(S)))           # rows may be logged every few 30 Hz steps
    g = first_close(A[:, 7])
    res: dict[str, Any] = {"demo_close_s": None if dg is None else round(dg / fps, 2)}
    if g is None or dg is None:
        return {**res, "closed": False}
    tg = int(step[g])
    later = min(int(np.searchsorted(step, tg + int(2 * fps))), len(S) - 1)
    best = int(np.argmin(np.abs(S[:g + 1][:, arm] - D[dg, arm]).sum(1)))
    return {**res, "closed": True, "close_s": round(tg / fps, 2), "vs_demo_s": round((tg - dg) / fps, 2),
            "finger_state_2s_later": round(float(S[later, 7]), 3),
            "to_demo_grasp_pose_rad": round(float(np.abs(S[g, arm] - D[dg, arm]).sum()), 3),
            "to_demo_same_time_rad": round(float(np.abs(S[g, arm] - D[min(tg, len(D) - 1), arm]).sum()), 3),
            "closest_before_rad": round(float(np.abs(S[best, arm] - D[dg, arm]).sum()), 3),
            "closest_before_s": round(int(step[best]) / fps, 2)}


def episode_report(ref: dict[str, Any], run: dict[str, np.ndarray], every_s: float = 5.0, fps: float = 30.0
                   ) -> dict[str, Any]:
    """One sim episode vs the demo: a row every `every_s` (how far each arm has moved from the start pose, the demo's
    at the same time, left gripper command / state, fingertips-to-box and box height) and the summary."""
    D, ready = np.array(ref["demo_states"]), np.array(ref["initial_state"])
    step, S, A = run["step"], run["state"], run["action"]
    box, tips = run.get("box"), run.get("left_tips")
    rows = []
    for t in range(0, int(step[-1]) + 1, int(every_s * fps)):
        i = int(np.searchsorted(step, t))
        if i >= len(S):
            break
        d = D[min(t, len(D) - 1)]
        rows.append({"t_s": round(t / fps, 1),
                     "left_moved_rad": round(float(np.abs(S[i, :7] - ready[:7]).sum()), 2),
                     "demo_left_moved_rad": round(float(np.abs(d[:7] - ready[:7]).sum()), 2),
                     "left_grip_cmd": round(float(A[i, 7]), 2), "left_grip_state": round(float(S[i, 7]), 2),
                     "right_moved_rad": round(float(np.abs(S[i, 8:15] - ready[8:15]).sum()), 2),
                     "demo_right_moved_rad": round(float(np.abs(d[8:15] - ready[8:15]).sum()), 2),
                     "tips_to_box_mm": (round(float(np.linalg.norm(tips[i].mean(0) - box[i]) * 1000))
                                        if tips is not None and box is not None else None),
                     "box_height_mm": round(float(box[i, 1] * 1000), 1) if box is not None else None})
    n = min(len(S), len(D))
    idx = np.minimum(step[:n], len(D) - 1)
    err = np.abs(S[:n][:, ARM] - D[idx][:, ARM])
    res: dict[str, Any] = {"rows": rows, "steps": len(S), "mean_joint_distance_to_demo_rad": round(float(err.mean()), 3),
                           "max_joint_distance_to_demo_rad": round(float(err.max()), 3)}
    if tips is not None and box is not None:
        dist = np.linalg.norm(tips.mean(1) - box, axis=1) * 1000
        k = int(dist.argmin())
        res["closest_tips_to_box_mm"] = round(float(dist[k]))
        res["closest_at_s"] = round(float(step[k] / fps), 1)
        res["box_minus_tips_mm"] = np.round((box[k] - tips[k].mean(0)) * 1000).tolist()
    return res
