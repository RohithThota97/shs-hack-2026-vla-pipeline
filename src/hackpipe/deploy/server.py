"""Policy server for the FR3 Duo Isaac Sim bridge (`hackpipe serve`), after the team's fr3duo_policy_server.py.

Same protocol as the team's Isaac Sim client (Environment/fr3duo_isaac_client.py, fr3duo_isaac_eval.py):

    GET  /info   -> {"checkpoint", "model", "policy_type", "input_features", "output_features", "fake",
                     "n_action_steps", "chunk_size", "policy_hz", "grasp_extend_s", "grasp_hold_s"}
    POST /reset  -> clears the action queue; call at the start of every episode
    POST /act    body: .npz, one array per dataset feature ("observation.state" (16,), "observation.images.
                       external_camera" / left_wrist_camera / right_wrist_camera as uint8 (H, W, 3)) + "task"
                 -> {"action": [16 floats], "latency_ms", "fake"}
    POST /probe  (ours) one recorded frame of the probe source through /act's path → {"ok", "action_shape",
                 "finite", "latency_ms"}; the deploy health check

Clients send the dataset's camera names; checkpoints trained with lerobot-train expect SmolVLA's camera1..3, so
the observation is renamed with configs/dataset.yaml → preprocessing.rename_map. The server refuses a
checkpoint that does not match the FR3 Duo contract (state 16, action 16, three 224×224 cameras).
"""
from __future__ import annotations

import io
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

STATE_DIM = ACTION_DIM = 16
CAMERAS = ["observation.images.external_camera", "observation.images.left_wrist_camera",
           "observation.images.right_wrist_camera"]
TASK = "Hand the box from the left gripper to the right gripper and place it on the serving tray."


def rename_map() -> dict[str, str]:
    from hackpipe.common.config import dataset_config
    return (dataset_config().get("preprocessing") or {}).get("rename_map") or {}


class DummyPolicy:
    """Holds the pose of the first observation after /reset — bridge tests only, flagged FAKE."""
    fake, checkpoint, model, policy_type = True, None, None, "dummy-hold-start-pose"
    input_features = {"observation.state": [STATE_DIM], **{c: [3, 224, 224] for c in CAMERAS}}
    output_features = {"action": [ACTION_DIM]}
    n_action_steps = chunk_size = None

    def __init__(self) -> None:
        self.hold: np.ndarray | None = None

    def reset(self) -> None:
        self.hold = None

    def act(self, obs: dict[str, np.ndarray], task: str) -> np.ndarray:
        if self.hold is None:
            self.hold = np.asarray(obs["observation.state"], np.float32).copy()
        return self.hold


class LeRobotPolicy:
    fake = False

    def __init__(self, checkpoint: str, device: str = "auto", n_action_steps: int | None = None,
                 model: str | None = None) -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.policies.utils import prepare_observation_for_inference
        self._torch, self._prepare = torch, prepare_observation_for_inference
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.pretrained_path = checkpoint
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        cfg.device = str(self.device)
        if n_action_steps:                 # actions executed from each predicted chunk before predicting again
            if not 1 <= n_action_steps <= cfg.chunk_size:
                raise ValueError(f"n_action_steps must be 1…{cfg.chunk_size} (the checkpoint's chunk)")
            cfg.n_action_steps = n_action_steps
        self.n_action_steps, self.chunk_size = cfg.n_action_steps, getattr(cfg, "chunk_size", None)
        self.policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg).to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, checkpoint, preprocessor_overrides={"device_processor": {"device": str(self.device)}})
        self.checkpoint, self.model, self.policy_type = checkpoint, model, cfg.type
        self.input_features = {k: list(v.shape) for k, v in cfg.input_features.items()}
        self.output_features = {k: list(v.shape) for k, v in cfg.output_features.items()}
        names = rename_map()                      # checkpoint key ← dataset key, for camera1..3 checkpoints
        self.source_of = {k: next((d for d, c in names.items() if c == k), k) for k in self.input_features}

    def reset(self) -> None:
        self.policy.reset()
        for p in (self.pre, self.post):
            if hasattr(p, "reset"):
                p.reset()

    def act(self, obs: dict[str, np.ndarray], task: str) -> np.ndarray:
        frame = self._prepare({k: np.asarray(obs[self.source_of[k]]) for k in self.input_features},
                              self.device, task=task)
        with self._torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(frame)))
        return action.squeeze(0).float().cpu().numpy()


class GraspDeepener:
    """Inference-time grasp shaping, no retraining (the team's --grasp_extend). When the model first closes a
    gripper: keep it at the pre-grasp opening and continue the arm's approach (mean joint velocity of the last second,
    decaying, ≤ max_extra_rad per joint) for extend_s, then close and hold the arm for hold_s, then re-plan."""
    ARMS = {"left": (list(range(7)), 7), "right": (list(range(8, 15)), 15)}

    def __init__(self, policy: Any, extend_s: float, hold_s: float = 0.7, hz: float = 6.0, close_below: float = 0.2,
                 decay: float = 0.95, max_extra_rad: float = 0.15) -> None:
        self.p, self.extend, self.hold = policy, int(round(extend_s * hz)), int(round(hold_s * hz))
        self.window = max(2, int(round(hz)))
        self.close_below, self.decay, self.max_extra = close_below, decay, max_extra_rad
        for k in ("fake", "checkpoint", "model", "policy_type", "input_features", "output_features",
                  "n_action_steps", "chunk_size"):
            setattr(self, k, getattr(policy, k, None))
        self.grasp_extend_s, self.grasp_hold_s = extend_s, hold_s
        self.reset()

    def reset(self) -> None:
        self.p.reset()
        self.hist: list[np.ndarray] = []
        self.plan: list[np.ndarray] = []
        self.done: set[str] = set()

    def act(self, obs: dict[str, np.ndarray], task: str) -> np.ndarray:
        if self.plan:                             # shaping in progress: replay it, then let the model re-plan
            a = self.plan.pop(0)
            if not self.plan:
                self.p.reset()
        else:
            a = np.asarray(self.p.act(obs, task), np.float32)
            for side, (arm, g) in self.ARMS.items():
                prev = self.hist[-1] if self.hist else None
                if side in self.done or prev is None or len(self.hist) < 3:
                    continue
                if prev[g] >= self.close_below and a[g] < self.close_below:     # the model starts closing
                    self.done.add(side)
                    self.plan = self._shape(arm, g, float(a[g]))
                    a = self.plan.pop(0)
                    break
        self.hist = (self.hist + [np.asarray(a, np.float32)])[-(self.window + 2):]
        return a

    def _shape(self, arm: list[int], g: int, close_cmd: float) -> list[np.ndarray]:
        base = self.hist[-1].copy()
        w = min(self.window, len(self.hist) - 1)
        v = (self.hist[-1][arm] - self.hist[-1 - w][arm]) / w
        plan, extra, cur = [], np.zeros(len(arm), np.float32), base.copy()
        for i in range(self.extend):
            extra = np.clip(extra + v * self.decay ** (i + 1), -self.max_extra, self.max_extra)
            cur = base.copy()
            cur[arm] = base[arm] + extra
            plan.append(cur.copy())
        closing = cur.copy()
        closing[g] = close_cmd
        return plan + [closing.copy() for _ in range(max(1, self.hold))]


def contract_problems(policy: Any) -> list[str]:
    """Differences between the checkpoint's features and the FR3 Duo contract (dataset or SmolVLA camera names)."""
    names = rename_map()
    feats = {next((d for d, c in names.items() if c == k), k): v for k, v in policy.input_features.items()}
    expected = {"observation.state": [STATE_DIM], **{c: [3, 224, 224] for c in CAMERAS}}
    problems = [f"missing input {k} {s}" for k, s in expected.items() if k not in feats]
    problems += [f"input {k} has shape {feats[k]}, expected {s}" for k, s in expected.items()
                 if k in feats and feats[k] != s]
    problems += [f"unexpected input {k} {feats[k]}" for k in sorted(set(feats) - set(expected))]
    if policy.output_features.get("action") != [ACTION_DIM]:
        problems.append(f"action has shape {policy.output_features.get('action')}, expected [{ACTION_DIM}]")
    return problems


def probe_frame(source: str | Path) -> dict[str, Any]:
    """First frame of the first episode of a recorded dataset, as a client would send it (uint8 HWC cameras)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from hackpipe.common.config import resolve
    root = resolve(source)
    item = LeRobotDataset(f"local/{root.name}", root=root, episodes=[0], video_backend="pyav")[0]
    obs = {c: (item[c].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for c in CAMERAS}
    obs["observation.state"] = item["observation.state"].numpy().astype(np.float32)
    return {"obs": obs, "task": str(item.get("task") or TASK), "source": str(source)}


def make_handler(policy: Any, info_extra: dict[str, Any], probe_source: str | None):
    import threading
    lock = threading.Lock()                       # one policy, one queue: requests are served one at a time

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _act(self, obs: dict[str, np.ndarray], task: str) -> dict[str, Any]:
            t0 = time.perf_counter()
            a = np.asarray(policy.act(obs, task), np.float64)
            return {"action": a.tolist(), "latency_ms": (time.perf_counter() - t0) * 1000, "fake": policy.fake}

        def do_GET(self) -> None:
            if self.path != "/info":
                return self._reply(404, {"error": f"unknown path {self.path}"})
            self._reply(200, {"checkpoint": policy.checkpoint, "model": getattr(policy, "model", None),
                              "policy_type": policy.policy_type, "input_features": policy.input_features,
                              "output_features": policy.output_features, "fake": policy.fake,
                              "n_action_steps": getattr(policy, "n_action_steps", None),
                              "chunk_size": getattr(policy, "chunk_size", None),
                              "grasp_extend_s": getattr(policy, "grasp_extend_s", 0),
                              "grasp_hold_s": getattr(policy, "grasp_hold_s", 0), **info_extra})

        def do_POST(self) -> None:
            try:
                with lock:
                    if self.path == "/reset":
                        policy.reset()
                        return self._reply(200, {"ok": True})
                    if self.path == "/probe":
                        if not probe_source:
                            return self._reply(400, {"error": "no probe source (serve --probe-source)"})
                        f = probe_frame(probe_source)
                        policy.reset()
                        r = self._act(f["obs"], f["task"])
                        policy.reset()
                        a = np.asarray(r["action"])
                        return self._reply(200, {"ok": a.shape == (ACTION_DIM,) and bool(np.isfinite(a).all()),
                                                 "action_shape": list(a.shape), "finite": bool(np.isfinite(a).all()),
                                                 "latency_ms": round(r["latency_ms"], 1), "fake": policy.fake,
                                                 "frame": f"{f['source']} episode 0 frame 0"})
                    if self.path != "/act":
                        return self._reply(404, {"error": f"unknown path {self.path}"})
                    body = self.rfile.read(int(self.headers["Content-Length"]))
                    with np.load(io.BytesIO(body), allow_pickle=False) as npz:
                        arrays = {k: npz[k] for k in npz.files}
                    task = str(arrays.pop("task", TASK))
                    need = ["observation.state", *CAMERAS]
                    missing = [k for k in need if k not in arrays]
                    if missing:
                        return self._reply(400, {"error": f"observation is missing {missing}"})
                    self._reply(200, self._act(arrays, task))
            except Exception as e:  # noqa: BLE001 — report to the client instead of dropping the connection
                self._reply(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *a: Any) -> None:
            pass

    return Handler


def build_policy(checkpoint: str | None = None, model: str | None = None, dummy: bool = False,
                 n_action_steps: int | None = None, grasp_extend: float = 0.0, grasp_hold: float = 0.7,
                 policy_hz: float = 6.0, device: str = "auto", any_robot: bool = False) -> Any:
    if dummy:
        return DummyPolicy()
    policy: Any = LeRobotPolicy(str(checkpoint), device, n_action_steps, model)
    problems = contract_problems(policy)
    if problems and not any_robot:
        raise ValueError("checkpoint does not match the FR3 Duo contract: " + "; ".join(problems))
    if grasp_extend > 0:
        policy = GraspDeepener(policy, grasp_extend, grasp_hold, policy_hz)
    return policy


def serve(policy: Any, host: str = "127.0.0.1", port: int = 8090, policy_hz: float = 6.0,
          probe_source: str | None = None) -> None:
    srv = ThreadingHTTPServer((host, port), make_handler(policy, {"policy_hz": policy_hz}, probe_source))
    print(f"policy {policy.policy_type} from {policy.checkpoint} on http://{host}:{port}"
          + (" — FAKE results" if policy.fake else ""), flush=True)
    srv.serve_forever()
