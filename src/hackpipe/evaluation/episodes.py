"""Why a test episode goes well or not: the model's commanded gripper path vs the demo's, in 3D.

From the executed stream of an offline test (offline.trajectories): both arms' grasp points by forward kinematics
(kinematics.py), the distance between model and demo per step (cm), the first step where it stays above
`drift_cm` for a while, and the demo's gripper close / open moments the model misses. Teacher-forced: every
prediction starts from the recorded state, so this shows what the model would command at each moment, not a
closed-loop rollout (that is the Isaac Sim test).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from hackpipe.evaluation.kinematics import arm_paths
from hackpipe.evaluation.offline import GRIP, gripper_events

DRIFT_CM = 5.0          # "off the demo": the grasp point more than this far from the demo's …
DRIFT_STEPS = 3         # … for this many policy steps in a row (0.5 s at 6 Hz)


def paths(ep: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """{"model": {"left", "right"}, "demo": {…}} grasp-point paths (n, 3) in metres."""
    return {"model": arm_paths(ep["pred"]), "demo": arm_paths(ep["gt"])}


def summarize(ep: dict[str, Any], hz: float = 6.0, drift_cm: float = DRIFT_CM,
              drift_steps: int = DRIFT_STEPS) -> dict[str, Any]:
    p = paths(ep)
    err = {s: np.linalg.norm(p["model"][s] - p["demo"][s], axis=1) * 100 for s in ("left", "right")}   # cm
    both = np.maximum(err["left"], err["right"])
    above = both > drift_cm
    run = np.convolve(above.astype(int), np.ones(drift_steps, int), "valid") if len(above) >= drift_steps else []
    first = int(np.argmax(np.asarray(run) == drift_steps)) if len(run) and (np.asarray(run) == drift_steps).any() else None
    missed = []
    tol = int(round(hz))
    for g, side in zip(GRIP, ("left", "right")):
        pred = gripper_events(ep["pred"], g)
        for t, kind in gripper_events(ep["gt"], g):
            if not any(k == kind and abs(pp - t) <= tol for pp, k in pred):
                missed.append({"gripper": side, "kind": kind, "t_s": round(t / hz, 1), "subtask": str(ep["sub"][t])})
    worst = int(np.argmax(both)) if len(both) else 0
    arm = "left" if err["left"][worst] >= err["right"][worst] else "right"
    out = {"source": ep["name"], "ep": int(ep["ep"]), "steps": int(len(both)),
           "mean_cm": round(float(both.mean()), 1) if len(both) else None,
           "max_cm": round(float(both.max()), 1) if len(both) else None,
           "max_at_s": round(worst / hz, 1), "max_arm": arm, "max_subtask": str(ep["sub"][worst]) if len(both) else "",
           "drift_at_s": round(first / hz, 1) if first is not None else None, "drift_step": first,
           "drift_arm": (("left" if err["left"][first] >= err["right"][first] else "right") if first is not None
                         else None),
           "drift_subtask": str(ep["sub"][first]) if first is not None else None,
           "missed_gripper": missed}
    out["follows_demo"] = first is None and not missed
    out["why"] = explain(out, drift_cm)
    return out


def explain(s: dict[str, Any], drift_cm: float = DRIFT_CM) -> str:
    if s["follows_demo"]:
        return (f"Follows the demo: the grasp points stay within {drift_cm:g} cm (mean {s['mean_cm']} cm, max "
                f"{s['max_cm']} cm) and every gripper close / open comes on time.")
    parts = []
    if s["drift_at_s"] is not None:
        parts.append(f"drifts off the demo at {s['drift_at_s']} s during “{s['drift_subtask']}” (largest gap "
                     f"{s['max_cm']} cm, {s['max_arm']} arm, at {s['max_at_s']} s in “{s['max_subtask']}”)")
    for m in s["missed_gripper"][:3]:
        word = "grasp (close)" if m["kind"] == "close" else "release (open)"
        parts.append(f"misses the {m['gripper']} gripper {word} at {m['t_s']} s (“{m['subtask']}”)")
    if len(s["missed_gripper"]) > 3:
        parts.append(f"and {len(s['missed_gripper']) - 3} more gripper moments")
    return "Off the demo: " + "; ".join(parts) + "."
