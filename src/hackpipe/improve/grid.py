"""Box grid player: one handover demo → many valid demos with the box somewhere else on the table.

All 84 handover demos are the same motion (same box pose; only the lighting differs), so the policy never saw the
box anywhere else. The grid player shifts the box over a grid of offsets and bends the LEFT arm's pick so its grasp
point follows the box: the shift ramps in during the approach, holds through grasp and lift, and ramps out before
the handover, so handover, release and place stay the demo's. Each frame's grasp-point pose (position + the demo's
orientation) is solved by inverse kinematics on the FR3 (evaluation/kinematics.py). A cell is valid only if every
frame is reachable within joint limits, the IK error is tiny, and no joint moves faster than the demo allows.

Pure numpy (no hackpipe config): Isaac Sim's Python imports it in scripts/isaac/record_demos.py --mode grid, which
puts the box at each valid cell, replays the bent demo, records the cameras and keeps the episodes whose box ends on
the tray. Offsets here are in the left arm's base frame; record_demos.py converts the table-plane grid from world.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from hackpipe.evaluation.kinematics import DQ_MAX, Q_MAX, Q_MIN, fk_T, ik

LEFT, LEFT_GRIP, RIGHT_GRIP = slice(0, 7), 7, 15
CLOSED_BELOW = 0.25           # gripper command: 1.0 start, 0.48 open, 0.0 closed on the box

DEFAULTS = {
    "fps": 30.0,
    "approach_end_s": 0.3,    # the full shift is reached this long before the left grasp
    "lift_hold_s": 1.5,       # … and held this long after it (grasp + lift with the box)
    "return_before_handover_s": 1.0,   # back on the demo this long before the right gripper closes
    "max_ik_mm": 2.0, "max_ik_deg": 1.0,
    "limit_margin_rad": 0.05,
    "max_speed_share": 0.5,   # a joint may move at most this share of the FR3's speed limit
}


def grid_offsets(x_cm: tuple[float, float, float] = (-8, 8, 1), y_cm: tuple[float, float, float] = (-3.5, 3.5, 1)
                 ) -> list[tuple[float, float]]:
    """(dx, dy) in metres over a table-plane grid (start, stop, step in cm, both inclusive): 17 × 8 = 136 cells by
    default."""
    xs = np.round(np.arange(x_cm[0], x_cm[1] + 1e-9, x_cm[2]), 4)
    ys = np.round(np.arange(y_cm[0], y_cm[1] + 1e-9, y_cm[2]), 4)
    return [(float(x) / 100, float(y) / 100) for x in xs for y in ys]


def phases(actions: np.ndarray, cfg: dict[str, Any] | None = None) -> dict[str, int]:
    """Frames of the left grasp, the end of the approach ramp, the end of the lift hold and the end of the return."""
    c = {**DEFAULTS, **(cfg or {})}
    fps = c["fps"]
    lg = np.flatnonzero(actions[:, LEFT_GRIP] < CLOSED_BELOW)
    rg = np.flatnonzero(actions[:, RIGHT_GRIP] < CLOSED_BELOW)
    if not len(lg) or not len(rg):
        raise ValueError("the demo has no left / right grasp (gripper command < 0.25)")
    grasp, handover = int(lg[0]), int(rg[0])
    a1 = max(1, grasp - int(round(c["approach_end_s"] * fps)))
    l1 = grasp + int(round(c["lift_hold_s"] * fps))
    r1 = handover - int(round(c["return_before_handover_s"] * fps))
    if not 0 < a1 < grasp < l1 < r1 < handover:
        raise ValueError(f"demo timing does not fit the grid ramps (grasp {grasp}, lift end {l1}, return {r1}, "
                         f"handover {handover})")
    return {"grasp": grasp, "approach_end": a1, "lift_end": l1, "return_end": r1, "handover": handover}


def weight(n: int, ph: dict[str, int]) -> np.ndarray:
    """0 → 1 (cosine) over the approach, 1 through grasp + lift, 1 → 0 back to the demo before the handover."""
    w = np.zeros(n)
    t = np.arange(n)
    up = t < ph["approach_end"]
    w[up] = 0.5 - 0.5 * np.cos(np.pi * t[up] / ph["approach_end"])
    w[(t >= ph["approach_end"]) & (t < ph["lift_end"])] = 1.0
    down = (t >= ph["lift_end"]) & (t < ph["return_end"])
    w[down] = 0.5 + 0.5 * np.cos(np.pi * (t[down] - ph["lift_end"]) / (ph["return_end"] - ph["lift_end"]))
    return w


def retarget(actions: np.ndarray, offset: np.ndarray, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Bend the demo (T, 16) so the left grasp point is shifted by `offset` (3,) m in the left arm's base frame while
    the box is in the left hand. Returns {"ok", "reason", "actions" (T, 16), "max_ik_mm", "max_ik_deg", "phases"}."""
    c = {**DEFAULTS, **(cfg or {})}
    A = np.asarray(actions, np.float64)
    off = np.asarray(offset, np.float64).reshape(3)
    ph = phases(A, c)
    w = weight(len(A), ph)
    out = A.copy()
    q = A[0, LEFT].copy()
    max_mm = max_deg = 0.0
    for t in np.flatnonzero(w > 0):
        T = fk_T(A[t, LEFT][None])[0]
        T[:3, 3] += w[t] * off
        q, pe, re = ik(T, q if t > 0 else A[t, LEFT], q_ref=A[t, LEFT])      # keep the demo's elbow posture
        out[t, LEFT] = q
        max_mm, max_deg = max(max_mm, pe * 1000), max(max_deg, float(np.degrees(re)))
    res = {"actions": out.astype(np.float32), "max_ik_mm": round(max_mm, 3), "max_ik_deg": round(max_deg, 3),
           "phases": ph, "offset_m": off.round(4).tolist()}
    step = np.abs(np.diff(out[:, LEFT], axis=0)).max(0)
    m = c["limit_margin_rad"]
    if max_mm > c["max_ik_mm"] or max_deg > c["max_ik_deg"]:
        return {**res, "ok": False, "reason": f"not reachable (IK error {max_mm:.1f} mm / {max_deg:.2f}°)"}
    if (out[:, LEFT] < Q_MIN + m).any() or (out[:, LEFT] > Q_MAX - m).any():
        return {**res, "ok": False, "reason": "too close to a joint limit"}
    limit = c["max_speed_share"] * DQ_MAX / c["fps"]                    # rad per frame
    if (step > limit).any():
        j = int(np.argmax(step / limit))
        return {**res, "ok": False, "reason": f"joint {j + 1} would move too fast ({step[j] * c['fps']:.2f} rad/s)"}
    return {**res, "ok": True, "reason": ""}


def plan(actions: np.ndarray, offsets: list[tuple[float, float]], rotate: np.ndarray | None = None,
         cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Retarget the demo for every table-plane offset. rotate (3, 3): table-plane offset → left-arm base frame (from
    the scene; identity = the base's x / y are the table plane). Returns one result per cell (with "cell", "dx",
    "dy")."""
    R = np.eye(3) if rotate is None else np.asarray(rotate, np.float64)
    out = []
    for i, (dx, dy) in enumerate(offsets):
        r = retarget(actions, R @ np.array([dx, dy, 0.0]), cfg)
        out.append({**r, "cell": i, "dx": dx, "dy": dy})
    return out
