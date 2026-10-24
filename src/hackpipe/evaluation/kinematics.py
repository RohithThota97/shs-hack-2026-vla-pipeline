"""Forward kinematics of one Franka FR3 arm (the FR3 Duo has two), for the 3D trajectories on the Evaluate page.

Franka's published kinematics (modified DH, Craig convention; the FR3 has the Panda's geometry): per joint
(a, d, alpha), theta = the joint angle; then the flange (d = 0.107) and the Franka Hand's grasp point (TCP,
0.1034 m further along the flange axis). Positions are in the arm's own base frame: the dual-arm mounting offset is
not in the dataset, so each arm is drawn on its own.
"""
from __future__ import annotations

import numpy as np

DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
      (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]
FLANGE_D = 0.107
TCP_D = 0.1034


def _mdh(a: float, d: float, alpha: float, theta: np.ndarray) -> np.ndarray:
    """(n, 4, 4) transforms RotX(alpha) · TransX(a) · RotZ(theta) · TransZ(d)."""
    n = theta.shape[0]
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    T = np.zeros((n, 4, 4))
    T[:, 0, 0], T[:, 0, 1], T[:, 0, 3] = ct, -st, a
    T[:, 1, 0], T[:, 1, 1], T[:, 1, 2], T[:, 1, 3] = st * ca, ct * ca, -sa, -sa * d
    T[:, 2, 0], T[:, 2, 1], T[:, 2, 2], T[:, 2, 3] = st * sa, ct * sa, ca, ca * d
    T[:, 3, 3] = 1.0
    return T


def fk(q: np.ndarray, tcp: bool = True) -> np.ndarray:
    """q (n, 7) joint angles (rad) → (n, 3) position of the grasp point (tcp=True) or the flange, in metres."""
    q = np.asarray(q, np.float64).reshape(-1, 7)
    T = np.repeat(np.eye(4)[None], len(q), 0)
    for i, (a, d, alpha) in enumerate(DH):
        T = T @ _mdh(a, d, alpha, q[:, i])
    T = T @ _mdh(0.0, FLANGE_D + (TCP_D if tcp else 0.0), 0.0, np.zeros(len(q)))
    return T[:, :3, 3]


def arm_paths(actions: np.ndarray) -> dict[str, np.ndarray]:
    """FR3 Duo action / state rows (n, 16: left 0-6, left gripper 7, right 8-14, right gripper 15) → grasp-point paths
    {"left": (n, 3), "right": (n, 3)} in each arm's base frame."""
    a = np.asarray(actions, np.float64)
    return {"left": fk(a[:, 0:7]), "right": fk(a[:, 8:15])}
