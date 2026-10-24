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


# FR3 joint limits (rad), from Franka's FR3 datasheet
Q_MIN = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
Q_MAX = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
DQ_MAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])          # joint speed limits (rad/s)


def fk_T(q: np.ndarray, tcp: bool = True) -> np.ndarray:
    """q (n, 7) → (n, 4, 4) pose of the grasp point (or flange) in the arm's base frame."""
    q = np.asarray(q, np.float64).reshape(-1, 7)
    T = np.repeat(np.eye(4)[None], len(q), 0)
    for i, (a, d, alpha) in enumerate(DH):
        T = T @ _mdh(a, d, alpha, q[:, i])
    return T @ _mdh(0.0, FLANGE_D + (TCP_D if tcp else 0.0), 0.0, np.zeros(len(q)))


def _rot_err(R_target: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotation vector taking R to R_target (axis · angle, rad)."""
    Re = R_target @ R.T
    c = np.clip((np.trace(Re) - 1) / 2, -1.0, 1.0)
    ang = np.arccos(c)
    v = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    if ang < 1e-9:
        return 0.5 * v
    if np.pi - ang < 1e-6:                      # 180°: axis from the symmetric part
        axis = np.sqrt(np.maximum((np.diag(Re) + 1) / 2, 0))
        return axis * ang
    return v * ang / (2 * np.sin(ang))


def ik(T_target: np.ndarray, q0: np.ndarray, iters: int = 40, damping: float = 1e-2, tol_m: float = 2e-4,
       tol_rad: float = 2e-3, q_ref: np.ndarray | None = None, k_null: float = 0.5) -> tuple[np.ndarray, float, float]:
    """Damped-least-squares IK for one 4×4 target pose, from q0 (7,). The FR3 has 7 joints for a 6-D pose: with
    q_ref, the spare degree of freedom is pulled toward it (null-space term), so the solution stays in that posture
    instead of drifting. Returns (q, position error m, rotation error rad); joint limits are enforced by clipping."""
    q = np.asarray(q0, np.float64).copy()
    eps = 1e-6
    for _ in range(iters):
        T = fk_T(q[None])[0]
        e = np.r_[T_target[:3, 3] - T[:3, 3], _rot_err(T_target[:3, :3], T[:3, :3])]
        if np.linalg.norm(e[:3]) < tol_m and np.linalg.norm(e[3:]) < tol_rad and q_ref is None:
            break
        Qp = q[None] + eps * np.eye(7)                         # numerical Jacobian, one batched FK
        Tp = fk_T(Qp)
        J = np.zeros((6, 7))
        for j in range(7):
            J[:3, j] = (Tp[j, :3, 3] - T[:3, 3]) / eps
            J[3:, j] = _rot_err(Tp[j, :3, :3], T[:3, :3]) / eps
        Jp = J.T @ np.linalg.inv(J @ J.T + damping ** 2 * np.eye(6))
        dq = Jp @ e
        if q_ref is not None:                                 # posture: stay near q_ref without moving the grasp point
            dq += (np.eye(7) - Jp @ J) @ (k_null * (np.asarray(q_ref) - q))
        q = np.clip(q + dq, Q_MIN, Q_MAX)
        if q_ref is not None and np.linalg.norm(e[:3]) < tol_m and np.linalg.norm(e[3:]) < tol_rad \
                and np.linalg.norm(dq) < 1e-5:
            break
    T = fk_T(q[None])[0]
    return q, float(np.linalg.norm(T_target[:3, 3] - T[:3, 3])), float(np.linalg.norm(_rot_err(T_target[:3, :3], T[:3, :3])))
