"""Where things are in the FR3 Duo scene, without Isaac Sim: the arm bases, the cart top and the head camera.

The numbers come from the scene USD (Hackathon_FR3Duo_flattened.usd), extracted once with Isaac Sim's USD libraries
by our team's scripts (github.com/yugalrk/Robotics--VLA-training: tools/extract_scene_kinematics.py,
tools/head_camera_model.py) into geometry/*.json next to this file:
- fr3duo_scene_chain.json: per arm the world pose of link0 (the arm base on the mount), the joint frames, the
  scene's joint limits, the gripper centre (between the Robotiq fingertips) in link7's frame, and one authored pose
  with its link7 world matrix (self-check). Franka's DH chain (evaluation/kinematics.py) on top of link0's pose
  reproduces that link7 pose to < 1 µm, so the DH model works in room coordinates with these base poses.
- head_camera.json: the head camera's world pose and the recorder's intrinsics (224 × 224, fx = fy = 112 px).

The world is Y-up, in metres; the table plane is room X / Z. Pure numpy: Isaac Sim's Python imports it too.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

GEOMETRY = Path(__file__).resolve().parent / "geometry"

# Cart top surface (scene mesh, y 0.82–0.83) and what the box needs to stand on it
CART_TOP_X = (-5.154, -4.086)
CART_TOP_Z = (22.475, 23.089)
CART_Y = 0.829                 # cart top: the box rests here
CART_MARGIN = 0.03 + 0.055     # rim + half the box footprint: the box centre stays this far inside the edges
BOX_MID_Y = 0.88               # height of the box centre (back-projection plane for the camera)

# table-plane offset (dx, dy, 0) → world (dx, 0, dy): the plane's axes are room X and Z
TABLE_TO_WORLD = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@lru_cache(maxsize=None)
def chain() -> dict[str, Any]:
    return json.loads((GEOMETRY / "fr3duo_scene_chain.json").read_text())["arms"]


@lru_cache(maxsize=None)
def head_camera() -> dict[str, Any]:
    return json.loads((GEOMETRY / "head_camera.json").read_text())


def arm_base(side: str = "left") -> np.ndarray:
    """4 × 4 world pose of the arm's base (link0); `side` is "left" or "right"."""
    return np.array(chain()[side]["link0_world"])


def joint_limits(side: str = "left") -> tuple[np.ndarray, np.ndarray]:
    """The scene's joint limits (rad), as the simulator enforces them (slightly wider than the FR3 datasheet's)."""
    lim = np.radians(np.array([j["limits_deg"] for j in chain()[side]["joints"]]))
    return lim[:, 0], lim[:, 1]


def tcp_offset(side: str = "left") -> float:
    """Distance of the gripper centre (between the Robotiq fingertips) from link7 along its axis, in metres."""
    return float(np.linalg.norm(chain()[side]["tcp_in_link7"]))


def to_world(side: str, p_base: np.ndarray) -> np.ndarray:
    """(n, 3) points in the arm's base frame → world."""
    T = arm_base(side)
    return np.asarray(p_base, np.float64) @ T[:3, :3].T + T[:3, 3]


def tcp_world(side: str, q: np.ndarray) -> np.ndarray:
    """(n, 7) joint angles → (n, 3) world position of the gripper centre (between the Robotiq fingertips)."""
    from hackpipe.evaluation.kinematics import FLANGE_D, fk_T
    T = fk_T(q, tcp=False)                                                  # flange, in the arm's base frame
    return to_world(side, T[:, :3, 3] + T[:, :3, 2] * (tcp_offset(side) - FLANGE_D))


def link_points(side: str, q: np.ndarray) -> np.ndarray:
    """(7,) joint angles → (9, 3) world points of the arm drawn as a stick figure: base, the 7 joint frames (Franka's
    DH origins), gripper centre."""
    from hackpipe.evaluation.kinematics import DH, _mdh
    q = np.asarray(q, np.float64).reshape(7)
    T, pts = np.eye(4), [np.zeros(3)]
    for i, (a, d, alpha) in enumerate(DH):
        T = T @ _mdh(a, d, alpha, q[i:i + 1])[0]
        pts.append(T[:3, 3].copy())
    return np.vstack([to_world(side, np.array(pts)), tcp_world(side, q[None])])


def table_to_base(side: str = "left") -> np.ndarray:
    """3 × 3: table-plane offset (dx, dy, 0) → the same shift in the arm's base frame (the arms are mounted tilted,
    so the base's x / y are not the table plane)."""
    return arm_base(side)[:3, :3].T @ TABLE_TO_WORLD


def cart_limits(grasp_world: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Table-plane offsets (m) of the box from the demo's grasp point that keep it standing on the cart top:
    ((dx min, dx max), (dy min, dy max)), dy along room Z."""
    g = np.asarray(grasp_world, np.float64)
    return ((CART_TOP_X[0] + CART_MARGIN - g[0], CART_TOP_X[1] - CART_MARGIN - g[0]),
            (CART_TOP_Z[0] + CART_MARGIN - g[2], CART_TOP_Z[1] - CART_MARGIN - g[2]))


# ---------------------------------------------------------------- head camera (USD: looks along -Z, +Y up)
def world_to_pixel(P: np.ndarray, cam: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(n, 3) world points → (n, 2) pixels (u right, v down) and their depth along the view direction."""
    c = cam or head_camera()
    T = np.linalg.inv(np.array(c["world_from_camera"]))
    P = np.atleast_2d(np.asarray(P, np.float64))
    Pc = P @ T[:3, :3].T + T[:3, 3]
    z = -Pc[:, 2]
    return np.c_[c["cx"] + c["fx"] * Pc[:, 0] / z, c["cy"] - c["fy"] * Pc[:, 1] / z], z


def pixel_to_plane(uv: np.ndarray, height: float = BOX_MID_Y, cam: dict[str, Any] | None = None) -> np.ndarray:
    """(n, 2) pixels → (n, 3) world points on the horizontal plane y = height."""
    c = cam or head_camera()
    W = np.array(c["world_from_camera"])
    uv = np.atleast_2d(np.asarray(uv, np.float64))
    d = np.c_[(uv[:, 0] - c["cx"]) / c["fx"], -(uv[:, 1] - c["cy"]) / c["fy"], -np.ones(len(uv))] @ W[:3, :3].T
    s = (height - W[1, 3]) / d[:, 1]
    return W[:3, 3] + s[:, None] * d


def detect_box(img: np.ndarray, expect_uv: np.ndarray, radius: int = 16) -> np.ndarray | None:
    """Centroid (u, v) of the box in a head-camera frame (h, w, 3): the bright, low-saturation blob nearest to
    expect_uv on the dark cart, brightness-weighted for sub-pixel accuracy. None if there is no such blob."""
    a = np.asarray(img, np.float64)
    m = (a.mean(2) > 150) & ((a.max(2) - a.min(2)) < 40)
    h, w = m.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ex, ey = float(expect_uv[0]), float(expect_uv[1])
    near = (xx - ex) ** 2 + (yy - ey) ** 2
    seeds = np.argwhere(m & (near < radius ** 2))
    if not len(seeds):
        return None
    blob, stack = np.zeros_like(m), [tuple(seeds[np.argmin(near[seeds[:, 0], seeds[:, 1]])])]
    while stack:                                         # flood fill, kept within 3 radii of the expected spot
        y, x = stack.pop()
        if 0 <= y < h and 0 <= x < w and m[y, x] and not blob[y, x] and near[y, x] < (3 * radius) ** 2:
            blob[y, x] = True
            stack += [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
    ys, xs = np.nonzero(blob)
    wgt = a.mean(2)[ys, xs] - 140
    return np.array([(xs * wgt).sum() / wgt.sum(), (ys * wgt).sum() / wgt.sum()])


def box_offset(uv: np.ndarray, ref_world: np.ndarray, bias_m: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Box centroid pixel → its table-plane offset (dx, dy) in metres from ref_world (the demo's grasp point).
    bias_m: the offset measured on a frame where the box IS at the demo's spot (centroid vs grasp point)."""
    P = pixel_to_plane(uv)[0]
    ref = np.asarray(ref_world, np.float64)
    return np.array([P[0] - ref[0], P[2] - ref[2]]) - np.asarray(bias_m)


def nearest_cell(offset: np.ndarray, cells: list[dict[str, Any]]) -> dict[str, Any]:
    """The grid cell (dicts with "dx", "dy" in metres) closest to a table-plane offset."""
    o = np.asarray(offset, np.float64)
    return min(cells, key=lambda c: (c["dx"] - o[0]) ** 2 + (c["dy"] - o[1]) ** 2)
