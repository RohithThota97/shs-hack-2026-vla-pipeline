"""Static explainer pages (`hackpipe pages <page>`): each builder reads a recorded dataset (and the sub-task labels of
the newest dataset version that has them) and writes a folder of static files — index.html (a template from
templates/), data.json, plotly.min.js and, where the page plays it, the video. Serve a folder with `hackpipe pages
serve --dir <folder>` (HTTP Range support, so the videos can seek).

Pages, after our team repo's demo_ui/ (github.com/yugalrk/Robotics--VLA-training):
  site       the handover explorer: 1 the data (joint matrix + video), 2 the robot model (forward kinematics, and
             inverse kinematics with and without one extra rule), 3 camera → grid → trajectory (the grid player)
  grid       box grid player: the whole-cart grid (improve/grid.py), each cell's bent trajectory, the head camera
             with a draggable box (camera → table plane → nearest cell)
  subtasks   how the sub-task labeller (dataset/subtasks.py) cuts one handover episode
  xyz        both grippers' XYZ (forward kinematics) next to the video, for every episode in one video file
  tcp        one gripper's path over a time window (default: the 20-action chunk before the left grasp), one file
  retarget   one box shift: the demo's gripper path, the bent one, and the joint changes

All positions are room coordinates (metres, Y up) from the scene geometry (sim/scene.py).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hackpipe.common.config import resolve
from hackpipe.common.lerobot_io import DatasetFolder
from hackpipe.sim import scene

TEMPLATES = Path(__file__).resolve().parent / "templates"
SOURCE = "data/raw/fr3_duo_handover_84"
CAMS = ["external_camera", "left_wrist_camera", "right_wrist_camera"]
NAMES = [f"left joint {i}" for i in range(1, 8)] + ["left gripper"] + [f"right joint {i}" for i in range(1, 8)] \
    + ["right gripper"]
SCENE_BOX = {"min": [-4.265, 0.829, 22.721], "max": [-4.234, 0.901, 22.828]}   # the hand-out scene's box (bounds)


# ---------------------------------------------------------------- helpers
def plotly_js() -> Path:
    try:
        import plotly
    except ImportError as e:                                    # the console extra brings it
        raise RuntimeError("the demo pages need plotly: pip install -e '.[console]'") from e
    return Path(plotly.__file__).parent / "package_data" / "plotly.min.js"


def _page(out: Path, template: str, data: dict[str, Any] | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATES / template, out / "index.html")
    shutil.copy(plotly_js(), out / "plotly.min.js")
    if data is not None:
        (out / "data.json").write_text(json.dumps(data))
    return out


def episode(ds: DatasetFolder, ep: int) -> pd.DataFrame:
    d = ds.frames()
    d = d[d["episode_index"] == ep].sort_values("frame_index")
    if d.empty:
        raise ValueError(f"episode {ep} not in {ds.root}")
    return d


def arrays(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = d["timestamp"].to_numpy(float) if "timestamp" in d else d["frame_index"].to_numpy(float) / 30
    return np.stack(d["observation.state"]).astype(float), np.stack(d["action"]).astype(float), t


def labels_for(source: str) -> tuple[dict[str, Any], Path] | None:
    """(labels.json entry, its folder) for a source from the newest dataset version with sub-task labels."""
    best = None
    for f in resolve("outputs/datasets").glob("*/subtasks/labels.json"):
        idx = json.loads(f.read_text())
        for key, v in idx.get("sources", {}).items():
            if Path(key).name == Path(source).name and (best is None or idx.get("created_at", "") > best[0]):
                best = (idx.get("created_at", ""), v, f.parent)
    return (best[1], best[2]) if best else None


def frame_subtasks(source: str, ep: int, n: int) -> list[str]:
    """Sub-task text per frame of one episode (empty strings if the source isn't labelled)."""
    lab = labels_for(source)
    if not lab:
        return [""] * n
    seg = pd.read_csv(lab[1] / lab[0]["segments"])
    seg = seg[seg["episode_index"] == ep]
    out = [""] * n
    for r in seg.itertuples():
        for k in range(int(r.start_frame), min(int(r.end_frame), n)):
            out[k] = r.text
    return out


def _video_of(ds: DatasetFolder, cam: str, ep: int) -> tuple[Path, float]:
    k = f"videos/observation.images.{cam}"
    e = ds.episodes_meta().set_index("episode_index").loc[ep]
    return (ds.episode_video_file(f"observation.images.{cam}", e[k + "/chunk_index"], e[k + "/file_index"]),
            float(e[k + "/from_timestamp"]))


# ---------------------------------------------------------------- site: 1 data, 2 model, 3 grid
def page_data(ds: DatasetFolder, source: str, out: Path, every: int = 2) -> dict[str, Any]:
    """Every episode stored in the first external-camera video file: state / action matrix + sub-task per frame."""
    k = "videos/observation.images.external_camera/"
    eps = ds.episodes_meta()
    eps = eps[(eps[k + "chunk_index"] == 0) & (eps[k + "file_index"] == 0)].sort_values("episode_index")
    res = {"names": NAMES, "fps": ds.fps / every, "episodes": []}
    for e in eps.itertuples(index=False):
        ep = int(getattr(e, "episode_index"))
        d = episode(ds, ep)
        S, A, t = arrays(d)
        lab = frame_subtasks(source, ep, len(S))[::every]
        res["episodes"].append({"episode": ep, "video_from": float(eps.loc[eps.episode_index == ep, k + "from_timestamp"]
                                                                   .iloc[0]),
                                "t": t[::every].round(3).tolist(), "state": S[::every].round(4).tolist(),
                                "action": A[::every].round(4).tolist(), "subtask": lab})
    _page(out, "site/data.html", res)
    shutil.copy(_video_of(ds, "external_camera", int(eps.episode_index.iloc[0]))[0], out / "video.mp4")
    return {"episodes": len(res["episodes"])}


def recover_joints(q_rec: np.ndarray, fix7: bool, iters: int = 50, damping: float = 1e-3) -> np.ndarray:
    """Joints back from the gripper poses of q_rec (n, 7), frame by frame. fix7=False: no extra rule — the spare
    7th degree of freedom only stays near the previous frame, so the joints may drift from the recording (many
    solutions). fix7=True: joint 7 is given (the recorded angle), the other 6 are solved: one solution per frame."""
    from hackpipe.evaluation.kinematics import Q_MAX, Q_MIN, _rot_err, fk_T
    targets = fk_T(q_rec)
    q, out, eps = q_rec[0].astype(float).copy(), [], 1e-6
    for k, Tt in enumerate(targets):
        if fix7:
            q[6] = q_rec[k, 6]
        prev = q.copy()
        for _ in range(iters):
            T = fk_T(q[None])[0]
            e = np.r_[Tt[:3, 3] - T[:3, 3], _rot_err(Tt[:3, :3], T[:3, :3])]
            if np.linalg.norm(e) < 1e-9:
                break
            Tp = fk_T(q[None] + eps * np.eye(7))
            J = np.zeros((6, 7))
            for j in range(7):
                J[:3, j] = (Tp[j, :3, 3] - T[:3, 3]) / eps
                J[3:, j] = _rot_err(Tp[j, :3, :3], T[:3, :3]) / eps
            if fix7:
                J6 = J[:, :6]
                q[:6] += J6.T @ np.linalg.solve(J6 @ J6.T + damping ** 2 * np.eye(6), e)
            else:
                Jp = J.T @ np.linalg.inv(J @ J.T + damping ** 2 * np.eye(6))
                q += Jp @ e + (np.eye(7) - Jp @ J) @ (0.5 * (prev - q))
            q = np.clip(q, Q_MIN - 0.2, Q_MAX + 0.2)       # the scene's limits are a little wider than the datasheet's
        out.append(q.copy())
    return np.array(out)


def page_model(ds: DatasetFolder, out: Path, ep: int = 0, every: int = 2) -> dict[str, Any]:
    from hackpipe.evaluation.kinematics import fk_T
    S, _, t = arrays(episode(ds, ep))
    S, t = S[::every], t[::every]
    QL, QR = S[:, :7], S[:, 8:15]
    QA, QB = recover_joints(QL, fix7=False), recover_joints(QL, fix7=True)
    P = fk_T(QL)[:, :3, 3]
    err = lambda Q: float(np.linalg.norm(fk_T(Q)[:, :3, 3] - P, axis=1).max() * 1000)  # noqa: E731
    stats = {"A_pose_err_mm": round(err(QA), 4), "B_pose_err_mm": round(err(QB), 4),
             "A_joint_diff_deg": round(float(np.degrees(np.abs(QA - QL).max())), 2),
             "B_joint_diff_deg": round(float(np.degrees(np.abs(QB - QL).max())), 4)}
    pts = lambda side, Q: np.array([scene.link_points(side, q) for q in Q]).round(4).tolist()  # noqa: E731
    _page(out, "site/model.html", {
        "t": t.round(3).tolist(), "left": pts("left", QL), "right": pts("right", QR), "leftA": pts("left", QA),
        "q_rec": np.degrees(QL).round(3).tolist(), "q_A": np.degrees(QA).round(3).tolist(),
        "q_B": np.degrees(QB).round(3).tolist(), "grip_l": S[:, 7].round(3).tolist(),
        "grip_r": S[:, 15].round(3).tolist(), "stats": stats})
    return stats


def build_site(out: Path, source: str = SOURCE, step_cm: float = 5.0) -> dict[str, Any]:
    ds = DatasetFolder(resolve(source))
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATES / "site" / "index.html", out / "index.html")
    return {"1-data": page_data(ds, source, out / "1-data"), "2-model": page_model(ds, out / "2-model"),
            "3-lut": build_grid(out / "3-lut", source, step_cm)}


# ---------------------------------------------------------------- grid player
def cell_stats(A: np.ndarray, r: dict[str, Any], ph: dict[str, int]) -> dict[str, Any]:
    """The player's numbers for one planned cell (improve/grid.py plan())."""
    from hackpipe.evaluation.kinematics import Q_MAX, Q_MIN
    base = {"box_dx_mm": round(r["dx"] * 1000, 1), "box_dz_mm": round(r["dy"] * 1000, 1), "status": r["status"],
            "reason": r["reason"], "episode_index": -1}
    if r["status"] == "off_cart":
        return base
    Q, Qn = A[:, :7], np.asarray(r["actions"], float)[:, :7]
    g = scene.tcp_world("left", Qn[ph["grasp"]][None])[0]
    return {**base, "max_pos_err_mm": r["max_ik_mm"], "max_rot_err_deg": r["max_ik_deg"],
            "max_joint_change_deg": round(float(np.degrees(np.abs(Qn - Q).max())), 2),
            "max_step_deg": round(float(np.degrees(np.abs(np.diff(Qn, axis=0)).max())), 3),
            "min_limit_margin_deg": round(float(np.degrees(np.minimum(Qn - Q_MIN, Q_MAX - Qn).min())), 1),
            "grasp_x": round(float(g[0]), 5), "grasp_y": round(float(g[1]), 5), "grasp_z": round(float(g[2]), 5)}


def camera_assets(ds: DatasetFolder, A: np.ndarray, out: Path, ep: int = 0, at_s: float = 1.0) -> dict[str, Any] | None:
    """Head-camera frame with the box removed (background), the box as a sprite, and the calibration: in this real
    frame the box IS at the demo's spot, so its measured offset is the constant bias (box centroid vs grasp point)."""
    from PIL import Image

    from hackpipe.dataset.subtasks import Videos
    from hackpipe.improve.grid import grasp_world
    img = Videos(ds).frames(ep, "external_camera", [int(round(at_s * ds.fps))])[0]
    ref = grasp_world(A)
    ref_uv = scene.world_to_pixel(np.array([[ref[0], scene.BOX_MID_Y, ref[2]]]))[0][0]
    c = scene.detect_box(img, ref_uv)
    if c is None:
        return None
    a = img.astype(float)
    m = (a.mean(2) > 150) & ((a.max(2) - a.min(2)) < 40)
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    blob = m & ((xx - c[0]) ** 2 + (yy - c[1]) ** 2 < 16 ** 2)
    ys, xs = np.nonzero(blob)
    pad = np.zeros_like(blob)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            pad[np.clip(ys + dy, 0, m.shape[0] - 1), np.clip(xs + dx, 0, m.shape[1] - 1)] = True
    ring = np.zeros_like(blob)
    ring[max(0, ys.min() - 4):ys.max() + 5, max(0, xs.min() - 4):xs.max() + 5] = True
    ring &= ~pad
    bg = img.copy()
    bg[pad] = np.median(img[ring], axis=0).astype(np.uint8)
    y0, y1, x0, x1 = max(0, ys.min() - 1), ys.max() + 2, max(0, xs.min() - 1), xs.max() + 2
    rgba = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    rgba[..., :3] = img[y0:y1, x0:x1]
    rgba[..., 3] = (pad[y0:y1, x0:x1] * 255).astype(np.uint8)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(bg).save(out / "background.png")
    Image.fromarray(rgba, "RGBA").save(out / "box_sprite.png")
    bias = scene.box_offset(c, ref) * 100
    p0 = np.array([[ref[0] + bias[0] / 100, scene.BOX_MID_Y, ref[2] + bias[1] / 100]])
    cam = scene.head_camera()
    info = {"camera": cam, "box_mid_y": scene.BOX_MID_Y, "cart_y": scene.CART_Y, "bias_cm": bias.round(3).tolist(),
            "ref_xz": [float(ref[0]), float(ref[2])], "ref_uv": ref_uv.round(2).tolist(),
            "box_centroid_px": c.round(2).tolist(), "box_depth": round(float(scene.world_to_pixel(p0)[1][0]), 4),
            "sprite": {"x0": int(x0), "y0": int(y0), "w": int(x1 - x0), "h": int(y1 - y0)},
            "frame": f"episode {ep}, {at_s} s",
            "px_per_cm": round(float(np.linalg.norm(scene.world_to_pixel(
                np.array([[ref[0] + 0.01, scene.BOX_MID_Y, ref[2]]]))[0][0] - ref_uv)), 2)}
    (out / "camera.json").write_text(json.dumps(info))
    return info


def build_grid(out: Path, source: str = SOURCE, step_cm: float = 5.0, every: int = 4, ep: int = 0
               ) -> dict[str, Any]:
    from hackpipe.improve.grid import cart_offsets, grid_offsets, phases, plan
    ds = DatasetFolder(resolve(source))
    S, A, t = arrays(episode(ds, ep))
    fps = ds.fps
    ph = phases(A, {"fps": fps})
    xr, yr = cart_offsets(A, step_cm, {"fps": fps})
    cells = plan(A, grid_offsets(xr, yr), cfg={"fps": fps}, cart=True)
    (out / "paths").mkdir(parents=True, exist_ok=True)
    rows, n_ok = [], 0
    for r in cells:
        row = cell_stats(A, r, ph)
        if r["ok"]:
            row["episode_index"] = n_ok
            Qn = np.asarray(r["actions"], float)[:, :7]
            (out / "paths" / f"{n_ok}.json").write_text(json.dumps({
                "path": scene.tcp_world("left", Qn[::every]).round(5).tolist(),
                "dq_deg": np.degrees(Qn - A[:, :7])[::every].round(2).tolist()}))
            n_ok += 1
        rows.append(row)
    lim = scene.cart_limits(scene.tcp_world("left", A[ph["grasp"], :7][None])[0])
    axis = lambda r: (np.round(np.arange(r[0], r[1] + 1e-9, r[2]), 3) * 10).tolist()  # noqa: E731 — mm
    index = {"dx_mm": axis(xr), "dz_mm": axis(yr), "step_mm": step_cm * 10, "t": t[::every].round(3).tolist(),
             "grasp_t": float(t[ph["grasp"]]), "orig": scene.tcp_world("left", A[::every, :7]).round(5).tolist(),
             "grip": S[::every, 7].round(3).tolist(),
             "blend": [[0.0, ph["approach_end"] / fps], [ph["lift_end"] / fps, ph["return_end"] / fps]],
             "cart_limits_cm": {"dx": [v * 100 for v in lim[0]], "dz": [v * 100 for v in lim[1]]}, "cells": rows}
    _page(out, "grid_player.html")
    (out / "index.json").write_text(json.dumps(index))
    cam = camera_assets(ds, A, out / "camera", ep)
    counts = {s: sum(r["status"] == s for r in rows) for s in ("ok", "out_of_reach", "off_cart")}
    return {**counts, "camera": cam is not None}


# ---------------------------------------------------------------- sub-task cuts
def build_subtasks(out: Path, source: str = SOURCE, ep: int = 0) -> dict[str, Any]:
    from PIL import Image

    from hackpipe.dataset import subtasks as sb
    ds = DatasetFolder(resolve(source))
    S, A, _ = arrays(episode(ds, ep))
    fps, j = ds.fps, sb.Joints(sb.subtasks_config())
    mv_l, mv_r = sb.moving(S, j.l_arm, fps, j.still), sb.moving(S, j.r_arm, fps, j.still)
    b, ev, still = sb.segment("handover", S, A, fps, j)
    problems = sb.check("handover", b, ev, still, A, j)
    if b is None:
        raise ValueError(f"episode {ep}: {problems}")
    _, subs = sb.texts("handover")
    lc_done = sb.grip_settled(S, j.l_g, ev["left_close"], fps) or ev["left_close"]
    rc_done = sb.grip_settled(S, j.r_g, ev["right_close"], fps) or ev["right_close"]
    starts = [sb.first(mv_l, lc_done), sb.first(mv_r, lc_done), ev["left_open"], sb.first(mv_r, rc_done)]
    why = ["first left-arm motion after the left gripper finished closing",
           "first right-arm motion after that (the right arm starts its reach)",
           "the left gripper command starts opening",
           "first right-arm motion after the right gripper finished closing"]
    (out / "snaps").mkdir(parents=True, exist_ok=True)
    videos = sb.Videos(ds)
    cuts = []
    for k, (e, n) in enumerate(zip(starts, still)):
        tiles = [videos.frames(ep, c, [b[k + 1]])[0] for c in CAMS]
        Image.fromarray(np.concatenate(tiles, 1)).save(out / "snaps" / f"cut{k + 1}.png")
        cuts.append({"frame": b[k + 1], "event_frame": int(e), "pause_from": int(e - n), "pause_to": int(e),
                     "why": why[k]})
    qa = {}
    for kind, src in (("handover", source), ("cubes", "data/raw/fr3_duo_rand500")):
        lab = labels_for(src)
        qa[kind] = ({"episodes": lab[0]["episodes"], "flagged": len(lab[0]["flagged"]),
                     "durations": lab[0].get("subtask_seconds", {}), "problems": {}} if lab else
                    {"episodes": 0, "flagged": 0, "durations": {}, "problems": {}})
    speed = lambda arm: np.r_[np.abs(np.diff(S[:, arm], axis=0)).sum(1) * fps, 0.0]  # noqa: E731
    video, start = _video_of(ds, "external_camera", ep)
    _page(out, "subtasks.html", {
        "episode": ep, "fps": fps, "n": len(S), "video_from": start, "t": (np.arange(len(S)) / fps).round(3).tolist(),
        "speed_l": speed(j.l_arm).round(4).tolist(), "speed_r": speed(j.r_arm).round(4).tolist(),
        "moving_l": mv_l.astype(int).tolist(), "moving_r": mv_r.astype(int).tolist(),
        "grip_cmd_l": A[:, j.l_g].round(3).tolist(), "grip_cmd_r": A[:, j.r_g].round(3).tolist(),
        "threshold": j.still, "bounds": [int(x) for x in b], "subtasks": subs, "cuts": cuts,
        "events": {k: (None if v is None else int(v)) for k, v in ev.items()}, "problems": problems, "qa": qa})
    shutil.copy(video, out / "video.mp4")
    return {"bounds": [int(x) for x in b], "problems": problems}


# ---------------------------------------------------------------- XYZ next to the video
def build_xyz(out: Path, source: str = SOURCE, camera: str = "external_camera", chunk: int = 0, file: int = 0
              ) -> dict[str, Any]:
    ds = DatasetFolder(resolve(source))
    k = f"videos/observation.images.{camera}/"
    eps = ds.episodes_meta()
    eps = eps[(eps[k + "chunk_index"] == chunk) & (eps[k + "file_index"] == file)].sort_values("episode_index")
    if eps.empty:
        raise ValueError(f"no episode in {camera} chunk {chunk} file {file}")
    res = {"camera": camera, "video": f"file-{file:03d}.mp4", "fps": ds.fps, "episodes": []}
    for e in eps.itertuples(index=False):
        ep = int(getattr(e, "episode_index"))
        S, A, t = arrays(episode(ds, ep))
        res["episodes"].append({
            "episode": ep, "video_from": float(eps.loc[eps.episode_index == ep, k + "from_timestamp"].iloc[0]),
            "t": t.round(4).tolist(), "left": scene.tcp_world("left", S[:, :7]).round(4).tolist(),
            "right": scene.tcp_world("right", S[:, 8:15]).round(4).tolist(),
            "grip_left": S[:, 7].round(3).tolist(), "grip_right": S[:, 15].round(3).tolist(),
            "cmd_left": A[:, 7].round(3).tolist(), "cmd_right": A[:, 15].round(3).tolist(),
            "subtask": frame_subtasks(source, ep, len(S))})
    _page(out, "xyz_verify.html", res)
    shutil.copy(ds.episode_video_file(f"observation.images.{camera}", chunk, file), out / "video.mp4")
    return {"episodes": len(res["episodes"])}


# ---------------------------------------------------------------- one gripper over a time window
def build_tcp(out_file: Path, source: str = SOURCE, ep: int = 0, arm: str = "left", start_s: float = 7.6,
              seconds: float = 20 / 6) -> dict[str, Any]:
    """One self-contained HTML file (plotly inlined): the gripper centre over [start_s, start_s + seconds], with
    the policy's 6 Hz steps marked; default = the 20-action chunk that ends at the left grasp."""
    ds = DatasetFolder(resolve(source))
    S, _, _ = arrays(episode(ds, ep))
    fps = ds.fps
    a, b = int(round(start_s * fps)), min(len(S) - 1, int(round((start_s + seconds) * fps)))
    sl, g = (slice(0, 7), 7) if arm == "left" else (slice(8, 15), 15)
    P = scene.tcp_world(arm, S[a:b + 1, sl])
    data = {"x": P[:, 0].tolist(), "y": P[:, 1].tolist(), "z": P[:, 2].tolist(),
            "t": (np.arange(a, b + 1) / fps).round(3).tolist(), "grip": S[a:b + 1, g].round(3).tolist(),
            "steps": list(range(0, b - a + 1, max(1, int(round(fps / 6))))), "arm": arm, "episode": ep,
            "box": SCENE_BOX, "path_len_mm": float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() * 1000),
            "displacement_mm": np.round((P[-1] - P[0]) * 1000, 1).tolist()}
    html = (TEMPLATES / "tcp_window.html").read_text(encoding="utf-8")
    html = html.replace("/*DATA*/null", json.dumps(data)).replace(
        "/*PLOTLY*/", plotly_js().read_text(encoding="utf-8").replace("</script", "<\\/script"))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html, encoding="utf-8")
    return {"frames": len(P), "path_len_mm": round(data["path_len_mm"], 1), "displacement_mm": data["displacement_mm"]}


# ---------------------------------------------------------------- one box shift
def build_retarget(out: Path, dx_cm: float = 3.0, dy_cm: float = 0.0, source: str = SOURCE, ep: int = 0
                   ) -> dict[str, Any]:
    from hackpipe.evaluation.kinematics import Q_MAX, Q_MIN
    from hackpipe.improve.grid import phases, retarget, weight
    ds = DatasetFolder(resolve(source))
    S, A, t = arrays(episode(ds, ep))
    cfg = {"fps": ds.fps}
    ph = phases(A, cfg)
    r = retarget(A, scene.table_to_base("left") @ np.array([dx_cm / 100, dy_cm / 100, 0.0]), cfg)
    Q, Qn = A[:, :7], np.asarray(r["actions"], float)[:, :7]
    fps = ds.fps
    margin = np.degrees(np.minimum(Qn - Q_MIN, Q_MAX - Qn).min(0))
    data = {"name": f"dx {dx_cm:+g} cm, dz {dy_cm:+g} cm", "shift_mm": [dx_cm * 10, 0.0, dy_cm * 10],
            "blend_in": [0.0, ph["approach_end"] / fps], "blend_out": [ph["lift_end"] / fps, ph["return_end"] / fps],
            "t": t.round(4).tolist(), "w": weight(len(A), ph).round(4).tolist(),
            "orig": scene.tcp_world("left", Q).round(5).tolist(), "new": scene.tcp_world("left", Qn).round(5).tolist(),
            "q_orig": np.degrees(Q).round(3).tolist(), "q_new": np.degrees(Qn).round(3).tolist(),
            "grip": S[:, 7].round(3).tolist(), "grasp_t": float(t[ph["grasp"]]),
            "stats": {"ok": r["ok"], "reason": r["reason"], "max_pos_err_mm": r["max_ik_mm"],
                      "max_rot_err_deg": r["max_ik_deg"],
                      "max_joint_change_deg": round(float(np.degrees(np.abs(Qn - Q).max())), 2),
                      "max_step_deg": round(float(np.degrees(np.abs(np.diff(Qn, axis=0)).max())), 3),
                      "rec_step_deg": round(float(np.degrees(np.abs(np.diff(Q, axis=0)).max())), 3),
                      "min_limit_margin_deg": round(float(margin.min()), 1)}}
    _page(out, "retarget.html", data)
    return data["stats"]


# ---------------------------------------------------------------- static server with Range (seekable videos)
def make_server(directory: str | Path, port: int = 8779, host: str = "127.0.0.1") -> Any:
    import os
    import re
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    class RangeHandler(SimpleHTTPRequestHandler):
        def send_head(self):  # noqa: ANN202
            rng = self.headers.get("Range")
            path = self.translate_path(self.path)
            if not rng or not os.path.isfile(path):
                return super().send_head()
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            size = os.path.getsize(path)
            start = int(m[1]) if m and m[1] else 0
            end = min(int(m[2]) if m and m[2] else size - 1, size - 1)
            if start >= size:
                self.send_error(416, "Requested Range Not Satisfiable")
                return None
            f = open(path, "rb")  # noqa: SIM115 — closed by the base class after copyfile
            f.seek(start)
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            self._remaining = end - start + 1
            return f

        def copyfile(self, source, outputfile):  # noqa: ANN001, ANN202
            remaining = getattr(self, "_remaining", None)
            if remaining is None:
                return super().copyfile(source, outputfile)
            while remaining > 0:
                chunk = source.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
            self._remaining = None

        def log_message(self, *a: Any) -> None:
            pass

    return ThreadingHTTPServer((host, port), partial(RangeHandler, directory=str(directory)))


def serve(directory: str | Path, port: int = 8779, host: str = "127.0.0.1") -> None:
    srv = make_server(directory, port, host)
    print(f"serving {Path(directory).resolve()} on http://{host}:{port}", flush=True)
    srv.serve_forever()
