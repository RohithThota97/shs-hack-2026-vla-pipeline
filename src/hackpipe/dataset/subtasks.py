"""Sub-task labels for curriculum training, from robot signals only (ported from the team's make_subtasks.py and
label_rand500_colors.py).

- An arm "moves" when its 7 joints change faster than `still_rad_s`; events are gripper close / open (start of the
  command ramp) and the first arm motion after an event. Each boundary sits in the middle of the still period
  before the event that starts the next sub-task.
- handover (5 sub-tasks): pick (left) | bring to handover point | grasp (right) | release (left) | place (right)
- cubes (4 sub-tasks):    pick A (left) | put A on B | pick C (right) | stack C on top. Cube colours: wrist camera
  at the moment each gripper closes (A, C); head camera first vs last frame for the base cube B.
- Checks per episode: boundaries in order and inside a still period, each sub-task holds its expected gripper
  event, gripper state at each boundary, duration outliers (> max(4 MAD, 30%) from the median). A flagged episode
  is kept in the labels but the curriculum trainer leaves it out.

`label(base)` labels the train/val sources of an approved version into outputs/labels/subtasks-<sha>/ (labels.json
+ one segments / episodes table per source + QA + review images). `hackpipe dataset create --base <base>
--subtasks <dir>` then makes the next draft version with the same demos and split plus these labels.
Nothing is copied or re-encoded: training reads the original folders.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hackpipe.common.config import config_dir, dataset_config, load_yaml, project_config, resolve
from hackpipe.common.ids import now_iso, sha256_file
from hackpipe.common.lerobot_io import DatasetFolder

LABEL_VERSION = 1
N_SUBTASKS = {"handover": 5, "cubes": 4}
# sub-task position → skill (curriculum stage it belongs to)
SKILLS = {"handover": ["reach_grasp", "transport_place", "bimanual", "bimanual", "transport_place"],
          "cubes": ["reach_grasp", "transport_place", "reach_grasp", "transport_place"]}
HANDOVER_FULL = "Hand the box from the left gripper to the right gripper and place it on the serving tray."
HANDOVER_SUBS = ["Pick up the box with the left gripper.", "Bring the box to the handover position.",
                 "Grasp the box with the right gripper.", "Release the box from the left gripper.",
                 "Place the box on the serving tray with the right gripper."]


def subtasks_config() -> dict[str, Any]:
    p = config_dir() / "subtasks.yaml"
    return load_yaml(p) if p.exists() else {"sources": {}}


def kind_of(source: str, cfg: dict[str, Any] | None = None) -> str | None:
    """Labelling rules for a source (by folder name), or None if it isn't labelled."""
    return ((cfg or subtasks_config()).get("sources") or {}).get(Path(source).name)


# ---------------------------------------------------------------- signals
class Joints:
    def __init__(self, cfg: dict[str, Any]):
        j = cfg.get("joints") or {}
        self.l_arm, self.r_arm = slice(*j.get("left_arm", [0, 7])), slice(*j.get("right_arm", [8, 15]))
        self.l_g, self.r_g = int(j.get("left_gripper", 7)), int(j.get("right_gripper", 15))
        self.still = float(cfg.get("still_rad_s", 0.02))


def moving(S: np.ndarray, arm: slice, fps: float, still: float) -> np.ndarray:
    """True where the arm moves; motion must last >= 0.1 s so settling jitter is ignored."""
    v = np.abs(np.diff(S[:, arm], axis=0)).sum(1) * fps
    m = np.r_[v > still, False]
    k = max(2, int(fps) // 10)
    sustained = np.array([m[t:t + k].all() for t in range(len(m))])
    out = sustained.copy()
    for t in np.flatnonzero(sustained):        # extend each sustained run over its own frames
        out[t:t + k] = True
    return out & m


def first(mask: np.ndarray, start: int = 0) -> int | None:
    idx = np.flatnonzero(mask[start:])
    return int(start + idx[0]) if len(idx) else None


def ramp_start(a: np.ndarray, t: int, direction: int) -> int:
    """Walk back from t to where the gripper command started changing (-1 closing, +1 opening)."""
    while t > 0 and (a[t] - a[t - 1]) * direction > 1e-6:
        t -= 1
    return t


def close_onset(A: np.ndarray, g: int) -> int | None:
    t = first(A[:, g] < 0.25)
    return None if t is None else ramp_start(A[:, g], t, -1)


def open_onset(A: np.ndarray, g: int, after: int) -> int | None:
    return first(np.r_[np.diff(A[:, g]) > 1e-4, False], after)


def grip_settled(S: np.ndarray, g: int, onset: int | None, fps: float) -> int | None:
    """First frame after a close/open command where the measured gripper has stopped moving (for 0.2 s)."""
    if onset is None:
        return None
    d = np.r_[np.abs(np.diff(S[:, g])), 0.0]
    k = max(2, int(fps) // 5)
    for t in range(onset + 1, len(S) - k):
        if (d[t:t + k] < 0.005).all() and abs(S[t, g] - S[onset, g]) > 0.05:
            return t
    return None


def still_mid_before(t: int, mv_l: np.ndarray, mv_r: np.ndarray) -> tuple[int, int]:
    """Middle of the still period (both arms still) that ends at event t, and its length."""
    s = t
    while s > 0 and not mv_l[s - 1] and not mv_r[s - 1]:
        s -= 1
    return (s + t) // 2, t - s


def segment(kind: str, S: np.ndarray, A: np.ndarray, fps: float, j: Joints
            ) -> tuple[list[int] | None, dict[str, int | None], list[int] | None]:
    """(boundaries [0, b1, …, len], gripper events, still frames before each boundary)."""
    mv_l, mv_r = moving(S, j.l_arm, fps, j.still), moving(S, j.r_arm, fps, j.still)
    lc, rc = close_onset(A, j.l_g), close_onset(A, j.r_g)
    if lc is None or rc is None:
        return None, {"left_close": lc, "right_close": rc}, None
    # a sub-task that ends with a grasp ends at the first motion after the gripper has *finished* closing
    lc_done, rc_done = grip_settled(S, j.l_g, lc, fps) or lc, grip_settled(S, j.r_g, rc, fps) or rc
    lo, ro = open_onset(A, j.l_g, lc + 1), open_onset(A, j.r_g, rc + 1)
    ev = {"left_close": lc, "right_close": rc, "left_open": lo, "right_open": ro}
    if kind == "handover":
        events = [first(mv_l, lc_done), first(mv_r, lc_done), lo, first(mv_r, rc_done)]
    else:
        lo_done = grip_settled(S, j.l_g, lo, fps) or lo
        events = [first(mv_l, lc_done), first(mv_r, lo_done) if lo_done is not None else None, first(mv_r, rc_done)]
    bounds, still = [], []
    for e in events:
        if e is None:
            return None, ev, None
        b, n = still_mid_before(e, mv_l, mv_r)
        bounds.append(b)
        still.append(n)
    return [0, *bounds, len(S)], ev, still


def check(kind: str, b: list[int] | None, ev: dict[str, int | None], still: list[int] | None,
          A: np.ndarray, j: Joints) -> list[str]:
    """Problems of one episode (empty = OK)."""
    if b is None:
        return [f"an event was not found ({', '.join(k for k, v in ev.items() if v is None) or 'arm motion'})"]
    if any(x >= y for x, y in zip(b, b[1:])):
        return [f"boundaries out of order {b}"]
    p = []
    if any(n < 2 for n in still or []):
        p.append(f"boundary not in a still period (still frames {still})")
    expect = {"handover": [("left_close", 0), ("right_close", 2), ("left_open", 3), ("right_open", 4)],
              "cubes": [("left_close", 0), ("left_open", 1), ("right_close", 2), ("right_open", 3)]}[kind]
    for e, k in expect:
        t = ev.get(e)
        if not (t is not None and b[k] <= t < b[k + 1]):
            p.append(f"{e} (frame {t}) not inside sub-task {k}")
    hold = lambda g, t: A[min(t, len(A) - 1), g] < 0.25  # noqa: E731 — gripper command: holding
    if not hold(j.l_g, b[1]):
        p.append("left gripper not holding at end of pick")
    if kind == "handover":
        if not (hold(j.l_g, b[3]) and hold(j.r_g, b[3])):
            p.append("both grippers should hold at the handover")
        if hold(j.l_g, b[4]) or not hold(j.r_g, b[4]):
            p.append("after release only the right gripper should hold")
    else:
        if hold(j.l_g, b[2]):
            p.append("left gripper should have released before the right arm starts")
        if not hold(j.r_g, b[3]):
            p.append("right gripper not holding at end of its pick")
    return p


def texts(kind: str, colors: dict[str, str] | None = None) -> tuple[str, list[str]]:
    """(complete instruction, sub-task instructions). Cube texts name the colours."""
    if kind == "handover":
        return HANDOVER_FULL, list(HANDOVER_SUBS)
    c = colors or {}
    a, bc, cc = c.get("cube_left", "unknown"), c.get("cube_base", "unknown"), c.get("cube_right", "unknown")
    base = f"the {bc} cube" if bc != "unknown" else "the other cube"
    return (f"Stack the {a} cube on {base}, then put the {cc} cube on top.",
            [f"Pick up the {a} cube with the left gripper.", f"Put the {a} cube on {base}.",
             f"Pick up the {cc} cube with the right gripper.", f"Stack the {cc} cube on top of the {a} cube."])


# ---------------------------------------------------------------- cube colours (label_rand500_colors.py)
def color_name(rgb: Any) -> str:
    import colorsys
    r, g, b = (np.asarray(rgb, float) / 255).tolist()
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360
    if s < 0.25:
        return "white" if v > 0.75 else "black" if v < 0.2 else "gray"
    if (h < 40 or h >= 345) and v < 0.5 and s < 0.55:
        return "brown"
    for lim, name in [(15, "red"), (40, "orange"), (70, "yellow"), (165, "green"), (200, "cyan"),
                      (255, "blue"), (290, "purple"), (345, "pink"), (361, "red")]:
        if h < lim:
            return name
    return "red"


def _hue(rgb: Any) -> float:
    import colorsys
    r, g, b = (np.asarray(rgb, float) / 255).tolist()
    return colorsys.rgb_to_hsv(r, g, b)[0] * 360


def _hue_dist(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


class Videos:
    """Decode single frames of one episode (LeRobot v3: many episodes per mp4, offset by from_timestamp)."""

    def __init__(self, ds: DatasetFolder):
        self.ds, self.fps = ds, ds.fps
        self.E = ds.episodes_meta().set_index("episode_index")

    def frames(self, ep: int, cam: str, idx: list[int]) -> list[np.ndarray]:
        import av
        k = f"videos/observation.images.{cam}"
        e = self.E.loc[ep]
        f = self.ds.episode_video_file(f"observation.images.{cam}", e[k + "/chunk_index"], e[k + "/file_index"])
        want = sorted(float(e[k + "/from_timestamp"]) + i / self.fps for i in idx)
        out: list[np.ndarray] = []
        with av.open(str(f)) as c:
            s = c.streams.video[0]
            c.seek(int(max(0.0, want[0] - 1) / s.time_base), stream=s)
            for fr in c.decode(s):
                if want and fr.time is not None and fr.time >= want[0] - 1e-3:
                    out.append(fr.to_ndarray(format="rgb24"))
                    want.pop(0)
                if not want:
                    break
        if len(out) != len(idx):
            raise ValueError(f"episode {ep} {cam}: decoded {len(out)} of {len(idx)} frames")
        return out


def _held_cube_rgb(img: np.ndarray) -> np.ndarray:
    """Median colour between the fingers (bottom-centre of the wrist image)."""
    return np.median(img[150:205, 95:130].reshape(-1, 3), 0)


def _base_cube_rgb(img: np.ndarray, img_end: np.ndarray, picked_hues: list[float]) -> tuple[np.ndarray | None, int]:
    """The base cube in the head camera: a small coloured blob in the same place in the first and last frame."""
    import colorsys

    from scipy.ndimage import median_filter
    crop = img[110:220, 30:210].astype(float)
    end = img_end[110:220, 30:210].astype(float)
    background = median_filter(crop, size=(21, 21, 1))      # cubes (~10 px) vanish, floor / shadows stay
    local = np.linalg.norm(crop - background, axis=2).reshape(-1)
    unchanged = (np.linalg.norm(crop - end, axis=2) < 30).reshape(-1)
    region = crop.reshape(-1, 3)
    hsv = np.array([colorsys.rgb_to_hsv(*(p / 255)) for p in region])
    bg = np.array([colorsys.rgb_to_hsv(*(p / 255)) for p in background.reshape(-1, 3)])
    dh = np.abs(hsv[:, 0] - bg[:, 0]) * 360
    dh = np.minimum(dh, 360 - dh)
    not_shadow = (dh > 25) | (hsv[:, 1] - bg[:, 1] > 0.3)
    cand = (local > 40) & unchanged & not_shadow & (hsv[:, 1] > 0.3) & (hsv[:, 2] > 0.12)
    if picked_hues:
        hd = np.array([min(_hue_dist(h * 360, ph) for ph in picked_hues) for h in hsv[:, 0]])
        cand &= hd > 25
    if cand.sum() < 8:
        return None, int(cand.sum())
    return np.median(region[cand], 0), int(cand.sum())


def cube_colors(videos: Videos, ep: int, A: np.ndarray, j: Joints) -> tuple[dict[str, str], list[str]]:
    """Colours of cube A (left), base B and C (right); problems if a gripper never closes."""
    cl, cr = first(A[:, j.l_g] < .5), first(A[:, j.r_g] < .5)
    if cl is None or cr is None:                 # argmax would silently read frame 1
        return {"cube_left": "unknown", "cube_base": "unknown", "cube_right": "unknown"}, \
               ["colour: a gripper never closes"]
    last = len(A) - 1
    lw = videos.frames(ep, "left_wrist_camera", [min(cl + 1, last)])[0]
    rw = videos.frames(ep, "right_wrist_camera", [min(cr + 1, last)])[0]
    head, head_end = videos.frames(ep, "external_camera", [0, last])
    a, c = _held_cube_rgb(lw), _held_cube_rgb(rw)
    b, _ = _base_cube_rgb(head, head_end, [_hue(a), _hue(c)])
    return {"cube_left": color_name(a), "cube_base": color_name(b) if b is not None else "unknown",
            "cube_right": color_name(c)}, []


# ---------------------------------------------------------------- one source
def label_source(source: str, kind: str, episodes: list[int], cfg: dict[str, Any] | None = None,
                 progress: Any = print) -> dict[str, Any]:
    """Segments (one row per sub-task) and episode records (instruction, colours, problems) for `episodes`."""
    cfg = cfg or subtasks_config()
    j = Joints(cfg)
    cols = dataset_config()["columns"]
    ds = DatasetFolder(resolve(source))
    fps = ds.fps
    df = ds.frames()[[cols["episode"], cols["frame"], cols["state"], cols["action"]]]
    df = df[df[cols["episode"]].isin(episodes)]
    videos = Videos(ds) if kind == "cubes" else None
    seg_rows, ep_rows = [], []
    for n, (ep, g) in enumerate(df.groupby(cols["episode"])):
        g = g.sort_values(cols["frame"])
        S, A = np.stack(g[cols["state"]]).astype(float), np.stack(g[cols["action"]]).astype(float)
        b, ev, still = segment(kind, S, A, fps, j)
        problems = check(kind, b, ev, still, A, j)
        colors: dict[str, str] = {}
        if videos is not None:
            colors, cp = cube_colors(videos, int(ep), A, j)
            problems += cp
        full, subs = texts(kind, colors)
        ep_rows.append({"episode_index": int(ep), "n_frames": len(g), "full_text": full,
                        "colors": json.dumps(colors), "events": json.dumps(ev), "problems": problems})
        if b is not None and not any(x >= y for x, y in zip(b, b[1:])):
            for k in range(N_SUBTASKS[kind]):
                seg_rows.append({"episode_index": int(ep), "subtask_order": k, "start_frame": int(b[k]),
                                 "end_frame": int(b[k + 1]), "duration_s": round((b[k + 1] - b[k]) / fps, 2),
                                 "text": subs[k], "skill": SKILLS[kind][k]})
        if progress and n % 50 == 0:
            progress(f"  {Path(source).name}: episode {ep} {'OK' if not problems else problems[0]}")
    seg = pd.DataFrame(seg_rows, columns=["episode_index", "subtask_order", "start_frame", "end_frame",
                                          "duration_s", "text", "skill"])
    eps = pd.DataFrame(ep_rows)
    # duration outliers per sub-task position (demos are uniform, so 4 MAD alone would flag ~1 s variations)
    for k, g in seg.groupby("subtask_order"):
        med = g.duration_s.median()
        tol = max(4 * ((g.duration_s - med).abs().median() or 0.1), 0.3 * med)
        for _, r in g[(g.duration_s - med).abs() > tol].iterrows():
            i = eps.index[eps.episode_index == r.episode_index][0]
            eps.at[i, "problems"] = [*eps.at[i, "problems"], f"sub-task {k} duration {r.duration_s}s vs median {med}s"]
    eps["flagged"] = eps.problems.map(bool)
    eps["problems"] = eps.problems.map(json.dumps)
    return {"source": source, "kind": kind, "fps": fps, "segments": seg, "episodes": eps}


def review_sheet(source: str, seg: pd.DataFrame, episodes: list[int], out_png: Path) -> None:
    """Head + wrist camera frames at every boundary of the given episodes, labelled with the next sub-task."""
    from PIL import Image, ImageDraw
    ds = DatasetFolder(resolve(source))
    v = Videos(ds)
    cams = ["external_camera", "left_wrist_camera", "right_wrist_camera"]
    rows, labels = [], []
    for ep in episodes:
        b = seg[seg.episode_index == ep].sort_values("subtask_order")
        frames = b.start_frame.tolist()[1:]
        tiles = [np.concatenate([v.frames(ep, c, [t])[0][::2, ::2] for c in cams], 0) for t in frames]
        rows.append(np.concatenate(tiles, 1))
        labels.append((ep, b.text.tolist()))
    img = Image.fromarray(np.concatenate(rows, 0))
    d = ImageDraw.Draw(img)
    h, w = rows[0].shape[0], rows[0].shape[1] // max(1, len(labels[0][1]) - 1)
    for r, (ep, txt) in enumerate(labels):
        for k in range(len(txt) - 1):
            d.rectangle([k * w, r * h, k * w + w - 1, r * h + 22], fill=(0, 0, 0))
            d.text((k * w + 2, r * h + 1), f"ep{ep} -> {k + 2}", fill=(255, 255, 0))
            d.text((k * w + 2, r * h + 11), txt[k + 1][:22], fill=(255, 255, 255))
    img.save(out_png)


# ---------------------------------------------------------------- a whole version
def code_sha() -> str:
    """Identifies the labelling rules: this file + configs/subtasks.yaml."""
    h = hashlib.sha256(Path(__file__).read_bytes())
    p = config_dir() / "subtasks.yaml"
    if p.exists():
        h.update(p.read_bytes())
    return h.hexdigest()


def label(base: str, registry: Any = None, review_images: bool | None = None, progress: Any = print) -> Path:
    """Label the train/val episodes of an approved version → outputs/labels/subtasks-<sha12>/ (returned)."""
    from hackpipe.common.registry import Registry
    from hackpipe.dataset.versioning import load_manifest
    reg = registry or Registry()
    m = load_manifest(base, reg)
    if m["status"] != "approved":
        raise RuntimeError(f"{base} is '{m['status']}': label an approved version")
    cfg = subtasks_config()
    eps: dict[str, list[int]] = {}
    for split in ("train", "val"):
        for s, i in m["episodes"][split]:
            eps.setdefault(s, []).append(int(i))
    unknown = [s for s in eps if kind_of(s, cfg) is None]
    if unknown:
        raise RuntimeError(f"No labelling rules for {unknown}: add them to configs/subtasks.yaml → sources")
    outputs = resolve(project_config()["project"]["outputs_dir"]) / "labels"
    tmp = outputs / f".tmp-{base}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    index: dict[str, Any] = {"label_version": LABEL_VERSION, "code_sha256": code_sha(), "base": base,
                             "created_at": now_iso(), "sources": {}}
    for s in sorted(eps):
        kind = kind_of(s, cfg)
        progress(f"labelling {s} ({kind}, {len(eps[s])} episodes)")
        r = label_source(s, kind, sorted(eps[s]), cfg, progress)
        name = Path(s).name
        r["segments"].to_csv(tmp / f"{name}_segments.csv", index=False)
        r["episodes"].to_csv(tmp / f"{name}_episodes.csv", index=False)
        flagged = sorted(int(e) for e in r["episodes"].loc[r["episodes"].flagged, "episode_index"])
        unlabelled = sorted(set(eps[s]) - set(r["segments"].episode_index))
        index["sources"][s] = {
            "kind": kind, "fps": r["fps"], "episodes": len(eps[s]), "flagged": flagged, "unlabelled": unlabelled,
            "segments": f"{name}_segments.csv", "episodes_file": f"{name}_episodes.csv",
            "sha256": hashlib.sha256((sha256_file(tmp / f"{name}_segments.csv") +
                                      sha256_file(tmp / f"{name}_episodes.csv")).encode()).hexdigest(),
            "subtask_seconds": {int(k): {"median": float(g.duration_s.median()), "min": float(g.duration_s.min()),
                                         "max": float(g.duration_s.max())}
                                for k, g in r["segments"].groupby("subtask_order")}}
        progress(f"  {name}: {len(eps[s]) - len(flagged)}/{len(eps[s])} clean, flagged {flagged}")
        if review_images if review_images is not None else cfg.get("review_images", True):
            ok = sorted(set(r["segments"].episode_index) - set(flagged))
            rng = np.random.default_rng(0)
            sample = sorted(rng.choice(ok, min(8 if kind == "cubes" else 3, len(ok)), replace=False).tolist()) if ok else []
            bad = [e for e in flagged if e in set(r["segments"].episode_index)][:8]
            for tag, which in (("sample", sample), ("flagged", bad)):
                if which:
                    try:
                        review_sheet(s, r["segments"], which, tmp / f"{name}_boundaries_{tag}.png")
                    except Exception as e:  # noqa: BLE001 — review images are a help, not a requirement
                        progress(f"  review images skipped: {type(e).__name__}: {e}")
    index["sha256"] = hashlib.sha256(json.dumps({s: v["sha256"] for s, v in index["sources"].items()},
                                                sort_keys=True).encode()).hexdigest()
    (tmp / "labels.json").write_text(json.dumps(index, indent=1))
    out = outputs / f"subtasks-{index['sha256'][:12]}"
    if out.exists():
        shutil.rmtree(tmp)                      # same labels already there
    else:
        tmp.rename(out)
    progress(f"labels: {out}")
    return out


def load_labels(directory: str | Path) -> dict[str, Any]:
    """labels.json + tables per source: {"index": …, source: {"segments": df, "episodes": df}}."""
    d = resolve(directory)
    index = json.loads((d / "labels.json").read_text())
    out: dict[str, Any] = {"index": index}
    for s, v in index["sources"].items():
        out[s] = {"segments": pd.read_csv(d / v["segments"]), "episodes": pd.read_csv(d / v["episodes_file"])}
    return out
