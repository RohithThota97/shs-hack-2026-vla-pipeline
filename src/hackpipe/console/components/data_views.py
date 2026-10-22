"""Data page building blocks (spec: docs/ui/01_data.md)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hackpipe.common import writelock
from hackpipe.common.config import project_config, resolve
from hackpipe.common.lerobot_io import DatasetFolder
from hackpipe.common.registry import Registry
from hackpipe.console.theme import TOKENS, badge
from hackpipe.dataset.versioning import approve, frozen_test_ids, set_excluded
from hackpipe.jobs import runner
from hackpipe.jobs.estimates import fmt_duration
from hackpipe.pipeline.preview import candidate_folders, preview
from hackpipe.quality.report import report_name


def ordered_sources(demos: list[dict]) -> list[str]:
    """Sources on this machine with real (non-duplicate) demos first, then the rest."""
    def rank(src: str) -> tuple:
        real = any(d["source"] == src and d["status"] != "DUPLICATE" for d in demos)
        return (not resolve(src).exists(), not real, src)
    return sorted({d["source"] for d in demos}, key=rank)


def writer_ok() -> tuple[bool, str]:
    w = writelock.holder()
    if w and w["machine"] != writelock.machine_id():
        return False, f"{w['machine']} is the registry writer — take over in the jobs panel first."
    return True, ""


# ---------- 1 · Add demonstrations ----------
def add_tab(reg: Registry) -> None:
    st.markdown("Pick a LeRobot v3 folder. **Run pipeline** imports it, checks only new or changed demos, "
                "recognises duplicates and creates the next draft dataset version for your approval.")
    folders = candidate_folders()
    choice = st.selectbox("Folder (data/incoming, data/raw)", folders + ["Other path…"], index=None,
                          placeholder="Choose a dataset folder")
    if choice == "Other path…":
        choice = st.text_input("Path to the dataset folder (contains meta/info.json)")
    if not choice:
        return
    try:
        p = preview(choice, reg)
    except FileNotFoundError as e:
        st.error(f"Not a LeRobot v3 dataset folder: {e}")
        return
    c = st.columns(4)
    c[0].metric("Episodes", p["episodes"])
    c[1].metric("New", p["new"])
    c[2].metric("Already registered", p["registered"])
    c[3].metric("Waiting for QC", p["pending_qc"])
    st.caption(f"{p['frames']:,} frames · {p['fps']:g} fps · cameras {', '.join(p['cameras']) or 'none'} · "
               f"robot {p['robot_type']} · LeRobot {p['codebase_version']} · task: {'; '.join(p['tasks'])}")
    plan = pd.DataFrame([
        {"step": "1 · import (hash files, find duplicates)", "runs on": "CPU", "work": f"{p['episodes']} episodes"},
        {"step": "2 · quality check (new / changed only)", "runs on": "CPU",
         "work": f"{p['work_units']:,} video frames to decode"},
        {"step": "3 · next draft dataset version", "runs on": "CPU", "work": "seconds"},
    ])
    st.dataframe(plan, hide_index=True, use_container_width=True)
    est = p["estimate"]
    st.markdown(f"**Expected time:** {fmt_duration(est['seconds'])} — {est['basis']}. Registered demos with "
                "unchanged files are skipped; files changed since import are re-checked.")
    ok, why = writer_ok()
    if st.button("Run pipeline", type="primary", disabled=not ok, help=why or None):
        jid = runner.submit("auto", runner.hackpipe_cmd("auto", "--root", p["source"]),
                            work_units=p["work_units"] or None, label=f"Pipeline · {Path(p['source']).name}",
                            registry=reg)
        st.success(f"Started {jid}. Follow it in the jobs panel; the result appears under Versions.")
    if not ok:
        st.caption(why)


# ---------- 2 · Demos ----------
def versions_by_demo(reg: Registry) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for d in reg.datasets():
        try:
            m = json.loads(Path(d["manifest_path"]).read_text())
        except (OSError, ValueError):
            continue
        for ids in m["demo_ids"].values():
            for i in ids:
                out.setdefault(i, []).append(d["version"])
    return out


def demos_frame(reg: Registry) -> pd.DataFrame:
    in_versions = versions_by_demo(reg)
    rows = []
    for d in reg.demos():
        m, env = d["meta"], d["meta"].get("environment") or {}
        qc = m.get("qc") or {}
        light = env.get("lighting") if isinstance(env.get("lighting"), dict) else {}
        rows.append({"demo": d["demo_id"][:8], "source": d["source"], "episode": d["episode_index"],
                     "status": "EXCLUDED" if m.get("excluded") else d["status"],
                     "reasons": ", ".join(qc.get("hard", []) + qc.get("soft", []))
                     or (f"duplicate of {m['duplicate_of'][:8]}" if m.get("duplicate_of") else "")
                     or ((m.get("excluded") or {}).get("reason") or ""),
                     "seed": env.get("seed"), "light K": round(light["color_temperature_K"])
                     if light.get("color_temperature_K") else None,
                     "outcome": (m.get("outcome") or {}).get("status"), "frames": m.get("n_frames"),
                     "versions": ", ".join(sorted(set(in_versions.get(d["demo_id"], [])))), "demo_id": d["demo_id"]})
    return pd.DataFrame(rows)


def demos_tab(reg: Registry) -> None:
    df = demos_frame(reg)
    if df.empty:
        st.info("No demos yet — add a folder in the first tab.")
        return
    f1, f2 = st.columns(2)
    srcs = f1.multiselect("Source", sorted(df["source"].unique()))
    stats = f2.multiselect("Status", sorted(df["status"].unique()))
    mask = pd.Series(True, index=df.index)
    if srcs:
        mask &= df["source"].isin(srcs)
    if stats:
        mask &= df["status"].isin(stats)
    view = df[mask]
    st.dataframe(view.drop(columns=["demo_id"]), hide_index=True, use_container_width=True, height=360)
    st.caption(f"{len(view)} of {len(df)} demos")
    with st.expander("Exclude / include a demo"):
        labels = {f"{r.source} · ep {r.episode} ({r.status})": r.demo_id for r in view.itertuples()}
        pick = st.selectbox("Demo", list(labels), index=None)
        reason = st.text_input("Reason (required to exclude)")
        by = st.text_input("Your name")
        ok, why = writer_ok()
        b1, b2 = st.columns(2)
        if b1.button("Exclude from future versions", disabled=not (ok and pick)):
            try:
                writelock.acquire()
                set_excluded(labels[pick], True, reason, by, reg)
                st.success("Excluded. The next version will list it under removed / excluded.")
            except ValueError as e:
                st.error(str(e))
        if b2.button("Include again", disabled=not (ok and pick)):
            writelock.acquire()
            set_excluded(labels[pick], False, registry=reg)
            st.success("Included again.")
        if why:
            st.caption(why)


# ---------- 3 · Episode viewer ----------
@st.cache_data(show_spinner=False)
def _frames(root: str, mtime: float) -> pd.DataFrame:
    return DatasetFolder(root).frames()


def joint_figure(ep: pd.DataFrame, idx: list[int], names: list[str], title: str) -> go.Figure:
    t = ep["timestamp"].to_numpy()
    s = np.stack(ep["observation.state"].to_numpy())
    a = np.stack(ep["action"].to_numpy())
    fig = go.Figure()
    for k, i in enumerate(idx):
        colour = f"hsl({(k * 47) % 360},55%,45%)"
        fig.add_trace(go.Scatter(x=t, y=s[:, i], name=f"{names[i]} state", line={"color": colour, "width": 1.5}))
        fig.add_trace(go.Scatter(x=t, y=a[:, i], name=f"{names[i]} action",
                                 line={"color": colour, "width": 1, "dash": "dot"}))
    fig.update_layout(title=title, height=300, margin={"l": 10, "r": 10, "t": 40, "b": 10},
                      xaxis_title="time (s)", legend={"font": {"size": 10}}, font={"family": "Inter"},
                      plot_bgcolor="#fff", paper_bgcolor="#fff", colorway=[TOKENS["accent"]])
    fig.update_xaxes(gridcolor=TOKENS["line"])
    fig.update_yaxes(gridcolor=TOKENS["line"])
    return fig


def viewer_tab(reg: Registry) -> None:
    demos = reg.demos()
    if not demos:
        st.info("No demos yet.")
        return
    srcs = ordered_sources(demos)
    c1, c2, c3 = st.columns([2, 1, 1])
    src = c1.selectbox("Source", srcs)
    eps = sorted(d["episode_index"] for d in demos if d["source"] == src)
    ep_i = c2.selectbox("Episode", eps)
    root = resolve(src)
    try:
        ds = DatasetFolder(root)
    except FileNotFoundError:
        st.warning(f"{src} is not on this machine.")
        return
    cams = ds.video_keys
    cam = c3.selectbox("Camera", cams, format_func=lambda k: k.rsplit(".", 1)[-1]) if cams else None
    meta = ds.episodes_meta()
    row = meta[meta["episode_index"] == ep_i].iloc[0] if len(meta) else None
    v, side = st.columns([3, 2])
    if cam and row is not None and f"videos/{cam}/chunk_index" in row:
        f = ds.episode_video_file(cam, row[f"videos/{cam}/chunk_index"], row[f"videos/{cam}/file_index"])
        t0, t1 = float(row[f"videos/{cam}/from_timestamp"]), float(row[f"videos/{cam}/to_timestamp"])
        if f.exists():
            v.video(str(f), start_time=int(t0), end_time=int(np.ceil(t1)))
            v.caption(f"{f.name} · {t0:.1f}–{t1:.1f} s of the file (AV1 needs Chrome/Edge/Firefox)")
    demo = next(d for d in demos if d["source"] == src and d["episode_index"] == ep_i)
    with side:
        st.markdown(f"**{src} · episode {ep_i}** {badge(demo['status'])}", unsafe_allow_html=True)
        m = demo["meta"]
        st.json({"demo_id": demo["demo_id"], "task": m.get("task"), "environment": m.get("environment"),
                 "outcome": m.get("outcome"), "qc": m.get("qc")}, expanded=1)
    df = _frames(str(root), (root / "data").stat().st_mtime)
    ep = df[df["episode_index"] == ep_i].sort_values("frame_index")
    dim = len(ep["observation.state"].iloc[0]) if len(ep) else 0
    names = ds.features.get("observation.state", {}).get("names")
    if not isinstance(names, list) or len(names) != dim:      # e.g. rand500 has no joint names
        names = [f"joint {i}" for i in range(dim)]
    if dim == 16:
        groups = [("Left arm", list(range(7))), ("Right arm", list(range(8, 15))), ("Grippers", [7, 15])]
    else:
        groups = [("Joints", list(range(dim)))]
    for title, idx in groups:
        st.plotly_chart(joint_figure(ep, idx, names, f"{title} — state (solid) vs action (dotted)"),
                        use_container_width=True)


# ---------- 4 · Quality ----------
def quality_tab(reg: Registry) -> None:
    srcs = ordered_sources(reg.demos())
    if not srcs:
        st.info("No QC reports yet.")
        return
    src = st.selectbox("Source ", srcs)
    path = resolve(Path(project_config()["project"]["outputs_dir"]) / "qc" / report_name(src) / "qc_report.json")
    if not path.exists():
        st.info("No QC report for this source yet.")
        return
    r = json.loads(path.read_text())
    c = st.columns(6)
    for col, k in zip(c, ("VALID", "REVIEW", "INVALID")):
        col.metric(k, r["status_counts"][k])
    c[3].metric("Duplicates", len(r.get("duplicates") or []))
    c[4].metric("Checked in last run", r.get("checked_now", r["n_episodes"]))
    c[5].metric("Reused results", r.get("reused", 0))
    st.caption(f"Generated {r['generated_at']} · {r['n_episodes']} episodes · {r['n_frames']:,} frames · "
               f"length {r['episode_frames']['min']}–{r['episode_frames']['max']} frames")
    red, vids = r.get("redundancy") or {}, r.get("videos") or {}
    if red:
        st.markdown(f"**Redundancy:** {red['distinct_trajectories']} distinct state/action trajectories in "
                    f"{r['n_episodes']} episodes (largest identical group: {red['largest_identical_group']}).")
    if vids.get("decode_check") == "done":
        st.markdown(f"**Videos (last run):** {vids['checked_files']} files decoded, {vids['decoded_frames']:,} "
                    f"frames, {len(vids['corrupted'])} corrupted.")
    for issue in r["dataset_issues"]:
        st.warning(issue)
    flagged = [e for e in r["episodes"] if e["status"] != "VALID"]
    if flagged:
        st.markdown("**Flagged episodes**")
        st.dataframe(pd.DataFrame([{"episode": e["episode_index"], "status": e["status"],
                                    "reasons": ", ".join(e["hard"] + e["soft"])} for e in flagged]),
                     hide_index=True, use_container_width=True)
    else:
        st.success("No flagged episodes.")


# ---------- 5 · Versions ----------
def versions_tab(reg: Registry) -> None:
    ds = sorted(reg.datasets(), key=lambda d: int(d["version"].split("-v", 1)[1]), reverse=True)
    if not ds:
        st.info("No dataset version yet — run the pipeline on a folder.")
        return
    frozen = frozen_test_ids(reg)
    for d in ds:
        m = json.loads(Path(d["manifest_path"]).read_text())
        lin = m.get("lineage") or {}
        head = f"{d['version']} · {d['status']} · {d['fingerprint']}" + (f" · based on {lin['base']}"
                                                                          if lin.get("base") else "")
        with st.expander(head, expanded=d is ds[0]):
            st.markdown(badge(d["status"]) + (f" approved by {d['meta'].get('approved_by')} at "
                                              f"{d['meta'].get('approved_at')}" if d["meta"].get("approved_by")
                                              else ""), unsafe_allow_html=True)
            c = st.columns(5)
            for col, k in zip(c, ("train", "val", "test")):
                col.metric(k, len(m["demo_ids"][k]))
            c[3].metric("Inherited", len(lin.get("inherited", [])) if lin.get("base") else "—")
            c[4].metric("Added", len(lin.get("added", [])) if lin.get("base") else "—")
            test_ok = not frozen or frozen[1] == sorted(m["demo_ids"]["test"])
            st.markdown(f"Test split {'unchanged (frozen) ✓' if test_ok else 'differs from the frozen test ✗'} · "
                        "leakage guard passed ✓ · " + (f"removed {len(lin['removed'])} demo(s)"
                                                       if lin.get("removed") else "nothing removed"))
            if m.get("needs_materialize"):
                st.info("Several sources: before training, run `hackpipe dataset materialize "
                        f"{d['version']}` on the GPU machine (merges them into one folder)."
                        + (" ✓ done" if m.get("materialized") else ""))
            card = Path(d["meta"].get("card", ""))
            with st.expander("Dataset card"):
                st.markdown(card.read_text() if card.exists() else "card missing")
            if d["status"] == "draft":
                ok, why = writer_ok()
                name = st.text_input("Approved by (your name)", key=f"by-{d['version']}")
                if st.button(f"Approve {d['version']}", type="primary", key=f"approve-{d['version']}",
                             disabled=not (ok and name.strip()), help=why or "A human decision; name is recorded"):
                    writelock.acquire()
                    approve(d["version"], reg, by=name.strip())
                    st.success(f"{d['version']} approved by {name.strip()}.")
                    st.rerun()
