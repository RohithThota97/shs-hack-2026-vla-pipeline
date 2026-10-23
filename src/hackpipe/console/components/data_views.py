"""Data page building blocks (spec: docs/ui/01_data.md)."""
from __future__ import annotations

import json
import os
import secrets
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hackpipe.common import writelock
from hackpipe.common.config import project_config, resolve, source_key
from hackpipe.common.lerobot_io import DatasetFolder
from hackpipe.common.registry import Registry
from hackpipe.console.theme import TOKENS, badge
from hackpipe.dataset.versioning import approve, frozen_test_ids, set_excluded
from hackpipe.jobs import runner
from hackpipe.jobs.estimates import fmt_duration
from hackpipe.pipeline.preview import candidate_folders, new_folders, preview
from hackpipe.pipeline.sources import KEY_DIR, dataset_name, missing_packages
from hackpipe.pipeline.watch import RAW, unlinked, watch_dirs
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
SOURCES = {"This machine (folder / zip, mounted Drive)": "local", "Upload a zip": "upload",
           "Hugging Face": "hf", "GitHub": "github", "Google Drive link": "gdrive",
           "Cloud bucket (S3 / GCS / Azure)": "cloud"}


def _secret(label: str, env_var: str, key: str) -> str:
    """Password field; empty means: use the variable from this machine's environment (if set)."""
    have = bool(os.environ.get(env_var))
    return st.text_input(label, type="password", key=key,
                         placeholder=f"using {env_var} from the environment" if have else "",
                         help=f"Only passed to the fetch job as {env_var}; not saved, not logged.")


def _private_file(data: bytes, name: str) -> str:
    """Owner-only file in KEY_DIR (created 0600, never world-readable); the fetch job deletes it after use."""
    KEY_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    f = KEY_DIR / name
    fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(data)
    return str(f)


@st.cache_data(ttl=20, show_spinner=False)
def _scan(folders: tuple[str, ...], n_demos: int, n_imported: int) -> tuple[list[dict], list[str]]:
    """new_folders() for the page, cached briefly: it previews every folder (keyed on what would change it)."""
    errors: list[str] = []
    return new_folders(Registry(), folders=list(folders), errors=errors), errors


def _drop_stale_uploads(d: Path, reg: Registry) -> None:
    """Uploads older than a day that no queued / running job uses (abandoned picks; a fetch removes its own)."""
    busy = {a for s in ("queued", "running") for j in reg.jobs(status=s) for a in j["meta"].get("argv", [])}
    for old in d.glob("*.zip"):
        try:
            if str(old) not in busy and time.time() - old.stat().st_mtime > 86400:
                old.unlink()
        except OSError:          # removed meanwhile by a finishing fetch
            pass


def _dir_stamp() -> tuple:
    """Changes when something is linked / fetched into data/raw or data/incoming (directory mtimes)."""
    out = []
    for d in ("data/raw", "data/incoming"):
        try:
            out.append(resolve(d).stat().st_mtime_ns)
        except OSError:
            out.append(None)
    return tuple(out)


@st.cache_data(ttl=20, show_spinner=False)
def _folders_cached(stamp: tuple) -> list[str]:
    return candidate_folders()


def _folders() -> list[str]:
    """candidate_folders() walks data/raw + data/incoming (maybe a Drive mount): cached, but refreshed as soon as
    a dataset is linked or fetched there."""
    return _folders_cached(_dir_stamp())


def _forget_scans() -> None:
    """After starting a job that adds datasets: rescan on the next render instead of showing them as waiting."""
    for f in (_folders_cached, _unlinked, _scan):
        f.clear()


@st.cache_data(ttl=20, show_spinner=False)
def _unlinked(folders: tuple[str, ...], dirs: tuple[str, ...]) -> list[dict]:
    """Datasets in the watch dirs not linked yet (re-scanned when data/raw, data/incoming or the dirs change)."""
    return unlinked(dirs=[Path(d) for d in dirs])


def scan_section(reg: Registry, folders: list[str], waiting: list[dict]) -> None:
    """Every dataset with demos the registry hasn't seen — in data/incoming, data/raw or a watch dir → one click."""
    demos = reg.demos()
    new, errors = _scan(tuple(folders), len(demos), sum(d["status"] == "imported" for d in demos))
    for e in errors:
        st.warning(f"Can't read {e}")
    if not new and not waiting:
        dirs = ", ".join(f"`{source_key(d)}`" for d in watch_dirs()) or "none on this machine"
        st.caption(f"No new datasets — everything in data/raw, data/incoming and the watch folders ({dirs}) "
                   "is registered.")
        return
    rows = [{"dataset": p["source"], "where": "data/raw · data/incoming", "episodes": p["episodes"],
             "new": p["new"], "waiting for QC": p["pending_qc"], "changed files": p["changed_files"]} for p in new]
    rows += [{"dataset": f"{RAW}/{d['name']}", "where": f"{'zip' if d['kind'] == 'zip' else 'folder'} · "
              f"{source_key(d['path'])} — will be {'unzipped and ' if d['kind'] == 'zip' else ''}linked",
              "episodes": d["episodes"], "new": d["episodes"], "waiting for QC": 0, "changed files": 0}
             for d in waiting]
    st.markdown(f"**{len(rows)} dataset(s) with new demonstrations found**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    units = sum(p["work_units"] for p in new)
    ok, why = writer_ok()
    if st.button("Run pipeline on all new datasets", type="primary", disabled=not ok, help=why or None):
        jid = runner.submit("auto", runner.hackpipe_cmd("auto", "--all"), work_units=units or None,
                            label=f"Pipeline · {len(rows)} dataset(s)", registry=reg)
        st.success(f"Started {jid}: link new ones as {RAW}/<name> → import → QC (new only) → one draft version.")
        _forget_scans()


def fetch_section(reg: Registry) -> None:
    """Bring a dataset in from elsewhere; a background job fetches it into data/incoming and runs the pipeline."""
    kind = SOURCES[st.radio("From", list(SOURCES), horizontal=True)]
    env: dict[str, str] = {}
    args: list[str] = []
    src = default_name = ""
    if kind == "local":
        src = st.text_input("Folder or .zip on this machine", placeholder="/content/drive/MyDrive/shs-hack-2026/data")
        copy = st.checkbox("Copy to local disk (faster than reading from a Drive mount)")
        if copy:
            args.append("--copy")
        st.caption("A folder containing several datasets is fine — each one with meta/info.json is found.")
    elif kind == "upload":
        up = st.file_uploader("Zip of a LeRobot v3 dataset (meta/, data/, videos/), up to 5 GB", type=["zip"],
                              help="The browser upload is held in memory (~1× its size). On Colab (~12 GB RAM) prefer the "
                                   "Drive link or 'This machine' for big zips.")
        if up:
            saved = st.session_state.setdefault("saved_uploads", {})
            if up.file_id not in saved:        # write each upload once, even after the job removed the zip
                d = resolve("data/incoming/_uploads")
                d.mkdir(parents=True, exist_ok=True)
                _drop_stale_uploads(d, reg)
                dest = d / f"{dataset_name(up.name)}-{up.file_id[:8]}.zip"
                with open(dest, "wb") as out:
                    shutil.copyfileobj(up, out, 16 << 20)
                saved[up.file_id] = (str(dest), dataset_name(up.name))
            src, default_name = saved[up.file_id]
        kind = "local"
    elif kind == "hf":
        src = st.text_input("Dataset repo ID", placeholder="your-user/fr3_duo_handover")
        c = st.columns(2)
        rev = c[0].text_input("Revision (optional)", placeholder="main")
        sub = c[1].text_input("Subfolder (optional)")
        args += [*(["--revision", rev] if rev else []), *(["--subdir", sub] if sub else [])]
        env["HF_TOKEN"] = _secret("Hugging Face token (private repos)", "HF_TOKEN", "k-hf")
    elif kind == "github":
        src = st.text_input("Repository", placeholder="owner/repo or https://github.com/owner/repo")
        rev = st.text_input("Branch / tag (optional)")
        args += ["--revision", rev] if rev else []
        env["GITHUB_TOKEN"] = _secret("GitHub token (private repos)", "GITHUB_TOKEN", "k-gh")
        st.caption("Large files in Git LFS are pulled when git-lfs is installed on this machine.")
    elif kind == "gdrive":
        src = st.text_input("Share link (file or folder, 'anyone with the link')",
                            placeholder="https://drive.google.com/drive/folders/…")
        st.caption("Works only for links shared as 'Anyone with the link'. For team-only data install Google Drive "
                   "for desktop and use 'This machine' with the folder's path (private files work then). A zip "
                   "is fastest; a folder link may stop at 50 files.")
    elif kind == "cloud":
        prov = st.selectbox("Provider", ["Amazon S3 / R2 / MinIO", "Google Cloud Storage", "Azure Blob"])
        if prov.startswith("Amazon"):
            src = st.text_input("URL", placeholder="s3://bucket/path/to/dataset (or …/dataset.zip)")
            c = st.columns(2)
            env["AWS_ACCESS_KEY_ID"] = c[0].text_input("Access key ID", key="k-s3id")
            env["AWS_SECRET_ACCESS_KEY"] = c[1].text_input("Secret access key", type="password", key="k-s3s")
            c = st.columns(2)
            env["AWS_DEFAULT_REGION"] = c[0].text_input("Region (optional)", key="k-s3r")
            env["AWS_ENDPOINT_URL"] = c[1].text_input("Endpoint URL (R2 / MinIO, optional)", key="k-s3e")
            st.caption("Empty fields use this machine's AWS environment / ~/.aws credentials.")
        elif prov.startswith("Google"):
            src = st.text_input("URL", placeholder="gs://bucket/path/to/dataset")
            sa = st.file_uploader("Service-account key (JSON, optional)", type=["json"], key="k-gcs",
                                  help="Stored owner-only in ~/.cache/hackpipe/keys until the fetch job used it.")
            if sa:
                env["GOOGLE_APPLICATION_CREDENTIALS"] = f"@{sa.file_id}"      # written at submit time
        else:
            src = st.text_input("URL", placeholder="az://container/path/to/dataset")
            env["AZURE_STORAGE_CONNECTION_STRING"] = st.text_input("Connection string", type="password",
                                                                   key="k-az")
    env = {k: v for k, v in env.items() if v}
    if not src:
        return
    auto = kind == "gdrive"
    name = st.text_input("Name under data/incoming", value="" if auto else (default_name or dataset_name(src)),
                         placeholder="empty = the folder / zip name on Drive" if auto else "",
                         help="Demo IDs depend on this path — use the same name on every machine.")
    missing = missing_packages(kind, src)
    if missing:
        st.error(f"Needs {', '.join(missing)} on this machine: `uv pip install {' '.join(missing)}` "
                 "(or `pip install -e '.[sources]'` / `'.[cloud]'`), then restart the console.")
    ok, why = writer_ok()
    if st.button("Fetch and run pipeline", type="primary", disabled=not (ok and (name.strip() or auto))
                 or bool(missing), help=why or None):
        if env.get("GOOGLE_APPLICATION_CREDENTIALS", "").startswith("@"):
            env["GOOGLE_APPLICATION_CREDENTIALS"] = _private_file(st.session_state["k-gcs"].getvalue(),
                                                                  f"gcs-{secrets.token_hex(6)}.json")
        argv = runner.hackpipe_cmd("fetch", "--from", kind, src, *(["--name", name.strip()] if name.strip() else []),
                                   *args, "--auto")
        jid = runner.submit("fetch", argv, label=f"Fetch · {name.strip() or dataset_name(src)}", registry=reg,
                            env=env)
        st.success(f"Started {jid}: fetch → import → QC (new only) → draft version. Follow it in the jobs panel.")
        _forget_scans()


def link_panel(reg: Registry, d: dict) -> None:
    """A dataset in a watch dir: what it is, and one button to link it as data/raw/<name> and run the pipeline."""
    c = st.columns(3)
    c[0].metric("Episodes", d["episodes"])
    c[1].metric("Frames", f"{d['frames']:,}" if d["frames"] else "—")
    c[2].metric("FPS", d["fps"])
    zipped = d["kind"] == "zip"
    st.caption(f"{'Zip' if zipped else 'Folder'}: {d['path']}" + (f" (inside: {d['inner']})" if d["inner"] else ""))
    name = st.text_input(f"Name under {RAW}", value=d["name"], key=f"link-{d['fingerprint']}",
                         help="Demo IDs depend on this path — the Colab notebook uses the same name.")
    st.markdown(f"**Run pipeline:** {'unzip to `data/cache/` → ' if zipped else ''}link as `{RAW}/{name}` → "
                "import → QC (new only) → next draft version.")
    ok, why = writer_ok()
    if st.button(f"Link as {RAW}/{name} and run pipeline", type="primary", disabled=not (ok and name.strip()),
                 help=why or None):
        argv = runner.hackpipe_cmd("auto", "--link", d["path"], "--name", name.strip())
        jid = runner.submit("auto", argv, label=f"Pipeline · {name.strip()}", registry=reg)
        _forget_scans()
        st.success(f"Started {jid}. Follow it in the jobs panel; the result appears under Versions.")


def add_tab(reg: Registry) -> None:
    st.markdown("New demonstrations are found by themselves: every LeRobot v3 dataset under `data/incoming`, "
                "`data/raw` or a watch folder (organiser download, Google Drive — `dataset.watch_dirs`) that has "
                "demos the registry hasn't seen is listed here. **Run pipeline** imports it, "
                "checks only new or changed demos, recognises duplicates and creates the next draft dataset "
                "version for your approval.")
    folders = _folders()
    waiting = _unlinked(tuple(folders), tuple(str(d) for d in watch_dirs()))
    scan_section(reg, folders, waiting)
    with st.expander("Bring in a dataset from elsewhere — this machine, upload, Hugging Face, GitHub, Drive, "
                     "cloud", expanded=not folders and not waiting):
        fetch_section(reg)
    st.markdown("**Check one dataset**")
    found = {f"{RAW}/{d['name']} — not linked yet ({d['kind']}: {source_key(d['path'])})": d for d in waiting}
    choice = st.selectbox("Dataset (data/raw, data/incoming, watch folders)", folders + list(found) + ["Other path…"],
                          index=None, placeholder="Choose a dataset")
    if choice == "Other path…":
        choice = st.text_input("Path to the dataset folder (contains meta/info.json)")
    if not choice:
        return
    if choice in found:
        link_panel(reg, found[choice])
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
            if m.get("mixed_schemas"):
                fps = sorted({f"{s['fps']:g} fps" for s in m["preprocessing"].get("per_source_schema", {}).values()})
                st.info(f"Sources differ in fps / schema ({', '.join(fps)}): kept together as data — this "
                        "version can't be merged or trained as one folder (training on it not decided yet).")
            elif m.get("needs_materialize"):
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
