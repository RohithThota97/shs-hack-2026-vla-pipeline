"""Train page (R4): setup + dry run, timing run, step budget, full run / resume, runs. Spec: docs/ui/02_train.md."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hackpipe.common import writelock
from hackpipe.common.config import REPO_ROOT, load_yaml
from hackpipe.common.registry import Registry
from hackpipe.console.components.shell import machine_info
from hackpipe.console.theme import TOKENS, badge
from hackpipe.jobs import runner
from hackpipe.jobs.estimates import fmt_duration
from hackpipe.training import progress
from hackpipe.dataset.versioning import load_manifest, trainable_sources
from hackpipe.training.train import parse_overrides, set_config_steps, train

reg = Registry()
st.markdown("### Train")
st.caption("03 Fine-Tune — SmolVLA on an approved dataset version. Every number here comes from a logged run.")

gpus = machine_info()["gpus"]
if not gpus:
    st.warning("This machine has no GPU. Dry runs work here; the timing run and training need the console on "
               "Colab (notebook: “Console” step).")

approved = sorted((d["version"] for d in reg.datasets() if d["status"] == "approved"),
                  key=lambda v: int(v.split("-v", 1)[1]), reverse=True)
configs = sorted(p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "configs" / "train").glob("*.yaml"))


def writer_ok() -> tuple[bool, str]:
    w = writelock.holder()
    if w and w["machine"] != writelock.machine_id():
        return False, f"{w['machine']} is the registry writer — take over in the jobs panel first."
    return True, ""


def latest_job(kind: str) -> dict | None:
    jobs = reg.jobs(kind=kind, limit=20)
    return jobs[0] if jobs else None


def set_args(pairs: list[str]) -> list[str]:
    return [a for kv in pairs for a in ("--set", kv)]


if not approved:
    st.info("Approve a dataset version first (Data → Versions).")
else:
    c1, c2 = st.columns(2)
    ds = c1.selectbox("Dataset (approved only)", approved)
    _train_src, _data_only = trainable_sources(load_manifest(ds, reg))
    if _data_only:
        st.caption(f"{ds} trains on {', '.join(f'`{x}`' for x in _train_src)}; kept as data only (other "
                   f"fps / schema, not trained): {', '.join(f'`{x}`' for x in _data_only)}.")
    cfg_path = c2.selectbox("Train config", configs,
                            index=configs.index("configs/train/smolvla_fr3_duo.yaml")
                            if "configs/train/smolvla_fr3_duo.yaml" in configs else 0)
    cfg = load_yaml(cfg_path)
    args = cfg.get("args") or {}
    ov_text = st.text_area("Overrides (`key=value`, one per line — stored in the run record)", height=68,
                           placeholder="batch_size=4")
    try:
        overrides = parse_overrides([l.strip() for l in ov_text.splitlines() if l.strip()])
    except ValueError as e:
        st.error(str(e))
        overrides = {}
    ov_list = [f"{k}={v}" for k, v in overrides.items()]
    batch = int(overrides.get("batch_size", args.get("batch_size", 8)))
    save_freq = int(overrides.get("save_freq", args.get("save_freq", 1000)))
    log_freq = int(overrides.get("log_freq", args.get("log_freq", 50)))
    frames = progress.train_frames(ds, reg)

    tabs = st.tabs(["Setup & dry run", "Timing run", "Step budget", "Full run", "Runs"])

    # ---------- Setup & dry run ----------
    with tabs[0]:
        m = st.columns(4)
        m[0].metric("Train frames", f"{frames:,}")
        m[1].metric("Batch size", batch)
        m[2].metric("Steps (config)", int(overrides.get("steps", args.get("steps", 0))))
        m[3].metric("Checkpoint every", save_freq)
        st.caption(f"Policy {args.get('policy.path', args.get('policy.type', '?'))} · seed {cfg.get('seed')} · "
                   f"{args.get('num_workers', '?')} data workers")
        if st.button("Dry run — show the exact lerobot-train command"):
            try:
                rec = train(cfg_path, ds, dry_run=True, registry=reg, overrides=overrides)
                st.code(" \\\n  ".join(rec["command"]), language="bash")
                st.caption("Check: dataset root and episodes (train split only), rename map → camera1..3, policy "
                           "path, batch size, steps, save frequency, seed, output dir, logging.")
            except Exception as e:  # noqa: BLE001 — show any configuration error to the user
                st.error(f"{type(e).__name__}: {e}")

    # ---------- Timing run ----------
    with tabs[1]:
        st.markdown("A short GPU run (200 steps) to **measure** seconds per step, GPU memory and data-loading "
                    "share. Its model ID is labelled *timing run, not a result*.")
        tj = latest_job("train_timing")
        ok, why = writer_ok()
        if st.button("Start timing run (200 steps)", type="primary", disabled=not (gpus and ok),
                     help=why or (None if gpus else "needs a GPU")):
            jid = runner.submit("train_timing", runner.hackpipe_cmd(
                "train", "--config", cfg_path, "--dataset", ds, "--purpose", "timing",
                *set_args(ov_list + ["steps=200", "save_freq=200"])),
                needs="gpu", work_units=200, label=f"Timing run · {ds}", registry=reg,
                extra_meta={"target_steps": 200, "dataset": ds, "config": cfg_path, "log_freq": log_freq})
            st.success(f"Started {jid}.")
            tj = reg.get_job(jid)
        if tj:
            t = progress.timing_summary(tj, tj["meta"].get("log_freq", log_freq))
            st.markdown(f"Latest: **{tj['job_id']}** {badge(tj['status'])}", unsafe_allow_html=True)
            r = st.columns(4)
            r[0].metric("Seconds / step", t["seconds_per_step"] if t["seconds_per_step"] else "n/a")
            r[1].metric("Peak GPU memory", f"{t['peak_mem_gb']:.1f} GB" if t["peak_mem_gb"] else "n/a")
            r[2].metric("Data-loading share", f"{t['data_share']:.0%}" if t["data_share"] is not None else "n/a")
            r[3].metric("Start-up", fmt_duration(t["startup_s"]))
            if t["data_share"] is not None and t["data_share"] > 0.3:
                st.warning("Data loading takes a large share of each step: raise `num_workers` or try "
                           "`dataset.video_backend=pyav`, then time again.")
            if t["peak_mem_gb"] and gpus and t["peak_mem_gb"] > 0.9 * gpus[0]["memory_total_mib"] / 1024:
                st.warning("Close to the GPU's memory limit: use `batch_size=4` for the full run.")
            if tj["status"] == "failed":
                st.error("The timing run failed — open its log in the jobs panel.")
        else:
            st.info("No timing run yet.")

    # ---------- Step budget ----------
    with tabs[2]:
        tj = next((j for j in reg.jobs(kind="train_timing", limit=20) if j["status"] == "done"), None)
        t = progress.timing_summary(tj, tj["meta"].get("log_freq", log_freq)) if tj else None
        if not t or not t["seconds_per_step"]:
            st.info("Needs a finished timing run (seconds per step are measured, never assumed).")
        else:
            hours = st.number_input("Training slot (hours)", 0.25, 48.0, 3.5, 0.25,
                                    help="Free Colab sessions often end after a few hours; checkpoints allow resume.")
            steps = progress.budget(hours, t["seconds_per_step"], save_freq, t["startup_s"] or 0)
            b = st.columns(3)
            b[0].metric("Steps", f"{steps:,}")
            b[1].metric("Epochs", progress.epochs(steps, batch, frames) or "n/a")
            b[2].metric("Expected time", fmt_duration(progress.expected_seconds(steps, t["seconds_per_step"],
                                                                                t["startup_s"])))
            st.caption(f"From {t['job_id']}: {t['seconds_per_step']} s/step, start-up {fmt_duration(t['startup_s'])}; "
                       f"rounded down to whole checkpoints ({save_freq} steps).")
            if steps <= 0:
                fit = int((hours * 3600 - (t["startup_s"] or 0)) / t["seconds_per_step"])
                st.warning(f"Only about {max(fit, 0):,} steps fit — less than one checkpoint ({save_freq} steps). "
                           "Use a longer slot, a faster GPU, or a smaller `save_freq` override.")
            if st.button(f"Write steps = {steps} into {cfg_path}", disabled=steps <= 0):
                set_config_steps(cfg_path, steps)
                st.success("Written. Commit the config and note the reason in docs/decisions.md — the same steps "
                           "are used for model_v2.")

    # ---------- Full run ----------
    with tabs[3]:
        target = int(overrides.get("steps", args.get("steps", 0)))
        tj = next((j for j in reg.jobs(kind="train_timing", limit=20) if j["status"] == "done"), None)
        t = progress.timing_summary(tj, tj["meta"].get("log_freq", log_freq)) if tj else None
        est = ({"seconds": progress.expected_seconds(target, t["seconds_per_step"], t["startup_s"]),
                "basis": f"timing run {t['job_id']}: {t['seconds_per_step']} s/step × {target} steps + start-up"}
               if t and t["seconds_per_step"] else None)
        st.markdown(f"**{target:,} steps** on {ds} · expected **{fmt_duration(est['seconds']) if est else 'n/a'}**"
                    + (f" ({est['basis']})" if est else " — run the timing run first"))
        ok, why = writer_ok()
        if st.button("Start training", type="primary", disabled=not (gpus and ok and target > 0),
                     help=why or (None if gpus else "needs a GPU")):
            jid = runner.submit("train", runner.hackpipe_cmd("train", "--config", cfg_path, "--dataset", ds,
                                                             *set_args(ov_list)),
                                needs="gpu", work_units=target, label=f"Training · {ds}", registry=reg,
                                estimate_override=est, extra_meta={"target_steps": target, "dataset": ds,
                                                                   "config": cfg_path, "log_freq": log_freq})
            st.success(f"Started {jid}. Checkpoints every {save_freq} steps.")
        for j in [j for j in reg.jobs(kind="train", limit=10) if j["status"] in ("running", "queued")]:
            lv = progress.live(j, j["meta"].get("target_steps"), j["meta"].get("log_freq"))
            with st.container(border=True):
                st.markdown(f"**{j['job_id']}** {badge(j['status'])} · {j['meta'].get('dataset')}",
                            unsafe_allow_html=True)
                if lv["step"] is not None and j["meta"].get("target_steps"):
                    st.progress(min(lv["step"] / j["meta"]["target_steps"], 1.0),
                                text=f"step {lv['step']:,} / {j['meta']['target_steps']:,} (from the log)")
                    st.caption(f"loss {lv['loss']} · {lv['seconds_per_step']} s/step · "
                               f"GPU {lv['mem_gb'] or 'n/a'} GB · time left {fmt_duration(lv['eta_s'])}")
                else:
                    st.caption("starting — no training log line yet")
        resumable = [m for m in reg.models() if m["status"] in ("training", "failed")
                     and (Path(m["meta"].get("output_dir", "")) / "checkpoints" / "last").exists()]
        if resumable:
            st.markdown("**Resume** (from the last checkpoint; no overrides — LeRobot reuses the saved config)")
            pick = st.selectbox("Model", [m["version"] for m in resumable])
            if st.button("Resume", disabled=not (gpus and ok)):
                mdl = next(m for m in resumable if m["version"] == pick)
                jid = runner.submit("train", runner.hackpipe_cmd("train", "--dataset", mdl["dataset_version"],
                                                                 "--resume", pick),
                                    needs="gpu", label=f"Resume · {pick}", registry=reg,
                                    extra_meta={"dataset": mdl["dataset_version"], "resume": pick})
                st.success(f"Started {jid}.")

    # ---------- Runs ----------
    with tabs[4]:
        models = reg.models()
        if not models:
            st.info("No models yet.")
        else:
            rows = []
            for mdl in models:
                rec = {}
                if mdl.get("run_record_path") and Path(mdl["run_record_path"]).exists():
                    rec = json.loads(Path(mdl["run_record_path"]).read_text())
                loss_csv = Path(mdl["run_record_path"] or ".").parent / "loss.csv"
                final = loss_csv.read_text().splitlines()[-1].split(",")[-1] if loss_csv.exists() else None
                rows.append({"model": mdl["version"],
                             "purpose": "timing run, not a result" if mdl["meta"].get("purpose") == "timing"
                             else "model", "dataset": mdl["dataset_version"], "status": mdl["status"],
                             "steps": ((rec.get("train_config") or {}).get("args") or {}).get("steps"),
                             "wall time": fmt_duration(rec.get("duration_s")), "final loss": final or "n/a"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            pick = st.selectbox("Loss curve of", [m["version"] for m in models], index=len(models) - 1)
            csv = Path(next(m for m in models if m["version"] == pick)["run_record_path"] or ".").parent / "loss.csv"
            if csv.exists():
                df = pd.read_csv(csv)
                fig = go.Figure(go.Scatter(x=df["step"], y=df["loss"], line={"color": TOKENS["accent"]}))
                fig.update_layout(height=300, margin={"l": 10, "r": 10, "t": 10, "b": 10}, xaxis_title="step",
                                  yaxis_title="training loss", plot_bgcolor="#fff", font={"family": "Inter"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No loss.csv yet: `python scripts/plot_loss.py --log <log> --model "
                           f"{pick}` after training.")
            if len(models) >= 2:
                a, b = st.columns(2)
                ma = a.selectbox("Compare run record", [m["version"] for m in models], index=0)
                mb = b.selectbox("with", [m["version"] for m in models], index=len(models) - 1)
                ra, rb = (json.loads(Path(reg.get_model(v)["run_record_path"]).read_text())
                          if reg.get_model(v)["run_record_path"] and Path(reg.get_model(v)["run_record_path"]).exists()
                          else {} for v in (ma, mb))
                diff = progress.run_record_diff(ra, rb)
                if diff:
                    st.dataframe(pd.DataFrame(diff), hide_index=True, use_container_width=True)
                    if all(d["dataset_field"] for d in diff):
                        st.success("Only dataset fields differ — the change can be attributed to the data.")
                else:
                    st.caption("No differences.")
