"""Train page (R4): mode (config from the base, or fine-tune the best model), what this dataset already trained,
start bar, then setup + dry run, timing run, step budget, live progress / resume, runs. Spec: docs/ui/02_train.md."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hackpipe.common import writelock
from hackpipe.common.config import REPO_ROOT, load_yaml, resolve
from hackpipe.common.registry import Registry
from hackpipe.console.components.shell import machine_info
from hackpipe.console.theme import TOKENS, badge
from hackpipe.jobs import runner
from hackpipe.jobs.estimates import fmt_duration
from hackpipe.training import progress
from hackpipe.dataset.versioning import load_manifest, trainable_sources
from hackpipe.training import curriculum as cur
from hackpipe.training import ranking
from hackpipe.training.train import apply_overrides, is_curriculum, parse_overrides, set_config_steps, train

reg = Registry()
st.markdown("### Train")
st.caption("03 Fine-Tune — SmolVLA on an approved dataset version (lerobot-train, or the curriculum trainer on "
           "sub-task labels). Every number here comes from a logged run.")

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


def config_rel(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return resolve(path).relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def run_record(mdl: dict) -> dict:
    p = Path(mdl.get("run_record_path") or "")
    return json.loads(p.read_text()) if p.is_file() else {}


AUTO = "Automatic — fine-tune the best model"
MANUAL = "Choose a config (start from its base policy)"


if not approved:
    st.info("Approve a dataset version first (Data → Versions).")
else:
    ranked = ranking.ranking(reg)
    best = ranking.best_model(reg, ranked)
    mode = st.radio("How to train", [MANUAL, AUTO], horizontal=True,
                    help="Automatic: the best model so far (sim success → offline test error → final loss) is the "
                         "starting checkpoint, with its own config and overrides; only the dataset is yours to pick.")
    c1, c2 = st.columns(2)
    ds = c1.selectbox("Dataset (approved only)", approved)
    manifest = load_manifest(ds, reg)
    blocked = ""                       # why training can't start with this dataset + config
    init_from = None
    best_rec: dict = {}
    if mode == AUTO and best is None:
        st.warning("No model can be fine-tuned yet: it needs a finished run (not a timing run) with a checkpoint on "
                   "this machine and some evidence (sim eval, offline eval or loss.csv). Train one from a config "
                   "first.")
        blocked = "no best model yet"
    elif mode == AUTO:
        bm = best["model"]
        init_from = bm["version"]
        best_rec = run_record(bm)
        leak = ranking.init_leakage(bm, manifest, reg)
        with st.container(border=True):
            st.markdown(f"**Starting point: {init_from}** — {best['label']} · trained on {bm['dataset_version']} "
                        f"with `{config_rel(bm['meta'].get('train_config_file'))}`"
                        + (f" (itself fine-tuned from {bm['meta']['init_from']})" if bm["meta"].get("init_from")
                           else ""))
            if best["tier"] == "loss":
                st.caption("Ranked by training loss only — no sim or offline eval beats hold-still yet. Run "
                           "`hackpipe eval --model … --offline` for a better choice.")
            if leak:
                blocked = f"{init_from} was trained on {len(leak)} test episodes of {ds}"
                st.error(f"{blocked} (e.g. {leak[:3]}): its scores on {ds}'s test split would not be honest. "
                         "Pick a dataset whose test split it never saw.")
    with st.expander("Model ranking (how the best model is chosen)", expanded=False):
        st.dataframe(pd.DataFrame([{"model": r["model"]["version"], "dataset": r["model"]["dataset_version"],
                                    "evidence": r["tier"] or "—", "score": r["label"],
                                    "can fine-tune": "yes" if r["eligible"] else r["why_not"]} for r in ranked]),
                     hide_index=True, use_container_width=True)
        st.caption("Order: sim success rate (higher better) → offline test arm error, only when it beats hold-still "
                   "(lower better) → final training loss (lower better; not comparable across datasets).")

    on_ds = [m for m in reg.models() if m["dataset_version"] == ds and m["meta"].get("purpose") != "timing"]
    trained_cfgs: dict[str, list[str]] = {}
    for m in on_ds:
        if m["status"] in ranking.FINISHED:
            trained_cfgs.setdefault(config_rel(m["meta"].get("train_config_file")) or "?", []).append(m["version"])
    if mode == AUTO and init_from:
        cfg_path = config_rel(bm["meta"]["train_config_file"])
        c2.selectbox("Train config (the starting model's)", [cfg_path], disabled=True)
    else:
        cfg_path = c2.selectbox("Train config", configs,
                                index=configs.index("configs/train/smolvla_fr3_duo.yaml")
                                if "configs/train/smolvla_fr3_duo.yaml" in configs else 0,
                                format_func=lambda c: c + (f"  · trained on {ds} → {', '.join(trained_cfgs[c])}"
                                                           if c in trained_cfgs else ""))
    if on_ds:
        with st.expander(f"Already trained on {ds}: {len(on_ds)} model(s)", expanded=True):
            sc = {r["model"]["version"]: r for r in ranked}
            st.dataframe(pd.DataFrame([{
                "model": m["version"], "config": config_rel(m["meta"].get("train_config_file")) or "external",
                "started from": m["meta"].get("init_from") or "base policy", "status": m["status"],
                "steps": ((run_record(m).get("train_config") or {}).get("args") or {}).get("steps"),
                "score": sc[m["version"]]["label"]} for m in on_ds]), hide_index=True, use_container_width=True)
        if mode == MANUAL and cfg_path in trained_cfgs:
            st.info(f"`{cfg_path}` already trained {', '.join(trained_cfgs[cfg_path])} on {ds}. Starting it again "
                    "makes a new model with the same data and config (only useful with changed overrides, e.g. "
                    "another seed). To compare, pick another config — or switch to **Automatic** to fine-tune the "
                    "best model.")
    cfg = load_yaml(cfg_path)
    best_ov = {k: v for k, v in (best_rec.get("overrides") or {}).items() if k != "policy.path"}
    plan_ov = {k: best_ov.pop(k) for k in ("curriculum", "no_curriculum") if k in best_ov}
    cfg = apply_overrides(cfg, plan_ov)       # the plan editor and baseline toggle start from the best model's plan
    args = cfg.get("args") or {}
    curriculum = is_curriculum(cfg)
    _train_src, _data_only = trainable_sources(manifest)
    if _data_only and not curriculum:
        st.caption(f"{ds} trains on {', '.join(f'`{x}`' for x in _train_src)}; kept as data only (other "
                   f"fps / schema, not trained): {', '.join(f'`{x}`' for x in _data_only)}. The curriculum "
                   "config (`smolvla_curriculum.yaml`) trains both at one rate.")
    prefill = "\n".join(f"{k}={json.dumps(v)}" for k, v in best_ov.items())    # JSON is YAML: values round-trip
    ov_text = st.text_area("Overrides (`key=value`, one per line — stored in the run record)"
                           + (f" — prefilled with {init_from}'s" if init_from else ""), value=prefill, height=68,
                           placeholder="batch_size=4", key=f"ov-{init_from or 'manual'}")
    try:
        overrides = parse_overrides([l.strip() for l in ov_text.splitlines() if l.strip()])
    except ValueError as e:
        st.error(str(e))
        overrides = {}

    # ---------- curriculum: labels + stage plan ----------
    plan_info = None
    if curriculum:
        if not manifest.get("subtasks"):
            blocked = f"{ds} has no sub-task labels"
            st.warning(f"The curriculum trainer needs sub-task labels, and **{ds}** has none. Label it: this "
                       f"creates a draft **next version** with the same demos and split plus the labels, which "
                       "you check (review images, flagged episodes) and approve in Data → Versions.")
            ok_w, why_w = writer_ok()
            if st.button(f"Label {ds} and create the draft", disabled=not ok_w, help=why_w or None):
                jid = runner.submit("subtasks", runner.hackpipe_cmd("subtasks", "label", "--base", ds, "--create"),
                                    needs="cpu", label=f"Sub-task labels · {ds}", registry=reg,
                                    extra_meta={"dataset": ds})
                st.success(f"Started {jid} (cube colours are read from the videos: a few minutes).")
        else:
            sub = manifest["subtasks"]
            st.caption(f"Sub-task labels `{sub['sha256'][:12]}` · flagged episodes left out: "
                       + "; ".join(f"`{s}` {v['flagged'] or 'none'}" for s, v in sorted(sub["sources"].items())))
        with st.expander("Curriculum plan — changes are stored as overrides in the run record", expanded=not blocked):
            base_plan = cfg.get("curriculum") or cur.DEFAULT_PLAN
            norm = {"new_skill_weight": float(base_plan.get("new_skill_weight", 0.7)),
                    "stages": [{"name": x["name"], "frac": float(x["frac"]), "skills": list(x["skills"])}
                               for x in base_plan["stages"]]}
            baseline = st.toggle("Baseline: all skills mixed from the start (no curriculum) — same data and steps, "
                                 "to show whether the curriculum helps", value=bool(args.get("no_curriculum")))
            nsw = st.slider("Share of a stage's samples from the skills it adds (the rest rehearses earlier skills)",
                            0.3, 1.0, norm["new_skill_weight"], 0.05, disabled=baseline)
            ed = st.data_editor(pd.DataFrame([{"stage": x["name"], "share of steps": x["frac"],
                                               "skills": ", ".join(x["skills"])} for x in norm["stages"]]),
                                num_rows="dynamic", hide_index=True, disabled=baseline, key=f"plan-{cfg_path}",
                                use_container_width=True)
            st.caption(f"Skills: {', '.join(f'`{k}`' for k in cur.SKILLS)}. Shares are normalised to 1.")
            edited = {"new_skill_weight": float(nsw),
                      "stages": [{"name": str(r["stage"]), "frac": float(r["share of steps"] or 0),
                                  "skills": [k.strip() for k in str(r["skills"] or "").split(",") if k.strip()]}
                                 for r in ed.to_dict("records") if str(r.get("stage") or "").strip()]}
            if not baseline and edited != norm:
                overrides["curriculum"] = edited
            if baseline != bool(args.get("no_curriculum")):
                overrides["no_curriculum"] = baseline
            for k, v in plan_ov.items():          # unchanged in the editor: still the best model's plan
                overrides.setdefault(k, v)
            if not blocked:
                try:
                    plan_info = cur.describe(apply_overrides(cfg, overrides), manifest)
                except (RuntimeError, ValueError, KeyError) as e:
                    blocked = str(e)
                    st.error(blocked)
            if plan_info:
                st.dataframe(pd.DataFrame([
                    {"stage": x["name"], "steps": n,
                     "sample mix": " · ".join(f"{g.split(':')[0].replace('fr3_duo_', '')} {g.split(':')[1]} {v:.0%}"
                                              for g, v in mix.items())}
                    for x, n, mix in zip(plan_info["plan"]["stages"], plan_info["stage_steps"],
                                         plan_info["sample_mix"])]), hide_index=True, use_container_width=True)
                st.caption(f"{plan_info['rate_hz']} Hz, {plan_info['chunk']} actions per prediction · train episodes "
                           + ", ".join(f"`{s}` {n}" for s, n in plan_info["train_episodes"].items()))
    ov_list = [f"{k}={json.dumps(v)}" for k, v in overrides.items()]
    batch = int(overrides.get("batch_size", args.get("batch_size", 8)))
    save_freq = int(overrides.get("save_freq", args.get("save_freq", 1000)))
    log_freq = int(overrides.get("log_freq", args.get("log_freq", 50)))
    rate = float(overrides.get("rate_hz", args.get("rate_hz", cur.DEFAULTS["rate_hz"])))
    frames = (progress.curriculum_frames(ds, reg, rate, overrides.get("train_sources", args.get("train_sources")))
              if curriculum and manifest.get("subtasks")
              else progress.train_frames(ds, reg))
    init_args = ["--init-from", init_from] if init_from else []

    # ---------- start bar ----------
    target = int(overrides.get("steps", args.get("steps", 0)))
    done_tj = next((j for j in reg.jobs(kind="train_timing", limit=20) if j["status"] == "done"), None)
    done_t = t = progress.timing_summary(done_tj, done_tj["meta"].get("log_freq", log_freq)) if done_tj else None
    est = ({"seconds": progress.expected_seconds(target, t["seconds_per_step"], t["startup_s"]),
            "basis": f"timing run {t['job_id']}: {t['seconds_per_step']} s/step × {target} steps + start-up"}
           if t and t["seconds_per_step"] else None)
    ok, why = writer_ok()
    with st.container(border=True):
        a, b = st.columns([3, 1])
        a.markdown(f"**{target:,} steps** on **{ds}** with `{cfg_path}`, starting from "
                   + (f"**{init_from}** (fine-tune)" if init_from else f"`{args.get('policy.path', '?')}`")
                   + f" · expected **{fmt_duration(est['seconds']) if est else 'n/a'}**"
                   + (f" ({est['basis']})" if est else " — run the timing run first for an estimate"))
        if b.button("Start training", type="primary", use_container_width=True,
                    disabled=not (gpus and ok and target > 0) or bool(blocked),
                    help=why or blocked or (None if gpus else "needs a GPU — open the console on Colab")):
            jid = runner.submit("train", runner.hackpipe_cmd("train", "--config", cfg_path, "--dataset", ds,
                                                             *init_args, *set_args(ov_list)),
                                needs="gpu", work_units=target,
                                label=f"Training · {ds}" + (f" from {init_from}" if init_from else ""), registry=reg,
                                estimate_override=est, extra_meta={"target_steps": target, "dataset": ds,
                                                                   "config": cfg_path, "log_freq": log_freq,
                                                                   "init_from": init_from})
            st.success(f"Started {jid}. Checkpoints every {save_freq} steps — progress under **Progress & resume**.")
        if blocked:
            a.caption(f"Can't start: {blocked}.")

    tabs = st.tabs(["Setup & dry run", "Timing run", "Step budget", "Progress & resume", "Runs"])

    # ---------- Setup & dry run ----------
    with tabs[0]:
        m = st.columns(4)
        m[0].metric(f"Train samples ({rate:g} Hz)" if curriculum else "Train frames", f"{frames:,}")
        m[1].metric("Batch size", batch)
        m[2].metric("Steps (config)", int(overrides.get("steps", args.get("steps", 0))))
        m[3].metric("Checkpoint every", save_freq)
        st.caption(f"Policy {f'{init_from} (fine-tune)' if init_from else args.get('policy.path', args.get('policy.type', '?'))} · seed {cfg.get('seed')} · "
                   f"{args.get('num_workers', '?')} data workers")
        if st.button("Dry run — show the exact training command"):
            try:
                rec = train(cfg_path, ds, dry_run=True, registry=reg, overrides=overrides, init_from=init_from)
                st.code(" \\\n  ".join(rec["command"]), language="bash")
                if curriculum:
                    st.caption("Check: dataset version with sub-task labels, stage plan and steps (above), "
                               f"{rec['curriculum']['rate_hz']} Hz, batch size, steps, save frequency, seed, output "
                               "dir. The resolved config is saved next to the dry-run record.")
                else:
                    st.caption("Check: dataset root and episodes (train split only), rename map → camera1..3, "
                               "policy path, batch size, steps, save frequency, seed, output dir, logging.")
            except Exception as e:  # noqa: BLE001 — show any configuration error to the user
                st.error(f"{type(e).__name__}: {e}")

    # ---------- Timing run ----------
    with tabs[1]:
        st.markdown("A short GPU run (200 steps) to **measure** seconds per step, GPU memory and data-loading "
                    "share. Its model ID is labelled *timing run, not a result*.")
        tj = latest_job("train_timing")
        ok, why = writer_ok()
        if st.button("Start timing run (200 steps)", type="primary", disabled=not (gpus and ok) or bool(blocked),
                     help=why or blocked or (None if gpus else "needs a GPU")):
            jid = runner.submit("train_timing", runner.hackpipe_cmd(
                "train", "--config", cfg_path, "--dataset", ds, "--purpose", "timing", *init_args,
                *set_args(ov_list + ["steps=200", "save_freq=200"]
                          + (["stage_end_eval=false", "val_every=1000000"] if curriculum else []))),
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
        tj, t = done_tj, done_t
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

    # ---------- Progress & resume ----------
    with tabs[3]:
        running = [j for j in reg.jobs(kind="train", limit=10) if j["status"] in ("running", "queued")]
        if not running:
            st.caption("No training running. Start one with **Start training** above.")
        for j in running:
            lv = progress.live(j, j["meta"].get("target_steps"), j["meta"].get("log_freq"))
            with st.container(border=True):
                st.markdown(f"**{j['job_id']}** {badge(j['status'])} · {j['meta'].get('dataset')}"
                            + (f" · from {j['meta']['init_from']}" if j["meta"].get("init_from") else ""),
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
                             else "model", "dataset": mdl["dataset_version"],
                             "started from": mdl["meta"].get("init_from") or "base policy", "status": mdl["status"],
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
            clog = Path(next(m for m in models if m["version"] == pick)["meta"].get("output_dir") or ".") \
                / "curriculum_log.json"
            if clog.is_file():
                val = json.loads(clog.read_text()).get("val") or []
                keys = sorted({k for v in val for k in v if "/" in k})
                if val:
                    fig = go.Figure([go.Scatter(x=[v["step"] for v in val if k in v],
                                                y=[v[k]["l1_norm"] for v in val if k in v], name=k, mode="lines+markers")
                                     for k in keys])
                    for v in val:
                        if v.get("end_of_stage"):
                            fig.add_vline(x=v["step"], line_dash="dot", line_color=TOKENS["muted"],
                                          annotation_text=v["stage"].split("_", 1)[0])
                    fig.update_layout(height=320, margin={"l": 10, "r": 10, "t": 30, "b": 10}, xaxis_title="step",
                                      yaxis_title="validation L1 (normalised)", plot_bgcolor="#fff",
                                      font={"family": "Inter"}, legend={"orientation": "h"})
                    st.markdown("**Validation per dataset × instruction** (sub-task / complete task; dotted: end "
                                "of a stage)")
                    st.plotly_chart(fig, use_container_width=True)
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
