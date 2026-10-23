"""Evaluate page (R5): offline eval on the held-out test source, Isaac Sim imports, gates → verdict, comparison.
Spec: docs/ui/03_evaluate.md. Metrics and gates after the team's post-training pipeline (AI-project-work)."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hackpipe.common import writelock
from hackpipe.common.config import REPO_ROOT, gates_config
from hackpipe.common.registry import Registry
from hackpipe.console.components.shell import machine_info
from hackpipe.console.theme import TOKENS, badge
from hackpipe.evaluation.offline import is_running, pending
from hackpipe.evaluation.runner import load_results
from hackpipe.gates.verdict import offline_result, sim_result, verdict
from hackpipe.jobs import runner

reg = Registry()
st.markdown("### Evaluate")
st.caption("04 Evaluate — offline on the held-out test source (lighting10, never trained on), closed loop from "
           "Isaac Sim runs, then the gates in `configs/gates.yaml` decide: DEPLOY · MORE TRAINING · HOLD.")
gpus = machine_info()["gpus"]
from hackpipe.training.ranking import FINISHED  # noqa: E402 — same "finished" as the ranking and pending()
models = [m for m in reg.models() if m["meta"].get("purpose") != "timing" and m["status"] in FINISHED]
dataset_of = {m["version"]: m["dataset_version"] for m in models}


def writer_ok() -> tuple[bool, str]:
    w = writelock.holder()
    if w and w["machine"] != writelock.machine_id():
        return False, f"{w['machine']} is the registry writer — take over in the jobs panel first."
    return True, ""


def chart(fig: go.Figure, height: int = 280, **layout) -> None:
    fig.update_layout(height=height, margin={"l": 10, "r": 10, "t": 30, "b": 10}, plot_bgcolor="#fff",
                      font={"family": "Inter"}, **layout)
    st.plotly_chart(fig, use_container_width=True)


def gate_table(rows: list[dict]) -> None:
    if rows:
        st.dataframe(pd.DataFrame([{"gate": r["gate"], "value": r["value"], "limit": r["limit"],
                                    "result": "pass" if r["pass"] else ("FAIL" if r["blocking"] else "warn")}
                                   for r in rows]), hide_index=True, use_container_width=True)


ok, why = writer_ok()
running = [j for j in reg.jobs(limit=30) if j["kind"].startswith("eval") and j["status"] in ("running", "queued")]
for j in running:
    st.markdown(f"**{j['job_id']}** {badge(j['status'])} · {j['meta'].get('label', j['kind'])}",
                unsafe_allow_html=True)

if not models:
    st.info("No trained models yet (Train page). Timing runs are not evaluated.")
    st.stop()

todo = pending(reg)
offs = {m["version"]: offline_result(m["version"]) for m in models}         # each file read once per rerun
deployed = reg.current_deployed_model()
ref = offs.get(deployed) if deployed in offs else (offline_result(deployed) if deployed else None)
verdicts = {v: verdict(v, reg, write=False, off=o, deployed=deployed, ref=ref) for v, o in offs.items()}
busy = {m["version"] for m in models if is_running(m["version"])}
busy_jobs = bool([j for j in running if j["kind"] == "eval_offline"])
c1, c2 = st.columns([2, 1])
pick = c1.selectbox("Model", [m["version"] for m in models][::-1],
                    format_func=lambda v: f"{v} · {dataset_of[v]} · {verdicts[v]['verdict']}"
                    + (" · evaluating …" if v in busy else ""))
model = reg.get_model(pick)
c2.markdown("&nbsp;")
if c2.button(f"Evaluate all pending ({len(todo)})", disabled=not todo or not ok or busy_jobs,
             help=why or ("an offline eval job is already queued or running" if busy_jobs else None),
             use_container_width=True):
    jid = runner.submit("eval_offline", runner.hackpipe_cmd("eval", "--model", "pending", "--offline"),
                        needs="gpu" if gpus else "cpu", label=f"Offline eval · {', '.join(todo)}", registry=reg,
                        extra_meta={"models": todo})
    st.success(f"Started {jid}.")

v = verdicts[pick]
with st.container(border=True):
    a, b = st.columns([3, 1])
    a.markdown(f"#### {pick} {badge(v['verdict'])}", unsafe_allow_html=True)
    a.caption(f"trained on {model['dataset_version']} · status {model['status']}"
              + (f" · regression reference: deployed {v['reference']}" if v.get("reference") else ""))
    for t in v["todo"]:
        a.markdown(f"- {t}")
    if b.button("Run offline eval (test split)", type="primary", use_container_width=True,
                disabled=not ok or pick in busy or busy_jobs,
                help=why or ("an offline eval is already running" if pick in busy or busy_jobs else
                             "" if gpus else "no GPU here: runs on the CPU (slow — about an hour or more)")):
        jid = runner.submit("eval_offline", runner.hackpipe_cmd("eval", "--model", pick, "--offline"),
                            needs="gpu" if gpus else "cpu", label=f"Offline eval · {pick}", registry=reg,
                            extra_meta={"models": [pick]})
        st.success(f"Started {jid}.")
    if v["offline_gates"] or v["online_gates"]:
        st.markdown("**Gates** (`configs/gates.yaml`)")
        gate_table(v["offline_gates"] + v["online_gates"])
        st.caption(f"Isaac Sim: **{v['online_state']}**"
                   + (f" ({v['sim_eval']})" if v.get("sim_eval") else ""))

tabs = st.tabs(["Offline (test split)", "Isaac Sim", "Compare models"])

# ---------- offline ----------
with tabs[0]:
    off = offs[pick]
    if off is None:
        st.info("No offline eval yet. **Run offline eval** above (the test split is the held-out lighting10 source; "
                "about 15–30 min on a T4 with the robustness variants).")
    else:
        ex = off.get("executed") or {}
        allrow = next((r for r in off["rows"] if r["instruction"] == "full" and r["subtask"].endswith("· all")), {})
        m = st.columns(5)
        m[0].metric("Arm error, executed", f"{ex.get('arm_mean', allrow.get('chunk_arm_rad'))} rad")
        m[1].metric("Hold-still (do nothing)", f"{allrow.get('hold_still_arm_rad')} rad",
                    help="Error of a policy that keeps the current pose — the model must be clearly below it")
        m[2].metric("Arm error p99", f"{ex.get('arm_p99', 'n/a')} rad")
        ge = off.get("gripper_events") or {}
        m[3].metric("Gripper events caught", f"{ge['hit_rate']:.0%} of {ge['n']}" if ge.get("hit_rate") is not None
                    else "n/a")
        lat = off.get("latency_ms") or {}
        m[4].metric("Re-plan latency", f"{lat['replan_median']} ms" if lat else "n/a",
                    help=f"one prediction, batch 1; budget at the policy rate {lat.get('budget_ms')} ms" if lat else None)
        st.caption(f"{off.get('dataset_version')} · split {off.get('split')} · episodes {off.get('episodes')} · "
                   f"{off.get('rate_hz')} Hz, every {off.get('every')} steps · {off.get('predictions')} predictions in "
                   f"{off.get('seconds')} s · checkpoint `{off.get('checkpoint')}`")
        if ex.get("per_subtask"):
            st.markdown("**Per sub-task** (executed stream)")
            st.dataframe(pd.DataFrame(ex["per_subtask"]), hide_index=True, use_container_width=True)
        c = st.columns(2)
        with c[0]:
            hz = [x for x in off.get("horizon_arm_rad") or [] if x is not None]
            if hz:
                st.markdown("**Error by steps ahead**")
                chart(go.Figure(go.Scatter(x=list(range(1, len(hz) + 1)), y=hz, mode="lines+markers",
                                           line={"color": TOKENS["accent"]})),
                      xaxis_title="step ahead in the predicted chunk", yaxis_title="arm error (rad)")
        with c[1]:
            rob = off.get("robustness") or {}
            if rob.get("variants"):
                st.markdown(f"**Robustness** (× baseline {rob['baseline_chunk_arm_rad']} rad, "
                            f"{rob['episodes_per_source']} episodes)")
                lim = gates_config()["offline"]["block"]["robustness_ratio_max"]
                ks = [k for k in rob["variants"] if rob["variants"][k].get("ratio") is not None]
                chart(go.Figure(go.Bar(x=ks, y=[rob["variants"][k]["ratio"] for k in ks],
                                       marker_color=[TOKENS["bad"] if rob["variants"][k]["ratio"] > lim
                                                     else TOKENS["info"] for k in ks])),
                      yaxis_title="error ÷ baseline", shapes=[{"type": "line", "xref": "paper", "x0": 0, "x1": 1,
                                                               "y0": lim, "y1": lim, "line": {"dash": "dot"}}])
                for k, why_skip in (rob.get("skipped") or {}).items():
                    st.caption(f"`{k}` not measured: {why_skip}")
            else:
                st.caption("Robustness not measured (run without `--no-robustness`).")
        sm = off.get("smoothness") or {}
        if sm:
            st.caption(f"Largest joint jump per step (p95): within a prediction {sm.get('within_prediction_p95')} rad · "
                       f"at re-plan {sm.get('at_replan_p95')} rad · demo {sm.get('demo_p95')} rad")
        if ge.get("events"):
            with st.expander(f"Gripper events ({ge['n']}, timing error {ge.get('mean_abs_timing_error_s')} s)"):
                st.dataframe(pd.DataFrame(ge["events"]), hide_index=True, use_container_width=True)
        with st.expander("Instruction kinds (prediction error per sub-task, full chunk)"):
            st.dataframe(pd.DataFrame(off["rows"]), hide_index=True, use_container_width=True)

# ---------- Isaac Sim ----------
with tabs[1]:
    evs = [e for e in reg.evals(pick) if e["metrics"].get("stage_counts_0to4") is not None
           and not e["metrics"].get("fake_results")]                         # same rule as the verdict
    with st.expander("Import an Isaac Sim run (eval_isaac.py on the site PC)", expanded=not evs):
        st.caption("Run eval_isaac.py with `--ref` from `hackpipe sim reference` (held-out lighting) — see the runbook. "
                   "Paths are on this machine.")
        suites = sorted(p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "configs" / "eval").glob("sim_*.yaml"))
        f = st.columns(2)
        suite = f[0].selectbox("Suite", suites) if suites else None
        res = f[1].text_input("results.json")
        f = st.columns(2)
        replay = f[0].text_input("replay.json (eval_isaac --mode replay, same scene)",
                                 help="Without it the sim result can't be verified: verdict HOLD")
        gifs = f[1].text_input("GIF folder (optional)")
        if st.button("Import", disabled=not (suite and res and ok), help=why or None):
            missing = [x for x in (res, replay, gifs) if x and not Path(x).exists()]
            if missing:
                st.error(f"not found on this machine: {missing}")
            else:
                args = ["eval", "--model", pick, "--suite", suite, "--import", res]
                args += ["--replay", replay] if replay else []
                args += ["--gifs", gifs] if gifs else []
                jid = runner.submit("eval_import", runner.hackpipe_cmd(*args), needs="cpu",
                                    label=f"Isaac Sim import · {pick}", registry=reg)
                st.success(f"Started {jid}.")
    if not evs:
        st.info("No Isaac Sim eval for this model yet.")
    for ev in reversed(evs):
        mtr = ev["metrics"]
        ts = mtr["task_success"]
        rep = mtr.get("replay")
        with st.container(border=True):
            st.markdown(f"**{ev['eval_id']}** · suite {ev['test_suite']} · success **{ts['successes']}/{ts['n']}** "
                        f"({(ts['rate'] or 0):.0%}, 95% CI {ts['ci95'][0]:.0%}–{ts['ci95'][1]:.0%}) · demo replay "
                        + (badge("pass") if rep and rep.get("success") else badge("unverified")),
                        unsafe_allow_html=True)
            chart(go.Figure(go.Bar(x=["0 not lifted", "1 lifted", "2 halfway", "3 over tray", "4 on tray"],
                                   y=mtr["stage_counts_0to4"], marker_color=TOKENS["info"])), height=220,
                  yaxis_title="episodes")
            try:
                eps = load_results(ev["results_dir"])
            except FileNotFoundError:
                eps = []
            if eps:
                st.dataframe(pd.DataFrame([{"seed (lighting)": e.seed, "success": e.success,
                                            "failure stage": e.failure_stage, "notes": e.notes} for e in eps]),
                             hide_index=True, use_container_width=True)
                gif_eps = [e for e in eps if e.video_path and (Path(ev["results_dir"]) / e.video_path).exists()]
                if gif_eps:
                    g = st.selectbox("Camera GIF", [e.video_path for e in gif_eps], key=f"gif-{ev['eval_id']}")
                    st.image(str(Path(ev["results_dir"]) / g))

# ---------- compare ----------
with tabs[2]:
    rows = []
    for m in models:
        off = offs[m["version"]]
        ex = (off or {}).get("executed") or {}
        allrow = next((r for r in (off or {}).get("rows", []) if r["instruction"] == "full"
                       and r["subtask"].endswith("· all")), {})
        sim = sim_result(m["version"], reg)
        subs = [s for s in ex.get("per_subtask") or [] if s["arm_rad"] is not None]
        rows.append({"model": m["version"], "dataset": m["dataset_version"],
                     "started from": m["meta"].get("init_from") or "base policy",
                     "verdict": verdicts[m["version"]]["verdict"], "arm error (rad)": ex.get("arm_mean"),
                     "÷ hold-still": round(allrow["chunk_arm_rad"] / allrow["hold_still_arm_rad"], 2)
                     if allrow.get("hold_still_arm_rad") else None,
                     "worst sub-task (rad)": max((s["arm_rad"] for s in subs), default=None),
                     "gripper events": (off or {}).get("gripper_events", {}).get("hit_rate"),
                     "sim success": (sim or {}).get("task_success", {}).get("rate")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Offline numbers are comparable only on the same test source; sim success only on the same suite.")
