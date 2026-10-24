"""Evaluate page: best checkpoint → test on the held-out lighting10 episodes → simple checks → verdict, plus Isaac Sim
results from the site PC. Spec: docs/ui/03_evaluate.md."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hackpipe.common.config import REPO_ROOT, gates_config
from hackpipe.common.registry import Registry
from hackpipe.console.components.shell import machine_info
from hackpipe.console.components.data_views import writer_ok
from hackpipe.console.theme import TOKENS, badge
from hackpipe.evaluation.offline import is_current, is_running
from hackpipe.evaluation.runner import load_results
from hackpipe.gates.verdict import offline_result, sim_result, verdict
from hackpipe.jobs import runner
from hackpipe.training import best as best_ck
from hackpipe.training.ranking import FINISHED

reg = Registry()
st.markdown("### Evaluate")
st.caption("04 Evaluate — is the trained model good enough? It is tested on **fr3_duo_handover_lighting10**: 10 "
           "handover episodes with lighting it never saw in training. Then a few simple checks decide what's next.")
with st.expander("How the test works", expanded=False):
    st.markdown(
        "1. **Best checkpoint** — training saves a checkpoint every few hundred steps and measures its error on the "
        "validation episodes of `fr3_duo_handover_84`. The checkpoint with the lowest error is used, not simply the "
        "last one.\n"
        "2. **Test** — for every 3rd step (6 Hz) of each lighting10 episode, the model sees the recorded cameras + "
        "joint state and predicts its next actions. These are compared with what the demo did. The robot does not "
        "move, so this measures prediction quality, not closed-loop success.\n"
        "3. **Doing nothing** is the yardstick: the error of a policy that just keeps the current pose. A useful model "
        "is far below it.\n"
        "4. **Isaac Sim** (site PC) then measures real success: does the box end up on the tray?")
gpus = machine_info()["gpus"]
models = [m for m in reg.models() if m["meta"].get("purpose") != "timing" and m["status"] in FINISHED]


def chart(fig: go.Figure, height: int = 260, **layout) -> None:
    fig.update_layout(height=height, margin={"l": 10, "r": 10, "t": 30, "b": 10}, plot_bgcolor="#fff",
                      font={"family": "Inter"}, **layout)
    st.plotly_chart(fig, use_container_width=True)


ok, why = writer_ok()
running = [j for j in reg.jobs(limit=30) if j["kind"].startswith("eval") and j["status"] in ("running", "queued")]
for j in running:
    st.markdown(f"**{j['job_id']}** {badge(j['status'])} · {j['meta'].get('label', j['kind'])}", unsafe_allow_html=True)
if not models:
    st.info("No trained model yet — start one on the Train page (timing runs are not tested).")
    st.stop()

pick = st.selectbox("Model", [m["version"] for m in models][::-1],
                    format_func=lambda v: f"{v} · {reg.get_model(v)['dataset_version']}")
model = reg.get_model(pick)

# ---------- 1. best checkpoint ----------
st.markdown("#### 1 · Best checkpoint")
cands = best_ck.candidates(model)
b = best_ck.best(model)
chosen = model["meta"].get("best_step")
if not cands:
    st.caption("No validation log (curriculum runs only) — the model's last checkpoint is used.")
else:
    c = st.columns([3, 1])
    with c[0]:
        steps = [r["step"] for r in cands]
        fig = go.Figure(go.Scatter(x=steps, y=[r["val_full_arm_rad"] for r in cands], mode="lines+markers",
                                   name="validation error", line={"color": TOKENS["info"]}))
        if b:
            fig.add_trace(go.Scatter(x=[b["step"]], y=[b["val_full_arm_rad"]], mode="markers", name="best",
                                     marker={"size": 14, "color": TOKENS["ok"], "symbol": "star"}))
        chart(fig, 220, xaxis_title="training step", yaxis_title="val arm error (rad)", legend={"orientation": "h"})
    with c[1]:
        if b:
            st.metric("Best step", f"{b['step']:,}", help="lowest validation error among the saved checkpoints")
            st.metric("Its validation error", f"{b['val_full_arm_rad']} rad")
            if chosen == b["step"]:
                st.success("In use for test and serving.")
            elif st.button("Use this checkpoint", type="primary", disabled=not ok, help=why or None):
                best_ck.select(pick, reg)
                st.rerun()
        else:
            st.caption("No saved checkpoint matches a validation step yet.")
    with st.expander("All checkpoints"):
        st.dataframe(pd.DataFrame([{"step": r["step"], "stage": r["stage"], "val error, full task (rad)":
                                    r["val_full_arm_rad"], "val error, sub-task (rad)": r["val_sub_arm_rad"],
                                    "saved": "yes" if r["checkpoint"] else "deleted",
                                    "": "best" if b and r["step"] == b["step"] else ""} for r in cands]),
                     hide_index=True, use_container_width=True)

# ---------- 2. test ----------
st.markdown("#### 2 · Test on lighting10 (held out)")
off = offline_result(pick)
busy = is_running(pick) or any(pick in (j["meta"].get("models") or []) for j in running)
c = st.columns([3, 1])
c[0].caption(f"Checkpoint: `{model['checkpoint_dir']}`" + (f" (step {chosen})" if chosen else ""))
if c[1].button("Run the test" if off is None else "Test again", type="primary", use_container_width=True,
               disabled=not ok or busy or (bool(cands) and chosen is None),
               help=why or ("a test of this model is running" if busy else
                            "pick the best checkpoint first (step 1)" if cands and chosen is None else
                            "~15–25 min on a GPU" if gpus else "no GPU here: slow on the CPU")):
    jid = runner.submit("eval_offline", runner.hackpipe_cmd("eval", "--model", pick, "--offline"),
                        needs="gpu" if gpus else "cpu", label=f"Test on lighting10 · {pick}", registry=reg,
                        extra_meta={"models": [pick]})
    st.success(f"Started {jid}. Results appear here when it ends.")
if busy:
    st.info("Test running …")
if off is None or not (off.get("executed") or {}).get("arm_mean"):
    if off is not None:
        st.warning("This test is from an older version without the checks below — run it again.")
    st.stop()
if not is_current(off, model):
    st.warning(f"This test is of another checkpoint (`{off.get('checkpoint')}`) than the one in use now — "
               "**Test again**. The numbers below are the old checkpoint's.")

ex = off["executed"]
allrow = next((r for r in off["rows"] if r["instruction"] == "full" and r["subtask"].endswith("· all")), {})
still = allrow.get("hold_still_arm_rad")
ge = off.get("gripper_events") or {}
lat = off.get("latency_ms") or {}
m = st.columns(4)
m[0].metric("Arm error", f"{ex['arm_mean']:.3f} rad",
            help="mean difference between the joints the model commands and the demo's, over all test steps")
m[1].metric("Doing nothing", f"{still:.3f} rad" if still else "n/a", help="error of just keeping the current pose")
m[2].metric("Gripper on time", f"{ge['hit_rate']:.0%}" if ge.get("hit_rate") is not None else "n/a",
            help="share of the demo's gripper close / open moments the model matches within ±1 s")
m[3].metric("One prediction", f"{lat['replan_median']:.0f} ms" if lat else "n/a",
            help=f"time to predict once (batch 1); a 6 Hz robot has {lat.get('budget_ms', 167)} ms")
if still:
    ratio = ex["arm_mean"] / still
    need = gates_config()["offline"]["block"]["max_ratio_to_hold_still"]
    st.markdown(f"The model's error is **{ratio:.0%} of doing nothing** "
                + (f"— clearly better than standing still (needed: ≤ {need:.0%})." if ratio <= need else
                   "— not yet much better than standing still." if ratio < 1 else "— worse than standing still."))

subs = [s for s in ex.get("per_subtask") or [] if s.get("arm_rad") is not None]
stills = {r["subtask"]: r["hold_still_arm_rad"] for r in off["rows"] if r["instruction"] == "full"}
c = st.columns(2)
with c[0]:
    if subs:
        st.markdown("**Per sub-task** (model vs doing nothing)")
        names = [s["subtask"].split(" · ", 1)[-1] for s in subs]
        chart(go.Figure([go.Bar(name="model", x=names, y=[s["arm_rad"] for s in subs], marker_color=TOKENS["info"]),
                         go.Bar(name="doing nothing", x=names, y=[stills.get(s["subtask"]) for s in subs],
                                marker_color=TOKENS["line"])]), barmode="group", yaxis_title="arm error (rad)",
              legend={"orientation": "h"})
with c[1]:
    rob = (off.get("robustness") or {}).get("variants") or {}
    ks = [k for k in rob if rob[k].get("ratio") is not None]
    if ks:
        lim = gates_config()["offline"]["block"]["robustness_ratio_max"]
        st.markdown("**Robustness** (error ÷ normal; dotted line = limit)")
        chart(go.Figure(go.Bar(x=ks, y=[rob[k]["ratio"] for k in ks],
                               marker_color=[TOKENS["bad"] if rob[k]["ratio"] > lim else TOKENS["ok"] for k in ks])),
              yaxis_title="× normal", shapes=[{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": lim,
                                               "y1": lim, "line": {"dash": "dot"}}])
st.caption(f"{sum((off.get('episodes') or {}).values())} episodes · {off.get('predictions')} predictions in "
           f"{off.get('seconds')} s · {off.get('created_at')}")

# ---------- 3. checks + verdict ----------
st.markdown("#### 3 · Checks and verdict")
v = verdict(pick, reg, write=False, off=off)
rows = v["offline_gates"] + v["online_gates"]
st.markdown(f"Verdict: {badge(v['verdict'])}", unsafe_allow_html=True)
st.dataframe(pd.DataFrame([{"check": r["gate"], "value": r["value"], "needed": r["limit"],
                            "": "✅" if r["pass"] else ("❌" if r["blocking"] else "⚠️ (info)")} for r in rows]),
             hide_index=True, use_container_width=True)
for t in v["todo"]:
    st.markdown(f"- {t}")
st.caption("Limits: `configs/gates.yaml`. Next step for READY FOR SIM TEST: Deploy page → serve the model, run "
           "Isaac Sim on the site PC, import the result below.")

# ---------- 4. Isaac Sim ----------
st.markdown("#### 4 · Isaac Sim results (site PC)")
evs = [e for e in reg.evals(pick) if e["metrics"].get("stage_counts_0to4") is not None
       and not e["metrics"].get("fake_results")]
for ev in reversed(evs):
    mtr, rep = ev["metrics"], ev["metrics"].get("replay")
    ts = mtr["task_success"]
    with st.container(border=True):
        st.markdown(f"**{ts['successes']}/{ts['n']} on the tray** ({(ts['rate'] or 0):.0%}, 95% CI "
                    f"{ts['ci95'][0]:.0%}–{ts['ci95'][1]:.0%}) · demo replay "
                    + (badge("pass") if rep and rep.get("success") else badge("unverified"))
                    + f" · {ev['eval_id']}", unsafe_allow_html=True)
        chart(go.Figure(go.Bar(x=["not lifted", "lifted", "halfway", "over tray", "on tray"],
                               y=mtr["stage_counts_0to4"], marker_color=TOKENS["info"])), 200, yaxis_title="episodes")
        try:
            eps = load_results(ev["results_dir"])
        except FileNotFoundError:
            eps = []
        gifs = [e for e in eps if e.video_path and (Path(ev["results_dir"]) / e.video_path).exists()]
        if gifs:
            g = st.selectbox("Camera GIF", [e.video_path for e in gifs], key=f"gif-{ev['eval_id']}")
            st.image(str(Path(ev["results_dir"]) / g))
with st.expander("Import an Isaac Sim run", expanded=not evs):
    st.caption("On the site PC: serve this model (Deploy page), run eval_isaac.py with the held-out lighting reference "
               "(`hackpipe sim reference`, runbook), copy results.json + replay.json here.")
    suites = sorted(p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "configs" / "eval").glob("sim_*.yaml"))
    f = st.columns(3)
    suite = f[0].selectbox("Suite", suites) if suites else None
    res = f[1].text_input("results.json (path)")
    replay = f[2].text_input("replay.json (path)", help="the demo replayed in the same scene — without it the run "
                                                        "can't be verified")
    if st.button("Import", disabled=not (suite and res and ok), help=why or None):
        missing = [x for x in (res, replay) if x and not Path(x).exists()]
        if missing:
            st.error(f"not found on this machine: {missing}")
        else:
            args = ["eval", "--model", pick, "--suite", suite, "--import", res] + (["--replay", replay] if replay else [])
            jid = runner.submit("eval_import", runner.hackpipe_cmd(*args), needs="cpu",
                                label=f"Isaac Sim import · {pick}", registry=reg)
            st.success(f"Started {jid}.")
