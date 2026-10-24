"""Evaluate page: best checkpoint → test on the held-out lighting10 episodes → simple checks → verdict, plus Isaac Sim
results from the site PC. Spec: docs/ui/03_evaluate.md."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hackpipe.common.config import REPO_ROOT, gates_config
from hackpipe.common.registry import Registry
from hackpipe.console.components.shell import machine_info
from hackpipe.console.components.data_views import writer_ok
from hackpipe.console.theme import TOKENS, badge
from hackpipe.evaluation import episodes as ep_explain
from hackpipe.common.config import resolve
from hackpipe.evaluation.offline import (is_current, is_running, load_result, result_for, result_path, step_of,
                                         trajectories)
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
models = [m for m in reg.models() if m["meta"].get("purpose") != "timing" and m["status"] in (*FINISHED, "training")]
training_now = bool(reg.jobs(kind="train", status="running"))
shared_busy = any(j["kind"] == "eval_offline" and j["status"] in ("running", "queued") and j["meta"].get("shared_gpu")
                  for j in reg.jobs(kind="eval_offline", limit=50))


def submit_test(model_version: str, step: int | None = None, shared: bool = False) -> str:
    """shared: a step test of the model being trained runs next to the training (no GPU lock, one at a time);
    every other test waits for the GPU like any job."""
    args = ["eval", "--model", model_version, "--offline"] + (["--step", str(step)] if step is not None else [])
    label = f"Test on lighting10 · {model_version}" + (f" step {step}" if step is not None else "")
    return runner.submit("eval_offline", runner.hackpipe_cmd(*args), needs="cpu" if shared or not gpus else "gpu",
                         label=label, registry=reg, extra_meta={"models": [model_version], "step": step,
                                                                "shared_gpu": shared})


@st.cache_data(show_spinner=False)
def step_summary(path: str, mtime: float) -> str:          # cached by file + mtime: the table re-reads nothing
    import json as _json
    r = _json.loads(Path(path).read_text())
    ex_ = r.get("executed") or {}
    if not ex_.get("arm_mean"):
        return ""
    a = next((x for x in r.get("rows") or [] if x["instruction"] == "full" and x["subtask"].endswith("· all")), {})
    g = (r.get("gripper_events") or {}).get("hit_rate")
    ratio = (f"{a['chunk_arm_rad'] / a['hold_still_arm_rad']:.0%} of doing nothing"   # 3.3 s, as the checks
             if a.get("hold_still_arm_rad") and a.get("chunk_arm_rad") is not None else f"{ex_['arm_mean']} rad")
    return ratio + (f", gripper {g:.0%}" if g is not None else "")


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
                    format_func=lambda v: f"{v} · {reg.get_model(v)['dataset_version']}"
                    + (" · training now" if reg.get_model(v)["status"] == "training" else ""))
model = reg.get_model(pick)
if model["status"] == "training":
    st.info("This model is still training. Saved checkpoints can be tested now (step 1); the test runs next to the "
            "training on the same GPU.")

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
    def tested(step: int) -> str:
        f = result_path(pick, step=step)
        if is_running(pick, step=step):
            return "testing …"
        return step_summary(str(f), f.stat().st_mtime) if f.is_file() else ""
    st.dataframe(pd.DataFrame([{"step": r["step"], "stage": r["stage"], "val error, full task (rad)":
                                r["val_full_arm_rad"], "val error, sub-task (rad)": r["val_sub_arm_rad"],
                                "saved": "yes" if r["checkpoint"] else "deleted",
                                "test on lighting10": tested(r["step"]),
                                "": "best" if b and r["step"] == b["step"] else ""} for r in cands]),
                 hide_index=True, use_container_width=True)
    saved = [r["step"] for r in cands if r["checkpoint"]]
    if saved:
        t = st.columns([2, 1])
        step_pick = t[0].selectbox("Test a saved checkpoint now (also while training)", saved[::-1],
                                   format_func=lambda x: f"step {x:,}" + (" (best so far)" if b and x == b["step"] else ""))
        shared = model["status"] == "training" and training_now
        if t[1].button("Test this checkpoint", use_container_width=True,
                       disabled=not ok or is_running(pick, step=step_pick) or (shared and shared_busy),
                       help=why or ("one test at a time runs next to the training — wait for it" if shared and
                                    shared_busy else "~15–25 min on the A100")):
            st.success(f"Started {submit_test(pick, step_pick, shared)} — the result appears in the table.")

# ---------- 2. test ----------
st.markdown("#### 2 · Test on lighting10 (held out)")
off = result_for(model)                 # the test of the checkpoint in use (also one made during training)
cur_step = step_of(resolve(model["checkpoint_dir"] or ".").resolve())
busy = is_running(pick, step=cur_step) or any(pick in (j["meta"].get("models") or []) and
                                               j["meta"].get("step") in (None, cur_step) for j in running)
c = st.columns([3, 1])
c[0].caption(f"Checkpoint: `{model['checkpoint_dir']}`" + (f" (step {chosen})" if chosen else ""))
if c[1].button("Run the test" if off is None else "Test again", type="primary", use_container_width=True,
               disabled=not ok or busy or (bool(cands) and chosen is None),
               help=why or ("a test of this model is running" if busy else
                            "pick the best checkpoint first (step 1)" if cands and chosen is None else
                            "~15–25 min on a GPU" if gpus else "no GPU here: slow on the CPU")):
    st.success(f"Started {submit_test(pick)}. Results appear here when it ends.")
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
still = allrow.get("hold_still_arm_rad")      # doing nothing over the whole 3.3 s prediction (as the checks)
pred_err = allrow.get("chunk_arm_rad")
ge = off.get("gripper_events") or {}
lat = off.get("latency_ms") or {}
m = st.columns(4)
m[0].metric("Arm error", f"{pred_err:.3f} rad" if pred_err is not None else "n/a",
            help="mean difference between the joints the model predicts and the demo's, over its whole 3.3 s "
                 f"prediction (the actions it would execute: {ex['arm_mean']:.3f} rad)")
m[1].metric("Doing nothing", f"{still:.3f} rad" if still else "n/a",
            help="error of just keeping the current pose, over the same 3.3 s")
m[2].metric("Gripper on time", f"{ge['hit_rate']:.0%}" if ge.get("hit_rate") is not None else "n/a",
            help="share of the demo's grasps (close) and releases (open) the model matches within ±1 s")
m[3].metric("One prediction", f"{lat['replan_median']:.0f} ms" if lat else "n/a",
            help=f"time to predict once (batch 1); a 6 Hz robot has {lat.get('budget_ms', 167)} ms")
if still and pred_err is not None:
    ratio = pred_err / still
    need = gates_config()["offline"]["block"]["max_ratio_to_hold_still"]
    st.markdown(f"The model's error is **{ratio:.0%} of doing nothing** "
                + (f"— clearly better than standing still (needed: ≤ {need:.0%})." if ratio <= need else
                   "— not yet much better than standing still." if ratio < 1 else "— worse than standing still."))

subs = [{"subtask": r["subtask"], "arm_rad": r["chunk_arm_rad"]} for r in off["rows"]
        if r["instruction"] == "full" and not r["subtask"].endswith("· all")]
stills = {r["subtask"]: r["hold_still_arm_rad"] for r in off["rows"] if r["instruction"] == "full"}
c = st.columns(2)
with c[0]:
    if subs:
        st.markdown("**Per sub-task** (model vs doing nothing, 3.3 s prediction)")
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

# ---------- 3. why: episodes in 3D ----------
st.markdown("#### 3 · Why: the test episodes in 3D")
eps = trajectories(off)
if not eps:
    st.caption("No trajectories saved with this test (tests made before 24 Oct) — test again to see them.")
else:
    hz = float(off.get("rate_hz") or 6)
    sums = [ep_explain.summarize(e, hz) for e in eps]
    n_ok = sum(x["follows_demo"] for x in sums)
    st.markdown(f"**{n_ok} of {len(sums)}** test episodes follow the demo (grasp points within "
                f"{ep_explain.DRIFT_CM:g} cm, every grasp and release on time).")
    st.dataframe(pd.DataFrame([{"episode": x["ep"], "": "✅" if x["follows_demo"] else "❌", "mean gap (cm)":
                                x["mean_cm"], "max gap (cm)": x["max_cm"], "drifts at (s)": x["drift_at_s"],
                                "missed grasps / releases": len(x["missed_gripper"]), "why": x["why"]} for x in sums]),
                 hide_index=True, use_container_width=True)
    epi = st.selectbox("Show episode in 3D", range(len(eps)),
                       format_func=lambda i: f"episode {sums[i]['ep']} " + ("✅" if sums[i]["follows_demo"] else "❌"))
    e, sm = eps[epi], sums[epi]
    P = ep_explain.paths(e)
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("left arm (its base frame)", "right arm (its base frame)"))
    from hackpipe.evaluation.offline import GRIP, gripper_events
    for col, (side, g) in enumerate((("left", GRIP[0]), ("right", GRIP[1])), start=1):
        d, mo = P["demo"][side], P["model"][side]
        gap = np.linalg.norm(mo - d, axis=1) * 100
        fig.add_trace(go.Scatter3d(x=d[:, 0], y=d[:, 1], z=d[:, 2], mode="lines", name="demo",
                                   line={"color": "#9E9E9E", "width": 6}, showlegend=col == 1), row=1, col=col)
        fig.add_trace(go.Scatter3d(x=mo[:, 0], y=mo[:, 1], z=mo[:, 2], mode="markers", name="model (gap cm)",
                                   marker={"size": 3, "color": gap, "colorscale": "RdYlGn_r", "cmin": 0,
                                           "cmax": max(2 * ep_explain.DRIFT_CM, 1), "showscale": col == 2,
                                           "colorbar": {"title": "gap cm"}},
                                   text=[f"{t / hz:.1f} s · {s_} · gap {gg:.1f} cm" for t, (s_, gg) in
                                         enumerate(zip(e["sub"], gap))], hoverinfo="text",
                                   showlegend=col == 1), row=1, col=col)
        for t, kind in gripper_events(e["gt"], g):
            fig.add_trace(go.Scatter3d(x=[d[t, 0]], y=[d[t, 1]], z=[d[t, 2]], mode="markers+text",
                                       text=["grasp" if kind == "close" else "release"], textposition="top center",
                                       marker={"size": 7, "color": TOKENS["info"], "symbol": "diamond"},
                                       name="demo grasp / release", showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter3d(x=[d[0, 0]], y=[d[0, 1]], z=[d[0, 2]], mode="markers+text", text=["start"],
                                   marker={"size": 6, "color": "#000"}, showlegend=False), row=1, col=col)
    if sm["drift_step"] is not None:                     # on the arm that drifts, at the step summarize() found
        t = sm["drift_step"]
        for col, side in ((1, "left"), (2, "right")):
            if side != sm["drift_arm"]:
                continue
            q = P["model"][side][t]
            fig.add_trace(go.Scatter3d(x=[q[0]], y=[q[1]], z=[q[2]], mode="markers+text", text=["drifts here"],
                                       marker={"size": 9, "color": TOKENS["bad"], "symbol": "x"},
                                       showlegend=False), row=1, col=col)
    fig.update_layout(height=520, margin={"l": 0, "r": 0, "t": 30, "b": 0}, legend={"orientation": "h"},
                      scene={"aspectmode": "data"}, scene2={"aspectmode": "data"})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Episode {sm['ep']}:** {sm['why']}")
    st.caption("Grey: the demo's grasp-point path; dots: what the model commands at each step, coloured by the gap "
               "to the demo (green close, red far); diamonds: the demo's grasps and releases. Paths come from the "
               "joint angles through the FR3's kinematics. Teacher-forced: every prediction starts from the recorded "
               "state, so this shows where the model's commands go wrong, not a closed-loop run — that is the "
               "Isaac Sim test (step 5).")

# ---------- 4. checks + verdict ----------
st.markdown("#### 4 · Checks and verdict")
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

# ---------- 5. Isaac Sim ----------
st.markdown("#### 5 · Isaac Sim results (site PC)")
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
