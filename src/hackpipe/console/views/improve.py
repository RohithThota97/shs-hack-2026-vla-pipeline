"""Improve page: the box grid player and the loop that turns its Isaac Sim recordings into a better model.
Spec: docs/ui/05_improve.md."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hackpipe.common.registry import Registry
from hackpipe.console.components.data_views import writer_ok
from hackpipe.console.components.shell import machine_info
from hackpipe.console.theme import TOKENS, badge
from hackpipe.improve import loop
from hackpipe.jobs import runner

reg = Registry()
cfg = loop.improve_config()
st.markdown("### Improve")
st.caption("06 Improve — more varied data without new teleoperation: the box grid player moves the box over the "
           "table, bends the demo's pick to follow it, Isaac Sim records it, and the pipeline turns it into a new "
           "dataset version and a fine-tuned model — then checks whether it beats the best one.")
with st.expander("Why, and how the grid player works", expanded=False):
    st.markdown(
        "- **Why:** all 84 handover demos are the same motion with the box in the same place (only the lighting "
        "changes). The model never saw the box anywhere else.\n"
        "- **Grid:** the box is shifted over the whole cart top in 5 cm steps (`configs/improve.yaml` → grid); "
        "positions where the box would not stand on the cart are skipped. The arm-base poses, the cart and the "
        "camera come from the scene file (`sim/scene.py`), so the laptop plans in the same frame as Isaac Sim.\n"
        "- **Bend the demo:** for each position the left arm's approach, grasp and lift are moved with the box "
        "by inverse kinematics on the FR3 (same gripper orientation, the demo's elbow posture); the motion goes back "
        "to the demo before the handover, so handover, release and place stay the same. A position counts only if "
        "every frame is reachable, within joint limits and below half the FR3's joint speed.\n"
        "- **Isaac Sim (site PC):** each valid position is replayed with a training episode's lighting and recorded "
        "(3 cameras, state, action). Episodes whose box doesn't end on the tray are dropped.\n"
        "- **Back into the pipeline:** import + quality checks → new dataset version with sub-task labels (the "
        "copies keep their original's split; lighting10 stays the untouched test) → approved automatically "
        "(sim-generated only, `configs/improve.yaml`) → fine-tune the best model → best checkpoint → test → "
        "compare with the best so far.")

ok, why = writer_ok()
gpus = machine_info()["gpus"]
rows = loop.status(reg)
nxt = next((r for r in rows if not r["done"]), None)

# ---------- the loop ----------
st.markdown("#### The loop")
for i, r in enumerate(rows, 1):
    mark = "✅" if r["done"] else ("👉" if r is nxt else "·")
    st.markdown(f"{mark} **{i} · {r['title']}**" + (f" — {r['detail']}" if r["detail"] else ""))
running = [j for j in reg.jobs(kind="improve_loop", limit=10) if j["status"] in ("running", "queued")]
for j in running:
    st.markdown(f"**{j['job_id']}** {badge(j['status'])} · {j['meta'].get('label')}", unsafe_allow_html=True)

c = st.columns([2, 1, 1])
steps = c[1].number_input("Fine-tune steps", 100, 20000, int((cfg["train"].get("overrides") or {}).get("steps", 3000)),
                          100, help="steps of this loop's training run (configs/improve.yaml → train)")
needs_gpu = bool(nxt and nxt["step"] in ("train", "evaluate"))
blocked = (nxt is None) or (nxt["step"] == "render") or (needs_gpu and not gpus)
reason = ("the loop is complete" if nxt is None else
          "waiting for the Isaac Sim recordings from the site PC (below)" if nxt["step"] == "render" else
          "the next step needs a GPU: open the console on the GPU machine" if needs_gpu and not gpus else None)
if c[2].button("Run the loop", type="primary", use_container_width=True, disabled=blocked or not ok or bool(running),
               help=why or reason or "runs every step that can run now and stops where it has to wait"):
    jid = runner.submit("improve_loop", runner.hackpipe_cmd("loop", "run", "--set", f"steps={int(steps)}"),
                        needs="gpu" if needs_gpu or any(r["step"] == "train" and not r["done"] for r in rows) and gpus
                        else "cpu", label=f"Improve loop from step “{nxt['title']}”", registry=reg)
    st.success(f"Started {jid}.")
if reason:
    c[0].caption(reason)

# ---------- grid ----------
plan = loop._state().get("plan")
st.markdown("#### Box positions")
if not plan:
    st.caption("Not planned yet — **Run the loop** does it first (a few minutes).")
else:
    g = pd.DataFrame(plan["grid"])
    fig = go.Figure()
    if "status" not in g:                               # plans from before the cart check
        g["status"] = np.where(g["ok"], "ok", "out_of_reach")
    for status, colour, name in (("ok", TOKENS["ok"], "valid"), ("out_of_reach", TOKENS["bad"], "not reachable"),
                                 ("off_cart", "#c8c8c8", "off the cart")):
        sub = g[g["status"] == status]
        if len(sub):
            fig.add_trace(go.Scatter(x=sub["dx"] * 100, y=sub["dy"] * 100, mode="markers", name=name,
                                     marker={"size": 12, "color": colour, "symbol": "square"},
                                     text=sub["reason"], hovertemplate="%{x} cm, %{y} cm %{text}"))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", text=["demo"], textposition="top center",
                             marker={"size": 16, "color": "#000", "symbol": "x"}, name="demo's box"))
    fig.update_layout(height=300, margin={"l": 10, "r": 10, "t": 20, "b": 10}, plot_bgcolor="#fff",
                      xaxis_title="box shift, room X (cm)", yaxis_title="box shift, room Z (cm)", legend={"orientation": "h"},
                      font={"family": "Inter"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{plan['valid']} of {plan['cells']} positions valid ({plan.get('off_cart', 0)} off the cart) · checked on {plan['reference']} · worst IK error "
               f"{plan['max_ik_mm']} mm. Isaac Sim checks them again in the real scene frame before recording.")

# ---------- site PC ----------
st.markdown("#### On the site PC (Isaac Sim)")
st.code(loop.site_pc_commands(cfg), language="bash")
st.caption("The reference file comes from step 2. Copy the recordings folder to this machine (e.g. through Drive) "
           f"into `{cfg['grid']['recordings_dir']}`, then **Run the loop** again: it goes on by itself up to the "
           "comparison.")

# ---------- result ----------
cmp_ = loop._state().get("compare")
if cmp_:
    st.markdown("#### Result")
    (st.success if cmp_.get("best") else st.warning)(cmp_["summary"])
    st.caption("Details: Evaluate page → that model (checks, 3D episodes).")
log = next((Path(j["log_path"]) for j in reg.jobs(kind="improve_loop", limit=1) if j.get("log_path")), None)
if log and log.is_file():
    with st.expander("Last loop run (log)"):
        st.code(log.read_text(errors="replace")[-6000:])
