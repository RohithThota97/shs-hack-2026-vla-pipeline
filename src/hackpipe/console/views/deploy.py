"""Deploy page (R10): what serves the Isaac Sim bridge now, stop / start / rollback, gated deploy with a staged
health check, history. Spec: docs/ui/04_deploy.md. After the team's pipeline UI (AI-project-work)."""
import pandas as pd
import streamlit as st

from hackpipe.common.registry import Registry
from hackpipe.console.components.data_views import writer_ok
from hackpipe.console.theme import badge
from hackpipe.deploy import serving
from hackpipe.gates.verdict import offline_result, verdict
from hackpipe.jobs import runner
from hackpipe.training.ranking import FINISHED

reg = Registry()
cfg = serving.settings()
PORT, STAGE = int(cfg["port"]), int(cfg["stage_port"])
st.markdown("### Deploy")
st.caption(f"05 Deploy — serve a tested model for Isaac Sim on port {PORT}. The team's Isaac Sim client (site PC) "
           "sends camera images + joint state there and gets the robot's next action back.")
with st.expander("How serving works", expanded=False):
    st.markdown(
        f"1. **Who may be served:** a model whose verdict (Evaluate page) is **READY FOR SIM TEST** (to run the sim "
        "test) or **READY TO DEPLOY**. Anything else only with an override and your name — recorded in the history.\n"
        f"2. **Safe switch:** the new model is first loaded on a spare port ({STAGE}) and checked on a recorded "
        f"lighting10 frame (it must return 16 finite joint values). Only then does it replace what serves {PORT}. If "
        "anything fails, the old model keeps serving.\n"
        "3. **Actions per prediction:** the model predicts 20 steps ahead (3.3 s); the robot executes this many of "
        "them, then asks again. 5 = re-plan every 0.8 s (the sim suite's setting).\n"
        "4. **Roll back** puts the previous served model back, with its settings.")


ok, why = writer_ok()
if st.session_state.get("deploy_msg"):              # result of the last button, shown after the rerun
    st.success(st.session_state.pop("deploy_msg"))
jobs = [j for j in reg.jobs(limit=30) if j["kind"] in ("deploy", "serving") and j["status"] in ("running", "queued")]
for j in jobs:
    st.markdown(f"**{j['job_id']}** {badge(j['status'])} · {j['meta'].get('label', j['kind'])}", unsafe_allow_html=True)
busy = bool(jobs)

# ---------- serving now ----------
deps = reg.deployments()
current = reg.current_deployment()
stt = serving.status(PORT)
with st.container(border=True):
    a, b = st.columns([3, 1])
    live = stt["info"] or {}
    if stt["live"]:
        a.markdown(f"#### Port {PORT} {badge('running')} · {live.get('model') or live.get('checkpoint')}"
                   + (" · FAKE" if live.get("fake") else ""), unsafe_allow_html=True)
        a.caption(f"{live.get('policy_type')} · executes {live.get('n_action_steps')} of {live.get('chunk_size')} "
                  f"actions per prediction at {live.get('policy_hz')} Hz"
                  + (f" · grasp extension {live.get('grasp_extend_s')} s" if live.get("grasp_extend_s") else "")
                  + (f" · since {stt['state']['started_at']}" if stt.get("state") else ""))
        if stt["foreign"]:
            a.warning(f"Port {PORT} is held by a server hackpipe did not start (e.g. the team's "
                      "fr3duo_policy_server.py). Deploys refuse to replace it — stop it where it was started.")
    else:
        a.markdown(f"#### Port {PORT} {badge('stopped')} · nothing served", unsafe_allow_html=True)
    if current:
        a.caption(f"Deployed: **{current['model_version']}** ({current['dep_id']}, {current['created_at']}"
                  + (", supervised" if (current['meta'] or {}).get("supervised") else "") + ")")
    if b.button("Stop serving", disabled=not (stt["live"] and stt["ours"]) or busy, use_container_width=True):
        with st.spinner(f"stopping the server on {PORT} …"):
            stopped = serving.stop(PORT)
        st.session_state["deploy_msg"] = f"Port {PORT} stopped." if stopped else "Nothing of ours was serving."
        st.rerun()
    if b.button("Start (deployed model)", disabled=stt["live"] or not current or busy or not ok,
                use_container_width=True, help=why or "serves the deployed model again, with its recorded settings"):
        jid = runner.submit("serving", runner.hackpipe_cmd("serving", "start"), needs="cpu",
                            label=f"Serve {current['model_version']} on {PORT}", registry=reg)
        st.success(f"Started {jid} (model load: about a minute).")
    n_deployed = sum(1 for d in deps if d["status"] == "deployed")
    if b.button("Roll back", disabled=n_deployed < 2 or busy or not ok, use_container_width=True,
                help=why or "re-deploy the previous model (staged + health-checked) and serve it"):
        jid = runner.submit("deploy", runner.hackpipe_cmd("rollback", "--serve"), needs="cpu",
                            label="Rollback", registry=reg)
        st.success(f"Started {jid}.")

# ---------- deploy ----------
st.markdown("**Deploy a model**")
models = [m for m in reg.models() if m["meta"].get("purpose") != "timing" and m["status"] in FINISHED]
if not models:
    st.info("No trained models yet.")
else:
    dep_model = current["model_version"] if current else None
    c = st.columns([2, 1, 1])
    pick = c[0].selectbox("Model", [m["version"] for m in models][::-1],
                          format_func=lambda v: f"{v} · {next(m['dataset_version'] for m in models if m['version'] == v)}")
    nas = c[1].number_input("Actions per prediction", 1, 50, int(cfg.get("n_action_steps") or 5),
                            help="executed before re-planning; the sim suite's n_action_steps")
    ge = c[2].number_input("Grasp extension (s)", 0.0, 3.0, 0.0, 0.1,
                           help="inference-time grasp shaping (the team's --grasp_extend); 0 = off")
    v = verdict(pick, reg, write=False, deployed=dep_model,           # only the picked model, once per rerun
                ref=offline_result(dep_model) if dep_model and dep_model != pick else None)
    st.markdown(f"Verdict for **{pick}**: {badge(v['verdict'])}", unsafe_allow_html=True)
    for t in v["todo"]:
        st.markdown(f"- {t}")
    ready = v["verdict"] in ("READY FOR SIM TEST", "READY TO DEPLOY")
    override = st.checkbox("Serve anyway (override — recorded with your name)", disabled=ready,
                           help="e.g. to look at a model in the sim that hasn't passed the offline checks")
    by = st.text_input("Your name (recorded)")
    allowed = ready or override
    label = "Serve for the sim test" if v["verdict"] != "READY TO DEPLOY" else "Deploy"
    if st.button(f"{label}: {pick} on port {PORT}", type="primary",
                 disabled=not (allowed and by.strip() and ok) or busy or stt["foreign"],
                 help=why or ("needs a READY verdict (Evaluate page) or the override, and your name"
                              if not allowed or not by.strip() else None)):
        args = ["deploy", "--model", pick, "--serve", "--n-action-steps", str(int(nas)), "--by", by.strip()]
        args += ["--grasp-extend", str(float(ge))] if ge else []
        args += ["--override"] if override and not ready else []
        jid = runner.submit("deploy", runner.hackpipe_cmd(*args), needs="cpu", label=f"Deploy {pick}", registry=reg)
        st.success(f"Started {jid}: loading on {STAGE}, checking on a recorded frame, then port {PORT} (about a "
                   "minute). The old model keeps serving if anything fails.")
    st.caption("The server uses GPU memory while it runs; stop it before a training run on a small GPU.")

# ---------- history ----------
if deps:
    st.markdown("**History**")
    st.dataframe(pd.DataFrame([{"deployment": d["dep_id"], "model": d["model_version"], "status": d["status"],
                                "when": d["created_at"], "verdict": (d["meta"] or {}).get("verdict"),
                                "by": (d["meta"] or {}).get("by"),
                                "purpose": (d["meta"] or {}).get("purpose"),
                                "override": (d["meta"] or {}).get("supervised"),
                                "rollback from": (d["meta"] or {}).get("rollback_from"),
                                "health check": ((d["meta"] or {}).get("serving") or {}).get("health_check", {})
                                .get("first_call_ms"), "error": (d["meta"] or {}).get("error")}
                               for d in reversed(deps)]), hide_index=True, use_container_width=True)
