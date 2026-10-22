"""R9 dashboard — `streamlit run dashboard/app.py` (needs `pip install -e ".[dashboard]"`)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from hackpipe.common.config import project_config, resolve
from hackpipe.common.registry import Registry
from hackpipe.evaluation.metrics import compare
from hackpipe.evaluation.runner import load_results
from hackpipe.explain.failures import categorize, summary

st.set_page_config(page_title="Demo-to-Deployment Pipeline", layout="wide")
reg = Registry()
st.title("Demo-to-Deployment Pipeline")

tab_models, tab_eval, tab_eps, tab_data, tab_card = st.tabs(
    ["Models", "Evaluation", "Episodes", "Data quality", "Quality statement"])

evals = reg.evals()

with tab_models:
    models = reg.models()
    if not models:
        st.info("No models registered yet.")
    else:
        rows = []
        for m in models:
            ev = [e for e in evals if e["model_version"] == m["version"]]
            last = ev[-1]["metrics"] if ev else None
            rows.append({"model": m["version"], "policy": m["policy_name"], "dataset": m["dataset_version"],
                         "status": m["status"],
                         "success": None if not last else last["task_success"]["rate"],
                         "ci95": None if not last else last["task_success"]["ci95"],
                         "fake": None if not last else last.get("fake_results")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption(f"Deployed: {reg.current_deployed_model() or 'none'}")

with tab_eval:
    if not evals:
        st.info("No evaluations yet.")
    else:
        ids = [e["eval_id"] for e in evals]
        sel = st.selectbox("Evaluation", ids, index=len(ids) - 1)
        ev = reg.get_eval(sel)
        m = ev["metrics"]
        if m.get("fake_results"):
            st.error("FAKE results from the dummy adapter — do not use these numbers.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Task success", f"{m['task_success']['rate']:.0%}", help=f"95% CI {m['task_success']['ci95']}")
        c2.metric("Unseen success", "n/a" if m["unseen"]["rate"] is None else f"{m['unseen']['rate']:.0%}")
        c3.metric("Collision rate", "n/a" if m["collision_rate"] is None else f"{m['collision_rate']:.0%}")
        c4.metric("Latency p95 (ms)", m["inference_latency_ms"]["p95"])
        st.subheader("Success by category")
        st.dataframe(pd.DataFrame([{"category": k, **v} for k, v in m["by_category"].items()]),
                     use_container_width=True)
        st.subheader("Failure categories")
        st.bar_chart(pd.Series(m["failure_categories"], name="episodes"))
        others = [e for e in evals if e["test_suite"] == ev["test_suite"] and e["eval_id"] != sel]
        if others:
            ref = st.selectbox("Compare with", [e["eval_id"] for e in others])
            st.dataframe(pd.DataFrame(compare(m, reg.get_eval(ref)["metrics"])), use_container_width=True)

with tab_eps:
    if evals:
        sel = st.selectbox("Evaluation ", [e["eval_id"] for e in evals], index=len(evals) - 1)
        results = load_results(reg.get_eval(sel)["results_dir"])
        cats = sorted({categorize(r) for r in results})
        pick = st.multiselect("Failure category", cats, default=[c for c in cats if c != "success"])
        for r in [r for r in results if categorize(r) in pick]:
            with st.expander(summary(r)):
                st.json(r.to_dict())
                if r.video_path and Path(r.video_path).exists():
                    st.video(r.video_path)

with tab_data:
    qc = resolve(Path(project_config()["project"]["outputs_dir"]) / "qc" / "qc_report.json")
    if qc.exists():
        rep = json.loads(qc.read_text())
        st.write(rep["status_counts"])
        st.write(rep["reason_counts"])
        st.dataframe(pd.DataFrame(rep["episodes"]), use_container_width=True)
    else:
        st.info("Run `hackpipe qc` first.")
    for d in reg.datasets():
        with st.expander(f"{d['version']} · {d['status']} · {d['fingerprint']}"):
            card = Path(d["meta"].get("card", ""))
            st.markdown(card.read_text() if card.exists() else "card missing")

with tab_card:
    if evals:
        sel = st.selectbox("Evaluation  ", [e["eval_id"] for e in evals], index=len(evals) - 1)
        card = Path(reg.get_eval(sel)["results_dir"], "model_card.md")
        st.markdown(card.read_text() if card.exists() else "Run `hackpipe card --eval <id>` first.")
