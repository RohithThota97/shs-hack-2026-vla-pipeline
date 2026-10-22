"""Global page: the challenge, the five stages with live status, what to do next, known data gaps."""
import html

import streamlit as st

from hackpipe.common.registry import Registry
from hackpipe.console.theme import card
from hackpipe.status import overview

reg = Registry()
ov = overview(reg)

st.markdown("### Teach the Robot — from demonstration to deployed, evaluated robot model")
st.markdown('<span class="hp-muted">Leading question: how can a human demonstration become a deployed, evaluated '
            "and explainable robot model within 72 hours — and improve with each iteration?</span>",
            unsafe_allow_html=True)

st.markdown('<div class="hp-next"><b>Next:</b> ' + " ".join(html.escape(s) for s in ov["next"]) + "</div>",
            unsafe_allow_html=True)
st.write("")

cols = st.columns(5)
for col, s in zip(cols, ov["stages"]):
    col.markdown(card(s["title"], s["lines"], s["status"], s["id"]), unsafe_allow_html=True)
st.caption("Stage 05 feeds the next learning cycle: new demonstrations start again at 01.")

st.write("")
a, b = st.columns(2)
a.markdown(card("Intelligent Success Judgement", ["Assess rollouts with state traces or a VLM; explain where "
                                                  "execution degraded."], "todo", "A", wide=True), unsafe_allow_html=True)
b.markdown(card("Self-Improving Pipeline", ["Find capability gaps, recommend missing data, propose the next "
                                            "training strategy."], "todo", "B", wide=True), unsafe_allow_html=True)

st.write("")
st.markdown("#### Known gaps in the data")
if ov["known_gaps"]:
    for g in ov["known_gaps"]:
        st.markdown(f"- {g}")
else:
    st.caption("Shown once a dataset version exists.")
