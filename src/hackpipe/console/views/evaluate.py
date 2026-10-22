"""Evaluate page — placeholder until branch `ui/evaluate`; the spec is docs/ui/03_evaluate.md."""
import streamlit as st

from hackpipe.common.config import REPO_ROOT

st.markdown("### Evaluate")
st.info("Not built yet (Test suite runs · success with confidence intervals · failures). Planned in branch `ui/evaluate` after the spec is reviewed.")
spec = REPO_ROOT / "docs" / "ui" / "03_evaluate.md"
with st.expander("UI spec: docs/ui/03_evaluate.md"):
    st.markdown(spec.read_text() if spec.exists() else "spec missing")
