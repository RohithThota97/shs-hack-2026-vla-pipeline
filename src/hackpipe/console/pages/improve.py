"""Improve page — placeholder until branch `ui/improve`; the spec is docs/ui/05_improve.md."""
import streamlit as st

from hackpipe.common.config import REPO_ROOT

st.markdown("### Improve")
st.info("Not built yet (Capability gaps · next data · next training strategy). Planned in branch `ui/improve` after the spec is reviewed.")
spec = REPO_ROOT / "docs" / "ui" / "05_improve.md"
with st.expander("UI spec: docs/ui/05_improve.md"):
    st.markdown(spec.read_text() if spec.exists() else "spec missing")
