"""Data page — placeholder until branch `ui/data`; the spec is docs/ui/01_data.md."""
import streamlit as st

from hackpipe.common.config import REPO_ROOT

st.markdown("### Data")
st.info("Not built yet (Add demonstrations · demos · episode viewer · quality · versions). Planned in branch `ui/data` after the spec is reviewed.")
spec = REPO_ROOT / "docs" / "ui" / "01_data.md"
with st.expander("UI spec: docs/ui/01_data.md"):
    st.markdown(spec.read_text() if spec.exists() else "spec missing")
