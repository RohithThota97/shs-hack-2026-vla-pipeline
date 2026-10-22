"""Train page — placeholder until branch `ui/train`; the spec is docs/ui/02_train.md."""
import streamlit as st

from hackpipe.common.config import REPO_ROOT

st.markdown("### Train")
st.info("Not built yet (Dry run · timing run · step budget · training jobs · loss curves). Planned in branch `ui/train` after the spec is reviewed.")
spec = REPO_ROOT / "docs" / "ui" / "02_train.md"
with st.expander("UI spec: docs/ui/02_train.md"):
    st.markdown(spec.read_text() if spec.exists() else "spec missing")
