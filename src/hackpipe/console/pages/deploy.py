"""Deploy page — placeholder until branch `ui/deploy`; the spec is docs/ui/04_deploy.md."""
import streamlit as st

from hackpipe.common.config import REPO_ROOT

st.markdown("### Deploy")
st.info("Not built yet (Gates · quality statement · deploy / rollback · Hub publishing). Planned in branch `ui/deploy` after the spec is reviewed.")
spec = REPO_ROOT / "docs" / "ui" / "04_deploy.md"
with st.expander("UI spec: docs/ui/04_deploy.md"):
    st.markdown(spec.read_text() if spec.exists() else "spec missing")
