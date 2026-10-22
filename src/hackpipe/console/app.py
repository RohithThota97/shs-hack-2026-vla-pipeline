"""Console entry point: `hackpipe console` (or `streamlit run src/hackpipe/console/app.py` from the repo root)."""
from __future__ import annotations

import hmac
import os
from pathlib import Path

import streamlit as st

from hackpipe.common.registry import Registry
from hackpipe.console import theme
from hackpipe.console.components.shell import header, jobs_rail

PAGES = Path(__file__).parent / "views"   # not "pages/": Streamlit would auto-discover that folder

st.set_page_config(page_title="FR3 Duo · Demo-to-Deployment", layout="wide")
theme.apply()

# Optional access token (set when the console is exposed through a public tunnel, e.g. from Colab).
TOKEN = os.environ.get("HACKPIPE_CONSOLE_TOKEN")
if TOKEN and not st.session_state.get("authed"):
    given = st.query_params.get("token") or st.text_input("Access token", type="password")
    if not hmac.compare_digest(str(given or ""), TOKEN):
        st.info("This console is protected. Open the link printed by the notebook, or enter the token.")
        st.stop()
    st.session_state["authed"] = True

nav = st.navigation({
    "Pipeline": [st.Page(PAGES / "overview.py", title="Overview", icon=":material/dashboard:", default=True)],
    "Stages": [
        st.Page(PAGES / "data.py", title="Data", icon=":material/database:"),
        st.Page(PAGES / "train.py", title="Train", icon=":material/model_training:"),
        st.Page(PAGES / "evaluate.py", title="Evaluate", icon=":material/fact_check:"),
        st.Page(PAGES / "deploy.py", title="Deploy", icon=":material/rocket_launch:"),
        st.Page(PAGES / "improve.py", title="Improve", icon=":material/autorenew:"),
    ],
})

reg = Registry()
header(reg)
main, rail = st.columns([3.3, 1], gap="large")
with main:
    nav.run()
with rail:
    jobs_rail()
