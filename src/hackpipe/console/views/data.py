"""Data page (R1–R3): add demonstrations, demos, episode viewer, quality, versions. Spec: docs/ui/01_data.md."""
import streamlit as st

from hackpipe.common.registry import Registry
from hackpipe.console.components import data_views as v

reg = Registry()
st.markdown("### Data")
st.caption("01 Ingest & Transform · 02 Dataset Intelligence — new demonstrations are checked once, "
           "versions build on the last approved one.")
tabs = st.tabs(["Add demonstrations", "Demos", "Episode viewer", "Quality", "Versions"])
with tabs[0]:
    v.add_tab(reg)
with tabs[1]:
    v.demos_tab(reg)
with tabs[2]:
    v.viewer_tab(reg)
with tabs[3]:
    v.quality_tab(reg)
with tabs[4]:
    v.versions_tab(reg)
