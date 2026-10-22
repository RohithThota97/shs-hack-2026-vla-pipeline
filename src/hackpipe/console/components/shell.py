"""Parts every page shares: the header (versions, machine, registry writer) and the jobs rail."""
from __future__ import annotations

import html
import time
from datetime import datetime, timezone

import streamlit as st

from hackpipe.common import writelock
from hackpipe.common.registry import Registry
from hackpipe.console.theme import badge
from hackpipe.jobs import machine, runner
from hackpipe.jobs.estimates import fmt_duration
from hackpipe.status import overview


@st.cache_data(ttl=30, show_spinner=False)
def machine_info() -> dict:
    return machine.info()


def chip(label: str, value: str | None) -> str:
    return f'<span class="hp-chip">{html.escape(label)} <b>{html.escape(value or "—")}</b></span>'


def header(reg: Registry) -> None:
    ov = overview(reg)
    v = ov["versions"]
    m = machine_info()
    w = writelock.holder()
    ds = f"{v['dataset']} ({v['dataset_status']})" if v["dataset"] else None
    chips = [chip("Dataset", ds), chip("Model", v["model"]), chip("Deployed", v["deployed"]),
             chip("Machine", m["label"]), chip("Writer", w["machine"] if w else "free")]
    st.markdown(
        '<div class="hp-header"><div class="hp-title">FR3 Duo · Demo-to-Deployment'
        "<small>SHS Hackathon 2026 · Challenge 2 — Teach the Robot</small></div>"
        f'<div class="hp-chips">{"".join(chips)}</div></div>', unsafe_allow_html=True)


def _elapsed(job: dict) -> float | None:
    if not job["started_at"]:
        return None
    start = datetime.strptime(job["started_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    end = (datetime.strptime(job["ended_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
           if job["ended_at"] else datetime.now(timezone.utc))
    return (end - start).total_seconds()


def job_box(job: dict, reg: Registry) -> None:
    el = _elapsed(job)
    st.markdown(f"**{html.escape(job['meta'].get('label', job['kind']))}** {badge(job['status'])}",
                unsafe_allow_html=True)
    st.caption(f"{job['job_id']} · {job['needs'].upper()} · {job['machine']}"
               + (" · waiting for the GPU" if job["meta"].get("waiting_for") else ""))
    est = job["estimate_s"]
    st.caption(f"expected {fmt_duration(est)} · elapsed {fmt_duration(el)}"
               + ("" if est else f" · {job['meta'].get('estimate_basis', '')}"))
    if job["status"] == "running" and est and el is not None:
        st.progress(min(el / est, 0.99), text="by expected time (not measured progress)")
    tail = runner.log_tail(job, 8)
    if tail:
        with st.expander("log"):
            st.code(tail, language=None)
    if job["status"] in ("running", "queued"):
        if st.button("Cancel", key=f"cancel-{job['job_id']}"):
            runner.cancel(job["job_id"], reg)
            st.rerun()


@st.fragment(run_every=5)
def jobs_rail() -> None:
    reg = Registry()
    runner.reconcile(reg)
    st.markdown('<div class="hp-rail-title">Jobs</div>', unsafe_allow_html=True)
    jobs = reg.jobs(limit=8)
    if not jobs:
        st.markdown('<span class="hp-muted">No jobs yet. Pipeline steps started here appear with their '
                    "expected time and machine.</span>", unsafe_allow_html=True)
    for j in jobs:
        with st.container(border=True):
            job_box(j, reg)
    writer_box()
    st.caption(f"updated {time.strftime('%H:%M:%S')}")


def writer_box() -> None:
    w = writelock.holder()
    me = writelock.machine_id()
    with st.container(border=True):
        st.markdown("**Registry writer**")
        if not w:
            st.caption(f"Free. This machine ({me}) claims it on its first write.")
        elif w["machine"] == me:
            st.caption(f"This machine ({me}) since {w['since']}.")
            if st.button("Release", key="writer-release"):
                writelock.release()
                st.rerun()
        else:
            st.caption(f"{w['machine']} since {w['since']}. This machine can read but not write.")
            sure = st.checkbox("I'm sure the other machine has stopped writing", key="writer-sure")
            if st.button("Take over", key="writer-take", disabled=not sure):
                writelock.acquire(force=True)
                st.rerun()
