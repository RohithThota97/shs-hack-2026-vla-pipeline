"""Look and feel: Siemens Healthineers-like (orange accent, greys, white cards), free Inter font, no logos.
Tokens are defined once here; pages use the CSS classes below instead of inline colours."""
from __future__ import annotations

import html

import streamlit as st

TOKENS = {
    "accent": "#EC6602",      # orange accent (approximation, not an official brand file)
    "accent_soft": "#FDEBDD",
    "ink": "#1A1A1A",
    "muted": "#6B6B6B",
    "line": "#E3E3E3",
    "panel": "#F6F6F6",
    "ok": "#2E7D32",
    "warn": "#B26A00",
    "bad": "#C62828",
    "info": "#00677F",
}

STATUS_COLOURS = {"done": "ok", "active": "info", "running": "info", "queued": "muted", "waiting": "warn",
                  "todo": "muted", "failed": "bad", "draft": "warn", "approved": "ok", "VALID": "ok",
                  "REVIEW": "warn", "INVALID": "bad", "DUPLICATE": "muted", "imported": "muted", "trained": "ok",
                  "training": "info", "evaluated": "ok", "deployed": "ok", "rolled_back": "muted",
                  "READY TO DEPLOY": "ok", "NEEDS MORE TRAINING": "bad", "READY FOR SIM TEST": "info",
                  "NOT TESTED": "muted", "best": "ok",
                  "pass": "ok", "FAIL": "bad", "warn": "warn", "unverified": "warn", "skipped": "muted",
                  "fail": "bad"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 3.2rem; }
.hp-header { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
  border-bottom: 3px solid %(accent)s; padding: .4rem 0 .7rem 0; margin-bottom: 1rem; }
.hp-title { font-size: 1.35rem; font-weight: 700; color: %(ink)s; }
.hp-title small { display:block; font-size:.8rem; font-weight:500; color:%(muted)s; }
.hp-chips { display:flex; gap:.4rem; flex-wrap:wrap; }
.hp-chip { border:1px solid %(line)s; border-radius: 999px; padding:.15rem .65rem; font-size:.78rem;
  color:%(ink)s; background:#fff; white-space:nowrap; }
.hp-chip b { color:%(accent)s; font-weight:600; }
.hp-card { border:1px solid %(line)s; border-radius:10px; padding:.8rem .9rem; background:#fff; min-height:185px; }
.hp-card.hp-wide { min-height:0; }
.hp-card h4 { margin:0 0 .35rem 0; font-size:.95rem; color:%(ink)s; }
.hp-card .hp-id { color:%(accent)s; font-weight:700; margin-right:.35rem; }
.hp-card .hp-badge { margin-bottom:.45rem; }
.hp-card p { margin:.1rem 0; font-size:.83rem; color:%(muted)s; }
.hp-badge { display:inline-block; border-radius:6px; padding:.05rem .45rem; font-size:.72rem; font-weight:600;
  color:#fff; text-transform:uppercase; letter-spacing:.02em; }
.hp-next { border-left:4px solid %(accent)s; background:%(accent_soft)s; padding:.6rem .9rem; border-radius:6px; }
.hp-rail-title { font-weight:700; font-size:.95rem; margin-bottom:.3rem; }
.hp-muted { color:%(muted)s; font-size:.8rem; }
</style>
""" % TOKENS


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def badge(status: str) -> str:
    colour = TOKENS[STATUS_COLOURS.get(status, "muted")]
    return f'<span class="hp-badge" style="background:{colour}">{html.escape(str(status))}</span>'


def card(title: str, lines: list[str], status: str | None = None, ident: str | None = None,
         wide: bool = False) -> str:
    head = (f'<span class="hp-id">{html.escape(ident)}</span>' if ident else "") + html.escape(title)
    body = "".join(f"<p>{html.escape(str(l))}</p>" for l in lines)
    return (f'<div class="hp-card{" hp-wide" if wide else ""}"><h4>{head}</h4>'
            f'{badge(status) if status else ""}{body}</div>')
