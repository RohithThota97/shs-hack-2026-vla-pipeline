"""Draws docs/architecture/pipeline_architecture.svg (16:9, for slides). Edit the text below and re-run.
The "Now:" card and the footer date are a snapshot — update them when a new dataset version is approved.


    python3 docs/architecture/make_diagram.py
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --hide-scrollbars \\
        --force-device-scale-factor=2 --window-size=1920,1080 \\
        --screenshot=docs/architecture/pipeline_architecture.png "file://$PWD/docs/architecture/pipeline_architecture.svg"
"""
import sys
import textwrap
from pathlib import Path
from html import escape as e
W, H = 1920, 1080
FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"
INK, MUTED, LINE = "#1f2937", "#4b5563", "#9ca3af"
cols = [
 ("1 · Sources", "auto-discovered", "#475569", "#f1f5f9", [
   ("Organiser download", "data/C2_Teach-the-robot_VLA (folders or zips)"),
   ("Google Drive", "Colab mount · Drive for desktop · shared link"),
   ("Upload / local folder", "zip up to 5 GB; all sources de-duplicated → data/raw/<name>"),
   ("Hugging Face · GitHub", "your own token · pinned revision"),
   ("Cloud bucket", "S3 · GCS · Azure — keys only as job env vars"),
 ]),
 ("2 · Data", "Person 1", "#1d4ed8", "#eff6ff", [
   ("R1 · Ingest", "LeRobot v3 → demo records, stable IDs, checksums"),
   ("R2 · Quality checks", "NaN, gaps, video decode, length ≥ 50 s → VALID / REVIEW / INVALID"),
   ("R3 · Dataset version ds-vN", "split by episode (fixed seed), leakage guard, frozen test, fingerprint, card"),
   ("Human approval", "draft → approved; training refuses drafts"),
   ("Now: ds-v2 approved", "handover 76/8 · rand500 441/49 (data only) · test lighting10 (10)"),
 ]),
 ("3 · Training", "Person 2", "#6d28d9", "#f5f3ff", [
   ("Approved ds-vN", "trainable sources only; other-schema subtasks stay data only"),
   ("R4 · lerobot-train", "SmolVLA (smolvla_base) on a Colab T4, batch 8"),
   ("Dry run → timing → full run", "steps set from the measured seconds per step"),
   ("Run record", "command, config, dataset fingerprint, git commit, packages, GPU"),
   ("Model m-vN", "checkpoints on Drive (last 2 kept) + loss curve"),
 ]),
 ("4 · Evaluation", "Person 3 + 4", "#0f766e", "#f0fdfa", [
   ("Test suite test-vN", "frozen; the same scenarios and seeds for every model"),
   ("Simulator adapter", "organisers' 3D sim (Isaac Sim), safe resets, stop rules"),
   ("R5 · Metrics", "success rate with 95% CI, collisions, safety stops, latency"),
   ("Option A · Success judge", "success and progress from state / vision traces"),
   ("R8 · Explain failures", "failure category + one-line summary per episode"),
 ]),
 ("5 · Release", "Person 5", "#b45309", "#fffbeb", [
   ("R6 · Quality gates", "data · training · eval → deploy / supervised / do not deploy"),
   ("R7 · Model card", "results with CIs, failures, limitations, comparison"),
   ("R10 · Deploy", "smoke test, deployment record, one-step rollback"),
   ("Hub publish", "model + card only (private) — the dataset never leaves Drive"),
 ]),
]
out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">',
 '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#6b7280"/></marker></defs>',
 f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
 f'<text x="40" y="58" font-size="34" font-weight="700" fill="{INK}">hackpipe — demo-to-deployment pipeline</text>',
 f'<text x="40" y="90" font-size="18" fill="{MUTED}">FR3 duo handover · LeRobot v3 data · SmolVLA policy · every step versioned, checked and recorded</text>']
X0, GAP, CW = 40, 40, 336
TOP, CARD_H, CARD_GAP, CARD_Y0 = 112, 104, 14, 176
BOTTOM = 770


def lines(text, width, max_lines):
    """Wrapped description lines; stops with an error instead of letting text run out of its box."""
    out = textwrap.wrap(text, width)
    if len(out) > max_lines:
        sys.exit(f"too long for its box ({len(out)} lines > {max_lines}), shorten it: {text!r}")
    return out


for i, (name, owner, c, bg, cards) in enumerate(cols):
    x = X0 + i * (CW + GAP)
    out.append(f'<rect x="{x}" y="{TOP}" width="{CW}" height="{BOTTOM-TOP}" rx="16" fill="{bg}" stroke="{c}" stroke-opacity=".35"/>')
    out.append(f'<rect x="{x}" y="{TOP}" width="{CW}" height="48" rx="16" fill="{c}"/><rect x="{x}" y="{TOP+30}" width="{CW}" height="18" fill="{c}"/>')
    out.append(f'<text x="{x+18}" y="{TOP+32}" font-size="21" font-weight="700" fill="#fff">{e(name)}</text>')
    out.append(f'<text x="{x+CW-18}" y="{TOP+31}" font-size="14" fill="#fff" fill-opacity=".9" text-anchor="end">{e(owner)}</text>')
    for j, (t, d) in enumerate(cards):
        y = CARD_Y0 + j * (CARD_H + CARD_GAP)
        status = t.startswith("Now:")
        dash = 'stroke-dasharray="5 4"' if status else ""
        out.append(f'<rect x="{x+14}" y="{y}" width="{CW-28}" height="{CARD_H}" rx="10" fill="{"#fff" if not status else bg}" stroke="{c}" stroke-width="{1.6 if not status else 1.4}" {dash}/>')
        if not status:
            out.append(f'<rect x="{x+14}" y="{y}" width="6" height="{CARD_H}" rx="3" fill="{c}"/>')
        out.append(f'<text x="{x+32}" y="{y+30}" font-size="18" font-weight="700" fill="{c if status else INK}">{e(t)}</text>')
        for k, line in enumerate(lines(d, 38, 3)):
            out.append(f'<text x="{x+32}" y="{y+56+k*20}" font-size="14.5" fill="{MUTED}">{e(line)}</text>')
        if i in (1, 2, 3, 4) and j < len(cards) - 1 and not cards[j+1][0].startswith("Now:"):
            cx = x + CW / 2
            out.append(f'<line x1="{cx}" y1="{y+CARD_H+1}" x2="{cx}" y2="{y+CARD_H+CARD_GAP-1}" stroke="#6b7280" stroke-width="2" marker-end="url(#a)"/>')
    if i < len(cols) - 1:
        ay = 444
        out.append(f'<line x1="{x+CW+2}" y1="{ay}" x2="{x+CW+GAP-3}" y2="{ay}" stroke="#374151" stroke-width="3" marker-end="url(#a)"/>')
# R11 loop: Evaluation and Release feed back into Data; the label sits under the line, not on it
ly = 804
x_rel = X0 + 4 * (CW + GAP) + CW / 2; x_ev = X0 + 3 * (CW + GAP) + CW / 2; x_data = X0 + 1 * (CW + GAP) + CW / 2
out.append(f'<path d="M{x_rel} {BOTTOM} V{ly} H{x_data} V{BOTTOM+2}" fill="none" stroke="#dc2626" stroke-width="3" stroke-dasharray="9 6" marker-end="url(#a)"/>')
out.append(f'<path d="M{x_ev} {BOTTOM} V{ly}" fill="none" stroke="#dc2626" stroke-width="3" stroke-dasharray="9 6"/>')
lab = "R11 · Option B self-improving loop: weak scenario categories → ~20 targeted demos each → ds-v(N+1) → m-v(N+1) → same test suite"
out.append(f'<rect x="{(x_data+x_ev)/2-500}" y="{ly+12}" width="1000" height="30" rx="15" fill="#fef2f2" stroke="#dc2626"/>')
out.append(f'<text x="{(x_data+x_ev)/2}" y="{ly+33}" font-size="15.5" font-weight="600" fill="#b91c1c" text-anchor="middle">{e(lab)}</text>')
# platform band
by = 870
out.append(f'<text x="40" y="{by+6}" font-size="15" font-weight="700" fill="{MUTED}" letter-spacing="1.5">SHARED PLATFORM — USED BY EVERY STAGE</text>')
plat = [("SQLite registry", "demos · datasets · models · evals · deployments; one writer at a time"),
        ("Background job runner", "every long step runs as a job with its own log; secrets only as env vars"),
        ("Web console + hackpipe CLI", "Streamlit pages: Data · Train · Evaluate · Deploy · Improve"),
        ("Storage: Google Drive (15 GB)", "datasets as zips · last 2 checkpoints · no datasets on HF / Kaggle / GitHub")]
PW = (W - 80 - 3 * 24) / 4
for k, (t, d) in enumerate(plat):
    x = 40 + k * (PW + 24)
    out.append(f'<rect x="{x}" y="{by+22}" width="{PW}" height="112" rx="12" fill="#f9fafb" stroke="#374151" stroke-width="1.5"/>')
    out.append(f'<text x="{x+20}" y="{by+56}" font-size="18" font-weight="700" fill="{INK}">{e(t)}</text>')
    for m, line in enumerate(lines(d, 52, 2)):
        out.append(f'<text x="{x+20}" y="{by+84+m*21}" font-size="14.5" fill="{MUTED}">{e(line)}</text>')
out.append(f'<text x="{W-40}" y="{H-22}" font-size="13" fill="{LINE}" text-anchor="end">23 Oct 2026 · SHS Hackathon 2026 · CathCrew</text>')
out.append('</svg>')
Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("pipeline_architecture.svg")).write_text(
    "\n".join(out), encoding="utf-8")
