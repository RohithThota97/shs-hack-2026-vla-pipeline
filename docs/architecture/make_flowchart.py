"""Draws the black-and-white pipeline flowchart in two formats from one description:

    pipeline_flowchart.drawio   open and edit in draw.io (app.diagrams.net or the desktop app)
    pipeline_flowchart.svg      for slides; export a PNG with
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --hide-scrollbars \\
            --force-device-scale-factor=2 --window-size=1920,1080 \\
            --screenshot=docs/architecture/pipeline_flowchart.png "file://$PWD/docs/architecture/pipeline_flowchart.svg"

    python3 docs/architecture/make_flowchart.py

Shapes: pill = start / end, box = step, diamond = decision, cylinder = stored data, [ bracket ] = note.
"""
import sys
import textwrap
from html import escape as e
from pathlib import Path

W, H = 1920, 1080
OUT = Path(__file__).parent
FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
INK = "#222222"

# id: (kind, centre x, centre y, width, height, label)
NODES = {
    "src":      ("pill",     130, 300, 200, 70, "Demonstrations"),
    "fetch":    ("box",      400, 300, 190, 84, "Discover & fetch"),
    "ingest":   ("box",      660, 300, 190, 84, "R1 · Ingest"),
    "qc":       ("box",      920, 300, 190, 84, "R2 · Quality checks"),
    "valid":    ("diamond", 1170, 300, 170, 120, "VALID?"),
    "version":  ("box",     1430, 300, 190, 84, "R3 · Dataset version ds-vN"),
    "approved": ("diamond", 1720, 300, 190, 130, "Human approval?"),
    "train":    ("box",     1720, 760, 190, 84, "R4 · Train m-vN"),
    "eval":     ("box",     1430, 760, 190, 84, "R5 · Evaluate"),
    "gates":    ("diamond", 1130, 760, 190, 130, "R6 · Gates pass?"),
    "deploy":   ("box",      790, 760, 190, 84, "R10 · Deploy"),
    "out":      ("pill",     500, 760, 200, 70, "Deployed policy"),
    "improve":  ("box",     1130, 960, 190, 84, "R11 · Improve"),
    "excluded": ("box",     1170, 520, 190, 64, "Manual review"),
    "drive":    ("cylinder", 400, 100, 190, 110, "Google Drive\n15 GB"),
    "registry": ("cylinder", 660, 540, 190, 120, "Registry (SQLite)"),
    "suite":    ("cylinder", 1430, 540, 170, 120, "Frozen test suite"),
    "hub":      ("cylinder", 790, 960, 170, 110, "Hugging Face Hub"),
}

# (from, side[@fraction], to, side[@fraction], label, style, waypoints)   style: main | thin | dashed
EDGES = [
    ("src", "right", "fetch", "left", "", "main", []),
    ("fetch", "right", "ingest", "left", "", "main", []),
    ("ingest", "right", "qc", "left", "", "main", []),
    ("qc", "right", "valid", "left", "", "main", []),
    ("valid", "right", "version", "left", "yes", "main", []),
    ("valid", "bottom", "excluded", "top", "no", "thin", []),
    ("excluded", "right", "version", "left@0.7", "accepted", "thin", [(1300, 520), (1300, 316.8)]),
    ("version", "right", "approved", "left", "", "main", []),
    ("approved", "bottom", "train", "top", "yes", "main", []),
    ("train", "left", "eval", "right", "", "main", []),
    ("eval", "left", "gates", "right", "", "main", []),
    ("gates", "left", "deploy", "right", "deploy / supervised", "main", []),
    ("deploy", "left", "out", "right", "", "main", []),
    ("gates", "bottom", "improve", "top", "do not deploy", "thin", []),
    ("improve", "bottom", "src", "bottom", "next round of demos", "dashed", [(1130, 1050), (130, 1050)]),
    ("drive", "bottom@0.35", "fetch", "top@0.35", "", "thin", []),
    ("fetch", "top@0.65", "drive", "bottom@0.65", "", "thin", []),
    ("ingest", "bottom@0.35", "registry", "top@0.35", "", "thin", []),
    ("registry", "top@0.65", "ingest", "bottom@0.65", "", "thin", []),
    ("version", "bottom", "suite", "top", "freeze test split", "thin", []),
    ("suite", "bottom", "eval", "top", "same suite for every model", "thin", []),
    ("deploy", "bottom", "hub", "top", "manual publish: model + R7 card", "dashed", []),
]

# (text, centre x, centre y, leader to (x, y) or None)
NOTES = [
    ("Drive · HF · GitHub · S3 / GCS / Azure · zip upload ≤ 5 GB", 400, 430, (400, 342)),
    ("stable demo IDs · checksums", 660, 190, (660, 258)),
    ("NaN · gaps · video · length 50–83 s", 920, 140, (920, 258)),
    ("split by episode · leakage guard", 1430, 190, (1430, 258)),
    ("no → stays draft", 1720, 150, (1720, 235)),
    ("records every step", 950, 560, (755, 560)),
    ("INVALID / DUPLICATE → excluded", 1170, 630, (1170, 552)),
    ("SmolVLA · Colab T4 · run record", 1720, 870, (1720, 802)),
    ("simulator · 95% CI · success judge", 1430, 870, (1430, 802)),
    ("smoke test · rollback", 790, 650, (790, 718)),
    ("weak scenarios → targeted demos → ds-v(N+1)", 1450, 990, (1225, 960)),
]


def port(nid, spec):
    side, _, frac = spec.partition("@")
    f = float(frac) if frac else 0.5
    _, cx, cy, w, h, _ = NODES[nid]
    x0, y0 = cx - w / 2, cy - h / 2
    fx, fy = {"left": (0, f), "right": (1, f), "top": (f, 0), "bottom": (f, 1)}[side]
    return x0 + fx * w, y0 + fy * h, fx, fy


def text_w(s, size):
    return len(s) * size * 0.53


# ---------- SVG ----------
def svg():
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="{FONT}" fill="none" stroke="{INK}">',
         '<defs><marker id="a" viewBox="0 0 10 10" refX="9.5" refY="5" markerWidth="8" markerHeight="8" '
         f'orient="auto"><path d="M0 0L10 5L0 10z" fill="{INK}" stroke="none"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="#ffffff" stroke="none"/>']
    for a, sa, b, sb, label, style, via in EDGES:
        x1, y1, *_ = port(a, sa)
        x2, y2, *_ = port(b, sb)
        pts = [(x1, y1), *via, (x2, y2)]
        width = {"main": 2.6, "thin": 1.4, "dashed": 1.6}[style]
        dash = ' stroke-dasharray="7 5"' if style == "dashed" else ""
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts))
        o.append(f'<path d="{d}" stroke-width="{width}"{dash} marker-end="url(#a)"/>')
        if label:
            (px, py), (qx, qy) = max(zip(pts, pts[1:]), key=lambda s: abs(s[1][0] - s[0][0]) + abs(s[1][1] - s[0][1]))
            mx, my = (px + qx) / 2, (py + qy) / 2
            if py == qy:      # horizontal: label above the line
                o.append(f'<text x="{mx}" y="{my - 10}" font-size="14.5" fill="{INK}" stroke="none" '
                         f'text-anchor="middle">{e(label)}</text>')
            else:             # vertical: label to the right of the line
                o.append(f'<text x="{mx + 10}" y="{my + 5}" font-size="15" fill="{INK}" stroke="none">{e(label)}</text>')
    for kind, cx, cy, w, h, label in NODES.values():
        x, y = cx - w / 2, cy - h / 2
        if kind == "box":
            o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="#fff" stroke-width="2.2"/>')
        elif kind == "pill":
            o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" fill="#fff" stroke-width="2.2"/>')
        elif kind == "diamond":
            o.append(f'<polygon points="{cx},{y} {x + w},{cy} {cx},{y + h} {x},{cy}" fill="#fff" stroke-width="2.2"/>')
        else:
            r = 16
            o.append(f'<path d="M{x} {y + r} V{y + h - r} A{w / 2} {r} 0 0 0 {x + w} {y + h - r} V{y + r}" '
                     f'fill="#fff" stroke-width="2.2"/>')
            o.append(f'<ellipse cx="{cx}" cy="{y + r}" rx="{w / 2}" ry="{r}" fill="#fff" stroke-width="2.2"/>')
            cy += r / 2
        lines = wrap(label, 18 if kind != "diamond" else 12)
        for i, line in enumerate(lines):
            ly = cy + (i - (len(lines) - 1) / 2) * 21 + 6
            o.append(f'<text x="{cx}" y="{ly}" font-size="17" font-weight="600" fill="{INK}" stroke="none" '
                     f'text-anchor="middle">{e(line)}</text>')
    for text, cx, cy, lead in NOTES:
        tw = text_w(text, 15.5) + 26
        x0, x1 = cx - tw / 2, cx + tw / 2
        if lead:
            lx, ly = lead
            sx, sy = (cx, cy + 16 if ly > cy else cy - 16) if abs(ly - cy) > abs(lx - cx) else \
                     (x0 if lx < cx else x1, cy)
            o.append(f'<path d="M{sx} {sy} L{lx} {ly}" stroke-width="1.3" stroke-dasharray="4 4"/>')
        for bx, s in ((x0, 1), (x1, -1)):
            o.append(f'<path d="M{bx + 7 * s} {cy - 16} H{bx} V{cy + 16} H{bx + 7 * s}" stroke-width="1.6" '
                     f'fill="#fff"/>')
        o.append(f'<text x="{cx}" y="{cy + 5.5}" font-size="15.5" font-weight="500" fill="{INK}" stroke="none" '
                 f'text-anchor="middle">{e(text)}</text>')
    o.append("</svg>")
    return "\n".join(o)


def wrap(s, n):
    return [ln for part in s.split("\n") for ln in textwrap.wrap(part, n)]


# ---------- draw.io ----------
STYLE = {
    "box": "rounded=0;whiteSpace=wrap;html=1;strokeWidth=2;fontStyle=1;fontSize=16;",
    "pill": "rounded=1;arcSize=50;whiteSpace=wrap;html=1;strokeWidth=2;fontStyle=1;fontSize=16;",
    "diamond": "rhombus;whiteSpace=wrap;html=1;strokeWidth=2;fontStyle=1;fontSize=16;",
    "cylinder": "shape=cylinder3;boundedLbl=1;backgroundOutline=1;size=15;whiteSpace=wrap;html=1;strokeWidth=2;"
                "fontStyle=1;fontSize=16;",
}
EDGE_STYLE = {
    "main": "endArrow=block;endFill=1;strokeWidth=2.5;",
    "thin": "endArrow=block;endFill=1;strokeWidth=1.3;",
    "dashed": "endArrow=block;endFill=1;strokeWidth=1.5;dashed=1;",
}
BASE = "fillColor=#ffffff;strokeColor=#222222;fontColor=#222222;fontFamily=Helvetica;"


def html_value(text):
    """draw.io html=1 labels are HTML inside an XML attribute: escape for HTML, then for XML."""
    return e(e(text).replace("\n", "<br>"))


def drawio():
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    for nid, (kind, cx, cy, w, h, label) in NODES.items():
        cells.append(f'<mxCell id="{nid}" value="{html_value(label)}" style="{STYLE[kind]}{BASE}" vertex="1" parent="1">'
                     f'<mxGeometry x="{cx - w / 2}" y="{cy - h / 2}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    for i, (a, sa, b, sb, label, style, via) in enumerate(EDGES):
        *_, ex, ey = port(a, sa)
        *_, nx, ny = port(b, sb)
        pts = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in via)
        cells.append(f'<mxCell id="e{i}" value="{html_value(label)}" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;'
                     f'exitX={ex};exitY={ey};entryX={nx};entryY={ny};{EDGE_STYLE[style]}{BASE}fontSize=14;'
                     f'labelBackgroundColor=#ffffff;" edge="1" parent="1" source="{a}" target="{b}">'
                     f'<mxGeometry relative="1" as="geometry"><Array as="points">{pts}</Array></mxGeometry></mxCell>')
    for i, (text, cx, cy, lead) in enumerate(NOTES):
        tw = text_w(text, 15.5) + 28
        cells.append(f'<mxCell id="n{i}" value="{html_value(text)}" style="shape=partialRectangle;top=0;bottom=0;'
                     f'whiteSpace=wrap;html=1;strokeWidth=1.5;fontSize=14;{BASE}" vertex="1" parent="1">'
                     f'<mxGeometry x="{cx - tw / 2}" y="{cy - 16}" width="{tw}" height="32" as="geometry"/></mxCell>')
        if lead:
            cells.append(f'<mxCell id="l{i}" style="endArrow=none;dashed=1;strokeWidth=1.2;{BASE}" edge="1" '
                         f'parent="1" source="n{i}"><mxGeometry relative="1" as="geometry">'
                         f'<mxPoint x="{lead[0]}" y="{lead[1]}" as="targetPoint"/></mxGeometry></mxCell>')
    return ('<mxfile host="hackpipe"><diagram name="Pipeline flowchart" id="pipeline">'
            f'<mxGraphModel dx="{W}" dy="{H}" grid="1" gridSize="10" page="1" pageWidth="{W}" pageHeight="{H}">'
            f'<root>{"".join(cells)}</root></mxGraphModel></diagram></mxfile>\n')


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    (out / "pipeline_flowchart.svg").write_text(svg(), encoding="utf-8")
    (out / "pipeline_flowchart.drawio").write_text(drawio(), encoding="utf-8")
