"""Training metrics: loss vs step from a lerobot-train log → loss.csv (+ loss.png if matplotlib is installed)."""
from __future__ import annotations

import re
from pathlib import Path

_STEP = re.compile(r"\bstep:\s*([0-9.]+)([KMB]?)\b")
_LOSS = re.compile(r"\bloss:\s*([-+0-9.eE]+)")
_MULT = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


def parse_log(text: str) -> list[tuple[int, float]]:
    """(step, loss) from lerobot-train log lines like `step:2K smpl:16K ... loss:0.123 grdn:...`.
    LeRobot abbreviates big steps (2K), so steps are as precise as the log prints them."""
    out = []
    for line in text.splitlines():
        s, l = _STEP.search(line), _LOSS.search(line)
        if s and l:
            out.append((int(round(float(s.group(1)) * _MULT[s.group(2)])), float(l.group(1))))
    return out


def export(log_path: str | Path, out_dir: str | Path) -> dict[str, str | int | float | None]:
    rows = parse_log(Path(log_path).read_text(errors="replace"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "loss.csv").write_text("step,loss\n" + "".join(f"{s},{l}\n" for s, l in rows))
    result: dict[str, str | int | float | None] = {"points": len(rows), "csv": str(out / "loss.csv"),
                                                   "final_loss": rows[-1][1] if rows else None, "png": None}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return result
    if rows:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot([s for s, _ in rows], [l for _, l in rows], linewidth=1.2)
        ax.set_xlabel("step")
        ax.set_ylabel("training loss")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "loss.png", dpi=120)
        plt.close(fig)
        result["png"] = str(out / "loss.png")
    return result
