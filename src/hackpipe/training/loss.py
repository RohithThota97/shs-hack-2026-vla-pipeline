"""Training metrics: loss vs step from a lerobot-train log → loss.csv (+ loss.png if matplotlib is installed)."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

_STEP = re.compile(r"\bstep:\s*([0-9.]+)([KMBTQ]?)(?=\s|$)")
_FIELD = {k: re.compile(rf"(?:^|\s){re.escape(k)}:\s*([-+0-9.eE]+)") for k in
          ("loss", "grdn", "lr", "updt_s", "data_s", "smp/s", "mem_gb", "epch")}
_MULT = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 10**12, "Q": 10**15}
_TIME = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def parse_metrics(text: str, log_freq: int | None = None) -> list[dict[str, float]]:
    """One dict per lerobot-train log line: step, loss, grdn, lr, updt_s, data_s, smp/s, mem_gb, epch.

    LeRobot prints big steps without decimals (step 1500 → "step:2K"). With log_freq known, those steps are
    rebuilt as previous step + log_freq (checked against the printed value); otherwise they stay as printed."""
    rows: list[dict[str, float]] = []
    prev: int | None = None
    for line in text.splitlines():
        m = _STEP.search(line)
        if not m or not _FIELD["loss"].search(line):
            continue
        mult = _MULT[m.group(2)]
        printed = float(m.group(1)) * mult
        step = int(round(printed))
        if mult > 1 and log_freq and prev is not None and abs(prev + log_freq - printed) <= mult / 2:
            step = prev + log_freq
        row: dict[str, float] = {"step": step}
        t = _TIME.search(line)
        if t:                                   # LeRobot's log timestamp (local time, 1 s resolution)
            row["t"] = datetime.strptime(t.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        for k, rx in _FIELD.items():
            f = rx.search(line)
            if f:
                row[k] = float(f.group(1))
        rows.append(row)
        prev = step
    return rows


def parse_log(text: str, log_freq: int | None = None) -> list[tuple[int, float]]:
    """(step, loss) pairs, see parse_metrics."""
    return [(int(r["step"]), r["loss"]) for r in parse_metrics(text, log_freq)]


def export(log_path: str | Path, out_dir: str | Path, log_freq: int | None = None) -> dict[str, str | int | float | None]:
    """loss.csv (step,loss) + metrics.csv (every logged field) + loss.png."""
    metrics = parse_metrics(Path(log_path).read_text(errors="replace"), log_freq)
    rows = [(int(r["step"]), r["loss"]) for r in metrics]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "loss.csv").write_text("step,loss\n" + "".join(f"{s},{l}\n" for s, l in rows))
    keys = ["step", *[k for k in _FIELD if any(k in r for r in metrics)]]
    (out / "metrics.csv").write_text(",".join(keys) + "\n" + "".join(
        ",".join("" if r.get(k) is None else f"{r[k]:g}" for k in keys) + "\n" for r in metrics))
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
