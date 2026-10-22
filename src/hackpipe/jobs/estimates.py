"""Expected job time, only from measured runs: work units / median throughput of earlier finished jobs of the
same kind (same machine preferred). Without a measurement there is no estimate — never a guess."""
from __future__ import annotations

from datetime import datetime
from statistics import median
from typing import Any

from hackpipe.common.registry import Registry

NO_ESTIMATE = "no estimate yet (first run)"


def _seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (datetime.strptime(end, fmt) - datetime.strptime(start, fmt)).total_seconds()


def throughput_history(kind: str, registry: Registry) -> list[dict[str, Any]]:
    out = []
    for j in registry.jobs(kind=kind, status="done"):
        dur = _seconds(j["started_at"], j["ended_at"])
        if j["work_units"] and dur and dur > 0:
            out.append({"machine": j["machine"], "rate": j["work_units"] / dur, "job_id": j["job_id"]})
    return out


def estimate(kind: str, work_units: float | None, registry: Registry, machine: str | None = None) -> dict[str, Any]:
    if not work_units:
        return {"seconds": None, "basis": NO_ESTIMATE}
    hist = throughput_history(kind, registry)
    same = [h for h in hist if machine and h["machine"] == machine]
    use = same or hist
    if not use:
        return {"seconds": None, "basis": NO_ESTIMATE}
    rate = median(h["rate"] for h in use)
    where = machine if same else f"{', '.join(sorted({h['machine'] for h in use}))} (other machine)"
    return {"seconds": round(work_units / rate, 1),
            "basis": f"from {len(use)} earlier run(s) on {where}: {rate:,.0f} units/s"}


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {sec:02d}s" if m else f"{sec}s")
