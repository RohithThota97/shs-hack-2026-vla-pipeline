"""Numbers for the Train page, all measured from lerobot-train logs and job records — never guessed.

timing_summary(): seconds per step (from log timestamps), peak GPU memory (LeRobot's mem_gb), data-loading share
(data_s vs updt_s), start-up time. budget(): steps that fit a time slot. live(): step, loss and time left of a run.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from hackpipe.common.registry import Registry
from hackpipe.dataset.versioning import load_manifest, trainable_sources
from hackpipe.training.loss import parse_metrics


def job_metrics(job: dict[str, Any], log_freq: int | None = None) -> list[dict[str, float]]:
    p = Path(job.get("log_path") or "")
    return parse_metrics(p.read_text(errors="replace"), log_freq) if p.exists() else []


def seconds_per_step(rows: list[dict[str, float]]) -> float | None:
    """Wall time between log lines ÷ steps between them; the first line (warm-up) is skipped when possible."""
    timed = [r for r in rows if "t" in r]
    if len(timed) >= 3:
        timed = timed[1:]
    if len(timed) < 2 or timed[-1]["step"] <= timed[0]["step"]:
        return None
    return (timed[-1]["t"] - timed[0]["t"]) / (timed[-1]["step"] - timed[0]["step"])


def _utc(ts: str) -> float:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def timing_summary(job: dict[str, Any], log_freq: int | None = None) -> dict[str, Any]:
    rows = job_metrics(job, log_freq)
    sps = seconds_per_step(rows)
    upd = [r["updt_s"] for r in rows if "updt_s" in r]
    dat = [r["data_s"] for r in rows if "data_s" in r]
    mem = [r["mem_gb"] for r in rows if "mem_gb" in r]
    startup = None
    timed = [r for r in rows if "t" in r]
    if timed and job.get("started_at") and sps:
        # first log line comes after start-up + its first log_freq steps
        startup = max(0.0, timed[0]["t"] - _utc(job["started_at"]) - timed[0]["step"] * sps)
    return {"job_id": job["job_id"], "status": job["status"], "log_lines": len(rows),
            "last_step": int(rows[-1]["step"]) if rows else None, "last_loss": rows[-1]["loss"] if rows else None,
            "seconds_per_step": round(sps, 4) if sps else None,
            "peak_mem_gb": max(mem) if mem else None,
            "data_share": round(mean(dat) / (mean(dat) + mean(upd)), 3) if dat and upd and mean(dat) + mean(upd) else None,
            "startup_s": round(startup, 1) if startup is not None else None}


def budget(slot_hours: float, seconds_per_step: float, save_freq: int, startup_s: float = 0.0) -> int:
    """Steps that fit the slot, rounded down to a whole number of checkpoints."""
    usable = slot_hours * 3600 - (startup_s or 0.0)
    if usable <= 0 or seconds_per_step <= 0:
        return 0
    return int(usable / seconds_per_step) // save_freq * save_freq


def expected_seconds(steps: int, seconds_per_step: float, startup_s: float | None) -> float:
    return (startup_s or 0.0) + steps * seconds_per_step


def train_frames(dataset_version: str, registry: Registry) -> int:
    """Frames training reads: train demos of the trainable sources (a data-only subtask is not trained)."""
    m = load_manifest(dataset_version, registry)
    ids, used = set(m["demo_ids"]["train"]), set(trainable_sources(m)[0])
    return sum(int(d["meta"].get("n_frames", 0)) for d in registry.demos()
               if d["demo_id"] in ids and (not used or d["source"] in used))   # old manifests: no sources


def curriculum_frames(dataset_version: str, registry: Registry, rate_hz: float) -> int:
    """Sample rows the curriculum trainer reads: train demos of every labelled source on the rate_hz grid, flagged
    episodes left out (each row is used for a sub-task and a complete-task sample)."""
    import math
    m = load_manifest(dataset_version, registry)
    sub = (m.get("subtasks") or {}).get("sources") or {}
    ids = set(m["demo_ids"]["train"])
    total = 0
    for d in registry.demos():
        info = sub.get(d["source"])
        if d["demo_id"] in ids and info and d["episode_index"] not in set(info["flagged"]):
            stride = max(1, round(float(d["meta"].get("fps") or info["fps"]) / rate_hz))
            total += math.ceil(int(d["meta"].get("n_frames", 0)) / stride)
    return total


def epochs(steps: int, batch_size: int, frames: int) -> float | None:
    return round(steps * batch_size / frames, 2) if frames else None


def live(job: dict[str, Any], target_steps: int | None, log_freq: int | None = None) -> dict[str, Any]:
    rows = job_metrics(job, log_freq)
    if not rows:
        return {"step": None, "loss": None, "eta_s": None, "seconds_per_step": None, "mem_gb": None}
    recent = rows[-6:]
    sps = seconds_per_step(recent) or seconds_per_step(rows)
    step = int(rows[-1]["step"])
    return {"step": step, "loss": rows[-1]["loss"], "mem_gb": rows[-1].get("mem_gb"),
            "seconds_per_step": round(sps, 4) if sps else None,
            "eta_s": round((target_steps - step) * sps, 1) if sps and target_steps and target_steps > step else None}


def run_record_diff(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    """Fields that differ between two run records (flattened). For v1 vs v2, only dataset fields should."""
    skip = {"model_version", "started_at", "ended_at", "duration_s", "exit_code", "output_dir", "environment",
            "command"}

    def flat(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in d.items():
            if prefix == "" and k in skip:
                continue
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(flat(v, key + "."))
            else:
                out[key] = json.dumps(v, sort_keys=True, default=str)
        return out

    fa, fb = flat(a), flat(b)
    return [{"field": k, "a": fa.get(k), "b": fb.get(k), "dataset_field": k.startswith("dataset")}
            for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
