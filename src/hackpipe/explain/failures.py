"""R8 — Rule-based failure categories and one-line summaries from logged episode fields."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hackpipe.evaluation.schema import EpisodeResult

CATEGORIES = ["success", "safety_stop", "collision", "wrong_object", "grasp_failed", "object_dropped",
              "placement_off_target", "timeout", "unknown"]


def categorize(r: "EpisodeResult") -> str:
    """First matching rule wins (most severe first)."""
    if r.success:
        return "success"
    if r.safety_stop:
        return "safety_stop"
    if r.collision:
        return "collision"
    if r.correct_object is False:
        return "wrong_object"
    if r.grasp_attempts > 0 and r.grasp_successes == 0:
        return "grasp_failed"
    if r.dropped:
        return "object_dropped"
    if r.in_target_zone is False:
        return "placement_off_target"
    if r.timeout:
        return "timeout"
    return "unknown"


def summary(r: "EpisodeResult") -> str:
    cat = categorize(r)
    where = f" during {r.failure_stage}" if r.failure_stage else ""
    if cat == "success":
        t = f" in {r.completion_time_s:.1f}s" if r.completion_time_s is not None else ""
        err = f", placement error {r.placement_error_m * 100:.1f} cm" if r.placement_error_m is not None else ""
        text = f"Succeeded{t}{err}."
    else:
        text = f"Failed ({cat.replace('_', ' ')}){where}; stop reason: {r.stop_reason}."
    if r.interventions:
        text += f" {r.interventions} human intervention(s)."
    if r.fake:
        text += " [FAKE — dummy adapter]"
    return f"{r.scenario_id} seed {r.seed}: {text}"
