"""R5 / Phase 8 — Metrics from episode results. Unmeasured fields (None) are excluded, never guessed."""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from hackpipe.common.stats import percentile, rate, wilson_interval
from hackpipe.evaluation.schema import EpisodeResult
from hackpipe.explain.failures import categorize


def _success_block(rs: list[EpisodeResult]) -> dict[str, Any]:
    k, n = sum(r.success for r in rs), len(rs)
    lo, hi = wilson_interval(k, n)
    return {"n": n, "successes": k, "rate": rate(k, n), "ci95": [round(lo, 4), round(hi, 4)]}


def compute_metrics(results: list[EpisodeResult]) -> dict[str, Any]:
    n = len(results)
    with_grasp = [r for r in results if r.grasp_successes > 0]
    obj_known = [r for r in results if r.correct_object is not None]
    zone_known = [r for r in results if r.in_target_zone is not None]
    errors = [r.placement_error_m for r in results if r.placement_error_m is not None]
    times = [r.completion_time_s for r in results if r.success and r.completion_time_s is not None]
    lat = [x for r in results for x in r.inference_latency_ms]

    by_scenario: dict[str, list[EpisodeResult]] = defaultdict(list)
    by_category: dict[str, list[EpisodeResult]] = defaultdict(list)
    for r in results:
        by_scenario[r.scenario_id].append(r)
        by_category[r.category].append(r)

    return {
        "n_runs": n,
        "fake_results": any(r.fake for r in results),
        "task_success": _success_block(results),
        "correct_object_rate": rate(sum(bool(r.correct_object) for r in obj_known), len(obj_known)),
        "grasp_success_rate": rate(sum(r.grasp_successes for r in results), sum(r.grasp_attempts for r in results)),
        "object_drop_rate": rate(sum(r.dropped for r in with_grasp), len(with_grasp)),
        "placement_in_zone_rate": rate(sum(bool(r.in_target_zone) for r in zone_known), len(zone_known)),
        "placement_error_m_mean": mean(errors) if errors else None,
        "collision_rate": rate(sum(r.collision for r in results), n),
        "timeout_rate": rate(sum(r.timeout for r in results), n),
        "safety_stop_rate": rate(sum(r.safety_stop for r in results), n),
        "completion_time_s": {"mean": mean(times) if times else None, "median": median(times) if times else None},
        "interventions_per_episode": rate(sum(r.interventions for r in results), n),
        "intervention_time_s_total": sum(r.intervention_time_s for r in results),
        "inference_latency_ms": {"p50": percentile(lat, 50), "p95": percentile(lat, 95)},
        "seen": _success_block([r for r in results if r.seen]),
        "unseen": _success_block([r for r in results if not r.seen]),
        "by_category": {c: _success_block(rs) for c, rs in sorted(by_category.items())},
        "by_scenario": {s: _success_block(rs) for s, rs in sorted(by_scenario.items())},
        "failure_categories": dict(Counter(categorize(r) for r in results if not r.success).most_common()),
    }


def compare(new: dict[str, Any], old: dict[str, Any]) -> list[dict[str, Any]]:
    """Metric-by-metric comparison of two metrics dicts (new − old)."""
    rows = []
    pairs = [
        ("task_success", lambda m: m["task_success"]["rate"], True),
        ("unseen_success", lambda m: m["unseen"]["rate"], True),
        ("grasp_success_rate", lambda m: m["grasp_success_rate"], True),
        ("collision_rate", lambda m: m["collision_rate"], False),
        ("safety_stop_rate", lambda m: m["safety_stop_rate"], False),
        ("timeout_rate", lambda m: m["timeout_rate"], False),
        ("latency_p95_ms", lambda m: m["inference_latency_ms"]["p95"], False),
    ]
    for name, get, higher_better in pairs:
        a, b = get(new), get(old)
        delta = None if a is None or b is None else a - b
        better = None if delta is None else (delta > 0 if higher_better else delta < 0)
        rows.append({"metric": name, "new": a, "old": b, "delta": delta, "improved": better})
    return rows
