"""R11 / Option B — Find capability gaps, recommend missing data and the next training strategy.

Rules are simple, explicit heuristics so every recommendation can be traced to a metric. Tune them on site.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hackpipe.common.config import gates_config
from hackpipe.common.registry import Registry

STRATEGY_HINTS = {
    "grasp_failed": "More demos around the grasp phase for the weak scenarios; check gripper action range and "
                    "normalisation in the dataset stats.",
    "wrong_object": "Check that the instruction reaches the policy (language-conditioned model, e.g. SmolVLA); "
                    "add demos with varied item order.",
    "object_dropped": "Add demos with careful lift/transfer; check gripper force/closing in the action space.",
    "placement_off_target": "Add demos with varied target positions; check camera coverage of the target area.",
    "collision": "Add obstacle demos; add or tighten safety limits in the adapter.",
    "timeout": "Check inference latency vs action-chunk duration (consider RTC or remote inference); "
               "check for hesitation in demos.",
    "safety_stop": "Review safety limits and the trajectories that triggered them before any retraining.",
    "unknown": "Log more fields in the adapter so failures can be categorised.",
}


def recommend(eval_id: str, registry: Registry | None = None, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    reg = registry or Registry()
    cfg = (cfg or gates_config())["improve"]
    ev = reg.get_eval(eval_id)
    if ev is None:
        raise KeyError(eval_id)
    m = ev["metrics"]
    target = cfg["target_success_rate"]

    gaps = [{"category": c, "success_rate": b["rate"], "n": b["n"]}
            for c, b in m["by_category"].items() if b["rate"] is not None and b["rate"] < target]
    gaps.sort(key=lambda g: g["success_rate"])
    data_recs = [f"Collect ~{cfg['demos_per_gap']} targeted demos for '{g['category']}' "
                 f"(success {g['success_rate']:.0%} over {g['n']} runs, target {target:.0%})." for g in gaps]
    failures = m.get("failure_categories", {})
    strategy = [{"failure": f, "episodes": n, "hint": STRATEGY_HINTS.get(f, STRATEGY_HINTS["unknown"])}
                for f, n in failures.items()]
    if m["unseen"]["rate"] is not None and m["seen"]["rate"] is not None \
            and m["seen"]["rate"] - m["unseen"]["rate"] >= 0.2:
        strategy.insert(0, {"failure": "generalisation_gap", "episodes": m["unseen"]["n"],
                            "hint": "Seen vs unseen gap ≥ 20 points: add variation (positions, lighting) to the "
                                    "data before training longer."})
    out = {"eval_id": eval_id, "model_version": ev["model_version"], "target_success_rate": target,
           "capability_gaps": gaps, "data_recommendations": data_recs, "training_strategy": strategy,
           "fake_results": m.get("fake_results", False)}
    Path(ev["results_dir"], "recommendations.json").write_text(json.dumps(out, indent=2))
    return out
