"""R6 — Quality gates. A model is ready for deployment only if every gate passes."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hackpipe.common.config import gates_config
from hackpipe.common.registry import Registry


@dataclass
class GateResult:
    gate: str
    passed: bool
    detail: str


def evaluate_gates(eval_id: str, registry: Registry | None = None,
                   cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    reg = registry or Registry()
    cfg = cfg or gates_config()
    ev = reg.get_eval(eval_id)
    if ev is None:
        raise KeyError(eval_id)
    m = ev["metrics"]
    model = reg.get_model(ev["model_version"])
    ds = reg.get_dataset(model["dataset_version"]) if model else None
    e = cfg["evaluation"]
    results: list[GateResult] = []

    # Data gate
    approved = bool(ds and ds["status"] == "approved")
    results.append(GateResult("data.approved_dataset", approved or not cfg["data"]["require_approved_dataset"],
                              f"dataset {model['dataset_version'] if model else '?'} status: "
                              f"{ds['status'] if ds else 'missing'}"))
    # Training gate
    has_record = bool(model and model.get("run_record_path") and Path(model["run_record_path"]).exists())
    results.append(GateResult("training.run_record", has_record or not cfg["training"]["require_run_record"],
                              "run record found" if has_record else "run record missing"))
    # Evaluation gates
    n = m["n_runs"]
    results.append(GateResult("evaluation.min_runs", n >= e["min_runs"], f"{n} runs (min {e['min_runs']})"))
    sr = m["task_success"]["rate"] or 0.0
    results.append(GateResult("evaluation.min_success_rate", sr >= e["min_success_rate"],
                              f"success {sr:.1%} (target {e['min_success_rate']:.0%}), "
                              f"95% CI {m['task_success']['ci95']}"))
    results.append(GateResult("evaluation.real_results", not m.get("fake_results", False),
                              "FAKE dummy-adapter results" if m.get("fake_results") else "simulator/robot results"))

    current = reg.current_deployed_model()
    if current and current != ev["model_version"]:
        cur_evals = [x for x in reg.evals(current) if x["test_suite"] == ev["test_suite"]]
        if cur_evals:
            cm = cur_evals[-1]["metrics"]
            if e.get("no_worse_than_current"):
                c_sr = cm["task_success"]["rate"] or 0.0
                results.append(GateResult("evaluation.no_worse_than_current", sr >= c_sr,
                                          f"{sr:.1%} vs deployed {current} {c_sr:.1%}"))
            for key, limit_key in (("collision_rate", "max_collision_rate_increase"),
                                   ("safety_stop_rate", "max_safety_stop_rate_increase")):
                a, b = m.get(key) or 0.0, cm.get(key) or 0.0
                results.append(GateResult(f"evaluation.{key}", a - b <= e[limit_key],
                                          f"{a:.1%} vs deployed {b:.1%}"))
        else:
            results.append(GateResult("evaluation.no_worse_than_current", False,
                                      f"deployed model {current} has no eval on {ev['test_suite']}"))
    if e.get("max_latency_p95_ms") is not None:
        p95 = m["inference_latency_ms"]["p95"]
        results.append(GateResult("evaluation.latency_p95", p95 is not None and p95 <= e["max_latency_p95_ms"],
                                  f"p95 {p95} ms (max {e['max_latency_p95_ms']})"))

    all_pass = all(r.passed for r in results)
    success_ok = next(r.passed for r in results if r.gate == "evaluation.min_success_rate")
    if all_pass:
        rec = "deploy"
    elif success_ok and m.get("fake_results") is False:
        rec = "deploy with supervision only after review"
    else:
        rec = "do not deploy"
    out = {"eval_id": eval_id, "model_version": ev["model_version"], "all_passed": all_pass,
           "recommendation": rec, "gates": [asdict(r) for r in results]}
    Path(ev["results_dir"], "gates.json").write_text(json.dumps(out, indent=2))
    return out
