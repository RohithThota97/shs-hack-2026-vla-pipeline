"""R7 — Generate the quality statement (model card) for one evaluation of one model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hackpipe.common.registry import Registry
from hackpipe.evaluation.metrics import compare
from hackpipe.gates.gates import evaluate_gates


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def build_card(eval_id: str, registry: Registry | None = None) -> str:
    reg = registry or Registry()
    ev = reg.get_eval(eval_id)
    if ev is None:
        raise KeyError(eval_id)
    m = ev["metrics"]
    model = reg.get_model(ev["model_version"])
    ds = reg.get_dataset(model["dataset_version"])
    record: dict[str, Any] = {}
    if model.get("run_record_path") and Path(model["run_record_path"]).exists():
        record = json.loads(Path(model["run_record_path"]).read_text())
    gates = evaluate_gates(eval_id, reg)
    ts = m["task_success"]
    lo, hi = ts["ci95"]

    lines = [f"# Quality statement · {model['version']}", ""]
    if m.get("fake_results"):
        lines += ["> **WARNING: FAKE RESULTS from the dummy simulator adapter. Do not use these numbers.**", ""]
    lines += [
        f"**{model['version']} succeeded in {ts['successes']} of {ts['n']} runs ({_pct(ts['rate'])}, "
        f"95% CI {lo:.0%}–{hi:.0%}) on {ev['test_suite']}.** Recommendation: **{gates['recommendation']}**.", "",
        "## Identity", "",
        "| Field | Value |", "|---|---|",
        f"| Model version | {model['version']} ({model['policy_name']}) |",
        f"| Dataset version | {model['dataset_version']} · fingerprint `{ds['fingerprint'] if ds else '?'}` |",
        f"| Training config | `{record.get('train_config_file', 'n/a')}` · seed "
        f"{(record.get('train_config') or {}).get('seed', 'n/a')} |",
        f"| Code version | {(record.get('environment') or {}).get('git', {}).get('commit', 'n/a')} |",
        f"| Evaluation environment | {ev['env_version']} |",
        f"| Test suite | {ev['test_suite']} |",
        f"| Evaluation runs | {m['n_runs']} |", "",
        "## Results", "",
        "| Metric | Value |", "|---|---|",
        f"| Task success | {_pct(ts['rate'])} (95% CI {lo:.0%}–{hi:.0%}) |",
        f"| Seen scenarios | {_pct(m['seen']['rate'])} ({m['seen']['successes']}/{m['seen']['n']}) |",
        f"| Unseen scenarios | {_pct(m['unseen']['rate'])} ({m['unseen']['successes']}/{m['unseen']['n']}) |",
        f"| Correct object selection | {_pct(m['correct_object_rate'])} |",
        f"| Grasp success | {_pct(m['grasp_success_rate'])} |",
        f"| Placement in target zone | {_pct(m['placement_in_zone_rate'])} |",
        f"| Collision rate | {_pct(m['collision_rate'])} |",
        f"| Object drop rate | {_pct(m['object_drop_rate'])} |",
        f"| Timeout rate | {_pct(m['timeout_rate'])} |",
        f"| Safety stop rate | {_pct(m['safety_stop_rate'])} |",
        f"| Median completion time | {m['completion_time_s']['median'] or 'n/a'} s |",
        f"| Inference latency p50 / p95 | {m['inference_latency_ms']['p50']} / {m['inference_latency_ms']['p95']} ms |",
        "", "## Failure breakdown", "",
        "| Category | Episodes |", "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in m["failure_categories"].items()] or ["| none | 0 |"]
    weak = [(s, b) for s, b in m["by_scenario"].items() if (b["rate"] or 0) < 0.5]
    lines += ["", "## Known limitations", ""]
    lines += [f"- Scenario `{s}`: {_pct(b['rate'])} success ({b['successes']}/{b['n']})" for s, b in weak] \
        or ["- No scenario below 50% success in this test suite."]
    lines += ["- Only tested on the scenarios listed in the test suite; behaviour outside them is unknown."]
    lines += ["", "## Quality gates", "", "| Gate | Result | Detail |", "|---|---|---|"]
    lines += [f"| {g['gate']} | {'pass' if g['passed'] else 'FAIL'} | {g['detail']} |" for g in gates["gates"]]

    prev = [x for x in reg.evals() if x["test_suite"] == ev["test_suite"] and x["model_version"] != model["version"]
            and x["created_at"] <= ev["created_at"]]
    lines += ["", "## Comparison with previous model", ""]
    if prev:
        p = prev[-1]
        lines += [f"Compared with {p['model_version']} on the same test suite:", "",
                  "| Metric | New | Previous | Change |", "|---|---|---|---|"]
        for row in compare(m, p["metrics"]):
            d = "n/a" if row["delta"] is None else f"{row['delta']:+.3f}"
            lines.append(f"| {row['metric']} | {row['new']} | {row['old']} | {d} |")
    else:
        lines += ["- No previous model evaluated on this test suite."]
    card = "\n".join(lines) + "\n"
    Path(ev["results_dir"], "model_card.md").write_text(card)
    return card
