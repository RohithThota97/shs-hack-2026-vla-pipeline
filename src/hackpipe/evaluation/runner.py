"""R5 — Run the same versioned test suite for any registered model and store every episode."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hackpipe.common.config import load_yaml, project_config, resolve
from hackpipe.common.ids import now_iso
from hackpipe.common.registry import Registry
from hackpipe.evaluation.metrics import compute_metrics
from hackpipe.evaluation.schema import EpisodeResult
from hackpipe.sim.adapter import SimAdapter


def run_suite(model_version: str, suite_path: str, adapter: SimAdapter,
              registry: Registry | None = None) -> dict[str, Any]:
    reg = registry or Registry()
    model = reg.get_model(model_version)
    if model is None:
        raise KeyError(f"Unknown model {model_version}")
    suite = load_yaml(suite_path)
    eval_id = f"eval-{model_version}-{suite['version']}-{now_iso().replace(':', '').replace('-', '')}"
    out_dir = resolve(Path(project_config()["project"]["outputs_dir"]) / "eval" / eval_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    identity = {"eval_id": eval_id, "model_version": model_version, "dataset_version": model["dataset_version"],
                "test_suite": suite["version"], "env_version": adapter.env_version}

    results: list[EpisodeResult] = []
    with open(out_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for sc in suite["scenarios"]:
            for run in range(int(sc.get("n_runs", 1))):
                seed = int(sc.get("seed_base", 0)) + run
                r = adapter.run_episode(model, sc, seed, float(sc.get("time_limit_s", suite.get(
                    "default_time_limit_s", 60))), identity)
                results.append(r)
                f.write(json.dumps(r.to_dict()) + "\n")
                f.flush()  # keep results even if a later episode crashes

    metrics = compute_metrics(results)
    metrics["suite_version"] = suite["version"]
    metrics["env_version"] = adapter.env_version
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    reg.add_eval(eval_id, model_version, suite["version"], adapter.env_version, str(out_dir), metrics)
    reg.set_model_status(model_version, "evaluated")
    return {"eval_id": eval_id, "results_dir": str(out_dir), "metrics": metrics}


def load_results(results_dir: str | Path) -> list[EpisodeResult]:
    lines = Path(results_dir, "episodes.jsonl").read_text().splitlines()
    return [EpisodeResult.from_dict(json.loads(l)) for l in lines if l.strip()]
