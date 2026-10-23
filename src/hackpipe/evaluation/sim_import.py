"""R5 for Isaac Sim runs made outside the pipeline: import the team's `eval_isaac.py` results.json as an evaluation
of a registered model, so metrics (95% CI), gates and the model card work as for any other suite.

Checked before anything is stored:
- every episode's lighting comes from the suite's held-out lighting source (`sim.lighting_source`), i.e. the run
  used a reference file from `hackpipe sim reference` — not the team's val lighting (episodes in our train split)
- the policy re-planned as the suite says (`sim.n_action_steps`), so results of different settings never mix

eval_isaac.py stages: 0 not lifted · 1 lifted · 2 carried halfway · 3 over tray · 4 resting on tray (= success).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from hackpipe.common.config import load_yaml, resolve, source_key
from hackpipe.common.lerobot_io import DatasetFolder
from hackpipe.common.registry import Registry
from hackpipe.evaluation.schema import EpisodeResult

# failure stage (EpisodeResult vocabulary) by the furthest eval_isaac stage reached
FAILURE_STAGE = {0: "grasp", 1: "transfer", 2: "transfer", 3: "place"}


class ImportRefused(ValueError):
    pass


def to_episode(e: dict[str, Any], identity: dict[str, Any], scenario: dict[str, Any], gif: str | None
               ) -> EpisodeResult:
    stage = int(e.get("stage", 0))
    success = bool(e.get("success"))
    lat = e.get("latency_ms") or {}
    return EpisodeResult(
        **identity, scenario_id=scenario["id"], category=scenario.get("category", "environment_variation"),
        seen=bool(scenario.get("seen", False)), seed=int(e.get("lighting_from_val_episode", e.get("episode", 0))),
        instruction=scenario.get("instruction", ""), success=success,
        stop_reason="success" if success else "timeout",       # eval_isaac always runs to max_seconds
        failure_stage=None if success else FAILURE_STAGE.get(stage, "release"),
        grasp_attempts=1, grasp_successes=int(stage >= 1),
        in_target_zone=success, placement_error_m=(e["dist_to_demo_final_mm"] / 1000
                                                   if success and e.get("dist_to_demo_final_mm") is not None else None),
        timeout=not success, completion_time_s=None,             # not measured by eval_isaac
        inference_latency_ms=[float(lat["median"])] if lat.get("median") is not None else [],
        video_path=gif,
        notes=(f"stage {stage} ({e.get('stage_name')}), lift {e.get('max_lift_mm')} mm, tilt {e.get('box_tilt_deg')}°,"
               f" final box {e.get('final_box_pos')}, latency p95 {lat.get('p95')} ms"),
    )


def import_results(model_version: str, suite_path: str, results_json: str | Path, gif_dir: str | Path | None = None,
                   registry: Registry | None = None) -> dict[str, Any]:
    from hackpipe.evaluation.runner import new_eval, store_eval
    reg = registry or Registry()
    if reg.get_model(model_version) is None:
        raise KeyError(f"Unknown model {model_version}")
    suite = load_yaml(suite_path)
    sim = suite.get("sim") or {}
    if len(suite.get("scenarios") or []) != 1:
        raise ImportRefused("a sim suite has exactly one scenario (eval_isaac.py runs one task)")
    scenario = suite["scenarios"][0]
    raw = json.loads(Path(results_json).read_text())
    episodes = raw.get("episodes") or []
    if not episodes:
        raise ImportRefused(f"{results_json} has no episodes")
    src = source_key(resolve(sim["lighting_source"]))
    allowed = {int(i) for i in DatasetFolder(resolve(src)).episode_extras()}
    used = [int(e.get("lighting_from_val_episode", -1)) for e in episodes]
    if not set(used) <= allowed:
        raise ImportRefused(f"episodes lit like {sorted(set(used) - allowed)}, which are not {src} episodes — run "
                            "eval_isaac.py with --ref from `hackpipe sim reference` (held-out lighting)")
    if len(set(used)) != len(used):
        raise ImportRefused(f"lighting episodes repeat ({sorted(used)}): each episode of the suite is one lighting")
    n_runs = int(scenario.get("n_runs", len(episodes)))
    if len(episodes) != n_runs:
        raise ImportRefused(f"{len(episodes)} episodes, the suite has {n_runs} (eval_isaac.py --episodes {n_runs}); "
                            "a short or crashed run isn't comparable with full ones")
    want = sim.get("n_action_steps")
    got = (raw.get("policy") or {}).get("n_action_steps")
    if want is not None and got is not None and int(got) != int(want):
        raise ImportRefused(f"the policy executed {got} actions per prediction; the suite says {want} "
                            f"(serve_policy.py --n_action_steps {want}) — or create a new suite version")
    if want is not None and got is None:
        raise ImportRefused("results.json doesn't say how many actions per prediction the policy executed")
    eval_id, out, identity = new_eval(model_version, suite, suite["env_version"], reg)
    shutil.copyfile(results_json, out / "results_eval_isaac.json")
    results = []
    for e in episodes:
        gif = None
        if gif_dir:
            g = Path(gif_dir) / f"policy_ep{int(e.get('episode', 0)):02d}.gif"
            if g.exists():
                shutil.copyfile(g, out / g.name)
                gif = g.name
        results.append(to_episode(e, identity, scenario, gif))
    with open(out / "episodes.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")
    stages = [int(e.get("stage", 0)) for e in episodes]
    return store_eval(eval_id, out, model_version, suite, suite["env_version"], results, reg,
                      {"stage_counts_0to4": [stages.count(k) for k in range(5)], "imported_from": str(results_json),
                       "policy": raw.get("policy"), "lighting_source": src})
