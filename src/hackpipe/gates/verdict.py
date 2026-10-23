"""Post-training verdict for one model: offline gates + Isaac Sim gates → DEPLOY / MORE TRAINING / HOLD.

After the team's post_train_eval.py (AI-project-work), on our registry and test split:
- offline: outputs/runs/<model>/offline_eval_test.json (`hackpipe eval --model … --offline`)
- online: the model's latest real sim eval (`hackpipe eval --import results.json --replay replay.json`); it counts
  only if replaying the recorded demo in that scene succeeded — otherwise the scene, not the model, is measured
- reference: the deployed model's offline eval (regression gate)

Verdicts: NOT EVALUATED (no offline eval) · MORE TRAINING (an offline gate fails, or a verified sim eval fails) ·
HOLD: SIM UNVERIFIED (offline passes, sim missing or its demo replay failed) · DEPLOY.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

from hackpipe.common.config import gates_config, project_config, resolve
from hackpipe.common.ids import now_iso
from hackpipe.common.registry import Registry


def offline_result(model_version: str, split: str = "test") -> dict[str, Any] | None:
    f = resolve(project_config()["project"]["outputs_dir"]) / "runs" / model_version / f"offline_eval_{split}.json"
    return json.loads(f.read_text()) if f.is_file() else None


def sim_result(model_version: str, reg: Registry) -> dict[str, Any] | None:
    """Latest real Isaac Sim eval (one with stage counts) of the model."""
    for ev in reversed(reg.evals(model_version)):
        m = ev["metrics"]
        if m.get("stage_counts_0to4") is not None and not m.get("fake_results"):
            return {"eval_id": ev["eval_id"], **m}
    return None


def _gate(rows: list[dict[str, Any]], name: str, value: Any, ok: bool, limit: str, blocking: bool = True,
          hint: str = "") -> None:
    rows.append({"gate": name, "value": value, "limit": limit, "pass": bool(ok), "blocking": blocking, "hint": hint})


def offline_gates(off: dict[str, Any], ref: dict[str, Any] | None, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    b, w, rows = cfg["block"], cfg["warn"], []
    ex = off.get("executed") or {}
    if not ex:
        _gate(rows, "offline eval has the executed-stream metrics", "missing", False, "re-run hackpipe eval --offline",
              hint="the offline eval predates these metrics: run it again")
        return rows
    train = "train longer, or give that sub-task's skill more weight in the curriculum"
    _gate(rows, "arm error, mean (rad)", ex["arm_mean"], ex["arm_mean"] <= b["arm_error_mean_max"],
          f"≤ {b['arm_error_mean_max']}", hint=train)
    _gate(rows, "arm error, p99 (rad)", ex["arm_p99"], ex["arm_p99"] <= b["arm_error_p99_max"],
          f"≤ {b['arm_error_p99_max']}", hint=train)
    subs = [s for s in ex.get("per_subtask") or [] if s["arm_rad"] is not None]
    if subs:
        worst = max(subs, key=lambda s: s["arm_rad"])
        _gate(rows, f"worst sub-task: {worst['subtask']}", worst["arm_rad"],
              worst["arm_rad"] <= b["worst_subtask_error_max"], f"≤ {b['worst_subtask_error_max']}", hint=train)
    ge = off.get("gripper_events") or {}
    if ge.get("hit_rate") is not None:
        _gate(rows, "gripper events caught within 1 s", ge["hit_rate"],
              ge["hit_rate"] >= b["gripper_event_hit_rate_min"], f"≥ {b['gripper_event_hit_rate_min']}",
              hint="check gripper labels / timing in the data (gripper_lag_fix), or weight the grasp sub-tasks")
    rob = (off.get("robustness") or {}).get("variants") or {}
    for k in b.get("robustness_variants") or []:
        if k in rob and rob[k].get("ratio") is not None:
            _gate(rows, f"robustness: {k} (× baseline)", rob[k]["ratio"], rob[k]["ratio"] <= b["robustness_ratio_max"],
                  f"≤ {b['robustness_ratio_max']}", hint="add that variation to training (image_aug, "
                  "state_noise_std, paraphrase_prob)")
        else:
            _gate(rows, f"robustness: {k}", "not measured", False, "run without --no-robustness",
                  hint="re-run the offline eval with robustness")
    if ref and (ref.get("executed") or {}).get("arm_mean"):
        ratio = round(ex["arm_mean"] / ref["executed"]["arm_mean"], 3)
        _gate(rows, f"vs deployed {ref['model']} (× its arm error)", ratio, ratio <= b["regression_vs_deployed_max"],
              f"≤ {b['regression_vs_deployed_max']}", hint="worse than the deployed model: keep the deployed one")
    lat = (off.get("latency_ms") or {}).get("replan_median")
    if lat is not None:
        _gate(rows, "re-plan latency (ms, batch 1)", lat, lat <= w["replan_latency_budget_ms"],
              f"≤ {w['replan_latency_budget_ms']}", blocking=False)
    sm = off.get("smoothness") or {}
    if sm.get("at_replan_p95") is not None and sm.get("demo_p95"):
        r = round(sm["at_replan_p95"] / sm["demo_p95"], 2)
        _gate(rows, "joint jump at re-plan (× demo jumps)", r, r <= w["replan_jump_vs_demo_max"],
              f"≤ {w['replan_jump_vs_demo_max']}", blocking=False)
    te = ge.get("mean_abs_timing_error_s")
    if te is not None:
        _gate(rows, "gripper timing error (s)", te, te <= w["gripper_timing_error_s_max"],
              f"≤ {w['gripper_timing_error_s_max']}", blocking=False)
    return rows


def online_gates(sim: dict[str, Any] | None, cfg: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """(state, rows): state skipped | unverified | pass | fail."""
    if sim is None:
        return "skipped", []
    replay = sim.get("replay")
    if cfg.get("require_replay_ok", True) and not (replay and replay.get("success")):
        return "unverified", []
    b, rows = cfg["block"], []
    sr = (sim.get("task_success") or {}).get("rate") or 0.0
    counts = sim["stage_counts_0to4"]
    stages = [k for k, c in enumerate(counts) for _ in range(c)]
    med = float(median(stages)) if stages else 0.0
    _gate(rows, "sim success rate", round(sr, 3), sr >= b["success_rate_min"], f"≥ {b['success_rate_min']}")
    _gate(rows, "sim median stage (0–4)", med, med >= b["median_stage_min"], f"≥ {b['median_stage_min']}")
    return ("pass" if all(r["pass"] for r in rows) else "fail"), rows


def verdict(model_version: str, registry: Registry | None = None, cfg: dict[str, Any] | None = None,
            write: bool = True) -> dict[str, Any]:
    reg = registry or Registry()
    if reg.get_model(model_version) is None:
        raise KeyError(f"Unknown model {model_version}")
    cfg = cfg or gates_config()
    off = offline_result(model_version)
    deployed = reg.current_deployed_model()
    ref = offline_result(deployed) if deployed and deployed != model_version else None
    sim = sim_result(model_version, reg)
    out: dict[str, Any] = {"model": model_version, "created_at": now_iso(), "reference": deployed if ref else None,
                           "sim_eval": sim["eval_id"] if sim else None}
    if off is None:
        out.update(verdict="NOT EVALUATED", offline_gates=[], online_gates=[], online_state="skipped",
                   todo=[f"run the offline eval: hackpipe eval --model {model_version} --offline"])
        return out
    rows = offline_gates(off, ref, cfg["offline"])
    state, sim_rows = online_gates(sim, cfg["online"])
    offline_ok = all(r["pass"] for r in rows if r["blocking"])
    todo = []
    if not offline_ok:
        v = "MORE TRAINING"
        todo += [f"{r['gate']} = {r['value']} (limit {r['limit']}): {r['hint']}" for r in rows
                 if r["blocking"] and not r["pass"]]
    elif state == "fail":
        v = "MORE TRAINING"
        todo.append("accurate offline but fails in closed loop: errors compound when the robot acts on its own "
                    "predictions — add recovery data (DART recordings, scripts/isaac/record_demos.py) or state noise")
    elif state in ("unverified", "skipped"):
        v = "HOLD: SIM UNVERIFIED"
        todo.append("offline gates pass; closed-loop success is unknown: " + (
            "the recorded demo does not succeed in that scene (fix the scene), or the import had no --replay"
            if state == "unverified" else "no Isaac Sim eval yet (configs/eval/sim_handover_v1.yaml, runbook)"))
    else:
        v = "DEPLOY"
    todo += [f"note: {r['gate']} = {r['value']} (limit {r['limit']})" for r in rows if not r["blocking"] and not r["pass"]]
    out.update(verdict=v, offline_gates=rows, online_gates=sim_rows, online_state=state, todo=todo,
               offline_file=f"outputs/runs/{model_version}/offline_eval_test.json")
    if write:
        d = resolve(project_config()["project"]["outputs_dir"]) / "runs" / model_version
        d.mkdir(parents=True, exist_ok=True)
        (d / "verdict.json").write_text(json.dumps(out, indent=1))
    return out
