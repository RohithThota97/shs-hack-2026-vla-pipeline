"""Post-training verdict for one model: offline gates + Isaac Sim gates → DEPLOY / MORE TRAINING / HOLD.

After the team's post_train_eval.py (AI-project-work), on our registry and test split:
- offline: outputs/runs/<model>/offline_eval_test.json (`hackpipe eval --model … --offline`)
- online: the model's latest real sim eval (`hackpipe eval --import results.json --replay replay.json`); it counts
  only if replaying the recorded demo in that scene succeeded — otherwise the scene, not the model, is measured
- reference: the deployed model's offline eval (regression gate)

Verdicts: NOT TESTED (no current offline test) · NEEDS MORE TRAINING (an offline check fails, or a verified sim run
fails) · READY FOR SIM TEST (offline passes; no verified sim run yet) · READY TO DEPLOY (sim passes too).
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
    from hackpipe.evaluation.offline import load_result
    return load_result(model_version, split)


def sim_result(model_version: str, reg: Registry) -> dict[str, Any] | None:
    """Latest real Isaac Sim eval (one with stage counts) of the model."""
    for ev in reversed(reg.evals(model_version)):
        m = ev["metrics"]
        if m.get("stage_counts_0to4") is not None and not m.get("fake_results"):
            return {"eval_id": ev["eval_id"], **m}
    return None


def _gate(rows: list[dict[str, Any]], name: str, value: Any, ok: bool, limit: str, blocking: bool = True,
          hint: str = "") -> None:
    rows.append({"gate": name, "value": value, "limit": limit, "pass": bool(ok), "blocking": blocking, "hint": hint,
                 "missing": False})


def _check(rows: list[dict[str, Any]], name: str, value: Any, test, limit: str, blocking: bool = True,
           hint: str = "") -> None:
    """A gate on a measured value; a missing / non-finite value is 'not measured' (verdict NOT EVALUATED, not a
    training problem)."""
    import math
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        rows.append({"gate": name, "value": "not measured", "limit": limit, "pass": False, "blocking": blocking,
                     "hint": "re-run the offline eval (with robustness)", "missing": True})
    else:
        _gate(rows, name, value, test(value), limit, blocking, hint)


def _hold_still(off: dict[str, Any]) -> dict[str, float]:
    """Do-nothing error per sub-task (and "all") of the complete-task rows: the scale every model is judged on."""
    out = {}
    for r in off.get("rows") or []:
        if r["instruction"] == "full" and r.get("hold_still_arm_rad"):
            out[r["subtask"]] = r["hold_still_arm_rad"]
    return out


def offline_gates(off: dict[str, Any], ref: dict[str, Any] | None, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    b, w, rows = cfg["block"], cfg["warn"], []
    ex = off.get("executed") or {}
    # model vs doing nothing over the whole predicted chunk (3.3 s at 6 Hz), both on the same steps: over the next
    # 0.5 s alone the arm barely moves, so standing still scores almost like a perfect model
    chunk = {r["subtask"]: r for r in off.get("rows") or [] if r["instruction"] == "full"
             and r.get("hold_still_arm_rad") and r.get("chunk_arm_rad") is not None}
    allrow = next((r for k, r in chunk.items() if k.endswith("· all")), None)
    train = "train longer (more steps), or give the weak sub-task more weight in the curriculum"
    arm = ex.get("arm_mean")                     # executed actions: the regression check against the served model
    _check(rows, "arm error ÷ doing nothing (3.3 s prediction)",
           round(allrow["chunk_arm_rad"] / allrow["hold_still_arm_rad"], 3) if allrow else None,
           lambda x: x <= b["max_ratio_to_hold_still"], f"≤ {b['max_ratio_to_hold_still']}", hint=train)
    subs = [(k, r["chunk_arm_rad"] / r["hold_still_arm_rad"]) for k, r in chunk.items() if not k.endswith("· all")]
    if subs:
        name, ratio = max(subs, key=lambda x: x[1])
        _check(rows, f"worst sub-task ÷ doing nothing: {name.split(' · ', 1)[-1]}", round(ratio, 3),
               lambda x: x <= b["max_worst_subtask_ratio"], f"≤ {b['max_worst_subtask_ratio']}", hint=train)
    ge = off.get("gripper_events") or {}
    if ge.get("n"):                                   # no gripper change in the demo: nothing to catch
        _check(rows, "gripper close / open on time (±1 s)", ge.get("hit_rate"),
               lambda x: x >= b["gripper_event_hit_rate_min"], f"≥ {b['gripper_event_hit_rate_min']}",
               hint="check the gripper labels (gripper_lag_fix) or give the grasp / release sub-tasks more weight")
    robust = off.get("robustness") or {}
    rob, skipped = robust.get("variants") or {}, robust.get("skipped") or {}
    for k in b.get("robustness_variants") or []:
        if k in skipped:
            _gate(rows, f"robustness: {k}", "not applicable", True, skipped[k], blocking=False)
        else:
            _check(rows, f"robustness: {k} (× normal)", (rob.get(k) or {}).get("ratio"),
                   lambda x: x <= b["robustness_ratio_max"], f"≤ {b['robustness_ratio_max']}",
                   hint="add that variation to training (image_aug, state_noise_std, paraphrase_prob)")
    ref_arm = ((ref or {}).get("executed") or {}).get("arm_mean")
    if ref_arm and arm is not None:
        _check(rows, f"vs served model {ref['model']} (× its arm error)", round(arm / ref_arm, 3),
               lambda x: x <= b["regression_vs_deployed_max"], f"≤ {b['regression_vs_deployed_max']}",
               hint="worse than the model being served: keep that one")
    lat = (off.get("latency_ms") or {}).get("replan_median")
    if lat is not None:                 # the robot asks again every n_action_steps steps: that is the time budget
        from hackpipe.common.config import project_config
        n = int((project_config().get("deploy") or {}).get("n_action_steps") or 1)
        budget = round(1000 * n / float(off.get("rate_hz") or 6), 1)
        _check(rows, f"time for one prediction (ms; re-plan every {n} steps)", lat, lambda x: x <= budget,
               f"≤ {budget}", blocking=False)
    sm = off.get("smoothness") or {}
    if sm.get("at_replan_p95") is not None and sm.get("demo_p95"):
        _check(rows, "jump when re-planning (× demo step)", round(sm["at_replan_p95"] / sm["demo_p95"], 2),
               lambda x: x <= w["replan_jump_vs_demo_max"], f"≤ {w['replan_jump_vs_demo_max']}", blocking=False)
    te = ge.get("mean_abs_timing_error_s")
    if te is not None:
        _check(rows, "gripper timing error (s)", te, lambda x: x <= w["gripper_timing_error_s_max"],
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


_UNSET: Any = object()


def verdict(model_version: str, registry: Registry | None = None, cfg: dict[str, Any] | None = None,
            write: bool = True, off: Any = _UNSET, deployed: Any = _UNSET, ref: Any = _UNSET) -> dict[str, Any]:
    """off / deployed / ref: already loaded (the Evaluate page loads every model's offline eval once)."""
    from hackpipe.evaluation.offline import is_current
    reg = registry or Registry()
    model = reg.get_model(model_version)
    if model is None:
        raise KeyError(f"Unknown model {model_version}")
    cfg = cfg or gates_config()
    if off is _UNSET:
        from hackpipe.evaluation.offline import result_for
        off = result_for(model)                       # the test of the checkpoint in use (also one made in training)
    deployed = reg.current_deployed_model() if deployed is _UNSET else deployed
    if deployed == model_version:
        ref = None
    elif ref is _UNSET:
        ref = offline_result(deployed) if deployed else None
    sim = sim_result(model_version, reg)
    out: dict[str, Any] = {"model": model_version, "created_at": now_iso(), "reference": deployed if ref else None,
                           "sim_eval": sim["eval_id"] if sim else None}
    if not is_current(off, model):
        why = ("run the offline eval" if not off else "the offline eval predates the gate metrics: re-run it"
               if not (off.get("executed") or {}).get("arm_mean") else
               "the model now points at another checkpoint than the one tested: test it again")
        out.update(verdict="NOT TESTED", offline_gates=[], online_gates=[], online_state="skipped",
                   todo=[why
                         + f" (hackpipe eval --model {model_version} --offline, or Evaluate all pending)"])
        return out
    rows = offline_gates(off, ref, cfg["offline"])
    state, sim_rows = online_gates(sim, cfg["online"])
    offline_ok = all(r["pass"] for r in rows if r["blocking"])
    missing = [r for r in rows if r["blocking"] and r.get("missing")]
    todo = []
    if missing:                                       # unmeasured is not a training problem
        v = "NOT TESTED"
        todo.append("not measured: " + ", ".join(r["gate"] for r in missing)
                    + f" — re-run hackpipe eval --model {model_version} --offline (without --no-robustness)")
    elif not offline_ok:
        v = "NEEDS MORE TRAINING"
        todo += [f"{r['gate']} = {r['value']} (limit {r['limit']}): {r['hint']}" for r in rows
                 if r["blocking"] and not r["pass"]]
    elif state == "fail":
        v = "NEEDS MORE TRAINING"
        todo.append("accurate offline but fails in closed loop: errors compound when the robot acts on its own "
                    "predictions — add recovery data (DART recordings, scripts/isaac/record_demos.py) or state noise")
    elif state in ("unverified", "skipped"):
        v = "READY FOR SIM TEST"
        todo.append("offline gates pass; closed-loop success is unknown: " + (
            "the recorded demo does not succeed in that scene (fix the scene), or the import had no --replay"
            if state == "unverified" else "no Isaac Sim eval yet (configs/eval/sim_handover_v1.yaml, runbook)"))
    else:
        v = "READY TO DEPLOY"
    todo += [f"note: {r['gate']} = {r['value']} (limit {r['limit']})" for r in rows if not r["blocking"] and not r["pass"]]
    out.update(verdict=v, offline_gates=rows, online_gates=sim_rows, online_state=state, todo=todo,
               offline_file=f"outputs/runs/{model_version}/offline_eval_test.json")
    if write:
        d = resolve(project_config()["project"]["outputs_dir"]) / "runs" / model_version
        d.mkdir(parents=True, exist_ok=True)
        (d / "verdict.json").write_text(json.dumps(out, indent=1))
    return out
