"""Best checkpoint of a curriculum run: the saved step with the lowest validation error.

The curriculum trainer validates every `val_every` steps and at each stage end (curriculum_log.json → val:
per dataset × instruction, `arm_rad` = mean arm-joint error of the predicted chunk on the val episodes). Only
steps whose checkpoint still exists count (keep_checkpoints deletes older ones). The score is the complete-task
instruction's arm error (what the robot gets at test time), averaged over the trained datasets.

`select(model)` points the model's checkpoint_dir at that step, so the offline test, serving and deploy all use it;
the last checkpoint stays in meta.last_checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from hackpipe.common.config import resolve
from hackpipe.common.registry import Registry, portable


def candidates(model: dict[str, Any]) -> list[dict[str, Any]]:
    """[{step, stage, end_of_stage, val_full_arm_rad, val_sub_arm_rad, checkpoint (or None if deleted)}], by step."""
    out_dir = resolve(model["meta"].get("output_dir") or ".")
    log = out_dir / "curriculum_log.json"
    if not log.is_file():
        return []
    rows = []
    for v in json.loads(log.read_text()).get("val") or []:
        full = [x["arm_rad"] for k, x in v.items() if isinstance(x, dict) and k.endswith("/full") and "arm_rad" in x]
        sub = [x["arm_rad"] for k, x in v.items() if isinstance(x, dict) and k.endswith("/sub") and "arm_rad" in x]
        ck = out_dir / "checkpoints" / f"{int(v['step']):06d}" / "pretrained_model"
        rows.append({"step": int(v["step"]), "stage": v.get("stage"), "end_of_stage": bool(v.get("end_of_stage")),
                     "val_full_arm_rad": round(mean(full), 4) if full else None,
                     "val_sub_arm_rad": round(mean(sub), 4) if sub else None,
                     "checkpoint": str(ck) if (ck / "model.safetensors").is_file() else None})
    by_step = {r["step"]: r for r in rows}          # a stage end that is also a val step is logged once per step
    return [by_step[s] for s in sorted(by_step)]


def best(model: dict[str, Any]) -> dict[str, Any] | None:
    ok = [r for r in candidates(model) if r["checkpoint"] and r["val_full_arm_rad"] is not None]
    return min(ok, key=lambda r: (r["val_full_arm_rad"], -r["step"])) if ok else None


def select(model_version: str, registry: Registry | None = None) -> dict[str, Any]:
    """Point the model at its best checkpoint (recorded in the model's meta). Returns the chosen row."""
    reg = registry or Registry()
    m = reg.get_model(model_version)
    if m is None:
        raise KeyError(f"Unknown model {model_version}")
    b = best(m)
    if b is None:
        raise RuntimeError(f"{model_version} has no validated checkpoint left (curriculum runs only; keep_checkpoints "
                           "may have deleted them)")
    meta = {"best_step": b["step"], "best_val_full_arm_rad": b["val_full_arm_rad"],
            "last_checkpoint": m["meta"].get("last_checkpoint") or portable(m["checkpoint_dir"] or "")}
    with reg.conn:
        reg.conn.execute("UPDATE models SET checkpoint_dir=? WHERE version=?", (portable(b["checkpoint"]), model_version))
    reg.update_model_meta(model_version, meta)
    return b
