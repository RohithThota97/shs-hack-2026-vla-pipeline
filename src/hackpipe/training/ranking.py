"""Which trained model is best, for the Train page's "fine-tune the best model" mode.

Evidence, strongest first (a model is ranked by the strongest it has):
1. sim — task success rate of its latest real (not fake) simulator eval; higher is better
2. offline — mean chunk arm error on the held-out test split (offline_eval_test.json, complete-task instruction);
   lower is better, and only counted when it beats the hold-still baseline
3. loss — final training loss (loss.csv), or for curriculum runs without one the last validation L1
   (curriculum_log.json); lower is better, but not comparable across datasets: a weak signal
Timing runs, failed / unfinished runs and models without a checkpoint on this machine are never candidates.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from hackpipe.common.config import project_config, resolve
from hackpipe.common.registry import Registry

TIERS = ("sim", "offline", "loss")
FINISHED = ("trained", "evaluated", "deployed", "rolled_back")


def _sim(model: str, reg: Registry) -> float | None:
    for ev in reversed(reg.evals(model)):
        m = ev.get("metrics") or {}
        if isinstance(m, str):
            m = json.loads(m)
        rate = (m.get("task_success") or {}).get("rate")
        if not m.get("fake_results") and rate is not None:
            return float(rate)
    return None


def _offline(model: str) -> tuple[float, float] | None:
    """(chunk arm error, hold-still error) on the test split, averaged over its sources."""
    f = resolve(project_config()["project"]["outputs_dir"]) / "runs" / model / "offline_eval_test.json"
    if not f.is_file():
        return None
    rows = [r for r in json.loads(f.read_text()).get("rows") or []
            if r["instruction"] == "full" and r["subtask"].endswith("· all")]
    if not rows:
        return None
    return mean(r["chunk_arm_rad"] for r in rows), mean(r["hold_still_arm_rad"] for r in rows)


def _final_loss(model: dict[str, Any]) -> tuple[float, str] | None:
    """(value, what it is): training loss from loss.csv, else the curriculum's last validation L1."""
    csv = Path(model.get("run_record_path") or ".").parent / "loss.csv"
    if csv.is_file():
        lines = [l for l in csv.read_text().splitlines()[1:] if l.strip()]
        try:
            if lines:
                return float(lines[-1].split(",")[-1]), "final loss"
        except ValueError:
            pass
    clog = resolve(model["meta"].get("output_dir") or ".") / "curriculum_log.json"
    val = json.loads(clog.read_text()).get("val") if clog.is_file() else None
    errs = [x["l1_norm"] for x in val[-1].values() if isinstance(x, dict) and "l1_norm" in x] if val else []
    return (mean(errs), "validation L1") if errs else None


def score(model: dict[str, Any], reg: Registry) -> dict[str, Any]:
    """{"tier", "value", "label", "eligible", "why_not"} for one model."""
    v = model["version"]
    why_not = None
    if model["meta"].get("purpose") == "timing":
        why_not = "timing run, not a result"
    elif model["status"] not in FINISHED:
        why_not = f"status {model['status']}"
    elif not (resolve(model.get("checkpoint_dir") or ".") / "config.json").is_file():
        why_not = "no checkpoint on this machine"
    elif not model["meta"].get("train_config_file"):
        why_not = "trained outside the pipeline (no train config to repeat)"
    out: dict[str, Any] = {"tier": None, "value": None, "label": "no evaluation or loss yet",
                           "eligible": why_not is None, "why_not": why_not}
    sim = _sim(v, reg)
    off = _offline(v)
    if sim is not None:
        out.update(tier="sim", value=sim, label=f"sim success {sim:.0%}")
    elif off and off[0] < off[1]:
        out.update(tier="offline", value=off[0],
                   label=f"offline test arm error {off[0]:.3f} rad (hold-still {off[1]:.3f})")
    else:
        loss = _final_loss(model)
        if loss is not None:
            out.update(tier="loss", value=loss[0], label=f"{loss[1]} {loss[0]:.4f}"
                       + (f" · offline {off[0]:.3f} rad, worse than hold-still {off[1]:.3f}" if off else ""))
        elif off:
            out["label"] = f"offline {off[0]:.3f} rad, worse than hold-still {off[1]:.3f}"
    return out


def _key(s: dict[str, Any]) -> tuple[int, float]:
    t = TIERS.index(s["tier"]) if s["tier"] else len(TIERS)
    v = s["value"] if s["value"] is not None else float("inf")
    return t, -v if s["tier"] == "sim" else v


def ranking(reg: Registry) -> list[dict[str, Any]]:
    """All models with their score, best first; candidates (eligible with evidence) before the rest."""
    rows = [{"model": m, **score(m, reg)} for m in reg.models()]
    return sorted(rows, key=lambda r: (not (r["eligible"] and r["tier"]), *_key(r)))


def best_model(reg: Registry) -> dict[str, Any] | None:
    top = ranking(reg)
    return top[0] if top and top[0]["eligible"] and top[0]["tier"] else None


def _ids(manifest: dict[str, Any], *splits: str) -> set[tuple[str, int]]:
    return {(Path(s).name, int(i)) for sp in splits for s, i in manifest["episodes"].get(sp, [])}


def init_leakage(init_model: dict[str, Any], target_manifest: dict[str, Any], reg: Registry) -> list[tuple[str, int]]:
    """Test episodes of the target version that the starting model was trained or validated on."""
    from hackpipe.dataset.versioning import load_manifest
    seen = _ids(load_manifest(init_model["dataset_version"], reg), "train", "val")
    return sorted(seen & _ids(target_manifest, "test"))
