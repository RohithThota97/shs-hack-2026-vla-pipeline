"""Which trained model is best, for the Train page's "fine-tune the best model" mode.

Evidence, strongest first (a model is ranked by the strongest it has):
1. sim — task success rate of its latest real (not fake) simulator eval; higher is better
2. offline — chunk arm error on the held-out test split ÷ the hold-still error on the same frames
   (offline_eval_test.json, complete-task instruction): a ratio, so models scored on different versions' test
   splits stay comparable; lower is better, and only counted when it beats hold-still (ratio < 1)
3. loss — final training loss (loss.csv), or for curriculum runs without one the last validation L1
   (curriculum_log.json); lower is better, but not comparable across datasets: a weak signal
Timing runs, failed / unfinished runs, models without a checkpoint or train config on this machine, and
non-finite values (a diverged run's nan loss) are never candidates.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from hackpipe.common.config import project_config, resolve
from hackpipe.common.registry import Registry

TIERS = ("sim", "offline", "loss")
FINISHED = ("trained", "evaluated", "deployed", "rolled_back")


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _sim(model: str, reg: Registry) -> float | None:
    for ev in reversed(reg.evals(model)):
        m = ev.get("metrics") or {}
        if isinstance(m, str):
            m = json.loads(m)
        rate = (m.get("task_success") or {}).get("rate")
        if not m.get("fake_results") and _finite(rate):
            return float(rate)
    return None


def _offline(model: str) -> tuple[float, float] | None:
    """(chunk arm error, hold-still error) on the test split, averaged over its sources."""
    from hackpipe.evaluation.offline import load_result
    res = load_result(model)
    if res is None:
        return None
    rows = [r for r in res.get("rows") or []
            if r["instruction"] == "full" and r["subtask"].endswith("· all")]
    if not rows:
        return None
    err, still = mean(r["chunk_arm_rad"] for r in rows), mean(r["hold_still_arm_rad"] for r in rows)
    return (err, still) if _finite(err) and _finite(still) and still > 0 else None


def _final_loss(model: dict[str, Any]) -> tuple[float, str] | None:
    """(value, what it is): training loss from loss.csv, else the curriculum's last validation L1."""
    csv = Path(model.get("run_record_path") or ".").parent / "loss.csv"
    if csv.is_file():
        lines = [l for l in csv.read_text().splitlines()[1:] if l.strip()]
        try:
            if lines and _finite(v := float(lines[-1].split(",")[-1])):
                return v, "final loss"
        except ValueError:
            pass
    clog = resolve(model["meta"].get("output_dir") or ".") / "curriculum_log.json"
    val = json.loads(clog.read_text()).get("val") if clog.is_file() else None
    errs = [x["l1_norm"] for x in val[-1].values() if isinstance(x, dict) and "l1_norm" in x] if val else []
    v = mean(errs) if errs else None
    return (v, "validation L1") if _finite(v) else None


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
    elif not resolve(model["meta"]["train_config_file"]).is_file():
        why_not = f"train config {model['meta']['train_config_file']} not found on this machine"
    out: dict[str, Any] = {"tier": None, "value": None, "label": "no evaluation or loss yet",
                           "eligible": why_not is None, "why_not": why_not}
    sim = _sim(v, reg)
    off = _offline(v)
    if sim is not None:
        out.update(tier="sim", value=sim, label=f"sim success {sim:.0%}")
    elif off and off[0] < off[1]:
        out.update(tier="offline", value=off[0] / off[1],
                   label=f"offline test arm error {off[0]:.3f} rad = {off[0] / off[1]:.0%} of hold-still")
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


def best_model(reg: Registry, ranked: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    top = ranking(reg) if ranked is None else ranked
    return top[0] if top and top[0]["eligible"] and top[0]["tier"] else None


def _ids(manifest: dict[str, Any], *splits: str) -> set[tuple[str, int]]:
    return {(Path(s).name, int(i)) for sp in splits for s, i in manifest["episodes"].get(sp, [])}


def init_leakage(init_model: dict[str, Any], target_manifest: dict[str, Any], reg: Registry) -> list[tuple[str, int]]:
    """Test episodes of the target version that the starting model — or any model it was fine-tuned from — was
    trained or validated on."""
    from hackpipe.dataset.versioning import load_manifest
    seen: set[tuple[str, int]] = set()
    m, visited = init_model, set()
    while m and m.get("version") not in visited:
        visited.add(m.get("version"))
        seen |= _ids(load_manifest(m["dataset_version"], reg), "train", "val")
        parent = (m.get("meta") or {}).get("init_from")
        m = reg.get_model(parent) if parent else None
    return sorted(seen & _ids(target_manifest, "test"))
