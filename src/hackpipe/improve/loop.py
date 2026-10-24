"""Improve loop: box grid → Isaac Sim recordings → dataset → fine-tuned model → test → better than the best?

Each step reads the pipeline's own state (files, registry), so `run()` can be called again at any time and goes on
from where the loop stands; it stops where a step needs something it can't do itself (the Isaac Sim rendering on
the site PC, a GPU for training) and says what to do.

  1 plan       the grid of box positions, checked by inverse kinematics on the reference demo (improve/grid.py)
  2 reference  the reference file for the site PC (`hackpipe sim reference`: demo actions + training lighting)
  3 render     site PC: record_demos.py --mode grid → recordings (waits for them)
  4 convert    recordings → LeRobot v3 folder data/incoming/<source_name>
  5 import     import + quality checks of the new demos
  6 version    next dataset version: base = latest approved + the new demos, sub-task labels for all
  7 approve    automatic when every new demo comes from a sim-generated source (configs/improve.yaml), else waits
  8 train      fine-tune the best model on the new version (curriculum, handover + grid sources)
  9 evaluate   best checkpoint → test on lighting10 (the held-out test source)
 10 compare    is the new model the best one now (sim success → offline test → loss, training/ranking.py)?
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from hackpipe.common.config import CONFIG_DIR, config_dir, load_yaml, project_config, resolve
from hackpipe.common.ids import now_iso
from hackpipe.common.registry import Registry

STEPS = [("plan", "Grid of box positions"), ("reference", "Reference file for the site PC"),
         ("render", "Render in Isaac Sim (site PC)"), ("convert", "Recordings → dataset"),
         ("import", "Import + quality checks"), ("version", "New dataset version + sub-task labels"),
         ("approve", "Approval"), ("train", "Fine-tune the best model"), ("evaluate", "Best checkpoint + test"),
         ("compare", "Better than the best so far?")]


def improve_config() -> dict[str, Any]:
    f = config_dir() / "improve.yaml"
    return load_yaml(f if f.is_file() else CONFIG_DIR / "improve.yaml")


def _dir() -> Path:
    d = resolve(project_config()["project"]["outputs_dir"]) / "improve"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state() -> dict[str, Any]:
    f = _dir() / "loop.json"
    return json.loads(f.read_text()) if f.is_file() else {}


def _save(**kw: Any) -> dict[str, Any]:
    st = {**_state(), **kw, "updated_at": now_iso()}
    (_dir() / "loop.json").write_text(json.dumps(st, indent=1))
    return st


def _source_root(cfg: dict[str, Any]) -> Path:
    return resolve("data/incoming") / cfg["grid"]["source_name"]


def _recordings(cfg: dict[str, Any]) -> list[Path]:
    d = resolve(cfg["grid"]["recordings_dir"])
    return sorted(p for p in d.iterdir() if (p / "meta.json").is_file()) if d.is_dir() else []


def _version_with_source(reg: Registry, source: str) -> dict[str, Any] | None:
    """Newest dataset version that contains demos of this source."""
    from hackpipe.dataset.versioning import load_manifest
    for d in sorted(reg.datasets(), key=lambda d: int(d["version"].split("-v", 1)[1]), reverse=True):
        m = load_manifest(d["version"], reg)
        if any(Path(s).name == source for split in ("train", "val") for s, _ in m["episodes"].get(split, [])):
            return {**d, "manifest": m}
    return None


def _model_on(reg: Registry, version: str) -> dict[str, Any] | None:
    ms = [m for m in reg.models() if m["dataset_version"] == version and m["meta"].get("purpose") != "timing"
          and m["meta"].get("improve_loop")]
    return ms[-1] if ms else None


def status(registry: Registry | None = None) -> list[dict[str, Any]]:
    """[{step, title, done, detail}] in order; the first not-done step is where the loop stands."""
    from hackpipe.evaluation.offline import is_current, result_for
    reg = registry or Registry()
    cfg = improve_config()
    st = _state()
    src = cfg["grid"]["source_name"]
    rows: dict[str, tuple[bool, str]] = {}
    plan = st.get("plan")
    rows["plan"] = (bool(plan), f"{plan['valid']} of {plan['cells']} box positions valid" if plan else "")
    ref = st.get("reference")
    rows["reference"] = (bool(ref and Path(ref).is_file()), ref or "")
    recs = _recordings(cfg)
    rows["render"] = (bool(recs) or _source_root(cfg).is_dir(),
                      f"{len(recs)} recordings in {cfg['grid']['recordings_dir']}" if recs else
                      f"waiting for {cfg['grid']['recordings_dir']}")
    root = _source_root(cfg)
    rows["convert"] = ((root / "meta" / "info.json").is_file(), str(root.relative_to(resolve("."))) if root.is_dir() else "")
    demos = [d for d in reg.demos() if Path(d["source"]).name == src]
    pending = [d for d in demos if d["status"] == "imported"]
    rows["import"] = (bool(demos) and not pending, f"{len(demos)} demos, " + ", ".join(
        f"{k} {sum(d['status'] == k for d in demos)}" for k in sorted({d['status'] for d in demos})) if demos else "")
    ver = _version_with_source(reg, src)
    rows["version"] = (bool(ver), f"{ver['version']} ({ver['status']})" if ver else "")
    rows["approve"] = (bool(ver and ver["status"] == "approved"),
                       ((ver.get("meta") or {}).get("approved_by") or "") if ver else "")
    model = _model_on(reg, ver["version"]) if ver else None
    rows["train"] = (bool(model and model["status"] in ("trained", "evaluated", "deployed")),
                     f"{model['version']} ({model['status']})" if model else "")
    res = result_for(model) if model else None
    rows["evaluate"] = (bool(model and is_current(res, model)), f"{model['version']} tested" if model and res else "")
    cmp_ = st.get("compare")
    rows["compare"] = (bool(cmp_ and model and cmp_.get("model") == model["version"]),
                       cmp_.get("summary", "") if cmp_ else "")
    return [{"step": k, "title": t, "done": rows[k][0], "detail": rows[k][1]} for k, t in STEPS]


def site_pc_commands(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or improve_config()
    ref = _state().get("reference") or "outputs/improve/ref.json"
    x, y = (",".join(str(v) for v in cfg["grid"][k]) for k in ("x_cm", "y_cm"))
    return (f"<isaac>/python.sh scripts/isaac/record_demos.py --mode grid --usd <Hackathon_FR3Duo_flattened.usd> "
            f"--ref {ref} --out {cfg['grid']['recordings_dir']} --grid_x={x} --grid_y={y}\n"
            f"# then copy {cfg['grid']['recordings_dir']} to this machine (Drive) and run: hackpipe loop run")


# ---------------------------------------------------------------- steps
def step_plan(reg: Registry, cfg: dict[str, Any], log: Callable[[str], None]) -> None:
    import numpy as np

    from hackpipe.common.lerobot_io import DatasetFolder
    from hackpipe.improve.grid import grid_offsets, plan
    ds = DatasetFolder(resolve(cfg["reference_demo_source"]))
    df = ds.frames()
    ep0 = df["episode_index"].min()
    A = np.stack(df[df["episode_index"] == ep0].sort_values("frame_index")["action"]).astype(np.float32)
    cells = plan(A, grid_offsets(tuple(cfg["grid"]["x_cm"]), tuple(cfg["grid"]["y_cm"])))
    ok = [c for c in cells if c["ok"]]
    summary = {"cells": len(cells), "valid": len(ok), "reference": f"{cfg['reference_demo_source']} episode {ep0}",
               "grid": [{"dx": c["dx"], "dy": c["dy"], "ok": c["ok"], "reason": c["reason"]} for c in cells],
               "max_ik_mm": max((c["max_ik_mm"] for c in ok), default=None)}
    _save(plan=summary)
    log(f"plan: {len(ok)} of {len(cells)} box positions valid")


def step_reference(reg: Registry, cfg: dict[str, Any], log: Callable[[str], None]) -> None:
    from hackpipe.dataset.versioning import latest_approved
    from hackpipe.sim.reference import write_reference
    base = latest_approved(reg)
    out = _dir() / "ref.json"
    write_reference(base, "configs/eval/sim_handover_v1.yaml", out, reg)
    _save(reference=str(out.relative_to(resolve("."))) if out.is_relative_to(resolve(".")) else str(out))
    log(f"reference: {out} (from {base})")


def step_convert(reg: Registry, cfg: dict[str, Any], log: Callable[[str], None]) -> None:
    from hackpipe.ingest.sim_recording import convert
    r = convert(resolve(cfg["grid"]["recordings_dir"]), _source_root(cfg))
    log(f"convert: {r}")


def step_import(reg: Registry, cfg: dict[str, Any], log: Callable[[str], None]) -> None:
    from hackpipe.pipeline.auto import _import_and_check
    out, _ = _import_and_check(str(_source_root(cfg)), reg)
    log(f"import + QC: {out['import']} · {out['qc'] and out['qc']['status_counts']}")


def step_version(reg: Registry, cfg: dict[str, Any], log: Callable[[str], None]) -> None:
    from hackpipe.dataset.subtasks import label
    from hackpipe.dataset.versioning import create_dataset_version, latest_approved
    base = latest_approved(reg)
    labels = label(base, reg, review_images=False, progress=log)
    m = create_dataset_version(reg, notes=f"improve loop: {base} + {cfg['grid']['source_name']} (box grid)",
                               base=base, subtasks=labels)
    _save(version=m["version"])
    log(f"version: {m['version']} (draft) from {base} + {cfg['grid']['source_name']}")


def step_approve(reg: Registry, cfg: dict[str, Any], log: Callable[[str], None]) -> bool:
    from hackpipe.dataset.versioning import approve
    from hackpipe.quality.checks import VALID
    ver = _version_with_source(reg, cfg["grid"]["source_name"])
    added = set((ver["manifest"].get("lineage") or {}).get("added") or [])
    demos = {d["demo_id"]: d for d in reg.demos()}
    allowed = set(cfg.get("auto_approve_sources") or [])
    foreign = [i for i in added if Path(demos[i]["source"]).name not in allowed] if added else []
    bad = [i for i in added if demos[i]["status"] != VALID]
    if not added or foreign or bad:
        log(f"approve: {ver['version']} needs a human approval (Data → Versions): "
            + ("no new demos" if not added else f"{len(foreign)} new demos not sim-generated" if foreign
               else f"{len(bad)} new demos not VALID"))
        return False
    approve(ver["version"], reg, by="auto: sim-generated (configs/improve.yaml)")
    log(f"approve: {ver['version']} approved automatically ({len(added)} sim-generated demos, all VALID)")
    return True


def step_train(reg: Registry, cfg: dict[str, Any], log: Callable[[str], None],
               overrides: dict[str, Any] | None = None) -> None:
    from hackpipe.training import best as best_ck
    from hackpipe.training.ranking import best_model
    from hackpipe.training.train import train
    ver = _version_with_source(reg, cfg["grid"]["source_name"])
    t = cfg["train"]
    ov = {**(t.get("overrides") or {}), **(overrides or {})}
    ov["train_sources"] = sorted({*(ov.get("train_sources") or ["fr3_duo_handover_84"]), cfg["grid"]["source_name"]})
    best = best_model(reg) if t.get("fine_tune_best", True) else None
    init = best["model"]["version"] if best else None
    log(f"train: {t['config']} on {ver['version']}" + (f", fine-tuned from {init}" if init else "") + f", {ov}")
    rec = train(t["config"], ver["version"], registry=reg, overrides=ov, init_from=init)
    reg.update_model_meta(rec["model_version"], {"improve_loop": True})
    if rec.get("exit_code") != 0:
        raise RuntimeError(f"training {rec['model_version']} failed (exit {rec.get('exit_code')})")
    try:
        b = best_ck.select(rec["model_version"], reg)
        log(f"train: {rec['model_version']} done, best checkpoint step {b['step']} (val {b['val_full_arm_rad']} rad)")
    except (RuntimeError, OSError, ValueError) as e:
        log(f"train: {rec['model_version']} done; no best checkpoint ({e}), the last one is tested")


def step_evaluate(reg: Registry, cfg: dict[str, Any], log: Callable[[str], None],
                  every: int | None = None, robustness: bool | None = None) -> None:
    from hackpipe.evaluation.offline import offline_eval
    ver = _version_with_source(reg, cfg["grid"]["source_name"])
    model = _model_on(reg, ver["version"])
    e = cfg.get("evaluate") or {}
    r = offline_eval(model["version"], reg, every=every or int(e.get("every", 3)),
                     robustness=bool(e.get("robustness", True)) if robustness is None else robustness, progress=log)
    log(f"evaluate: {model['version']} arm error {(r.get('executed') or {}).get('arm_mean')} rad")


def step_compare(reg: Registry, cfg: dict[str, Any], log: Callable[[str], None]) -> None:
    from hackpipe.gates.verdict import verdict
    from hackpipe.training.ranking import ranking
    ver = _version_with_source(reg, cfg["grid"]["source_name"])
    model = _model_on(reg, ver["version"])
    ranked = [r for r in ranking(reg) if r["eligible"] and r["tier"]]
    pos = next((i for i, r in enumerate(ranked) if r["model"]["version"] == model["version"]), None)
    v = verdict(model["version"], reg, write=False)
    prev = next((r for r in ranked if r["model"]["version"] != model["version"]), None)
    summary = (f"{model['version']} is the best model now ({ranked[0]['label']}); verdict {v['verdict']}" if pos == 0
               else f"{model['version']} ({ranked[pos]['label'] if pos is not None else 'not ranked'}) is not better "
                    f"than {ranked[0]['model']['version']} ({ranked[0]['label']}); verdict {v['verdict']}")
    _save(compare={"model": model["version"], "best": pos == 0, "verdict": v["verdict"], "summary": summary,
                   "previous_best": prev["model"]["version"] if prev else None})
    log(f"compare: {summary}")


def _print(msg: str) -> None:
    print(msg, flush=True)                      # a job's log is a file: show each step as it happens


def run(registry: Registry | None = None, log: Callable[[str], None] = _print, until: str | None = None,
        train_overrides: dict[str, Any] | None = None, eval_every: int | None = None,
        eval_robustness: bool | None = None) -> list[dict[str, Any]]:
    """Run every step that can run now, in order; stop at the first that has to wait (or after `until`)."""
    from hackpipe.jobs.machine import gpus
    reg = registry or Registry()
    cfg = improve_config()
    for row in status(reg):
        if row["done"]:
            if until == row["step"]:
                break
            continue
        k = row["step"]
        if k == "render":
            log("render: waiting for the site PC — " + site_pc_commands(cfg).splitlines()[0])
            break
        if k in ("train", "evaluate") and not gpus():
            log(f"{k}: needs a GPU — run `hackpipe loop run` on the GPU machine (or the console there)")
            break
        if k == "approve":
            if not step_approve(reg, cfg, log):
                break
        elif k == "train":
            step_train(reg, cfg, log, train_overrides)
        elif k == "evaluate":
            step_evaluate(reg, cfg, log, eval_every, eval_robustness)
        else:
            {"plan": step_plan, "reference": step_reference, "convert": step_convert, "import": step_import,
             "version": step_version, "compare": step_compare}[k](reg, cfg, log)
        if until == k:
            break
    return status(reg)
