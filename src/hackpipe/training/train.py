"""R4 — Reproducible training.

Builds a `lerobot-train` command from a YAML config and an approved dataset version, saves a complete run
record (command, config, dataset fingerprint, git commit, packages, GPU), runs it, and registers the model.
LeRobot itself writes train_config.json into every checkpoint; resume with --resume.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from hackpipe.common.config import load_yaml, project_config, resolve
from hackpipe.common.envinfo import environment
from hackpipe.common.ids import next_version, now_iso
from hackpipe.common.registry import Registry
from hackpipe.dataset.versioning import load_manifest, needs_merge, trainable_sources


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, dict)):
        return json.dumps(v, separators=(",", ":"))
    return str(v)


def parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """["steps=200", "policy.device=cuda"] → {"steps": 200, "policy.device": "cuda"} (values parsed as YAML)."""
    out: dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"--set expects key=value, got {pair!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def set_config_steps(config_path: str | Path, steps: int) -> None:
    """Write the chosen step budget into the train config (keeps comments); commit it afterwards."""
    import re
    p = resolve(config_path)
    text = p.read_text()
    new, n = re.subn(r"^(\s*steps:\s*)\d+", rf"\g<1>{int(steps)}", text, count=1, flags=re.MULTILINE)
    if n != 1:
        raise ValueError(f"no `steps:` line in {p}")
    p.write_text(new)


TOP_LEVEL = ("seed", "policy_name", "keep_checkpoints", "trainer", "curriculum")


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """`seed`, `policy_name`, `keep_checkpoints`, `trainer` and `curriculum` (the stage plan) are top-level config
    keys; everything else is a trainer arg (lerobot-train: --key=value)."""
    cfg = {**cfg, "args": dict(cfg.get("args") or {})}
    for k, v in overrides.items():
        if k in TOP_LEVEL:
            cfg[k] = v
        else:
            cfg["args"][k] = v
    return cfg


def training_input(manifest: dict[str, Any], dataset_root: str | None = None) -> dict[str, Any]:
    """Which folder, repo id and episode indices training reads: the merged folder for multi-source versions,
    else the version's single source (falling back to the project dataset root for old manifests)."""
    train, data_only = trainable_sources(manifest)
    mat = manifest.get("materialized")
    if mat:
        return {"repo_id": mat["repo_id"], "root": str(resolve(mat["root"])), "episodes": mat["splits"]["train"],
                "data_only_sources": data_only}
    if data_only and len(train) == 1:      # mixed version: train on the main-schema source, skip the subtask
        eps = sorted(i for s, i in manifest["episodes"]["train"] if s == train[0])
        return {"repo_id": manifest["repo_id"], "root": str(resolve(train[0])), "episodes": eps,
                "data_only_sources": data_only}
    if needs_merge(manifest):
        raise RuntimeError(f"{manifest['version']} has several sources; merge them first: "
                           f"hackpipe dataset materialize {manifest['version']}")
    src = manifest.get("source")
    root = str(resolve(src)) if src else dataset_root
    return {"repo_id": manifest["repo_id"], "root": root, "episodes": manifest["splits"]["train"],
            "data_only_sources": data_only}


def is_curriculum(cfg: dict[str, Any]) -> bool:
    return cfg.get("trainer") == "curriculum"


def curriculum_command(dataset_version: str, config_json: Path, output_dir: Path, resume: bool = False) -> list[str]:
    cmd = [sys.executable, "-m", "hackpipe.training.curriculum", "--dataset", dataset_version,
           "--config", str(config_json), "--output_dir", str(output_dir)]
    return cmd + ["--resume"] if resume else cmd


def build_command(train_cfg: dict[str, Any], manifest: dict[str, Any], output_dir: Path, job_name: str,
                  dataset_root: str | None) -> list[str]:
    inp = training_input(manifest, dataset_root)
    cmd = ["lerobot-train", f"--dataset.repo_id={inp['repo_id']}"]
    if inp["root"]:
        cmd.append(f"--dataset.root={inp['root']}")
    if manifest.get("revision") and not manifest.get("materialized"):
        cmd.append(f"--dataset.revision={manifest['revision']}")
    cmd.append(f"--dataset.episodes={_fmt(inp['episodes'])}")
    cmd += [f"--output_dir={output_dir}", f"--job_name={job_name}", f"--seed={train_cfg.get('seed', 1000)}"]
    rename = manifest["preprocessing"]["preprocessing"].get("rename_map") or {}
    if rename:
        cmd.append(f"--rename_map={_fmt(rename)}")
    for k, v in (train_cfg.get("args") or {}).items():
        cmd.append(f"--{k}={_fmt(v)}")
    return cmd


class PreflightError(RuntimeError):
    pass


def prune_checkpoints(output_dir: Path, keep: int, protect: set[str] | Callable[[], set[str]] | None = None
                      ) -> list[str]:
    """Delete all but the newest `keep` step checkpoints (and never the one `last` points to, nor the step folders in
    `protect`, e.g. the model's chosen best checkpoint), so a run fits a 15 GB Drive. lerobot-train writes
    checkpoints/<step>/ and then moves the `last` link to it."""
    ck = Path(output_dir) / "checkpoints"
    if keep < 1 or not ck.is_dir():
        return []
    steps = sorted((d for d in ck.iterdir() if d.is_dir() and not d.is_symlink() and d.name.isdigit()),
                   key=lambda d: int(d.name))
    last = (ck / "last").resolve() if (ck / "last").exists() else None
    protect = (protect() if callable(protect) else protect) or set()
    removed = []
    for d in steps[:-keep]:
        if (last is not None and d.resolve() == last) or d.name in protect:
            continue
        shutil.rmtree(d, ignore_errors=True)
        removed.append(d.name)
    return removed


def _pruner(output_dir: Path, keep: int, stop: threading.Event, every_s: float = 60,
            protect: set[str] | Callable[[], set[str]] | None = None) -> threading.Thread:
    def loop() -> None:
        while not stop.wait(every_s):
            try:
                prune_checkpoints(output_dir, keep, protect)
            except OSError as e:     # e.g. lerobot renaming files right now: try again next round
                print(f"checkpoint pruning skipped this round: {e}", flush=True)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def preflight() -> None:
    """`lerobot-train --help` must start: it imports every policy, so dependency clashes show up here
    (e.g. Colab's preinstalled torchao vs diffusers: `cannot import name 'FqnToConfig'`)."""
    exe = shutil.which("lerobot-train")
    if exe is None:
        raise PreflightError("lerobot-train not found: install LeRobot (pip install 'lerobot[smolvla,dataset]==0.6.1')")
    proc = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-6:])
        hint = ("\nFix: pip uninstall -y torchao  (SmolVLA doesn't need it; the preinstalled version clashes "
                "with diffusers)") if "FqnToConfig" in tail or "torchao" in tail else ""
        raise PreflightError(f"lerobot-train cannot start:\n{tail}{hint}")


def train(config_path: str, dataset_version: str, dry_run: bool = False, registry: Registry | None = None,
          resume_model: str | None = None, overrides: dict[str, Any] | None = None,
          purpose: str = "model", init_from: str | None = None) -> dict[str, Any]:
    """purpose: "model" or "timing" (a short run to measure speed — registered, but labelled as not a result).
    init_from: a trained model whose checkpoint is the starting point (fine-tune) instead of the config's
    policy.path; the config defaults to that model's own, and its trainer must match."""
    reg = registry or Registry()
    manifest = load_manifest(dataset_version, reg)
    if manifest["status"] != "approved":
        raise RuntimeError(f"{dataset_version} is '{manifest['status']}'. Approve it first: "
                           f"hackpipe dataset approve {dataset_version}")
    if config_path is None and resume_model:      # `--resume` alone (the Train page): the run's own config file
        config_path = ((reg.get_model(resume_model) or {}).get("meta") or {}).get("train_config_file")
    init = None
    if init_from:
        if resume_model:
            raise ValueError("--init-from starts a new model; it cannot be combined with --resume")
        init = reg.get_model(init_from)
        if init is None:
            raise KeyError(f"Unknown model {init_from}")
        if not (resolve(init["checkpoint_dir"] or ".") / "config.json").is_file():
            raise FileNotFoundError(f"{init_from} has no checkpoint on this machine ({init['checkpoint_dir']})")
        config_path = config_path or init["meta"].get("train_config_file")
    if config_path is None:
        raise ValueError("a train config is required")
    cfg = apply_overrides(load_yaml(config_path), overrides or {})
    if init:
        trainer = "curriculum" if is_curriculum(cfg) else "lerobot-train"
        if init["meta"].get("trainer", "lerobot-train") != trainer:
            raise ValueError(f"{init_from} was trained with {init['meta'].get('trainer', 'lerobot-train')}, "
                             f"{config_path} uses {trainer}: rate and chunk differ, pick a matching config")
        from hackpipe.training.ranking import init_leakage
        leak = init_leakage(init, manifest, reg)
        if leak:
            raise RuntimeError(f"{init_from} was trained on {len(leak)} test episodes of {dataset_version} "
                               f"(e.g. {leak[:3]}): its scores on this test split would not be honest")
        cfg["args"].pop("policy.type", None)          # the checkpoint's config sets the type
        cfg["args"]["policy.path"] = str(resolve(init["checkpoint_dir"]))
    outputs = resolve(project_config()["project"]["outputs_dir"])

    if resume_model and overrides:
        raise ValueError("--set cannot be combined with --resume: lerobot-train resumes from the saved "
                         "train_config.json, so the overrides would be recorded but not applied.")
    curriculum = None
    if resume_model:
        model = reg.get_model(resume_model)
        if model is None:
            raise KeyError(resume_model)
        if model["dataset_version"] != dataset_version:
            raise ValueError(f"{resume_model} was trained on {model['dataset_version']}, not {dataset_version}: "
                             f"resume with --dataset {model['dataset_version']}")
        version, output_dir = resume_model, Path(model["meta"]["output_dir"])
        if model["meta"].get("trainer") == "curriculum":      # resumes with the run's own saved config
            cfg = json.loads((output_dir / "train_config.json").read_text())
            cmd = curriculum_command(dataset_version, output_dir / "train_config.json", output_dir, resume=True)
        else:
            cmd = ["lerobot-train", "--resume=true",
                   f"--config_path={output_dir / 'checkpoints' / 'last' / 'pretrained_model' / 'train_config.json'}"]
    else:
        version = next_version([m["version"] for m in reg.models()], "m")
        output_dir = outputs / "train" / version
        if is_curriculum(cfg):
            from hackpipe.training.curriculum import describe
            curriculum = describe(cfg, manifest)          # fails early if the version has no sub-task labels
            name = "train_config_dryrun.json" if dry_run else "train_config.json"
            cmd = curriculum_command(dataset_version, outputs / "runs" / version / name, output_dir)
        else:
            root = project_config()["dataset"].get("root")
            cmd = build_command(cfg, manifest, output_dir, f"{version}-{cfg.get('policy_name', 'policy')}",
                                str(resolve(root)) if root else None)

    record = {
        "model_version": version,
        "dataset_version": dataset_version,
        "dataset_fingerprint": manifest["fingerprint"],
        "policy_name": cfg.get("policy_name"),
        "train_config_file": str(config_path),
        "train_config": cfg,
        "overrides": overrides or {},
        "init_from": init_from,
        "command": cmd,
        "output_dir": str(output_dir),
        "environment": environment(),
        "purpose": purpose,
        "trainer": "curriculum" if is_curriculum(cfg) else "lerobot-train",
        "started_at": now_iso(),
        "dry_run": dry_run,
    }
    if curriculum:
        record["curriculum"] = curriculum
        record["subtask_labels"] = manifest["subtasks"]["sha256"]
    run_dir = outputs / "runs" / version
    run_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / ("run_record_resume.json" if resume_model else "run_record.json")
    if curriculum:                      # the resolved config the trainer reads (the file named in the command)
        Path(cmd[cmd.index("--config") + 1]).write_text(json.dumps(cfg, indent=2))

    if dry_run:
        # Separate file so a dry run never looks like a real run record (the version is not registered).
        record_path.with_name("run_record_dryrun.json").write_text(json.dumps(record, indent=2))
        print(" ".join(cmd))
        return record

    preflight()                     # a broken LeRobot install fails here, before a model ID is used up
    # Register before launching, so a session that dies mid-run (e.g. Colab) can still be resumed by model ID.
    ckpt = output_dir / "checkpoints" / "last" / "pretrained_model"
    record_path.write_text(json.dumps(record, indent=2))
    if resume_model:
        reg.set_model_status(version, "training")
    else:
        reg.add_model(version, dataset_version, cfg.get("policy_name", "policy"), "training", str(ckpt),
                      str(record_path), {"output_dir": str(output_dir), "train_config_file": str(config_path),
                                         "purpose": purpose, "trainer": record["trainer"],
                                         **({"init_from": init_from} if init_from else {})})
    print(f"{version}: training started — resume after a crash with "
          f"`hackpipe train --dataset {dataset_version} --resume {version}`", flush=True)
    t0 = time.time()
    keep = int(cfg.get("keep_checkpoints") or 0)          # 0 = keep all
    reg_path = reg.path

    def protect() -> set[str]:          # re-read each round: the best checkpoint can be chosen while it trains
        r = Registry(reg_path)          # own connection: the pruner runs in another thread
        try:
            m = r.get_model(version) or {}
        finally:
            r.close()
        keep_dirs = {f"{int(m['meta']['best_step']):06d}"} if (m.get("meta") or {}).get("best_step") is not None else set()
        from hackpipe.evaluation.offline import step_of
        if step_of(m.get("checkpoint_dir")) is not None:
            keep_dirs.add(f"{step_of(m['checkpoint_dir']):06d}")
        return keep_dirs
    stop = threading.Event()
    if keep:
        _pruner(output_dir, keep, stop, protect=protect)
    try:
        proc = subprocess.run(cmd)
    except BaseException as e:                 # lerobot-train missing, Ctrl+C, …: never leave 'training' behind
        record.update({"ended_at": now_iso(), "duration_s": round(time.time() - t0, 1), "exit_code": None,
                       "error": f"{type(e).__name__}: {e}"})
        record_path.write_text(json.dumps(record, indent=2))
        reg.set_model_status(version, "failed")
        raise
    finally:
        stop.set()
        if keep:
            prune_checkpoints(output_dir, keep, protect)
    record.update({"ended_at": now_iso(), "duration_s": round(time.time() - t0, 1), "exit_code": proc.returncode})
    record_path.write_text(json.dumps(record, indent=2))
    reg.set_model_status(version, "trained" if proc.returncode == 0 else "failed")
    return record


def register_external(checkpoint_dir: str, dataset_version: str, policy_name: str,
                      registry: Registry | None = None, notes: str = "") -> str:
    """Register a checkpoint trained outside the wrapper (e.g. pi0.5 LoRA via openpi)."""
    reg = registry or Registry()
    load_manifest(dataset_version, reg)
    version = next_version([m["version"] for m in reg.models()], "m")
    run_dir = resolve(project_config()["project"]["outputs_dir"]) / "runs" / version
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {"model_version": version, "dataset_version": dataset_version, "policy_name": policy_name,
              "checkpoint_dir": checkpoint_dir, "external": True, "notes": notes,
              "environment": environment(), "registered_at": now_iso()}
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=2))
    reg.add_model(version, dataset_version, policy_name, "trained", checkpoint_dir,
                  str(run_dir / "run_record.json"), {"external": True, "notes": notes})
    return version
