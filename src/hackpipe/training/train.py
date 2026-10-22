"""R4 — Reproducible training.

Builds a `lerobot-train` command from a YAML config and an approved dataset version, saves a complete run
record (command, config, dataset fingerprint, git commit, packages, GPU), runs it, and registers the model.
LeRobot itself writes train_config.json into every checkpoint; resume with --resume.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from hackpipe.common.config import load_yaml, project_config, resolve
from hackpipe.common.envinfo import environment
from hackpipe.common.ids import next_version, now_iso
from hackpipe.common.registry import Registry
from hackpipe.dataset.versioning import load_manifest


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


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """`seed` and `policy_name` are top-level config keys; everything else is a lerobot-train arg."""
    cfg = {**cfg, "args": dict(cfg.get("args") or {})}
    for k, v in overrides.items():
        if k in ("seed", "policy_name"):
            cfg[k] = v
        else:
            cfg["args"][k] = v
    return cfg


def training_input(manifest: dict[str, Any], dataset_root: str | None = None) -> dict[str, Any]:
    """Which folder, repo id and episode indices training reads: the merged folder for multi-source versions,
    else the version's single source (falling back to the project dataset root for old manifests)."""
    mat = manifest.get("materialized")
    if mat:
        return {"repo_id": mat["repo_id"], "root": str(resolve(mat["root"])), "episodes": mat["splits"]["train"]}
    if manifest.get("needs_materialize"):
        raise RuntimeError(f"{manifest['version']} has several sources; merge them first: "
                           f"hackpipe dataset materialize {manifest['version']}")
    src = manifest.get("source")
    root = str(resolve(src)) if src else dataset_root
    return {"repo_id": manifest["repo_id"], "root": root, "episodes": manifest["splits"]["train"]}


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


def train(config_path: str, dataset_version: str, dry_run: bool = False, registry: Registry | None = None,
          resume_model: str | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    reg = registry or Registry()
    manifest = load_manifest(dataset_version, reg)
    if manifest["status"] != "approved":
        raise RuntimeError(f"{dataset_version} is '{manifest['status']}'. Approve it first: "
                           f"hackpipe dataset approve {dataset_version}")
    cfg = apply_overrides(load_yaml(config_path), overrides or {})
    outputs = resolve(project_config()["project"]["outputs_dir"])

    if resume_model and overrides:
        raise ValueError("--set cannot be combined with --resume: lerobot-train resumes from the saved "
                         "train_config.json, so the overrides would be recorded but not applied.")
    if resume_model:
        model = reg.get_model(resume_model)
        if model is None:
            raise KeyError(resume_model)
        version, output_dir = resume_model, Path(model["meta"]["output_dir"])
        cmd = ["lerobot-train", "--resume=true",
               f"--config_path={output_dir / 'checkpoints' / 'last' / 'pretrained_model' / 'train_config.json'}"]
    else:
        version = next_version([m["version"] for m in reg.models()], "m")
        output_dir = outputs / "train" / version
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
        "command": cmd,
        "output_dir": str(output_dir),
        "environment": environment(),
        "started_at": now_iso(),
        "dry_run": dry_run,
    }
    run_dir = outputs / "runs" / version
    run_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / ("run_record_resume.json" if resume_model else "run_record.json")

    if dry_run:
        # Separate file so a dry run never looks like a real run record (the version is not registered).
        record_path.with_name("run_record_dryrun.json").write_text(json.dumps(record, indent=2))
        print(" ".join(cmd))
        return record

    # Register before launching, so a session that dies mid-run (e.g. Colab) can still be resumed by model ID.
    ckpt = output_dir / "checkpoints" / "last" / "pretrained_model"
    record_path.write_text(json.dumps(record, indent=2))
    if resume_model:
        reg.set_model_status(version, "training")
    else:
        reg.add_model(version, dataset_version, cfg.get("policy_name", "policy"), "training", str(ckpt),
                      str(record_path), {"output_dir": str(output_dir), "train_config_file": str(config_path)})
    print(f"{version}: training started — resume after a crash with "
          f"`hackpipe train --dataset {dataset_version} --resume {version}`", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd)
    except BaseException as e:                 # lerobot-train missing, Ctrl+C, …: never leave 'training' behind
        record.update({"ended_at": now_iso(), "duration_s": round(time.time() - t0, 1), "exit_code": None,
                       "error": f"{type(e).__name__}: {e}"})
        record_path.write_text(json.dumps(record, indent=2))
        reg.set_model_status(version, "failed")
        raise
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
