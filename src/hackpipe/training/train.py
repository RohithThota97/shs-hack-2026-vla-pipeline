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


def build_command(train_cfg: dict[str, Any], manifest: dict[str, Any], output_dir: Path, job_name: str,
                  dataset_root: str | None) -> list[str]:
    cmd = ["lerobot-train", f"--dataset.repo_id={manifest['repo_id']}"]
    if dataset_root:
        cmd.append(f"--dataset.root={dataset_root}")
    cmd.append(f"--dataset.episodes={_fmt(manifest['splits']['train'])}")
    cmd += [f"--output_dir={output_dir}", f"--job_name={job_name}", f"--seed={train_cfg.get('seed', 1000)}"]
    rename = manifest["preprocessing"]["preprocessing"].get("rename_map") or {}
    if rename:
        cmd.append(f"--rename_map={_fmt(rename)}")
    for k, v in (train_cfg.get("args") or {}).items():
        cmd.append(f"--{k}={_fmt(v)}")
    return cmd


def train(config_path: str, dataset_version: str, dry_run: bool = False, registry: Registry | None = None,
          resume_model: str | None = None) -> dict[str, Any]:
    reg = registry or Registry()
    manifest = load_manifest(dataset_version, reg)
    if manifest["status"] != "approved":
        raise RuntimeError(f"{dataset_version} is '{manifest['status']}'. Approve it first: "
                           f"hackpipe dataset approve {dataset_version}")
    cfg = load_yaml(config_path)
    outputs = resolve(project_config()["project"]["outputs_dir"])

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
        cmd = build_command(cfg, manifest, output_dir, f"{version}-{cfg.get('policy_name', 'policy')}",
                            project_config()["dataset"].get("root"))

    record = {
        "model_version": version,
        "dataset_version": dataset_version,
        "dataset_fingerprint": manifest["fingerprint"],
        "policy_name": cfg.get("policy_name"),
        "train_config_file": str(config_path),
        "train_config": cfg,
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

    t0 = time.time()
    proc = subprocess.run(cmd)
    record.update({"ended_at": now_iso(), "duration_s": round(time.time() - t0, 1), "exit_code": proc.returncode})
    record_path.write_text(json.dumps(record, indent=2))
    status = "trained" if proc.returncode == 0 else "failed"
    ckpt = output_dir / "checkpoints" / "last" / "pretrained_model"
    if resume_model:
        reg.set_model_status(version, status)
    else:
        reg.add_model(version, dataset_version, cfg.get("policy_name", "policy"), status, str(ckpt),
                      str(record_path), {"output_dir": str(output_dir), "train_config_file": str(config_path)})
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
