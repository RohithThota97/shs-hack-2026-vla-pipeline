"""Publish a trained model and its model card to the Hugging Face Hub (model only — never the dataset).

What goes up: the checkpoint's `pretrained_model/` files, a README.md model card built from the run record,
the dataset manifest summary (version, fingerprint, episode counts — no data) and, once evaluated, the quality
statement (R7). Repos are private unless `--public` is given. Files that look like data are refused.
The Hub repo and commit are written back into the registry (models.meta.hub).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from hackpipe.common.config import project_config, resolve
from hackpipe.common.ids import now_iso
from hackpipe.common.registry import Registry
from hackpipe.dataset.versioning import known_gaps, load_manifest

DATA_SUFFIXES = {".parquet", ".mp4", ".hdf5", ".h5", ".zip", ".db"}


def _record(model: dict[str, Any]) -> dict[str, Any]:
    p = model.get("run_record_path")
    return json.loads(Path(p).read_text()) if p and Path(p).exists() else {}


def _final_loss(run_dir: Path) -> str:
    p = run_dir / "loss.csv"
    if not p.exists():
        return "n/a"
    rows = [r for r in p.read_text().splitlines()[1:] if r.strip()]
    return rows[-1].split(",")[-1] if rows else "n/a"


def build_model_card(model_version: str, repo_id: str, registry: Registry | None = None) -> str:
    reg = registry or Registry()
    model = reg.get_model(model_version)
    if model is None:
        raise KeyError(model_version)
    rec = _record(model)
    manifest = load_manifest(model["dataset_version"], reg)
    args = (rec.get("train_config") or {}).get("args") or {}
    base = args.get("policy.path") or "n/a"
    env = rec.get("environment") or {}
    counts = {k: len(v) for k, v in manifest["splits"].items()}
    demo_ids = {i for v in manifest["demo_ids"].values() for i in v}
    gaps = known_gaps([d for d in reg.demos() if d["demo_id"] in demo_ids])
    evals = reg.evals(model_version)
    run_dir = Path(model["run_record_path"]).parent if model.get("run_record_path") else Path(".")

    front = ["---", "library_name: lerobot", "pipeline_tag: robotics",
             "tags: [robotics, lerobot, " + f"{model['policy_name']}, fr3_duo, vla]"]
    if base != "n/a":
        front.append(f"base_model: {base}")
    front.append("---")
    lines = front + [
        "", f"# {repo_id.split('/')[-1]} · {model['version']}", "",
        f"{model['policy_name']} policy for the FR3 Duo dual-arm robot (simulation), fine-tuned from `{base}` on "
        "demonstrations provided by the SHS Hackathon 2026 organisers (Challenge 2). "
        "The training data is **not** published with this model.", "",
    ]
    if not evals:
        lines += ["> **Not evaluated yet.** No success rate exists for this model. Do not use it on a robot.", ""]
    lines += [
        "## Identity", "", "| Field | Value |", "|---|---|",
        f"| Model version | {model['version']} ({model['policy_name']}) |",
        f"| Base model | `{base}` |",
        f"| Dataset version | {manifest['version']} · fingerprint `{manifest['fingerprint']}` |",
        f"| Episodes (train / val / held-out test) | {counts.get('train', 0)} / {counts.get('val', 0)} / "
        f"{counts.get('test', 0)} |",
        f"| Task | {', '.join(sorted({str(d['meta'].get('task')) for d in reg.demos() if d['demo_id'] in demo_ids}))} |",
        f"| Steps · batch size · seed | {args.get('steps', 'n/a')} · {args.get('batch_size', 'n/a')} · "
        f"{(rec.get('train_config') or {}).get('seed', 'n/a')} |",
        f"| Overrides | `{json.dumps(rec.get('overrides') or {})}` |",
        f"| Training time | {rec.get('duration_s', 'n/a')} s · exit code {rec.get('exit_code', 'n/a')} |",
        f"| Final training loss | {_final_loss(run_dir)} |",
        f"| Code version | {(env.get('git') or {}).get('commit', 'n/a')} |",
        f"| GPU | {env.get('gpu') or 'n/a'} |", "",
        "## Inputs and outputs", "",
        f"- Cameras: {', '.join((manifest['preprocessing'].get('schema') or {}).get('cameras', {}) or ['n/a'])}",
        f"- Camera rename map: `{json.dumps(manifest['preprocessing']['preprocessing'].get('rename_map') or {})}`",
        f"- State / action: {(manifest['preprocessing'].get('schema') or {}).get('state', {}).get('shape')} / "
        f"{(manifest['preprocessing'].get('schema') or {}).get('action', {}).get('shape')} — left q1..q7, left gripper, "
        "right q1..q7, right gripper; joint targets in rad, gripper 1 = open, 0 = closed",
        f"- FPS: {(manifest['preprocessing'].get('schema') or {}).get('fps', 'n/a')}", "",
        "## Evaluation", "",
    ]
    if evals:
        from hackpipe.reporting.model_card import build_card
        latest = sorted(evals, key=lambda e: e["created_at"])[-1]
        lines += [build_card(latest["eval_id"], reg).replace("# Quality statement", "### Quality statement"), ""]
    else:
        lines += ["n/a — not evaluated yet.", ""]
    lines += ["## Known limitations", ""] + [f"- {g}" for g in gaps] + [
        "- Trained and (if evaluated) tested in simulation only.", "",
        "## Load", "", "```python",
        "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy",
        f"policy = SmolVLAPolicy.from_pretrained(\"{repo_id}\")",
        "```", "",
        "Run record (`run_record.json`) and dataset summary (`dataset_summary.json`) are in this repo.", "",
    ]
    return "\n".join(lines)


def stage(model_version: str, repo_id: str, registry: Registry | None = None) -> tuple[Path, list[Path]]:
    """Write card + metadata to outputs/hub/<model>/ and list the checkpoint files to upload."""
    reg = registry or Registry()
    model = reg.get_model(model_version)
    if model is None:
        raise KeyError(model_version)
    ckpt = Path(model["checkpoint_dir"] or "")
    if not (ckpt / "config.json").exists():
        raise FileNotFoundError(f"No policy checkpoint at {ckpt} (needs config.json + weights)")
    out = resolve(Path(project_config()["project"]["outputs_dir"]) / "hub" / model_version)
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(build_model_card(model_version, repo_id, reg))
    manifest = load_manifest(model["dataset_version"], reg)
    (out / "dataset_summary.json").write_text(json.dumps(
        {k: manifest[k] for k in ("version", "fingerprint", "created_at", "splits", "notes")} |
        {"preprocessing": manifest["preprocessing"].get("schema"),
         "normalisation_stats_sha256": (manifest["preprocessing"].get("normalisation_stats") or {}).get("sha256")},
        indent=2))
    run_dir = Path(model["run_record_path"]).parent if model.get("run_record_path") else None
    for name in ("run_record.json", "loss.csv", "loss.png"):
        if run_dir and (run_dir / name).exists():
            shutil.copyfile(run_dir / name, out / name)
    ckpt_files = sorted(p for p in ckpt.rglob("*") if p.is_file())
    bad = [p for p in ckpt_files + sorted(out.iterdir()) if p.suffix.lower() in DATA_SUFFIXES]
    if bad:
        raise RuntimeError(f"Refusing to upload data-like files: {[str(p) for p in bad]}")
    return out, ckpt_files


def publish(model_version: str, repo_id: str | None = None, private: bool = True, dry_run: bool = False,
            registry: Registry | None = None, api: Any = None) -> dict[str, Any]:
    reg = registry or Registry()
    repo_id = repo_id or (project_config().get("hub") or {}).get("model_repo_id")
    if not repo_id:
        raise ValueError("No repo id: pass --repo <hf-user>/<name> or set hub.model_repo_id in configs/project.yaml")
    out, ckpt_files = stage(model_version, repo_id, reg)
    model = reg.get_model(model_version)
    ckpt = Path(model["checkpoint_dir"])
    plan = {"repo_id": repo_id, "private": private, "staged": str(out),
            "files": [p.relative_to(ckpt).as_posix() for p in ckpt_files] + sorted(p.name for p in out.iterdir())}
    if dry_run:
        return plan | {"uploaded": False}

    if api is None:
        from huggingface_hub import HfApi  # optional dependency: pip install -e ".[hub]"
        api = HfApi()
    from huggingface_hub import CommitOperationAdd
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=p.relative_to(ckpt).as_posix(), path_or_fileobj=str(p)) for p in ckpt_files]
    ops += [CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=str(p)) for p in sorted(out.iterdir())]
    info = api.create_commit(repo_id=repo_id, repo_type="model", operations=ops,
                             commit_message=f"{model_version}: checkpoint + model card")
    commit = getattr(info, "oid", None)
    hub = {"repo_id": repo_id, "private": private, "commit": commit, "published_at": now_iso()}
    reg.update_model_meta(model_version, {"hub": hub})
    return plan | {"uploaded": True, "commit": commit, "url": f"https://huggingface.co/{repo_id}"}
