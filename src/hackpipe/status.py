"""Pipeline status for the console's global page: one card per challenge stage, what to do next, known gaps.
Everything is read from the registry and the dataset manifests — no numbers from anywhere else."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from hackpipe.common.registry import Registry

DONE, ACTIVE, WAITING, TODO = "done", "active", "waiting", "todo"


def _vnum(v: str) -> int:
    try:
        return int(v.rsplit("-v", 1)[1])
    except (IndexError, ValueError):
        return 0


def overview(registry: Registry | None = None) -> dict[str, Any]:
    reg = registry or Registry()
    demos = reg.demos()
    counts = Counter(d["status"] for d in demos)
    datasets = sorted(reg.datasets(), key=lambda d: _vnum(d["version"]))
    approved = [d for d in datasets if d["status"] == "approved"]
    latest_ds = datasets[-1] if datasets else None
    models = sorted(reg.models(), key=lambda m: _vnum(m["version"]))
    evals = reg.evals()
    deployed = reg.current_deployed_model()
    training = [m for m in models if m["status"] == "training"]

    stages = [
        {"id": "01", "title": "Ingest & Transform",
         "status": DONE if demos else TODO,
         "lines": [f"{len(demos)} demos", f"{len({d['source'] for d in demos})} source(s)"]},
        {"id": "02", "title": "Dataset Intelligence",
         "status": DONE if approved else (WAITING if latest_ds else (ACTIVE if demos else TODO)),
         "lines": [f"{counts.get('VALID', 0)} valid · {counts.get('REVIEW', 0)} review · "
                   f"{counts.get('INVALID', 0)} invalid · {counts.get('DUPLICATE', 0)} duplicate",
                   f"latest {latest_ds['version']} ({latest_ds['status']})" if latest_ds else "no dataset version"]},
        {"id": "03", "title": "Fine-Tune",
         "status": ACTIVE if training else (DONE if any(m["status"] == "trained" for m in models) else TODO),
         "lines": [f"{len(models)} model(s)",
                   f"latest {models[-1]['version']} ({models[-1]['status']})" if models else "no model yet"]},
        {"id": "04", "title": "Evaluate",
         "status": DONE if evals else TODO,
         "lines": [f"{len(evals)} evaluation(s)"]},
        {"id": "05", "title": "Version & Serve",
         "status": DONE if deployed else TODO,
         "lines": [f"deployed: {deployed or 'none'}"]},
    ]
    return {"stages": stages, "next": next_steps(demos, counts, datasets, models, evals, deployed),
            "known_gaps": known_gaps_of(approved[-1] if approved else latest_ds, reg),
            "versions": {"dataset": latest_ds["version"] if latest_ds else None,
                         "dataset_status": latest_ds["status"] if latest_ds else None,
                         "approved_dataset": approved[-1]["version"] if approved else None,
                         "model": models[-1]["version"] if models else None, "deployed": deployed}}


def next_steps(demos, counts, datasets, models, evals, deployed) -> list[str]:
    if not demos:
        return ["Add demonstrations: Data → Add demonstrations."]
    if counts.get("imported"):
        return [f"{counts['imported']} demo(s) imported but not quality-checked: run the pipeline on their folder."]
    drafts = [d for d in datasets if d["status"] == "draft"]
    if not datasets:
        return ["Create the first dataset version: Data → Versions."]
    if drafts:
        return [f"Review and approve {drafts[-1]['version']}: Data → Versions."]
    if not models:
        return ["Train model_v1 on the approved dataset (Colab: dry run, then timing run)."]
    if any(m["status"] == "training" for m in models):
        return ["Training is running: watch the jobs panel or the Colab log."]
    if not evals:
        return ["Evaluate the latest trained model."]
    if not deployed:
        return ["Check gates and the quality statement, then deploy."]
    return ["Add new demonstrations to start the next iteration."]


def known_gaps_of(dataset: dict[str, Any] | None, reg: Registry) -> list[str]:
    if not dataset:
        return []
    from hackpipe.dataset.versioning import known_gaps
    try:
        manifest = json.loads(Path(dataset["manifest_path"]).read_text())
    except (OSError, ValueError):
        return []
    ids = {i for v in manifest["demo_ids"].values() for i in v}
    return known_gaps([d for d in reg.demos() if d["demo_id"] in ids])
