"""R2 — Apply check results to the registry and write the data quality report (JSON + Markdown)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from hackpipe.common.config import dataset_config, project_config, resolve
from hackpipe.common.ids import demo_id, now_iso
from hackpipe.common.registry import Registry
from hackpipe.ingest.importer import build_demo_records
from hackpipe.quality.checks import INVALID, REVIEW, VALID, check_dataset


def run_qc(root: str | Path, registry: Registry | None = None, out_dir: str | Path | None = None,
           camera_report: str | Path | None = None) -> dict[str, Any]:
    reg = registry or Registry()
    source_before = build_demo_records(root)
    result = check_dataset(root) if camera_report is None else check_dataset(root, camera_report=camera_report)
    source = result["root"]
    current_records = build_demo_records(root)
    def signatures(records):
        return {record["episode_index"]: (record["status"], record["meta"].get("content_sha256"),
                tuple(record["meta"].get("ingest_errors", []))) for record in records}
    source_changed_during_qc = signatures(source_before) != signatures(current_records)
    if source_changed_during_qc:
        issue = "source content changed while QC was running"
        result["dataset_issues"].append(issue)
        result["blocking_issues"].append(issue)
        blockers = result["blocking_issues"]
        result["readiness"] = {"status": "BLOCKED", "ready_for_training": False,
                                "blocking_issues": blockers}
        for episode in result["episodes"]:
            episode.hard.append(f"dataset_blocked: {issue}")
            episode.finalize()
    current_hashes = {record["episode_index"]: record["meta"].get("content_sha256")
                      for record in current_records if record["status"] != "FAILED"}
    accept_review = dataset_config()["quality"].get("accept_review_as_valid", False)
    readiness = result["readiness"]
    for e in result["episodes"]:
        did = demo_id(source, e.episode_index)
        if reg.get_demo(did) is None:
            continue  # run `hackpipe ingest` first
        status = INVALID if readiness["status"] == "BLOCKED" else e.status
        unresolved_video = (bool(result["videos"].get("declared_keys")) and
                            result["videos"].get("decode_check", "").startswith("skipped"))
        if readiness["status"] == "REVIEW" and unresolved_video:
            status = REVIEW
        duplicate_numeric = any("potential_duplicate_numeric_trajectory" in reason for reason in e.soft)
        if (e.status == REVIEW and accept_review and readiness["status"] != "BLOCKED" and
                not unresolved_video and not duplicate_numeric):
            status = VALID
        reg.set_demo_status(did, status, {"qc": {"status": e.status, "hard": e.hard, "soft": e.soft,
                                                 "stats": e.stats, "checked_at": now_iso(),
                                                 "dataset_readiness": readiness,
                                                 "content_sha256": current_hashes.get(e.episode_index),
                                                 "videos": result["videos"],
                                                 "video_check_inconclusive": unresolved_video}})
    report = summarize(result)
    out = resolve(out_dir or Path(project_config()["project"]["outputs_dir"]) / "qc")
    out.mkdir(parents=True, exist_ok=True)
    (out / "qc_report.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "qc_report.md").write_text(to_markdown(report))
    report["paths"] = {"json": str(out / "qc_report.json"), "md": str(out / "qc_report.md")}
    return report


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    eps = result["episodes"]
    statuses = Counter(e.status for e in eps)
    reasons = Counter(r.split(" ")[0] for e in eps for r in e.hard + e.soft)
    lengths = np.array([e.stats.get("n_frames", 0) for e in eps]) if eps else np.array([0])
    return {
        "generated_at": now_iso(),
        "root": result["root"],
        "fps": result["fps"],
        "n_episodes": result["n_episodes"],
        "n_frames": result["n_frames"],
        "status_counts": {s: statuses.get(s, 0) for s in (VALID, REVIEW, INVALID)},
        "reason_counts": dict(reasons.most_common()),
        "episode_frames": {"min": int(lengths.min()), "median": float(np.median(lengths)),
                           "max": int(lengths.max())},
        "dataset_issues": result["dataset_issues"],
        "readiness": result["readiness"],
        "potential_duplicate_trajectories": result["potential_duplicate_trajectories"],
        "videos": result["videos"],
        "episodes": [{"episode_index": e.episode_index, "status": e.status, "hard": e.hard, "soft": e.soft}
                     for e in eps],
    }


def to_markdown(r: dict[str, Any]) -> str:
    lines = [
        "# Data quality report", "",
        f"- Generated: {r['generated_at']}",
        f"- Source: `{r['root']}`",
        f"- Episodes: {r['n_episodes']} · Frames: {r['n_frames']} · FPS: {r['fps']}",
        f"- Readiness: **{r['readiness']['status']}** (ready for training: {r['readiness']['ready_for_training']})",
        f"- Episode length (frames): min {r['episode_frames']['min']}, median {r['episode_frames']['median']:.0f}, "
        f"max {r['episode_frames']['max']}", "",
        "| Status | Episodes |", "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in r["status_counts"].items()]
    lines += ["", "## Reasons", "", "| Reason | Episodes |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in r["reason_counts"].items()] or ["| none | 0 |"]
    lines += ["", "## Dataset-level issues", ""]
    lines += [f"- {i}" for i in r["dataset_issues"]] or ["- none"]
    flagged = [e for e in r["episodes"] if e["status"] != VALID]
    if flagged:
        lines += ["", "## Flagged episodes", "", "| Episode | Status | Reasons |", "|---|---|---|"]
        lines += [f"| {e['episode_index']} | {e['status']} | {', '.join(e['hard'] + e['soft'])} |" for e in flagged]
    return "\n".join(lines) + "\n"
