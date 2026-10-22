"""R2 — Apply check results to the registry and write the data quality report (JSON + Markdown).

One report per dataset folder: outputs/qc/<folder name>/qc_report.{json,md}.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from hackpipe.common.config import dataset_config, project_config, resolve, source_key
from hackpipe.common.ids import demo_id, now_iso
from hackpipe.common.registry import Registry
from hackpipe.quality.checks import INVALID, REVIEW, VALID, EpisodeCheck, check_dataset


def run_qc(root: str | Path, registry: Registry | None = None, out_dir: str | Path | None = None,
           full: bool = False) -> dict[str, Any]:
    """Incremental by default: checks the source's demos that are new or changed (status "imported");
    earlier results are reused. full=True re-checks every episode. DUPLICATE demos are never checked."""
    reg = registry or Registry()
    source = source_key(root)
    registered = {d["episode_index"]: d for d in reg.demos(source=source)}
    todo = None if full or not registered else {i for i, d in registered.items() if d["status"] == "imported"}
    result = check_dataset(root, only_episodes=todo)
    accept_review = dataset_config()["quality"].get("accept_review_as_valid", False)
    checked = {e.episode_index for e in result["episodes"]}
    for e in result["episodes"]:
        did = demo_id(source, e.episode_index)
        if reg.get_demo(did) is None or reg.get_demo(did)["status"] == "DUPLICATE":
            continue  # not ingested yet / duplicate of a demo from another folder
        status = VALID if (e.status == REVIEW and accept_review) else e.status
        reg.set_demo_status(did, status, {"qc": {"status": e.status, "hard": e.hard, "soft": e.soft,
                                                 "stats": e.stats, "checked_at": now_iso()}})
    for e in result["episodes"]:
        if reg.get_demo(demo_id(source, e.episode_index)) is None:
            result["dataset_issues"].append("some episodes are not in the registry yet: run `hackpipe ingest` first")
            break
    reused = [_stored_check(d) for i, d in sorted(registered.items())
              if i not in checked and d["status"] != "DUPLICATE" and d["meta"].get("qc")]
    result["episodes"] = sorted(result["episodes"] + reused, key=lambda e: e.episode_index)
    report = summarize(result)
    report["checked_now"], report["reused"] = len(checked), len(reused)
    report["source"] = source                     # the registry's name (data/raw/…), not the resolved path
    report["duplicates"] = sorted(i for i, d in registered.items() if d["status"] == "DUPLICATE")
    out = resolve(out_dir or Path(project_config()["project"]["outputs_dir"]) / "qc" / report_name(source))
    out.mkdir(parents=True, exist_ok=True)
    (out / "qc_report.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "qc_report.md").write_text(to_markdown(report))
    report["paths"] = {"json": str(out / "qc_report.json"), "md": str(out / "qc_report.md")}
    return report


def _stored_check(demo: dict[str, Any]) -> EpisodeCheck:
    qc = demo["meta"]["qc"]
    return EpisodeCheck(demo["episode_index"], qc["status"], list(qc["hard"]), list(qc["soft"]), dict(qc["stats"]))


def report_name(source: str) -> str:
    """Folder name for a dataset's report: its name inside the repo; name + path hash outside it, so two
    folders with the same name elsewhere don't overwrite each other."""
    name = Path(source).name
    return name if not Path(source).is_absolute() else f"{name}-{hashlib.sha256(source.encode()).hexdigest()[:8]}"


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    eps = result["episodes"]
    statuses = Counter(e.status for e in eps)
    reasons = Counter(r.split(" ")[0] for e in eps for r in e.hard + e.soft)
    lengths = np.array([e.stats.get("n_frames", 0) for e in eps]) if eps else np.array([0])
    return {
        "generated_at": now_iso(),
        "root": result["root"],
        "source": source_key(result["root"]),
        "fps": result["fps"],
        "n_episodes": result["n_episodes"],
        "n_frames": result["n_frames"],
        "status_counts": {s: statuses.get(s, 0) for s in (VALID, REVIEW, INVALID)},
        "reason_counts": dict(reasons.most_common()),
        "episode_frames": {"min": int(lengths.min()), "median": float(np.median(lengths)),
                           "max": int(lengths.max())},
        "dataset_issues": result["dataset_issues"],
        "redundancy": result.get("redundancy"),
        "videos": result["videos"],
        "episodes": [{"episode_index": e.episode_index, "status": e.status, "hard": e.hard, "soft": e.soft}
                     for e in eps],
    }


def to_markdown(r: dict[str, Any]) -> str:
    lines = [
        "# Data quality report", "",
        f"- Generated: {r['generated_at']}",
        f"- Source: `{r.get('source', r['root'])}` (files: `{r['root']}`)",
        f"- Episodes: {r['n_episodes']} · Frames: {r['n_frames']} · FPS: {r['fps']}",
        f"- Checked in this run: {r.get('checked_now', r['n_episodes'])} · reused earlier results: "
        f"{r.get('reused', 0)} · duplicates (not checked): {len(r.get('duplicates') or [])}",
        f"- Episode length (frames): min {r['episode_frames']['min']}, median {r['episode_frames']['median']:.0f}, "
        f"max {r['episode_frames']['max']}", "",
        "| Status | Episodes |", "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in r["status_counts"].items()]
    lines += ["", "## Reasons", "", "| Reason | Episodes |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in r["reason_counts"].items()] or ["| none | 0 |"]
    lines += ["", "## Dataset-level issues", ""]
    lines += [f"- {i}" for i in r["dataset_issues"]] or ["- none"]
    red = r.get("redundancy") or {}
    if red:
        lines += ["", "## Redundancy", "",
                  f"- {red['distinct_trajectories']} distinct state/action trajectories in {r['n_episodes']} episodes; "
                  f"largest group of identical episodes: {red['largest_identical_group']}.",
                  "- Identical trajectories differ at most in the camera images (e.g. lighting)."
                  if red["distinct_trajectories"] < r["n_episodes"] else "- No two episodes share a trajectory."]
    vids = r.get("videos") or {}
    if vids.get("decode_check") == "done":
        lines += ["", "## Videos", "", f"- {vids['checked_files']} files decoded start to end, "
                  f"{vids['decoded_frames']} frames; corrupted: {len(vids['corrupted'])}."]
    flagged = [e for e in r["episodes"] if e["status"] != VALID]
    if flagged:
        lines += ["", "## Flagged episodes", "", "| Episode | Status | Reasons |", "|---|---|---|"]
        lines += [f"| {e['episode_index']} | {e['status']} | {', '.join(e['hard'] + e['soft'])} |" for e in flagged]
    return "\n".join(lines) + "\n"
