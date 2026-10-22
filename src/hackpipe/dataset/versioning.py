"""R3 — Create a versioned dataset manifest from approved demos.

- Splits by episode (never by frame), optionally keeping whole groups (e.g. task) in one split.
- Fixed seed; explicit held-out test episodes supported.
- Leakage guard: a test demo or test group can never appear in train/val.
- Fingerprint = hash(sorted demo content checksums + preprocessing config).
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hackpipe.common.config import dataset_config, project_config, resolve
from hackpipe.common.ids import fingerprint, next_version, now_iso
from hackpipe.common.registry import Registry
from hackpipe.quality.checks import VALID


class LeakageError(RuntimeError):
    pass


def split_demos(demos: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    s = cfg["splits"]
    rng = random.Random(s["seed"])
    explicit_test = set(s.get("test_episodes") or [])
    group_key = s.get("group_key")

    test = [d for d in demos if d["episode_index"] in explicit_test]
    rest = [d for d in demos if d["episode_index"] not in explicit_test]

    if group_key:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for d in rest:
            groups[str(d["meta"].get(group_key))].append(d)
        keys = sorted(groups)
        rng.shuffle(keys)
        units = [groups[k] for k in keys]
    else:
        units = [[d] for d in sorted(rest, key=lambda d: (d["source"], d["episode_index"]))]
        rng.shuffle(units)

    n_total = sum(len(u) for u in units) + len(test)
    targets = {"test": 0 if explicit_test else round(s["test"] * n_total), "val": round(s["val"] * n_total)}
    out = {"train": [], "val": [], "test": list(test)}
    for unit in units:
        if len(out["test"]) < targets["test"]:
            out["test"] += unit
        elif len(out["val"]) < targets["val"]:
            out["val"] += unit
        else:
            out["train"] += unit
    return out


def leakage_guard(splits: dict[str, list[dict[str, Any]]], group_key: str | None) -> None:
    ids = {k: {d["demo_id"] for d in v} for k, v in splits.items()}
    for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
        overlap = ids[a] & ids[b]
        if overlap:
            raise LeakageError(f"{len(overlap)} demos in both {a} and {b}")
    if group_key:
        test_groups = {str(d["meta"].get(group_key)) for d in splits["test"]}
        for k in ("train", "val"):
            bad = test_groups & {str(d["meta"].get(group_key)) for d in splits[k]}
            if bad:
                raise LeakageError(f"test {group_key} values also in {k}: {sorted(bad)[:5]}")


def create_dataset_version(registry: Registry | None = None, cfg: dict[str, Any] | None = None,
                           notes: str = "") -> dict[str, Any]:
    reg = registry or Registry()
    cfg = cfg or dataset_config()
    demos = reg.demos(status=VALID)
    if not demos:
        raise RuntimeError("No VALID demos. Run `hackpipe ingest` and `hackpipe qc` first.")
    sources = sorted({d["source"] for d in demos})
    if len(sources) > 1:
        raise RuntimeError(f"Demos from several sources {sources}; merge them into one dataset first "
                           "(lerobot-edit-dataset) or filter by source.")
    splits = split_demos(demos, cfg)
    leakage_guard(splits, cfg["splits"].get("group_key"))

    preprocessing = {"columns": cfg["columns"], "preprocessing": cfg["preprocessing"], "splits": cfg["splits"]}
    fp = fingerprint([d["meta"].get("content_sha256", d["demo_id"]) for d in demos], preprocessing)
    version = next_version([d["version"] for d in reg.datasets()], "ds")
    excluded = [{"demo_id": d["demo_id"], "episode_index": d["episode_index"], "status": d["status"],
                 "reasons": d["meta"].get("qc", {}).get("hard", []) + d["meta"].get("qc", {}).get("soft", [])}
                for d in reg.demos() if d["status"] != VALID]
    manifest = {
        "version": version,
        "fingerprint": fp,
        "created_at": now_iso(),
        "source": sources[0],
        "repo_id": project_config()["dataset"]["repo_id"],
        "splits": {k: sorted(d["episode_index"] for d in v) for k, v in splits.items()},
        "demo_ids": {k: sorted(d["demo_id"] for d in v) for k, v in splits.items()},
        "excluded": excluded,
        "preprocessing": preprocessing,
        "notes": notes,
    }
    out_dir = resolve(Path(project_config()["project"]["outputs_dir"]) / "datasets" / version)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (out_dir / "dataset_card.md").write_text(dataset_card(manifest, splits))
    reg.add_dataset(version, fp, str(manifest_path), {"counts": {k: len(v) for k, v in splits.items()},
                                                      "card": str(out_dir / "dataset_card.md")})
    return manifest


def dataset_card(manifest: dict[str, Any], splits: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        f"# Dataset card · {manifest['version']}", "",
        f"- Fingerprint: `{manifest['fingerprint']}`",
        f"- Created: {manifest['created_at']}",
        f"- Source: `{manifest['source']}`",
        f"- Split seed: {manifest['preprocessing']['splits']['seed']}", "",
        "| Split | Episodes | Frames |", "|---|---|---|",
    ]
    for k in ("train", "val", "test"):
        frames = sum(int(d["meta"].get("n_frames", 0)) for d in splits[k])
        lines.append(f"| {k} | {len(splits[k])} | {frames} |")
    tasks = Counter(str(d["meta"].get("task")) for v in splits.values() for d in v)
    lines += ["", "## Episodes per task", "", "| Task | Episodes |", "|---|---|"]
    lines += [f"| {t} | {n} |" for t, n in tasks.most_common()]
    lines += ["", f"## Excluded demos ({len(manifest['excluded'])})", ""]
    lines += [f"- episode {e['episode_index']}: {e['status']} — {', '.join(e['reasons']) or 'no reason recorded'}"
              for e in manifest["excluded"][:50]] or ["- none"]
    lines += ["", "## Known gaps", "", "- Fill in after review (e.g. layouts or item positions with few demos).", ""]
    return "\n".join(lines)


def approve(version: str, registry: Registry | None = None) -> None:
    (registry or Registry()).set_dataset_status(version, "approved")


def load_manifest(version: str, registry: Registry | None = None) -> dict[str, Any]:
    d = (registry or Registry()).get_dataset(version)
    if d is None:
        raise KeyError(f"Unknown dataset version {version}")
    return {**json.loads(Path(d["manifest_path"]).read_text()), "status": d["status"]}
