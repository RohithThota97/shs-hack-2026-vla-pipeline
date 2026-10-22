"""R3 — Create a versioned dataset manifest from approved demos.

- Splits by episode (never by frame), optionally keeping whole groups (e.g. task) in one split.
- Fixed seed; held-out test = explicit episodes or whole held-out source datasets (`splits.test_sources`).
- Test split is frozen: once a dataset version is approved, later versions must keep exactly the same test demos.
- Leakage guard: a test demo, test group or test variation (`splits.leakage_keys`, e.g. seed) can never
  appear in train/val.
- Preprocessing record: fps, cameras, state/action keys and shapes, image size, rename map, normalisation stats
  (meta/stats.json, copied + hashed), excluded episodes.
- Fingerprint = hash(sorted demo content checksums + preprocessing record).
- Training selects the train episodes with --dataset.episodes, so episode indices stay those of the source.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hackpipe.common.config import dataset_config, project_config, resolve, source_key
from hackpipe.common.ids import fingerprint, next_version, now_iso
from hackpipe.common.lerobot_io import DatasetFolder
from hackpipe.common.registry import Registry
from hackpipe.quality.checks import VALID


class LeakageError(RuntimeError):
    pass


def _get(meta: dict[str, Any], dotted: str) -> Any:
    """_get(meta, "environment.seed") → meta["environment"]["seed"] or None."""
    cur: Any = meta
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def split_demos(demos: list[dict[str, Any]], cfg: dict[str, Any],
                held_out: list[dict[str, Any]] | None = None) -> dict[str, list[dict[str, Any]]]:
    """held_out: demos of held-out test sources; when given they are the whole test split."""
    s = cfg["splits"]
    rng = random.Random(s["seed"])
    explicit_test = set(s.get("test_episodes") or [])
    group_key = s.get("group_key")

    test = [d for d in demos if d["episode_index"] in explicit_test] + list(held_out or [])
    rest = [d for d in demos if d["episode_index"] not in explicit_test]
    if held_out:
        explicit_test = explicit_test or {None}     # test split is fixed; no ratio-based test from `rest`

    if group_key:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for d in rest:
            groups[str(_get(d["meta"], group_key))].append(d)
        keys = sorted(groups)
        rng.shuffle(keys)
        units = [groups[k] for k in keys]
    else:
        units = [[d] for d in sorted(rest, key=lambda d: (d["source"], d["episode_index"]))]
        rng.shuffle(units)

    n_total = sum(len(u) for u in units) + (0 if held_out else len(test))
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


def leakage_guard(splits: dict[str, list[dict[str, Any]]], group_key: str | None,
                  leakage_keys: list[str] | None = None) -> None:
    ids = {k: {d["demo_id"] for d in v} for k, v in splits.items()}
    for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
        overlap = ids[a] & ids[b]
        if overlap:
            raise LeakageError(f"{len(overlap)} demos in both {a} and {b}")
    for key in ([group_key] if group_key else []) + list(leakage_keys or []):
        test_vals = {str(_get(d["meta"], key)) for d in splits["test"] if _get(d["meta"], key) is not None}
        for k in ("train", "val"):
            bad = test_vals & {str(_get(d["meta"], key)) for d in splits[k] if _get(d["meta"], key) is not None}
            if bad:
                raise LeakageError(f"test {key} values also in {k}: {sorted(bad)[:5]}")


def demo_fingerprint_item(d: dict[str, Any]) -> str:
    """Demo identity + tabular content + checksums of its files (videos too: images can differ when
    trajectories are identical)."""
    files = d["meta"].get("files_sha256") or {}
    return "|".join([d["demo_id"], str(d["meta"].get("content_sha256")),
                     *(f"{k}={v}" for k, v in sorted(files.items()))])


def frozen_test_ids(registry: Registry) -> tuple[str, list[str]] | None:
    """(version, test demo_ids) of the first approved dataset version: the frozen test split.
    Drafts don't freeze anything, so a wrong draft can simply be replaced."""
    versions = sorted((d for d in registry.datasets() if d["status"] == "approved"),
                      key=lambda d: int(d["version"].split("-v", 1)[1]))
    if not versions:
        return None
    first = json.loads(Path(versions[0]["manifest_path"]).read_text())
    return versions[0]["version"], sorted(first["demo_ids"]["test"])


def preprocessing_record(source: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Everything training needs to reproduce the inputs: schema, camera mapping, normalisation stats."""
    ds = DatasetFolder(resolve(source))
    cols = cfg["columns"]
    stats = ds.stats_path()
    return {
        "columns": cols,
        "preprocessing": cfg["preprocessing"],
        "splits": cfg["splits"],
        "schema": {
            "fps": ds.fps,
            "cameras": {k: ds.features[k].get("shape") for k in ds.video_keys},
            "state": {"key": cols["state"], "shape": ds.features.get(cols["state"], {}).get("shape"),
                      "names": ds.features.get(cols["state"], {}).get("names")},
            "action": {"key": cols["action"], "shape": ds.features.get(cols["action"], {}).get("shape"),
                       "names": ds.features.get(cols["action"], {}).get("names")},
            "codebase_version": ds.info.get("codebase_version"),
        },
        "normalisation_stats": {"file": "meta/stats.json",
                                "sha256": hashlib.sha256(stats.read_bytes()).hexdigest() if stats.exists() else None},
    }


def create_dataset_version(registry: Registry | None = None, cfg: dict[str, Any] | None = None,
                           notes: str = "") -> dict[str, Any]:
    reg = registry or Registry()
    cfg = cfg or dataset_config()
    demos = reg.demos(status=VALID)
    if not demos:
        raise RuntimeError("No VALID demos. Run `hackpipe ingest` and `hackpipe qc` first.")
    test_sources = sorted({source_key(resolve(p)) for p in (cfg["splits"].get("test_sources") or [])})
    held_out = [d for d in demos if d["source"] in test_sources]
    demos = [d for d in demos if d["source"] not in test_sources]
    missing = [t for t in test_sources if t not in {d["source"] for d in held_out}]
    if missing:
        raise RuntimeError(f"Held-out test sources have no VALID demos: {missing}. Ingest + QC them first.")
    sources = sorted({d["source"] for d in demos})
    if len(sources) != 1:
        raise RuntimeError(f"Train/val must come from one source, found {sources or 'none'}; merge them first "
                           "(lerobot-edit-dataset --operation.type merge) or list extra ones in splits.test_sources.")
    splits = split_demos(demos, cfg, held_out)
    leakage_guard(splits, cfg["splits"].get("group_key"), cfg["splits"].get("leakage_keys"))
    frozen = frozen_test_ids(reg)
    if cfg["splits"].get("freeze_test", True) and frozen and frozen[1] != sorted(d["demo_id"] for d in splits["test"]):
        raise LeakageError(f"The test split is frozen since {frozen[0]}; this version's test demos differ. "
                           "Keep the split config, or start a new test suite on purpose.")

    preprocessing = preprocessing_record(sources[0], cfg)
    fp = fingerprint([demo_fingerprint_item(d) for d in demos + held_out], preprocessing)
    version = next_version([d["version"] for d in reg.datasets()], "ds")
    in_scope = {sources[0], *test_sources}
    excluded = [{"demo_id": d["demo_id"], "source": d["source"], "episode_index": d["episode_index"],
                 "status": d["status"],
                 "reasons": d["meta"].get("qc", {}).get("hard", []) + d["meta"].get("qc", {}).get("soft", [])}
                for d in reg.demos() if d["status"] != VALID and d["source"] in in_scope]
    manifest = {
        "version": version,
        "fingerprint": fp,
        "created_at": now_iso(),
        "source": sources[0],
        "test_sources": test_sources,
        "repo_id": project_config()["dataset"]["repo_id"],
        "revision": project_config()["dataset"].get("revision"),
        "splits": {k: sorted(d["episode_index"] for d in v) for k, v in splits.items()},
        "demo_ids": {k: sorted(d["demo_id"] for d in v) for k, v in splits.items()},
        "excluded": excluded,
        "preprocessing": preprocessing,
        "notes": notes,
    }
    out_dir = resolve(Path(project_config()["project"]["outputs_dir"]) / "datasets" / version)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = DatasetFolder(resolve(sources[0])).stats_path()
    if stats.exists():
        shutil.copyfile(stats, out_dir / "stats.json")
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (out_dir / "dataset_card.md").write_text(dataset_card(manifest, splits))
    reg.add_dataset(version, fp, str(manifest_path), {"counts": {k: len(v) for k, v in splits.items()},
                                                      "card": str(out_dir / "dataset_card.md")})
    return manifest


def _range(values: list[Any]) -> str:
    vals = [v for v in values if isinstance(v, (int, float))]
    return f"{min(vals):g} – {max(vals):g}" if vals else "n/a"


def known_gaps(demos: list[dict[str, Any]]) -> list[str]:
    """Facts about what the data does not cover, derived from the demo records."""
    gaps = []
    outcomes = [d["meta"].get("outcome") for d in demos]
    if outcomes and all(o and o.get("success") for o in outcomes):
        gaps.append(f"All {len(demos)} demos are successful: no failure, recovery or near-miss demonstrations.")
    traj = [d["meta"].get("trajectory_sha256") for d in demos if d["meta"].get("trajectory_sha256")]
    if traj and len(set(traj)) < len(traj):
        gaps.append(f"{len(traj)} demos share only {len(set(traj))} distinct state/action trajectories: identical "
                    "robot motion, only the camera images differ. The policy sees no variation in object pose "
                    "or motion.")
    tasks = {d["meta"].get("task") for d in demos}
    if len(tasks) == 1:
        gaps.append(f"One task only ({next(iter(tasks))!r}); the language instruction never changes.")
    env_keys = sorted({k for d in demos for k in (d["meta"].get("environment") or {})})
    varying = [k for k in env_keys if len({json.dumps(d["meta"].get("environment", {}).get(k), sort_keys=True,
                                                      default=str) for d in demos}) > 1]
    fixed = [k for k in env_keys if k not in varying]
    if env_keys:
        gaps.append(f"Recorded environment fields that vary: {', '.join(varying) or 'none'}; "
                    f"constant: {', '.join(fixed) or 'none'}.")
    return gaps


def dataset_card(manifest: dict[str, Any], splits: dict[str, list[dict[str, Any]]]) -> str:
    pre = manifest["preprocessing"]
    schema = pre.get("schema", {})
    lines = [
        f"# Dataset card · {manifest['version']}", "",
        f"- Fingerprint: `{manifest['fingerprint']}`",
        f"- Created: {manifest['created_at']}",
        f"- Train/val source: `{manifest['source']}`",
        f"- Held-out test sources: {', '.join(f'`{t}`' for t in manifest.get('test_sources', [])) or 'none (ratio split)'}",
        f"- Dataset revision: {manifest.get('revision') or 'local folder, not published'}",
        f"- Split seed: {pre['splits']['seed']} · split by episode, never by frame",
        f"- Notes: {manifest.get('notes') or '—'}", "",
        "| Split | Episodes | Frames |", "|---|---|---|",
    ]
    for k in ("train", "val", "test"):
        frames = sum(int(d["meta"].get("n_frames", 0)) for d in splits[k])
        lines.append(f"| {k} | {len(splits[k])} | {frames} |")
    lines += ["", "## Episodes per split", ""]
    for k in ("train", "val", "test"):
        eps = sorted((d["source"], d["episode_index"]) for d in splits[k])
        by_src: dict[str, list[int]] = defaultdict(list)
        for src, i in eps:
            by_src[src].append(i)
        for src, idx in by_src.items():
            lines.append(f"- **{k}** `{src}`: {idx}")
    lines += ["", "Demo IDs per split: `manifest.json` → `demo_ids`.", ""]
    tasks = Counter(str(d["meta"].get("task")) for v in splits.values() for d in v)
    lines += ["## Episodes per task", "", "| Task | Episodes |", "|---|---|"]
    lines += [f"| {t} | {n} |" for t, n in tasks.most_common()]
    lines += ["", "## Variation per split", "",
              "| Split | Seeds | Light colour temp (K) | Rect light intensity | Dome light intensity |",
              "|---|---|---|---|---|"]
    for k in ("train", "val", "test"):
        env = [d["meta"].get("environment") or {} for d in splits[k]]
        light = [e.get("lighting") if isinstance(e.get("lighting"), dict) else {} for e in env]
        lines.append(f"| {k} | {_range([e.get('seed') for e in env])} | "
                     f"{_range([l.get('color_temperature_K') for l in light])} | "
                     f"{_range([l.get('rect_intensity') for l in light])} | {_range([l.get('dome_intensity') for l in light])} |")
    lines += ["", "## Preprocessing", "", "| Field | Value |", "|---|---|",
              f"| FPS | {schema.get('fps', 'n/a')} |",
              f"| Cameras | {', '.join(f'`{k}` {v}' for k, v in (schema.get('cameras') or {}).items()) or 'n/a'} |",
              f"| State | `{(schema.get('state') or {}).get('key')}` {(schema.get('state') or {}).get('shape')} |",
              f"| Action | `{(schema.get('action') or {}).get('key')}` {(schema.get('action') or {}).get('shape')} |",
              f"| Image size | {pre['preprocessing'].get('image_size') or 'dataset resolution'} |",
              f"| Rename map | `{json.dumps(pre['preprocessing'].get('rename_map') or {})}` |",
              f"| Normalisation stats | `meta/stats.json` sha256 `{(pre.get('normalisation_stats') or {}).get('sha256')}` (copied as stats.json) |",
              f"| Leakage guard keys | {', '.join(pre['splits'].get('leakage_keys') or []) or 'demo IDs only'} — passed |"]
    lines += ["", f"## Excluded demos ({len(manifest['excluded'])})", ""]
    lines += [f"- `{e.get('source', '')}` episode {e['episode_index']}: {e['status']} — "
              f"{', '.join(e['reasons']) or 'no reason recorded'}" for e in manifest["excluded"][:50]] or ["- none"]
    all_demos = [d for v in splits.values() for d in v]
    lines += ["", "## Known gaps", ""] + [f"- {g}" for g in known_gaps(all_demos)]
    lines += ["- (Reviewer: add gaps the metadata cannot show, e.g. object poses or layouts never demonstrated.)", ""]
    return "\n".join(lines)


def approve(version: str, registry: Registry | None = None) -> None:
    (registry or Registry()).set_dataset_status(version, "approved")


def load_manifest(version: str, registry: Registry | None = None) -> dict[str, Any]:
    d = (registry or Registry()).get_dataset(version)
    if d is None:
        raise KeyError(f"Unknown dataset version {version}")
    return {**json.loads(Path(d["manifest_path"]).read_text()), "status": d["status"]}
