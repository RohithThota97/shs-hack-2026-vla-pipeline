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

import pandas as pd

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
                  leakage_keys: list[str] | None = None, other_task: set[str] | None = None) -> None:
    """other_task: sources of another task / schema (data-only subtask) — their seeds etc. are numbered on their
    own, so they aren't compared with the test split's (demo IDs still are)."""
    ids = {k: {d["demo_id"] for d in v} for k, v in splits.items()}
    other_task = other_task or set()
    for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
        overlap = ids[a] & ids[b]
        if overlap:
            raise LeakageError(f"{len(overlap)} demos in both {a} and {b}")
    for key in ([group_key] if group_key else []) + list(leakage_keys or []):
        test_vals = {str(_get(d["meta"], key)) for d in splits["test"] if _get(d["meta"], key) is not None}
        for k in ("train", "val"):
            bad = test_vals & {str(_get(d["meta"], key)) for d in splits[k]
                               if _get(d["meta"], key) is not None and d["source"] not in other_task}
            if bad:
                raise LeakageError(f"test {key} values also in {k}: {sorted(bad)[:5]}")


def _schema(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record["schema"].items() if k != "codebase_version"}


def _schema_key(schema: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in schema.items() if k != "codebase_version"}, sort_keys=True)


def schema_diff(schemas: dict[str, dict[str, Any]]) -> list[str]:
    """Which schema parts differ between sources: e.g. ["fps", "state names"]."""
    vals = list(schemas.values())
    out = []
    for part, label in (("fps", "fps"), ("cameras", "cameras")):
        if len({json.dumps(v.get(part), sort_keys=True) for v in vals}) > 1:
            out.append(label)
    for part in ("state", "action"):
        for sub in ("shape", "names"):
            if len({json.dumps((v.get(part) or {}).get(sub)) for v in vals}) > 1:
                out.append(f"{part} {sub}")
    if not out and len({_schema_key(v) for v in vals}) > 1:
        out.append("other schema fields")
    return out


def primary_source(sources: list[str]) -> str | None:
    """The source whose schema a mixed version trains with: the project's dataset.root (handover), if it is one of
    the version's sources (None otherwise — no guessing by alphabetical order)."""
    root = project_config()["dataset"].get("root")
    key = source_key(resolve(root)) if root else None
    return key if key in sources else None


def trainable_sources(manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(sources training uses, data-only sources). A version mixing schemas (e.g. the 6 fps cube-stacking subtask
    next to 30 fps handover) trains on the sources with the primary source's schema; the rest is kept as data.
    Read from the manifest (train_sources, stored at creation) so a later config change can't alter what an
    approved version trains; older manifests without it are derived once from their per-source schemas."""
    if manifest.get("train_sources") is not None:          # fixed when the version was created
        return list(manifest["train_sources"]), list(manifest.get("data_only_sources") or [])
    sources = manifest.get("sources") or ([manifest["source"]] if manifest.get("source") else [])
    per = (manifest.get("preprocessing") or {}).get("per_source_schema")
    if not manifest.get("mixed_schemas") or not per:
        return sources, []
    primary = primary_source(sources)
    if primary is None:
        raise RuntimeError(f"{manifest.get('version')}: mixed sources but dataset.root is none of {sources}")
    want = _schema_key(per[primary])
    train = [s for s in sources if _schema_key(per[s]) == want]
    return train, [s for s in sources if s not in train]


def needs_merge(manifest: dict[str, Any]) -> bool:
    """Training needs one merged folder: more than one trainable source."""
    return len(trainable_sources(manifest)[0]) > 1


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


class NothingNewError(RuntimeError):
    pass


def _excluded_by_hand(d: dict[str, Any]) -> bool:
    return bool(d["meta"].get("excluded"))


def _subtask_labels(directory: str | Path, episodes: dict[str, list[list[Any]]]) -> dict[str, Any]:
    """Check sub-task labels (hackpipe subtasks label) cover every train/val episode of the version."""
    d = resolve(directory)
    index = json.loads((d / "labels.json").read_text())
    want: dict[str, set[int]] = defaultdict(set)
    for split in ("train", "val"):
        for s, i in episodes[split]:
            want[s].add(int(i))
    missing = {}
    for s, eps in want.items():
        v = index["sources"].get(s)
        have = set() if v is None else {int(e) for e in pd.read_csv(d / v["episodes_file"]).episode_index}
        if eps - have:
            missing[s] = len(eps - have)
    if missing:
        raise RuntimeError(f"Sub-task labels in {d} don't cover {missing} train/val episodes (source: count). "
                           "Label this version's episodes: hackpipe subtasks label --base <version>")
    return index


def create_dataset_version(registry: Registry | None = None, cfg: dict[str, Any] | None = None,
                           notes: str = "", base: str | None = None, subtasks: str | Path | None = None
                           ) -> dict[str, Any]:
    """Create the next ds-vN (draft).

    Without base: split all VALID demos (fixed seed). With base (an approved version): keep the base's train/val
    assignment for demos that are still VALID, split only the added demos, keep the frozen test split, and
    record inherited / added / removed. Demos excluded by hand (meta.excluded) and DUPLICATEs never enter.
    subtasks: a labels folder from `hackpipe subtasks label` — copied into the version (part of its fingerprint) for
    the curriculum trainer; with a base it is reason enough for a new version even if no demo changed. Without it,
    the base's labels are kept if they still cover every train/val episode."""
    reg = registry or Registry()
    cfg = cfg or dataset_config()
    valid = [d for d in reg.demos(status=VALID) if not _excluded_by_hand(d)]
    if not valid:
        raise RuntimeError("No VALID demos. Run `hackpipe ingest` and `hackpipe qc` first.")
    test_sources = sorted({source_key(resolve(p)) for p in (cfg["splits"].get("test_sources") or [])})
    held_out = [d for d in valid if d["source"] in test_sources]
    pool = [d for d in valid if d["source"] not in test_sources]
    missing = [t for t in test_sources if t not in {d["source"] for d in held_out}]
    if missing:
        raise RuntimeError(f"Held-out test sources have no VALID demos: {missing}. Ingest + QC them first.")

    lineage: dict[str, Any] = {"base": base}
    if base:
        bm = load_manifest(base, reg)
        if bm["status"] != "approved":
            raise RuntimeError(f"Base {base} is '{bm['status']}'; a new version builds on an approved one.")
        by_id = {d["demo_id"]: d for d in valid}
        base_ids = {k: set(bm["demo_ids"][k]) for k in ("train", "val", "test")}
        in_base = base_ids["train"] | base_ids["val"] | base_ids["test"]
        added = [d for d in pool if d["demo_id"] not in in_base]
        removed = sorted(i for i in base_ids["train"] | base_ids["val"] if i not in by_id)
        if not added and not removed and subtasks is None:
            raise NothingNewError(f"Nothing new since {base}: no added or removed demos, so no new version.")
        new_cfg = {**cfg, "splits": {**cfg["splits"], "test": 0.0, "test_episodes": []}}
        new = split_demos(added, new_cfg) if added else {"train": [], "val": [], "test": []}
        splits = {
            "train": [by_id[i] for i in sorted(base_ids["train"]) if i in by_id] + new["train"],
            "val": [by_id[i] for i in sorted(base_ids["val"]) if i in by_id] + new["val"],
            "test": held_out if test_sources else [by_id[i] for i in sorted(base_ids["test"]) if i in by_id],
        }
        lineage.update({"inherited": sorted((base_ids["train"] | base_ids["val"]) - set(removed)),
                        "added": sorted(d["demo_id"] for d in added), "removed": removed,
                        "added_split": {k: sorted(d["demo_id"] for d in new[k]) for k in ("train", "val")}})
    else:
        splits = split_demos(pool, cfg, held_out)
    sources = sorted({d["source"] for d in splits["train"] + splits["val"]})
    records = {s: preprocessing_record(s, cfg) for s in sources}
    schemas = {s: _schema(r) for s, r in records.items()}
    mixed = len({_schema_key(v) for v in schemas.values()}) > 1
    if mixed and not (cfg.get("versions") or {}).get("allow_mixed_schemas"):
        raise RuntimeError(f"Sources have different schemas ({', '.join(schema_diff(schemas))}): {sorted(schemas)}. "
                           "To keep them in one version anyway (only the sources matching the main dataset are "
                           "trained), set versions.allow_mixed_schemas: true in configs/dataset.yaml")
    if mixed:
        primary = primary_source(sources)
        if primary is None:
            raise RuntimeError("Sources have different schemas and dataset.root (configs/project.yaml) is not one of "
                               f"the train/val sources {sources}: set it to the main dataset, whose schema is trained.")
    else:
        primary = sources[0]           # as before: fingerprints of unmixed versions don't depend on dataset.root
    other_task = {s for s in sources if _schema_key(schemas[s]) != _schema_key(schemas[primary])}
    leakage_guard(splits, cfg["splits"].get("group_key"), cfg["splits"].get("leakage_keys"), other_task)
    frozen = frozen_test_ids(reg)
    if cfg["splits"].get("freeze_test", True) and frozen and frozen[1] != sorted(d["demo_id"] for d in splits["test"]):
        raise LeakageError(f"The test split is frozen since {frozen[0]}; this version's test demos differ. "
                           "Keep the split config, or start a new test suite on purpose.")
    preprocessing = records[primary]   # the trained schema (single source: its only one, as before)
    if mixed:                          # recorded per source; part of the fingerprint
        preprocessing["per_source_schema"] = schemas
    warnings = []
    for t in test_sources:
        d = schema_diff({"trained": schemas[primary], t: _schema(preprocessing_record(t, cfg))})
        if d:
            warnings.append(f"test source {t} has another schema than the trained sources ({', '.join(d)})")
    if len(sources) > 1:            # single-source records stay as before, so fingerprints stay comparable
        preprocessing["normalisation_stats"]["per_source"] = {s: r["normalisation_stats"]["sha256"]
                                                              for s, r in records.items()}
    episodes = {k: sorted([d["source"], d["episode_index"]] for d in v) for k, v in splits.items()}
    labels_dir = subtasks
    if labels_dir is None and base and (bm.get("subtasks") or {}).get("dir"):
        try:                                   # keep the base's labels while they still fit
            _subtask_labels(bm["subtasks"]["dir"], episodes)
            labels_dir = bm["subtasks"]["dir"]
        except RuntimeError:
            warnings.append(f"sub-task labels of {base} dropped: they don't cover the added episodes "
                            "(hackpipe subtasks label, then create again with --subtasks)")
    labels = _subtask_labels(labels_dir, episodes) if labels_dir is not None else None
    if (base and subtasks is not None and labels and not lineage.get("added") and not lineage.get("removed")
            and labels["sha256"] == (bm.get("subtasks") or {}).get("sha256")):
        raise NothingNewError(f"Nothing new since {base}: same demos and the same sub-task labels.")
    if labels:                         # part of the fingerprint: other labels = another version
        preprocessing["subtask_labels"] = {"sha256": labels["sha256"], "label_version": labels["label_version"],
                                           "code_sha256": labels["code_sha256"]}
    used = splits["train"] + splits["val"] + splits["test"]
    fp = fingerprint([demo_fingerprint_item(d) for d in used], preprocessing)
    version = next_version([d["version"] for d in reg.datasets()], "ds")
    in_scope = {*sources, *test_sources}
    excluded = [{"demo_id": d["demo_id"], "source": d["source"], "episode_index": d["episode_index"],
                 "status": "EXCLUDED" if _excluded_by_hand(d) else d["status"],
                 "reasons": ([f"excluded by {d['meta']['excluded'].get('by')}: {d['meta']['excluded'].get('reason')}"]
                             if _excluded_by_hand(d) else
                             [f"duplicate of {d['meta'].get('duplicate_of')}"] if d["status"] == "DUPLICATE" else
                             d["meta"].get("qc", {}).get("hard", []) + d["meta"].get("qc", {}).get("soft", []))}
                for d in reg.demos() if (d["status"] != VALID or _excluded_by_hand(d)) and d["source"] in in_scope]
    manifest = {
        "version": version,
        "fingerprint": fp,
        "created_at": now_iso(),
        "source": sources[0] if len(sources) == 1 else None,
        "sources": sources,
        "test_sources": test_sources,
        "needs_materialize": len(sources) - len(other_task) > 1,
        "mixed_schemas": mixed,     # e.g. 30 fps + 6 fps: trains on `train_sources`, the rest is data only
        "train_sources": [x for x in sources if x not in other_task],
        "data_only_sources": sorted(other_task),
        "warnings": warnings,
        "repo_id": project_config()["dataset"]["repo_id"],
        "revision": project_config()["dataset"].get("revision"),
        # Episode indices per split: valid on `source` (train/val) and the test source; multi-source versions
        # are trained from a merged folder (`hackpipe dataset materialize`).
        "splits": {k: sorted(d["episode_index"] for d in v) for k, v in splits.items()},
        "episodes": episodes,
        "demo_ids": {k: sorted(d["demo_id"] for d in v) for k, v in splits.items()},
        "lineage": lineage,
        "excluded": excluded,
        "preprocessing": preprocessing,
        "notes": notes,
    }
    out_dir = resolve(Path(project_config()["project"]["outputs_dir"]) / "datasets" / version)
    out_dir.mkdir(parents=True, exist_ok=True)
    if labels:
        shutil.rmtree(out_dir / "subtasks", ignore_errors=True)
        shutil.copytree(resolve(labels_dir), out_dir / "subtasks")
        rel = Path(project_config()["project"]["outputs_dir"]) / "datasets" / version / "subtasks"
        manifest["subtasks"] = {"dir": rel.as_posix(), "sha256": labels["sha256"], "from": str(labels_dir),
                                "label_version": labels["label_version"], "base": labels.get("base"),
                                "sources": {s: {k: v[k] for k in ("kind", "fps", "flagged", "unlabelled")}
                                            for s, v in labels["sources"].items()}}
    stats = DatasetFolder(resolve(primary)).stats_path()          # normalisation stats of the trained schema
    if stats.exists():
        shutil.copyfile(stats, out_dir / "stats.json")
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (out_dir / "dataset_card.md").write_text(dataset_card(manifest, splits))
    reg.add_dataset(version, fp, str(manifest_path), {"counts": {k: len(v) for k, v in splits.items()},
                                                      "card": str(out_dir / "dataset_card.md"), "base": base})
    return manifest


def latest_approved(registry: Registry) -> str | None:
    approved = [d["version"] for d in registry.datasets() if d["status"] == "approved"]
    return max(approved, key=lambda v: int(v.split("-v", 1)[1])) if approved else None


def set_excluded(demo_id: str, excluded: bool, reason: str = "", by: str = "",
                 registry: Registry | None = None) -> None:
    """Exclude a demo from future versions by hand (reason + name required), or include it again."""
    reg = registry or Registry()
    d = reg.get_demo(demo_id)
    if d is None:
        raise KeyError(demo_id)
    if excluded and not (reason.strip() and by.strip()):
        raise ValueError("Excluding a demo needs a reason and a name.")
    reg.set_demo_status(demo_id, d["status"], {"excluded": {"reason": reason, "by": by, "at": now_iso()}
                                               if excluded else None})


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
        f"- Train/val source(s): {', '.join(f'`{x}`' for x in manifest.get('sources') or [manifest['source']])}"
        + (f" — sources differ ({', '.join(schema_diff(pre.get('per_source_schema') or {}))}): training uses "
           f"{', '.join(f'`{x}`' for x in trainable_sources(manifest)[0])}; data only: "
           f"{', '.join(f'`{x}`' for x in trainable_sources(manifest)[1])}" if manifest.get("mixed_schemas") else "")
        + (" — several sources: train from the merged folder (`hackpipe dataset materialize`)"
           if needs_merge(manifest) else ""),
        f"- Based on: {(manifest.get('lineage') or {}).get('base') or 'nothing (first version)'}",
        f"- Held-out test sources: {', '.join(f'`{t}`' for t in manifest.get('test_sources', [])) or 'none (ratio split)'}",
        f"- Dataset revision: {manifest.get('revision') or 'local folder, not published'}",
        f"- Split seed: {pre['splits']['seed']} · split by episode, never by frame",
        f"- Notes: {manifest.get('notes') or '—'}", "",
        "| Split | Episodes | Frames |", "|---|---|---|",
    ]
    if pre.get("per_source_schema"):
        rows = ["| Source | fps | cameras | state | action |", "|---|---|---|---|---|"]
        for src, sc in sorted(pre["per_source_schema"].items()):
            rows.append(f"| `{src}` | {sc['fps']:g} | {len(sc['cameras'])} | {sc['state']['shape']} | "
                        f"{sc['action']['shape']} |")
        at = lines.index("| Split | Episodes | Frames |")
        lines[at:at] = ["## Sources (different schemas)", "", *rows, ""]
    for w in manifest.get("warnings") or []:
        lines.insert(lines.index("| Split | Episodes | Frames |"), f"> ⚠ {w}\n")
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
    lin = manifest.get("lineage") or {}
    if lin.get("base"):
        added_split = lin.get("added_split") or {}
        lines += [f"## Inherited vs new (base {lin['base']})", "",
                  "| | Demos |", "|---|---|",
                  f"| Inherited from {lin['base']} (kept their train/val split, not re-validated) | "
                  f"{len(lin.get('inherited', []))} |",
                  f"| Added (new VALID demos) | {len(lin.get('added', []))} — train "
                  f"{len(added_split.get('train', []))}, val {len(added_split.get('val', []))} |",
                  f"| Removed (excluded / no longer VALID) | {len(lin.get('removed', []))} |",
                  "| Test split | unchanged (frozen) ✓ |", ""]
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
    sub = manifest.get("subtasks")
    if sub:
        lines += ["", "## Sub-task labels (curriculum training)", "",
                  f"Labels `{sub['sha256'][:12]}` (rules v{sub['label_version']}, from the robot signals), copied to "
                  f"`{sub['dir']}`. Flagged episodes keep their split but the curriculum trainer leaves them out.", "",
                  "| Source | Rules | Flagged episodes |", "|---|---|---|"]
        lines += [f"| `{s}` | {v['kind']} | {v['flagged'] or 'none'} |" for s, v in sorted(sub["sources"].items())]
    lines += ["", f"## Excluded demos ({len(manifest['excluded'])})", ""]
    lines += [f"- `{e.get('source', '')}` episode {e['episode_index']}: {e['status']} — "
              f"{', '.join(e['reasons']) or 'no reason recorded'}" for e in manifest["excluded"][:50]] or ["- none"]
    all_demos = [d for v in splits.values() for d in v]
    lines += ["", "## Known gaps", ""] + [f"- {g}" for g in known_gaps(all_demos)]
    lines += ["- (Reviewer: add gaps the metadata cannot show, e.g. object poses or layouts never demonstrated.)", ""]
    return "\n".join(lines)


def approve(version: str, registry: Registry | None = None, by: str = "") -> None:
    reg = registry or Registry()
    reg.set_dataset_status(version, "approved")
    reg.update_dataset_meta(version, {"approved_by": by or None, "approved_at": now_iso()})


def load_manifest(version: str, registry: Registry | None = None) -> dict[str, Any]:
    d = (registry or Registry()).get_dataset(version)
    if d is None:
        raise KeyError(f"Unknown dataset version {version}")
    return {**json.loads(Path(d["manifest_path"]).read_text()), "status": d["status"]}
