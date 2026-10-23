"""Create no-copy, episode-level training handoff artifacts for native LeRobot v3 data.

Only reads the source datasets. Writes manifests and train-only numeric statistics
under the repository; never copies, moves, or edits dataset files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "handoff_sources.json"
DEFAULT_OUTPUT = HERE / "handoff_data"
MARKER = ".hackpipe-handoff-artifacts-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_dataset(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"unsupported LeRobot dataset version at {root}: {info.get('codebase_version')}")
    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not episode_files or not data_files:
        raise ValueError(f"missing LeRobot episode or data parquet files at {root}")
    episodes = pq.read_table(episode_files).to_pylist()
    rows = pq.read_table(data_files).to_pylist()
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError(f"episode metadata count disagrees with info.json at {root}")
    task_file = root / "meta" / "tasks.parquet"
    tasks = pq.read_table(task_file).to_pylist() if task_file.is_file() else []
    return info, episodes, rows


def _episode_hash(columns: dict[str, list[Any]], indexes: list[int], episode_id: int) -> tuple[str, int]:
    indexes.sort(key=lambda i: int(columns["frame_index"][i]))
    if not indexes:
        raise ValueError(f"episode {episode_id} has no frame rows")
    digest = hashlib.sha256()
    frame_count = len(indexes)
    for key in ("observation.state", "action"):
        if key not in columns:
            raise ValueError(f"required numeric feature missing: {key}")
        array = np.stack([np.asarray(columns[key][i], dtype="<f8").reshape(-1) for i in indexes])
        if not np.isfinite(array).all():
            raise ValueError(f"episode {episode_id} has non-finite {key} values")
        digest.update(key.encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest(), frame_count


def _split_groups(groups: dict[str, list[int]], seed: int, n_episodes: int,
                  ratios: dict[str, float]) -> dict[str, list[int]] | None:
    """Return disjoint episode splits; None means fewer than 3 unique groups."""
    if len(groups) < 3:
        return None
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    n_test = max(1, int(math.floor(n_episodes * ratios["test"] + 0.5)))
    n_val = max(1, int(math.floor(n_episodes * ratios["validation"] + 0.5)))
    targets = {"validation": n_val, "test": n_test}
    assigned: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    remaining: list[str] = []
    # Whole groups only: greedily fill validation and test near their episode targets.
    for split in ("validation", "test"):
        for group in ordered:
            if group in remaining:
                continue
            episodes = groups[group]
            before = abs(targets[split] - len(assigned[split]))
            after = abs(targets[split] - len(assigned[split]) - len(episodes))
            if len(assigned[split]) < targets[split] and (after < before or not assigned[split]):
                assigned[split].extend(episodes)
                remaining.append(group)
        # don't leave so few unique groups for train that it becomes empty
    for group in ordered:
        if group not in remaining:
            assigned["train"].extend(groups[group])
    if not all(assigned.values()):
        # Balanced deterministic fallback for grouped data: keep first groups in train,
        # and reserve whole groups for the two holdouts when that is possible.
        assigned = {"train": [], "validation": [], "test": []}
        split_order = ("validation", "test", "train")
        cursor = 0
        for split in split_order:
            target = targets.get(split, n_episodes - n_val - n_test)
            while cursor < len(ordered) and (len(assigned[split]) < target or not assigned[split]):
                group = ordered[cursor]
                cursor += 1
                assigned[split].extend(groups[group])
        while cursor < len(ordered):
            assigned["train"].extend(groups[ordered[cursor]])
            cursor += 1
    if not all(assigned.values()):
        return None
    sets = [set(assigned[name]) for name in ("train", "validation", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("episode split overlap")
    group_split = {name: {g for g, ids in groups.items() if set(ids) & sets[index]}
                   for index, name in enumerate(("train", "validation", "test"))}
    if any(group_split[a] & group_split[b] for a, b in (("train", "validation"),
                                                       ("train", "test"),
                                                       ("validation", "test"))):
        raise AssertionError("duplicate trajectory group crossed splits")
    return {name: sorted(ids) for name, ids in assigned.items()}


def _training_stats(rows: list[dict[str, Any]], train_ids: list[int]) -> dict[str, Any]:
    train_id_set = set(train_ids)
    selected = [row for row in rows if int(row["episode_index"]) in train_id_set]
    result: dict[str, Any] = {"source": "train split only", "frame_count": len(selected), "features": {}}
    for key in ("observation.state", "action"):
        values = np.stack([np.asarray(row[key], dtype=np.float64) for row in selected])
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values in training split: {key}")
        result["features"][key] = {
            "count": int(values.shape[0]),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0, ddof=0).tolist(),
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
        }
    return result


def prepare(config_path: Path = DEFAULT_CONFIG, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported handoff source config")
    if output_dir.exists():
        marker = output_dir / MARKER
        if not marker.is_file():
            raise FileExistsError(f"refusing to overwrite unowned output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / MARKER).write_text("Generated by training_preparation/prepare_handoff.py\n", encoding="utf-8")
    ratios = config["split_ratios"]
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1")
    seed = int(config["seed"])
    prepared: list[dict[str, Any]] = []

    for source in config["datasets"]:
        root = Path(source["root"])
        info, episode_meta, rows = _read_dataset(root)
        columns = {key: [row.get(key) for row in rows] for key in
                   ("episode_index", "frame_index", "observation.state", "action", "success")}
        groups: dict[str, list[int]] = {}
        episode_frames: dict[int, int] = {}
        row_indexes: dict[int, list[int]] = {}
        for index, row in enumerate(rows):
            row_indexes.setdefault(int(row["episode_index"]), []).append(index)
        for indexes in row_indexes.values():
            indexes.sort(key=lambda index: int(columns["frame_index"][index]))
        for ep in sorted(episode_meta, key=lambda item: int(item["episode_index"])):
            episode_id = int(ep["episode_index"])
            key, frames = _episode_hash(columns, row_indexes.get(episode_id, []), episode_id)
            if frames != int(ep["length"]):
                raise ValueError(f"episode {episode_id} parquet rows disagree with episode metadata")
            groups.setdefault(key, []).append(episode_id)
            episode_frames[episode_id] = frames

        task_rows = pq.read_table(root / "meta" / "tasks.parquet").to_pylist()
        tasks = sorted({str(task["task"]) for task in task_rows})
        final_success = {
            episode_id: bool(columns["success"][indexes[-1]])
            for episode_id, indexes in row_indexes.items()
        }
        camera_features = sorted(key for key, feature in info["features"].items()
                                 if feature.get("dtype") == "video")
        # Bind the split to all source content, including video bytes, without relying
        # on the prior scan cache or file size/mtime.
        source_files = sorted(path for path in root.rglob("*") if path.is_file())
        source_content = []
        for index, path in enumerate(source_files, start=1):
            source_content.append({"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)})
            if index % 25 == 0 or index == len(source_files):
                print(f"{source['id']}: content-hashed {index}/{len(source_files)} source files", flush=True)
        fingerprint = _canonical_hash(source_content)
        split = _split_groups(groups, seed, len(episode_meta), ratios)
        split_available = split is not None
        if split is None:
            split = {"train": [], "validation": [], "test": []}
        trainable_candidate = split_available
        names_known = bool(info["features"].get("observation.state", {}).get("names") and
                           info["features"].get("action", {}).get("names"))
        status = "CANDIDATE_NEEDS_ACTION_SEMANTICS_REVIEW" if trainable_candidate and not names_known else (
            "CANDIDATE" if trainable_candidate else "BLOCKED_NO_INDEPENDENT_SPLITS")
        manifest = {
            "schema_version": 1,
            "dataset_id": source["id"],
            "task_group": source["task_group"],
            "task_descriptions": tasks,
            "dataset_root": str(root),
            "format": info["codebase_version"],
            "source_fingerprint": fingerprint,
            "source_fingerprint_scope": "SHA-256 of every file under the source dataset root, including videos",
            "episode_count": len(episode_meta),
            "frame_count": len(rows),
            "fps": float(info["fps"]),
            "robot_type": info.get("robot_type"),
            "cameras": {key: info["features"][key].get("shape") for key in camera_features},
            "features": {
                key: {"dtype": info["features"][key].get("dtype"),
                      "shape": info["features"][key].get("shape"),
                      "names": info["features"][key].get("names")}
                for key in ("observation.state", "action")
            },
            "success_label": {
                "feature_present": "success" in info["features"],
                "all_episode_final_values_true": all(final_success.values()) if "success" in info["features"] else None,
                "interpretation": "source-provided labels; not independently verified",
            },
            "seed": seed,
            "split_ratios": ratios,
            "splits_file": f"{source['id']}/splits.json",
            "split_counts": {key: len(value) for key, value in split.items()},
            "trajectory_group_count": len(groups),
            "duplicate_trajectory_groups": [sorted(ids) for ids in groups.values() if len(ids) > 1],
            "split_status": "PASS" if split_available else "BLOCKED",
            "training_candidate": trainable_candidate,
            "status": status,
            "limitations": [] if split_available else [
                "Exact numeric state/action trajectories form fewer than three independent groups; no leakage-safe train/validation/test split is emitted."
            ],
        }
        out = output_dir / source["id"]
        out.mkdir(parents=True, exist_ok=True)
        _atomic_json(out / "dataset_manifest.json", manifest)
        _atomic_json(out / "splits.json", split)
        if trainable_candidate:
            _atomic_json(out / "train_state_action_stats.json", _training_stats(rows, split["train"]))
        elif (out / "train_state_action_stats.json").exists():
            (out / "train_state_action_stats.json").unlink()
        prepared.append({"source": source, "manifest": manifest, "groups": groups})

    overlap: list[dict[str, Any]] = []
    for i, left in enumerate(prepared):
        for right in prepared[i + 1:]:
            shared = sorted(set(left["groups"]) & set(right["groups"]))
            if shared:
                overlap.append({
                    "dataset_a": left["source"]["id"],
                    "dataset_b": right["source"]["id"],
                    "exact_numeric_trajectory_hashes": shared,
                    "episode_ids_a": sorted(e for key in shared for e in left["groups"][key]),
                    "episode_ids_b": sorted(e for key in shared for e in right["groups"][key]),
                    "interpretation": "numeric overlap only; visual demonstration identity is not established",
                })
    catalog = {
        "schema_version": 1,
        "seed": seed,
        "source_config": config_path.name,
        "task_mixing": "disabled; each dataset has its own root and split manifest",
        "datasets": [{
            "dataset_id": item["manifest"]["dataset_id"],
            "task_group": item["manifest"]["task_group"],
            "dataset_root": item["manifest"]["dataset_root"],
            "status": item["manifest"]["status"],
            "training_candidate": item["manifest"]["training_candidate"],
            "split_counts": item["manifest"]["split_counts"],
            "source_fingerprint": item["manifest"]["source_fingerprint"],
            "manifest_file": f"{item['source']['id']}/dataset_manifest.json",
            "splits_file": f"{item['source']['id']}/splits.json",
            "stats_file": (f"{item['source']['id']}/train_state_action_stats.json"
                           if item["manifest"]["training_candidate"] else None),
        } for item in prepared],
        "cross_dataset_numeric_overlap": overlap,
        "source_data_modified": False,
        "training_started": False,
    }
    _atomic_json(output_dir / "catalog.json", catalog)
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    catalog = prepare(args.config, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir),
                      "datasets": [{"id": item["dataset_id"], "status": item["status"],
                                    "split_counts": item["split_counts"]}
                                   for item in catalog["datasets"]],
                      "cross_dataset_numeric_overlap_groups": len(catalog["cross_dataset_numeric_overlap"])},
                     indent=2))


if __name__ == "__main__":
    main()
