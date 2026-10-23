"""R1 — Import LeRobot v3 episodes with content integrity metadata.

Existing demo IDs remain source-path/episode based for registry compatibility.
Content hashes are separate and cover episode rows, referenced metadata, data
shards and camera files so copied inputs can be detected without conflating IDs.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hackpipe.common.config import dataset_config, project_config
from hackpipe.common.ids import demo_id, now_iso, sha256_bytes
from hackpipe.common.lerobot_io import DatasetFolder
from hackpipe.common.registry import Registry


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _episode_checksum(ep: pd.DataFrame, cols: list[str] | None = None) -> str:
    """Hash canonical episode rows; the optional ``cols`` keeps old callers valid."""
    columns = sorted(cols if cols is not None else ep.columns)
    rows = [{column: _json_value(row[column]) for column in columns if column in ep.columns}
            for _, row in ep.iterrows()]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=_json_value,
                         allow_nan=True).encode("utf-8")
    return sha256_bytes(payload)


def _file_checksum(path: Path, cache: dict[Path, str]) -> str:
    """Hash file bytes in chunks. Cache lifetime is one import call, never persisted."""
    path = path.resolve()
    if path not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        cache[path] = digest.hexdigest()
    return cache[path]


def _episode_content_checksum(ds: DatasetFolder, episode: pd.DataFrame,
                              episode_meta: dict[str, Any], task_text: str | None,
                              file_hashes: dict[Path, str]) -> str:
    referenced_files: dict[str, str] = {}
    data_path = ds.data_path_for_episode(episode_meta)
    referenced_files["data"] = _file_checksum(data_path, file_hashes)
    for camera in ds.video_keys:
        referenced_files[f"camera:{camera}"] = _file_checksum(
            ds.video_path_for_episode(episode_meta, camera), file_hashes
        )
    metadata = {
        key: _json_value(value) for key, value in episode_meta.items()
        if key not in {"episode_index"}
    }
    content = {
        "rows_sha256": _episode_checksum(episode),
        "episode_metadata": metadata,
        "task": task_text,
        "fps": ds.info.get("fps"),
        "features": ds.features,
        "file_sha256": referenced_files,
    }
    payload = json.dumps(_json_value(content), sort_keys=True, separators=(",", ":"),
                         default=_json_value, allow_nan=True).encode("utf-8")
    return sha256_bytes(payload)


def build_demo_records(root: str | Path, source: str | None = None) -> list[dict[str, Any]]:
    """Build one record per episode; preserve recoverable per-episode failures."""
    ds = DatasetFolder(root)
    cfg = dataset_config()["columns"]
    proj = project_config()
    df = ds.frames()
    source = source or str(Path(root).resolve())
    ep_col = cfg["episode"]
    if ep_col not in df.columns:
        raise ValueError(f"data parquet is missing required column {ep_col!r}")

    audit = ds.validate_structure(df, cfg["task_index"])
    metadata = audit["episode_metadata"]
    meta_by_id: dict[int, dict[str, Any]] = {}
    if "episode_index" in metadata.columns:
        for row in metadata.to_dict(orient="records"):
            try:
                meta_by_id[int(row["episode_index"])] = row
            except (KeyError, TypeError, ValueError):
                continue

    frame_ids = {int(value) for value in df[ep_col].dropna().unique()}
    episode_ids = sorted(frame_ids | set(meta_by_id))
    tasks = ds.tasks()
    file_hashes: dict[Path, str] = {}
    records: list[dict[str, Any]] = []
    for episode_index in episode_ids:
        demo = demo_id(source, episode_index)  # Stable legacy identity: do not migrate IDs.
        episode = df[df[ep_col] == episode_index]
        errors = list(audit["blocking_issues"])
        errors.extend(audit["episode_issues"].get(episode_index, []))
        if episode.empty:
            errors.append("episode metadata has no matching data rows")
        if not errors:
            try:
                frame_col = cfg["frame"]
                if frame_col not in episode:
                    raise ValueError(f"data parquet is missing required column {frame_col!r}")
                episode = episode.sort_values(frame_col)
                task_col = cfg["task_index"]
                task_ids = set(int(value) for value in episode[task_col].dropna().unique()) if task_col in episode else set()
                if len(task_ids) > 1:
                    raise ValueError(f"episode contains multiple task_index values: {sorted(task_ids)}")
                task_text = tasks.get(next(iter(task_ids))) if task_ids else None
                episode_meta = meta_by_id.get(episode_index)
                if episode_meta is None:
                    raise ValueError("no matching meta/episodes row")
                checksum = _episode_content_checksum(ds, episode, episode_meta, task_text, file_hashes)
                ts_col = cfg["timestamp"]
                ts = episode[ts_col].to_numpy(dtype=float) if ts_col in episode else np.array([])
                records.append({
                    "demo_id": demo, "source": source, "episode_index": episode_index,
                    "status": "imported",
                    "meta": {
                        "n_frames": int(len(episode)), "fps": ds.fps,
                        "duration_s": float(ts[-1] - ts[0]) if ts.size > 1 else 0.0,
                        "task": task_text,
                        "robot": {"type": ds.robot_type or proj["robot"]["type"],
                                  "embodiment": proj["robot"].get("embodiment")},
                        "cameras": {key: ds.features[key].get("shape") for key in ds.video_keys},
                        "state_dim": ds.feature_dim(cfg["state"]),
                        "action_dim": ds.feature_dim(cfg["action"]),
                        "operator": proj["demo_defaults"]["operator"],
                        "environment": proj["demo_defaults"]["environment"],
                        "content_sha256": checksum,
                        "codebase_version": ds.info.get("codebase_version"),
                        "imported_at": now_iso(),
                    },
                })
            except Exception as exc:  # a malformed episode must not discard its neighbors
                errors.append(f"{type(exc).__name__}: {exc}")
        if errors:
            records.append({
                "demo_id": demo, "source": source, "episode_index": episode_index,
                "status": "FAILED",
                "meta": {"ingest_errors": sorted(set(errors)), "imported_at": now_iso(),
                         "codebase_version": ds.info.get("codebase_version")},
            })
    return records


def ingest_report(root: str | Path, registry: Registry | None = None) -> dict[str, Any]:
    """Import records and report failed episodes and content duplicates."""
    reg = registry or Registry()
    records = build_demo_records(root)
    existing = reg.demos()
    by_checksum: dict[str, list[dict[str, Any]]] = {}
    for old in existing:
        checksum = old["meta"].get("content_sha256")
        if checksum:
            by_checksum.setdefault(checksum, []).append(old)

    failures: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    imported = 0
    for record in records:
        meta = dict(record["meta"])
        if record["status"] == "FAILED":
            failures.append({"episode_index": record["episode_index"], "reasons": meta["ingest_errors"]})
            reg.upsert_demo(record["demo_id"], record["source"], record["episode_index"],
                            meta, status="FAILED")
            continue

        checksum = meta["content_sha256"]
        matches = [old for old in by_checksum.get(checksum, [])
                   if old["demo_id"] != record["demo_id"]]
        if matches:
            duplicate_ids = sorted(old["demo_id"] for old in matches)
            meta["potential_duplicate_imports"] = duplicate_ids
            duplicates.append({"demo_id": record["demo_id"], "episode_index": record["episode_index"],
                               "content_sha256": checksum, "duplicate_of": duplicate_ids,
                               "sources": sorted({old["source"] for old in matches} | {record["source"]})})
        prior = reg.get_demo(record["demo_id"])
        retry_status = "imported" if prior and prior["status"] == "FAILED" else None
        reg.upsert_demo(record["demo_id"], record["source"], record["episode_index"], meta,
                        status=retry_status)
        imported += 1
        by_checksum.setdefault(checksum, []).append({**record, "meta": meta})
    return {"root": str(Path(root).resolve()), "imported": imported, "failed": failures,
            "duplicates": duplicates, "records": len(records)}


def ingest(root: str | Path, registry: Registry | None = None) -> int:
    """Compatibility wrapper returning the number of successfully imported episodes."""
    return int(ingest_report(root, registry)["imported"])


def watch(root: str | Path, interval_s: int = 60, registry: Registry | None = None) -> None:
    """Poll a dataset folder and import new episodes. Stop with Ctrl+C."""
    reg = registry or Registry()
    source = str(Path(root).resolve())
    seen = {demo["demo_id"] for demo in reg.demos(source=source)}
    print(f"Watching {root} every {interval_s}s ({len(seen)} demos already registered)")
    while True:
        try:
            records = build_demo_records(root)
            new = [record for record in records if record["demo_id"] not in seen]
            result = _register_records(new, reg)
            seen.update(record["demo_id"] for record in new if record["status"] != "FAILED")
            if new:
                print(f"{now_iso()} imported {result['imported']}; failed {len(result['failed'])}; "
                      f"duplicates {len(result['duplicates'])}")
                for failure in result["failed"]:
                    print(f"episode {failure['episode_index']} failed: {'; '.join(failure['reasons'])}")
        except Exception as exc:
            print(f"import scan failed: {type(exc).__name__}: {exc}")
        time.sleep(interval_s)


def _register_records(records: list[dict[str, Any]], registry: Registry) -> dict[str, Any]:
    """Shared registration path for watcher batches without rebuilding existing records."""
    existing = registry.demos()
    by_checksum: dict[str, list[dict[str, Any]]] = {}
    for old in existing:
        checksum = old["meta"].get("content_sha256")
        if checksum:
            by_checksum.setdefault(checksum, []).append(old)
    failures, duplicates = [], []
    imported = 0
    for record in records:
        meta = dict(record["meta"])
        if record["status"] == "FAILED":
            failures.append({"episode_index": record["episode_index"], "reasons": meta["ingest_errors"]})
            registry.upsert_demo(record["demo_id"], record["source"], record["episode_index"],
                                 meta, status="FAILED")
            continue
        checksum = meta["content_sha256"]
        matches = [old for old in by_checksum.get(checksum, []) if old["demo_id"] != record["demo_id"]]
        if matches:
            duplicate_ids = sorted(old["demo_id"] for old in matches)
            meta["potential_duplicate_imports"] = duplicate_ids
            duplicates.append({"demo_id": record["demo_id"], "episode_index": record["episode_index"],
                               "content_sha256": checksum, "duplicate_of": duplicate_ids,
                               "sources": sorted({old["source"] for old in matches} | {record["source"]})})
        prior = registry.get_demo(record["demo_id"])
        retry_status = "imported" if prior and prior["status"] == "FAILED" else None
        registry.upsert_demo(record["demo_id"], record["source"], record["episode_index"], meta,
                             status=retry_status)
        imported += 1
        by_checksum.setdefault(checksum, []).append({**record, "meta": meta})
    return {"imported": imported, "failed": failures, "duplicates": duplicates}
