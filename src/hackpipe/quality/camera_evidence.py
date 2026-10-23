"""Validation for externally completed, full-decode FFmpeg camera reports."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from hackpipe.common.lerobot_io import DatasetFolder


class CameraEvidenceError(ValueError):
    """Camera report is stale, incomplete, mismatched, or reports decode failures."""

    def __init__(self, message: str, *, blocking: bool = False):
        super().__init__(message)
        self.blocking = blocking


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_sha256(ds: DatasetFolder) -> str:
    digest = hashlib.sha256()
    metadata_root = ds.root / "meta"
    for path in sorted(p for p in metadata_root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(ds.root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _path_key(value: str | Path) -> str:
    return os.path.normcase(str(Path(value).resolve()))


def _inventory(ds: DatasetFolder) -> list[dict[str, Any]]:
    rows = ds.episode_metadata().to_dict(orient="records")
    inventory = []
    for row in rows:
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        for camera in ds.video_keys:
            prefix = f"videos/{camera}"
            start = float(row[f"{prefix}/from_timestamp"])
            end = float(row[f"{prefix}/to_timestamp"])
            if not math.isfinite(start) or not math.isfinite(end) or end <= start or length <= 0:
                raise CameraEvidenceError(f"invalid camera timing/length metadata for episode {episode_index}, {camera}")
            inventory.append({
                "episode_index": episode_index,
                "camera": camera,
                "path": _path_key(ds.video_path_for_episode(row, camera)),
                "expected_frames": length,
                "start": start,
                "end": end,
            })
    return sorted(inventory, key=lambda item: (item["episode_index"], item["camera"], item["path"]))


def dataset_identity(ds: DatasetFolder) -> dict[str, Any]:
    episodes = ds.episode_metadata()
    inventory = _inventory(ds)
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "root": _path_key(ds.root),
        "metadata_sha256": _metadata_sha256(ds),
        "fps": ds.fps,
        "total_episodes": ds.info.get("total_episodes"),
        "discovered_episodes": len(episodes),
        "episode_ids": sorted(int(value) for value in episodes["episode_index"].tolist()),
        "feature_names": sorted(ds.features),
        "camera_features": sorted(ds.video_keys),
        "episode_camera_stream_associations": len(inventory),
        "inventory_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def validate_ffmpeg_report(ds: DatasetFolder, report_path: str | Path) -> dict[str, Any]:
    """Validate report provenance, exact inventory, and current camera bytes.

    A report is usable only when it asserts a complete FFmpeg full decode and every
    unique referenced video has a matching fresh SHA-256 and successful decoder exit.
    """
    path = Path(report_path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CameraEvidenceError(f"cannot read FFmpeg camera report {path}: {exc}") from exc
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise CameraEvidenceError("unsupported or missing FFmpeg camera report schema_version")

    verification = report.get("verification")
    if (not isinstance(verification, dict) or verification.get("method") != "ffmpeg_full_decode" or
            (verification.get("scanner") not in (None, "hackpipe-camera-verifier-v3")) or
            not verification.get("tool_version")):
        raise CameraEvidenceError("report does not identify the FFmpeg full-decode verifier")

    expected_identity = dataset_identity(ds)
    if report.get("dataset_identity") != expected_identity:
        raise CameraEvidenceError("FFmpeg report dataset identity or metadata is stale/mismatched")
    if (_path_key(report.get("root", "")) != expected_identity["root"] or
            report.get("expected_episodes") != expected_identity["total_episodes"] or
            report.get("discovered_episodes") != expected_identity["discovered_episodes"] or
            report.get("declared_camera_features") != expected_identity["camera_features"] or
            report.get("episode_camera_stream_associations") != expected_identity["episode_camera_stream_associations"]):
        raise CameraEvidenceError("FFmpeg report top-level dataset identity or readiness is inconsistent")

    expected_inventory = _inventory(ds)
    reported_associations = report.get("episode_camera_associations")
    if not isinstance(reported_associations, list):
        raise CameraEvidenceError("FFmpeg report has no episode-camera association inventory")

    def normalized_association(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "episode_index": int(item["episode_index"]),
            "camera": str(item["camera"]),
            "path": _path_key(item["path"]),
            "expected_frames": int(item["expected_frames"]),
            "start": float(item["start"]),
            "end": float(item["end"]),
        }

    try:
        actual_inventory = sorted((normalized_association(item) for item in reported_associations),
                                  key=lambda item: (item["episode_index"], item["camera"], item["path"]))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise CameraEvidenceError(f"malformed episode-camera inventory: {exc}") from exc
    if actual_inventory != expected_inventory:
        raise CameraEvidenceError("FFmpeg report episode-camera inventory does not match current metadata")

    video_outcomes = report.get("video_outcomes")
    if not isinstance(video_outcomes, list):
        raise CameraEvidenceError("FFmpeg report has no per-video outcomes")
    expected_paths = sorted({item["path"] for item in expected_inventory})
    normalized_outcomes: dict[str, dict[str, Any]] = {}
    for item in video_outcomes:
        if not isinstance(item, dict) or not item.get("path"):
            raise CameraEvidenceError("malformed per-video decode result")
        key = _path_key(item["path"])
        if key in normalized_outcomes:
            raise CameraEvidenceError(f"duplicate per-video result for {key}")
        normalized_outcomes[key] = item
    if sorted(normalized_outcomes) != expected_paths:
        raise CameraEvidenceError("FFmpeg report per-video inventory is incomplete or mismatched")
    cameras_by_path: dict[str, set[str]] = {}
    for association in expected_inventory:
        cameras_by_path.setdefault(association["path"], set()).add(association["camera"])
    for video_path in expected_paths:
        result = normalized_outcomes[video_path]
        if (result.get("decoder") != "ffmpeg" or result.get("full_decode") is not True or
                result.get("outcome") != "fully_decoded" or result.get("decoder_exit_code") != 0):
            failed = result.get("outcome") in {"failed", "partially_decoded"}
            raise CameraEvidenceError(
                f"video decode is not a successful full FFmpeg decode: {video_path} "
                f"({result.get('outcome', 'missing outcome')})", blocking=failed)
        try:
            frame_count = int(result["frame_count"])
            reported_hash = str(result["sha256"])
            video_fps = float(result["fps"])
            duration = float(result["duration"])
            width, height = int(result["width"]), int(result["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CameraEvidenceError(f"incomplete FFmpeg result for {video_path}: {exc}") from exc
        if (frame_count <= 0 or not math.isfinite(video_fps) or video_fps <= 0 or
                not math.isfinite(duration) or duration <= 0 or width <= 0 or height <= 0):
            raise CameraEvidenceError(f"FFmpeg reported invalid stream metrics for {video_path}", blocking=True)
        if result.get("reported_frame_count") is not None and int(result["reported_frame_count"]) != frame_count:
            raise CameraEvidenceError(f"decoded/container frame counts disagree for {video_path}", blocking=True)
        for camera in cameras_by_path[video_path]:
            shape = ds.features.get(camera, {}).get("shape")
            if shape and (height, width, 3) != tuple(shape):
                raise CameraEvidenceError(f"decoded resolution disagrees with declared camera shape: {video_path}",
                                          blocking=True)
        if abs(video_fps - ds.fps) > max(0.1, ds.fps * 0.01):
            raise CameraEvidenceError(f"decoded camera FPS disagrees with dataset FPS: {video_path}", blocking=True)
        actual_path = Path(video_path)
        if not actual_path.is_file():
            raise CameraEvidenceError(f"referenced video disappeared: {video_path}")
        if _file_sha256(actual_path) != reported_hash:
            raise CameraEvidenceError(f"video content SHA-256 changed since FFmpeg scan: {video_path}")

    if (report.get("unique_referenced_video_files") != len(expected_paths) or
            report.get("videos_fully_decoded") != len(expected_paths) or
            report.get("videos_partially_decoded") != 0 or report.get("videos_failed") != 0 or
            report.get("videos_skipped") != 0):
        raise CameraEvidenceError("FFmpeg report aggregate video outcomes are incomplete or inconsistent")
    if report.get("camera_readiness") != "PASS":
        raise CameraEvidenceError("FFmpeg report camera readiness is not PASS")

    per_episode: dict[int, list[dict[str, Any]]] = {}
    for association in reported_associations:
        expected = next(item for item in expected_inventory
                        if item["episode_index"] == int(association["episode_index"])
                        and item["camera"] == str(association["camera"]))
        try:
            count = int(association["segment_frame_count"])
            first_pts, last_pts = float(association["first_pts"]), float(association["last_pts"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CameraEvidenceError(f"incomplete decoded timing for episode-camera association: {exc}") from exc
        if (association.get("video_outcome") != "fully_decoded" or count != expected["expected_frames"] or
                not math.isfinite(first_pts) or not math.isfinite(last_pts) or last_pts < first_pts):
            raise CameraEvidenceError("episode-camera frame/timing result is failed or inconsistent", blocking=True)
        metadata_duration = expected["end"] - expected["start"]
        expected_duration = expected["expected_frames"] / ds.fps
        tolerance = 0.5 / ds.fps
        actual_duration = last_pts - first_pts + 1.0 / ds.fps
        if abs(metadata_duration - expected_duration) > tolerance or abs(actual_duration - metadata_duration) > tolerance:
            raise CameraEvidenceError("episode-camera duration disagrees with metadata", blocking=True)
        association["_start_offset"] = first_pts - expected["start"]
        association["_end_offset"] = last_pts - (expected["end"] - 1.0 / ds.fps)
        per_episode.setdefault(expected["episode_index"], []).append(association)
    for episode_index, associations in per_episode.items():
        if len(associations) != len(ds.video_keys):
            raise CameraEvidenceError(f"incomplete camera timing inventory for episode {episode_index}")
        starts = [item["_start_offset"] for item in associations]
        ends = [item["_end_offset"] for item in associations]
        if max(starts) - min(starts) > 0.5 / ds.fps or max(ends) - min(ends) > 0.5 / ds.fps:
            raise CameraEvidenceError(f"cross-camera timing disagreement for episode {episode_index}", blocking=True)

    if verification.get("complete") is not True:
        raise CameraEvidenceError("FFmpeg report does not attest to a complete inventory-wide scan")

    checks = report.get("checks")
    required_checks = ("frame_count", "duration", "fps", "resolution", "cross_camera_timing")
    if not isinstance(checks, dict) or any(checks.get(key) != "PASS" for key in required_checks):
        raise CameraEvidenceError("FFmpeg report contains missing or inconclusive camera consistency checks")
    discrepancies = report.get("discrepancies")
    if not isinstance(discrepancies, dict) or any(discrepancies.get(key) != 0 for key in required_checks):
        raise CameraEvidenceError("FFmpeg report contains camera consistency discrepancies")

    return {
        "declared_keys": list(ds.video_keys),
        "missing_keys": [],
        "missing_files": [],
        "corrupted": [],
        "checked_files": len(expected_paths),
        "timelines": {},
        "decode_check": "done",
        "verification_source": "FFmpeg full decode",
        "full_decode": True,
        "evidence_report": str(path.resolve()),
        "evidence_sha256_validated": True,
        "frame_count_discrepancies": 0,
        "duration_discrepancies": 0,
        "fps_discrepancies": 0,
        "resolution_discrepancies": 0,
        "cross_camera_timing_discrepancies": 0,
    }
