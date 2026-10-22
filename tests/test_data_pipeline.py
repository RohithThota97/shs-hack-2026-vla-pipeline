import json
import shutil
import sys
from types import SimpleNamespace

import pytest

from hackpipe.dataset.versioning import LeakageError, approve, create_dataset_version, leakage_guard
from hackpipe.ingest.importer import ingest, ingest_report
from hackpipe.quality.checks import INVALID, REVIEW, VALID, check_dataset
from hackpipe.quality.report import run_qc


def test_quality_flags_broken_episodes(dataset):
    res = check_dataset(dataset)
    by_ep = {e.episode_index: e for e in res["episodes"]}
    assert by_ep[0].status == VALID
    assert by_ep[2].status == INVALID and any("action_nan" in h for h in by_ep[2].hard)
    assert by_ep[4].status == REVIEW and any("missing_frames" in s for s in by_ep[4].soft)
    assert by_ep[6].status == REVIEW and any("too_short" in s for s in by_ep[6].soft)
    assert by_ep[8].status == INVALID and "timestamps_not_increasing" in by_ep[8].hard


def test_nonfinite_timestamp_is_invalid(tmp_path):
    from conftest import make_dataset
    ds = make_dataset(tmp_path / "timestamp", n_episodes=1, broken={0: "timestamp_nan"})
    result = check_dataset(ds)
    assert "timestamps_not_finite" in result["episodes"][0].hard


def test_configured_missing_success_label_is_unknown(tmp_path):
    from conftest import make_dataset
    from hackpipe.common.config import dataset_config
    ds = make_dataset(tmp_path / "success-label", n_episodes=1)
    cfg = dataset_config()
    cfg["columns"]["success"] = "next.success"
    result = check_dataset(ds, cfg)
    episode = result["episodes"][0]
    assert episode.stats["success_label"] == "UNKNOWN"
    assert "success_label_unavailable" in episode.soft


def test_malformed_episode_metadata_isolated_during_ingest(tmp_path, workspace):
    from conftest import make_dataset
    ds = make_dataset(tmp_path / "malformed", n_episodes=3)
    path = ds / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = __import__("pandas").read_parquet(path)
    table.loc[1, "length"] = -1
    table.to_parquet(path)
    _, reg = workspace
    report = ingest_report(ds, reg)
    assert report["imported"] == 2
    assert report["failed"][0]["episode_index"] == 1
    assert "episode metadata length must be positive" in report["failed"][0]["reasons"][0]


def test_cross_path_duplicate_content_is_reported(tmp_path, workspace):
    from conftest import make_dataset
    first = make_dataset(tmp_path / "copy-a", n_episodes=2)
    second = tmp_path / "copy-b"
    shutil.copytree(first, second)
    _, reg = workspace
    ingest(first, reg)
    report = ingest_report(second, reg)
    assert len(report["duplicates"]) == 2
    assert all(set(item["sources"]) == {str(first.resolve()), str(second.resolve())}
               for item in report["duplicates"])


def test_duplicate_numeric_trajectory_is_only_potential_duplicate(tmp_path):
    import pandas as pd
    from conftest import make_dataset
    ds = make_dataset(tmp_path / "numeric-duplicate", n_episodes=2)
    data_path = ds / "data" / "chunk-000" / "file-000.parquet"
    table = pd.read_parquet(data_path)
    episode_zero = table.index[table.episode_index == 0].tolist()
    episode_one = table.index[table.episode_index == 1].tolist()
    for column in ("observation.state", "action"):
        for source_row, target_row in zip(episode_zero, episode_one):
            table.at[target_row, column] = table.at[source_row, column].copy()
    table.to_parquet(data_path)
    report = check_dataset(ds)
    assert report["potential_duplicate_trajectories"][0]["episode_indices"] == [0, 1]
    assert "camera images were not compared" in report["potential_duplicate_trajectories"][0]["interpretation"]


def test_corrupt_camera_blocks_readiness(tmp_path, monkeypatch):
    from conftest import make_dataset
    ds = make_dataset(tmp_path / "corrupt-camera", n_episodes=1, cameras=True)

    class BrokenContainer:
        streams = SimpleNamespace(video=[SimpleNamespace(average_rate=10)])
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def decode(self, **kwargs): raise ValueError("synthetic decode error")

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda _: BrokenContainer()))
    report = check_dataset(ds)
    assert report["videos"]["corrupted"]
    assert report["readiness"]["status"] == "BLOCKED"


def test_inconclusive_camera_decode_prevents_version_creation(workspace, tmp_path, monkeypatch):
    from conftest import make_dataset
    ds = make_dataset(tmp_path / "skipped-camera", n_episodes=1, cameras=True)
    _, reg = workspace
    monkeypatch.setitem(sys.modules, "av", None)
    assert ingest(ds, reg) == 1
    report = run_qc(ds, reg)
    assert report["videos"]["decode_check"] == "skipped (PyAV unavailable)"
    assert report["readiness"]["status"] == "REVIEW"
    demo = reg.demos()[0]
    # Even an erroneous VALID registry status must not override inconclusive camera QC.
    reg.set_demo_status(demo["demo_id"], VALID)
    with pytest.raises(RuntimeError, match="missing or inconclusive QC evidence"):
        create_dataset_version(reg)


def _ffmpeg_camera_report(root, *, outcome="fully_decoded"):
    import hashlib
    from hackpipe.common.lerobot_io import DatasetFolder
    from hackpipe.quality.camera_evidence import dataset_identity

    ds = DatasetFolder(root)
    associations = []
    videos = {}
    for row in ds.episode_metadata().to_dict(orient="records"):
        for camera in ds.video_keys:
            prefix = f"videos/{camera}"
            path = ds.video_path_for_episode(row, camera).resolve()
            associations.append({
                "episode_index": int(row["episode_index"]), "camera": camera,
                "path": str(path), "expected_frames": int(row["length"]),
                "start": float(row[f"{prefix}/from_timestamp"]),
                "end": float(row[f"{prefix}/to_timestamp"]),
                "segment_frame_count": int(row["length"]),
                "first_pts": float(row[f"{prefix}/from_timestamp"]),
                "last_pts": float(row[f"{prefix}/to_timestamp"]) - 1.0 / ds.fps,
                "video_outcome": outcome,
            })
            if str(path) not in videos:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                videos[str(path)] = {
                    "path": str(path), "sha256": digest, "outcome": outcome,
                    "decoder": "ffmpeg", "full_decode": outcome == "fully_decoded",
                    "decoder_exit_code": 0 if outcome == "fully_decoded" else 1,
                    "frame_count": int(row["length"]), "fps": ds.fps,
                    "duration": int(row["length"]) / ds.fps,
                    "width": 16, "height": 16,
                }
    discrepancies = {key: 0 for key in
                     ("frame_count", "duration", "fps", "resolution", "cross_camera_timing")}
    return {
        "schema_version": 1,
        "verification": {"method": "ffmpeg_full_decode", "scanner": "hackpipe-camera-verifier-v3",
                         "complete": outcome == "fully_decoded", "tool_version": "test ffmpeg"},
        "dataset_identity": dataset_identity(ds),
        "root": str(ds.root.resolve()),
        "expected_episodes": ds.info["total_episodes"],
        "discovered_episodes": len(ds.episode_metadata()),
        "declared_camera_features": sorted(ds.video_keys),
        "episode_camera_stream_associations": len(associations),
        "unique_referenced_video_files": len(videos),
        "videos_fully_decoded": len(videos) if outcome == "fully_decoded" else 0,
        "videos_partially_decoded": len(videos) if outcome == "partially_decoded" else 0,
        "videos_failed": len(videos) if outcome == "failed" else 0,
        "videos_skipped": 0,
        "camera_readiness": "PASS" if outcome == "fully_decoded" else "FAIL",
        "checks": {key: "PASS" for key in discrepancies},
        "discrepancies": discrepancies,
        "episode_camera_associations": associations,
        "video_outcomes": list(videos.values()),
    }


def test_valid_ffmpeg_evidence_is_consumed_by_qc(tmp_path, monkeypatch):
    from conftest import make_dataset
    ds = make_dataset(tmp_path / "valid-ffmpeg-evidence", n_episodes=1, cameras=True)
    report = tmp_path / "camera-report.json"
    report.write_text(json.dumps(_ffmpeg_camera_report(ds)))
    monkeypatch.setitem(sys.modules, "av", None)

    result = check_dataset(ds, camera_report=report)

    assert result["videos"]["verification_source"] == "FFmpeg full decode"
    assert result["videos"]["full_decode"] is True
    assert result["videos"]["evidence_sha256_validated"] is True
    assert result["readiness"]["status"] == "READY"


def test_ffmpeg_evidence_preserves_numeric_duplicate_review(tmp_path, monkeypatch):
    import pandas as pd
    from conftest import make_dataset
    ds = make_dataset(tmp_path / "ffmpeg-numeric-duplicate", n_episodes=2, cameras=True)
    data_path = ds / "data" / "chunk-000" / "file-000.parquet"
    table = pd.read_parquet(data_path)
    for column in ("observation.state", "action"):
        first = table.index[table.episode_index == 0].tolist()
        second = table.index[table.episode_index == 1].tolist()
        for source_row, target_row in zip(first, second):
            table.at[target_row, column] = table.at[source_row, column].copy()
    table.to_parquet(data_path)
    report = tmp_path / "camera-report.json"
    report.write_text(json.dumps(_ffmpeg_camera_report(ds)))
    monkeypatch.setitem(sys.modules, "av", None)

    result = check_dataset(ds, camera_report=report)

    assert result["videos"]["verification_source"] == "FFmpeg full decode"
    assert result["potential_duplicate_trajectories"][0]["episode_indices"] == [0, 1]
    assert result["readiness"]["status"] == "REVIEW"


def test_run_qc_persists_ffmpeg_source_as_conclusive_evidence(workspace, tmp_path, monkeypatch):
    from conftest import make_dataset
    ds = make_dataset(tmp_path / "run-qc-ffmpeg-evidence", n_episodes=1, cameras=True)
    report = tmp_path / "camera-report.json"
    report.write_text(json.dumps(_ffmpeg_camera_report(ds)))
    _, registry = workspace
    assert ingest(ds, registry) == 1
    monkeypatch.setitem(sys.modules, "av", None)

    result = run_qc(ds, registry, out_dir=tmp_path / "qc-report", camera_report=report)

    demo = registry.demos()[0]
    assert result["videos"]["verification_source"] == "FFmpeg full decode"
    assert demo["meta"]["qc"]["video_check_inconclusive"] is False
    assert demo["meta"]["qc"]["videos"]["verification_source"] == "FFmpeg full decode"


def test_stale_ffmpeg_video_hash_is_rejected(tmp_path, monkeypatch):
    from conftest import make_dataset
    ds = make_dataset(tmp_path / "stale-ffmpeg-evidence", n_episodes=1, cameras=True)
    report = tmp_path / "camera-report.json"
    report.write_text(json.dumps(_ffmpeg_camera_report(ds)))
    video = next((ds / "videos").rglob("*.mp4"))
    video.write_bytes(video.read_bytes() + b"changed")
    monkeypatch.setitem(sys.modules, "av", None)

    result = check_dataset(ds, camera_report=report)

    assert result["videos"]["verification_source"] == "skipped"
    assert "SHA-256 changed" in result["videos"]["evidence_error"]
    assert result["readiness"]["status"] == "REVIEW"


def test_mismatched_ffmpeg_camera_inventory_is_rejected(tmp_path, monkeypatch):
    from conftest import make_dataset
    ds = make_dataset(tmp_path / "mismatched-ffmpeg-evidence", n_episodes=1, cameras=True)
    evidence = _ffmpeg_camera_report(ds)
    evidence["episode_camera_associations"].clear()
    report = tmp_path / "camera-report.json"
    report.write_text(json.dumps(evidence))
    monkeypatch.setitem(sys.modules, "av", None)

    result = check_dataset(ds, camera_report=report)

    assert result["videos"]["verification_source"] == "skipped"
    assert "inventory" in result["videos"]["evidence_error"]
    assert result["readiness"]["status"] == "REVIEW"


def test_missing_ffmpeg_report_is_inconclusive(tmp_path, monkeypatch):
    from conftest import make_dataset
    ds = make_dataset(tmp_path / "missing-ffmpeg-evidence", n_episodes=1, cameras=True)
    monkeypatch.setitem(sys.modules, "av", None)

    result = check_dataset(ds, camera_report=tmp_path / "does-not-exist.json")

    assert result["videos"]["verification_source"] == "skipped"
    assert "cannot read FFmpeg camera report" in result["videos"]["evidence_error"]
    assert result["readiness"]["status"] == "REVIEW"


@pytest.mark.parametrize("outcome", ["partially_decoded", "failed"])
def test_partial_or_failed_ffmpeg_evidence_blocks_readiness(tmp_path, monkeypatch, outcome):
    from conftest import make_dataset
    ds = make_dataset(tmp_path / f"{outcome}-ffmpeg-evidence", n_episodes=1, cameras=True)
    report = tmp_path / f"{outcome}.json"
    report.write_text(json.dumps(_ffmpeg_camera_report(ds, outcome=outcome)))
    monkeypatch.setitem(sys.modules, "av", None)

    result = check_dataset(ds, camera_report=report)

    assert result["videos"]["verification_source"] == "skipped"
    assert result["videos"]["corrupted"]
    assert result["readiness"]["status"] == "BLOCKED"


def test_blocking_qc_metadata_prevents_version_creation(workspace):
    _, reg = workspace
    reg.upsert_demo("demo", "source", 0, {"n_frames": 10, "qc": {
        "dataset_readiness": {"status": "BLOCKED", "blocking_issues": ["bad totals"]}}}, status=VALID)
    with pytest.raises(RuntimeError, match="blocking QC errors"):
        create_dataset_version(reg)


def test_missing_qc_evidence_prevents_version_creation(workspace, tmp_path):
    from conftest import make_dataset
    ds = make_dataset(tmp_path / "missing-qc", n_episodes=1)
    _, reg = workspace
    ingest(ds, reg)
    demo = reg.demos()[0]
    reg.set_demo_status(demo["demo_id"], VALID)
    with pytest.raises(RuntimeError, match="missing or inconclusive QC evidence"):
        create_dataset_version(reg)


def test_malformed_readiness_evidence_prevents_version_creation(workspace, tmp_path):
    from conftest import make_dataset
    ds = make_dataset(tmp_path / "malformed-qc", n_episodes=1)
    _, reg = workspace
    ingest(ds, reg)
    demo = reg.demos()[0]
    reg.set_demo_status(demo["demo_id"], VALID, {"qc": {
        "status": VALID, "checked_at": "present", "content_sha256": demo["meta"]["content_sha256"],
        "dataset_readiness": {"unexpected": "value"}, "video_check_inconclusive": False}})
    with pytest.raises(RuntimeError, match="missing or inconclusive QC evidence"):
        create_dataset_version(reg)


def test_source_change_after_qc_prevents_version_creation(workspace, tmp_path):
    import pandas as pd
    from conftest import make_dataset
    ds = make_dataset(tmp_path / "stale-qc", n_episodes=1)
    _, reg = workspace
    ingest(ds, reg)
    run_qc(ds, reg)
    data_path = ds / "data" / "chunk-000" / "file-000.parquet"
    table = pd.read_parquet(data_path)
    table.at[0, "action"] = table.at[0, "action"] + 0.25
    table.to_parquet(data_path)
    with pytest.raises(RuntimeError, match="stale, or inconclusive"):
        create_dataset_version(reg)


def test_dataset_blocker_prevents_version_creation(workspace, tmp_path):
    from conftest import make_dataset
    ds = make_dataset(tmp_path / "blocked-dataset", n_episodes=2)
    info_path = ds / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] += 1
    info_path.write_text(json.dumps(info))
    _, reg = workspace
    assert ingest(ds, reg) == 0
    report = run_qc(ds, reg)
    assert report["readiness"]["status"] == "BLOCKED"
    with pytest.raises(RuntimeError, match="No VALID demos"):
        create_dataset_version(reg)
    assert reg.datasets() == []


def test_source_change_during_qc_is_blocked(workspace, tmp_path, monkeypatch):
    import pandas as pd
    import hackpipe.quality.report as report_module
    from conftest import make_dataset
    ds = make_dataset(tmp_path / "changing-source", n_episodes=1)
    _, reg = workspace
    ingest(ds, reg)
    original_check = report_module.check_dataset

    def check_then_change(root):
        result = original_check(root)
        data_path = ds / "data" / "chunk-000" / "file-000.parquet"
        table = pd.read_parquet(data_path)
        table.at[0, "action"] = table.at[0, "action"] + 0.5
        table.to_parquet(data_path)
        return result

    monkeypatch.setattr(report_module, "check_dataset", check_then_change)
    result = run_qc(ds, reg)
    assert result["readiness"]["status"] == "BLOCKED"
    assert "source content changed while QC was running" in result["dataset_issues"]
    assert reg.demos(status=VALID) == []


def test_ingest_qc_dataset_flow(workspace, dataset):
    tmp, reg = workspace
    assert ingest(dataset, reg) == 10
    assert ingest(dataset, reg) == 10            # idempotent: same demo_ids
    assert len(reg.demos()) == 10
    report = run_qc(dataset, reg)
    assert report["status_counts"] == {VALID: 6, REVIEW: 2, INVALID: 2}
    m = create_dataset_version(reg)
    assert m["version"] == "ds-v1"
    all_eps = sum(len(v) for v in m["splits"].values())
    assert all_eps == 6                           # only VALID demos
    assert not set(m["splits"]["train"]) & set(m["splits"]["test"])
    assert reg.get_dataset("ds-v1")["status"] == "draft"
    approve("ds-v1", reg)
    assert reg.get_dataset("ds-v1")["status"] == "approved"
    # Same inputs → same fingerprint
    m2 = create_dataset_version(reg)
    assert m2["version"] == "ds-v2" and m2["fingerprint"] == m["fingerprint"]


def test_leakage_guard_detects_overlap():
    d = {"demo_id": "x", "meta": {}}
    with pytest.raises(LeakageError):
        leakage_guard({"train": [d], "val": [], "test": [d]}, None)


def test_train_dry_run_builds_command(workspace, dataset):
    from hackpipe.training.train import train
    tmp, reg = workspace
    ingest(dataset, reg)
    run_qc(dataset, reg)
    m = create_dataset_version(reg)
    with pytest.raises(RuntimeError):
        train("configs/train/act.yaml", "ds-v1", dry_run=True, registry=reg)   # not approved yet
    approve("ds-v1", reg)
    rec = train("configs/train/act.yaml", "ds-v1", dry_run=True, registry=reg)
    cmd = rec["command"]
    assert cmd[0] == "lerobot-train"
    assert f"--dataset.episodes={json.dumps(m['splits']['train'], separators=(',', ':'))}" in cmd
    assert "--policy.type=act" in cmd and "--seed=1000" in cmd
