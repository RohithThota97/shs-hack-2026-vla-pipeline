import json

import pytest

from hackpipe.dataset.versioning import LeakageError, approve, create_dataset_version, leakage_guard
from hackpipe.ingest.importer import ingest
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


@pytest.mark.parametrize("fps,frames,short", [(30, 2247, False), (6, 345, False), (30, 1400, True), (6, 290, True)])
def test_length_limit_is_in_seconds(fps, frames, short):
    import pandas as pd
    from hackpipe.common.config import CONFIG_DIR, load_yaml
    from hackpipe.quality.checks import check_episode
    cfg = load_yaml(CONFIG_DIR / "dataset.yaml")                 # the real limits, not the test config
    cfg["quality"].update({"joint_limits": None, "final_state": None})
    ep = pd.DataFrame({"episode_index": 0, "frame_index": range(frames),
                       "timestamp": [i / fps for i in range(frames)]})
    res = check_episode(ep, fps, {}, cfg)
    assert any(s.startswith("too_short") for s in res.soft) == short
    assert not any(s.startswith("too_long") for s in res.soft)
