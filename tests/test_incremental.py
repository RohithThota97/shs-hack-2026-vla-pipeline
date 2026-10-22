"""Automatic, incremental data pipeline: nothing is checked twice, duplicates are recognised, ds-v2 builds on ds-v1."""
import json
import shutil

import pytest

from conftest import add_fr3_extras, make_dataset, write_test_configs
from hackpipe.common.config import source_key
from hackpipe.dataset.materialize import MaterializeError, plan, verify
from hackpipe.dataset.versioning import (NothingNewError, approve, create_dataset_version, latest_approved,
                                         load_manifest, set_excluded)
from hackpipe.ingest.importer import import_demos
from hackpipe.pipeline.auto import run_auto
from hackpipe.quality.report import run_qc


def src(root, n, seed0, broken=None):
    return add_fr3_extras(make_dataset(root, n_episodes=n, frames=30, broken=broken), n, 30,
                          seeds=[seed0 + i for i in range(n)], video=None)


def test_reimport_skips_unchanged_and_uses_hash_cache(workspace):
    tmp, reg = workspace
    a = src(tmp / "a", 6, 7000)
    first = import_demos(a, reg)
    assert len(first["new"]) == 6 and first["hash_cache"]["misses"] >= 1
    again = import_demos(a, reg)
    assert len(again["unchanged"]) == 6 and not again["new"] and again["hash_cache"]["misses"] == 0


def test_copy_of_a_folder_is_duplicate(workspace):
    tmp, reg = workspace
    a = src(tmp / "a", 6, 7000)
    import_demos(a, reg)
    run_qc(a, reg)
    shutil.copytree(a, tmp / "copy")
    r = import_demos(tmp / "copy", reg)
    assert len(r["duplicate"]) == 6
    dups = reg.demos(source=source_key(tmp / "copy"))
    assert {d["status"] for d in dups} == {"DUPLICATE"} and all(d["meta"]["duplicate_of"] for d in dups)
    originals = {d["demo_id"]: d["episode_index"] for d in reg.demos(source=source_key(a))}
    assert all(originals[d["meta"]["duplicate_of"]] == d["episode_index"] for d in dups)   # ep i ↔ ep i
    q = run_qc(tmp / "copy", reg)
    assert q["checked_now"] == 0 and q["duplicates"] == list(range(6))
    assert {d["status"] for d in reg.demos(source=source_key(tmp / "copy"))} == {"DUPLICATE"}


def test_qc_checks_only_new_demos(workspace):
    tmp, reg = workspace
    a = src(tmp / "a", 6, 7000, broken={2: "nan"})
    import_demos(a, reg)
    first = run_qc(a, reg)
    assert first["checked_now"] == 6 and first["status_counts"]["INVALID"] == 1
    again = run_qc(a, reg)
    assert again["checked_now"] == 0 and again["reused"] == 6 and again["status_counts"] == first["status_counts"]
    assert run_qc(a, reg, full=True)["checked_now"] == 6


def _v1(tmp, reg, test_configs):
    write_test_configs(test_configs, splits__train=0.8, splits__val=0.1, splits__test=0.1,
                       splits__leakage_keys=["environment.seed"])
    a = src(tmp / "a", 10, 7000)
    r = run_auto(str(a), reg)
    assert r["version"] == "ds-v1" and r["qc"]["checked_now"] == 10
    approve("ds-v1", reg, by="Rohith")
    return a


def test_v2_inherits_v1_and_adds_only_new(workspace, test_configs):
    tmp, reg = workspace
    _v1(tmp, reg, test_configs)
    v1 = load_manifest("ds-v1", reg)
    assert reg.get_dataset("ds-v1")["meta"]["approved_by"] == "Rohith"
    b = src(tmp / "b", 4, 8000)
    r = run_auto(str(b), reg)
    assert r["version"] == "ds-v2" and r["base"] == "ds-v1"
    assert r["qc"]["checked_now"] == 4                          # only the new demos were checked
    assert r["inherited"] == 9 and r["added"] == 4 and r["removed"] == 0 and r["needs_materialize"]
    v2 = load_manifest("ds-v2", reg)
    for k in ("train", "val"):
        assert set(v1["demo_ids"][k]) <= set(v2["demo_ids"][k])  # v1 assignment kept
    assert v2["demo_ids"]["test"] == v1["demo_ids"]["test"]     # frozen test
    added = {d["demo_id"] for d in reg.demos(source=source_key(b))}
    assert added <= set(v2["demo_ids"]["train"]) | set(v2["demo_ids"]["val"])
    card = (tmp / "outputs" / "datasets" / "ds-v2" / "dataset_card.md").read_text()
    assert "Inherited vs new (base ds-v1)" in card and "| Added (new VALID demos) | 4" in card


def test_auto_without_changes_creates_nothing(workspace, test_configs):
    tmp, reg = workspace
    a = _v1(tmp, reg, test_configs)
    r = run_auto(str(a), reg)
    assert r["version"] is None and "Nothing new since ds-v1" in r["reason"]
    assert r["import"]["unchanged"] == 10 and r["qc"] is None
    assert len(reg.datasets()) == 1


def test_auto_waits_for_pending_draft(workspace, test_configs):
    tmp, reg = workspace
    write_test_configs(test_configs, splits__leakage_keys=[])
    a = src(tmp / "a", 10, 7000)
    assert run_auto(str(a), reg)["version"] == "ds-v1"
    r = run_auto(str(a), reg)
    assert r["version"] is None and "ds-v1 is waiting for approval" in r["reason"]


def test_excluded_demo_is_removed_in_next_version(workspace, test_configs):
    tmp, reg = workspace
    a = _v1(tmp, reg, test_configs)
    v1 = load_manifest("ds-v1", reg)
    victim = v1["demo_ids"]["train"][0]
    with pytest.raises(ValueError):
        set_excluded(victim, True, registry=reg)                 # reason and name required
    set_excluded(victim, True, reason="gripper glitch in video", by="Rohith", registry=reg)
    m = create_dataset_version(reg, base=latest_approved(reg))
    assert victim not in m["demo_ids"]["train"] and m["lineage"]["removed"] == [victim]
    assert any(e["status"] == "EXCLUDED" and "gripper glitch" in e["reasons"][0] for e in m["excluded"])
    approve(m["version"], reg)
    set_excluded(victim, False, registry=reg)                    # include again → back in the next version
    m3 = create_dataset_version(reg, base=latest_approved(reg))
    assert victim in m3["lineage"]["added"]
    with pytest.raises(NothingNewError):
        approve(m3["version"], reg)
        create_dataset_version(reg, base=latest_approved(reg))


def test_base_must_be_approved(workspace, test_configs):
    tmp, reg = workspace
    write_test_configs(test_configs, splits__leakage_keys=[])
    run_auto(str(src(tmp / "a", 10, 7000)), reg)
    with pytest.raises(RuntimeError, match="approved"):
        create_dataset_version(reg, base="ds-v1")


def test_materialize_plan_and_verification(workspace, test_configs):
    tmp, reg = workspace
    _v1(tmp, reg, test_configs)
    run_auto(str(src(tmp / "b", 4, 8000)), reg)
    p = plan("ds-v2", reg)
    sa, sb = sorted([source_key(tmp / "a"), source_key(tmp / "b")])
    assert p["offsets"] == {sa: 0, sb: 10} and p["total_episodes"] == 14
    assert p["command"][:3] == ["lerobot-edit-dataset", "--new_repo_id", "local/ds-v2"]
    assert "merge" in p["command"] and json.loads(p["command"][-1])[1].endswith("/b")
    m = load_manifest("ds-v2", reg)
    assert sorted(p["splits"]["train"] + p["splits"]["val"]) == sorted(
        p["offsets"][s] + i for s, i in m["episodes"]["train"] + m["episodes"]["val"])
    make_dataset(tmp / "data" / "processed" / "ds-v2", n_episodes=14, frames=30)   # what a correct merge gives
    p["root"] = str(tmp / "data" / "processed" / "ds-v2")
    assert verify(p, m, reg) == len(m["episodes"]["train"]) + len(m["episodes"]["val"])
    shutil.rmtree(p["root"])
    wrong = sorted(p["offsets"][s] + i for s, i in m["episodes"]["train"])[0]
    make_dataset(tmp / "data" / "processed" / "ds-v2", n_episodes=14, frames=30, broken={wrong: "nan"})
    with pytest.raises(MaterializeError, match="did not concatenate"):
        verify(p, m, reg)


def test_training_needs_materialized_multi_source(workspace, test_configs):
    from hackpipe.training.train import train
    tmp, reg = workspace
    _v1(tmp, reg, test_configs)
    run_auto(str(src(tmp / "b", 4, 8000)), reg)
    approve("ds-v2", reg)
    with pytest.raises(RuntimeError, match="materialize ds-v2"):
        train("configs/train/smolvla_fr3_duo.yaml", "ds-v2", dry_run=True, registry=reg)
    p = plan("ds-v2", reg)
    path = reg.get_dataset("ds-v2")["manifest_path"]
    stored = json.loads(open(path).read())
    stored["materialized"] = {k: p[k] for k in ("root", "repo_id", "offsets", "total_episodes", "splits")}
    open(path, "w").write(json.dumps(stored))
    cmd = train("configs/train/smolvla_fr3_duo.yaml", "ds-v2", dry_run=True, registry=reg)["command"]
    assert "--dataset.repo_id=local/ds-v2" in cmd and any(c.endswith("data/processed/ds-v2") for c in cmd)
    assert f"--dataset.episodes={json.dumps(p['splits']['train'], separators=(',', ':'))}" in cmd
