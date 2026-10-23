"""A subtask recorded at another fps (rand500: 6 fps next to 30 fps handover) can be kept in a dataset version as
data, but such a version can't be merged or trained as one folder."""
import json

import pytest
import yaml
from conftest import make_dataset, write_test_configs

from hackpipe.dataset.versioning import approve, create_dataset_version, load_manifest
from hackpipe.common.config import source_key
from hackpipe.pipeline.auto import run_auto


def _set_fps(root, fps):
    """Same frames recorded at another rate: info.json fps and the timestamps both change."""
    import numpy as np
    import pandas as pd
    info = json.loads((root / "meta" / "info.json").read_text())
    f = root / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(f)
    df["timestamp"] = (df["frame_index"] / fps).astype(np.float32)
    df.to_parquet(f)
    info["fps"] = fps
    (root / "meta" / "info.json").write_text(json.dumps(info))
    return root


@pytest.fixture
def two_fps(workspace, monkeypatch, tmp_path):
    tmp, reg = workspace
    cfg = write_test_configs(tmp_path / "cfg")
    monkeypatch.setenv("HACKPIPE_CONFIG_DIR", str(cfg))
    a = make_dataset(tmp / "a", n_episodes=10, frames=30)                  # 10 fps, 3 s
    proj = yaml.safe_load((cfg / "project.yaml").read_text())
    proj["dataset"]["root"] = str(a)                                      # the main dataset (handover's role)
    (cfg / "project.yaml").write_text(yaml.safe_dump(proj))
    b = _set_fps(make_dataset(tmp / "b", n_episodes=10, frames=40), 5)     # 5 fps, 8 s (other content)
    run_auto(str(a), reg)
    approve("ds-v1", reg, by="t")
    return reg, a, b, cfg


def test_mixed_version_kept_as_data_with_frozen_test(two_fps):
    reg, a, b, _ = two_fps
    out = run_auto(str(b), reg)
    m = load_manifest(out["version"], reg)
    assert out["version"] == "ds-v2" and m["mixed_schemas"] and out["inherited"] > 0
    fps = {s: sc["fps"] for s, sc in m["preprocessing"]["per_source_schema"].items()}
    assert sorted(fps.values()) == [5, 10]
    assert m["preprocessing"]["schema"]["fps"] == 10                      # the trained (primary) schema
    assert sorted(m["demo_ids"]["test"]) == sorted(load_manifest("ds-v1", reg)["demo_ids"]["test"])
    assert out["data_only_sources"] == [source_key(b)] and out["needs_materialize"] is False
    card = open(reg.get_dataset("ds-v2")["meta"]["card"]).read()
    assert "## Sources (different schemas)" in card and "sources differ (fps)" in card and "data only" in card


def test_mixed_schemas_refused_unless_allowed(two_fps, monkeypatch):
    reg, a, b, cfg = two_fps
    ds = yaml.safe_load((cfg / "dataset.yaml").read_text())
    ds["versions"]["allow_mixed_schemas"] = False
    (cfg / "dataset.yaml").write_text(yaml.safe_dump(ds))
    out = run_auto(str(b), reg)
    assert out["version"] is None and "allow_mixed_schemas" in out["reason"]


def test_mixed_version_trains_on_main_schema_only(two_fps):
    from hackpipe.dataset.materialize import materialize
    from hackpipe.training.train import training_input
    reg, a, b, _ = two_fps
    run_auto(str(b), reg)
    m2 = load_manifest("ds-v2", reg)
    inp = training_input(m2)
    a_train = sorted(i for s, i in m2["episodes"]["train"] if s == source_key(a))
    assert inp["root"].endswith("/a") and inp["episodes"] == a_train and inp["data_only_sources"] == [source_key(b)]
    assert a_train == load_manifest("ds-v1", reg)["splits"]["train"]       # same as ds-v1's handover train
    assert materialize("ds-v2", reg, run=False)["needed"] is False


def test_next_version_after_approving_mixed_still_trains(two_fps, tmp_path):
    """Approving the mixed ds-v2 must not block training on new main-task data (ds-v3 is based on ds-v2)."""
    from hackpipe.training.train import training_input
    reg, a, b, _ = two_fps
    run_auto(str(b), reg)
    approve("ds-v2", reg, by="t")
    c = make_dataset(tmp_path / "c", n_episodes=6, frames=50)              # new 10 fps main-task data
    out = run_auto(str(c), reg)
    m3 = load_manifest(out["version"], reg)
    assert out["version"] == "ds-v3" and m3["mixed_schemas"]
    assert out["needs_materialize"] is True                               # a + c: merge, b stays data only
    from hackpipe.dataset.materialize import plan
    assert source_key(b) not in plan("ds-v3", reg)["sources"]
    with pytest.raises(RuntimeError, match="merge them first"):
        training_input(m3)


def test_other_task_seeds_are_not_leakage():
    from hackpipe.dataset.versioning import LeakageError, leakage_guard

    def d(i, src, seed):
        return {"demo_id": f"{src}{i}", "source": src, "meta": {"environment": {"seed": seed}}}
    splits = {"train": [d(1, "main", 1), d(2, "cubes", 7)], "val": [], "test": [d(3, "test", 7)]}
    leakage_guard(splits, None, ["environment.seed"], other_task={"cubes"})
    with pytest.raises(LeakageError):
        leakage_guard(splits, None, ["environment.seed"])


def test_single_source_fingerprint_unchanged_by_versions_section(workspace, monkeypatch, tmp_path):
    tmp, reg = workspace
    a = make_dataset(tmp / "a", n_episodes=10, frames=30)
    fps = []
    for allow in (True, False, None):                 # None: no versions section at all (before it existed)
        cfg = write_test_configs(tmp_path / f"cfg{allow}")
        ds = yaml.safe_load((cfg / "dataset.yaml").read_text())
        if allow is None:
            del ds["versions"]
        else:
            ds["versions"]["allow_mixed_schemas"] = allow
        (cfg / "dataset.yaml").write_text(yaml.safe_dump(ds))
        monkeypatch.setenv("HACKPIPE_CONFIG_DIR", str(cfg))
        if not reg.demos():
            run_auto(str(a), reg)
        fps.append(create_dataset_version(reg, notes=str(allow))["fingerprint"])
    assert fps[0] == fps[1] == fps[2]


def test_train_frames_and_train_page_with_mixed_version(two_fps):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from hackpipe.training.progress import train_frames
    reg, a, b, _ = two_fps
    run_auto(str(b), reg)
    approve("ds-v2", reg, by="t")
    assert train_frames("ds-v2", reg) == train_frames("ds-v1", reg)        # the subtask isn't trained
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/hackpipe/console/views/train.py"),
                           default_timeout=30)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("kept as data only" in c.value for c in at.caption)


def test_mixed_without_main_dataset_is_refused(two_fps):
    reg, a, b, cfg = two_fps
    proj = yaml.safe_load((cfg / "project.yaml").read_text())
    proj["dataset"]["root"] = None
    (cfg / "project.yaml").write_text(yaml.safe_dump(proj))
    out = run_auto(str(b), reg)
    assert out["version"] is None and "dataset.root" in out["reason"]


def test_trained_sources_fixed_at_creation(two_fps):
    """Changing dataset.root after creating a version doesn't change what that version trains."""
    from hackpipe.training.train import training_input
    reg, a, b, cfg = two_fps
    run_auto(str(b), reg)
    proj = yaml.safe_load((cfg / "project.yaml").read_text())
    proj["dataset"]["root"] = str(b)
    (cfg / "project.yaml").write_text(yaml.safe_dump(proj))
    assert training_input(load_manifest("ds-v2", reg))["root"].endswith("/a")


def test_old_manifest_without_sources():
    from hackpipe.dataset.versioning import trainable_sources
    assert trainable_sources({"version": "ds-v1", "source": None, "preprocessing": {}}) == ([], [])
