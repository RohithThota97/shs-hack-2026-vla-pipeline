"""A subtask recorded at another fps (rand500: 6 fps next to 30 fps handover) can be kept in a dataset version as
data, but such a version can't be merged or trained as one folder."""
import json

import pytest
import yaml
from conftest import make_dataset, write_test_configs

from hackpipe.dataset.versioning import approve, create_dataset_version, load_manifest
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
    assert sorted(m["demo_ids"]["test"]) == sorted(load_manifest("ds-v1", reg)["demo_ids"]["test"])
    card = open(reg.get_dataset("ds-v2")["meta"]["card"]).read()
    assert "## Sources (different schemas)" in card and "not trainable as one folder" in card


def test_mixed_schemas_refused_unless_allowed(two_fps, monkeypatch):
    reg, a, b, cfg = two_fps
    ds = yaml.safe_load((cfg / "dataset.yaml").read_text())
    ds["versions"]["allow_mixed_schemas"] = False
    (cfg / "dataset.yaml").write_text(yaml.safe_dump(ds))
    out = run_auto(str(b), reg)
    assert out["version"] is None and "allow_mixed_schemas" in out["reason"]


def test_mixed_version_cannot_be_merged_or_trained(two_fps):
    from hackpipe.dataset.materialize import MaterializeError, materialize
    from hackpipe.training.train import training_input
    reg, a, b, _ = two_fps
    run_auto(str(b), reg)
    with pytest.raises(MaterializeError, match="Same fps"):
        materialize("ds-v2", reg, run=False)
    with pytest.raises(RuntimeError, match="different fps"):
        training_input(load_manifest("ds-v2", reg))
    assert training_input(load_manifest("ds-v1", reg))["episodes"]           # ds-v1 still trains


def test_single_source_fingerprint_unchanged_by_versions_section(workspace, monkeypatch, tmp_path):
    tmp, reg = workspace
    a = make_dataset(tmp / "a", n_episodes=10, frames=30)
    fps = []
    for allow in (True, False):
        cfg = write_test_configs(tmp_path / f"cfg{allow}")
        ds = yaml.safe_load((cfg / "dataset.yaml").read_text())
        ds["versions"]["allow_mixed_schemas"] = allow
        (cfg / "dataset.yaml").write_text(yaml.safe_dump(ds))
        monkeypatch.setenv("HACKPIPE_CONFIG_DIR", str(cfg))
        if not reg.demos():
            run_auto(str(a), reg)
        fps.append(create_dataset_version(reg, notes=str(allow))["fingerprint"])
    assert fps[0] == fps[1]
