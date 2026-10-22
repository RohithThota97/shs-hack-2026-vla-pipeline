"""Model + model card publishing to the Hub, with a fake API (no network)."""
import json

import pytest

from hackpipe.dataset.versioning import approve, create_dataset_version
from hackpipe.hub.publish import build_model_card, publish
from hackpipe.ingest.importer import ingest
from hackpipe.quality.report import run_qc


class FakeApi:
    def __init__(self):
        self.repos, self.commits = [], []

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        self.repos.append({"repo_id": repo_id, "private": private, "repo_type": repo_type})

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits.append({"repo_id": repo_id, "files": sorted(o.path_in_repo for o in operations)})
        return type("Info", (), {"oid": "abc123"})()


@pytest.fixture
def trained_model(workspace, dataset):
    tmp, reg = workspace
    ingest(dataset, reg)
    run_qc(dataset, reg)
    create_dataset_version(reg)
    approve("ds-v1", reg)
    ckpt = tmp / "outputs" / "train" / "m-v1" / "checkpoints" / "last" / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")
    (ckpt / "model.safetensors").write_bytes(b"weights")
    run_dir = tmp / "outputs" / "runs" / "m-v1"
    run_dir.mkdir(parents=True)
    record = {"model_version": "m-v1", "train_config": {"seed": 1000, "args": {
        "policy.path": "lerobot/smolvla_base", "steps": 200, "batch_size": 8}}, "overrides": {"steps": 200},
        "duration_s": 12.5, "exit_code": 0, "environment": {"git": {"commit": "deadbeef"}, "gpu": "Tesla T4"}}
    (run_dir / "run_record.json").write_text(json.dumps(record))
    (run_dir / "loss.csv").write_text("step,loss\n100,0.9\n200,0.42\n")
    reg.add_model("m-v1", "ds-v1", "smolvla", "trained", str(ckpt), str(run_dir / "run_record.json"), {})
    return tmp, reg, ckpt


def test_card_without_eval_says_not_evaluated(trained_model):
    _, reg, _ = trained_model
    card = build_model_card("m-v1", "user/smolvla-test", reg)
    assert card.startswith("---\nlibrary_name: lerobot")
    assert "base_model: lerobot/smolvla_base" in card
    assert "Not evaluated yet" in card and "n/a — not evaluated yet." in card
    assert "| Final training loss | 0.42 |" in card and "Tesla T4" in card and "deadbeef" in card
    assert "**not** published" in card


def test_dry_run_uploads_nothing(trained_model):
    tmp, reg, _ = trained_model
    api = FakeApi()
    plan = publish("m-v1", "user/smolvla-test", dry_run=True, registry=reg, api=api)
    assert plan["uploaded"] is False and plan["private"] is True and not api.commits
    assert {"config.json", "model.safetensors", "README.md", "run_record.json", "dataset_summary.json",
            "loss.csv"} <= set(plan["files"])
    assert "hub" not in reg.get_model("m-v1")["meta"]


def test_publish_private_and_registers(trained_model):
    _, reg, _ = trained_model
    api = FakeApi()
    out = publish("m-v1", "user/smolvla-test", registry=reg, api=api)
    assert api.repos == [{"repo_id": "user/smolvla-test", "private": True, "repo_type": "model"}]
    assert "model.safetensors" in api.commits[0]["files"] and "README.md" in api.commits[0]["files"]
    assert out["commit"] == "abc123"
    hub = reg.get_model("m-v1")["meta"]["hub"]
    assert hub["repo_id"] == "user/smolvla-test" and hub["commit"] == "abc123" and hub["private"] is True


def test_refuses_data_files(trained_model):
    _, reg, ckpt = trained_model
    (ckpt / "leak.parquet").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="data-like"):
        publish("m-v1", "user/smolvla-test", registry=reg, api=FakeApi())


def test_needs_repo_id(trained_model):
    _, reg, _ = trained_model
    with pytest.raises(ValueError, match="hub.model_repo_id"):
        publish("m-v1", None, registry=reg, api=FakeApi())


def test_repo_id_from_config(trained_model, test_configs):
    import yaml
    _, reg, _ = trained_model
    proj = yaml.safe_load((test_configs / "project.yaml").read_text())
    proj["hub"]["model_repo_id"] = "someone/from-config"
    (test_configs / "project.yaml").write_text(yaml.safe_dump(proj))
    assert publish("m-v1", None, dry_run=True, registry=reg)["repo_id"] == "someone/from-config"
