"""Best-model ranking and fine-tuning from it (Train page, automatic mode)."""
import json
from pathlib import Path

import pytest

CFG = "configs/train/smolvla_fr3_duo.yaml"


def _approved(workspace, dataset):
    from hackpipe.dataset.versioning import approve, create_dataset_version
    from hackpipe.ingest.importer import ingest
    from hackpipe.quality.report import run_qc
    tmp, reg = workspace
    ingest(dataset, reg)
    run_qc(dataset, reg)
    create_dataset_version(reg)
    approve("ds-v1", reg, by="t")
    return tmp, reg


def _model(reg, tmp, v, status="trained", purpose="model", trainer="lerobot-train", ckpt=True, loss=None,
           offline=None):
    from hackpipe.common.config import project_config, resolve
    ck = tmp / "ck" / v
    if ckpt:
        ck.mkdir(parents=True)
        (ck / "config.json").write_text("{}")
    run = resolve(project_config()["project"]["outputs_dir"]) / "runs" / v
    run.mkdir(parents=True, exist_ok=True)
    (run / "run_record.json").write_text(json.dumps({"overrides": {"batch_size": 4, "steps": 2000}}))
    if loss is not None:
        (run / "loss.csv").write_text(f"step,loss\n100,0.9\n200,{loss}\n")
    if offline:
        (run / "offline_eval_test.json").write_text(json.dumps({"rows": [
            {"instruction": "full", "subtask": "h · all", "chunk_arm_rad": offline[0], "hold_still_arm_rad": offline[1]},
            {"instruction": "sub", "subtask": "h · all", "chunk_arm_rad": 0.0, "hold_still_arm_rad": 1.0}]}))
    reg.add_model(v, "ds-v1", "smolvla", status, str(ck), str(run / "run_record.json"),
                  {"output_dir": str(tmp / "train" / v), "train_config_file": CFG, "purpose": purpose,
                   "trainer": trainer})


def test_ranking_tiers(workspace, dataset):
    from hackpipe.training import ranking
    tmp, reg = _approved(workspace, dataset)
    _model(reg, tmp, "m-v1", purpose="timing", loss=0.01)               # never a candidate
    _model(reg, tmp, "m-v2", loss=0.2)
    _model(reg, tmp, "m-v3", loss=0.5, offline=(0.04, 0.05))           # beats hold-still → offline tier
    _model(reg, tmp, "m-v4", loss=0.1, offline=(0.50, 0.05))           # worse than hold-still → loss tier
    _model(reg, tmp, "m-v5", status="training", loss=0.05)
    _model(reg, tmp, "m-v6", ckpt=False, offline=(0.01, 0.05))
    order = [r["model"]["version"] for r in ranking.ranking(reg)]
    assert order[:3] == ["m-v3", "m-v4", "m-v2"]
    assert ranking.best_model(reg)["model"]["version"] == "m-v3"
    s = {r["model"]["version"]: r for r in ranking.ranking(reg)}
    assert s["m-v1"]["why_not"] == "timing run, not a result" and "worse than hold-still" in s["m-v4"]["label"]
    assert s["m-v6"]["why_not"] == "no checkpoint on this machine"
    reg.add_eval("ev-1", "m-v2", "sim-handover-v1", "isaac", str(tmp), {"task_success": {"rate": 0.3}})
    reg.add_eval("ev-2", "m-v4", "sim-handover-v1", "isaac", str(tmp),
                 {"task_success": {"rate": 0.9}, "fake_results": True})        # fake results never count
    assert ranking.best_model(reg)["model"]["version"] == "m-v2"


def test_fine_tune_dry_run(workspace, dataset):
    from hackpipe.training.train import train
    tmp, reg = _approved(workspace, dataset)
    _model(reg, tmp, "m-v1", loss=0.2)
    rec = train(None, "ds-v1", dry_run=True, registry=reg, init_from="m-v1")    # config = the model's own
    assert rec["init_from"] == "m-v1" and rec["train_config_file"].endswith(CFG)
    assert f"--policy.path={tmp / 'ck' / 'm-v1'}" in rec["command"]
    with pytest.raises(ValueError, match="resume"):
        train(CFG, "ds-v1", registry=reg, init_from="m-v1", resume_model="m-v1")
    _model(reg, tmp, "m-v2", trainer="curriculum")
    with pytest.raises(ValueError, match="trained with curriculum"):
        train(CFG, "ds-v1", dry_run=True, registry=reg, init_from="m-v2")
    _model(reg, tmp, "m-v3", ckpt=False)
    with pytest.raises(FileNotFoundError):
        train(CFG, "ds-v1", dry_run=True, registry=reg, init_from="m-v3")


def test_init_leakage(workspace, dataset, monkeypatch):
    from hackpipe.training import ranking
    tmp, reg = _approved(workspace, dataset)
    trained = {"episodes": {"train": [["data/raw/a", 0], ["data/raw/a", 1]], "val": [["data/raw/a", 2]]}}
    monkeypatch.setattr("hackpipe.dataset.versioning.load_manifest", lambda v, r=None: trained)
    target = {"episodes": {"test": [["/elsewhere/a", 2], ["data/raw/b", 0]]}}
    assert ranking.init_leakage({"dataset_version": "ds-x"}, target, reg) == [("a", 2)]


def test_train_page_automatic_mode(workspace, dataset, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("HACKPIPE_MACHINE", "laptop-test")
    tmp, reg = _approved(workspace, dataset)
    _model(reg, tmp, "m-v1", loss=0.2)
    _model(reg, tmp, "m-v2", loss=0.5, offline=(0.04, 0.05))
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/hackpipe/console/views/train.py"),
                           default_timeout=30)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("Already trained on ds-v1: 2 model(s)" in e.label for e in at.expander)
    assert any("already trained m-v1, m-v2" in i.value for i in at.info)       # manual mode, same config
    start = next(b for b in at.button if b.label == "Start training")
    assert start.disabled                                                      # no GPU here
    at.radio[0].set_value("Automatic — fine-tune the best model").run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("Starting point: m-v2" in m.value for m in at.markdown)
    assert at.text_area[0].value == "batch_size=4\nsteps=2000"                 # m-v2's overrides
    assert any("starting from **m-v2** (fine-tune)" in m.value for m in at.markdown)


def test_curriculum_val_l1_when_no_loss_csv(workspace, dataset):
    from hackpipe.training import ranking
    tmp, reg = _approved(workspace, dataset)
    _model(reg, tmp, "m-v1", trainer="curriculum")
    out = tmp / "train" / "m-v1"
    out.mkdir(parents=True)
    (out / "curriculum_log.json").write_text(json.dumps({"val": [
        {"step": 500, "stage": "s1", "h/full": {"l1_norm": 0.9}},
        {"step": 1000, "stage": "s4", "h/full": {"l1_norm": 0.3}, "h/sub": {"l1_norm": 0.5}}]}))
    s = ranking.score(reg.get_model("m-v1"), reg)
    assert s["tier"] == "loss" and s["value"] == pytest.approx(0.4) and s["label"] == "validation L1 0.4000"


def test_review_guards(workspace, dataset, monkeypatch):
    from hackpipe.training import ranking
    from hackpipe.training.train import train
    tmp, reg = _approved(workspace, dataset)
    _model(reg, tmp, "m-v1", loss=float("nan"))                         # diverged run: never best
    _model(reg, tmp, "m-v2", loss=0.9)
    assert ranking.best_model(reg)["model"]["version"] == "m-v2"
    _model(reg, tmp, "m-v3", offline=(0.06, 0.12))                       # 50% of hold-still
    _model(reg, tmp, "m-v4", offline=(0.04, 0.05))                       # 80%: lower rad, worse ratio
    assert ranking.best_model(reg)["model"]["version"] == "m-v3"
    reg.update_model_meta("m-v3", {"train_config_file": "configs/train/gone.yaml"})
    assert "not found" in ranking.score(reg.get_model("m-v3"), reg)["why_not"]
    rec = train("configs/train/act.yaml", "ds-v1", dry_run=True, registry=reg, init_from="m-v2")
    assert not any(c.startswith("--policy.type") for c in rec["command"])


def test_leakage_follows_fine_tune_chain(workspace, dataset, monkeypatch):
    from hackpipe.training import ranking
    tmp, reg = _approved(workspace, dataset)
    _model(reg, tmp, "m-v1")
    _model(reg, tmp, "m-v2")
    reg.update_model_meta("m-v2", {"init_from": "m-v1"})
    reg.conn.execute("UPDATE models SET dataset_version='ds-v2' WHERE version='m-v2'")
    mans = {"ds-v1": {"episodes": {"train": [["data/raw/a", 3]], "val": []}},
            "ds-v2": {"episodes": {"train": [["data/raw/b", 0]], "val": []}}}
    monkeypatch.setattr("hackpipe.dataset.versioning.load_manifest", lambda v, r=None: mans[v])
    target = {"episodes": {"test": [["data/raw/a", 3]]}}
    assert ranking.init_leakage(reg.get_model("m-v2"), target, reg) == [("a", 3)]
