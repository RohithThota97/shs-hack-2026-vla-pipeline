import json
from pathlib import Path

from hackpipe.dataset.versioning import approve, create_dataset_version
from hackpipe.deploy.deploy import deploy, rollback, smoke_test
from hackpipe.evaluation.metrics import compute_metrics
from hackpipe.evaluation.runner import load_results, run_suite
from hackpipe.evaluation.schema import EpisodeResult
from hackpipe.explain.failures import categorize, summary
from hackpipe.gates.gates import evaluate_gates
from hackpipe.improve.gaps import recommend
from hackpipe.ingest.importer import ingest
from hackpipe.judge.judge import StateTraceJudge, agreement
from hackpipe.quality.report import run_qc
from hackpipe.reporting.model_card import build_card
from hackpipe.sim.adapter import DummySimAdapter

IDENT = dict(eval_id="e", model_version="m-v1", dataset_version="ds-v1", test_suite="test-v1",
             env_version="env-v1")


def ep(**kw):
    base = dict(**IDENT, scenario_id="s", category="seen", seen=True, seed=0, instruction="", success=False,
                stop_reason="failed")
    base.update(kw)
    return EpisodeResult(**base)


def test_failure_categories():
    assert categorize(ep(success=True, stop_reason="success")) == "success"
    assert categorize(ep(collision=True, correct_object=False)) == "collision"
    assert categorize(ep(correct_object=False)) == "wrong_object"
    assert categorize(ep(grasp_attempts=2, grasp_successes=0)) == "grasp_failed"
    assert categorize(ep(grasp_attempts=1, grasp_successes=1, dropped=True)) == "object_dropped"
    assert categorize(ep(timeout=True)) == "timeout"
    assert "Failed" in summary(ep(timeout=True))


def test_metrics_exclude_unmeasured():
    rs = [ep(success=True, stop_reason="success", correct_object=None, completion_time_s=10.0),
          ep(success=False, correct_object=False, seen=False)]
    m = compute_metrics(rs)
    assert m["task_success"]["successes"] == 1 and m["task_success"]["n"] == 2
    assert m["correct_object_rate"] == 0.0          # only the measured episode counts
    assert m["completion_time_s"]["median"] == 10.0
    assert m["unseen"]["n"] == 1


def _registered_model(reg, tmp, dataset):
    ingest(dataset, reg)
    run_qc(dataset, reg)
    create_dataset_version(reg)
    approve("ds-v1", reg)
    ckpt = tmp / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"type": "act"}))
    (ckpt / "model.safetensors").write_bytes(b"0")
    rr = tmp / "run_record.json"
    rr.write_text("{}")
    reg.add_model("m-v1", "ds-v1", "act", "trained", str(ckpt), str(rr), {})
    return ckpt


def test_end_to_end_with_dummy_adapter(workspace, dataset):
    tmp, reg = workspace
    _registered_model(reg, tmp, dataset)
    r = run_suite("m-v1", "configs/eval/test_suite_v1.yaml", DummySimAdapter(), reg)
    results = load_results(r["results_dir"])
    assert len(results) == r["metrics"]["n_runs"] == 20
    assert r["metrics"]["fake_results"] is True

    gates = evaluate_gates(r["eval_id"], reg)
    real_gate = next(g for g in gates["gates"] if g["gate"] == "evaluation.real_results")
    assert real_gate["passed"] is False                  # fake results can never pass
    assert gates["recommendation"] == "do not deploy"

    card = build_card(r["eval_id"], reg)
    assert "FAKE RESULTS" in card and "95% CI" in card
    recs = recommend(r["eval_id"], reg)
    assert "data_recommendations" in recs
    assert Path(r["results_dir"], "model_card.md").exists()

    dep = deploy("m-v1", reg)
    assert dep["status"] == "rejected"                   # gates fail on fake results


def test_judge_agreement():
    judge = StateTraceJudge()
    truth = [ep(success=True, stop_reason="success"), ep(failure_stage="grasp")]
    judged = [judge.judge(t) for t in truth]
    assert agreement(judged, truth)["accuracy"] == 1.0
    assert judged[1].degraded_at == "grasp"


def test_smoke_test_files(tmp_path):
    assert smoke_test(str(tmp_path))["passed"] is False
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"0")
    assert smoke_test(str(tmp_path))["passed"] is True


def test_rollback(workspace):
    tmp, reg = workspace
    reg.add_deployment("d1", "m-v1", "deployed", {})
    reg.add_deployment("d2", "m-v2", "deployed", {})
    out = rollback(reg)
    assert out == {"dep_id": out["dep_id"], "from": "m-v2", "to": "m-v1"}
    assert reg.current_deployed_model() == "m-v1"
