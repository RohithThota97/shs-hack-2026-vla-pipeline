"""Offline metrics after the team's eval_offline.py, the post-training verdict and the Evaluate page."""
import json
from pathlib import Path

import numpy as np
import pytest



def _table(n=60, stride=1):                  # same frame table as test_state_offline._table
    S = np.cumsum(np.full((n, 16), 0.01, np.float32), 0)
    A = np.roll(S, -1, 0)
    sub = np.array(["Pick" if i < n // 2 else "Place" for i in range(n)], object)
    return {"name": "h", "stride": stride, "S": S, "A": A, "rows": np.arange(0, n, 2),
            "sub_end": np.where(np.arange(n) < n // 2, n // 2, n), "ep_end": np.full(n, n),
            "sub_text": sub, "full_text": np.full(n, "Do it", object), "ep": np.zeros(n, int)}


def _oracle(tab, rows, mode, variant=None):
    from hackpipe.training.curriculum import chunk_for
    return np.stack([tab["A"][chunk_for(tab, r, 1 if mode == "full" else 0, 5, "mask")[0]] for r in rows])


def _frozen(tab, rows, mode, variant=None):          # never moves: keeps the first frame's pose and grippers
    return np.repeat(tab["S"][0][None, None], len(rows), 0).repeat(5, 1)


def test_stream_metrics_oracle_vs_hold_still():
    from hackpipe.evaluation.offline import ROBUSTNESS, score
    t = _table(120)
    t["S"][:, 7] = t["S"][:, 15] = np.where(np.arange(120) < 60, 1.0, 0.0)   # one gripper change per arm
    t["A"] = np.roll(t["S"], -1, 0)
    res = score([t], _oracle, chunk=5, every=1, batch=8, robust=ROBUSTNESS, robust_eps=1)
    ex = res["executed"]
    assert ex["arm_mean"] == 0 and ex["arm_p99"] == 0 and {s["dtw_rad"] for s in ex["per_subtask"]} == {0.0}
    ge = res["gripper_events"]
    assert ge["n"] == 2 and ge["hit_rate"] == 1.0 and ge["mean_abs_timing_error_s"] == 0
    assert set(res["horizon_arm_rad"]) == {0.0} and len(res["horizon_arm_rad"]) == 5
    assert set(res["robustness"]["variants"]) == set(ROBUSTNESS)
    still = score([t], _frozen, chunk=5, every=1)
    assert still["executed"]["arm_mean"] > 0 and still["gripper_events"]["hit_rate"] == 0.0
    assert "robustness" not in still
    with pytest.raises(ValueError, match="gaps"):
        score([t], _oracle, chunk=5, every=6)


def test_dtw():
    from hackpipe.evaluation.offline import dtw
    a = np.linspace(0, 1, 30)[:, None].repeat(3, 1)
    assert dtw(a, a) == 0
    assert dtw(a, np.roll(a, 2, 0)) < np.abs(a - np.roll(a, 2, 0)).mean()      # tolerant to a time shift


def _models(workspace, dataset):
    from hackpipe.dataset.versioning import approve, create_dataset_version
    from hackpipe.ingest.importer import ingest
    from hackpipe.quality.report import run_qc
    tmp, reg = workspace
    ingest(dataset, reg)
    run_qc(dataset, reg)
    create_dataset_version(reg)
    approve("ds-v1", reg, by="t")
    for v in ("m-v1", "m-v2"):
        ck = tmp / "ck" / v
        ck.mkdir(parents=True)
        (ck / "config.json").write_text("{}")
        reg.add_model(v, "ds-v1", "smolvla", "trained", str(ck), None, {"output_dir": str(tmp / v)})
    return tmp, reg


def _offline(v, arm=0.01, hit=1.0, rob=1.1, rows=None):
    from hackpipe.common.config import project_config, resolve
    d = resolve(project_config()["project"]["outputs_dir"]) / "runs" / v
    d.mkdir(parents=True, exist_ok=True)
    (d / "offline_eval_test.json").write_text(json.dumps({
        "model": v, "rows": rows or [{"instruction": "full", "subtask": "h · all", "n": 3, "first_arm_rad": arm,
                                      "chunk_arm_rad": arm, "gripper": 0.1, "hold_still_arm_rad": 0.05}],
        "executed": {"arm_mean": arm, "arm_p99": arm * 3, "per_subtask": [{"subtask": "h · Pick", "arm_rad": arm}]},
        "gripper_events": {"n": 4, "hit_rate": hit, "mean_abs_timing_error_s": 0.1, "events": []},
        "robustness": {"episodes_per_source": 3, "baseline_chunk_arm_rad": arm,
                       "variants": {k: {"ratio": rob} for k in ("images_darker_x0.6", "images_brighter_x1.4",
                                                                 "state_noise_0.01rad", "reworded_instruction")}},
        "latency_ms": {"replan_median": 90.0}, "smoothness": {"at_replan_p95": 0.01, "demo_p95": 0.01}}))


def _sim(reg, tmp, v, rate, replay_ok=True):
    reg.add_eval(f"ev-{v}-{rate}-{replay_ok}-{len(reg.evals())}", v, "sim-handover-v1", "isaac", str(tmp), {
        "task_success": {"rate": rate, "successes": int(rate * 10), "n": 10, "ci95": [0, 1]},
        "stage_counts_0to4": [0, 0, 0, 10 - int(rate * 10), int(rate * 10)],
        "replay": {"success": replay_ok}})


def test_verdicts(workspace, dataset):
    from hackpipe.evaluation.offline import pending
    from hackpipe.gates.verdict import verdict
    tmp, reg = _models(workspace, dataset)
    assert verdict("m-v1", reg)["verdict"] == "NOT TESTED" and pending(reg) == ["m-v1", "m-v2"]
    _offline("m-v1")
    assert pending(reg) == ["m-v2"]
    v = verdict("m-v1", reg)
    assert v["verdict"] == "READY FOR SIM TEST" and v["online_state"] == "skipped"
    _sim(reg, tmp, "m-v1", 0.9, replay_ok=False)
    assert verdict("m-v1", reg)["online_state"] == "unverified"         # demo replay failed: scene, not model
    _sim(reg, tmp, "m-v1", 0.9)
    v = verdict("m-v1", reg)
    assert v["verdict"] == "READY TO DEPLOY" and Path(v["offline_file"]).name == "offline_eval_test.json"
    _sim(reg, tmp, "m-v1", 0.5)
    assert verdict("m-v1", reg)["verdict"] == "NEEDS MORE TRAINING"            # verified sim fails
    _offline("m-v2", arm=0.03, hit=0.5, rob=2.0)
    v = verdict("m-v2", reg)
    failed = {r["gate"].split(":")[0] for r in v["offline_gates"] if not r["pass"]}
    assert v["verdict"] == "NEEDS MORE TRAINING" and {"arm error ÷ doing nothing", "robustness",
                                                      "gripper close / open on time (±1 s)"} <= failed
    assert any("image_aug" in t for t in v["todo"])
    reg.add_deployment("dep-1", "m-v1", "deployed", {})                   # regression vs the deployed model
    assert any(r["gate"].startswith("vs served model m-v1") and not r["pass"]
               for r in verdict("m-v2", reg)["offline_gates"])


def test_then_eval(workspace, dataset, monkeypatch):
    from hackpipe.cli import main
    tmp, reg = _models(workspace, dataset)
    calls = []
    monkeypatch.setattr("hackpipe.training.train.train", lambda *a, **k: {"model_version": "m-v3", "exit_code": 0})
    monkeypatch.setattr("hackpipe.evaluation.offline.offline_eval",
                        lambda mv, r, *a, **k: calls.append(mv) or {"file": "x", "executed": {"arm_mean": 0.1}})
    main(["train", "--config", "configs/train/smolvla_fr3_duo.yaml", "--dataset", "ds-v1", "--then-eval"])
    main(["train", "--config", "configs/train/smolvla_fr3_duo.yaml", "--dataset", "ds-v1"])
    assert calls == ["m-v3"]


def test_evaluate_page(workspace, dataset, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("HACKPIPE_MACHINE", "laptop-test")
    tmp, reg = _models(workspace, dataset)
    _offline("m-v1")
    _sim(reg, tmp, "m-v1", 0.9)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/hackpipe/console/views/evaluate.py"),
                           default_timeout=30)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    at.selectbox[0].select_index(1).run()                                     # newest first: m-v2, m-v1
    assert not at.exception, [e.message for e in at.exception]
    heads = [m.value for m in at.markdown]
    assert {"#### 1 · Best checkpoint", "#### 2 · Test on lighting10 (held out)", "#### 3 · Checks and verdict",
            "#### 4 · Isaac Sim results (site PC)"} <= set(heads)
    assert any("READY TO DEPLOY" in h for h in heads) and any("of doing nothing" in h for h in heads)
    assert {m.label for m in at.metric} >= {"Arm error", "Doing nothing", "Gripper on time", "One prediction"}

def _dtw_ref(a, b):
    n, m = len(a), len(b)
    c = np.abs(a[:, None] - b[None]).mean(-1)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0
    st = np.zeros((n + 1, m + 1), int)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            p = min((acc[i - 1, j - 1], st[i - 1, j - 1]), (acc[i - 1, j], st[i - 1, j]), (acc[i, j - 1], st[i, j - 1]))
            acc[i, j], st[i, j] = c[i - 1, j - 1] + p[0], p[1] + 1
    return acc[n, m] / st[n, m]


def test_dtw_vectorised_matches_reference():
    from hackpipe.evaluation.offline import dtw
    rng = np.random.default_rng(1)
    for n, m in ((7, 11), (13, 5), (1, 4)):
        a, b = rng.normal(size=(n, 3)), rng.normal(size=(m, 3))
        assert dtw(a, b) == pytest.approx(_dtw_ref(a, b))


def test_review_fixes(workspace, dataset, monkeypatch):
    import os
    from hackpipe.cli import main
    from hackpipe.common.config import project_config, resolve
    from hackpipe.evaluation.offline import pending, running_marker
    from hackpipe.gates.verdict import verdict
    tmp, reg = _models(workspace, dataset)
    runs = resolve(project_config()["project"]["outputs_dir"]) / "runs"
    (runs / "m-v1").mkdir(parents=True)
    (runs / "m-v1" / "offline_eval_test.json").write_text(json.dumps({"rows": []}))   # made before these metrics
    v = verdict("m-v1", reg)
    assert v["verdict"] == "NOT TESTED" and "predates" in v["todo"][0] and "m-v1" in pending(reg)
    running_marker("m-v2").parent.mkdir(parents=True, exist_ok=True)
    running_marker("m-v2").write_text(f"{os.getpid()} now")                           # an eval is running
    assert pending(reg) == ["m-v1"]
    running_marker("m-v2").write_text("999999999 old")                                # stale marker: dead pid
    assert "m-v2" in pending(reg)
    _offline("m-v1")
    off = json.loads((runs / "m-v1" / "offline_eval_test.json").read_text())
    off["robustness"] = {}                                                            # --no-robustness
    v = verdict("m-v1", reg, off=off)
    assert v["verdict"] == "NOT TESTED" and "not measured" in v["todo"][0]
    off["executed"]["arm_p99"] = float("nan")                                         # non-finite: no crash
    assert verdict("m-v1", reg, off=off)["verdict"] == "NOT TESTED"
    off = json.loads((runs / "m-v1" / "offline_eval_test.json").read_text())
    off["robustness"]["skipped"] = {"reworded_instruction": "no paraphrase matches"}
    del off["robustness"]["variants"]["reworded_instruction"]
    v = verdict("m-v1", reg, off=off)
    row = next(r for r in v["offline_gates"] if "reworded" in r["gate"])
    assert v["verdict"] == "READY FOR SIM TEST" and row["value"] == "not applicable" and not row["blocking"]
    reg.set_model_status("m-v2", "rolled_back")
    assert "m-v2" in pending(reg)
    monkeypatch.setattr("hackpipe.training.train.train", lambda *a, **k: {"model_version": "m-v2", "exit_code": 0})
    monkeypatch.setattr("hackpipe.evaluation.offline.offline_eval",
                        lambda *a, **k: (_ for _ in ()).throw(KeyError("camera")))
    main(["train", "--config", "configs/train/smolvla_fr3_duo.yaml", "--dataset", "ds-v1", "--then-eval"])  # no raise
