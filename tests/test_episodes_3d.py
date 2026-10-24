"""3D trajectories of test episodes: kinematics, per-episode explanation, saved streams, per-checkpoint tests."""
import json

import numpy as np
import pytest

READY = [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4]


def test_fr3_kinematics_known_poses():
    from hackpipe.evaluation.kinematics import arm_paths, fk
    assert fk(np.zeros((1, 7)), tcp=False)[0] == pytest.approx([0.088, 0, 0.926], abs=1e-4)
    assert fk(np.array([READY]), tcp=False)[0] == pytest.approx([0.3069, 0, 0.5903], abs=1e-3)
    row = np.r_[READY, 1.0, READY, 1.0]
    p = arm_paths(row[None])
    assert p["left"][0] == pytest.approx(p["right"][0]) and p["left"].shape == (1, 3)


def _episode(n=60, drift_from=None, miss_grasp=False):
    gt = np.tile(np.r_[READY, 0.48, READY, 0.48], (n, 1)).astype(float)
    gt[:, 0] = np.linspace(0, 0.8, n)                               # left base joint sweeps
    gt[20:, 7] = 0.0                                                  # left grasp at step 20
    pred = gt.copy()
    if drift_from is not None:
        pred[drift_from:, 1] += 0.3                                   # left shoulder off by 0.3 rad
    if miss_grasp:
        pred[:, 7] = 0.48
    sub = np.array(["Pick" if i < 30 else "Place" for i in range(n)])
    return {"name": "h", "ep": 3, "pred": pred, "gt": gt, "still": gt, "sub": sub, "replan": np.zeros(n, bool)}


def test_episode_explanations():
    from hackpipe.evaluation.episodes import summarize
    ok = summarize(_episode())
    assert ok["follows_demo"] and ok["max_cm"] == 0 and ok["why"].startswith("Follows the demo")
    bad = summarize(_episode(drift_from=40))
    assert not bad["follows_demo"] and bad["drift_at_s"] == pytest.approx(40 / 6, abs=0.1)
    assert bad["drift_subtask"] == "Place" and "drifts off the demo" in bad["why"] and bad["max_cm"] > 5
    miss = summarize(_episode(miss_grasp=True))
    assert miss["missed_gripper"][0]["kind"] == "close" and "left gripper grasp (close)" in miss["why"]


def test_gripper_events_are_grasps_and_releases():
    from hackpipe.evaluation.offline import gripper_events
    a = np.zeros((10, 16))
    a[:, 7] = [1.0, 0.48, 0.48, 0.0, 0.0, 0.0, 0.48, 0.48, 0.48, 0.48]   # pre-shape, grasp, release
    assert gripper_events(a, 7) == [(3, "close"), (6, "open")]


def test_saved_trajectories_and_step_results(workspace):
    from hackpipe.evaluation.offline import (load_result, result_for, result_path, save_trajectories, step_of,
                                             trajectories)
    tmp, reg = workspace
    out = tmp / "train" / "m-v1"
    ck = out / "checkpoints" / "001000" / "pretrained_model"
    ck.mkdir(parents=True)
    reg.add_model("m-v1", "ds-v1", "smolvla", "training", str(ck), None, {"output_dir": str(out)})
    assert step_of(ck) == 1000 and step_of(out / "checkpoints" / "last" / "pretrained_model") is None
    f = result_path("m-v1", step=1000)
    f.parent.mkdir(parents=True, exist_ok=True)
    ep = _episode()
    save_trajectories(f.with_suffix(".npz"), [ep])
    f.write_text(json.dumps({"model": "m-v1", "rows": [], "metrics_version": 2, "executed": {"arm_mean": 0.01}, "checkpoint": str(ck),
                             "file": str(f)}))
    r = result_for(reg.get_model("m-v1"))                        # no main result: the step test of the checkpoint
    assert r["checkpoint"] == str(ck) and load_result("m-v1") is None
    t = trajectories(r)
    assert len(t) == 1 and t[0]["ep"] == 3 and np.allclose(t[0]["gt"], ep["gt"]) and list(t[0]["sub"][:2]) == ["Pick"] * 2
    assert trajectories({"rows": []}) == []


def test_review_fixes_29(workspace, dataset, monkeypatch):
    import os
    from hackpipe.evaluation.offline import is_current, result_path, save_trajectories, trajectories
    from hackpipe.training.train import prune_checkpoints
    tmp, reg = workspace
    out = tmp / "train" / "m-v1"
    cks = out / "checkpoints"
    for s in (1000, 2000, 3000):
        (cks / f"{s:06d}" / "pretrained_model").mkdir(parents=True)
    (cks / "last").symlink_to("002000")
    last = cks / "last" / "pretrained_model"
    reg.add_model("m-v1", "ds-v1", "smolvla", "training", str(last), None, {"output_dir": str(out)})
    res = {"model": "m-v1", "metrics_version": 2, "executed": {"arm_mean": 0.01},
           "checkpoint": str(last.resolve())}                                       # stored resolved (step 2000)
    assert is_current(res, reg.get_model("m-v1"))
    (cks / "last").unlink()
    (cks / "last").symlink_to("003000")                                              # training moved on
    assert not is_current(res, reg.get_model("m-v1"))
    assert not is_current({**res, "metrics_version": 1, "checkpoint": str(last.resolve())}, reg.get_model("m-v1"))

    calls = []                                                                       # protect re-read every round
    prune_checkpoints(out, keep=1, protect=lambda: calls.append(1) or {"001000"})
    assert calls and (cks / "001000").exists() and not (cks / "002000").exists()

    f = result_path("m-v1")                                                          # after a state import
    f.parent.mkdir(parents=True, exist_ok=True)
    save_trajectories(f.with_suffix(".npz"), [_episode()])
    assert len(trajectories({"model": "m-v1", "file": "/other/machine/offline_eval_test.json"})) == 1


def test_deploy_a_training_model(workspace, dataset):
    from hackpipe.dataset.versioning import approve, create_dataset_version
    from hackpipe.deploy.deploy import deploy
    from hackpipe.evaluation.offline import result_path
    from hackpipe.ingest.importer import ingest
    from hackpipe.quality.report import run_qc
    tmp, reg = workspace
    ingest(dataset, reg)
    run_qc(dataset, reg)
    create_dataset_version(reg)
    approve("ds-v1", reg, by="t")
    out = tmp / "train" / "m-v1"
    ck = out / "checkpoints" / "002000" / "pretrained_model"
    ck.mkdir(parents=True)
    for f in ("config.json", "model.safetensors"):
        (ck / f).write_text("{}")
    (out / "checkpoints" / "last").symlink_to("002000")
    reg.add_model("m-v1", "ds-v1", "smolvla", "training", str(out / "checkpoints" / "last" / "pretrained_model"),
                  None, {"output_dir": str(out)})
    good = {"model": "m-v1", "metrics_version": 2, "rows": [{"instruction": "full", "subtask": "h · all",
            "hold_still_arm_rad": 0.1, "chunk_arm_rad": 0.01}], "executed": {"arm_mean": 0.01, "hold_still_arm_mean": 0.1,
            "per_subtask": []}, "gripper_events": {"n": 0}, "robustness": {"variants": {k: {"ratio": 1.0} for k in (
            "images_darker_x0.6", "images_brighter_x1.4", "state_noise_0.01rad", "reworded_instruction")}}}
    f = result_path("m-v1")
    f.parent.mkdir(parents=True, exist_ok=True)
    import json
    f.write_text(json.dumps({**good, "checkpoint": str(ck.resolve())}))
    r = deploy("m-v1", reg, by="r")                                       # still `last`: refused
    assert r["status"] == "rejected" and "pick a saved checkpoint" in r["error"]
    with reg.conn:
        reg.conn.execute("UPDATE models SET checkpoint_dir=? WHERE version='m-v1'", (str(ck),))
    r = deploy("m-v1", reg, by="r")
    assert r["status"] == "serving" and reg.deployments()[-1]["meta"]["purpose"] == "sim test"
    assert reg.get_model("m-v1")["status"] == "training"                  # the trainer still owns the status
