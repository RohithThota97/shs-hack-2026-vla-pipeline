"""Best checkpoint of a curriculum run (training/best.py)."""
import json


def _run(tmp, reg, steps_vals, saved):
    out = tmp / "train" / "m-v1"
    for s in saved:
        d = out / "checkpoints" / f"{s:06d}" / "pretrained_model"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(b"0")
        (d / "config.json").write_text("{}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "curriculum_log.json").write_text(json.dumps({"val": [
        {"step": s, "stage": "s", "end_of_stage": False, "h/full": {"arm_rad": v, "l1_norm": 1}, "h/sub": {"arm_rad": v + 1}}
        for s, v in steps_vals]}))
    reg.add_model("m-v1", "ds-v1", "smolvla", "trained", str(out / "checkpoints" / "last" / "pretrained_model"), None,
                  {"output_dir": str(out), "trainer": "curriculum"})
    return out


def test_best_picks_lowest_saved_val(workspace):
    from hackpipe.training import best
    tmp, reg = workspace
    out = _run(tmp, reg, [(500, 0.09), (1000, 0.02), (1500, 0.03), (2000, 0.05)], saved=[500, 1500, 2000])
    m = reg.get_model("m-v1")
    c = best.candidates(m)
    assert [r["step"] for r in c] == [500, 1000, 1500, 2000] and c[1]["checkpoint"] is None     # 1000 deleted
    b = best.best(m)
    assert b["step"] == 1500 and b["val_full_arm_rad"] == 0.03 and c[1]["val_sub_arm_rad"] == 1.02
    r = best.select("m-v1", reg)
    m = reg.get_model("m-v1")
    assert r["step"] == 1500 and m["checkpoint_dir"].endswith("checkpoints/001500/pretrained_model")
    assert m["meta"]["best_step"] == 1500 and m["meta"]["last_checkpoint"].endswith("last/pretrained_model")
    best.select("m-v1", reg)                                                  # again: last_checkpoint kept
    assert reg.get_model("m-v1")["meta"]["last_checkpoint"].endswith("last/pretrained_model")


def test_best_without_log(workspace):
    import pytest
    from hackpipe.training import best
    tmp, reg = workspace
    reg.add_model("m-v1", "ds-v1", "smolvla", "trained", str(tmp / "ck"), None, {"output_dir": str(tmp / "none")})
    assert best.candidates(reg.get_model("m-v1")) == [] and best.best(reg.get_model("m-v1")) is None
    with pytest.raises(RuntimeError, match="no validated checkpoint"):
        best.select("m-v1", reg)


def test_new_checkpoint_needs_a_new_test(workspace):
    from hackpipe.evaluation.offline import is_current, pending, result_path
    from hackpipe.gates.verdict import verdict
    from hackpipe.training import best
    tmp, reg = workspace
    out = _run(tmp, reg, [(500, 0.09), (1000, 0.02)], saved=[500, 1000])
    best.select("m-v1", reg)                                                   # → step 1000
    m = reg.get_model("m-v1")
    f = result_path("m-v1")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"model": "m-v1", "rows": [], "executed": {"arm_mean": 0.01},
                             "checkpoint": str(out / "checkpoints" / "000500" / "pretrained_model")}))
    assert not is_current(json.loads(f.read_text()), m)                        # tested 500, uses 1000
    v = verdict("m-v1", reg, write=False)
    assert v["verdict"] == "NOT TESTED" and "another checkpoint" in v["todo"][0]
    assert pending(reg) == ["m-v1"]


def test_pruner_keeps_the_best(tmp_path):
    from hackpipe.training.train import prune_checkpoints
    ck = tmp_path / "checkpoints"
    for s in (500, 1000, 1500, 2000):
        (ck / f"{s:06d}").mkdir(parents=True)
    (ck / "last").symlink_to("002000")
    removed = prune_checkpoints(tmp_path, keep=1, protect={"000500"})
    assert sorted(removed) == ["001000", "001500"] and (ck / "000500").exists()


def test_running_marker_from_another_machine(workspace):
    import os
    from hackpipe.evaluation.offline import is_running, running_marker
    f = running_marker("m-v9")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(f"{os.getpid()} some-other-host 2026-10-24T00:00:00Z")      # came with a state import
    assert not is_running("m-v9")
    import socket
    f.write_text(f"{os.getpid()} {socket.gethostname()} now")
    assert is_running("m-v9")


def test_pending_batch_continues_after_a_failure(workspace, monkeypatch, capsys):
    import pytest
    from hackpipe.cli import main
    tmp, reg = workspace
    monkeypatch.setattr("hackpipe.evaluation.offline.pending", lambda r, split="test": ["m-a", "m-b"])
    calls = []

    def fake(mv, *a, **k):
        calls.append(mv)
        if mv == "m-a":
            raise FileNotFoundError("no checkpoint")
        return {"file": "x", "predictions": 1, "seconds": 1, "rows": [], "executed": {"arm_mean": 0.1}}
    monkeypatch.setattr("hackpipe.evaluation.offline.offline_eval", fake)
    with pytest.raises(SystemExit, match="m-a"):
        main(["eval", "--model", "pending", "--offline"])
    assert calls == ["m-a", "m-b"]                                              # m-b still evaluated
