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
