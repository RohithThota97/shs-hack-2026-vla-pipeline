"""Box grid player (inverse kinematics on the FR3) and the improve loop."""
import json
from pathlib import Path

import numpy as np
import pytest

READY = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4])


def _demo(n=300):
    """A handover-like demo at 30 fps: left reaches and grasps at 100, right grasps at 220."""
    A = np.tile(np.r_[READY, 0.48, READY, 0.48], (n, 1)).astype(float)
    A[:, 1] += np.linspace(0, 0.3, n)                      # left shoulder moves the whole time
    A[100:, 7] = 0.0
    A[220:, 15] = 0.0
    return A.astype(np.float32)


def test_ik_reaches_and_keeps_the_posture():
    from hackpipe.evaluation.kinematics import fk_T, ik
    T = fk_T(READY[None])[0]
    T[:3, 3] += [0.04, -0.03, 0.02]
    q, pe, re = ik(T, READY, q_ref=READY)
    assert pe < 5e-4 and re < 5e-3
    assert np.abs(q - READY).max() < 0.3                     # stays near the reference posture
    far = fk_T(READY[None])[0]
    far[:3, 3] += [2.0, 0, 0]                                 # 2 m away: out of reach
    assert ik(far, READY)[1] > 0.5


def test_grid_retarget_and_plan():
    from hackpipe.evaluation.kinematics import fk
    from hackpipe.improve.grid import grid_offsets, phases, plan, retarget, weight
    A = _demo()
    ph = phases(A, {"approach_end_s": 0.3, "lift_hold_s": 1.0, "return_before_handover_s": 1.0})
    assert ph["grasp"] == 100 and ph["handover"] == 220 and ph["approach_end"] < 100 < ph["lift_end"] < ph["return_end"]
    w = weight(len(A), ph)
    assert w[0] == 0 and w[ph["approach_end"]] == 1 and w[ph["return_end"]] == 0 and np.all(np.diff(w[:ph["approach_end"]]) >= 0)
    cfg = {"lift_hold_s": 1.0}
    r = retarget(A, np.array([0.05, 0.02, 0.0]), cfg)
    assert r["ok"], r["reason"]
    g = ph["grasp"]
    assert fk(r["actions"][g:g + 1, :7])[0] - fk(A[g:g + 1, :7])[0] == pytest.approx([0.05, 0.02, 0.0], abs=1e-3)
    assert np.allclose(r["actions"][ph["return_end"]:], A[ph["return_end"]:])        # handover + place unchanged
    assert np.allclose(r["actions"][:, 8:], A[:, 8:])                                # right arm never changed
    bad = retarget(A, np.array([1.0, 0.0, 0.0]), cfg)
    assert not bad["ok"] and "not reachable" in bad["reason"]
    assert len(grid_offsets()) == 136
    cells = plan(A, [(0.0, 0.0), (0.03, -0.02)], cfg=cfg)
    assert [c["ok"] for c in cells] == [True, True] and cells[1]["dx"] == 0.03


def test_loop_status_and_auto_approve(workspace, monkeypatch):
    from hackpipe.improve import loop
    tmp, reg = workspace
    rows = loop.status(reg)
    assert [r["step"] for r in rows][:3] == ["plan", "reference", "render"] and not any(r["done"] for r in rows)
    assert "waiting for" in rows[2]["detail"]
    cfg = loop.improve_config()
    demos = [{"demo_id": "d1", "source": f"data/incoming/{cfg['grid']['source_name']}", "status": "VALID"},
             {"demo_id": "d2", "source": "data/raw/other", "status": "VALID"}]
    monkeypatch.setattr(reg, "demos", lambda **k: demos)
    approved = []
    monkeypatch.setattr("hackpipe.dataset.versioning.approve", lambda v, r, by="": approved.append((v, by)))
    ver = {"version": "ds-v9", "status": "draft", "manifest": {"lineage": {"added": ["d1"]}}}
    monkeypatch.setattr(loop, "_version_with_source", lambda r, s: ver)
    log = []
    assert loop.step_approve(reg, cfg, log.append) and approved == [("ds-v9", "auto: sim-generated (configs/improve.yaml)")]
    ver["manifest"]["lineage"]["added"] = ["d1", "d2"]                 # a non-sim demo: a human decides
    assert not loop.step_approve(reg, cfg, log.append) and "human approval" in log[-1]


def test_loop_run_stops_at_render(workspace, monkeypatch):
    from hackpipe.improve import loop
    tmp, reg = workspace
    done = []
    monkeypatch.setattr(loop, "step_plan", lambda r, c, log: (done.append("plan"), loop._save(plan={"cells": 1, "valid": 1, "grid": [], "reference": "x", "max_ik_mm": 0})))
    ref = tmp / "ref.json"
    ref.write_text("{}")
    monkeypatch.setattr(loop, "step_reference", lambda r, c, log: (done.append("reference"), loop._save(reference=str(ref))))
    msgs = []
    rows = loop.run(reg, log=msgs.append)
    assert done == ["plan", "reference"] and rows[0]["done"] and rows[1]["done"] and not rows[2]["done"]
    assert msgs[-1].startswith("render: waiting for the site PC") and "--mode grid" in msgs[-1]


def test_improve_page(workspace, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from hackpipe.improve import loop
    monkeypatch.setenv("HACKPIPE_MACHINE", "laptop-test")
    loop._save(plan={"cells": 2, "valid": 1, "reference": "demo", "max_ik_mm": 0.1,
                     "grid": [{"dx": 0, "dy": 0, "ok": True, "reason": ""}, {"dx": 0.08, "dy": 0, "ok": False,
                                                                           "reason": "too close to a joint limit"}]})
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/hackpipe/console/views/improve.py"),
                           default_timeout=30)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("1 · Grid of box positions" in m.value for m in at.markdown)
    assert any("--mode grid" in c.value for c in at.code)
