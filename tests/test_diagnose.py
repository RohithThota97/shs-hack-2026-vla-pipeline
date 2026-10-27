"""Diagnostic tools (hackpipe/diagnose.py) on synthetic probes, step logs and a dummy policy server, and the
trajectory baseline against a simulated robot that lags or gets blocked."""
import json
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from hackpipe import diagnose as dg


def _probe(n=4, horizon=50):
    rng = np.random.default_rng(0)
    S = rng.normal(0, 0.3, (n, 16)).astype(np.float32)
    A = S[:, None, :] + np.linspace(0, 0.2, horizon)[None, :, None]           # every joint drifts up from the state
    img = lambda: rng.integers(0, 255, (n, 224, 224, 3), dtype=np.uint8)  # noqa: E731
    return {"state": S, "actions": A.astype(np.float32), "ext": img(), "lw": img(), "rw": img(),
            "where": np.array([[0, 30 * i] for i in range(n)])}


def test_ablation_and_io_with_a_policy_that_follows_the_state(tmp_path):
    F = _probe()

    def predict(obs, task, seed):                       # "uses" the state only: images don't matter
        s = obs["observation.state"]
        steps = np.linspace(0, 0.2, 50)[np.minimum(np.arange(20) * 5, 49)]         # a 20-action chunk at 6 Hz
        return (s[None] + steps[:, None] + 0.001 * seed).astype(np.float32)

    r = dg.ablation(predict, F, n_actions=10, stride=5, seeds=3)
    e = r["error_rad"]
    assert e["normal (all inputs correct)"] < 0.01 and r["seed_spread_rad"] > 0
    assert e["all 3 cameras black"] == pytest.approx(e["normal (all inputs correct)"], abs=0.005)   # images unused
    assert e["correct images, state of a different moment"] > 5 * e["normal (all inputs correct)"]
    assert e["baseline: just hold the current pose"] > e["normal (all inputs correct)"]
    res = dg.io_example(predict, F, 1, tmp_path)
    assert (tmp_path / "inputs.png").exists() and json.loads((tmp_path / "io_example.json").read_text())["frame"] == 30
    assert len(res["output_chunk"]) == 20 and max(res["arm_error_per_step_rad"]) < 0.01


def test_score_variant_tells_absolute_from_delta():
    F = _probe()
    absolute = lambda obs, prompt: dg.probe_obs(F, 0)["observation.state"][None] * 0 + F["actions"][  # noqa: E731
        [i for i in range(len(F["state"])) if np.array_equal(F["state"][i], obs["observation.state"])][0]]
    r = dg.score_variant(absolute, F, 25)
    assert r["absolute"] < 1e-6 and r["as_delta"] > 0.1 and r["hold_pose"] > 0


def test_server_check_against_the_dummy_server():
    from hackpipe.deploy.server import DummyPolicy, make_handler
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(DummyPolicy(), {"policy_hz": 6.0}, None))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        F = _probe(2)
        r = dg.server_check(f"http://127.0.0.1:{srv.server_address[1]}", F, calls=3)
        assert r["server"]["policy_type"] == "dummy-hold-start-pose" and r["calls_per_frame"] == 3
        assert all(f["arm_error_rad"] == pytest.approx(f["hold_pose_rad"], abs=1e-4) for f in r["frames"])  # holds
    finally:
        srv.shutdown()


def test_grasp_timing_and_episode_report(tmp_path):
    T = 300
    D = np.zeros((T, 16))
    D[:, :7] = np.linspace(0, 1, T)[:, None]
    DA = D.copy()
    DA[:, 7] = np.where(np.arange(T) >= 150, 0.0, 1.0)                        # demo closes at 5 s
    ref = {"demo_states": D.tolist(), "demo_actions": DA.tolist(), "initial_state": D[0].tolist()}
    S, A = D.copy(), DA.copy()
    A[:, 7] = np.where(np.arange(T) >= 180, 0.0, 1.0)                         # the run closes 1 s late
    steps = tmp_path / "steps.jsonl"
    box = np.tile([0.0, 0.85, 0.0], (T, 1))
    tips = np.tile([[0.0, 0.95, 0.0], [0.0, 0.93, 0.0]], (T, 1, 1))
    steps.write_text("\n".join(json.dumps({"step": i, "state": S[i].tolist(), "action": A[i].tolist(),
                                           "box": box[i].tolist(), "left_tips": tips[i].tolist()}) for i in range(T)))
    run = dg.read_steps(steps)
    g = dg.grasp_timing(ref, run)
    assert g["closed"] and g["demo_close_s"] == 5.0 and g["vs_demo_s"] == pytest.approx(1.0)
    assert g["to_demo_same_time_rad"] == 0 and g["to_demo_grasp_pose_rad"] > 0 and g["closest_before_s"] == 5.0
    A[:, 7] = 1.0
    assert not dg.grasp_timing(ref, {"state": S, "action": A})["closed"]
    e = dg.episode_report(ref, run)
    assert e["mean_joint_distance_to_demo_rad"] == 0 and e["closest_tips_to_box_mm"] == 90
    assert [r["t_s"] for r in e["rows"]] == [0.0, 5.0] and e["rows"][1]["left_moved_rad"] == e["rows"][1]["demo_left_moved_rad"]


def test_trajectory_policy_waits_for_a_blocked_or_slow_robot():
    """The team's tools/test_trajectory_policy.py: robot = first-order lag towards the served target per 1/6 s call."""
    from hackpipe.deploy.server import TrajectoryPolicy
    n = 600                                                                    # 20 s at 30 fps
    S = np.zeros((n, 16), np.float32)
    S[:, :7] = np.linspace(0, 2, n)[:, None]
    A = np.vstack([S[1:], S[-1:]])
    p = TrajectoryPolicy(S, A, policy_hz=6)

    def run(gain, block=None, max_calls=2000):
        p.reset()
        s, log = S[0].copy(), []
        for k in range(max_calls):
            a = p.act({"observation.state": s}, "")
            if not (block and block[0] <= k / 6 < block[1]):
                s = s + gain * (a - s)
            log.append((k / 6, p.i))
            if p.i >= n - 1:
                break
        return np.array(log)

    good, slow = run(0.9), run(0.4)
    assert good[-1, 1] >= n - 1 and slow[-1, 1] >= n - 1 and slow[-1, 0] > good[-1, 0]   # slower, but gets there
    blocked = run(0.9, (8, 10))
    before = blocked[blocked[:, 0] < 8][-1, 1]
    during = blocked[(blocked[:, 0] >= 8) & (blocked[:, 0] < 10)][:, 1]
    assert during.max() - before <= 2 * p.step                                 # waited instead of running ahead
    p.reset()                                                                  # perfect tracking: served = recorded
    s, idx = S[0].copy(), []
    while p.i < n - 1 and len(idx) < 2000:
        a = p.act({"observation.state": s}, "")
        k = min(p.i + p.step - 1, n - 1)
        assert np.allclose(a, A[k])
        idx.append(p.i)
        s = S[min(k + 1, n - 1)]
    steps = np.diff(idx)
    assert (steps >= 0).all() and steps.max() <= p.step
