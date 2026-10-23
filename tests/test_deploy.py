"""Deploy: verdict gate, staged swap with health check, rollback, the policy server and its process manager."""
import io
import json
import socket
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

CAMS = ["observation.images.external_camera", "observation.images.left_wrist_camera",
        "observation.images.right_wrist_camera"]


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
        (ck / "model.safetensors").write_bytes(b"0")
        reg.add_model(v, "ds-v1", "smolvla", "trained", str(ck), None, {"output_dir": str(tmp / v)})
    return tmp, reg


def _offline(v, arm=0.01):
    from hackpipe.common.config import project_config, resolve
    d = resolve(project_config()["project"]["outputs_dir"]) / "runs" / v
    d.mkdir(parents=True, exist_ok=True)
    rob = {k: {"ratio": 1.0} for k in ("images_darker_x0.6", "images_brighter_x1.4", "state_noise_0.01rad",
                                         "reworded_instruction")}
    (d / "offline_eval_test.json").write_text(json.dumps({
        "model": v, "rows": [], "executed": {"arm_mean": arm, "arm_p99": arm, "per_subtask": []},
        "gripper_events": {"n": 2, "hit_rate": 1.0}, "robustness": {"variants": rob}}))


def _sim_pass(reg, tmp, v):
    reg.add_eval(f"ev-{v}", v, "sim-handover-v1", "isaac", str(tmp), {
        "task_success": {"rate": 0.9, "successes": 9, "n": 10, "ci95": [0.6, 1]},
        "stage_counts_0to4": [0, 0, 0, 1, 9], "replay": {"success": True}})


def test_deploy_follows_the_verdict(workspace, dataset):
    from hackpipe.deploy.deploy import deploy
    tmp, reg = _models(workspace, dataset)
    r = deploy("m-v1", reg)
    assert r["status"] == "rejected" and r["verdict"] == "NOT EVALUATED"
    _offline("m-v1")                                                   # offline passes, no sim → HOLD
    r = deploy("m-v1", reg)
    assert r["status"] == "rejected" and "supervision" in r["error"]
    r = deploy("m-v1", reg, allow_supervised=True, by="rohith")
    assert r["status"] == "deployed" and reg.current_deployed_model() == "m-v1"
    assert reg.deployments()[-1]["meta"]["supervised"] is True and reg.deployments()[-1]["meta"]["by"] == "rohith"
    _offline("m-v2", arm=0.5)                                          # MORE TRAINING: never, even supervised
    assert deploy("m-v2", reg, allow_supervised=True)["status"] == "rejected"


class FakeServing:
    """serving.* stand-in: records calls; production port 8090 starts out serving `initial`."""

    def __init__(self, initial=None, health_ok=True, prod_fails=False):
        self.calls, self.health_ok, self.prod_fails = [], health_ok, prod_fails
        self.ports = {8090: initial}

    def settings(self):
        return {"port": 8090, "stage_port": 8091, "n_action_steps": 5}

    def stop(self, port):
        self.calls.append(("stop", port))
        return self.ports.pop(port, None) is not None

    def start(self, port, model=None, n_action_steps=None, grasp_extend=0.0, **_):
        self.calls.append(("start", port, model, n_action_steps))
        if port == 8090 and self.prod_fails and model != (self.ports.get("prev") or "m-v0"):
            self.prod_fails = False
            raise RuntimeError("did not answer")
        self.ports[port] = {"model": model, "n_action_steps": n_action_steps}
        return {"pid": 1}

    def health_check(self, port):
        self.calls.append(("health", port))
        return {"ok": self.health_ok, "error": None if self.health_ok else "probe failed", "first_call_ms": 9}

    def status(self, port):
        s = self.ports.get(port)
        return {"live": s is not None, "ours": s is not None, "foreign": False, "state": s}


def test_swap_order_and_failures(workspace, dataset, monkeypatch):
    import hackpipe.deploy.serving as srv
    from hackpipe.deploy.deploy import deploy, rollback
    tmp, reg = _models(workspace, dataset)
    for v in ("m-v1", "m-v2"):
        _offline(v)
        _sim_pass(reg, tmp, v)

    fake = FakeServing()
    for k in ("settings", "stop", "start", "health_check", "status"):
        monkeypatch.setattr(srv, k, getattr(fake, k))
    r = deploy("m-v1", reg, serve=True)
    assert r["status"] == "deployed" and r["serving"]["n_action_steps"] == 5          # default from settings
    order = [c[:2] for c in fake.calls]
    assert order.index(("health", 8091)) < order.index(("stop", 8090)) < order.index(("start", 8090))

    fake.calls.clear()
    fake.health_ok = False                                             # broken model: production untouched
    r = deploy("m-v2", reg, serve=True)
    assert r["status"] == "rejected" and "health check failed" in r["error"]
    assert ("stop", 8090) not in [c[:2] for c in fake.calls] and fake.ports[8090]["model"] == "m-v1"
    assert reg.current_deployed_model() == "m-v1"

    fake.health_ok, fake.prod_fails = True, True                       # production start fails: previous back
    fake.ports["prev"] = "m-v1"
    r = deploy("m-v2", reg, serve=True, n_action_steps=3)
    assert r["status"] == "rejected" and fake.calls[-1][:3] == ("start", 8090, "m-v1")

    fake.ports.pop("prev")
    assert deploy("m-v2", reg, serve=True, n_action_steps=3)["status"] == "deployed"
    out = rollback(reg, serve=True)                                    # back to m-v1 with its own settings
    assert out["to"] == "m-v1" and fake.ports[8090] == {"model": "m-v1", "n_action_steps": 5}


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _npz(**arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def test_server_protocol_with_dummy_policy():
    from hackpipe.deploy.serving import http_json
    from hackpipe.deploy.server import DummyPolicy, make_handler
    port = _free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(DummyPolicy(), {"policy_hz": 6.0}, None))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{port}"
        info = http_json(base + "/info")
        assert info["fake"] is True and info["policy_hz"] == 6.0 and info["output_features"] == {"action": [16]}
        obs = {"observation.state": np.arange(16, dtype=np.float32),
               **{c: np.zeros((224, 224, 3), np.uint8) for c in CAMS}}
        assert http_json(base + "/reset", b"") == {"ok": True}
        a = http_json(base + "/act", _npz(task=np.array("go"), **obs))
        assert a["action"] == list(range(16)) and a["fake"] is True
        obs["observation.state"] = obs["observation.state"] + 1                       # holds the first pose
        assert http_json(base + "/act", _npz(**obs))["action"] == list(range(16))
        import urllib.error
        with pytest.raises(urllib.error.HTTPError) as e:
            http_json(base + "/act", _npz(**{"observation.state": obs["observation.state"]}))
        assert e.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as e:
            http_json(base + "/probe", b"")
        assert e.value.code == 400                                                     # no probe source
    finally:
        srv.shutdown()


def test_contract_and_grasp_deepener(monkeypatch):
    from hackpipe.deploy.server import GraspDeepener, contract_problems
    monkeypatch.setattr("hackpipe.deploy.server.rename_map", lambda: {                  # the repo's configs/dataset.yaml
        c: f"observation.images.camera{i}" for i, c in enumerate(CAMS, 1)})

    class P:
        fake, checkpoint, model, policy_type, n_action_steps, chunk_size = False, "c", "m", "smolvla", 5, 20
        input_features = {"observation.state": [16], **{f"observation.images.camera{i}": [3, 224, 224]
                                                         for i in (1, 2, 3)}}
        output_features = {"action": [16]}

        def __init__(self):
            self.i = 0

        def reset(self):
            pass

        def act(self, obs, task):
            self.i += 1
            a = np.full(16, 0.01 * self.i, np.float32)
            a[7] = 1.0 if self.i < 5 else 0.0                                         # closes the left gripper at 5
            a[15] = 1.0
            return a

    assert contract_problems(P()) == []                                                 # camera1..3 via rename_map
    bad = P()
    bad.output_features = {"action": [14]}
    assert any("action" in p for p in contract_problems(bad))
    g = GraspDeepener(P(), extend_s=0.5, hold_s=0.5, hz=6.0)
    acts = [g.act({}, "") for _ in range(12)]
    assert acts[4][7] == 1.0 and acts[6][7] == 1.0                    # kept open while approaching further
    assert acts[4][0] > acts[3][0]                                     # the arm keeps moving
    assert any(a[7] == 0.0 for a in acts[7:])                          # then closes


def test_serving_process_with_dummy(workspace):
    from hackpipe.deploy import serving
    port = _free_port()
    st = serving.start(port, dummy=True)
    try:
        assert st["info"]["fake"] is True and serving.status(port)["ours"]
        with pytest.raises(RuntimeError, match="already serving"):
            serving.start(port, dummy=True)
        hc = serving.health_check(port)
        assert hc["ok"] is False and "FAKE" in hc["error"]                       # a dummy is never deployed
    finally:
        assert serving.stop(port)
    assert not serving.status(port)["live"] and serving.stop(port) is False


def test_deploy_page(workspace, dataset, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("HACKPIPE_MACHINE", "laptop-test")
    tmp, reg = _models(workspace, dataset)
    _offline("m-v1")
    reg.add_deployment("dep-1", "m-v1", "deployed", {"verdict": "HOLD: SIM UNVERIFIED", "supervised": True, "by": "r"})
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/hackpipe/console/views/deploy.py"),
                           default_timeout=30)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    labels = {b.label: b.disabled for b in at.button}
    assert labels["Stop serving"] and labels["Roll back"]                  # nothing ours serving; one deployment
    deploy_btn = next(k for k in labels if k.startswith("Deploy m-v2"))
    assert labels[deploy_btn]                                              # m-v2 NOT EVALUATED
    at.selectbox[0].select_index(1).run()                                  # m-v1: HOLD
    at.checkbox[0].check().run()
    at.text_input[0].input("rohith").run()
    assert not at.exception
    assert not next(b for b in at.button if b.label.startswith("Deploy m-v1")).disabled
