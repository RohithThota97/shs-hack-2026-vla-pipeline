"""R10 — Deploy a model only after its post-training verdict allows it and a smoke test passes; serve it for the
Isaac Sim bridge with a staged health check (deploy/serving.py); one-command rollback.

The actual launch (simulator or robot) is filled in by Person 3 in `LAUNCH_COMMAND` once the organisers'
interface is known. LeRobot options: `lerobot-rollout` on the robot computer, or `lerobot-policy-server`
for inference on a GPU server (see LeRobot docs: inference, remote_inference).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hackpipe.common.ids import now_iso
from hackpipe.common.registry import Registry

REQUIRED_FILES = ["config.json", "model.safetensors"]   # LeRobot pretrained_model folder
LAUNCH_COMMAND = "TODO (Person 3): command that starts the policy in the organisers' simulator"


def smoke_test(checkpoint_dir: str | None, load: bool = False) -> dict[str, Any]:
    checks = []
    if not checkpoint_dir:
        return {"passed": False, "checks": [{"check": "checkpoint_dir", "passed": False, "detail": "missing"}]}
    p = Path(checkpoint_dir)
    for f in REQUIRED_FILES:
        checks.append({"check": f"file:{f}", "passed": (p / f).exists(), "detail": str(p / f)})
    if load and all(c["passed"] for c in checks):
        try:
            from lerobot.policies.factory import get_policy_class  # type: ignore

            policy_type = json.loads((p / "config.json").read_text()).get("type")
            policy = get_policy_class(policy_type).from_pretrained(str(p))
            n_params = sum(t.numel() for t in policy.parameters())
            checks.append({"check": "load_policy", "passed": True, "detail": f"{policy_type}, {n_params:,} parameters"})
            # TODO (Person 3): run one observation from a recorded episode and check action shape/range/latency.
        except Exception as e:  # noqa: BLE001
            checks.append({"check": "load_policy", "passed": False, "detail": f"{type(e).__name__}: {e}"[:300]})
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def _dep_id(model_version: str, kind: str = "") -> str:
    import secrets          # two deploys within one second must not collide
    return f"dep-{model_version}-{kind}{now_iso().replace(':', '').replace('-', '')}-{secrets.token_hex(2)}"


def _swap(model_version: str, n_action_steps: int | None, grasp_extend: float) -> dict[str, Any]:
    """Stage the model on stage_port, health-check it on a recorded frame, and only then replace what serves the
    production port. If staging fails, the old server keeps serving; if the new production server fails to start,
    the previous model is started again."""
    from hackpipe.deploy import serving
    cfg = serving.settings()
    port, stage = int(cfg["port"]), int(cfg["stage_port"])
    serving.stop(stage)
    try:
        serving.start(stage, model=model_version, n_action_steps=n_action_steps, grasp_extend=grasp_extend)
        health = serving.health_check(stage)
    finally:
        serving.stop(stage)
    if not health["ok"]:
        raise RuntimeError(f"health check failed: {health.get('error')}")
    before = serving.status(port)
    if before["foreign"]:
        raise RuntimeError(f"port {port} is held by a server hackpipe did not start: stop it first")
    previous = before["state"]
    serving.stop(port)
    try:
        started = serving.start(port, model=model_version, n_action_steps=n_action_steps, grasp_extend=grasp_extend)
    except Exception as e:
        if previous and previous.get("model"):
            try:
                serving.start(port, model=previous["model"], n_action_steps=previous.get("n_action_steps"),
                              grasp_extend=previous.get("grasp_extend") or 0.0)
            except Exception as e2:  # noqa: BLE001
                raise RuntimeError(f"{e} — and the previous model {previous['model']} could not be restarted "
                                   f"({e2}): port {port} serves nothing") from e
            raise RuntimeError(f"{e} — the previous model {previous['model']} serves again") from e
        raise
    return {"health_check": health, "port": port, "pid": started["pid"], "n_action_steps": n_action_steps,
            "grasp_extend": grasp_extend, "replaced": (previous or {}).get("model")}


def deploy(model_version: str, registry: Registry | None = None, allow_supervised: bool = False,
           serve: bool = False, n_action_steps: int | None = None, grasp_extend: float = 0.0,
           by: str = "") -> dict[str, Any]:
    """Gated by the post-training verdict (gates/verdict.py): READY TO DEPLOY → a deployment; READY FOR SIM TEST →
    served for the Isaac Sim test; anything else only with allow_supervised (= override, recorded with the name).
    serve: also stage + health-check + serve it on the production port (deploy/serving.py)."""
    from hackpipe.gates.verdict import verdict
    reg = registry or Registry()
    model = reg.get_model(model_version)
    if model is None:
        raise KeyError(model_version)
    v = verdict(model_version, reg)
    smoke = smoke_test(model["checkpoint_dir"])
    from hackpipe.evaluation.offline import step_of
    training = model["status"] == "training"
    ok_verdict = v["verdict"] in ("READY TO DEPLOY", "READY FOR SIM TEST")
    allowed = ok_verdict or allow_supervised
    dep_id = _dep_id(model_version)
    # a model still training is only served for a sim test: the trainer sets its status when it ends
    purpose = "deployment" if v["verdict"] == "READY TO DEPLOY" and not training else "sim test"
    meta: dict[str, Any] = {"verdict": v["verdict"], "todo": v["todo"], "smoke_test": smoke, "purpose": purpose,
                            "supervised": bool(allow_supervised and not ok_verdict), "by": by,
                            "launch": "hackpipe serve" if serve else LAUNCH_COMMAND}
    error = None
    if not allowed:
        error = f"verdict {v['verdict']}: serve it only with an override (--override, recorded)"
    elif allow_supervised and not ok_verdict and not by:
        error = "an override needs the name of who decides it (--by)"
    elif training and step_of(model["checkpoint_dir"]) is None:
        error = "the model is still training: pick a saved checkpoint first (Evaluate → Use this checkpoint)"
    elif not smoke["passed"]:
        error = "smoke test failed (checkpoint files missing)"
    elif serve:
        if n_action_steps is None:
            from hackpipe.deploy.serving import settings
            n_action_steps = settings().get("n_action_steps")
        try:
            meta["serving"] = _swap(model_version, n_action_steps, grasp_extend)
        except Exception as e:  # noqa: BLE001 — any failure is recorded as a rejected deployment
            error = f"{type(e).__name__}: {e}" if not isinstance(e, RuntimeError) else str(e)
    # only READY TO DEPLOY counts as a deployment (current model, regression reference, rollback target); a sim test
    # or an override is recorded as "serving"
    status = "rejected" if error else ("deployed" if purpose == "deployment" and not meta["supervised"] else "serving")
    if error:
        meta["error"] = error
    reg.add_deployment(dep_id, model_version, status, meta)
    if status == "deployed":
        reg.set_model_status(model_version, "deployed")
    return {"dep_id": dep_id, "status": status, "verdict": v["verdict"], "smoke_test": smoke, "error": error,
            "serving": meta.get("serving"), "launch": meta["launch"]}


def rollback(registry: Registry | None = None, serve: bool = False, by: str = "") -> dict[str, Any]:
    """Re-deploy the previously deployed model (no new gates: it passed them when it was deployed); serve: with
    its recorded serving settings, staged and health-checked like any deploy."""
    reg = registry or Registry()
    deployed = [d for d in reg.deployments() if d["status"] == "deployed"]
    if len(deployed) < 2:
        raise RuntimeError("No previous deployed model to roll back to.")
    current, previous = deployed[-1]["model_version"], deployed[-2]["model_version"]
    meta: dict[str, Any] = {"rollback_from": current, "by": by, "launch": "hackpipe serve" if serve else LAUNCH_COMMAND}
    if serve:
        from hackpipe.deploy.serving import settings
        prev_srv = (deployed[-2]["meta"] or {}).get("serving") or {}
        meta["serving"] = _swap(previous, prev_srv.get("n_action_steps") or settings().get("n_action_steps"),
                                prev_srv.get("grasp_extend") or 0.0)
    dep_id = _dep_id(previous, "rollback-")
    reg.add_deployment(dep_id, previous, "deployed", meta)
    reg.set_model_status(current, "rolled_back")
    reg.set_model_status(previous, "deployed")
    return {"dep_id": dep_id, "from": current, "to": previous}
