"""R10 — Deploy a model only after it passes the quality gates and a smoke test; one-command rollback.

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
from hackpipe.gates.gates import evaluate_gates

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


def deploy(model_version: str, registry: Registry | None = None, allow_supervised: bool = False) -> dict[str, Any]:
    reg = registry or Registry()
    model = reg.get_model(model_version)
    if model is None:
        raise KeyError(model_version)
    evals = reg.evals(model_version)
    if not evals:
        raise RuntimeError(f"{model_version} has no evaluation. Run `hackpipe eval` first.")
    gates = evaluate_gates(evals[-1]["eval_id"], reg)
    smoke = smoke_test(model["checkpoint_dir"])
    ok = smoke["passed"] and (gates["all_passed"] or
                              (allow_supervised and gates["recommendation"].startswith("deploy with supervision")))
    dep_id = f"dep-{model_version}-{now_iso().replace(':', '').replace('-', '')}"
    status = "deployed" if ok else "rejected"
    reg.add_deployment(dep_id, model_version, status, {"gates": gates, "smoke_test": smoke,
                                                       "launch": LAUNCH_COMMAND, "supervised": allow_supervised})
    if ok:
        reg.set_model_status(model_version, "deployed")
    return {"dep_id": dep_id, "status": status, "gates": gates, "smoke_test": smoke, "launch": LAUNCH_COMMAND}


def rollback(registry: Registry | None = None) -> dict[str, Any]:
    """Re-deploy the previously deployed model."""
    reg = registry or Registry()
    deployed = [d for d in reg.deployments() if d["status"] == "deployed"]
    if len(deployed) < 2:
        raise RuntimeError("No previous deployed model to roll back to.")
    current, previous = deployed[-1]["model_version"], deployed[-2]["model_version"]
    dep_id = f"dep-{previous}-rollback-{now_iso().replace(':', '').replace('-', '')}"
    reg.add_deployment(dep_id, previous, "deployed", {"rollback_from": current, "launch": LAUNCH_COMMAND})
    reg.set_model_status(current, "rolled_back")
    reg.set_model_status(previous, "deployed")
    return {"dep_id": dep_id, "from": current, "to": previous}
