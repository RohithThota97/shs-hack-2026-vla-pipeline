"""Simulator adapter interface.

Person 3 implements `OrganizerSimAdapter` once the provided 3D environment (FR3 Duo) is downloaded and its
observation/action interface is known. The evaluation runner only talks to this interface, so evaluation,
metrics, gates and reports work unchanged.

`DummySimAdapter` produces FAKE results for testing the pipeline. Its records carry fake=True and every
report shows a warning. Never use its numbers in the pitch.
"""
from __future__ import annotations

import random
from typing import Any, Protocol

from hackpipe.evaluation.schema import EpisodeResult


class SimAdapter(Protocol):
    env_version: str

    def run_episode(self, model: dict[str, Any], scenario: dict[str, Any], seed: int,
                    time_limit_s: float, identity: dict[str, Any]) -> EpisodeResult:
        """Reset the scene for `scenario` with `seed`, run the policy until success, stop or time limit,
        save video / trajectory / stage frames, and return one EpisodeResult."""
        ...


class OrganizerSimAdapter:
    """TODO (Person 3): connect to the organisers' simulator.

    Steps:
    1. Load the scene from the provided environment and reset it for the scenario (layout, lighting,
       obstacles) with a fixed seed.
    2. Load the policy from model["checkpoint_dir"] (LeRobot: config.json + model.safetensors) and map
       simulator observations to the policy's input keys (cameras, state); map policy actions to the robot.
    3. Step until success, safety stop, or time limit. Log per-step state/action, inference latency,
       contacts/collisions, grasp events, object poses.
    4. Save video, trajectory and one image at the start of each stage; fill EpisodeResult.
    """

    def __init__(self, env_version: str = "env-v1", **kwargs: Any):
        self.env_version = env_version
        self.kwargs = kwargs

    def run_episode(self, model, scenario, seed, time_limit_s, identity) -> EpisodeResult:  # noqa: D401
        raise NotImplementedError("Implement OrganizerSimAdapter.run_episode for the provided simulator.")


class DummySimAdapter:
    """FAKE outcomes, deterministic per seed. For pipeline tests only."""

    def __init__(self, base_success: float = 0.7, env_version: str = "dummy"):
        self.base_success = base_success
        self.env_version = env_version

    def run_episode(self, model, scenario, seed, time_limit_s, identity) -> EpisodeResult:
        rng = random.Random(f"{model['version']}:{scenario['id']}:{seed}")
        p = self.base_success - (0.0 if scenario.get("seen", True) else 0.15)
        success = rng.random() < p
        stage = None if success else rng.choice(["grasp", "transfer", "place"])
        timeout = (not success) and rng.random() < 0.2
        collision = (not success) and not timeout and rng.random() < 0.15
        grasped = success or stage in ("transfer", "place")
        return EpisodeResult(
            **identity, scenario_id=scenario["id"], category=scenario.get("category", "seen"),
            seen=bool(scenario.get("seen", True)), seed=seed, instruction=scenario.get("instruction", ""),
            success=success,
            stop_reason="success" if success else ("timeout" if timeout else "failed"),
            failure_stage=stage, correct_object=True if grasped else None,
            grasp_attempts=1, grasp_successes=1 if grasped else 0,
            dropped=(stage == "transfer"), in_target_zone=success if grasped else None,
            placement_error_m=round(rng.uniform(0.0, 0.03), 3) if success else None,
            collision=collision, timeout=timeout,
            completion_time_s=round(rng.uniform(15, 45), 1) if success else None,
            inference_latency_ms=[round(rng.uniform(40, 90), 1) for _ in range(5)],
            fake=True, notes="FAKE result from DummySimAdapter",
        )
