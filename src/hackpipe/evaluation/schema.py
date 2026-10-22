"""Episode result record — the contract between the simulator adapter (Person 3) and evaluation (Person 4).

One record per evaluation episode, stored as one JSON line in episodes.jsonl. Fields that cannot be
measured stay None (they are then excluded from that metric, never guessed).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EpisodeResult:
    # identity
    eval_id: str
    model_version: str
    dataset_version: str
    test_suite: str
    env_version: str
    scenario_id: str
    category: str
    seen: bool
    seed: int
    instruction: str
    # outcome
    success: bool
    stop_reason: str                      # success | timeout | safety_stop | operator_stop | error | failed
    failure_stage: str | None = None      # reach | grasp | lift | transfer | place | release
    correct_object: bool | None = None    # grasped item == instructed item
    grasp_attempts: int = 0
    grasp_successes: int = 0
    dropped: bool = False
    in_target_zone: bool | None = None
    placement_error_m: float | None = None
    collision: bool = False
    timeout: bool = False
    safety_stop: bool = False
    interventions: int = 0
    intervention_time_s: float = 0.0
    completion_time_s: float | None = None
    inference_latency_ms: list[float] = field(default_factory=list)
    # artefacts for explainability (paths relative to the eval folder)
    video_path: str | None = None
    trajectory_path: str | None = None
    stage_frames: dict[str, str] = field(default_factory=dict)   # stage → image path
    predicted_actions_path: str | None = None
    notes: str = ""
    fake: bool = False                    # True only for the dummy adapter

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EpisodeResult":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
