"""Option A — Intelligent Success Judgement.

Goal (organisers): assess rollouts with vision, VLMs or state traces; explain if, how well and where
execution degraded.

- `StateTraceJudge`: baseline from logged state/outcome fields (works now).
- `VLMJudge`: TODO (Person 4). Candidate backends already in LeRobot: ROBOMETER (progress + success per
  frame, checkpoint lerobot/Robometer-4B) and TOPReward (zero-shot from a VLM's token probabilities).
  Check GPU memory before running next to anything else on the 24 GB card.
- `agreement()`: measure how often the judge agrees with ground-truth success — report this, so the judge
  itself has a quality statement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from hackpipe.evaluation.schema import EpisodeResult
from hackpipe.explain.failures import categorize

STAGES = ["reach", "grasp", "lift", "transfer", "place", "release"]


@dataclass
class JudgeResult:
    success_prob: float                 # 0..1
    progress: float                     # 0..1, how far the task got
    degraded_at: str | None             # stage where execution degraded
    rationale: str
    per_frame_progress: list[float] = field(default_factory=list)


class SuccessJudge(Protocol):
    name: str

    def judge(self, episode: EpisodeResult, frames: list[Any] | None = None) -> JudgeResult: ...


class StateTraceJudge:
    name = "state_trace_v1"

    def judge(self, episode: EpisodeResult, frames: list[Any] | None = None) -> JudgeResult:
        if episode.success:
            return JudgeResult(1.0, 1.0, None, "All stages completed; item in target zone.")
        stage = episode.failure_stage
        progress = STAGES.index(stage) / len(STAGES) if stage in STAGES else 0.0
        return JudgeResult(0.0, round(progress, 2), stage,
                           f"Failed: {categorize(episode).replace('_', ' ')}"
                           + (f" at stage '{stage}'." if stage else "."))


class VLMJudge:
    name = "vlm_v1"

    def __init__(self, backend: str = "robometer", **kwargs: Any):
        self.backend = backend
        self.kwargs = kwargs

    def judge(self, episode: EpisodeResult, frames: list[Any] | None = None) -> JudgeResult:
        raise NotImplementedError(
            "TODO (Person 4): run a reward model / VLM on the rollout video frames with the instruction, "
            "return per-frame progress, success probability, and the first stage where progress drops.")


def agreement(judged: list[JudgeResult], truth: list[EpisodeResult], threshold: float = 0.5) -> dict[str, Any]:
    pairs = list(zip(judged, truth))
    if not pairs:
        return {"n": 0, "accuracy": None}
    correct = sum((j.success_prob >= threshold) == t.success for j, t in pairs)
    fp = sum(j.success_prob >= threshold and not t.success for j, t in pairs)
    fn = sum(j.success_prob < threshold and t.success for j, t in pairs)
    return {"n": len(pairs), "accuracy": correct / len(pairs), "false_success": fp, "missed_success": fn}
