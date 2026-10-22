# Option A · Intelligent Success Judgement

**Owner:** Person 4 · Evaluation

Judge rollouts from vision / state traces: success, progress, and where execution degraded.

- **Input:** `EpisodeResult` + rollout frames.
- **Output:** `JudgeResult` per episode + agreement with ground truth.

## To do on site

- [ ] Baseline: `StateTraceJudge` (works now).
- [ ] Implement `VLMJudge` (candidates in LeRobot: ROBOMETER, TOPReward).
- [ ] Report judge accuracy vs ground truth with `agreement()`.
