# Roles

| Person | Area | Owns | Output |
|---|---|---|---|
| 1 | Data engineering | R1–R3: ingest, quality checks, versioned dataset; data side of option B (coverage, gaps) | `ds-v1` + data quality report |
| 2 | Model training | R4: training environment, ACT baseline, SmolVLA fine-tune (π0.5 LoRA stretch), checkpoints, retraining | `m-v1`, `m-v2` + run records |
| 3 | Simulation & integration | Simulator adapter, safe resets and stop conditions, rollout recording; R10 deployment in simulation | Policy running in the simulator + rollout recordings |
| 4 | Evaluation & quality | R5 test suite and runner, metrics, R8 explainability, option A success judge | Evaluation suite + per-model results |
| 5 | MLOps & improvement | R6 gates, R7 quality statement, R9 registry and dashboard, R11 / option B loop, pitch numbers | Reproducible pipeline + dashboard + version comparison |
| Pitch | Shared (5 + 1) | Story, measured numbers, demo script, backup video | Final presentation |

## Hand-offs

| From → To | What | Where |
|---|---|---|
| 1 → 2 | Approved dataset version (manifest with train episodes) | `outputs/datasets/ds-vN/manifest.json` |
| 2 → 3 | Checkpoint folder + policy type | `registry.models.checkpoint_dir` |
| 3 → 4 | One `EpisodeResult` per episode + video/trajectory | `outputs/eval/<eval_id>/episodes.jsonl` |
| 4 → 5 | Metrics JSON per evaluation | `outputs/eval/<eval_id>/metrics.json` |
| 5 → all | Gates, model card, recommendations | `outputs/eval/<eval_id>/` |

Stand-ups: 09:00, 14:00, 20:00 — 10 minutes each.
