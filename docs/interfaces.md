# Interfaces (data contracts)

Change a contract only through a PR that updates this file and tells the affected owner.

## 1. Demo record (R1 → R2/R3) — `registry.demos`

```json
{
  "demo_id": "uuid5(source, episode_index)",
  "source": "/abs/path/to/dataset",
  "episode_index": 12,
  "status": "imported | VALID | REVIEW | INVALID",
  "meta": {
    "n_frames": 412, "fps": 30, "duration_s": 13.7, "task": "…",
    "robot": {"type": "…", "embodiment": "dual_arm"},
    "cameras": {"observation.images.head": [480, 640, 3]},
    "state_dim": 16, "action_dim": 16,
    "operator": "unknown", "environment": {"scene": "…", "layout_id": "…", "lighting": "…"},
    "content_sha256": "…", "imported_at": "…",
    "qc": {"status": "VALID", "hard": [], "soft": [], "stats": {}}
  }
}
```

## 2. Dataset manifest (R3 → R4) — `outputs/datasets/ds-vN/manifest.json`

```json
{
  "version": "ds-v1", "fingerprint": "16 hex", "source": "…", "repo_id": "local/provided_dataset",
  "splits": {"train": [0, 1, …], "val": […], "test": […]},
  "demo_ids": {"train": […], "val": […], "test": […]},
  "excluded": [{"demo_id": "…", "episode_index": 7, "status": "INVALID", "reasons": ["action_nan_or_inf"]}],
  "preprocessing": {"columns": {…}, "preprocessing": {…}, "splits": {…}}
}
```

## 3. Run record (R4 → R6/R7) — `outputs/runs/m-vN/run_record.json`

Model version, dataset version + fingerprint, policy, full training config, exact `lerobot-train` command,
output dir, environment (Python, platform, GPU, git commit, packages), start/end time, exit code.

## 4. Episode result (simulator adapter → evaluation) — one JSON line in `episodes.jsonl`

Defined in `src/hackpipe/evaluation/schema.py` (`EpisodeResult`). Required: identity (eval, model, dataset,
test suite, environment, scenario, category, seen, seed, instruction), `success`, `stop_reason`.
Measured if possible (else `null`): `correct_object`, grasp attempts/successes, `dropped`,
`in_target_zone`, `placement_error_m`, `collision`, `timeout`, `safety_stop`, interventions,
`completion_time_s`, `inference_latency_ms`, artefact paths (video, trajectory, stage frames, predicted actions).

## 5. Metrics (evaluation → gates, card, dashboard) — `metrics.json`

`n_runs`, `fake_results`, `task_success {n, successes, rate, ci95}`, rates (correct object, grasp, drop,
in-zone, collision, timeout, safety stop), completion time, interventions, latency p50/p95, `seen`/`unseen`,
`by_category`, `by_scenario`, `failure_categories`. Computed by `src/hackpipe/evaluation/metrics.py`.
