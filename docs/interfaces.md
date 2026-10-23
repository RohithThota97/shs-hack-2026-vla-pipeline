# Interfaces (data contracts)

Change a contract only through a PR that updates this file and tells the affected owner.

## 1. Demo record (R1 → R2/R3) — `registry.demos`

```json
{
  "demo_id": "uuid5(source, episode_index)",
  "source": "data/raw/fr3_duo_handover_84   (repo-relative, symlinks not followed; absolute if outside the repo)",
  "episode_index": 12,
  "status": "imported | VALID | REVIEW | INVALID | DUPLICATE",
  "meta": {
    "n_frames": 2247, "fps": 30, "duration_s": 74.87,
    "recording": {"start_s": 0.0, "end_s": 74.87, "wallclock_start": null},
    "task": "…", "instruction": "…",
    "robot": {"type": "fr3_duo", "embodiment": "dual_arm", "end_effector": "robotiq_2f85", "controller": "…"},
    "cameras": {"observation.images.external_camera": {"shape": [224, 224, 3], "fps": 30, "codec": "av1"}},
    "state_dim": 16, "action_dim": 16,
    "operator": "scripted_controller",
    "environment": {"scene": "…", "layout_id": "…", "items_present": ["…"],
                    "variation": "v9_night_light_s75012", "seed": 75012,
                    "lighting": {"color_temperature_K": 5841.7, "rect_intensity": 3212.8,
                                 "dome_intensity": 343.5, "rect_tilt_deg": [-10.1, 9.9]}},
    "outcome": {"status": "PASS", "success": true, "failure": null, "at_rest": true,
                "footprint_margin_mm": 55.96, "tilt_deg": 0.26, "placement_offset_mm": 0.35},
    "content_sha256": "frame/timestamp/state/action of this episode",
    "trajectory_sha256": "state/action only (finds redundant demos)",
    "files_sha256": {"data/chunk-000/file-000.parquet": "…", "videos/<camera>/chunk-000/file-003.mp4": "…"},
    "codebase_version": "v3.0",
    "pipeline": {"hackpipe": "0.1.0", "git_commit": "…"},
    "imported_at": "…",
    "qc": {"status": "VALID", "hard": [], "soft": [], "stats": {"video_frames:<camera>": 2247}}
  }
}
```

Also set when relevant: `duplicate_of` (demo ID of the same episode of the same files from another folder; status
`DUPLICATE`, never checked or versioned) and `excluded` (`{reason, by, at}`, set by hand; kept out of new versions).
Re-import: unchanged files → skipped; changed files → status back to `imported` (re-checked by QC).

`environment.variation/seed/lighting/layout_id` and `outcome` come from `meta/fr3_episodes_extra.jsonl` when the
dataset has it; otherwise `environment` holds `demo_defaults` from `configs/project.yaml` and `outcome` is null.

## 2. Dataset manifest (R3 → R4) — `outputs/datasets/ds-vN/manifest.json`

```json
{
  "version": "ds-v1", "fingerprint": "16 hex", "created_at": "…",
  "source": "data/raw/fr3_duo_handover_84   (null when several sources)",
  "sources": ["data/raw/fr3_duo_handover_84"], "test_sources": ["data/raw/fr3_duo_handover_lighting10"],
  "needs_materialize": false,
  "episodes": {"train": [["data/raw/…", 0], …], "val": […], "test": […]},
  "lineage": {"base": "ds-v1 | null", "inherited": [demo_id…], "added": […], "removed": […],
              "added_split": {"train": […], "val": […]}},
  "materialized": {"root": "data/processed/ds-v2", "repo_id": "local/ds-v2", "offsets": {source: n},
                   "splits": {"train": […], "val": […]}, "verified_episodes": 88}   (multi-source, after materialize)
  "repo_id": "local/fr3_duo_handover_84", "revision": null,
  "splits": {"train": [0, 1, …], "val": […], "test": […]},
  "demo_ids": {"train": […], "val": […], "test": […]},
  "excluded": [{"demo_id": "…", "source": "…", "episode_index": 7, "status": "INVALID", "reasons": ["…"]}],
  "preprocessing": {"columns": {…}, "preprocessing": {"image_size": null, "rename_map": {…}}, "splits": {…},
                    "schema": {"fps": 30, "cameras": {…}, "state": {"key", "shape", "names"}, "action": {…},
                               "codebase_version": "v3.0"},
                    "normalisation_stats": {"file": "meta/stats.json", "sha256": "…"}},
  "notes": "…"
}
```

`splits.test` episode indices belong to `test_sources` when those are set; train/val indices to `source`.
Training uses `--dataset.episodes=<train>` on `source`; a multi-source version is trained from its merged folder
(`materialized.splits.train` on `materialized.root`), after `hackpipe dataset materialize` verified the mapping by
content hash. A version with a `base` keeps the base's train/val assignment and frozen test split. `stats.json` is copied next to the manifest.
Fingerprint = hash of sorted (demo_id | content_sha256 | file checksums) + `preprocessing`.

## 3. Run record (R4 → R6/R7) — `outputs/runs/m-vN/run_record.json`

Model version, dataset version + fingerprint, policy, full training config, `overrides` (`--set`), exact
`lerobot-train` command, output dir, environment (Python, platform, GPU, git commit, packages), start/end time,
exit code, `purpose` (`model` | `timing`). The model is registered with status `training` before the run starts
(after a `lerobot-train --help` preflight), then `trained` / `failed`.
Training metrics: `loss.csv` (+ `loss.png`) next to the run record (`scripts/plot_loss.py`).
After `hackpipe publish`: `registry.models.meta.hub = {repo_id, private, commit, published_at}`.

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

## 6. Job (console → registry `jobs`)

`job_id` (job-NNNN), `kind` (auto, qc, train, …), `status` (queued → running → done / failed), `needs` (cpu / gpu),
`machine`, `pid`, `log_path` (`outputs/logs/jobs/<job_id>.log`), `work_units` (e.g. video frames to decode),
`estimate_s` (only from measured earlier jobs; null = no estimate yet), `exit_code`, `started_at`, `ended_at`,
`meta` (`argv`, `label`, `estimate_basis`, `error`).
