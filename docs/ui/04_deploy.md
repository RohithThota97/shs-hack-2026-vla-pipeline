# 04 · Deploy page (R6, R7, R10, Hub) — spec for review

## Goal
Show whether a model may be deployed, the trustworthy quality statement, and gated deploy / rollback. Publish the
model + card to the private Hugging Face repo.

## Panels
1. **Gates:** per gate pass/fail with value and threshold (`configs/gates.yaml`), and the recommendation
   (deploy / supervised / do not deploy).
2. **Quality statement:** the R7 card for the selected evaluation.
3. **Deploy / Rollback:** human buttons with name. They show the smoke-test result, the live version and the
   history.
4. **Publish to Hub:** dry run (card preview + file list) → Publish (private). Refused for models in
   `training`/`failed`; data files are never uploaded. Shows the Hub URL and commit once done.

## Backend
`gates/`, `reporting/model_card.py`, `deploy/`, `hub/publish.py` (exist).

## As built (23 Oct) — after the team's pipeline UI and fr3duo_policy_server.py (AI-project-work)
- **Serving** (`hackpipe serve --model m-vN`, `deploy/server.py`): the team's Isaac Sim client protocol on port
  8090 — `GET /info`, `POST /reset`, `POST /act` (npz with the dataset's feature names + `task` → 16 actions) — plus
  `POST /probe` (one recorded frame of the probe source, for the health check). Dataset camera names are renamed
  to the checkpoint's (camera1..3 for lerobot-train checkpoints, `configs/dataset.yaml` rename_map). Refuses
  checkpoints outside the FR3 Duo contract. Options: `--n-action-steps` (re-plan), `--grasp-extend` (the team's
  inference-time grasp shaping), `--dummy` (FAKE, bridge tests).
- **Process manager** (`deploy/serving.py`, `hackpipe serving status|stop|start`): servers run in their own session
  (they keep serving when the console or the job ends); state in `outputs/deploy/serving_<port>.json`. Only
  servers hackpipe started are stopped; a port held by something else is reported, never killed.
- **Deploy** (`hackpipe deploy --model m-vN --serve`): gated by the verdict — DEPLOY, or HOLD: SIM UNVERIFIED with
  `--allow-supervised` (recorded); MORE TRAINING / NOT EVALUATED are refused. Then: load on the stage port 8091 →
  health check (/info not FAKE, /reset, /probe → 16 finite values) → stop staging → replace 8090. A failed health
  check leaves the old server serving; if the new production server fails to start, the previous model is started
  again. Settings in `configs/project.yaml` → deploy.
- **Rollback** (`hackpipe rollback --serve`): the previous deployed model with its recorded settings, staged and
  health-checked the same way.
- **Page:** port 8090 now (model, re-plan setting, since; warning when a non-hackpipe server holds it), Stop
  serving / Start (deployed model) / Roll back; Deploy (model with verdict, actions per prediction, grasp
  extension, supervision checkbox for HOLD, your name), history with verdict, who, supervised, health check time,
  errors. Deploy / rollback / start run as jobs.
- Not built here: Hub publishing panel (`hackpipe publish` stays CLI).
- Verified with a real checkpoint (CPU): served in 14 s, health check on a lighting10 frame passed (16 finite
  values), client-style /act re-planned every 5 calls.
