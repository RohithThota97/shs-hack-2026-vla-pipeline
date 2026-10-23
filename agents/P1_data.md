# P1 · Data engineering (R1–R3)

**Goal:** an approved, versioned dataset `ds-v1` built from the 84-episode FR3 Duo handover set, with a data
quality report; the 10-episode lighting set checked and kept as the held-out test set.
**Owner:** Person 1 · Data. **Where:** runner session for `hackpipe` runs; code session for code changes.
**Read first:** `docs/interfaces.md` (demo record, dataset manifest), `configs/project.yaml`,
`configs/dataset.yaml`, `src/hackpipe/{ingest,quality,dataset}/`, and in each dataset `meta/info.json`,
`meta/fr3_conversion.json`, `meta/fr3_episodes_extra.jsonl`.
**Needs:** 00_setup done (data under `data/raw/`). **No GPU.**

Known facts (from the files — re-check, don't trust blindly):
- 84 episodes × 2247 frames = 188,748 frames, 30 fps, one task. Lighting set: 10 × 2247 = 22,470.
- Cameras `observation.images.{external_camera,left_wrist_camera,right_wrist_camera}`, 224×224, AV1.
- `observation.state` / `action`: 16 float32 — left q1..q7 (rad), left gripper, right q1..q7, right gripper.
  State gripper = normalised measured joint; action gripper = `1 − cmd/0.8`, 1 = open, 0 = closed.
- `action[k]` = command held at raw 60 Hz row `2k+2` (see `action_timing_rule`).
- `success` column; every episode in both sets is PASS / successful. **There are no failure demos.**
- Per episode in `fr3_episodes_extra.jsonl`: name, seed, lighting (requested / applied), final box pose,
  placement metrics.

## A · Configs and schema doc (code session, do first)

Status 22 Oct: steps 1, 3 and 4 are **done** (configs filled, importer reads the per-episode records, video /
final-state / redundancy checks built and tested). Left: the joint-limit `CONFIRM` and step 2.

1. Fill the `CONFIRM` values (done — joint limits still to confirm against the USD robot):
   - `configs/project.yaml`: `robot.type: fr3_duo`, `task.name: handover_box_to_tray`,
     `task.instruction_templates`: the exact task string above,
     `dataset.root: data/raw/fr3_duo_handover_84`, `dataset.repo_id: local/fr3_duo_handover_84`,
     `demo_defaults.environment.scene`: the USD scene name from the environment doc.
   - `configs/dataset.yaml`: `columns.success: success`; `quality.min_episode_seconds` /
     `max_episode_seconds` (50 s / 83.34 s; length = frames / fps, so 30 and 6 fps data share one rule);
     `quality.joint_limits` from FR3 joint
     limits (or dataset `stats.json` min/max with a margin — say which); splits: `train: 0.9`, `val: 0.1`,
     `test: 0.0` (the test set is the separate lighting10 dataset).
   Record every value and its source in `docs/decisions.md`.
2. Write `docs/data_schema_fr3_duo.md` from the files only: features and shapes, joint order, gripper
   contract, action timing rule, fps (30 Hz from 60 Hz sim), episode length, per-episode metadata fields,
   how the night-84 and lighting10 sets differ (seeds, lighting ranges).
3. Importer: if it doesn't read `meta/fr3_episodes_extra.jsonl`, add it — seed, lighting and placement
   metrics go into the demo record metadata. Test on a synthetic dataset in `tests/`.
4. Extra quality checks, if missing (each with a test): video frame count per camera == parquet rows;
   gripper values in [0, 1]; non-finite state/action; large jumps between consecutive joint targets.

## Automatic path (preferred, 23 Oct)

`hackpipe auto --root <folder>` (or Data → Add demonstrations → Run pipeline in `hackpipe console`) does import →
QC of new/changed demos only → next draft version based on the latest approved one. Duplicates of already
registered demos are recognised and skipped. You still STOP before approval; the human approves in the console or
with `hackpipe dataset approve ds-vN --by <name>`. The manual steps below are what `auto` runs.

## B · ds-v1 (runner session)

1. **Import:** `hackpipe ingest --root data/raw/fr3_duo_handover_84` → 84 demo records, one task.
2. **Quality check:** `hackpipe qc --root data/raw/fr3_duo_handover_84` → read
   `outputs/qc/fr3_duo_handover_84/qc_report.md`. Every REVIEW / INVALID has a reason. Expect all VALID and
   the Redundancy section to say 1 distinct trajectory (measured 22 Oct) — investigate anything else.
   Also ingest + QC the held-out set now (section C) — `dataset create` needs it.
3. **Create the version:**
   ```bash
   hackpipe dataset create --notes "v1: fr3_duo_handover_v9_night_84, 90/10 train/val; test = lighting10 (separate)"
   ```
   Check `outputs/datasets/ds-v1/dataset_card.md`: 76 train / 8 val / 10 test (lighting10), leakage guard on
   seed passed, fingerprint present, known gaps listed. (Measured on a scratch registry 22 Oct.)
4. **STOP.** Show the human the QC summary (VALID / REVIEW / INVALID counts and reasons) and the dataset card.
   The human runs `hackpipe dataset approve ds-v1`. You don't.

## C · Held-out test set (lighting10)

1. `hackpipe ingest --root data/raw/fr3_duo_handover_lighting10` and `hackpipe qc --root …` → report in
   `outputs/qc/fr3_duo_handover_lighting10/`. Demo IDs include the source path, so they don't collide with night84.
2. `configs/dataset.yaml` → `splits.test_sources` makes it the whole test split; the leakage guard refuses any
   test seed in train/val, and the test split is frozen from ds-v1 on.
3. Never in a training split. Used by evaluation (P4) only.

## Later · ds-v2

After the first evaluation and P5's improve step. The flags `dataset create --base/--boost` don't exist yet
**(to build)**. With only 84 successful demos and no failure demos, ds-v2 likely needs new demonstrations
from the simulator — to be planned with P3/P5.

## Done when

- [ ] `CONFIRM` values filled, sources in `docs/decisions.md`; `docs/data_schema_fr3_duo.md` written
- [ ] QC report for night84: 84 episodes, every flag has a reason
- [ ] `ds-v1` created, approved by a human
- [ ] lighting10 QC'd and documented as the held-out test set
- [ ] Any code change has tests; `pytest -q` green

## Hand-off (STATUS.md)

ds-v1 ID and who approved · VALID / REVIEW / INVALID counts · train / val episode counts and IDs file ·
lighting10 QC result · whether lighting10 was ingested.

## STOP and ask

Before `dataset approve` (always) · episode or frame counts differ from `info.json` · any INVALID episode ·
the importer can't keep two datasets apart.
