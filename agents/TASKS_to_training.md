# Task list · from data to the first trained model (draft for review)

Status: **Decisions made 22 Oct — FR3 Duo handover + SmolVLA.** Briefs 00 / P1 / P2 updated to match. Nothing run yet.

| Decision | Outcome |
|---|---|
| D1 | FR3 Duo is the main track; LIBERO dropped |
| D2 | Handover data (`v9_night_84`) trains model_v1; rand500 unused |
| D3 | `v9_lighting10` = held-out test set (assumed from the suggestion — change if you disagree) |
| D4 | SmolVLA from `lerobot/smolvla_base`; ACT optional baseline |
| D5 | **Open** — GPU box, internet, who runs it |
Scope stops at "model_v1 trained and registered". Evaluation, gates, deploy are out of scope here.

---

## What the review found

### The data we actually have (`data/C2_Teach-the-robot_VLA/`)

| Dataset | Task string | Episodes | Frames / ep | fps | Notes |
|---|---|---|---|---|---|
| `Dataset/fr3_duo_rand500.zip` = `Datasets/fr3_duo_rand500` (same zip, byte-identical) | "stack the three cubes into one tower" | **490** (not 500) | 339–369 (~58 s) | **6** | No joint names in `info.json`, no `fr3_conversion.json`, extra `norm_stats.json` (openpi format). Single `Dataset/` folder → probably our team-specific set. |
| `Datasets/fr3_duo_handover_84_ep` (`fr3_duo_handover_v9_night_84`) | "Hand the box from the left gripper to the right gripper and place it on the serving tray." | 84 | 2247 (~75 s) | 30 | Night-lighting randomisation, seeds 75000+. Has conversion + per-episode extra metadata. |
| `Datasets/FR3Duo_Handover_Dataset_lerobot3` (`v9_lighting10`) | same handover task | 10 | 2247 | 30 | Lighting randomisation, seeds 71100–71109. Described in the env doc. |

Common to all: LeRobot v3.0, `robot_type: fr3_duo`, 3 cameras (`external_camera`, `left_wrist_camera`,
`right_wrist_camera`) 224×224 AV1, `observation.state` and `action` 16-dim
(left q1–q7 rad, left gripper, right q1–q7, right gripper), gripper action 1 = open / 0 = closed,
`success` column, every episode successful (no failure demos anywhere).

Environment zip: USD scenes only (`Collected_Hackathon_FR3Duo_old` / `_v2`, Isaac Sim 5.1 assets) + one
markdown doc (`isaac_sim_hackathon_environment.md`). **No controller, no policy interface, no eval script.**
The doc describes the handover scene (box, tray); nothing about a cube-stacking scene.

### Mismatch between `agents/*.md` and the repo (important)

The briefs are written for a **LIBERO-Object first** plan ("Setup 1") and reference things that do not exist
in the repo at commit `cfb68e5`:

- Files: `CLAUDE.md`, `docs/setup1_libero.md`, `docs/wsl_setup.md`, `scripts/check_env.sh`,
  `configs/setups/{libero_object,fr3_duo}/*.yaml`, `configs/train/smolvla_libero.yaml`,
  `configs/eval/libero_object.yaml`.
- CLI: `HACKPIPE_SETUP` env var, `hackpipe libero-task-names`, `train --set`, `eval --set`,
  `dataset create --base/--boost`. Today the CLI only has `ingest/qc --root`, `dataset create --notes`,
  `train --config/--dataset/--dry-run/--resume/--register-only`.
- Paths: briefs say `docs/agents/…`; files live in `agents/`.

So either a newer version of the code exists somewhere and wasn't committed, or this all still needs building.
→ **Decision D1** below.

---

## Decisions needed before anyone starts (human)

| # | Decision | Options | My suggestion |
|---|---|---|---|
| D1 | Which plan is real? | (a) LIBERO-Object as the main loop + FR3 Duo as stretch (as briefs say) · (b) FR3 Duo data as the main track | (b), or keep LIBERO only as a fallback: the organisers' data, scene and DoD ("own expert policy") are FR3 Duo. Missing LIBERO code would have to be written first anyway. |
| D2 | Which FR3 Duo dataset trains model_v1? | rand500 (cube stack, 490 ep, 6 fps) · handover 84+10 (94 ep, 30 fps) · both (two tasks, language picks) | Ask organisers which one is "ours" (Q11). If no answer: handover 84+10 — it matches the provided USD scene, so a closed-loop eval is at least possible later. |
| D3 | Merge 84 + 10 handover sets? | merge into one LeRobot dataset · train on 84, keep 10 (lighting) as held-out test | Keep the 10-episode lighting set as the **held-out test** (different seeds + lighting = a real robustness split). |
| D4 | Policy for model_v1 | SmolVLA (`lerobot/smolvla_base`) · ACT baseline · π0.5 LoRA (openpi) | SmolVLA main, ACT baseline (as in `docs/decisions.md`). Note: `norm_stats.json` in rand500 hints the organisers used openpi. |
| D5 | Where does training run? | Windows RTX 4090 via WSL2 (briefs) · other | Confirm the box exists, has the data copied, and internet for HF model download. |

---

## Tasks

Legend — **Where:** GPU = runner session on the GPU box; LAPTOP = code session (no GPU); HUMAN = a person.
Each task: goal → steps → done when → STOP points.

### T0 · Organiser questions (HUMAN, now, 15 min)
Add to `docs/organizer_questions.md` and ask on site:
- Q11 Which dataset is our team's? Why 490 episodes not 500? Why is rand500 6 fps?
- Q12 Is there a cube-stacking scene? The env zip only has the handover scene.
- Q13 Is there a controller / Isaac Sim script to step the robot with joint targets, or do we write it?
- Q14 Official success definition and evaluation rules (the "Docs" link from the slides).
- Q15 rand500 state/action layout — same 16-dim order and gripper convention as the handover sets?

**Done when:** answers (or "no answer yet") written in the table. Feeds D1–D2.

### T1 · Fix the agent briefs to match the decision — ✅ done for 00 / P1 / P2; P3–P6 still LIBERO (banner added)
- Paths `docs/agents/` → `agents/` (or move the folder into `docs/`).
- If D1 = (b): rewrite `00_setup.md`, `P1_data.md`, `P2_training.md` for FR3 Duo; drop LIBERO steps or
  move them to an appendix "fallback".
- Remove or mark as TODO every command / flag that doesn't exist yet (list above).
- Add `CLAUDE.md` (agents are told to read it) — short: rules from `agents/README.md`, repo layout, test command.

**Done when:** every command in `00_setup`, `P1`, `P2` exists in `hackpipe --help` or is marked TODO with a task ID below.

### T2 · Environment on the GPU box (GPU, short) — from `00_setup.md`
1. `nvidia-smi`, `free -g`, `df -h ~` (data alone is ~4.5 GB unzipped + checkpoints), `tmux`.
2. Repo cloned, `python3 -m venv .venv && pip install -e ".[dev,dashboard,video]" && pytest -q` → note count.
3. LeRobot side by side (`pip install -e ".[smolvla]"`; `[libero]` only if D1 = a). Record commit + versions
   in `docs/decisions.md`. The datasets were written with **lerobot 0.6.1** (`fr3_conversion.json`) —
   pin ≥ that version, check it reads v3.0.
4. Check AV1 decoding works (the videos are AV1): load one frame with LeRobot's video backend. If it fails →
   install an ffmpeg/torchcodec with AV1 (dav1d) support. **This is a likely blocker, test it early.**
5. Copy the dataset(s) from D2 to `data/raw/<name>` on the box (not via git).
6. Skip the LIBERO reference model / smoke test unless D1 = (a).

**Done when:** tests green, LeRobot version pinned, one AV1 frame decoded, data present on the box.
**STOP:** GPU not visible in WSL · AV1 can't be decoded after 45 min · < 60 GB free disk.

### T3 · `scripts/check_env.sh` (LAPTOP, 30 min) — referenced by 00_setup but missing
Checks: python version, `hackpipe` importable, `lerobot` version, CUDA + GPU name + VRAM, free disk, AV1 decode,
`data/raw/<name>/meta/info.json` present. Prints PASS/WARN/FAIL per line.

### T4 · Configs for FR3 Duo (LAPTOP, 30 min) — resolve `CONFIRM` values
From the facts above (all from the dataset files / env doc):
- `configs/project.yaml`: `robot.type: fr3_duo`, `task.name` + exact instruction string(s) from `tasks.parquet`,
  `dataset.root`, `dataset.repo_id`.
- `configs/dataset.yaml`: `columns.success: success`; `quality.joint_limits` from the FR3 limits (or
  dataset stats min/max + margin); frame bounds (handover is always 2247 frames — a different length is
  REVIEW); `splits`: if D3 = held-out lighting set, set `test_episodes` or ingest it as a separate split.
- Record each value's source in `docs/decisions.md`.

**Done when:** no `CONFIRM` left that affects ingest/QC/training; `pytest -q` green.

### T5 · Data schema doc (LAPTOP, 30 min)
Write `docs/data_schema_fr3_duo.md` from the files only: features, dims, joint names, gripper contract
(`gripper_contract` in `fr3_conversion.json`: closed 0.8 rad, hold 0.51 norm), action timing rule
(action[k] = command at raw row 2k+2), fps, episode lengths, per-episode metadata (seed, lighting,
final box pose), differences between the three datasets (incl. rand500's missing joint names and 6 fps).

### T6 · Ingest + QC on the real data (GPU box or LAPTOP with data, no GPU) — P1 R1–R2
1. `hackpipe ingest --root data/raw/<name>` → check demo records: episode count (84 / 10 / 490), task,
   seed + lighting pulled from `fr3_episodes_extra.jsonl` into the demo record metadata (extend the importer
   if it doesn't read that file — with a synthetic test).
2. `hackpipe qc --root data/raw/<name>` → read the report. Expect all VALID; every REVIEW/INVALID needs a reason.
3. Add checks that matter for this data if missing: video frame count == parquet rows per camera,
   gripper values within [0, 1], no large joint jumps between frames, non-finite values.

**Done when:** QC report exists, counts match `info.json`, every flag explained.
**STOP:** episode count differs from `info.json` · any INVALID episode without an obvious reason.

### T7 · Create ds-v1 (no GPU) — P1 R3
1. `hackpipe dataset create --notes "v1: <dataset>, <split rule>"`.
2. Check `dataset_card.md`: train/val/test counts, leakage guard passed, fingerprint present, test split =
   whatever D3 decided.
3. If D3 needs a merged dataset (84 + 10 in one LeRobot dataset): do the merge with LeRobot's own tooling
   into `data/processed/…`, ingest that, and record the merge command in the card.

**Done when:** `ds-v1` exists as draft.
**STOP (always):** show the human the QC summary + dataset card. **The human runs
`hackpipe dataset approve ds-v1`.**

### T8 · Training code gaps (LAPTOP, 1–2 h) — needed by P2 before the GPU run
- `hackpipe train --set key=value` (repeatable) → override `args` in the config and store overrides in the
  run record. Needed for the timing run (`steps=200`). Test it.
- New config `configs/train/smolvla_fr3_duo.yaml`:
  `policy.path: lerobot/smolvla_base`; `rename_map` from our 3 camera keys to the policy's camera inputs
  (`camera1/2/3` or whatever smolvla_base expects — check its config); state/action 16 ≤ policy max dims
  (check `max_state_dim`/`max_action_dim`); `chunk_size`/`n_action_steps` sensible for 30 fps (or 6 fps for
  rand500); dataset revision/root; `wandb.enable=false` if no internet.
- Same for `configs/train/act.yaml` (baseline) — ACT has no language input; fine for a single-task set.
- `train --dry-run` prints the full `lerobot-train` command — unit test it with the new config.

**Done when:** dry-run prints a command that a human checks line by line; `pytest -q` green.

### T9 · Model choice note (LAPTOP, 20 min) — P2 §0
`docs/model_choice.md`: SmolVLA vs ACT vs π0.5/GR00T; camera/state mapping; 16-dim fits; why it fits 24 GB.
Facts from configs only.

### T10 · Dry run + timing run (GPU, ~15 min) — P2 §1–2
1. `hackpipe train --config configs/train/smolvla_fr3_duo.yaml --dataset ds-v1 --dry-run` → human checks
   dataset path, policy, batch size, steps, save_freq, seed, output dir, logging.
2. Timing run in tmux with `flock /tmp/hackpipe-gpu.lock`, `--set steps=200 --set save_freq=200`,
   VRAM logged via `nvidia-smi … -l 10`. OOM → `batch_size=8`, retry.
3. Record s/step, peak VRAM, batch size, data-loader throughput (AV1 decode may be the bottleneck → raise
   `num_workers` or pre-decode). The timing model ID goes to STATUS.md as "timing run, not a result".

**STOP:** OOM at batch 8 · GPU busy · s/step so slow the slot gives < ~5k steps.

### T11 · Fix the step budget (HUMAN + P2, 5 min) — P2 §3
Human gives the end of the training slot. `steps = slot_hours × 3600 / s_per_step`, rounded down to a
multiple of `save_freq`. Commit to the config, reason in `docs/decisions.md`. Frozen for model_v2.

### T12 · Full training run → model_v1 (GPU, overnight) — P2 §4–5
1. tmux + flock, log to `outputs/<setup>/logs/train_v1.log`. Wait for first loss lines, then hand off
   (tmux session, log path, expected end) in STATUS.md.
2. After: run record complete (command, config, overrides, dataset ID + fingerprint, seed, git commit,
   packages, GPU); `loss.csv` + `loss.png` (`scripts/plot_loss.py`); checkpoint has config + weights;
   model registered — write the printed model ID into STATUS.md.
3. Optional, only if the GPU is idle afterwards: ACT baseline on the same ds-v1, labelled "baseline".

**Done when:** model_v1 registered with a complete run record and loss curve. → Next stage is evaluation (out of scope here).
**STOP:** crash that `--resume` doesn't fix.

---

## Order / dependencies

```
T0 ──► D1–D5 ──► T1
                 T2 (GPU box) ─────────────┐
                 T3, T4, T5 (laptop) ─► T6 ─► T7 ─► [human approves ds-v1] ─┐
                 T8, T9 (laptop) ──────────────────────────────────────────┴─► T10 ─► T11 ─► T12
```
T2–T5, T8, T9 can run in parallel. T10 is the first GPU job.

## Housekeeping noticed
- `Dataset/fr3_duo_rand500.zip` and `Datasets/fr3_duo_rand500.zip` are identical (2.1 GB duplicate).
- Zips + unzipped copies both kept (~9 GB); `data/` is git-ignored, so only a disk-space issue.
- `agents/` is untracked in git — commit it once reviewed.
