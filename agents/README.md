# Agent tasks · Challenge 2 pipeline

One file per role. Each file is a complete brief for one Claude Code session: goal, what to read, steps,
"done when", where to stop and ask a human, and what to hand off. Roles follow the organisers' role cards
(Person 1–5) and the challenge stages 01–05.

**Track (decided 22 Oct):** the organisers' **FR3 Duo handover** data, fine-tuned **SmolVLA**.
LIBERO-Object is no longer part of the plan. See `docs/decisions.md`.

| Dataset | Use |
|---|---|
| `fr3_duo_handover_v9_night_84` (84 episodes, 30 fps) | ds-v1: train + validation |
| `FR3Duo_Handover_Dataset_lerobot3` (`v9_lighting10`, 10 episodes, other seeds and lighting) | held-out test set, never trained on |
| `fr3_duo_rand500` (cube stacking, 490 episodes, 6 fps) | not used for now |

Task string (both handover sets): *"Hand the box from the left gripper to the right gripper and place it on
the serving tray."*

| Stage (challenge slide) | Requirement | File | Owner |
|---|---|---|---|
| — | Environment on the GPU box | [00_setup.md](00_setup.md) | whoever is at the GPU box |
| 01 Ingest & Transform · 02 Dataset Intelligence | R1–R3 | [P1_data.md](P1_data.md) | Person 1 · Data |
| 03 Fine-Tune | R4 | [P2_training.md](P2_training.md) | Person 2 · Training |
| 04 Evaluate (simulator side) · 05 Serve | R5 / R10 | [P3_sim_integration.md](P3_sim_integration.md) | Person 3 · Sim & integration |
| 04 Evaluate · Option A | R5–R8 | [P4_evaluation.md](P4_evaluation.md) | Person 4 · Evaluation |
| 05 Version · feeds next cycle · Option B | R6, R9–R11 | [P5_mlops.md](P5_mlops.md) | Person 5 · MLOps |
| Pitch | — | [P6_pitch.md](P6_pitch.md) | pitch lead |

00, P1 and P2 are written for the FR3 Duo track. **P3–P6 still describe the old LIBERO plan** and must be
rewritten before evaluation starts (see the note at the top of each).

Hand-offs and IDs go in [STATUS.md](STATUS.md). Every agent reads it first and appends to it last.
The full task breakdown up to training is in [TASKS_to_training.md](TASKS_to_training.md).

## Start an agent

```bash
cd ~/shs-hack-2026-vla-pipeline && source .venv/bin/activate
claude
```

Prompt:

> Read agents/README.md, agents/STATUS.md and agents/P1_data.md. Do that task.
> Stop where the file says STOP. Log what you did in agents/STATUS.md.

Swap `P1_data.md` for the role's file.

## Order (critical path up to model_v1)

One GPU, so GPU jobs run strictly one after another.

| # | Task | Needs | GPU | When |
|---|---|---|---|---|
| 1 | 00_setup — Colab notebook steps 0–5 (incl. AV1 check, data from Drive) | — | short | Day 1 evening |
| 2 | P1 → ds-v1 from the 84-episode set (human approves) | 1 | no | Day 1 evening |
| 3 | P2 → code gaps + config (laptop), then dry run, timing run | ds-v1 approved | yes, ~15 min | Day 1 evening |
| 4 | P2 → model_v1 full run, then `hackpipe publish` (model + card, private) | 3 + step budget from human | yes, hours | Day 1 night |
| 5 | Evaluation, gates, improve, model_v2 | model_v1 | — | Day 2+ (briefs P3–P5 to be rewritten) |

In parallel, code only, no GPU: P1 (schema doc, extra QC checks, importer reads per-episode metadata),
P2 (`train --set`, FR3 Duo SmolVLA config, model choice note), P3 (read the Isaac Sim scene, plan the
closed-loop or offline evaluation path).

## Two kinds of sessions

**Runner session (GPU box, main checkout, branch `main`).** Runs the pipeline commands in the order above.
Only one runner at a time. The registry, `data/` and `outputs/` live in this checkout.

**Code sessions (any laptop, or a git worktree with its own venv).** Change code on a branch, run `pytest -q`,
open a PR. Tests use synthetic data and need no GPU or LeRobot.

```bash
# On a laptop
git clone <repo url> && cd shs-hack-2026-vla-pipeline
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev,dashboard]"
git switch -c p2-train-set

# Or on the GPU box (own venv, so the runner's install is not touched)
git worktree add ../wt-p2 -b p2-train-set && cd ../wt-p2
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev,dashboard]"
```

Never run training or evaluation from a code session: its registry and outputs are separate and the results
would be lost.

## Where things are written (from the code)

| What | Path |
|---|---|
| QC report | `outputs/qc/<dataset folder>/qc_report.md` (+ `.json`) |
| Dataset card / manifest | `outputs/datasets/<ds-vN>/` |
| Training output (checkpoints) | `outputs/train/<model>/` |
| Run record | `outputs/runs/<model>/run_record.json` |
| Loss curve | `outputs/runs/<model>/loss.csv`, `loss.png` |
| Hub upload staging (model card) | `outputs/hub/<model>/` |
| Logs (create it) | `outputs/logs/` |
| Registry | `registry/registry.db` |

## Rules for every agent

- **One GPU job at a time.** Wrap every training, evaluation or Isaac Sim command:
  `flock /tmp/hackpipe-gpu.lock <command>`. It waits until the GPU is free. Check `nvidia-smi` first.
- **Long jobs in tmux**, so they survive a closed terminal:
  `tmux new -d -s <name> '<setup commands> && flock /tmp/hackpipe-gpu.lock <command> 2>&1 | tee <log>'`.
  Logs go to `outputs/logs/`.
- **Humans approve.** The agent never runs `hackpipe dataset approve`, `hackpipe deploy`, never edits a
  frozen test suite, and never deletes `data/`, `outputs/` or `registry/`. It stops and asks.
- **The held-out test set stays held out.** No episode of `v9_lighting10` goes into any training or
  validation split.
- **Data never goes to the Hub.** Only models and model cards, private repos, via `hackpipe publish`.
- **Colab is run by a human.** Agents write code and read the results the notebook leaves on Drive.
- **Numbers only from logged runs.** Never estimate or invent a result. Unmeasured = `n/a`, never 0.
  Dummy-adapter results are labelled as fake everywhere.
- **IDs as printed.** Model and evaluation IDs are numbered in registration order. Use the ID the command
  printed and write it into STATUS.md. Never assume `m-v1`.
- **Trust the code over these files.** If a command or flag here doesn't exist, check `hackpipe <cmd> --help`
  and the source, use what is there, fix this file in the same PR and note it in STATUS.md. Don't invent flags.
  Steps marked **(to build)** need code first.
- **Tests stay green.** New behaviour gets a test in `tests/`. Contract changes go with `docs/interfaces.md`.
- **Decisions** in `docs/decisions.md` (date, decision, reason). **Never commit** data, outputs, registry or
  checkpoints.
