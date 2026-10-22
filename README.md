# Demo-to-Deployment Pipeline · SHS Robotics & AI Hackathon 2026

Challenge 2 — **VLA Robotics for Dynamic Storage and Cathlab Companion**.

> Leading question (organisers): *How can a human demonstration become a deployed, evaluated and explainable
> robot model within 72 hours — and improve with each iteration?*

This repo is our answer: an automated, versioned pipeline from the provided demonstrations (500 episodes,
FR3 Duo, 3D clinical environment) to a fine-tuned VLA that is evaluated, explained, gated and deployed in
simulation — plus optional modules **A (Intelligent Success Judgement)** and **B (Self-Improving Pipeline)**.

```
ingest → quality check → dataset version → train (fine-tune) → register
   ↑                                                            ↓
improve loop ← deploy ← gates + quality statement ← explain ← evaluate
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # pipeline code + tests (no GPU needed)
pytest                             # all tests should pass

# LeRobot is installed separately on the GPU machine (heavy, needs Python 3.12):
#   see docs/runbook.md → "Environment setup"
```

**Console (web UI):** `pip install -e ".[console]"` then `hackpipe console` → http://localhost:8501. Page specs in [`docs/ui/`](docs/ui/README.md).

**Training on free Colab:** open `notebooks/colab_train.ipynb` (data from your Google Drive, never from the Hub).

Typical flow (each step writes to the registry in `registry/registry.db`):

```bash
hackpipe ingest  --root data/raw/<dataset>          # R1  import episodes as demo records
hackpipe qc      --root data/raw/<dataset>          # R2  quality checks + report
hackpipe dataset create                              # R3  versioned splits → ds-v1 (draft)
hackpipe dataset approve ds-v1                       # R3  human approval
hackpipe train   --config configs/train/smolvla_fr3_duo.yaml --dataset ds-v1 [--set steps=200]   # R4
hackpipe publish --model <id> --repo <hf-user>/<name> [--dry-run]       # model + card → private HF repo
hackpipe eval    --model m-v1 --suite configs/eval/test_suite_v1.yaml    # R5 (needs sim adapter)
hackpipe gates   --eval <eval_id>                    # R6
hackpipe card    --eval <eval_id>                    # R7  quality statement (model card)
hackpipe improve --eval <eval_id>                    # R11 / option B  next-iteration recommendations
hackpipe deploy  --model m-v1                        # R10 gated deployment record + smoke test
streamlit run dashboard/app.py                       # R9  dashboard
```

## Folder map and owners

| Folder | Task | Owner | Status |
|---|---|---|---|
| `src/hackpipe/ingest/` | R1 Import provided episodes, demo records, metadata | Person 1 · Data | Working for LeRobotDataset v3; adapt to organiser schema |
| `src/hackpipe/quality/` | R2 Data quality checks, VALID / REVIEW / INVALID, report | Person 1 · Data | Working (video decode check optional) |
| `src/hackpipe/dataset/` | R3 Versioned splits, leakage guard, fingerprint, dataset card | Person 1 · Data | Working |
| `src/hackpipe/training/` | R4 Reproducible `lerobot-train` wrapper, run record, model registration | Person 2 · Training | Working (needs LeRobot on GPU box) |
| `src/hackpipe/sim/` | Adapter between trained policy and the organisers' simulator | Person 3 · Sim & integration | **Stub — implement once the 3D environment is downloaded** |
| `src/hackpipe/evaluation/` | R5 Test-suite runner, episode records, metrics, comparison | Person 4 · Evaluation | Working with any adapter (dummy adapter for tests only) |
| `src/hackpipe/explain/` | R8 Failure categories, episode summaries | Person 4 · Evaluation | Working (rules on logged fields) |
| `src/hackpipe/judge/` | Option A Intelligent success judgement | Person 4 · Evaluation | Interface + state-based baseline; VLM judge to implement |
| `src/hackpipe/gates/` | R6 Quality gates, deployment recommendation | Person 5 · MLOps | Working |
| `src/hackpipe/reporting/` | R7 Trustworthy quality statement (model card) | Person 5 · MLOps | Working |
| `src/hackpipe/improve/` | R11 / Option B Gap analysis, next data + training strategy | Person 5 + Person 1 | Working (heuristic rules) |
| `src/hackpipe/deploy/` | R10 Gated deployment, smoke test, rollback | Person 3 + Person 5 | Record + smoke test working; launch command to fill in |
| `src/hackpipe/common/` | Registry (SQLite), IDs, fingerprints, Wilson CI, environment capture | Person 5 | Working |
| `dashboard/` | R9 Streamlit dashboard over the registry | Person 5 | Working |
| `configs/` | Project, dataset, training, test suite, gates | All | Fill in `CONFIRM` values on Day 1 |
| `docs/` | Plan, roles, interfaces, runbook, decisions, organiser questions | All | Living documents |
| `docker/` | Inference / training container | Person 3 | Draft, build on Day 3 |
| `tests/` | Unit + end-to-end tests on synthetic data | All | `pytest` |

`data/`, `outputs/`, `registry/` and model weights are **git-ignored** — never commit datasets or checkpoints.

## Hand-offs between people

Defined in [`docs/interfaces.md`](docs/interfaces.md): demo record, dataset manifest, run record,
episode result (one JSON line per evaluation episode) and metrics JSON. Change a schema only with a PR that
updates that file.

## Rules

- Work on a branch, open a PR, one reviewer (see [`CONTRIBUTING.md`](CONTRIBUTING.md)).
- Only report numbers that come from logged runs. Results from the dummy adapter are **fake** and are marked
  as such everywhere.
- One heavy GPU job at a time on the RTX 4090 (24 GB). Never run Isaac Sim and training together.

## Docs

- [`docs/plan.md`](docs/plan.md) — goal, scope, model strategy, day plan, GPU schedule
- [`docs/roles.md`](docs/roles.md) — who owns what
- [`docs/interfaces.md`](docs/interfaces.md) — data contracts
- [`docs/runbook.md`](docs/runbook.md) — commands per day
- [`docs/decisions.md`](docs/decisions.md) — decision log
- [`docs/organizer_questions.md`](docs/organizer_questions.md) — open questions and answers
