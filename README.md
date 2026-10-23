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

Typical flow (each step writes to the registry in `registry/registry.db`):

```bash
hackpipe ingest  --root data/raw/<dataset>          # R1  import episodes as demo records
hackpipe qc      --root data/raw/<dataset>          # R2  quality checks + report
hackpipe dataset create                              # R3  versioned splits → ds-v1 (draft)
hackpipe dataset approve ds-v1                       # R3  human approval
hackpipe train   --config configs/train/act.yaml --dataset ds-v1        # R4
hackpipe eval    --model m-v1 --suite configs/eval/test_suite_v1.yaml    # R5 (needs sim adapter)
hackpipe gates   --eval <eval_id>                    # R6
hackpipe card    --eval <eval_id>                    # R7  quality statement (model card)
hackpipe improve --eval <eval_id>                    # R11 / option B  next-iteration recommendations
hackpipe deploy  --model m-v1                        # R10 gated deployment record + smoke test
streamlit run dashboard/app.py                       # R9  dashboard
```

## R1–R3 data integrity and readiness

This branch adds data-integrity checks around LeRobot v3 ingestion and quality control. The changes cover
R1 ingestion, R2 quality/readiness reporting, and the R3 guard that prevents dataset-version creation from
using missing, stale, inconclusive, or blocking QC evidence. They do not replace the existing split policy
or constitute a general R3 versioning redesign.

### Ingestion (R1)

- Validate `info.json`, required state/action feature declarations, episode metadata, data references, and
  declared camera references.
- Compute content hashes from source content rather than trusting a path or stale cache. Imports with
  identical content from different source paths are reported as potential duplicate imports.
- Isolate malformed episode records where possible and report the episode and failure reasons instead of
  silently treating them as successful imports.

### Quality and camera evidence (R2)

- Check finite, ordered timestamps and numeric state/action integrity; validate success labels when the
  selected dataset configuration declares one.
- Report missing camera references and decode outcomes explicitly. A file-open test is not considered a
  full decode.
- The QC command can consume a full-decode FFmpeg report with `--camera-report` when PyAV is unavailable.
  It checks the report schema and decode method, dataset root and metadata identity, exact episode/camera
  inventory, and a freshly computed SHA-256 for every referenced video. Missing, stale, incomplete, or
  mismatched evidence cannot satisfy the camera check. Without usable decoder evidence, camera validation
  remains inconclusive and readiness cannot be READY.
- Exact numeric state/action trajectory matches are reported as *potential* duplicates or leakage risks.
  Matching numeric values do not prove the camera demonstrations are visually identical.
- Dataset readiness is reported separately from per-episode quality. Video decode success alone does not
  make a dataset ready; other blockers and review findings remain in effect.

Example PowerShell commands for the three external datasets (the source folders remain in place):

```powershell
& .\.venv-r1r2\Scripts\hackpipe.exe qc `
  --root 'D:\Hackathon Challenges\C2_Teach-the-robot_VLA\Datasets\fr3_duo_rand500\fr3_duo_rand500' `
  --camera-report '.venv-r1r2\camera-verification\cube_stack490.json'

& .\.venv-r1r2\Scripts\hackpipe.exe qc `
  --root 'D:\Hackathon Challenges\C2_Teach-the-robot_VLA\Datasets\fr3_duo_handover_84_ep\fr3_duo_handover_v9_night_84' `
  --camera-report '.venv-r1r2\camera-verification\handover84.json'

& .\.venv-r1r2\Scripts\hackpipe.exe qc `
  --root 'D:\Hackathon Challenges\C2_Teach-the-robot_VLA\Datasets\FR3Duo_Handover_Dataset_lerobot3' `
  --camera-report '.venv-r1r2\camera-verification\handover10.json'
```

The report files shown above are local ignored audit artifacts and are not included in Git. Use a report
produced by the full-decode scanner for the exact current dataset contents; if the dataset changes, rerun
the scan. The pipeline validates the video hashes and metadata before consuming the evidence. If the
reports are unavailable, install/use a supported decoder only through the project's separately managed
environment, or accept that camera QC is inconclusive. Do not copy a PASS field into a report manually.

### Verification snapshot

The read-only full scan recorded 584 episodes and 261 unique referenced video files across the three
datasets; every discovered file completed a full FFmpeg decode with no reported frame-count, duration, FPS,
resolution, or timing discrepancies. Main-pipeline QC subsequently validated current video hashes against
those reports:

| Dataset | Episodes | Unique videos validated | QC readiness | Additional finding |
|---|---:|---:|---|---|
| Cube stacking (`fr3_duo_rand500`) | 490 | 120 | READY at scan time | Separate task; do not combine with handover |
| Handover (`fr3_duo_handover_v9_night_84`) | 84 | 126 | REVIEW | Exact numeric state/action duplicates; independence is unproven |
| Handover (`FR3Duo_Handover_Dataset_lerobot3`) | 10 | 15 | REVIEW | Exact numeric state/action duplicates; independence is unproven |

The handover duplicate groups remain REVIEW even though the videos decoded successfully. These outcomes are
a snapshot, not a promise that data remains unchanged; report validation re-hashes videos on each QC run.
The simulation-derived provenance indicated by handover metadata is not a verified real-world provenance
claim. The existing `ds-v1` manifest and draft registry entry were preserved and were not approved by this
work.

### Focused tests

Run the ingestion, QC, readiness-gate, and common tests in the isolated R1–R2 environment:

```powershell
& .\.venv-r1r2\Scripts\python.exe -m pytest -p no:cacheprovider `
  tests/test_data_pipeline.py tests/test_common.py -q
```

The tests use temporary synthetic datasets for malformed metadata, duplicate imports, numeric trajectory
matches, camera decode/evidence failures, stale hashes, and version-creation safeguards. They do not alter
the three source datasets.

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
