# 03 · Evaluate page (R5, R8, option A) — spec for review

## Goal
Run the frozen test suite on a model and show success with 95% confidence intervals, per-scenario results and
failure explanations. Only real simulator results; dummy-adapter results are marked FAKE.

## Panels
1. **Run:** model picker (trained models), suite (frozen, read-only), adapter. A GPU job, with the expected time
   from earlier evaluations (episodes × s/episode).
   - Without a closed-loop simulator adapter yet: an **offline check** on the held-out test set (predicted vs
     recorded actions, error per joint), labelled "offline, not closed-loop success".
2. **Results:** success rate + Wilson CI, seen/unseen, collision/drop/timeout rates (n/a when not measured),
   latency p50/p95.
3. **Episodes:** filter by failure category; each with a video, summary and full episode record.
4. **Compare:** two evaluations on the same suite, with differences called "clear" only when the CIs don't overlap.
5. **Option A:** the success judge's verdict per episode vs the simulator flag, with an agreement rate.

## Backend
`evaluation/runner.py`, `metrics.py`, `explain/failures.py`, `judge/judge.py` (exist). New: an offline-evaluation
job on lighting10.

## As built (23 Oct) — after the team's post-training pipeline (AI-project-work)
- **Offline eval** (`hackpipe eval --model m-vN --offline`, test split = lighting10). Besides the per-sub-task
  prediction error and hold-still baseline:
  - *executed stream*: the first `every` actions of each prediction, as a policy re-planning every `every` steps
    sends them (teacher-forced) → arm error mean / p50 / p95 / p99, per-sub-task error + DTW shape distance
  - gripper events: every crossing of 0.5 in the demo's gripper command, caught within ±1 s, timing error
  - smoothness: joint jump within a prediction vs at re-plan vs the demo
  - error by steps ahead (1 … chunk)
  - robustness (first 3 episodes per source): darker ×0.6 / brighter ×1.4 images, state noise 0.01 / 0.05 rad,
    reworded (the curriculum's paraphrases) and empty instruction — ratio to the unchanged baseline
  - re-plan latency: one prediction with batch 1
  - `--model pending` evaluates every finished model without one; `--no-robustness` skips the variants.
- **Isaac Sim** import: `hackpipe eval --model … --suite configs/eval/sim_handover_v1.yaml --import results.json
  --replay replay.json [--gifs dir]`. The demo replay in the same scene is the sanity check: without a successful
  replay the sim result is "unverified".
- **Verdict** (`hackpipe verdict --model m-vN`, `gates/verdict.py`, thresholds in `configs/gates.yaml` → offline /
  online): DEPLOY · MORE TRAINING (with what to change) · HOLD: SIM UNVERIFIED · NOT EVALUATED. Regression gate
  against the deployed model's offline eval. Written to `outputs/runs/<m>/verdict.json`.
- **Page:** model picker with verdicts; verdict card (to-dos, gate table); tabs Offline (metrics, per sub-task,
  error by steps ahead, robustness bars against the limit, gripper events), Isaac Sim (import form, success with
  95% CI, stage histogram, demo replay pass/unverified, episodes, camera GIFs), Compare models.
- **Auto-evaluate:** Train page → *Evaluate when done* (default on) adds `--then-eval`: the same GPU job runs the
  offline eval after a successful run. *Evaluate all pending* on this page catches up older models.
- Not taken from the team's version: their 10 night84 "held-out" episodes (they are in our train split), Windows
  process handling, LeRobot 0.4.4. Their gate numbers are kept (strict: arm mean ≤ 0.015 rad).
