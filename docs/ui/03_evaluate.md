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
