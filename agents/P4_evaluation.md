# P4 · Evaluation & quality (R5–R8, Option A)

> **Not yet updated for the FR3 Duo handover track (decided 22 Oct).** This brief still describes the old
> LIBERO-Object plan (per-object suite, `lerobot-eval`, MuJoCo). Rewrite it before evaluation starts —
> see `agents/README.md`. Don't run the LIBERO commands below.

**Goal:** every model is evaluated on the same frozen suite (100 simulator rollouts), with per-object success
and 95% confidence intervals, failure information, and a report comparing model versions.
**Owner:** Person 4 · Evaluation. **Where:** runner session for evaluations; code session for code.
**Read first:** `docs/setup1_libero.md` § 4, `configs/eval/libero_object.yaml`,
`src/hackpipe/{evaluation,explain,judge,reporting}/`, `docs/interfaces.md` (episode result, metrics),
`docs/jury_qa.md` § 3–4. Ask the human for the organisers' evaluation rules (file-share docs).
**GPU:** yes for evaluations (~ smoke-test time × 10 each, see STATUS.md).

## 1 · Metric map (now, no GPU)

Write `docs/eval_metrics_map.md`: one row per metric the organisers or the role card ask for — success rate,
collision rate, task duration, failure categories, latency, robustness.

| Metric | Asked by | Measured in Setup 1? | Source field | Shown as |
|---|---|---|---|---|

Anything the simulator doesn't report is `n/a` with the reason, never 0. Put the organisers' answer on
"what counts as success" into `docs/organizer_questions.md`.

## 2 · Freeze the suite (before the first full evaluation)

- `task_names` filled, seed fixed, 10 episodes per object → 100 rollouts.
- Agree with P3 how rollout videos are kept (suite file now, or one fixed `--set` for every model).
- After the first full evaluation the file is frozen. Any change = new suite version.

## 3 · Evaluate model_v1 (Day 2 morning)

```bash
tmux new -d -s eval_v1 'cd ~/shs-hack-2026-vla-pipeline && source .venv/bin/activate && \
  export HACKPIPE_SETUP=libero_object MUJOCO_GL=egl && \
  flock /tmp/hackpipe-gpu.lock hackpipe eval --model <model_v1> --suite configs/eval/libero_object.yaml \
  2>&1 | tee outputs/libero_object/logs/eval_v1.log'
```

Then `hackpipe card --eval <eval_id>`. Check: 100 rollouts, per-object rate with Wilson CI, the exact
`lerobot-eval` command stored, `fake_results` false.
Reference model (optional, only in a free GPU slot): same suite, always labelled external.

## 4 · Task duration

Check what `eval_info.json` holds per episode. If it has episode length, report steps (and seconds, if the
control rate is known) for successful episodes. If not, `n/a`.

## 5 · Failures (R8 + Option A)

LIBERO reports success only, so failure causes come from the videos. Never invent causes.
1. **Human labels:** for every failed episode, a person watches the video and picks one stage: wrong object ·
   missed grasp · dropped · didn't reach basket · timeout/stalled · other. Store
   `outputs/libero_object/eval/<eval_id>/failure_labels.csv` (episode ID, label, labeller). Add a small
   labelling view to the dashboard if quick; otherwise a CSV is fine. The card shows "human-labelled, n = …".
2. **Success judge (stretch):** run a vision-language model on rollout frames to predict success and failure
   stage. Report its agreement with the simulator's success and with the human labels (`agreement()` in
   `judge/`). It runs only when the GPU is free (or on CPU). Judge output is never presented as ground truth.

## 6 · Compare model versions (Day 3 morning)

1. Evaluate model_v2 exactly like step 3: same suite, same `--set` values.
2. `hackpipe card --eval <eval_v2>` → comparison table per object, "clear" only where the CIs don't overlap,
   otherwise "within noise". Overall rate over 100 rollouts with its CI.
3. Plot per-object success with CI bars for both models: `scripts/plot_compare.py` →
   `outputs/libero_object/reports/compare_<v1>_<v2>.png`. Numbers come from `metrics.json`, not typed in.

## Done when

- [ ] `docs/eval_metrics_map.md`
- [ ] Suite frozen; model_v1 and model_v2 evaluated with identical settings
- [ ] Quality statements generated for both
- [ ] Failure labels for every failed episode of both models
- [ ] Comparison plot; "clear" vs "within noise" stated per object
- [ ] (Stretch) judge agreement reported
- [ ] `pytest -q` green

## Hand-off (STATUS.md)

eval IDs · overall success with CI per model · weakest objects · failure label counts · plot path.

## STOP and ask

The suite would change after a full evaluation · GPU busy · evaluation estimate longer than the free slot ·
organisers' success definition differs from LIBERO's.
