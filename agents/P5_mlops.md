# P5 · MLOps & continuous improvement (R6, R9–R11, Option B)

> **Not yet updated for the FR3 Duo handover track (decided 22 Oct).** This brief still describes the old
> LIBERO-Object plan (per-object suite, `lerobot-eval`, MuJoCo). Rewrite it before evaluation starts —
> see `agents/README.md`. Don't run the LIBERO commands below.

**Goal:** quality gates decide deploy / supervised / do not deploy; every model has a quality statement; the
dashboard compares versions; the improve step turns evaluation results into ds-v2; the second iteration is
measured; deploy and rollback work.
**Owner:** Person 5 · MLOps. **Where:** code session for code; runner session for `gates` / `card` / `improve`.
**Read first:** `configs/gates.yaml`, `src/hackpipe/{gates,reporting,improve,deploy,common}/`,
`dashboard/app.py`, `docs/jury_qa.md` § 4–5, `docs/interfaces.md`.
**No GPU** (except what the dashboard starts).

## 1 · Gates config (now, before the first gates run)

Agree with the team and set in `configs/gates.yaml`: target success rate (a team target, not a standard),
minimum rollouts (100), data gate (approved dataset, not external), run record present, real (not fake)
results, not worse than the deployed model on the same suite. Record the values and reason in
`docs/decisions.md`. Later changes are logged there too.

## 2 · One GPU lock for the dashboard (now)

Dashboard buttons that start training, evaluation or Isaac Sim must run their CLI command through
`flock /tmp/hackpipe-gpu.lock`, the same lock the agents use. Don't add a lock inside the CLI as well — two
locks on the same file from parent and child would deadlock. Test that the built command has the prefix.

## 3 · Dashboard comparison (R9)

A page (or section) comparing model versions on the same suite: per-object success with CIs side by side,
overall rate, dataset version and fingerprint, code commit, eval ID, gate result, and a banner for external or
fake results. Show the rollout video of a selected episode (P3 stores the path) and the failure labels (P4).
Keep logic in testable functions; add tests.

## 4 · Traceability

`hackpipe status` (and the Overview page) shows, for every model: dataset version → fingerprint → run record
→ evaluations → gates → deployment. Fix any gap; test it.

## 5 · Iteration 1 → 2 (Day 2)

After P4 hands off the eval of model_v1:
```bash
hackpipe gates   --eval <eval_v1>
hackpipe card    --eval <eval_v1>
hackpipe improve --eval <eval_v1>
```
`improve` lists objects below target, held-back demos per object, and the exact `dataset create --base ds-v1
--boost ...` command. Put that command into STATUS.md for P1. The human approves ds-v2; P2 trains model_v2 with
the same config.

## 6 · Decide and deploy (Day 3)

After P4 hands off the eval of model_v2: `gates` and `card` again, then the comparison.
- Gates pass → **STOP**: the human runs `hackpipe deploy --model <model_v2>`; then rehearse `hackpipe rollback`
  and deploy again.
- Gates don't pass → the statement says so. `--allow-supervised` only on a human decision. Never loosen a gate
  to make a model pass.

## 7 · Repo and CI

`pytest -q` green on main; GitHub Actions green. Ask the human for the remote and push. Nothing from `data/`,
`outputs/`, `registry/` or checkpoints in git (`git status` before every commit).

## 8 · Reproducibility proof (optional)

From model_v1's run record, regenerate the training command (`--dry-run`) and diff it against the stored one.
Show the empty diff in the pitch.

## Done when

- [ ] Gates agreed and logged
- [ ] Dashboard GPU buttons use the lock (tested)
- [ ] Comparison view with CIs, video and labels
- [ ] `hackpipe status` shows the full chain for every model
- [ ] Improve output → ds-v2 command handed to P1
- [ ] Gates + statement for model_v2; deploy and rollback rehearsed
- [ ] CI green, repo pushed

## Hand-off (STATUS.md)

Gate thresholds · gate result per model · improve command · deploy / rollback result · dashboard pages ready
for the demo.

## STOP and ask

Before `deploy` (always) · before changing a gate after results exist · before pushing to a remote.
