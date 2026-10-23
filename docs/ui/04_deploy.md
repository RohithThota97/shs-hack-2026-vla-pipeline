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
