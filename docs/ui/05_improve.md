# 05 · Improve page (R11, option B) — spec for review

## Goal
Turn evaluation results and data gaps into the next iteration: which demonstrations to add, and which training
change to try. Feeds Data → Add demonstrations.

## Panels
1. **Capability gaps:** from the latest evaluation (failure categories, weak scenarios) and the dataset card's
   known gaps (e.g. one trajectory for all demos, no failure demos, lighting only).
2. **Recommendations:** `hackpipe improve --eval <id>` output, with the reason and expected effect, plus the exact
   next step (e.g. "record N demos with varied box pose", "dataset create --base ds-v1").
3. **Iteration history:** ds-v1 → model_v1 → eval → ds-v2 → model_v2, with the measured change per step
   (only measured numbers).

## Backend
`improve/gaps.py` (exists); data-gap rules from the dataset card.
