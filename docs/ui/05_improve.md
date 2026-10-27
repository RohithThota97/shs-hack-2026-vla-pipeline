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

## As built (24 Oct) — box grid player + improve loop
- **Why:** the 84 handover demos are one motion with the box in one place (only lighting varies).
- **Grid player** (`improve/grid.py`): box shifted over a table grid (configs/improve.yaml, default 17 × 8 = 136,
  ±8 cm × ±3.5 cm); the left arm's approach / grasp / lift bent by damped-least-squares IK on the FR3
  (`evaluation/kinematics.py`, grasp-point pose = demo orientation, null-space pull to the demo's elbow posture),
  back on the demo before the handover. Valid = IK error ≤ 2 mm / 1°, ≥ 0.05 rad from joint limits, ≤ half the FR3
  joint speed. On the real demo: 136 / 136 valid, worst IK error 0.13 mm, max joint speed 0.89 rad/s (= the demo).
- **Isaac Sim** (site PC): `record_demos.py --mode grid` checks the scene frame first (the demo's grasp point must
  be at the box), moves the box per cell, replays the bent demo with a training episode's lighting, records.
- **Loop** (`improve/loop.py`, `hackpipe loop status|run`, Improve page): plan → reference → render (site PC) →
  convert → import + QC → version with sub-task labels (copies keep their original's split; lighting10 stays the
  test) → auto-approve when every new demo is sim-generated and VALID (`auto_approve_sources`) → fine-tune the best
  model (curriculum, handover + grid) → best checkpoint + test → compare with the best so far. Each run goes on from
  the pipeline's state and stops where it has to wait (recordings, GPU).
