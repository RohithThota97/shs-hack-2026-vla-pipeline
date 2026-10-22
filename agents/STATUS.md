# Status board

Every agent reads this first and updates it last. IDs exactly as the CLI printed them.
Newest log entry at the top.

## Pipeline IDs (keep current)

| Item | ID / value | Set by | When |
|---|---|---|---|
| Track | FR3 Duo handover · SmolVLA (decided 22 Oct) | human | 22 Oct |
| LeRobot commit / version | | 00 | |
| Video backend that decodes AV1 | | 00 | |
| Train data path (night84) | `data/raw/fr3_duo_handover_84` | 00 | |
| Held-out test path (lighting10) | `data/raw/fr3_duo_handover_lighting10` | 00 | |
| ds-v1 (approved by) | | P1 | |
| Timing run model (not a result) | | P2 | |
| model_v1 | | P2 | |
| Eval of model_v1 | | P4 | |
| ds-v2 (approved by) | | P1 | |
| model_v2 | | P2 | |
| Eval of model_v2 | | P4 | |
| Deployed model | | human | |

## GPU now

| Job | Started | Expected end | tmux session | Log |
|---|---|---|---|---|
| | | | | |

## Open questions for humans

- Organiser questions Q11–Q15 (see `agents/TASKS_to_training.md` T0) — esp. success definition and whether a
  sim controller / policy interface exists.
- Who runs the GPU box, and is `lerobot/smolvla_base` downloadable there (internet)?

## Log

### 2026-10-22 · planning · Track switched to FR3 Duo handover + SmolVLA
- Did: reviewed data + briefs; rewrote README, 00_setup, P1, P2 for the FR3 Duo handover track; banner on P3–P6.
- Result: train = night84 (84 ep), held-out test = lighting10 (10 ep), rand500 unused.
- Next / blocked on: 00_setup on the GPU box; P1 configs; P2 `train --set` + `smolvla_fr3_duo.yaml`.

<!-- Copy this block for each entry, newest first.
### YYYY-MM-DD HH:MM · <role> · <one line>
- Did:
- Result (IDs, paths):
- Next / blocked on:
-->
