# Status board

Every agent reads this first and updates it last. IDs exactly as the CLI printed them.
Newest log entry at the top.

## Pipeline IDs (keep current)

| Item | ID / value | Set by | When |
|---|---|---|---|
| Track | FR3 Duo handover · SmolVLA (decided 22 Oct) | human | 22 Oct |
| LeRobot commit / version | | 00 | |
| Video backend that decodes AV1 | | 00 | |
| Training machine | free Colab T4 · `notebooks/colab_train.ipynb` · state on Drive `MyDrive/shs-hack-2026/state` | human | 22 Oct |
| HF model repo (private) | `RohithThota/smolvla-fr3duo-handover` | human | 22 Oct |
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

### 2026-10-22 · P1/P2 code · R1–R3 to spec, Colab notebook, Hub publishing (branch r1-r3-colab-hub)
- Did: R1 metadata (seed, lighting, layout, outcome, per-file SHA-256, pipeline version, watcher re-runs QC);
  R2 per-camera video checks, final state from outcome records, robot-config dims, redundancy; R3 held-out test
  source, seed leakage guard, frozen test, preprocessing + stats in manifest, richer card; `train --set`,
  `smolvla_fr3_duo.yaml`, model registered before training; `plot_loss.py`; `hackpipe publish`; Colab notebook.
- Result: scratch run on real data (not the real registry): night84 84 VALID, lighting10 10 VALID, 566,244 video
  frames decoded, ds-v1 76/8/10, fingerprint 01a77eac94728a96. **All 94 demos have an identical state/action
  trajectory** — only lighting differs.
- Next / blocked on: HF username; human runs the notebook (AV1 check on Colab, ds-v1 approval, timing run).

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
