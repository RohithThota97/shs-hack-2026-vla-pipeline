# R4 · Reproducible training

**Owner:** Person 2 · Training

Wrap `lerobot-train`: full command, config, dataset fingerprint, git commit, packages and GPU saved in a run record; model registered as m-vN.

- **Input:** Approved dataset version + `configs/train/*.yaml`.
- **Output:** `outputs/train/m-vN/` (LeRobot checkpoints), `outputs/runs/m-vN/run_record.json`, row in `registry.models`.

## To do on site

- [ ] Set `steps` from total_frames (steps = epochs × total_frames / batch_size).
- [ ] Use `--dry-run` first to check the command.
- [ ] Resume after a crash: `hackpipe train --dataset ds-vN --resume m-vN`.
- [ ] pi0.5 via openpi: train there, then `hackpipe train --register-only --checkpoint <dir> --dataset ds-vN --policy-name pi05_lora`.
