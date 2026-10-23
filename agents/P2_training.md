# P2 · Model training (R4)

**Goal:** `model_v1` = SmolVLA fine-tuned on `ds-v1` (FR3 Duo handover) with a complete run record and
training metrics. Later `model_v2` on `ds-v2` with the identical config and step count.
**Owner:** Person 2 · Training. **Where:** code session for §0–1; from §2 the GPU run happens in
`notebooks/colab_train.ipynb` on free Colab (T4, 16 GB), run by a human — or a runner session on a GPU box.
**Read first:** `configs/train/smolvla.yaml`, `src/hackpipe/training/train.py`, `docs/interfaces.md`
(run record), `docs/data_schema_fr3_duo.md` (from P1), the `config.json` of `lerobot/smolvla_base`.
**Needs:** for §2+: `ds-v1` approved (`hackpipe status`). **GPU: yes, hours.**

## 0 · Model choice note (no GPU, 20 min)

Write `docs/model_choice.md` (half a page), facts from config files and docs only:
- Starting point `lerobot/smolvla_base`; how our 3 cameras and 16-dim state/action map to its inputs
  (its expected image keys, `max_state_dim`, `max_action_dim`, `chunk_size`, `n_action_steps`).
- Why SmolVLA: language-conditioned, fits 24 GB, native LeRobot support, reads our LeRobot v3 data directly.
- What it doesn't solve: one task only, so language adds little now; long episodes (2247 frames ≈ 75 s);
  no failure demos.
- ACT = baseline candidate. π0.5 / GR00T = not on one 24 GB GPU (`docs/decisions.md`).

## 1 · Code gaps (code session, before any GPU run) — **done 22 Oct**

`hackpipe train --set key=value`, `configs/train/smolvla_fr3_duo.yaml` (T4-sized: batch 8, save every 1000,
2 workers), the camera `rename_map` in `configs/dataset.yaml` (→ `camera1..3`, checked against
`lerobot/smolvla_base`'s config.json), model registered before the run (resumable), `scripts/plot_loss.py`,
`hackpipe publish`. Tests in `tests/test_r1_r3.py` and `tests/test_publish.py`. What the config contains:

1. `--set` overrides are stored in the run record.
2. `configs/train/smolvla_fr3_duo.yaml`:
   - `policy.path: lerobot/smolvla_base`, `policy.device: cuda`, `policy.push_to_hub: false`
   - camera mapping: `preprocessing.rename_map` in `configs/dataset.yaml` (the wrapper passes it as
     `--rename_map`) from `observation.images.external_camera` / `left_wrist_camera` / `right_wrist_camera`
     to the image keys smolvla_base expects. Check the keys in its `config.json`; don't guess.
   - confirm 16 ≤ `max_state_dim` and `max_action_dim` (padding is automatic).
   - `wandb.enable: false` unless the box has internet and the team wants it.
   - `batch_size`, `num_workers`, `save_freq`: start from `smolvla.yaml`, adjust after timing.
   - `steps`: placeholder, set in §4.
3. Extend the dry-run test so it builds the full `lerobot-train` command from the new config.

## 2 · Dry run

```bash
hackpipe train --config configs/train/smolvla_fr3_duo.yaml --dataset ds-v1 --dry-run
```

Check the printed `lerobot-train` command line by line: `--dataset.repo_id`, `--dataset.root`
(absolute path ending in `data/raw/fr3_duo_handover_84`), `--dataset.episodes` = the 76 train episodes only
(no val, nothing from lighting10),
`--rename_map`, policy path, batch size, steps, save frequency, seed, output dir, logging target.
Show it to the human.

## 3 · Timing run (about 15 min)

Console alternative (Colab, notebook step 3b): Train → Timing run → Step budget → Full run. The same commands run
as GPU jobs, and the page shows s/step, peak memory and data-loading share measured from the log. Commands below
are the CLI equivalent (add `--purpose timing` so the model is labelled "timing run, not a result").

```bash
mkdir -p outputs/logs
nvidia-smi --query-gpu=memory.used --format=csv -l 10 > outputs/logs/vram_timing.csv &
flock /tmp/hackpipe-gpu.lock hackpipe train --config configs/train/smolvla_fr3_duo.yaml --dataset ds-v1 \
  --set steps=200 --set save_freq=200 2>&1 | tee outputs/logs/train_timing.log
kill %1
```

- Read seconds/step from the log and peak VRAM from the CSV.
- Watch GPU utilisation: if it is low, AV1 decoding is the bottleneck → raise `num_workers`, try the other
  video backend, and note it.
- Out of memory → `--set batch_size=4` and repeat (the config starts at 8 for the T4).
- T4 has no bf16: if SmolVLA's VLM runs in bf16 it is emulated and slow — the timing run shows it.
- The timing run also gets a model ID: write it into STATUS.md as "timing run, not a result".

## 4 · Set the step count

Ask the human when the training slot must end.
`steps = (slot hours × 3600) / seconds per step`, rounded down to a multiple of `save_freq`.
For reference, report the resulting epochs: `steps × batch_size / train frames` (train frames from the
dataset card). Set `steps` in `configs/train/smolvla_fr3_duo.yaml`, commit, record the reason in
`docs/decisions.md`. This value is now fixed for model_v2 too.

## 5 · Full run (overnight, tmux)

```bash
tmux new -d -s train_v1 'cd ~/shs-hack-2026-vla-pipeline && source .venv/bin/activate && \
  flock /tmp/hackpipe-gpu.lock hackpipe train --config configs/train/smolvla_fr3_duo.yaml --dataset ds-v1 \
  2>&1 | tee outputs/logs/train_v1.log'
```

Wait until the first loss lines appear and the run folder exists, then hand off. Don't block the session for
hours. Write into STATUS.md: tmux session, log path, expected end, and how to check:
`tmux attach -t train_v1`, `tail -f outputs/logs/train_v1.log`, `nvidia-smi`.

After a crash: `hackpipe train --config configs/train/smolvla_fr3_duo.yaml --dataset ds-v1 --resume <model_id>`.

## 6 · After training

- `outputs/runs/<model>/run_record.json` has: exact command, config, overrides, dataset ID and fingerprint,
  seed, git commit, packages, GPU. Fill any gap in the code (with a test).
- Training metrics: `python scripts/plot_loss.py --log outputs/logs/train_v1.log --model <model>` →
  `outputs/runs/<model>/loss.csv` + `loss.png`.
- Checkpoint folder `outputs/train/<model>/checkpoints/` contains the policy config and weights.
- Quick sanity check (no simulator): load the last checkpoint, run it on a few validation frames, compare
  predicted vs recorded actions (mean abs error per joint). Label it "offline sanity check, not a success rate".

## 6b · Model + model card → Hugging Face (private)

`hackpipe publish --model <model> --repo <hf-user>/<name> --dry-run` → read `outputs/hub/<model>/README.md`,
then without `--dry-run`. Uploads checkpoint, model card, run record, dataset summary (IDs/counts, no data),
loss curve; refuses data-like files; private unless `--public`. The card says "not evaluated" until P4 has an
evaluation — then publish again. Never upload the dataset.

## 7 · model_v2 (later)

Same command with `--dataset ds-v2`. No other change: same config file, same steps, same seed.
Afterwards compare the two run records and confirm that only dataset fields differ. Write that into STATUS.md.

## Optional · ACT baseline

Only if a GPU slot is free and the team agrees. Same ds-v1. Label it "baseline, no language input".

## Done when

- [ ] `docs/model_choice.md`
- [x] `train --set` and `smolvla_fr3_duo.yaml` merged with tests
- [ ] Dry-run command checked by a human
- [ ] Timing: seconds/step, peak VRAM, batch size, data-loader bottleneck (yes/no) recorded
- [ ] model_v1 trained; run record complete; `loss.csv` + `loss.png`; offline sanity check
- [ ] model_v1 + card published to the private Hub repo (HF username from the team)
- [ ] (later) model_v2 trained with identical settings; run-record diff checked

## Hand-off (STATUS.md)

model ID(s) · steps · epochs · seconds/step · peak VRAM · training wall time · final loss · checkpoint path ·
offline action error · run-record diff result (v2).

## STOP and ask

OOM at batch size 8 · the slot allows fewer steps than the team expects · GPU already busy · a crash that
resume doesn't fix · `smolvla_base` can't be downloaded.
