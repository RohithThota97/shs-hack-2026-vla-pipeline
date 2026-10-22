# 02 · Train page (R4) — spec for review

## Goal
Start, watch and resume training from the console on the GPU machine (Colab T4 via tunnel), with the expected
time from the timing run, and VRAM and step progress taken from the log.

## Panels
1. **Setup:** approved dataset picker (only approved versions), train config picker (`configs/train/*.yaml`),
   overrides (`--set`, key=value rows). **Dry run** shows the exact `lerobot-train` command (read-only check).
2. **Timing run:** one button, `--set steps=200 --set save_freq=200` as a GPU job. Afterwards it shows seconds per
   step (from the log), peak VRAM (polled `nvidia-smi`), and data-loader share (`data_s` vs `updt_s`).
3. **Step budget:** slot hours → steps = hours × 3600 ÷ s/step, rounded down to `save_freq`. The resulting epochs
   are shown. The value is saved to the config with a reason (decision log).
4. **Full run:** a GPU job with expected time = steps × measured s/step. Live: current step, loss (parsed from the
   log), elapsed, ETA from measured speed, GPU util/VRAM.
5. **Resume:** for models in `training`/`failed` with a checkpoint. Overrides are not allowed (the backend
   enforces it).
6. **Runs:** a table of models (version, dataset, status, steps, wall time, final loss) and a loss curve
   (`loss.csv`). Run records can be compared: a diff where only dataset fields should differ (v1 vs v2).

## Rules
GPU jobs queue on the GPU lock, one at a time. If the machine has no GPU, the buttons are disabled ("open the
console on Colab"). The timing-run model is labelled "timing run, not a result".

## Backend needed
Log parser for step/loss/s-per-step (`training/loss.py` exists; extend with `updt_s`/`data_s`), VRAM poller as part
of the job, `train` jobs with work units = steps.
