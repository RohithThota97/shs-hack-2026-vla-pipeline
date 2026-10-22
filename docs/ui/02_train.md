# 02 · Train page (R4)

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

## As built (23 Oct)
- **Tabs:** Setup & dry run · Timing run · Step budget · Full run (+ live progress, resume) · Runs (table, loss
  curve, run-record diff).
- **Seconds per step:** from LeRobot's log timestamps (the first, warm-up line is skipped). **Peak memory:**
  LeRobot's own `mem_gb`. **Data-loading share:** `data_s ÷ (data_s + updt_s)`. **Start-up:** first log line −
  job start − first steps.
- **Steps:** LeRobot prints them without decimals above 1000 (`step:2K`), so they are rebuilt from `log_freq`.
- **Timing run:** `hackpipe train --purpose timing`; the model is labelled "timing run, not a result" in the
  registry.
- **Full-run expected time:** start-up + steps × measured s/step, from the latest finished timing run.
- **No GPU** (laptop): dry run only; the GPU buttons are disabled.
- **Before registering a model,** `hackpipe train` checks that `lerobot-train` starts, so a broken install doesn't
  use up a model ID.

## Opening it on Colab
Notebook step **3b · Console** starts `hackpipe console` with a fresh access token and a `cloudflared` quick tunnel,
and prints `https://….trycloudflare.com/?token=…`. The link is public, so share it only with the team and stop
the tunnel when done. On Colab the registry writer is `colab`.

## Verified
- Training command: a 2-step run of the exact command on the laptop CPU with the real ds-v1 data (LeRobot 0.6.1)
  finished with exit 0, loss logged, and a checkpoint with config, weights and normalizer saved.
- The parser reads that real log (61 s/step on CPU), and the page shows those numbers.
