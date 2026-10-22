# 00 · Environment (Colab or GPU box)

**Goal:** a GPU machine runs the pipeline, LeRobot can read the FR3 Duo handover datasets (AV1 video),
and the data is in place for P1 and P2.

**Default: free Google Colab (T4, 16 GB)** — `notebooks/colab_train.ipynb` does every step below (install,
registry/outputs on Drive, data from Drive, AV1 check). A human runs it; agents prepare code and read results.
The steps below are for a dedicated GPU box (e.g. the RTX 4090), if one becomes available.

**Where:** GPU box, runner session, `~/shs-hack-2026-vla-pipeline`.
**Read first:** `agents/README.md`, `docs/runbook.md` ("Environment setup"),
`data/C2_Teach-the-robot_VLA/Environment/…/isaac_sim_hackathon_environment.md` (in the environment zip).

## Steps

1. **Machine checks.** `nvidia-smi` (RTX 4090 visible), `free -g`, `df -h ~` (datasets ~2.4 GB unzipped,
   checkpoints several GB each — want ≥ 60 GB free). Install `tmux` if missing: `sudo apt install -y tmux`.
   If the box is Windows: work inside WSL2 and keep the repo on the Linux filesystem (`~`), not `/mnt/c`.
2. **Repo.** `~/shs-hack-2026-vla-pipeline` must exist and be a git repo with at least one commit.
   If it doesn't exist: STOP and ask where the zip or GitHub remote is.
3. **Python env.** `python3 -m venv .venv && source .venv/bin/activate && pip install -U pip &&
   pip install -e ".[dev,dashboard,video,hub]" && pytest -q` → all tests pass. Note the count.
4. **LeRobot side by side**, not inside the repo:
   ```bash
   cd ~ && git clone https://github.com/huggingface/lerobot.git && cd lerobot
   pip install -e ".[smolvla]"
   git -C ~/lerobot rev-parse HEAD
   pip freeze | grep -i -E "lerobot|torch|torchcodec|av"
   ```
   The datasets were written with **lerobot 0.6.1** (`meta/fr3_conversion.json`). Use that version or newer;
   if `main` breaks reading them, check out the `v0.6.1` tag. Write the commit and the versions into
   `docs/decisions.md`.
5. **Copy the data** from the shared drive / laptop (never via git) and unzip so that these exist:
   ```
   data/raw/fr3_duo_handover_84/meta/info.json        # from fr3_duo_handover_84_ep.zip → fr3_duo_handover_v9_night_84/
   data/raw/fr3_duo_handover_lighting10/meta/info.json # from FR3Duo_Handover_Dataset_lerobot3.zip
   ```
   Check: `total_episodes` 84 and 10, `fps` 30, `codebase_version` v3.0.
6. **AV1 decode check (likely blocker — do it now).** All videos are AV1 (`libsvtav1`). Load one frame
   through LeRobot:
   ```bash
   python -c "
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   ds = LeRobotDataset('local/fr3_duo_handover_84', root='data/raw/fr3_duo_handover_84')
   x = ds[0]; print({k: getattr(v, 'shape', v) for k, v in x.items()})"
   ```
   (Import path may differ by LeRobot version — check `lerobot/datasets/`.) Expect three image tensors
   3×224×224, state and action of 16. If decoding fails: try the other video backend
   (`video_backend='pyav'` / `'torchcodec'`), then an ffmpeg build with `libdav1d`.
   Also time 200 random `ds[i]` reads — that is the data-loader ceiling for training.
7. **Check the box** — `scripts/check_env.sh` **(to build, see TASKS T3)**. Until it exists, the checks above
   are the check. (Colab: notebook steps 1–5 are the check.)
8. **No reference model, no smoke-test rollout.** There is no simulator policy interface yet (P3). Skip the
   old LIBERO steps.

## Done when

- [ ] `pytest -q` green, count noted
- [ ] LeRobot commit + package versions in `docs/decisions.md`
- [ ] Both handover datasets under `data/raw/`, counts match `info.json`
- [ ] One sample loaded through LeRobot with all three cameras decoded; reads/second noted

## Hand-off (STATUS.md)

LeRobot commit · torch/torchcodec/av versions · video backend that works · dataset paths · sample reads/s ·
any warning left open.

## STOP and ask

Repo missing · GPU not visible · AV1 can't be decoded after 45 min · less than 60 GB free disk ·
no internet for `lerobot/smolvla_base` (then the weights must be copied in).
