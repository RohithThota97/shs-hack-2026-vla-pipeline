# Runbook

## Environment setup (GPU machine, Day 1)

```bash
# Pipeline package
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dashboard,video]"
pytest

# LeRobot (Python 3.12) — see https://huggingface.co/docs/lerobot/installation
git clone https://github.com/huggingface/lerobot.git ../lerobot
pip install -e "../lerobot[training,smolvla]"
conda install -y ffmpeg -c conda-forge   # if using conda; or a system ffmpeg with PyTorch >= 2.10

nvidia-smi                               # confirm RTX 4090, 24 GB
```

openpi (π0.5 stretch) is tested on Ubuntu 22.04 — check the OS first.

## Day 1

```bash
# 1. Download dataset + environment from the organisers' file share into data/raw/ (never commit them)
# 2. Fill CONFIRM values in configs/project.yaml and configs/dataset.yaml from the data schema
hackpipe ingest --root data/raw/provided_dataset
hackpipe qc --root data/raw/provided_dataset          # read outputs/qc/qc_report.md
hackpipe dataset create
hackpipe dataset approve ds-v1
# 3. Set `steps` in configs/train/*.yaml from total_frames (meta/info.json)
hackpipe train --config configs/train/act.yaml --dataset ds-v1 --dry-run
hackpipe train --config configs/train/act.yaml --dataset ds-v1
hackpipe train --config configs/train/smolvla.yaml --dataset ds-v1   # overnight
```

Resume after a crash: `hackpipe train --dataset ds-v1 --resume m-v2`.

## Day 2

```bash
# Person 3: implement OrganizerSimAdapter (src/hackpipe/sim/adapter.py)
hackpipe eval --model m-v1 --suite configs/eval/test_suite_v1.yaml
hackpipe gates --eval <eval_id>
hackpipe card --eval <eval_id>
hackpipe improve --eval <eval_id>
streamlit run dashboard/app.py
```

Pipeline test without the simulator (FAKE numbers): `hackpipe eval --model m-v1 --adapter dummy`.

## Day 3

```bash
hackpipe deploy --model m-v2
hackpipe rollback            # demo the rollback
hackpipe status
```
