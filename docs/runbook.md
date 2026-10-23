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

## Isaac Sim on the site PC — diagnose, record, evaluate

The team's `eval_isaac.py` + `serve_policy.py` (github.com/yugalrk/Hack) run the policy in
`Hackathon_FR3Duo_flattened.usd`; our `scripts/isaac/record_demos.py` uses the same scene setup. Both read a
reference file built from an approved version — its eval lighting is lighting10's (held out), never the
team's val episodes (those are in our train split).

```bash
# 0. laptop / Colab: reference file (copy it to the site PC)
hackpipe sim reference --dataset ds-v2 --suite configs/eval/sim_handover_v1.yaml --out eval_assets/ref_ds-v2.json

# A. diagnose the current model (site PC)
python serve_policy.py --checkpoint <ckpt> --selftest                       # offline action error
python serve_policy.py --checkpoint <ckpt> --n_action_steps 5 --port 5555   # 20 / 5 / --temporal_ensemble 0.01
<isaac>/python.sh eval_isaac.py --mode policy --usd <scene> --ref ref_ds-v2.json --episodes 10 --out eval/<run>
<isaac>/python.sh scripts/isaac/record_demos.py --mode snapshot --usd <scene> --ref ref_ds-v2.json --out rec/snap --limit 5
hackpipe sim camera-gap --snapshots rec/snap/snapshots --out camera_gap.png     # sim vs training frames

# B. record new demos (site PC; ~2 GB per 84 episodes after conversion)
<isaac>/python.sh scripts/isaac/record_demos.py --mode render --usd <scene> --ref ref_ds-v2.json --out rec/render
<isaac>/python.sh scripts/isaac/record_demos.py --mode dart --usd <scene> --ref ref_ds-v2.json --out rec/dart --repeats 2
hackpipe sim convert --recordings rec/render --name fr3_duo_handover_sim_render   # → data/incoming/<name>
hackpipe sim convert --recordings rec/dart --name fr3_duo_handover_sim_dart       # failed episodes left out
hackpipe auto --all          # import + QC → draft ds-vN (copies keep their original's split) → approve in the UI
hackpipe subtasks label --base ds-vN --create                                     # labels for the curriculum

# D. evaluation into the registry (95% CI, gates, model card)
hackpipe train --register-only --checkpoint <ckpt> --dataset <ds> --policy-name <name>   # if trained elsewhere
hackpipe eval --model m-vN --suite configs/eval/sim_handover_v1.yaml --import eval/<run>/results.json --gifs eval/<run>
```

`eval --import` refuses runs whose lighting isn't lighting10's, or whose `n_action_steps` differs from the suite's.
