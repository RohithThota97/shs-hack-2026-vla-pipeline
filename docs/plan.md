# Plan

## Challenge

Challenge 2 — VLA Robotics for Dynamic Storage and Cathlab Companion. The organisers can record
teleoperation data and train policies, but there is no industrialised path from raw recordings to a served,
versioned model with a trustworthy quality statement; evaluation is manual, binary and not reproducible.

## What the organisers provide

- Team-specific dataset with **500 episodes** (local file share, link + QR + key).
- Team-specific **3D clinical environment with the FR3 Duo** (local file share).
- Detailed instructions: setup, **data schema and evaluation rules** (local file-share web app).
- Listed resources: Microsoft Physical AI Toolchain; Physical Intelligence openpi; NVIDIA GR00T;
  NVIDIA Isaac Sim; MuJoCo.
- Mentors on site; support channel.

## Scope

1. **Mandatory core:** data → quality → versioned dataset → reproducible training → automated evaluation →
   gates → quality statement → deployment in simulation (R1–R7, R9, R10).
2. **Optional modules:** A — Intelligent Success Judgement (`src/hackpipe/judge`), B — Self-Improving
   Pipeline (`src/hackpipe/improve`). C and D only if time remains.

## Model strategy (RTX 4090, 24 GB)

| Version | Model | Start weights | Fits 24 GB? | Role |
|---|---|---|---|---|
| v1 | ACT | from scratch | yes (~2–6 GB) | baseline, safety net |
| v2 | SmolVLA | `lerobot/smolvla_base` | yes (~10–16 GB) | main model (language input) |
| v3 | π0.5 LoRA (openpi) | π0.5 base | tight (openpi: > 22.5 GB for LoRA on an RTX 4090) | stretch, overnight only |
| — | GR00T N1.7 | — | fine-tuning: NVIDIA recommends 40 GB+ | skip |

GPU memory figures: LeRobot Compute HW Guide; openpi README; Isaac GR00T README.

## Day plan

| Day | Goal | Checkpoint |
|---|---|---|
| 1 | Download data + environment, read schema and evaluation rules, ingest + QC, ds-v1, ACT baseline, SmolVLA training overnight | ds-v1 approved, training running with run record |
| 2 | Simulator adapter, test-v1, automated evaluation, model card, judge (A), recommendations (B), m-v2 | two models compared on the same test suite |
| 3 | Gates, deployment in simulation, dashboard, pitch | gated deployment + live run |

## GPU schedule (one RTX 4090)

Never run Isaac Sim and training at the same time. Train at night, evaluate during the day.

| When | GPU job |
|---|---|
| Day 1 afternoon | ACT baseline (short) |
| Day 1 night | SmolVLA fine-tune |
| Day 2 day | Simulator evaluation of v1 models |
| Day 2 night | m-v2 on ds-v2 (or π0.5 LoRA trial) |
| Day 3 | Evaluation + deployment in simulation |
