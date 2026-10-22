# P3 · Simulation & robot integration (R5 / R10)

> **Not yet updated for the FR3 Duo handover track (decided 22 Oct).** This brief still describes the old
> LIBERO-Object plan (per-object suite, `lerobot-eval`, MuJoCo). Rewrite it before evaluation starts —
> see `agents/README.md`. Don't run the LIBERO commands below.

**Goal:** (A) every evaluation rollout is recorded and linked to its episode record; the simulator interface
and stop conditions are documented; "deployment" really loads and runs the policy, with rollback.
(B) The FR3 Duo interface is understood, and the FR3 Duo policy gets an evaluation path.
**Owner:** Person 3 · Sim & integration. **Where:** code session for code; runner session only for short checks.
**Read first:** `src/hackpipe/sim/adapter.py`, `src/hackpipe/evaluation/` (the `lerobot-eval` bridge and the
episode-result schema), `src/hackpipe/deploy/`, `docs/interfaces.md`, `configs/eval/libero_object.yaml`,
`configs/setups/fr3_duo/`.
**GPU:** short checks only, always through `flock /tmp/hackpipe-gpu.lock`. Never during training.

## A1 · Rollout recordings (before the first full evaluation, Day 2 morning)

1. Find where `lerobot-eval` writes rollout videos and how many episodes it renders (`lerobot-eval --help`;
   look for the option that limits rendered episodes).
2. Make every evaluation keep one video per episode (100) and store the video path in each episode record
   (`episodes.jsonl`), so the dashboard and P4 can open the rollout of any episode.
3. How to pass the setting: agree with P4 **before** the first full evaluation. Either the suite file gets it
   now (only smoke tests have used the suite so far), or it is a `--set` used identically for every model.
   Never edit the suite after the first full evaluation — that creates `libero_object_v2`.
4. Test the import with a small synthetic `eval_info.json` in `tests/`.

## A2 · Interface and stop conditions (Setup 1)

Write `docs/sim_interface.md`, facts from code and config only:
- Observation keys (cameras, resolution, state) and action space (dimension, range, meaning), control frequency.
- Reset: how each episode starts (LIBERO fixed initial states, seed).
- Termination: success signal, maximum steps per episode (find the value LeRobot uses for LIBERO-Object).
- Safety for our own runs: what happens on a non-finite action, and that actions are kept inside the
  dataset's range. If nothing guards this, add a check in our code (not in LeRobot) and note it.

## A3 · Deployment (R10)

1. Fill the launch / smoke test in `src/hackpipe/deploy/`: load the deployed checkpoint as a LeRobot policy
   and run one forward pass on one frame from the ds-v1 validation split; fail clearly if it doesn't load.
   Keep the test runnable without a GPU (mock the policy in `tests/`).
2. Rehearse on the reference model only as a dry run of the mechanics; the data gate must block it.
3. Demo sequence for Day 3 (the human runs deploy): `hackpipe deploy --model <model_v2>` → `hackpipe status`
   shows the live version → `hackpipe rollback` → status shows the previous version.

## A4 · Demo footage (Day 3)

From the evaluation videos pick, per model, one success and one failure on the same object. Copy them to
`outputs/libero_object/demo/` and list them with episode IDs in STATUS.md.

## B · FR3 Duo (after A; Isaac Sim needs the whole GPU)

1. Read the organisers' setup and schema docs. Write `docs/fr3_duo_interface.md`: cameras, state and action
   layout, gripper convention, fps, simulator version, how a scene is reset, how success is defined. The config
   notes (16-dim, left q1..q7 + gripper, right q1..q7 + gripper, gripper 1 = open, 30 fps, Isaac Sim 6.0.1)
   are earlier notes — confirm or correct each.
2. Choose the evaluation path with the human:
   - **Closed loop:** implement `OrganizerSimAdapter.run_episode` — reset with seed, load the policy, map
     observations to policy inputs and actions to the robot, step until success, stop or time limit. Stop
     conditions: time limit, joint limits, workspace box, non-finite action, large jump between consecutive
     joint targets. Save video and trajectory; return one `EpisodeResult`. Needs a GPU slot with nothing
     else running.
   - **Offline fallback:** on the held-out FR3 Duo episodes, predicted vs recorded actions (error per joint)
     plus a replay video. Label it "offline, not closed-loop success" everywhere.
3. Tests for any new code use synthetic data.

## Done when

- [ ] 100 rollout videos per evaluation, each linked from its episode record
- [ ] `docs/sim_interface.md`
- [ ] Deploy smoke test loads and runs the policy; rollback demonstrated
- [ ] Demo clips selected
- [ ] (Stretch) `docs/fr3_duo_interface.md` + one FR3 Duo evaluation path, clearly labelled
- [ ] `pytest -q` green

## Hand-off (STATUS.md)

How videos are kept and where · suite / `--set` decision agreed with P4 · deploy smoke-test result ·
demo clip paths · FR3 Duo path chosen and its status.

## STOP and ask

Any GPU use while training runs · the suite would need to change after a full evaluation · choosing the FR3 Duo
path · organiser docs missing.
