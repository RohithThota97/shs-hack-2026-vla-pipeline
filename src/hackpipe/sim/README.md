# Simulator adapter

**Owner:** Person 3 · Simulation & integration

Connect a trained policy to the organisers' 3D environment and return one `EpisodeResult` per episode.

- **Input:** Model checkpoint + scenario from the test suite + seed.
- **Output:** `EpisodeResult` records, videos, trajectories, stage images.

## To do on site

- [ ] Implement `OrganizerSimAdapter.run_episode` in `adapter.py`.
- [ ] Safe resets and stop conditions (time limit, collision, workspace limits).
- [ ] Never run Isaac Sim and training on the 24 GB GPU at the same time.
- [ ] `DummySimAdapter` is FAKE — only for testing the pipeline.

## Scene geometry (`scene.py`, `geometry/`)

Where things are in the FR3 Duo scene, without Isaac Sim: both arms' base poses, the cart top, and the head
camera (pixel ↔ table plane, box detection → grid cell). Extracted from `Hackathon_FR3Duo_flattened.usd` by our
team's scripts (github.com/yugalrk/Robotics--VLA-training). Franka's DH chain (`evaluation/kinematics.py`) on the
scene's base pose matches the scene's link7 pose to < 1 µm (`tests/test_improve_grid.py`). The box grid
(`improve/grid.py`) uses it to plan in the table plane and to skip positions off the cart.
