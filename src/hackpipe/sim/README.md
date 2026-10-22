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
