# R1 · Ingestion

**Owner:** Person 1 · Data

Import the provided episodes as demo records (deterministic UUID per episode, metadata, content checksum).

- **Input:** Dataset folder (LeRobotDataset v3 layout: `meta/info.json`, `data/`, `videos/`).
- **Output:** Rows in `registry.demos`.

## To do on site

- [ ] Check the organisers' data schema; if it is not LeRobot v3, add a converter here that writes LeRobot v3.
- [ ] Fill operator / environment fields if the dataset has them (else defaults from `configs/project.yaml`).
- [ ] `hackpipe ingest --root <dir> --watch` to import new episodes automatically.
