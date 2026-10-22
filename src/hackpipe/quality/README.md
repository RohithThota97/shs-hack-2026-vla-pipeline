# R2 · Data quality

**Owner:** Person 1 · Data

Per-episode checks → VALID / REVIEW / INVALID; data quality report (JSON + Markdown).

- **Input:** Dataset folder + `configs/dataset.yaml` (column names, thresholds).
- **Output:** `outputs/qc/qc_report.{json,md}`, demo statuses in the registry.

## To do on site

- [ ] Set column names to the organisers' schema in `configs/dataset.yaml`.
- [ ] Set `columns.success` if the dataset has a success/done flag.
- [ ] Review REVIEW episodes with `lerobot-dataset-viz` and approve/reject.
- [ ] Install `.[video]` (PyAV) for the video decode check.
