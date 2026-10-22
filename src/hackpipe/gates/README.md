# R6 · Quality gates

**Owner:** Person 5 · MLOps

Data, training and evaluation gates → deploy / deploy with supervision / do not deploy.

- **Input:** Evaluation in the registry + `configs/gates.yaml`.
- **Output:** `outputs/eval/<eval_id>/gates.json`.

## To do on site

- [ ] Agree thresholds on Day 1 (they are team targets).
- [ ] Set `max_latency_p95_ms` once the action-chunk duration is known.
