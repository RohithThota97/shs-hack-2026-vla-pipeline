# R5 · Automated evaluation

**Owner:** Person 4 · Evaluation

Run the same versioned test suite for every model; store every episode; compute metrics with 95% CIs; compare models.

- **Input:** Registered model + `configs/eval/test_suite_vN.yaml` + a simulator adapter.
- **Output:** `outputs/eval/<eval_id>/{episodes.jsonl,metrics.json}`, row in `registry.evals`.

## To do on site

- [ ] Align the test suite and success definition with the organisers' evaluation rules.
- [ ] Freeze test-v1 before evaluating the first model; changes create test-v2.
