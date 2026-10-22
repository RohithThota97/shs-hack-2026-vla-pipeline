# R11 · Option B · Self-improving pipeline

**Owner:** Person 5 + Person 1

Capability gaps by scenario category → targeted data recommendations and training strategy hints.

- **Input:** Evaluation metrics.
- **Output:** `outputs/eval/<eval_id>/recommendations.json`.

## To do on site

- [ ] Tune the heuristics in `STRATEGY_HINTS` on site.
- [ ] Close the loop: pick the recommended episodes/variations → ds-v(N+1) → m-v(N+1) → same test suite.
