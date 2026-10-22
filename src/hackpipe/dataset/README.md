# R3 · Dataset versioning

**Owner:** Person 1 · Data

Versioned splits by episode with fixed seed, leakage guard, fingerprint, dataset card.

- **Input:** VALID demos in the registry + `configs/dataset.yaml`.
- **Output:** `outputs/datasets/ds-vN/{manifest.json,dataset_card.md}`, row in `registry.datasets` (draft → approved).

## To do on site

- [ ] Decide `group_key` / `test_episodes` so the test split holds genuinely unseen variations.
- [ ] Approve with `hackpipe dataset approve ds-vN` — training refuses unapproved datasets.
