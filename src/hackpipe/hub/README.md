# Model publishing · Hugging Face Hub

**Owner:** Person 2 + Person 5

Pushes a trained model and its model card to a Hub **model** repo. The dataset is never uploaded.

- **Input:** a registered model (`hackpipe status`), its run record and dataset manifest; the quality statement
  if the model was evaluated.
- **Output:** private Hub repo with `pretrained_model` files, `README.md` (model card), `run_record.json`,
  `dataset_summary.json`, `loss.csv/png`; `models.meta.hub` in the registry (repo, commit).

```bash
pip install -e ".[hub]" && hf auth login
hackpipe publish --model <model_id> --repo <hf-user>/<name> --dry-run   # check card + file list
hackpipe publish --model <model_id> --repo <hf-user>/<name>             # private by default
```
