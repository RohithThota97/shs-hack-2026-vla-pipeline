"""Export loss vs step from a lerobot-train log into the model's run folder.

    python scripts/plot_loss.py --log outputs/logs/train_v1.log --model <model_id>
"""
import argparse
import json

from hackpipe.common.config import project_config, resolve
from hackpipe.training.loss import export

p = argparse.ArgumentParser()
p.add_argument("--log", required=True)
p.add_argument("--model", required=True, help="model ID as printed by hackpipe train")
a = p.parse_args()
print(json.dumps(export(a.log, resolve(project_config()["project"]["outputs_dir"]) / "runs" / a.model), indent=2))
