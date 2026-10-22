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
p.add_argument("--log-freq", type=int, help="default: from the model's run record")
a = p.parse_args()
run_dir = resolve(project_config()["project"]["outputs_dir"]) / "runs" / a.model
log_freq = a.log_freq
if log_freq is None and (run_dir / "run_record.json").exists():
    log_freq = ((json.loads((run_dir / "run_record.json").read_text()).get("train_config") or {})
                .get("args") or {}).get("log_freq")
print(json.dumps(export(a.log, run_dir, log_freq), indent=2))
