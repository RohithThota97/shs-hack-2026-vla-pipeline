"""Run on the training host with lerobot==0.6.1 and torch installed.

Loads native LeRobot samples and validates action chunks and train-only stats.
Does not download model weights or run model training.
"""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);args=p.parse_args()
    m=json.loads(args.manifest.read_text())
    if not m['episodes']:raise SystemExit('No episodes in this split; an independent training split cannot be created.')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset=LeRobotDataset(m['repo_id'],root=m['dataset_root'],episodes=m['episodes'],delta_timestamps={'action':[i/m['fps'] for i in range(m['chunk_size'])]},video_backend='pyav',tolerance_s=.002)
    expected=json.loads(Path(m['normalization_stats']).read_text())
    for key in ['observation.state','action']:
        if key in expected:
            for stat in ['mean','std']:
                assert np.allclose(dataset.meta.stats[key][stat],expected[key][stat])
    batch=next(iter(DataLoader(dataset,batch_size=2,num_workers=0)))
    assert batch['action'].shape==(2,m['chunk_size'],16)
    assert batch['observation.state'].shape==(2,16)
    for key,value in batch.items():
        if key.startswith('observation.images.'):
            assert value.shape==(2,3,224,224)
        if torch.is_tensor(value) and value.is_floating_point():assert torch.isfinite(value).all()
    print(json.dumps({'status':'PASS','policy':m['policy'],'shapes':{k:list(v.shape) for k,v in batch.items() if torch.is_tensor(v)}},indent=2))

if __name__=='__main__':main()
