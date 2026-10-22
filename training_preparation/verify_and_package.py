"""Verify snapshots, calculate reports, and assemble actual camera/state/action batches.

This is a data-level smoke test, not a PyTorch/LeRobot model forward pass.
"""
import csv, json, sys
from pathlib import Path
from pipeline import HERE, load, dump, digest, fingerprint, stats
import numpy as np
import pyarrow.parquet as pq
import av

def frame_at(path,t):
    with av.open(str(path)) as c:
        stream=c.streams.video[0]
        c.seek(max(0,int((t-.2)/float(stream.time_base))),stream=stream)
        best=None
        for f in c.decode(stream):
            error=abs(float(f.time)-t)
            if best is None or error<best[0]: best=(error,f.to_ndarray(format='rgb24'))
            if float(f.time)>t+.01: break
        assert best is not None and best[0]<.002,(path,t,best[0] if best else None)
        return best[1].transpose(2,0,1)

def main():
    output=[]
    for summary in load(HERE/'summary.json'):
        root=Path(summary['snapshot']);label=root.name.split('-ds-')[0]
        info=load(root/'meta/info.json')
        eps=pq.read_table(sorted((root/'meta/episodes').rglob('*.parquet'))).to_pylist()
        epmap={e['episode_index']:e for e in eps}
        data=pq.read_table(sorted((root/'data').rglob('*.parquet'))).to_pydict()
        ids=np.array(data['episode_index']);cameras=[k for k,v in info['features'].items() if v['dtype']=='video']
        report=HERE/'reports'/label
        records=load(report/'episodes.json')
        matrices={k:np.asarray(data[k],dtype=float) for k in ['observation.state','action']}
        for record in records:
            mask=ids==record['episode_index']
            record['feature_statistics']={k:stats(matrices[k][mask]) for k in matrices}
        dump(report/'episodes.json',records)
        with (report/'episodes.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['demo_id','episode_index','status','frames','duration_s','instruction','invalid_reasons','review_reasons'])
            w.writeheader();w.writerows({k:json.dumps(r[k]) if isinstance(r[k],list) else r[k] for k in w.fieldnames} for r in records)
        dump(report/'aggregate_statistics.json',{'features':{k:stats(np.asarray(data[k],dtype=float)) for k in ['observation.state','action']},'duration_s':stats(np.asarray([[r['duration_s']] for r in records])),'counts_by_task':{t['task']:len(records) for t in summary['tasks']},'counts_by_layout':{'UNKNOWN':len(records)},'normalization_use':'report only; full-dataset statistics must never be used for training'})
        original=Path(summary['source']); inventory=load(report/'file_inventory.json')
        verified=0
        for rel,entry in inventory.items():
            if rel in ['meta/info.json','meta/stats.json'] or rel.endswith('norm_stats.json'): continue
            assert digest(root/rel)==entry['sha256'],rel
            verified+=1
        split=load(root/'preparation/candidate_splits.json')
        trainstats=load(root/'meta/stats.json')
        for k in ['observation.state','action']:
            if split['train']:
                a=np.asarray(data[k],dtype=float)[np.isin(ids,split['train'])]
                for name in ['min','max','mean','std']:
                    assert np.allclose(trainstats[k][name],stats(a)[name],atol=1e-10)
        checks=[]
        for s,episode_ids in split.items():
            if not episode_ids:
                checks.append({'split':s,'status':'UNAVAILABLE','reason':'No independent episodes after trajectory grouping'})
                continue
            # First and last frames test both video offsets and action padding at boundaries.
            samples=[]
            for eid,at_end in [(episode_ids[0],False),(episode_ids[-1],True)]:
                ep=epmap[eid];ix=np.flatnonzero(ids==eid); j=len(ix)-1 if at_end else 0
                t=float(data['timestamp'][ix[j]])
                image={}
                for cam in cameras:
                    pre='videos/'+cam
                    rel=info['video_path'].format(video_key=cam,chunk_index=ep[pre+'/chunk_index'],file_index=ep[pre+'/file_index'])
                    image[cam]=frame_at(root/rel,t+ep[pre+'/from_timestamp'])
                chunk_indices=np.minimum(np.arange(j,j+50),len(ix)-1)
                sample={'observation.state':np.asarray(data['observation.state'][ix[j]],dtype=np.float32),'action':np.asarray([data['action'][ix[x]] for x in chunk_indices],dtype=np.float32),'action_is_pad':np.arange(j,j+50)>=len(ix),'episode_index':np.asarray(eid),'frame_index':np.asarray(j),**image}
                assert sample['action'].shape==(50,16)
                assert sample['observation.state'].shape==(16,)
                for cam in cameras:assert sample[cam].shape==(3,224,224)
                samples.append(sample)
            batch={k:np.stack([x[k] for x in samples]) for k in samples[0]}
            np.savez_compressed(report/f'{s}_batch.npz',**batch)
            checks.append({'split':s,'status':'PASS','batch_shapes':{k:list(v.shape) for k,v in batch.items()},'sample_instructions':[epmap[e]['tasks'] for e in [episode_ids[0],episode_ids[-1]]],'policies':['ACT raw input','SmolVLA raw input before tokenizer and policy preprocessing']})
        # Real LeRobot CLI argument lists, separate from the pipeline manifest format.
        for policy in ['act','smolvla']:
            features={'observation.state':{'type':'STATE','shape':[16]}}
            rename={}
            for i,cam in enumerate(cameras,1):
                target=cam if policy=='act' else f'observation.images.camera{i}'
                features[target]={'type':'VISUAL','shape':[3,224,224]}
                if policy=='smolvla':rename[cam]=target
            args=['lerobot-train',f'--dataset.repo_id=local/{label}',f'--dataset.root={root}','--dataset.episodes='+json.dumps(split['train'],separators=(',',':')),'--dataset.eval_split=0','--dataset.video_backend=pyav','--tolerance_s=0.002','--seed=20261022','--batch_size=8','--num_workers=0','--steps=20000','--policy.device=cuda','--policy.push_to_hub=false','--wandb.enable=false','--policy.chunk_size=50','--policy.n_action_steps=1','--policy.input_features='+json.dumps(features,separators=(',',':')),'--policy.output_features='+json.dumps({'action':{'type':'ACTION','shape':[16]}},separators=(',',':')),f'--output_dir=outputs/{label}_{policy}']
            if policy=='act': args+=['--policy.type=act']
            else: args+=['--policy.path=lerobot/smolvla_base','--rename_map='+json.dumps(rename,separators=(',',':')),'--policy.max_state_dim=32','--policy.max_action_dim=32','--policy.adapt_to_pi_aloha=false','--policy.use_delta_joint_actions_aloha=false']
            dump(report/f'{policy}_train_argv.json',{'status':'TEMPLATE_PENDING_FRAMEWORK_SMOKE_TEST_AND_DATA_APPROVAL','argv':args,'validation_episodes':split['val'],'test_episodes':split['test'],'note':'No automatic eval_split: validation must use explicit val manifest. No GPU job has been launched.'})
        record={'dataset':label,'verified_copied_files':verified,'train_only_statistics_verified':bool(split['train']),'batches':checks,'framework_model_smoke_test':'NOT_RUN: torch and lerobot not installed in this data-preparation runtime'}
        dump(report/'verification.json',record);output.append(record)
        payload={str(p.relative_to(root)).replace('\\','/'):digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='payload_checksums.json'}
        dump(root/'preparation/payload_checksums.json',payload)
    dump(HERE/'verification.json',output)
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
