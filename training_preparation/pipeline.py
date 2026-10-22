"""Local, non-destructive LeRobot v3 ingestion, QA and immutable candidate builds."""
from __future__ import annotations
import argparse, collections, concurrent.futures, datetime, hashlib, json, os, shutil, sys, time, uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import av
VERSION='1.0.0'
BASE=HERE.parent

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)

def load(path,default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def inspect_file(args):
    path,root,cache=args
    key=str(path.relative_to(root)).replace('\\','/')
    st=path.stat()
    old=cache.get(key)
    if old and old['size']==st.st_size and old['mtime_ns']==st.st_mtime_ns: return key,old
    out={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':digest(path)}
    if path.suffix=='.mp4':
        out.update(frames=0,decode_error=None,nonmonotonic_pts=0,bad_frame_intervals=0)
        try:
            pts=[]
            with av.open(str(path)) as c:
                stream=c.streams.video[0]
                stream.thread_type='AUTO'
                out['fps']=float(stream.average_rate)
                out['width']=stream.width; out['height']=stream.height
                for frame in c.decode(stream):
                    if frame.time is None: raise ValueError('Missing video timestamp')
                    pts.append(float(frame.time))
            ts=np.array(pts)
            out['frames']=len(ts)
            if len(ts):
                out['first_pts']=float(ts[0]);out['last_pts']=float(ts[-1])
                out['nonmonotonic_pts']=int(np.sum(np.diff(ts)<=0))
                out['bad_frame_intervals']=int(np.sum(np.abs(np.diff(ts)-1/out['fps'])>0.002))
                # Exact PTS retained for per-episode synchronization and frame counts.
                out['pts']=pts
            else: raise ValueError('Empty video')
        except Exception as e: out['decode_error']=str(e)
    return key,out

def stats(a):
    return {k:v.tolist() for k,v in {'min':a.min(0),'max':a.max(0),'mean':a.mean(0),'std':a.std(0),'q01':np.quantile(a,.01,axis=0),'q99':np.quantile(a,.99,axis=0),'count':np.array([len(a)])}.items()}

def source_build(label,root,cfg):
    report=HERE/'reports'/label
    report.mkdir(parents=True,exist_ok=True)
    info=load(root/'meta/info.json')
    if info['codebase_version']!='v3.0': raise ValueError('Only native LeRobot v3 is currently supported')
    eps=pq.read_table(sorted((root/'meta/episodes').rglob('*.parquet'))).to_pylist()
    tasks=pq.read_table(root/'meta/tasks.parquet').to_pylist()
    cameras=[k for k,v in info['features'].items() if v['dtype']=='video']
    cache=load(report/'file_inventory.json',{})
    files=sorted(p for p in root.rglob('*') if p.is_file())
    initial_signature=[(str(p.relative_to(root)),p.stat().st_size,p.stat().st_mtime_ns) for p in files]
    current={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['video_workers']) as pool:
        for i,(key,value) in enumerate(pool.map(inspect_file,[(p,root,cache) for p in files])):
            current[key]=value
            if i%10==0:
                print(f'{label}: inspected {i+1}/{len(files)} files',flush=True)
                dump(report/'file_inventory.json',current|{k:v for k,v in cache.items() if k not in current})
    dump(report/'file_inventory.json',current)
    final_signature=[(str(p.relative_to(root)),p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(root.rglob('*')) if p.is_file()]
    if initial_signature!=final_signature: raise RuntimeError('Source changed during ingestion; retry when recording is finalized')
    data=pa.concat_tables([pq.read_table(p) for p in sorted((root/'data').rglob('*.parquet'))])
    rows=data.to_pydict()
    epids=np.asarray(rows['episode_index'])
    extras={}
    if (root/'meta/fr3_episodes_extra.jsonl').exists():
        extras={int(x['episode_index']):x for x in [json.loads(line) for line in (root/'meta/fr3_episodes_extra.jsonl').read_text().splitlines()]}
    results=[]; arrays={}; duplicate_hashes=collections.defaultdict(list)
    for ep in eps:
        eid=int(ep['episode_index']); mask=np.flatnonzero(epids==eid); n=len(mask)
        invalid=[]; review=[]; checks={}
        if n!=ep['length'] or not n: invalid.append('episode_length_mismatch')
        required=['observation.state','action','timestamp','frame_index','index','task_index']
        if any(k not in rows for k in required): invalid.append('missing_required_columns')
        if invalid:
            results.append({'episode_index':eid,'status':'INVALID','invalid_reasons':invalid,'review_reasons':review}); continue
        arr={k:np.asarray([rows[k][i] for i in mask]) for k in required}
        arrays[eid]=arr
        ts=arr['timestamp'].astype(float)
        if not np.isfinite(ts).all(): invalid.append('nonfinite_timestamp')
        if n<2 or np.any(np.diff(ts)<=0): review.append('timestamp_not_strictly_increasing')
        if not np.allclose(ts,np.arange(n)/info['fps'],atol=cfg['sync_tolerance_s'],rtol=0): review.append('state_timestamp_grid_mismatch')
        if not np.array_equal(arr['frame_index'],np.arange(n)): invalid.append('frame_index_gap')
        if not np.array_equal(arr['index'],np.arange(ep['dataset_from_index'],ep['dataset_to_index'])): invalid.append('global_index_gap')
        datarel=info['data_path'].format(chunk_index=ep['data/chunk_index'],file_index=ep['data/file_index'])
        if datarel not in current: invalid.append('missing_data_file')
        else:
            indexed=pq.read_table(root/datarel,columns=['index','episode_index']).to_pydict()
            actual=[i for i,e in zip(indexed['index'],indexed['episode_index']) if e==eid]
            if actual!=arr['index'].tolist(): invalid.append('data_file_pointer_mismatch')
        for k in ['observation.state','action']:
            a=arr[k].astype(float)
            if a.shape!=(n,*info['features'][k]['shape']): invalid.append(k+':dimension_mismatch')
            if not np.isfinite(a).all(): invalid.append(k+':nonfinite')
            limits=cfg['sources'][label].get('limits',{}).get(k)
            if limits:
                if np.any(a<np.array(limits['min'])) or np.any(a>np.array(limits['max'])): invalid.append(k+':outside_configured_limits')
            else: review.append(k+':robot_limits_not_verified')
        camera_files={}
        for camera in cameras:
            prefix='videos/'+camera
            try:
                rel=info['video_path'].format(video_key=camera,chunk_index=ep[prefix+'/chunk_index'],file_index=ep[prefix+'/file_index'])
                camera_files[camera]=rel
                video=current[rel]
                if video.get('decode_error'): invalid.append(camera+':decode_error');continue
                start=ep[prefix+'/from_timestamp'];end=ep[prefix+'/to_timestamp']
                pts=np.array(video['pts'])
                segment=pts[(pts>=start-0.002)&(pts<end-0.002)]
                checks[camera]={'frames':len(segment),'expected_frames':n,'from_timestamp':start,'to_timestamp':end}
                if len(segment)!=n: review.append(camera+':frame_count_mismatch')
                if len(segment)==n and np.max(np.abs(segment-start-ts))>cfg['sync_tolerance_s']: review.append(camera+':timestamp_alignment')
                if abs(video['fps']-info['fps'])>0.001: review.append(camera+':fps_mismatch')
                if video['nonmonotonic_pts'] or video['bad_frame_intervals']: review.append(camera+':irregular_video_timestamps')
                shape=info['features'][camera]['shape']
                if [video['height'],video['width'],3]!=shape: invalid.append(camera+':resolution_mismatch')
            except (KeyError,FileNotFoundError) as e: invalid.append(camera+':missing_video_or_pointer:'+str(e))
        duration=n/info['fps']
        if not cfg['min_duration_s']<=duration<=cfg['max_duration_s']: review.append('duration_outside_bounds')
        if 'success' not in rows or not bool(rows['success'][mask[-1]]): review.append('final_success_not_recorded')
        extra=extras.get(eid,{})
        if extra.get('success') is False: review.append('extra_metadata_reports_failure')
        if not extra.get('final_metrics'): review.append('final_pose_or_operator_evidence_unavailable')
        if not info['features']['action'].get('names'): review.append('state_action_semantics_unverified')
        if label=='handover': review.append('claimed_real_source_has_simulation_metadata')
        # Serialized numeric content detects repeated trajectories, independent of source episode IDs.
        trajectory_hash=fingerprint({k:arr[k].tolist() for k in ['observation.state','action']})
        duplicate_hashes[trajectory_hash].append(eid)
        filekeys=sorted(set([datarel]+list(camera_files.values())))
        content_hash=fingerprint({'episode':eid,'files':{k:current[k]['sha256'] for k in filekeys if k in current},'range':[ep['dataset_from_index'],ep['dataset_to_index']]})
        demo_id=str(uuid.uuid5(uuid.NAMESPACE_URL,content_hash))
        results.append({'demo_id':demo_id,'episode_index':eid,'source':str(root),'source_domain_claim':cfg['sources'][label]['domain_claim'],'recording_start_utc':None,'recording_end_utc':None,'relative_start_s':float(ts[0]),'relative_end_s':float(ts[-1]),'duration_s':duration,'frames':n,'robot_type':info.get('robot_type'),'operator':None,'controller':extra.get('episode_extra',{}).get('controller'),'layout_id':extra.get('layout_id'),'instruction':ep['tasks'],'environment_metadata':extra,'camera_files':camera_files,'file_checksums':{k:current[k]['sha256'] for k in filekeys if k in current},'trajectory_hash':trajectory_hash,'content_hash':content_hash,'status':'INVALID' if invalid else 'NEEDS_REVIEW' if review else 'VALID','invalid_reasons':sorted(set(invalid)),'review_reasons':sorted(set(review)),'checks':checks,'import_time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pipeline_version':VERSION})
    dump(report/'episodes.json',results)
    # Duplicate trajectories are kept in the same group. No layout is invented.
    groups={h:sorted(ids) for h,ids in duplicate_hashes.items()}
    eligible={r['episode_index'] for r in results if r['status']!='INVALID'}
    groups={h:[e for e in ids if e in eligible] for h,ids in groups.items()}
    groups={h:ids for h,ids in groups.items() if ids}
    order=sorted(groups,key=lambda h:fingerprint({'seed':cfg['seed'],'group':h}))
    splits={'train':[],'val':[],'test':[]}
    frozen=load(report/'frozen_test.json')
    if frozen is None:
        target=max(1,round(len(eligible)*cfg['test_fraction']))
        testgroups=[]; count=0
        for h in order:
            if count>=target: break
            testgroups.append(h);count+=len(groups[h])
        frozen={'trajectory_hashes':testgroups,'seed':cfg['seed'],'method':'trajectory-grouped episode holdout; layout holdout not established'}
        dump(report/'frozen_test.json',frozen)
    for h in order:
        if h in frozen['trajectory_hashes']: splits['test']+=groups[h]
    remaining=[h for h in order if h not in frozen['trajectory_hashes']]
    target=max(1,round(len(eligible)*cfg['val_fraction']))
    for h in remaining:
        split='val' if len(splits['val'])<target else 'train'
        splits[split]+=groups[h]
    for k in splits:splits[k].sort()
    sets=[set(splits[k]) for k in splits]
    assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
    hashes={k:{r['trajectory_hash'] for r in results if r['episode_index'] in splits[k]} for k in splits}
    assert not any(hashes[a]&hashes[b] for a,b in [('train','val'),('train','test'),('val','test')])
    dump(report/'candidate_splits.json',splits)
    approved=load(HERE/'approvals.json',{})
    released={k:[e for e in ids if next(r for r in results if r['episode_index']==e)['status']=='VALID' or approved.get(next(r for r in results if r['episode_index']==e)['content_hash'])=={'decision':'APPROVE','reviewed':True}] for k,ids in splits.items()}
    dump(report/'approved_splits.json',released)
    preprocessing={'fps':info['fps'],'image_size':[224,224],'camera_mapping':{k:k for k in cameras},'state_key':'observation.state','action_key':'action','action_shape':info['features']['action']['shape'],'resampling':'none','seed':cfg['seed'],'pipeline_version':VERSION,'quality_config':cfg,'test_manifest':frozen}
    fp=fingerprint({'checksums':{k:v['sha256'] for k,v in sorted(current.items())},'preprocessing':preprocessing,'splits':splits})
    dest=HERE/'datasets'/f'{label}-ds-v1-{fp[:12]}'
    if not dest.exists():
        print(f'{label}: copying immutable native v3 snapshot to {dest}',flush=True)
        shutil.copytree(root,dest)
        # Original all-episode statistics are replaced; never use held-out rows for normalization.
        trainstats={k:stats(np.concatenate([arrays[e][k].astype(float) for e in splits['train']])) for k in ['observation.state','action']} if splits['train'] else {}
        # ACT uses fixed ImageNet visual stats; SmolVLA visual normalization is identity.
        for camera in cameras:
            trainstats[camera]={'mean':[[[x]] for x in [.485,.456,.406]],'std':[[[x]] for x in [.229,.224,.225]],'min':[[[0.]]] *3,'max':[[[1.]]] *3,'count':[sum(len(arrays[e]['timestamp']) for e in splits['train'])]}
        dump(dest/'meta/stats.json',trainstats)
        for p in dest.rglob('norm_stats.json'): p.unlink()
        newinfo=dict(info);newinfo['splits']={k:','.join(map(str,v)) for k,v in splits.items()}
        # Explicit dataset.episodes is authoritative; arbitrary splits are not ranges.
        dump(dest/'meta/info.json',newinfo)
        dump(dest/'preparation/preprocessing.json',preprocessing)
        dump(dest/'preparation/candidate_splits.json',splits)
        dump(dest/'preparation/fingerprint.json',{'fingerprint':fp,'status':'CANDIDATE_PENDING_REVIEW','source_root':str(root)})
        # Native per-episode summaries remain for provenance; v0.6.1 loads normalization from meta/stats.json.
        for policy in ['act','smolvla']:
            for split,ids in splits.items():
                dump(dest/f'preparation/{policy}_{split}.json',{'policy':policy,'dataset_root':str(dest),'repo_id':f'local/{label}','episodes':ids,'chunk_size':50,'n_action_steps':50,'fps':info['fps'],'chunk_duration_s':50/info['fps'],'seed':cfg['seed'],'normalization_stats':str(dest/'meta/stats.json'),'status':'CANDIDATE_PENDING_REVIEW'})
        dump(dest/'preparation/source_file_checksums.json',{k:{a:b for a,b in v.items() if a!='pts'} for k,v in current.items()})
    summary={'source':str(root),'snapshot':str(dest),'fingerprint':fp,'metadata_episodes':info['total_episodes'],'actual_episodes':len(eps),'actual_frames':len(epids),'fps':info['fps'],'tasks':tasks,'statuses':dict(collections.Counter(r['status'] for r in results)),'review_reasons':dict(collections.Counter(x for r in results for x in r['review_reasons'])),'invalid_reasons':dict(collections.Counter(x for r in results for x in r['invalid_reasons'])),'candidate_split_counts':{k:len(v) for k,v in splits.items()},'approved_split_counts':{k:len(v) for k,v in released.items()},'video_files':sum(p.suffix=='.mp4' for p in files),'video_frames_decoded':sum(v.get('frames',0) for v in current.values()),'leakage_checks':{'episode_overlap':False,'identical_trajectory_overlap':False,'layout_holdout_verified':False},'known_gaps':['layout identifiers unavailable; held-out layout generalization cannot be claimed','raw independent camera/state/action clock synchronization cannot be recovered from converted data','physical joint limits require robot/scene configuration','candidate artifacts must not be presented as approved training data','no live recorder endpoint supplied; adapter accepts finalized native v3 datasets only']}
    dump(report/'summary.json',summary)
    card=f"# {label} ds-v1\n\nStatus: CANDIDATE_PENDING_REVIEW\n\nFingerprint: `{fp}`\n\n"+json.dumps(summary,indent=2)+"\n\nSource files are preserved. Training statistics use only candidate train episode rows.\nSuccess flags are source labels, not independent task-success verification.\nUse explicit preparation/*_train.json episode lists; do not train on the full snapshot.\nACT does not condition on task text; SmolVLA uses the source task text.\n"
    (dest/'README.md').write_text(card,encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    return summary

def run():
    cfg=load(HERE/'config.json')
    summaries=[]
    for label,source in cfg['sources'].items():
        root=BASE/source['path']
        summaries.append(source_build(label,root,cfg))
    dump(HERE/'summary.json',summaries)

def watch(interval):
    cfg=load(HERE/'config.json'); last=None;stable=0
    while True:
        signature=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for s in cfg['sources'].values() for p in (BASE/s['path']).rglob('*') if p.is_file()]
        current=fingerprint(signature)
        if current==last: stable+=1
        else: stable=0
        if stable==2:
            try: run()
            except Exception as e:
                print(f'Import failed; retry after the next stable poll: {e}',flush=True)
                stable=1
        last=current;time.sleep(interval)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--interval',type=int,default=30)
    args=parser.parse_args()
    watch(args.interval) if args.watch else run()
