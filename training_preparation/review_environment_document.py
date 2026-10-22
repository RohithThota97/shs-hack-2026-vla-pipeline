"""Attach supplied environment documentation as evidence without changing frozen data."""
from pipeline import BASE,HERE,load,dump,digest

def main():
    source=BASE/'isaac_sim_hackathon_environment.md'
    handover=load(HERE/'reports/handover/summary.json')
    conversion=load(BASE/'fr3_duo_handover_84_ep/fr3_duo_handover_v9_night_84/meta/fr3_conversion.json')
    info=load(BASE/'fr3_duo_handover_84_ep/fr3_duo_handover_v9_night_84/meta/info.json')
    order=[f'{side}_{name}' for side in ['left','right'] for name in [*[f'joint{i}' for i in range(1,8)],'gripper']]
    assert info['features']['observation.state']['names']==order
    assert info['features']['action']['names']==order
    assert info['fps']==30
    assert 'rad, measured' in conversion['state']
    assert 'commanded target (rad)' in conversion['action']
    assert conversion['dataset_cfg']['gripper_contract']['closed_joint_rad']==0.8
    assert '1 - cmd/0.8' in conversion['action']
    evidence={
        'source_document':str(source),
        'source_document_sha256':digest(source),
        'document_dataset':'fr3_duo_handover_v9_lighting10',
        'document_episodes':10,
        'inspected_dataset':'fr3_duo_handover_v9_night_84',
        'inspected_episodes':84,
        'scope':'Supporting description of the handover environment family. Not a numerical robot-limit specification or a description of the stacking dataset.',
        'handover_control_contract':{
            'feature_order':order,
            'state':{'arm_values':'measured joint positions','arm_unit':'radian','gripper_values':'normalized measured joint state'},
            'action':{'arm_values':'absolute commanded joint targets','arm_unit':'radian','gripper_values':'normalized commands','open':1,'closed':0},
            'control_fps':30,'physics_fps':60,
            'cameras':{k:v['shape'] for k,v in info['features'].items() if v['dtype']=='video'},
            'action_timing_rule':conversion['action_timing_rule'],
            'evidence_status':'Order, units and command encoding cross-checked against the actual 84-episode feature and converter metadata. Gripper state and command normalizations are distinct.',
            'robot_limit_min':None,'robot_limit_max':None,
            'robot_limits_status':'Not supplied. Nominal ready pose is not a joint limit.',
            'provenance_assessment':'Actual converter/episode metadata and this family-level document support simulation origin. The document names another dataset; original real-camera claim remains unconfirmed.'
        },
        'stacking_control_contract':{
            'feature_order':None,'arm_unit':None,'action_semantics':None,'robot_limit_min':None,'robot_limit_max':None,
            'status':'Unresolved. Matching 16-dimensional shapes do not establish the same control contract for cube stacking.'
        },
        'release_effect':'No episode status changed; robot-limit and dataset-specific evidence gaps remain. Frozen candidates and test reservations were preserved.',
        'remaining_inputs':[
            'Scene/robot joint-limit tables or authored USD/URDF/config files, including any joint sign/offset transforms and gripper state bounds.',
            'Stacking converter/recorder specification establishing channel order, units, target/delta convention and gripper mapping.',
            'Stacking task-success criteria and evidence; dataset-specific confirmation of handover provenance.',
            'Independent handover trajectories and reliable layout/variation identifiers for the intended evaluation protocol.'
        ]
    }
    dump(HERE/'reports/environment_document_review.json',evidence)
    dump(HERE/'reports/handover/control_contract.json',evidence['handover_control_contract'])
    dump(HERE/'reports/stacking/control_contract.json',evidence['stacking_control_contract'])
    for label in ['handover','stacking']:
        path=HERE/'reports'/label/'episodes.json'
        episodes=load(path)
        for ep in episodes:
            ep['supporting_document_evidence']={
                'document_sha256':evidence['source_document_sha256'],
                'contract_report':f'reports/{label}/control_contract.json',
                'applicability':'handover-family description; actual control fields cross-checked' if label=='handover' else 'not established; document concerns a different task and dataset',
                'status_changed':False
            }
        dump(path,episodes)
    text='''# Environment document review

The supplied `isaac_sim_hackathon_environment.md` was inspected. It describes
`fr3_duo_handover_v9_lighting10` (10 episodes), rather than the supplied
`fr3_duo_handover_v9_night_84` (84 episodes) or the cube-stacking data.

The handover control description agrees with the actual 84-episode converter
metadata: left joints 1–7, left gripper, right joints 1–7, right gripper;
measured arm state in radians; absolute commanded arm targets in radians;
normalized gripper commands with 1=open and 0=closed. The measured gripper
state normalization and commanded gripper encoding are distinct and preserved.
The converter's original action timing rule remains authoritative.

The document strengthens the evidence that this handover family was generated
in Isaac Sim. It does not establish that the 84-episode package is a real-camera
recording. It contains no numerical joint limits; its nominal ready pose cannot
be used as a limit table. It also provides no stacking-specific control mapping
or task-success evidence. Those missing details cannot be inferred from matching
array dimensions.

Supporting contracts and document hashes were added to the mutable reports.
No episodes were automatically approved and no frozen dataset or test reservation
was modified. The existing candidate splits and previous technical checks stand.

The `isaac-sim` folder currently contains only an `Unconfirmed ... .crdownload`
file; no completed scene/config package was found there. An incomplete browser
download was not treated as authoritative robot configuration.

See `environment_document_review.json` for the structured evidence and remaining
inputs, and each source's `control_contract.json` for applicability.
'''
    (HERE/'reports/ENVIRONMENT_REVIEW.md').write_text(text,encoding='utf-8')
    print('Handover channel order, radians, absolute targets and gripper encoding cross-checked.')
    print('Document evidence added for all 574 episodes. No quality status or frozen artifact changed.')

if __name__=='__main__':main()
