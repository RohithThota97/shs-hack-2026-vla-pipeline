import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
import pyarrow.parquet as pq
base=Path(__file__).parent.parent
for folder in ['fr3_duo_handover_84_ep','fr3_duo_rand500']:
    root=next((base/folder).rglob('info.json')).parent.parent
    eps=pq.read_table(sorted((root/'meta/episodes').rglob('*.parquet'))).to_pylist()
    print(root, 'episodes',len(eps),'tasks',pq.read_table(root/'meta/tasks.parquet').to_pylist())
    print('first', {k:v for k,v in eps[0].items() if not k.startswith('stats/')})
    print('last', {k:v for k,v in eps[-1].items() if not k.startswith('stats/')})
