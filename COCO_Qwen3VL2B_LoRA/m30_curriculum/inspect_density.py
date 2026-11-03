"""Count COCO answer density with the established SFT annotation exclusions."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
root=Path(a.root)
source=root/'data/coco2017/annotations/instances_train2017.json'
data=json.loads(source.read_text())
counts=defaultdict(Counter)
for ann in data['annotations']:
    if ann.get('iscrowd',0):continue
    _,_,w,h=ann['bbox']
    if w>0 and h>0:counts[ann['image_id']][ann['category_id']]+=1
rows=[{'image_id':x['id'],'max_per_class':max(counts[x['id']].values(),default=0),
       'total_instances':sum(counts[x['id']].values())} for x in data['images']]
rows.sort(key=lambda x:x['image_id'])
report={'train_images':len(rows),'evaluable_instances':sum(x['total_instances'] for x in rows),
        'max_per_class':max(x['max_per_class'] for x in rows),
        'max_total':max(x['total_instances'] for x in rows),
        'paper_s1_per_class_le50':sum(x['max_per_class']<=50 for x in rows),
        'paper_s2_total_le500':sum(x['total_instances']<=500 for x in rows),
        'paper_s2_additional_dense_images':sum(x['max_per_class']>50 and x['total_instances']<=500 for x in rows),
        'per_class_caps':{str(t):sum(x['max_per_class']<=t for x in rows) for t in [1,2,3,4,5,8,10,20,30,50]},
        'total_caps':{str(t):sum(x['total_instances']<=t for x in rows) for t in [1,2,3,4,5,8,10,20,30,50]},
        'exclusions':'iscrowd or nonpositive width/height; identical to original SFT preparation'}
out=root/'m30_curriculum';out.mkdir(exist_ok=True)
(out/'density_report.json').write_text(json.dumps(report,indent=2)+'\n')
with (out/'image_density.jsonl').open('w') as f:
    for row in rows:f.write(json.dumps(row)+'\n')
print(json.dumps(report,indent=2))
