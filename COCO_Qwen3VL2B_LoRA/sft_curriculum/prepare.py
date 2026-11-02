"""Freeze paired M30-inspired curriculum/random orders; never edit targets or GT."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def prepare(root):
    folder=root/'sft_curriculum';out=folder/'prepared';out.mkdir(parents=True,exist_ok=True)
    annotation=root/'data/coco2017/annotations/instances_train2017.json'
    source=root/'data/coco2017/annotations/train_rex_token.jsonl'
    data=json.loads(annotation.read_text());counts={v['id']:Counter() for v in data['images']}
    for a in data['annotations']:
        if not a.get('iscrowd',0) and a['bbox'][2]>0 and a['bbox'][3]>0:
            counts[a['image_id']][a['category_id']]+=1
    ids=[];lines={}
    for line in source.open():
        image_id=json.loads(line)['image_id']
        if image_id in lines:raise ValueError('Duplicate source image')
        ids.append(image_id);lines[image_id]=line.rstrip('\n')+'\n'
    assert set(ids)==set(counts) and len(ids)==118287
    maximum={i:max(counts[i].values(),default=0) for i in ids}
    totals={i:sum(counts[i].values()) for i in ids}
    values=sorted(maximum.values());quantile=lambda q:values[math.ceil(q*(len(values)-1))]
    density={'images':len(ids),'max_per_class_quantiles':{str(q):quantile(q) for q in [0,.25,.5,.75,.9,.95,.99,1]},
        'max_total_boxes':max(totals.values()),'images_above_class50':sum(v>50 for v in maximum.values()),
        'images_above_total500':sum(v>500 for v in totals.values())}
    # Original50/500 is not meaningfully distinct on COCO if <10% are class-dense.
    # The fallback was fixed BEFORE looking at validation outcomes.
    threshold=50 if density['images_above_class50']/len(ids)>=.1 else quantile(.75)
    sparse=sorted(i for i in ids if maximum[i]<=threshold)
    dense=sorted(i for i in ids if maximum[i]>threshold)
    if not dense or len(sparse)<len(ids)//2:raise ValueError('No usable density curriculum')
    rng=random.Random(42);rng.shuffle(sparse);rng.shuffle(dense)
    boundary=(len(ids)//2//8)*8
    phase1=sparse[:boundary];phase2=sparse[boundary:]+dense
    random.Random(43).shuffle(phase2)
    curriculum=phase1+phase2
    control=sorted(ids);random.Random(42).shuffle(control)
    assert len(set(curriculum))==len(ids) and set(control)==set(curriculum)
    assert all(maximum[i]<=threshold for i in phase1)
    assert set(dense).issubset(phase2)
    input_hashes={'train_annotations':sha(annotation),'train_rows':sha(source),
        'val_rows':sha(root/'data/coco2017/annotations/val_rex_token.jsonl'),
        'baseline_train':sha(root/'sft_ablation/baseline_train.py'),
        'val_manifest':sha(root/'rl_val_manifest.json')}
    report={'method':'m30_density_order_ablation','seed':42,'phase2_shuffle_seed':43,
        'density':density,'sparse_max_per_class':threshold,'sparse_images':len(sparse),'dense_images':len(dense),
        'phase1_images':boundary,'phase2_images':len(phase2),'phase1_end_step':boundary//8,
        'unique_images':len(ids),'every_image_seen_exactly_once':True,'optimizer_updates':math.ceil(len(ids)/8),
        'threshold_rule':'class50 if >=10% images above50, else training-only75th percentile; no val-based choice',
        'scope':'M30-inspired ORDER ablation: fixed image/context budget, no pixel decoder/vision unfreezing; not full WeVisOne curriculum',
        'inputs':input_hashes,'files':{}}
    for variant,order in [('curriculum',curriculum),('random_control',control)]:
        for name,text in [(f'{variant}_ids.json',json.dumps(order)+'\n'),
                          (f'{variant}.jsonl',''.join(lines[i] for i in order))]:
            path=out/name
            if path.exists() and path.read_text()!=text:raise ValueError(f'Frozen preparation differs: {path}')
            if not path.exists():path.write_text(text)
            report['files'][name]=sha(path)
    path=folder/'density_report.json';path.write_text(json.dumps(density,indent=2)+'\n')
    path=out/'report.json'
    if path.exists() and json.loads(path.read_text())!=report:raise ValueError('Frozen metadata differs')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print('M30_PREPARED',json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    prepare(Path(parser.parse_args().root).resolve())
