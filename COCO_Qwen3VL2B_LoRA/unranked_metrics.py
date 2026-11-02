"""Score-free detection-set evaluation. Never read a prediction's score field."""
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

PROTOCOL = 'unranked_detection_set_v1'
THRESHOLDS = tuple(round(.50 + .05*i, 2) for i in range(10))


def match_count(overlaps, threshold):
    if not overlaps.size:
        return 0, np.zeros(overlaps.shape[0], dtype=bool)
    valid = overlaps >= threshold
    # Maximize match count FIRST, then sum IoU among those matchings.
    # A pure max-IoU assignment need not maximize the number of valid matches.
    bonus = min(overlaps.shape) + 1
    r, c = linear_sum_assignment(-(valid * (bonus + overlaps)))
    matched = np.zeros(overlaps.shape[0], dtype=bool)
    matched[r[valid[r, c]]] = True
    return int(matched.sum()), matched


def summarize(tp, fp, fn):
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    return {'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'precision': precision,
            'recall': recall, 'AP_set': precision*recall,
            'F1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def evaluate(dataset, predictions, image_ids=None, thresholds=THRESHOLDS):
    """All valid predictions, no ranking, top-k, score cutoff, or NMS.

    Match regular GT one-to-one within each image/class. Unmatched predictions
    overlapping crowd GT use COCO's crowd IoA rule and are reported as ignored.
    Every other unmatched prediction is FP, including duplicates.
    """
    if hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    ids = sorted(predictions if image_ids is None else image_ids)
    if len(ids) != len(set(ids)) or not set(ids).issubset(predictions):
        raise ValueError('Duplicate image IDs or missing prediction rows')
    known_images = {x['id'] for x in dataset['images']}
    categories = {x['id']: x['name'] for x in dataset['categories']}
    if not set(ids).issubset(known_images):
        raise ValueError('Unknown image ID')
    truth = defaultdict(list); truth_categories = defaultdict(set); wanted = set(ids)
    for ann in dataset['annotations']:
        if ann['image_id'] in wanted:
            truth[ann['image_id'], ann['category_id']].append(ann)
            truth_categories[ann['image_id']].add(ann['category_id'])
    counts = {t: defaultdict(lambda: np.zeros(4, dtype=np.int64)) for t in thresholds}
    total_predictions = 0
    for i in ids:
        groups = defaultdict(list)
        for prediction in predictions[i]:
            cat, box = int(prediction['category_id']), list(map(float, prediction['bbox']))
            if cat not in categories or len(box) != 4 or not all(map(math.isfinite,box)) or min(box[2:]) <= 0:
                raise ValueError(f'Invalid parsed detection, image {i}: {prediction}')
            groups[cat].append(box); total_predictions += 1
        for cat in sorted(set(groups) | truth_categories[i]):
            # Geometry canonicalization only resolves matching ties; never use
            # model confidence or generated sequence position in matching.
            boxes = sorted(groups[cat])
            regular = sorted([a['bbox'] for a in truth[i,cat] if not a.get('iscrowd',0)])
            crowd = sorted([a['bbox'] for a in truth[i,cat] if a.get('iscrowd',0)])
            overlaps = np.asarray(mask_utils.iou(boxes,regular,[0]*len(regular))) if boxes and regular else np.zeros((len(boxes),len(regular)))
            crowd_overlaps = np.asarray(mask_utils.iou(boxes,crowd,[1]*len(crowd))) if boxes and crowd else np.zeros((len(boxes),len(crowd)))
            for t in thresholds:
                tp, matched = match_count(overlaps,t)
                ignored = int(((~matched) & (crowd_overlaps >= t).any(axis=1)).sum()) if crowd else 0
                counts[t][cat] += [tp,len(boxes)-tp-ignored,len(regular)-tp,ignored]
    output = {'protocol': PROTOCOL, 'images': len(ids), 'detections': total_predictions,
              'score_used': False, 'max_detections': None, 'thresholds': {},
              'AP_definition': 'Custom single-operating-point PR area P*R per class/IoU; macro average over classes with GT or predictions, then IoU .50:.05:.95. Not standard ranked COCO AP.',
              'matching': 'same-image/class maximum-cardinality, then maximum-sum-IoU; geometry tie resolution',
              'crowd': 'unmatched predictions covered by same-class crowd IoA>=threshold ignored and counted separately'}
    for t in thresholds:
        total = sum(counts[t].values(), np.zeros(4,dtype=np.int64))
        per_category = {categories[c]: dict(summarize(*v[:3]),crowd_ignored=int(v[3])) for c,v in counts[t].items()}
        active = [v for v in per_category.values() if v['TP']+v['FP']+v['FN']]
        output['thresholds'][f'{t:.2f}'] = {
            'micro': dict(summarize(*total[:3]),crowd_ignored=int(total[3])),
            'macro': {name: float(np.mean([v[name] for v in active])) if active else 0. for name in ['precision','recall','F1','AP_set']},
            'macro_categories': len(active), 'per_category': per_category}
    output['mean_micro_F1_50_95'] = float(np.mean([x['micro']['F1'] for x in output['thresholds'].values()]))
    output['AP_set_50_95'] = float(np.mean([x['macro']['AP_set'] for x in output['thresholds'].values()]))
    output['AP_set_50'] = output['thresholds']['0.50']['macro']['AP_set'] if '0.50' in output['thresholds'] else None
    output['AP_set_75'] = output['thresholds']['0.75']['macro']['AP_set'] if '0.75' in output['thresholds'] else None
    return output


def load_predictions(path):
    rows = {}
    for line in Path(path).open():
        row = json.loads(line)
        if row['image_id'] in rows:
            raise ValueError(f'Duplicate image ID in {path}')
        rows[row['image_id']] = row['detections']
    return rows


def save_metrics(output, metrics):
    folder = Path(output); folder.mkdir(parents=True,exist_ok=True)
    destination = folder/'set_metrics.json'
    temporary = destination.with_name(f'{destination.stem}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(metrics,indent=2)+'\n'); temporary.replace(destination)
    return destination


def self_check():
    # Duplicate penalties and permutation/score invariance are protocol-critical.
    data = {'images':[{'id':1}], 'categories':[{'id':1,'name':'x'}],
            'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[0,0,10,10],'iscrowd':0}]}
    rows = {1:[{'category_id':1,'bbox':[0,0,10,10],'score':.01}]*2 +
              [{'category_id':1,'bbox':[20,20,10,10],'score':.99}]}
    result = evaluate(data,rows,thresholds=(.5,))
    v = result['thresholds']['0.50']['micro']; assert (v['TP'],v['FP'],v['FN']) == (1,2,0)
    assert abs(result['AP_set_50'] - 1/3) < 1e-12
    changed = {1:[dict(p,score=100-n) for n,p in enumerate(reversed(rows[1]))]}
    assert evaluate(data,changed,thresholds=(.5,)) == result
    many = {1:[{'category_id':1,'bbox':[0,0,10,10]}] +
              [{'category_id':1,'bbox':[20,20,10,10]}]*999}
    dense = evaluate(data,many,thresholds=(.5,))
    assert dense['detections']==1000 and abs(dense['AP_set_50']-.001)<1e-12
    empty = evaluate(data,{1:[]},thresholds=(.5,))
    assert empty['thresholds']['0.50']['micro']['FN']==1 and empty['AP_set_50']==0
    valid = np.array([[.6,.6],[1.,.3]])
    assert match_count(valid,.5)[0] == 2
    data['annotations'].append({'id':2,'image_id':1,'category_id':1,'bbox':[20,20,10,10],'iscrowd':1})
    v = evaluate(data,rows,thresholds=(.5,))['thresholds']['0.50']['micro']
    assert (v['TP'],v['FP'],v['FN'],v['crowd_ignored']) == (1,1,0,1)


if __name__ == '__main__':
    self_check(); print('Unranked metric invariance/duplicate/matching/crowd checks passed')
