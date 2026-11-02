"""COCO evaluation, shared by all variants, with no inference-time GT access."""
import contextlib
import io
from collections import defaultdict
import sys
from pathlib import Path

from pycocotools.coco import COCO
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unranked_metrics import evaluate as evaluate_set

def evaluate(gt, detections, image_ids):
    rows = {i: [] for i in image_ids}
    for prediction in detections:
        rows[prediction['image_id']].append(prediction)
    return evaluate_set(gt, rows, image_ids)


def evaluate_records(gt, records):
    variants = sorted(records[0]['variants'])
    ids = [row['image_id'] for row in records]
    output = {}
    for name in ['baseline', *variants]:
        detections = []
        for row in records:
            predictions = row['baseline'] if name == 'baseline' else row['variants'][name]
            detections.extend(dict(item, image_id=row['image_id']) for item in predictions)
        output[name] = evaluate(gt, detections, ids)
    for name in variants:
        output[name]['delta_F1_50_points'] = 100 * (output[name]['thresholds']['0.50']['micro']['F1'] - output['baseline']['thresholds']['0.50']['micro']['F1'])
        output[name]['delta_AP_set_points'] = 100 * (output[name]['AP_set_50_95'] - output['baseline']['AP_set_50_95'])
    return output


def bootstrap_dataset(gt, by_image_predictions, sampled_ids):
    # Duplicate images must get new IDs; repeated imgIds in COCOeval alone
    # would deduplicate the bootstrap draw and yield an invalid interval.
    images, annotations, detections = [], [], []
    for new_id, old_id in enumerate(sampled_ids):
        images.append(dict(gt.imgs[int(old_id)], id=new_id))
        for ann in gt.imgToAnns[int(old_id)]:
            annotations.append(dict(ann, id=len(annotations) + 1, image_id=new_id))
        detections.extend(dict(item, image_id=new_id) for item in by_image_predictions[int(old_id)])
    with contextlib.redirect_stdout(io.StringIO()):
        copied = COCO()
        copied.dataset = {'images': images, 'annotations': annotations, 'categories': gt.dataset['categories'], 'info': {}}
        copied.createIndex()
    return copied, detections, list(range(len(sampled_ids)))
