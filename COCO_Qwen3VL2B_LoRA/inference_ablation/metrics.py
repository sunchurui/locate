"""COCO evaluation, shared by all variants, with no inference-time GT access."""
import contextlib
import io
from collections import defaultdict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

METRIC_NAMES = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
                'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']


def evaluate(gt, detections, image_ids):
    with contextlib.redirect_stdout(io.StringIO()):
        if detections:
            dt = gt.loadRes(detections)
        else:
            dt = COCO()
            dt.dataset = {'images': [gt.imgs[i] for i in image_ids], 'categories': gt.dataset['categories'],
                          'annotations': []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, 'bbox')
        evaluator.params.imgIds = list(image_ids)
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    return {'images': len(image_ids), 'detections': len(detections),
            **{name: float(value) for name, value in zip(METRIC_NAMES, evaluator.stats)}}


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
        output[name]['delta_AP_points'] = (output[name]['AP'] - output['baseline']['AP']) * 100
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
