"""Class-aware detection reward for ms-swift GRPO."""
import json
import math

from swift.rewards import ORM, orms


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0


def score(completion, ground_truth):
    try:
        predictions = json.loads(completion)
        truth = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        if not isinstance(predictions, list) or not isinstance(truth, list):
            return 0.0
    except (ValueError, TypeError):
        return 0.0
    valid = []
    for item in predictions:
        try:
            label = item['label'].strip().lower()
            box = [float(v) for v in item['bbox_2d']]
            if len(box) != 4 or not label or not all(math.isfinite(v) and 0 <= v <= 1000 for v in box):
                continue
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            valid.append((label, box))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    candidates = []
    for pi, (label, box) in enumerate(valid):
        for gi, gt in enumerate(truth):
            if label == gt['label'].strip().lower():
                overlap = iou(box, gt['bbox_2d'])
                if overlap >= 0.5:
                    candidates.append((overlap, pi, gi))
    used_p, used_g, matched = set(), set(), []
    for overlap, pi, gi in sorted(candidates, reverse=True):
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
            matched.append(overlap)
    if not predictions and not truth:
        return 1.0
    f1 = 2 * len(matched) / (len(predictions) + len(truth)) if predictions or truth else 0.0
    mean_iou = sum(matched) / len(matched) if matched else 0.0
    return 0.1 + 0.7 * f1 + 0.2 * mean_iou


class CocoDetectionReward(ORM):
    def __call__(self, completions, ground_truth, **kwargs):
        if len(completions) != len(ground_truth):
            raise ValueError('completions and ground_truth length mismatch')
        return [score(c, gt) for c, gt in zip(completions, ground_truth)]


orms['coco_detection'] = CocoDetectionReward
