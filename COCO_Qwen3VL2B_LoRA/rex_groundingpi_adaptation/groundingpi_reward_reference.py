"""GroundingPI-inspired reward for Rex-Omni/Easy-R1's batch reward interface.

Uses Rex-Omni's BoxIoURewardFunction.calculate_iou. The strict reward's matching,
ordering, duplicate and size conventions are documented approximations: the paper
does not specify every implementation detail for those components.
"""
import json
import math
from typing import Any

from verl.configs.reward_func import BoxIoURewardFunction

_iou = BoxIoURewardFunction().calculate_iou
EPS = 1e-8


def _parse_prediction(text: str):
    try:
        items = json.loads(text)
    except (TypeError, ValueError):
        return [], False
    if not isinstance(items, list):
        return [], False
    parsed = []
    for item in items:
        try:
            label = item['label'].strip().lower()
            box = [float(v) for v in item['bbox_2d']]
            if (not label or len(box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1000 for v in box)
                    or box[2] <= box[0] or box[3] <= box[1]):
                return [], False
            parsed.append((label, box))
        except (KeyError, TypeError, ValueError, AttributeError):
            return [], False
    return parsed, True


def _parse_truth(value: Any):
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict) and 'ground_truth' in value:
        return _parse_truth(value['ground_truth'])
    if isinstance(value, dict) and 'answer' in value:
        width, height = value['resized_image_size']
        if width <= 0 or height <= 0:
            raise ValueError('invalid resized image dimensions')
        result = []
        for label, item in value['answer'].items():
            for x1, y1, x2, y2 in item.get('boxes', []):
                result.append((label.strip().lower(), [x1 * 1000 / width, y1 * 1000 / height,
                                                      x2 * 1000 / width, y2 * 1000 / height]))
        return result
    if not isinstance(value, list):
        raise ValueError('ground truth must be a list or a Rex-Omni answer')
    return [(item['label'].strip().lower(), [float(v) for v in item['bbox_2d']]) for item in value]


def _matching(pred, truth, threshold):
    # One-to-one maximum-cardinality matching, with deterministic IoU ordering.
    candidates = []
    for plabel, pbox in pred:
        options = [(j, _iou(pbox, gbox)) for j, (glabel, gbox) in enumerate(truth) if plabel == glabel]
        candidates.append(sorted(((j, v) for j, v in options if v >= threshold), key=lambda x: (-x[1], x[0])))
    matched = {}
    def assign(i, seen):
        for j, overlap in candidates[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matched or assign(matched[j][0], seen):
                matched[j] = (i, overlap)
                return True
        return False
    for i in range(len(pred)):
        assign(i, set())
    return [v for _, v in matched.values()]


def _set_reward(pred, truth):
    if not truth:
        # GroundingPI's paper defines this only for nonempty reference sets.
        return float(not pred)
    if not pred:
        return 0.0
    total = 0.0
    for glabel, gbox in truth:
        best = max(range(len(pred)), key=lambda k: _iou(gbox, pred[k][1]))
        total += _iou(gbox, pred[best][1]) if pred[best][0] == glabel else 0.0
    recall, precision = total / len(truth), total / len(pred)
    return 2 * recall * precision / (recall + precision + EPS)


def _strict_reward(pred, truth, valid):
    if not valid:
        return 0.0
    if not pred and not truth:
        return 1.0
    p, g = len(pred), len(truth)
    count = min(p, g) / max(p, g) if max(p, g) else 1.0
    matches = [_matching(pred, truth, t) for t in (0.5, 0.75, 0.95)]
    f1 = sum(2 * len(m) / (p + g) if p + g else 1.0 for m in matches) / 3
    mean_iou = sum(matches[0]) / len(matches[0]) if matches[0] else 0.0
    ordered = float(pred == sorted(pred, key=lambda item: (item[0], *item[1])))
    duplicates = sum(any(label == earlier_label and _iou(box, earlier_box) >= 0.9
                         for earlier_label, earlier_box in pred[:i])
                     for i, (label, box) in enumerate(pred)) / max(p, 1)
    oversized = sum((b[2] - b[0]) * (b[3] - b[1]) > 800000 for _, b in pred) / max(p, 1)
    return max(0.0, min(1.0, 0.10 + 0.10 * count + 0.50 * f1 + 0.25 * mean_iou
                        + 0.05 * ordered - 0.07 * oversized - 0.03 * duplicates))


def compute_score(predictions, ground_truths):
    if len(predictions) != len(ground_truths):
        raise ValueError('prediction and reference batch lengths differ')
    scores = []
    for text, truth_raw in zip(predictions, ground_truths):
        pred, valid = _parse_prediction(text)
        truth = _parse_truth(truth_raw)
        set_reward = _set_reward(pred, truth) if valid else 0.0
        strict_reward = _strict_reward(pred, truth, valid)
        # Overall is only for existing logging; advantages must use both raw
        # components separately (see rex_groundingpi.patch).
        scores.append({'overall': 0.7 * set_reward + 0.3 * strict_reward,
                       'groundingpi_set': set_reward, 'groundingpi_strict': strict_reward})
    return scores
