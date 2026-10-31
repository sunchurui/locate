"""Post-hoc, object-aligned credit on actual Rex response token IDs.

This module does not change the scalar rewards, optimizer, or GRPO loss.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

COORD_FIRST = 150643
COORD_LAST = 151642
MARKERS = ('<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>')


def flat_response_ids(value):
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError('Object credit supports single-turn responses only')
        value = value[0]
    return [int(v) for v in value]


def overlap(a, b):
    w = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    intersection = w * h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-12), intersection / max(area_a, 1e-12)


@dataclass
class Prediction:
    label: str
    box: list
    token_indices: list
    gt_index: int | None = None
    quality: float = 0.
    ignored: bool = False
    credit: float = 0.


@dataclass
class Annotation:
    response_ids: list
    valid: bool
    finish_reason: str | None
    predictions: list = field(default_factory=list)
    class_ends: dict = field(default_factory=dict)
    quality: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    # (actual response-token index, raw signed credit, reason)
    stops: list = field(default_factory=list)


class ObjectAnnotator:
    def __init__(self, tokenizer, format_parser, settings):
        self.tokenizer = tokenizer
        self.format_parser = format_parser
        self.settings = settings
        self.marker_ids = {}
        for marker in MARKERS:
            ids = tokenizer.encode(marker, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f'Expected one actual token for {marker}: {ids}')
            self.marker_ids[marker] = ids[0]
        for q in range(1000):
            if tokenizer.encode(f'<{q}>', add_special_tokens=False) != [COORD_FIRST + q]:
                raise ValueError(f'Coordinate vocabulary differs at <{q}>')
        eos = tokenizer.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def annotate(self, ids, finish_reason, truth, ignore_regions=()):
        ids = flat_response_ids(ids)
        text = self.decode(ids)
        _, valid = self.format_parser(text)
        result = Annotation(ids, valid, finish_reason, quality=[0.] * len(truth), missing=list(range(len(truth))))
        if not valid:
            return result
        ref_start, ref_end, box_start, box_end = (self.marker_ids[m] for m in MARKERS)
        position = 0
        while position < len(ids):
            if ids[position] != ref_start:
                position += 1
                continue
            end_label = ids.index(ref_end, position + 1)
            start_box = ids.index(box_start, end_label + 1)
            end_box = ids.index(box_end, start_box + 1)
            label = self.decode(ids[position + 1:end_label]).strip().lower()
            coords = [i for i in range(start_box + 1, end_box) if COORD_FIRST <= ids[i] <= COORD_LAST]
            if len(coords) % 4:
                raise ValueError('Strict-valid response has a non-four-coordinate box')
            for offset in range(0, len(coords), 4):
                indices = coords[offset:offset + 4]
                if indices != list(range(indices[0], indices[0] + 4)):
                    raise ValueError('Coordinates must be consecutive actual response tokens')
                box = [(ids[i] - COORD_FIRST) * 1000 / 999 for i in indices]
                result.predictions.append(Prediction(label, box, indices))
            result.class_ends[label] = end_box
            position = end_box + 1

        # A dummy column per prediction permits all rows to remain unmatched.
        # Input order is fixed. Canonicalize identical-box ties below so that a
        # later exact duplicate never takes credit from an earlier identical box.
        predictions = result.predictions
        if predictions and truth:
            weights = np.zeros((len(predictions), len(truth) + len(predictions)), dtype=np.float64)
            for i, pred in enumerate(predictions):
                for j, (label, box) in enumerate(truth):
                    if pred.label == label:
                        weights[i, j] = overlap(pred.box, box)[0]
            rows, cols = linear_sum_assignment(-weights)
            for i, j in zip(rows.tolist(), cols.tolist()):
                if j < len(truth) and weights[i, j] > 0:
                    predictions[i].gt_index = j
                    predictions[i].quality = float(weights[i, j])
            identical = defaultdict(list)
            for i, pred in enumerate(predictions):
                identical[(pred.label, tuple(pred.box))].append(i)
            for indices in identical.values():
                matched = sorted((predictions[i].gt_index, predictions[i].quality)
                                 for i in indices if predictions[i].gt_index is not None)
                for i in indices:
                    predictions[i].gt_index, predictions[i].quality = None, 0.
                for i, (j, quality) in zip(indices, matched):
                    predictions[i].gt_index, predictions[i].quality = j, quality

        for pred in predictions:
            if pred.gt_index is not None:
                result.quality[pred.gt_index] = pred.quality
            else:
                pred.ignored = any(pred.label == region['label'] and
                                   overlap(pred.box, region['bbox_2d'])[1] >= self.settings['crowd_ignore_prediction_ioa']
                                   for region in ignore_regions)
        matched_gt = {p.gt_index for p in predictions if p.gt_index is not None}
        result.missing = [j for j in range(len(truth)) if j not in matched_gt]

        # Only genuine, normal termination gets a stop penalty. A box_end is a
        # per-class stopping action; absent classes have no such action, so use
        # the actually sampled terminal EOS. Never invent or train an EOS.
        if finish_reason == 'stop':
            class_sizes = Counter(label for label, _ in truth)
            missing_counts = Counter(truth[j][0] for j in result.missing)
            absent_missing = 0
            for label, count in missing_counts.items():
                if label in result.class_ends:
                    result.stops.append((result.class_ends[label], -count / class_sizes[label], f'class_stop:{label}'))
                else:
                    absent_missing += count
            if absent_missing and ids and ids[-1] in self.eos_ids:
                result.stops.append((len(ids) - 1, -absent_missing / len(truth), 'eos_absent_classes'))
        return result


def assign_group_credit(annotations, settings):
    if len(annotations) != settings['group_size']:
        raise ValueError(f'Expected {settings["group_size"]} responses per image')
    sizes = {len(a.quality) for a in annotations}
    if len(sizes) != 1:
        raise ValueError('GT identity/order differs within an image group')
    mean_quality = np.mean([a.quality for a in annotations], axis=0)
    for annotation in annotations:
        for pred in annotation.predictions:
            if pred.gt_index is not None:
                pred.credit = pred.quality - float(mean_quality[pred.gt_index])
            elif not pred.ignored:
                pred.credit = settings['extra_credit']
    return mean_quality.tolist()
