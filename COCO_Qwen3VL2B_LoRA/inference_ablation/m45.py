"""Author GUI-RC spatial voting, repeated separately for predicted objects."""
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from common import PROMPT, detections, iou, parse


def clusters(samples, threshold):
    groups = defaultdict(list)
    for sample_index, predictions in enumerate(samples):
        by_label = defaultdict(list)
        for prediction in predictions:
            by_label[prediction['label']].append(prediction)
        for label, boxes in by_label.items():
            existing = groups[label]
            used = set()
            if existing:
                weights = np.array([[iou(p['box'], c['representative']) for c in existing] for p in boxes])
                rows, cols = linear_sum_assignment(-weights)
                for row, col in zip(rows.tolist(), cols.tolist()):
                    if weights[row, col] >= threshold:
                        existing[col]['members'].append(boxes[row])
                        existing[col]['sample_ids'].append(sample_index)
                        existing[col]['representative'] = np.mean([p['box'] for p in existing[col]['members']], axis=0).tolist()
                        used.add(row)
            for row, prediction in enumerate(boxes):
                if row not in used:
                    existing.append({'representative': prediction['box'], 'members': [prediction],
                                     'sample_ids': [sample_index]})
    return [(label, cluster) for label in sorted(groups) for cluster in groups[label]]


def run(engine, image, info, baseline, config, upstream, categories):
    settings = config['methods']['m45']
    variants, details = {}, {}
    for temperature in settings['temperatures']:
        bank = engine.generate(image, PROMPT, f'm45:{info["id"]}:T{temperature}',
                               samples=settings['samples'], temperature=temperature,
                               top_k=settings['top_k'], top_p=settings['top_p'])
        samples = [parse(item, engine.tokenizer, info['width'], info['height']) for item in bank]
        voted, member_boxes, records = [], [], []
        for name, cluster in clusters(samples, settings['association_iou']):
            region, votes = upstream.vote([p['box'] for p in cluster['members']], info['width'], info['height'])
            if region is None:
                continue
            confidence = votes / len(bank)
            center = ((region[0] + region[2]) / 2, (region[1] + region[3]) / 2)
            # Primary output is the author's maximum-vote consensus region.
            voted.append({'label': name, 'box': region, 'confidence': confidence})
            # Secondary box-task adaptation retains a sampled object's extent.
            # No GT is used: choose the member center nearest the vote region.
            member = min(cluster['members'], key=lambda p:
                         (((p['box'][0] + p['box'][2]) / 2 - center[0]) ** 2 +
                          ((p['box'][1] + p['box'][3]) / 2 - center[1]) ** 2, -p['confidence']))
            member_boxes.append(dict(member, confidence=confidence))
            records.append({'label': name, 'support': len(set(cluster['sample_ids'])), 'peak_votes': votes,
                            'region': region, 'representative_box': member['box']})
        prefix = f't{temperature}'
        variants[prefix + '_vote'] = detections(voted, categories, confidence=True)
        variants[prefix + '_member'] = detections(member_boxes, categories, confidence=True)
        variants[prefix + '_sample0'] = detections(samples[0], categories)
        details[prefix] = {'samples': len(bank), 'clusters': records}
    return variants, details
