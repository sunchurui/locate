"""Freeze four independent SFT datasets; never remove a COCO image's GT boxes."""
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from data import formatter, rng_for, serialize, sha256, resize_plan, transform_boxes


def spatial_distribution(rows, boxes_by_id, infos, grid, image_weights=None):
    counts = [0.] * (grid * grid)
    memberships = []
    for index, row in enumerate(rows):
        info = infos[row['image_id']]
        cells = []
        for boxes in boxes_by_id[row['image_id']].values():
            for x1, y1, x2, y2 in boxes:
                x = max(0, min(grid - 1, int((x1 + x2) / 2 / info['width'] * grid)))
                y = max(0, min(grid - 1, int((y1 + y2) / 2 / info['height'] * grid)))
                cell = y * grid + x
                counts[cell] += 1. if image_weights is None else image_weights[index]
                cells.append(cell)
        memberships.append(cells)
    return counts, memberships


def distribution_stats(counts):
    total = sum(counts)
    probabilities = [v / total for v in counts if v]
    entropy = -sum(p * math.log(p) for p in probabilities)
    return {'entropy': entropy, 'KL_to_uniform': math.log(len(counts)) - entropy,
            'centers': total, 'max_cell_fraction': max(probabilities)}


def freeze_rows(path, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)


def prepare(root, config):
    folder = root / 'sft_ablation/prepared'
    inputs = {name: sha256(root / config[key]) for name, key in
              [('train_rows', 'train_rows'), ('val_rows', 'val_rows'),
               ('train_annotations', 'train_annotations'), ('val_annotations', 'val_annotations')]}
    identity = {'config': config, 'input_sha256': inputs}
    if (folder / 'report.json').exists():
        old = json.loads((folder / 'report.json').read_text())
        if old['identity'] != identity:
            raise ValueError('Prepared data belongs to another immutable configuration')
        for name, digest in old['output_sha256'].items():
            if sha256(folder / name) != digest:
                raise ValueError(f'Prepared data checksum mismatch: {name}')
        print('PREPARED_DATA_VERIFIED', json.dumps(old['summary']), flush=True)
        return old
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Partial prepared directory exists; preserve it and choose another path')
    folder.mkdir(parents=True, exist_ok=True)
    task, upstream_hash = formatter(root, config['rex_commit'])
    annotations = json.loads((root / config['train_annotations']).read_text())
    infos = {i['id']: i for i in annotations['images']}
    categories = {i['id']: i['name'] for i in annotations['categories']}
    boxes = defaultdict(lambda: defaultdict(list))
    for ann in annotations['annotations']:
        if ann.get('iscrowd', 0):
            continue
        x, y, w, h = ann['bbox']
        if w > 0 and h > 0:
            boxes[ann['image_id']][categories[ann['category_id']]].append([x, y, x + w, y + h])
    del annotations
    rows = [json.loads(line) for line in (root / config['train_rows']).open()]
    if len(rows) != 118287 or len({r['image_id'] for r in rows}) != 118287:
        raise ValueError('Expected exactly the established 118287 COCO train images')
    if set(infos) != {r['image_id'] for r in rows}:
        raise ValueError('Training image pool differs from original COCO2017')
    # Exact original-target audit, including quantization and ordering.
    for row in rows:
        info = infos[row['image_id']]
        if serialize(task, boxes[row['image_id']], info['width'], info['height']) != row['answer']:
            raise ValueError(f'Baseline target differs from pinned Rex formatter: {row["image_id"]}')
    summary = {'training_images': len(rows), 'baseline_targets_verified': len(rows), 'seed': config['seed']}
    m34, m38 = [], []
    full_count = subset_count = queried_boxes = 0
    for row in rows:
        image_id = row['image_id']; info = infos[image_id]; targets = boxes[image_id]
        rng = rng_for(config['seed'], 'm34', image_id)
        names = sorted(targets)
        full = not names or rng.random() < config['methods']['m34']['full_query_probability']
        chosen = names if full else sorted(rng.sample(names, rng.randint(1, len(names))))
        record = dict(row)
        if not full:
            record['prompt'] = ('Detect all instances of these categories: ' + ', '.join(chosen) +
                '. Return each category with its boxes using the object and box markers and coordinate tokens. '
                'Return None if no objects are present.')
            record['answer'] = serialize(task, {k: targets[k] for k in chosen}, info['width'], info['height'])
        record['query_categories'] = chosen
        record['full_query'] = full
        full_count += full; subset_count += not full
        queried_boxes += sum(len(targets[k]) for k in chosen)
        m34.append(record)
        plan = resize_plan(info['width'], info['height'], config['methods']['m38'],
                           rng_for(config['seed'], 'm38', image_id))
        transformed = transform_boxes(targets, plan)
        if sum(map(len, transformed.values())) != sum(map(len, targets.values())):
            raise ValueError('M38 removed annotations')
        augmented = dict(row, augmentation=plan)
        augmented['answer'] = serialize(task, transformed, info['width'], info['height'])
        m38.append(augmented)
    summary['m34'] = {'full_queries': full_count, 'subset_queries': subset_count,
                      'queried_boxes': queried_boxes, 'visual_prompt_task': False}
    summary['m38'] = {'images': len(m38), 'all_annotation_counts_preserved': True,
                      'scale_range': [min(r['augmentation']['resized'][0] / r['augmentation']['canvas'][0] for r in m38),
                                      max(r['augmentation']['resized'][0] / r['augmentation']['canvas'][0] for r in m38)]}
    grid = config['methods']['m36']['grid']
    counts, memberships = spatial_distribution(rows, boxes, infos, grid)
    cutoff = sorted(counts)[int(grid * grid * config['methods']['m36']['sampling_factor'])]
    if cutoff <= 0:
        raise ValueError('M36 sampling factor gives an empty target distribution')
    weights = [sum(min(1., cutoff / counts[c]) for c in cells) / len(cells) if cells else None
               for cells in memberships]
    nonempty = [w for w in weights if w is not None]
    empty_weight = sum(nonempty) / len(nonempty)
    weights = [empty_weight if w is None else w for w in weights]
    rng = random.Random(config['seed'])
    indices = rng.choices(range(len(rows)), weights=weights, k=len(rows))
    m36 = [rows[i] for i in indices]
    expected, _ = spatial_distribution(rows, boxes, infos, grid, weights)
    observed, _ = spatial_distribution(m36, boxes, infos, grid)
    summary['m36'] = {'cell_cap_K': cutoff, 'weight_min': min(weights), 'weight_max': max(weights),
                      'unique_images_drawn': len({r['image_id'] for r in m36}),
                      'draws': len(m36), 'empty_weight': empty_weight,
                      'original_spatial': distribution_stats(counts),
                      'weighted_expected_spatial': distribution_stats(expected),
                      'drawn_spatial': distribution_stats(observed), 'all_GT_in_each_draw_retained': True}
    outputs = {'train_m34.jsonl': m34, 'train_m37.jsonl': rows, 'train_m38.jsonl': m38, 'train_m36.jsonl': m36}
    digests = {}
    for name, data in outputs.items():
        freeze_rows(folder / name, data); digests[name] = sha256(folder / name)
        freeze_rows(folder / ('smoke_' + name), data[:8]); digests['smoke_' + name] = sha256(folder / ('smoke_' + name))
    val_rows = [json.loads(line) for line in (root / config['val_rows']).open()]
    freeze_rows(folder / 'smoke_val.jsonl', val_rows[:4]); digests['smoke_val.jsonl'] = sha256(folder / 'smoke_val.jsonl')
    (folder / 'm36_sampling_manifest.json').write_text(json.dumps({'seed': config['seed'], 'indices': indices,
        'image_ids': [r['image_id'] for r in m36], 'weights': weights, 'grid_counts': counts}) + '\n')
    digests['m36_sampling_manifest.json'] = sha256(folder / 'm36_sampling_manifest.json')
    result = {'identity': identity, 'upstream_formatter_sha256': upstream_hash, 'summary': summary, 'output_sha256': digests}
    (folder / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PREPARED', json.dumps(summary), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    args = parser.parse_args(); root = Path(args.root).resolve()
    prepare(root, json.loads((root / 'sft_ablation/config.json').read_text()))
