"""Post-hoc 48/64 support check on the frozen M45 T=.5 member-box cache."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from micro_metrics import evaluate, save_metrics, PROTOCOL
from inference_ablation.m45_selection import select, detections


def main(root):
    source = root / 'inference_ablation/results/m45'
    files = sorted(source.glob('shard*/records/*.json'))
    records = [json.loads(p.read_text()) for p in files]
    fixed = set(json.loads((root / 'rl_val_manifest.json').read_text())['image_ids'])
    assert len(records) == len(fixed) == 256
    assert {r['image_id'] for r in records} == fixed
    folder = root / 'inference_ablation/results/m45_support48_v1'
    folder.mkdir(parents=True, exist_ok=True)
    config = {
        'protocol': PROTOCOL, 'temperature': .5, 'samples': 64,
        'extent': 'member', 'nms_iou': .7, 'minimum_support': [16, 32, 48],
        'interpretation': 'Post-hoc threshold extension requested after inspecting 16/32 results; exploratory on the same 256 images.',
        'record_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    frozen = folder / 'config.json'
    if frozen.exists():
        assert json.loads(frozen.read_text()) == config
    frozen.write_text(json.dumps(config, indent=2) + '\n')
    # Selection accesses only the class-name/ID schema, never GT boxes or labels.
    metadata = json.loads((root / 'data/coco2017/annotations/instances_val2017.json').read_text())
    categories = {c['name']: c['id'] for c in metadata['categories']}
    variants = {'baseline': {r['image_id']: r['baseline'] for r in records}}
    for minimum in [16, 32, 48]:
        name = f't0.5_member_s{minimum}_nms70'
        variants[name] = {
            r['image_id']: detections(
                select(r['details']['t0.5']['clusters'], minimum, .7, 'representative_box'),
                'representative_box', categories,
            ) for r in records
        }
        if minimum in (16, 32):
            old = root / 'inference_ablation/results/m45_selection_v1' / name / 'predictions.jsonl'
            rows = {r['image_id']: r['detections'] for r in map(json.loads, old.read_text().splitlines())}
            assert rows == variants[name], 'Previous predictions changed'
    # Freeze every variant before scoring any threshold.
    for name, rows in variants.items():
        target = folder / name
        target.mkdir(exist_ok=True)
        text = ''.join(json.dumps({'image_id': i, 'detections': rows[i]}) + '\n' for i in sorted(rows))
        path = target / 'predictions.jsonl'
        if path.exists():
            assert path.read_text() == text
        else:
            path.write_text(text)
    results = {}
    for name, rows in variants.items():
        metrics = evaluate(metadata, rows)
        metrics.update(source=str((folder / name / 'predictions.jsonl').relative_to(root)), image_ids=sorted(rows), coverage='complete')
        save_metrics(folder / name, metrics)
        results[name] = {k: metrics[k] for k in ['images', 'detections', 'meanIoU', 'IoU50', 'IoU95', 'native_limit']}
        print(name, json.dumps(results[name]), flush=True)
    lines = ['# M45 additional 48/64 support threshold', '', PROTOCOL, '',
             config['interpretation'], '',
             'Same frozen 64 generations per image, T=.5, member box, NMS IoU=.7. No new inference.', '',
             '| Variant | Boxes | P mIoU | R mIoU | F1 mIoU | F1@.50 | F1@.95 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, m in results.items():
        prf = m['meanIoU']
        lines.append(f'| {name} | {m["detections"]} | {100*prf["precision"]:.2f} | {100*prf["recall"]:.2f} | {100*prf["F1"]:.2f} | {100*m["IoU50"]["F1"]:.2f} | {100*m["IoU95"]["F1"]:.2f} |')
    summary = {'config': config, 'results': results}
    for base, name in [(folder, 'summary'), (root / 'reports', 'm45_support48_v1')]:
        (base / (name + '.json')).write_text(json.dumps(summary, indent=2) + '\n')
        (base / (name + '.md')).write_text('\n'.join(lines) + '\n')
    # Add only the new48 case;16/32 and baseline already have canonical entries.
    name = 't0.5_member_s48_nms70'
    metric_path = folder / name / 'micro_metrics.json'
    metrics = json.loads(metric_path.read_text())
    row = {k: metrics[k] for k in ['source', 'images', 'detections', 'coverage', 'meanIoU', 'IoU50', 'IoU95', 'native_limit', 'protocol']}
    row['metrics'] = str(metric_path.relative_to(root))
    row['interpretation'] = config['interpretation']
    for base in [root / 'evaluation/micro_summary', root / 'reports/micro_current']:
        path = base.with_suffix('.json')
        if not path.exists():
            continue
        registry = json.loads(path.read_text())
        assert registry['protocol'] == PROTOCOL
        registry['results'] = [r for r in registry['results'] if r['source'] != row['source']] + [row]
        path.write_text(json.dumps(registry, indent=2) + '\n')
        md = base.with_suffix('.md')
        text = '\n'.join(line for line in md.read_text().splitlines() if not line.startswith('| ' + row['source'] + ' |'))
        m = row['meanIoU']
        line = f'| {row["source"]} | 256 | complete | {100*m["precision"]:.2f} | {100*m["recall"]:.2f} | {100*m["F1"]:.2f} | {100*row["IoU50"]["F1"]:.2f} | {100*row["IoU95"]["F1"]:.2f} |'
        md.write_text(text.rstrip() + '\n' + line + '\n')
    print('M45_SUPPORT48_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    main(Path(parser.parse_args().root).resolve())
