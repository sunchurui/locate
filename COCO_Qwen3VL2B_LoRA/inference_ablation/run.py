#!/usr/bin/env python3
"""Run one frozen inference method and its paired greedy baseline."""
import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import shutil
import time
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO

from common import Engine, PROMPT, detections, parse, stable_seed
from metrics import evaluate_records
from setup_sources import ensure_sources
from upstream import Upstream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--method', choices=['m06', 'm44', 'm45', 'm46'], required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    folder = Path(__file__).resolve().parent
    config_text = (folder / 'config.json').read_text()
    config = json.loads(config_text)
    manifest_bytes = (root / config['manifest']).read_bytes()
    manifest = json.loads(manifest_bytes)
    annotation = root / 'data/coco2017/annotations/instances_val2017.json'
    if hashlib.sha256(annotation.read_bytes()).hexdigest() != manifest['annotation_sha256']:
        raise ValueError('Validation annotations differ from the fixed manifest')
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(annotation))
    image_ids = sorted(manifest['image_ids'])[args.shard_index::args.num_shards]
    image_ids.sort(key=lambda i: (stable_seed(config['seed'], 'image_order', i), i))
    if args.limit:
        image_ids = image_ids[:args.limit]
    if not image_ids:
        raise ValueError('Empty evaluation shard')
    output.mkdir(parents=True, exist_ok=True)
    code_snapshot = output / 'code_snapshots' / os.environ.get('SLURM_JOB_ID', 'local')
    code_snapshot.mkdir(parents=True, exist_ok=True)
    for source in list(folder.glob('*.py')) + [folder / 'config.json', folder / 'sources.lock.json']:
        if not (code_snapshot / source.name).exists():
            shutil.copy2(source, code_snapshot / source.name)
    snapshot = {'method': args.method, 'config': config, 'config_sha256': hashlib.sha256(config_text.encode()).hexdigest(),
                'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'image_ids': image_ids,
                'shard_index': args.shard_index, 'num_shards': args.num_shards,
                'adapter_sha256': hashlib.sha256((root / config['adapter'] / 'adapter_model.safetensors').read_bytes()).hexdigest()}
    snapshot_path = output / 'experiment.json'
    if snapshot_path.exists() and json.loads(snapshot_path.read_text()) != snapshot:
        raise ValueError('Output directory belongs to a different immutable experiment')
    snapshot_path.write_text(json.dumps(snapshot, indent=2) + '\n')
    ensure_sources(folder)
    upstream = Upstream(folder / 'upstream_sources')
    method = importlib.import_module(args.method)
    engine = Engine(root, config, output)
    (output / 'environment.json').write_text(json.dumps(engine.metadata, indent=2) + '\n')
    categories = {item['name'].lower(): item['id'] for item in gt.dataset['categories']}
    records = []
    record_folder = output / 'records'
    record_folder.mkdir(exist_ok=True)
    for image_id in image_ids:
        record_path = record_folder / f'{image_id}.json'
        if record_path.exists():
            records.append(json.loads(record_path.read_text()))
            continue
        started = time.perf_counter()
        info = gt.imgs[image_id]  # image metadata only; no GT annotations passed to methods
        image = Image.open(root / 'data/coco2017/val2017' / info['file_name']).convert('RGB')
        engine.accessed = []
        result = engine.generate(image, PROMPT, f'baseline:{image_id}')[0]
        baseline = parse(result, engine.tokenizer, info['width'], info['height'])
        base_keys = list(engine.accessed)
        variants, details = method.run(engine, image, info, baseline, config, upstream, categories)
        image.close()
        extra_keys = list(set(engine.accessed) - set(base_keys))
        record = {'image_id': image_id, 'baseline': detections(baseline, categories),
                  'variants': variants, 'details': details, 'baseline_cost': engine.cost(base_keys),
                  'extra_cost': engine.cost(extra_keys), 'processing_seconds': time.perf_counter() - started}
        temporary = record_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False) + '\n')
        temporary.replace(record_path)
        records.append(record)
        print(f'{args.method} shard {args.shard_index}: {len(records)}/{len(image_ids)} images; '
              f'extra responses {record["extra_cost"]["responses"]}; {record["processing_seconds"]:.1f}s', flush=True)
        if len(records) == config['pilot_images_per_shard']:
            pilot = evaluate_records(gt, records)
            (output / 'pilot_metrics.json').write_text(json.dumps({'exploratory_pilot_only': True, 'metrics': pilot}, indent=2) + '\n')
            print('REAL_IMAGE_PILOT_COMPLETE ' + json.dumps(pilot), flush=True)
    metrics = evaluate_records(gt, records)
    (output / 'coco_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    costs = {'images': len(records), 'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3,
             'processing_seconds': sum(r['processing_seconds'] for r in records)}
    for kind in ('baseline_cost', 'extra_cost'):
        costs[kind] = {key: sum(r[kind][key] for r in records)
                       for key in ('responses', 'tokens', 'generation_seconds', 'generate_batches')}
    (output / 'costs.json').write_text(json.dumps(costs, indent=2) + '\n')
    print('METHOD_COMPLETE ' + json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
