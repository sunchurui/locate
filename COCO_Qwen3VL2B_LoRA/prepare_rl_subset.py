#!/usr/bin/env python3
"""Create a portable, deterministic COCO subset and ms-swift GRPO JSONL."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

PROMPT = ('<image>Detect all objects from the 80 MS COCO categories. Return only a JSON array; '
          'each item must contain label and bbox_2d [x1,y1,x2,y2], with coordinates normalized to 0-1000.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--instances', required=True)
    p.add_argument('--image-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--split', choices=['train2017', 'val2017'], required=True)
    p.add_argument('--count', type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    raw = Path(a.instances).read_bytes()
    data = json.loads(raw)
    images = {int(i['id']): i for i in data['images']}
    if not 0 < a.count <= len(images):
        p.error('count must be between 1 and the number of images')
    def key(image_id):
        text = f'coco2017:{a.split}:seed={a.seed}:image_id={image_id}'
        return hashlib.sha256(text.encode('utf-8')).digest(), image_id
    ids = sorted(sorted(images, key=key)[:a.count])
    manifest = {
        'dataset': 'COCO 2017 instances', 'split': a.split, 'seed': a.seed,
        'selection': 'sort by SHA256 UTF-8 of coco2017:{split}:seed={seed}:image_id={id}, then numeric image ID; take first count; store IDs in ascending order',
        'count': a.count, 'annotation_sha256': hashlib.sha256(raw).hexdigest(), 'image_ids': ids,
    }
    manifest_path = Path(a.manifest)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing != manifest:
            raise ValueError(f'{manifest_path} does not match source annotations and selection settings')
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    categories = {c['id']: c['name'] for c in data['categories']}
    annotations = defaultdict(list)
    wanted = set(ids)
    for ann in data['annotations']:
        if ann['image_id'] in wanted and not ann.get('iscrowd', 0):
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                info = images[ann['image_id']]
                box = [max(0, min(1000, v)) for v in
                       (x * 1000 / info['width'], y * 1000 / info['height'],
                        (x + w) * 1000 / info['width'], (y + h) * 1000 / info['height'])]
                if box[2] > box[0] and box[3] > box[1]:
                    annotations[ann['image_id']].append({'label': categories[ann['category_id']], 'bbox_2d': box})
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_dir = Path(a.image_dir).resolve()
    with output.open('w', encoding='utf-8', newline='\n') as f:
        for image_id in ids:
            info = images[image_id]
            path = image_dir / info['file_name']
            if not path.is_file():
                raise FileNotFoundError(path)
            truth = sorted(annotations[image_id], key=lambda x: (x['label'], x['bbox_2d']))
            row = {'messages': [{'role': 'user', 'content': PROMPT}], 'images': [str(path)],
                   'image_id': image_id, 'ground_truth': json.dumps(truth, ensure_ascii=False)}
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'{output}: {len(ids)} images; annotations SHA256 {manifest["annotation_sha256"]}')


if __name__ == '__main__':
    main()
