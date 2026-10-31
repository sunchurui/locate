"""Keep COCO crowd/ignore regions for the exact existing train/validation IDs."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    result = {}
    sources = {}
    for split, manifest in [('train2017', 'rl_train_10000_manifest.json'), ('val2017', 'rl_val_manifest.json')]:
        fixed = json.loads((root / manifest).read_text())
        wanted = set(fixed['image_ids'])
        path = root / f'data/coco2017/annotations/instances_{split}.json'
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != fixed['annotation_sha256']:
            raise ValueError(f'{split} annotations differ from the fixed manifest')
        data = json.loads(raw)
        images = {i['id']: i for i in data['images'] if i['id'] in wanted}
        categories = {c['id']: c['name'].lower() for c in data['categories']}
        regions = defaultdict(list)
        for ann in data['annotations']:
            if ann['image_id'] not in wanted or not (ann.get('iscrowd') or ann.get('ignore')):
                continue
            image = images[ann['image_id']]
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            box = [max(0, min(1000, value)) for value in
                   (x * 1000 / image['width'], y * 1000 / image['height'],
                    (x + w) * 1000 / image['width'], (y + h) * 1000 / image['height'])]
            regions[ann['image_id']].append({'label': categories[ann['category_id']], 'bbox_2d': box,
                                           'annotation_id': ann['id']})
        if set(result) & {str(i) for i in wanted}:
            raise ValueError('Train/validation image IDs overlap')
        result.update({str(i): regions[i] for i in sorted(wanted)})
        sources[split] = {'annotation_sha256': digest, 'image_count': len(wanted),
                          'ignored_regions': sum(map(len, regions.values()))}
    output = root / 'rex_id_object_credit/ignore_metadata.json'
    output.write_text(json.dumps({'sources': sources, 'images': result}, ensure_ascii=False) + '\n')
    print(json.dumps(sources, indent=2))


if __name__ == '__main__':
    main()
