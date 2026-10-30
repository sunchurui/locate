#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-shards", type=int, default=2)
    args = ap.parse_args()

    annotation = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    image_ids = sorted(image["id"] for image in annotation["images"])
    rows = {}
    for shard in range(args.num_shards):
        path = Path(args.input_dir) / f"predictions.rank{shard}.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                image_id = row["image_id"]
                if image_id in rows:
                    raise ValueError(f"duplicate prediction for image {image_id}")
                rows[image_id] = row

    missing = set(image_ids) - rows.keys()
    extra = rows.keys() - set(image_ids)
    if missing or extra:
        raise ValueError(f"prediction coverage incomplete: missing={len(missing)}, extra={len(extra)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for image_id in image_ids:
            f.write(json.dumps(rows[image_id]) + "\n")
    print(f"merged predictions for all {len(image_ids)} validation images into {output}")


if __name__ == "__main__":
    main()
