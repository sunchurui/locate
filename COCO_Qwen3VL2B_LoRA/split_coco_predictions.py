#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-shards", type=int, default=2)
    args = ap.parse_args()
    if args.num_shards < 1:
        ap.error("num-shards must be positive")

    annotation = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    image_ids = sorted(image["id"] for image in annotation["images"])
    image_index = {image_id: i for i, image_id in enumerate(image_ids)}
    rows = {}
    with Path(args.predictions).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            image_id = row["image_id"]
            if image_id in rows:
                raise ValueError(f"duplicate prediction for image {image_id}")
            if image_id not in image_index:
                raise ValueError(f"unknown image id {image_id}")
            rows[image_id] = row

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for shard in range(args.num_shards):
        path = out_dir / f"predictions.rank{shard}.jsonl"
        with path.open("w", encoding="utf-8") as out:
            for image_id in image_ids:
                if image_id in rows and image_index[image_id] % args.num_shards == shard:
                    out.write(json.dumps(rows[image_id]) + "\n")
    print(f"split {len(rows)}/{len(image_ids)} existing image predictions across {args.num_shards} shards")


if __name__ == "__main__":
    main()
