#!/usr/bin/env python3
import argparse, json, os
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument("--instances", required=True)
p.add_argument("--image-dir", required=True)
p.add_argument("--output", required=True)
args = p.parse_args()

with open(args.instances, encoding="utf-8") as f:
    data = json.load(f)
cats = {c["id"]: c["name"] for c in data["categories"]}
by_image = defaultdict(list)
for ann in data["annotations"]:
    if ann.get("iscrowd", 0):
        continue
    x, y, w, h = ann["bbox"]
    if w <= 0 or h <= 0:
        continue
    by_image[ann["image_id"]].append((x, y, x+w, y+h, cats[ann["category_id"]]))
images = sorted(data["images"], key=lambda x: x["id"])
os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w", encoding="utf-8") as out:
    for image in images:
        items = sorted(by_image[image["id"]], key=lambda a: (a[0], a[1], a[4]))
        if items:
            target = "[\n" + ",\n".join(
                '  {"bbox_2d": <bbox>, "label": "<ref-object>"}' for _ in items
            ) + "\n]"
            refs = [x[4] for x in items]
            boxes = [[x[0], x[1], x[2], x[3]] for x in items]
        else:
            target, refs, boxes = "[]", [], []
        row = {
            "messages": [
                {"role": "user", "content": "<image>Detect all objects from the 80 MS COCO categories. Return only a JSON array; each item must contain label and bbox_2d [x1,y1,x2,y2], with coordinates normalized to 0-1000."},
                {"role": "assistant", "content": target},
            ],
            "images": [os.path.join(args.image_dir, image["file_name"])],
            "objects": {"ref": refs, "bbox": boxes},
        }
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"{args.output}: {len(images)} images")
