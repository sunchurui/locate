#!/usr/bin/env bash
set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA=$ROOT/data/coco2017
ARCH=$DATA/archives
MSZIP=$ARCH/modelscope/train2017_parallel.zip
"$ROOT/venv/bin/python" "$ROOT/download_coco_modelscope_ranges.py"
unzip -tq "$MSZIP"
unzip -q -n "$MSZIP" -d "$DATA"
count=$(find "$DATA/train2017" -maxdepth 1 -type f -name '*.jpg' | wc -l)
echo "train2017 image count: $count"
[[ "$count" -eq 118287 ]]
"$ROOT/venv/bin/python" "$ROOT/download_coco_remaining_ranges.py"
