#!/usr/bin/env bash
set -euo pipefail
ROOT="${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA=$ROOT/data/coco2017
ARCH=$DATA/archives
mkdir -p "$ARCH" "$DATA"
cd "$ARCH"
fetch() {
  local file="$1" url="$2" sha="$3"
  if [[ -f "$file" ]] && echo "$sha  $file" | sha1sum -c - >/dev/null 2>&1; then
    echo "verified existing $file"
    return 0
  fi
  for attempt in $(seq 1 20); do
    echo "download attempt $attempt for $file (resume enabled)"
    if curl -fL --connect-timeout 30 --retry 2 --retry-all-errors --retry-delay 10 \
      --speed-time 120 --speed-limit 1024 --continue-at - "$url" -o "$file"; then
      echo "$sha  $file" | sha1sum -c -
      return 0
    fi
    echo "transfer interrupted for $file; retrying from current file size" >&2
    sleep 15
  done
  echo "failed to download $file after retries" >&2
  return 1
}
TRAIN_IMAGES=$DATA/train2017
if [[ -d "$TRAIN_IMAGES" ]] && [[ $(find "$TRAIN_IMAGES" -maxdepth 1 -type f -name '*.jpg' | wc -l) -eq 118287 ]]; then
  echo "verified extracted COCO train2017 image count"
else
  fetch train2017.zip https://appcenter-deeplearning.sh1a.qingstor.com/dataset/coco/train2017.zip 10ad623668ab00c62c096f0ed636d6aff41faca5
fi
fetch val2017.zip https://appcenter-deeplearning.sh1a.qingstor.com/dataset/coco/val2017.zip 4950dc9d00dbe1c933ee0170f5797584351d2a41
fetch annotations_trainval2017.zip https://appcenter-deeplearning.sh1a.qingstor.com/dataset/coco/annotations_trainval2017.zip 8551ee4bb5860311e79dace7e79cb91e432e78b3
cd "$DATA"
unzip -q -n "$ARCH/val2017.zip" -d .
unzip -q -n "$ARCH/annotations_trainval2017.zip" -d .
echo "COCO validation and annotations download, checksum verification, and extraction completed"
