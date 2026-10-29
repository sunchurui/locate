#!/usr/bin/env python3
import concurrent.futures
import hashlib
import json
import os
import pathlib
import threading
import time
import requests

URL = "https://modelscope.cn/datasets/PAI/COCO2017/resolve/master/train2017.zip"
ROOT = pathlib.Path(__file__).resolve().parent
OUTPUT = ROOT / "data/coco2017/archives/modelscope/train2017_parallel.zip"
TOTAL = 19336861798
SHA256 = "69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929"
CHUNK = 8 * 1024 * 1024
WORKERS = int(os.environ.get("COCO_DOWNLOAD_WORKERS", "8"))
META = OUTPUT.with_suffix(".resume.json")
MARKERS = OUTPUT.parent / ".train2017_ranges"
LOCK = threading.Lock()

def fetch_chunk(fd, start, end, marker, counter, total_chunks, started):
    if marker.exists():
        with LOCK:
            counter[0] += 1
        return
    position = start
    failures = 0
    while position <= end:
        headers = {"Range": f"bytes={position}-{end}"}
        expected_range = f"bytes {position}-{end}/{TOTAL}"
        try:
            with requests.get(URL, headers=headers, stream=True, timeout=(30, 120)) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"HTTP {response.status_code}; byte range was not honored")
                cr = response.headers.get("Content-Range", "")
                if cr != expected_range:
                    raise RuntimeError(f"unexpected Content-Range {cr!r}; wanted {expected_range!r}")
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if not block:
                        continue
                    written = os.pwrite(fd, block, position)
                    if written != len(block):
                        raise IOError(f"short pwrite {written}/{len(block)}")
                    position += written
            if position != end + 1:
                raise IOError(f"short range ended at {position-1}, wanted {end}")
            marker.touch()
            with LOCK:
                counter[0] += 1
                elapsed = max(1, time.time() - started)
                done_bytes = min(TOTAL, counter[0] * CHUNK)
                print(f"ranges {counter[0]}/{total_chunks}; avg {done_bytes/1024/1024/elapsed:.2f} MiB/s", flush=True)
            return
        except Exception as error:
            failures += 1
            print(f"range {start}-{end} resumed at {position}, retry {failures}: {error}", flush=True)
            if failures >= 120:
                raise RuntimeError(f"failed range {start}-{end}") from error
            time.sleep(min(5, 1 + failures // 10))

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MARKERS.mkdir(exist_ok=True)
    if META.exists():
        meta = json.loads(META.read_text())
        prefix = int(meta["valid_prefix"])
        if int(meta["total"]) != TOTAL:
            raise RuntimeError("resume metadata total size mismatch")
    else:
        prefix = OUTPUT.stat().st_size if OUTPUT.exists() else 0
        if prefix > TOTAL:
            raise RuntimeError("existing archive exceeds expected size")
        META.write_text(json.dumps({"valid_prefix": prefix, "total": TOTAL}))
    ranges = []
    start = prefix
    while start < TOTAL:
        end = min(TOTAL - 1, start + CHUNK - 1)
        marker = MARKERS / f"{start}_{end}.done"
        if not marker.exists():
            ranges.append((start, end, marker))
        start = end + 1
    fd = os.open(OUTPUT, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, TOTAL)
    print(f"valid existing prefix: {prefix} bytes; missing ranges: {len(ranges)}; workers: {WORKERS}", flush=True)
    counter = [0]
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch_chunk, fd, a, b, m, counter, len(ranges), started) for a,b,m in ranges]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    os.close(fd)
    print("all ranges received; verifying ModelScope LFS SHA-256", flush=True)
    digest = hashlib.sha256()
    with OUTPUT.open("rb") as file:
        while True:
            block = file.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    actual = digest.hexdigest()
    print(f"sha256={actual}", flush=True)
    if actual != SHA256:
        raise RuntimeError(f"SHA-256 mismatch: expected {SHA256}, got {actual}")
    print("ModelScope train2017.zip verified", flush=True)

if __name__ == "__main__":
    main()
