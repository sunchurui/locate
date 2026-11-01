"""Fetch author files at immutable commits and verify their SHA256 checksums."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def ensure_sources(folder):
    folder = Path(folder)
    lock = json.loads((folder / 'sources.lock.json').read_text())
    target = folder / 'upstream_sources'
    target.mkdir(exist_ok=True)
    for item in lock['files']:
        output = target / item['name']
        if output.exists():
            data = output.read_bytes()
        else:
            url = f'https://raw.githubusercontent.com/{item["repo"]}/{item["commit"]}/{item["path"]}'
            data = urlopen(url, timeout=60).read()
            output.write_bytes(data)
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f'Author source checksum differs: {item["name"]}')


if __name__ == '__main__':
    ensure_sources(Path(__file__).resolve().parent)
