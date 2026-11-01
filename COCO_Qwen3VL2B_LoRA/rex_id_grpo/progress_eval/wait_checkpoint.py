"""A CPU-only readiness gate for a dependent checkpoint AP evaluation.

Computing nodes need no access to sbatch/sacct: all job dependencies are
submitted from the login node. No GPU is held while a checkpoint is pending.
"""
import argparse
import json
import time
from pathlib import Path


def ready(path, step):
    required = ('adapter_config.json', 'adapter_model.safetensors', 'trainer_state.json',
                'optimizer.pt', 'scheduler.pt', 'rng_state.pth')
    if not all((path / name).is_file() for name in required):
        return False
    try:
        return json.loads((path / 'trainer_state.json').read_text())['global_step'] == step
    except (OSError, ValueError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=15 * 3600)
    args = parser.parse_args()
    path = Path(args.checkpoint)
    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        if ready(path, args.step):
            print(f'Complete checkpoint ready: {path}', flush=True)
            return
        time.sleep(60)
    raise TimeoutError(f'No complete checkpoint within the bounded wait: {path}')


if __name__ == '__main__':
    main()
