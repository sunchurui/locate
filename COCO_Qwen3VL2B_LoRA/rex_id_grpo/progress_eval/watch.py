"""Submit independent AP jobs as complete training checkpoints become available.

Runs as a small CPU-only Slurm job. No training GPU is reserved, and neither
training process is restarted or modified. There are no locks or mutexes.
"""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def complete_checkpoint(path, step):
    required = ('adapter_config.json', 'adapter_model.safetensors', 'trainer_state.json',
                'optimizer.pt', 'scheduler.pt', 'rng_state.pth')
    if not all((path / name).is_file() for name in required):
        return False
    try:
        return json.loads((path / 'trainer_state.json').read_text())['global_step'] == step
    except (OSError, ValueError, KeyError):
        return False


def train_state(job_id):
    result = subprocess.run(['sacct', '-j', str(job_id), '-X', '--noheader', '--format=State', '-P'],
                            capture_output=True, text=True, check=True)
    rows = [line.strip().split('|')[0].split()[0] for line in result.stdout.splitlines() if line.strip()]
    return rows[0] if rows else 'UNKNOWN'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config_path = Path(args.config).resolve()
    terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL'}
    deadline = time.monotonic() + 15 * 3600
    while time.monotonic() < deadline:
        config = json.loads(config_path.read_text())
        pending = False
        for experiment in config['experiments']:
            state = train_state(experiment['training_job'])
            experiment['last_training_state'] = state
            for step in config['steps']:
                key = str(step)
                if key in experiment['evaluations']:
                    continue
                checkpoint = root / experiment['output'] / f'checkpoint-{step}'
                if complete_checkpoint(checkpoint, step):
                    output = root / 'evaluation/progress_val256' / f'{experiment["name"]}_step{step}'
                    result = subprocess.run([
                        'sbatch', '--parsable', f'--job-name=coco_ap_{experiment["name"]}_{step}',
                        f'--export=ALL,CHECKPOINT_PATH={checkpoint},EVAL_OUTPUT_DIR={output}',
                        str(root / 'rex_id_grpo/progress_eval/evaluate.sbatch')],
                        cwd=root, capture_output=True, text=True, check=True)
                    job = int(result.stdout.strip().split(';')[0])
                    experiment['evaluations'][key] = {'job_id': job, 'output': str(output)}
                    print(f'Submitted {experiment["name"]} step {step}: evaluation job {job}', flush=True)
                elif state in terminal:
                    experiment['evaluations'][key] = {'skipped': f'Training {state}; complete checkpoint unavailable'}
                    print(f'Skipped {experiment["name"]} step {step}: training {state}', flush=True)
                else:
                    pending = True
        config['last_checked_utc'] = datetime.now(timezone.utc).isoformat()
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        if not pending:
            print('All requested checkpoint evaluations submitted or unavailable after training ended.', flush=True)
            return
        time.sleep(60)
    raise TimeoutError('Checkpoint evaluation submission deadline reached; inspect training and this log')


if __name__ == '__main__':
    main()
