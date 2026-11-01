"""Submit four independent >=80GB SFT jobs and their full COCO evaluations."""
import argparse
import json
import subprocess
from pathlib import Path

from data import sha256


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    args = parser.parse_args(); root = Path(args.root).resolve(); folder = root / 'sft_ablation'
    if not (folder / 'prepared/audit.json').exists():
        raise ValueError('Run the actual COCO/token-batch audit before submission')
    registry = folder / 'jobs.json'
    jobs = json.loads(registry.read_text()) if registry.exists() else {}
    candidates = {'m34': 'nvidia_a100_80gb_pcie', 'm37': 'a100-sxm4-80gb',
                  'm38': 'nvidia_a800_80gb_pcie', 'm36': 'nvidia_h100_80gb_hbm3'}
    config_hash = sha256(folder / 'config.json')
    for method, gpu in candidates.items():
        entry = jobs.setdefault(method, {'config_sha256': config_hash, 'gpu': gpu})
        if entry['config_sha256'] != config_hash:raise ValueError('Job registry/config mismatch')
        common = [f'--export=ALL,METHOD={method}']
        if 'training_job' not in entry:
            command = ['sbatch','--parsable',f'--gres=gpu:{gpu}:1','--cpus-per-task=4',
                       f'--job-name=rex_sft_{method}'] + common + ['sft_ablation/train.sbatch']
            entry['training_job'] = int(subprocess.check_output(command,cwd=root,text=True).strip().split(';')[0])
            registry.write_text(json.dumps(jobs,indent=2)+'\n')
        if 'evaluation_job' not in entry:
            command = ['sbatch','--parsable',f'--dependency=afterok:{entry["training_job"]}',
                       f'--job-name=rex_sft_{method}_eval'] + common + ['sft_ablation/evaluate.sbatch']
            entry['evaluation_job'] = int(subprocess.check_output(command,cwd=root,text=True).strip().split(';')[0])
            registry.write_text(json.dumps(jobs,indent=2)+'\n')
        print(method,json.dumps(entry),flush=True)


if __name__ == '__main__':main()
