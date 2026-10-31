#!/usr/bin/env python3
"""Queue GPU alternatives in separate output directories for manual monitoring."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--existing-train-job', required=True, type=int)
    p.add_argument('--existing-eval-job', required=True, type=int)
    a = p.parse_args()
    root = Path(a.root).resolve()
    registry = root / 'rex_id_gpu_candidates.json'
    if registry.exists():
        raise FileExistsError(f'already submitted: {registry}')
    records = [{'label': 'a100_sxm80', 'training_job': a.existing_train_job,
                'evaluation_job': a.existing_eval_job, 'output': str(root / 'rex_id_sft_outputs')}]
    candidates = [
        ('a100_pcie80', 'nvidia_a100_80gb_pcie', '08:00:00', 4, 2),
        ('h100_hbm3', 'nvidia_h100_80gb_hbm3', '06:00:00', 4, 2),
        ('a800_80', 'nvidia_a800_80gb_pcie', '08:00:00', 4, 2),
    ]
    def save():
        registry.write_text(json.dumps(records, indent=2) + '\n')
    def submit(args):
        out = subprocess.check_output(['sbatch', '--parsable', *args], cwd=root, text=True).strip()
        return int(out.split(';')[0])
    save()
    for label, gpu, hours, batch, accum in candidates:
        output = root / f'rex_id_sft_outputs_{label}'
        row = {'label': label, 'output': str(output)}
        try:
            row['training_job'] = submit([f'--gres=gpu:{gpu}:1', '--cpus-per-task=8', '--mem=48G',
                f'--time={hours}', f'--job-name=coco_rex_id_{label}',
                f'--export=ALL,OUTPUT_DIR={output},TRAIN_BATCH_SIZE={batch},GRAD_ACCUM={accum},EVAL_BATCH_SIZE={batch}',
                'rex_token_sft/train_rex_id_lora.sbatch'])
            records.append(row)
            save()
            row['evaluation_job'] = submit([f'--dependency=afterok:{row["training_job"]}',
                f'--job-name=coco_rex_id_eval_{label}',
                f'--export=ALL,ADAPTER={output},EVAL_OUTPUT_DIR={root}/evaluation/rex_id_val2017_{label}',
                'rex_token_sft/evaluate_rex_id_coco.sbatch'])
        except subprocess.CalledProcessError as error:
            row['submission_error'] = str(error)
            if row not in records:
                records.append(row)
        save()
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
