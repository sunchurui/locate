"""Five-epoch fresh Rex-ID SFT; reuse the frozen original training entrypoint."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np
import torch
from transformers import TrainerCallback, set_seed


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def tune(trainer, target, folder):
    """Probe real long COCO answers, backward only; never change model weights."""
    total = torch.cuda.get_device_properties(0).total_memory
    if total < 79_000_000_000 or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('Training requires an >=80GB BF16 GPU')
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    indices = sorted(range(len(trainer.train_dataset.rows)),
                     key=lambda i: len(trainer.train_dataset.rows[i]['answer']), reverse=True)[:32]
    candidates, selected = [], None
    try:
        trainer.model.train()
        for size in (4, 8, 16, 32):
            batch = loss = None
            trainer.model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                batch = trainer._prepare_inputs(trainer.data_collator([trainer.train_dataset[i] for i in indices[:size]]))
                with trainer.compute_loss_context_manager():
                    loss = trainer.model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss on real long-answer probe')
                loss.backward()
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                safe = peak <= total * .80
                candidates.append({'microbatch': size, 'peak_gib': peak / 1024**3, 'loss': loss.item(), 'safe': safe})
                print('REAL_COCO_BATCH_PROBE', candidates[-1], flush=True)
                if safe:
                    selected = size
                else:
                    break
            except torch.cuda.OutOfMemoryError:
                candidates.append({'microbatch': size, 'oom': True})
                print('REAL_COCO_BATCH_PROBE_OOM', size, flush=True)
                break
            finally:
                batch = loss = None
                trainer.model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        random.setstate(rng[0]); np.random.set_state(rng[1])
        torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
    if selected is None:
        raise RuntimeError('Even microbatch4 failed the long-answer safety budget')
    trainer.args.per_device_train_batch_size = selected
    trainer.args.gradient_accumulation_steps = target // selected
    trainer._train_batch_size = selected
    write_json(folder / 'batch_probe.json', {'gpu': torch.cuda.get_device_name(), 'total_gib': total / 1024**3,
               'global_batch': target, 'microbatch': selected, 'accumulation': target // selected,
               'probe_image_ids': [trainer.train_dataset.rows[i]['image_id'] for i in indices],
               'probes': candidates, 'note': 'Forward/backward only, no optimizer step; 20% reserve for optimizer and length variation.'})


class EpochValidation(TrainerCallback):
    def __init__(self, root, output):
        self.root, self.output = root, output
        self.request_epoch = None

    def on_epoch_end(self, args, state, control, **kwargs):
        self.request_epoch = round(state.epoch)
        control.should_save = True
        control.should_evaluate = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if self.request_epoch is None or not state.is_world_process_zero:
            return control
        epoch = self.request_epoch
        self.request_epoch = None
        checkpoint = self.output / f'checkpoint-{state.global_step}'
        # Trainer checkpoints carry adapter weights; supply the exact saved SFT processor.
        self.processor.save_pretrained(checkpoint)
        write_json(self.output / 'epochs' / f'epoch{epoch}.json',
                   {'epoch': epoch, 'step': state.global_step, 'checkpoint': str(checkpoint)})
        torch.cuda.empty_cache()
        target = self.root / 'evaluation' / self.output.name / f'epoch{epoch}_val256'
        cmd = [sys.executable, str(self.root / 'rex_token_sft/evaluate_rex_id_coco.py'),
               '--model', str(self.root / 'model'), '--adapter', str(checkpoint),
               '--annotations', str(self.root / 'data/coco2017/annotations/instances_val2017.json'),
               '--image-dir', str(self.root / 'data/coco2017/val2017'), '--image-ids', str(self.root / 'rl_val_manifest.json'),
               '--output-dir', str(target), '--batch-size', '8']
        print('EPOCH_AP_START', epoch, flush=True)
        result = subprocess.run(cmd)
        write_json(self.output / 'epochs' / f'epoch{epoch}_evaluation.json',
                   {'epoch': epoch, 'returncode': result.returncode, 'metrics': str(target / 'coco_metrics.json')})
        print('EPOCH_AP_END', epoch, result.returncode, flush=True)
        return control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--global-batch', type=int, default=64, choices=[64])
    args = parser.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Fresh experiment requires a new empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    folder = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('frozen_rex_id_sft', folder / 'baseline_train.py')
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    original_args, original_trainer = base.TrainingArguments, base.Trainer
    callback = EpochValidation(root, output)

    def training_args(**kwargs):
        kwargs.update(num_train_epochs=5, eval_steps=250, save_steps=250,
                      save_total_limit=None, logging_steps=20)
        return original_args(**kwargs)

    class MultiEpochTrainer(original_trainer):
        def train(self, *positional, **kwargs):
            if kwargs.get('resume_from_checkpoint'):
                raise RuntimeError('This entrypoint starts fresh; no previous adapter or optimizer is loaded')
            if self.args.world_size != 1:
                raise RuntimeError('This batch64 launcher is configured for exactly one GPU worker')
            if len(self.train_dataset) != 118287 or len(self.eval_dataset) != 5000:
                raise RuntimeError('Expected the complete original COCO train2017/val2017 image sets')
            callback.processor = self.processing_class
            self.add_callback(callback)
            tune(self, args.global_batch, output)
            print('FORMAL_SFT_CONFIGURATION', {'epochs': self.args.num_train_epochs,
                  'microbatch': self.args.per_device_train_batch_size,
                  'accumulation': self.args.gradient_accumulation_steps,
                  'global_batch': self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps,
                  'train_images': len(self.train_dataset), 'val_images': len(self.eval_dataset)}, flush=True)
            return super().train(*positional, **kwargs)

    base.TrainingArguments, base.Trainer = training_args, MultiEpochTrainer
    set_seed(42)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for source in folder.iterdir():
        if source.is_file():
            shutil.copy2(source, snapshot / source.name)
    fingerprints = {}
    for name in ('train_rex_token.jsonl', 'val_rex_token.jsonl'):
        digest = hashlib.sha256()
        with (root / 'data/coco2017/annotations' / name).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        fingerprints[name] = digest.hexdigest()
    write_json(output / 'experiment.json', {'epochs': 5, 'initialization': 'Fresh model/ + freshly initialized LoRA; no SFT adapter',
               'global_batch': 64, 'seed': 42, 'lr': 1e-4, 'lora_r': 16, 'lora_alpha': 32,
               'coordinate_ids': [150643, 151642], 'loss_validation_steps': 250,
               'training_data_sha256': fingerprints,
               'baseline_source_sha256': hashlib.sha256((folder / 'baseline_train.py').read_bytes()).hexdigest(),
               'source_urls': ['https://github.com/IDEA-Research/Rex-Omni/tree/6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e/finetuning'],
               'note': 'Epoch count and effective batch both change relative to the old one-epoch batch8 experiment.'})
    sys.argv = [str(folder / 'baseline_train.py'), '--model', str(root / 'model'),
                '--tokenizer-dir', str(root / 'rex_id_tokenizer'),
                '--train-data', str(root / 'data/coco2017/annotations/train_rex_token.jsonl'),
                '--val-data', str(root / 'data/coco2017/annotations/val_rex_token.jsonl'),
                '--output-dir', str(output), '--epochs', '5', '--train-batch-size', '4', '--grad-accum', '16']
    base.main()


if __name__ == '__main__':
    main()
