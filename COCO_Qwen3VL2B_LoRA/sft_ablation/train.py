"""Single-factor variants of the exact established Rex-ID LoRA SFT entrypoint."""
import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import TrainerCallback, set_seed

import baseline_train as baseline
from data import load_image, sha256, user_message
from evaluate import run as evaluate


class Collator:
    def __init__(self, processor, max_length, order):
        self.processor, self.tokenizer, self.max_length, self.order = processor, processor.tokenizer, max_length, order

    def __call__(self, examples):
        ids, labels, pixels, grids = [], [], [], []
        for example in examples:
            image = load_image(example)
            prompt = self.processor.apply_chat_template(user_message(example['prompt'], order=self.order),
                                                       tokenize=False, add_generation_prompt=True)
            full = prompt + example['answer'] + '<|im_end|>\n'
            prompt_input = self.processor(text=[prompt], images=[image], return_tensors='pt')
            full_input = self.processor(text=[full], images=[image], return_tensors='pt')
            image.close()
            prefix, tokens = prompt_input['input_ids'][0], full_input['input_ids'][0]
            if not torch.equal(tokens[:len(prefix)], prefix):
                raise ValueError('Prompt/answer token boundary mismatch')
            if len(tokens) > self.max_length:
                raise ValueError(f'Image {example["image_id"]} exceeds {self.max_length} tokens; never truncate GT')
            target = tokens.clone(); target[:len(prefix)] = -100
            ids.append(tokens); labels.append(target)
            pixels.append(full_input['pixel_values']); grids.append(full_input['image_grid_thw'])
        batch_ids = pad_sequence(ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return {'input_ids': batch_ids, 'attention_mask': batch_ids.ne(self.tokenizer.pad_token_id).long(),
                'labels': pad_sequence(labels, batch_first=True, padding_value=-100),
                'pixel_values': torch.cat(pixels), 'image_grid_thw': torch.cat(grids)}


class Protocol(TrainerCallback):
    def __init__(self, root, metadata, processor, smoke):
        self.root, self.metadata, self.processor, self.smoke = root, metadata, processor, smoke

    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ('loss', 'grad_norm', 'eval_loss'):
            if logs and key in logs and not math.isfinite(float(logs[key])):
                raise ValueError(f'Nonfinite {key} at optimizer step {state.global_step}')

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if self.smoke:
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
                raise ValueError('Smoke optimizer step has missing or nonfinite gradients')

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not all(torch.isfinite(p).all().item() for p in model.parameters() if p.requires_grad):
            raise ValueError('Final trained adapter contains nonfinite values')

    def on_save(self, args, state, control, model=None, **kwargs):
        checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
        (checkpoint / 'sft_ablation.json').write_text(json.dumps(self.metadata, indent=2) + '\n')
        self.processor.save_pretrained(checkpoint)
        interval = self.metadata['config']['train']['progress_ap_steps']
        if not self.smoke and state.global_step % interval == 0:
            metrics = evaluate(model, self.processor, self.root, Path(args.output_dir) / 'progress_eval' / f'step-{state.global_step}',
                               self.metadata['input_order'], self.metadata['config']['train']['generation_batch'],
                               self.metadata['config']['train']['max_new_tokens'],
                               self.root / self.metadata['config']['progress_manifest'])
            with (Path(args.output_dir) / 'progress_ap.jsonl').open('a') as f:
                f.write(json.dumps({'step': state.global_step, **metrics}) + '\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    parser.add_argument('--method', choices=['m34','m37','m38','m36'], required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); root = Path(args.root).resolve(); folder = root / 'sft_ablation'
    config = json.loads((folder / 'config.json').read_text())
    if sha256(folder / 'baseline_train.py') != config['baseline_trainer_sha256']:
        raise ValueError('Baseline entrypoint differs from the established Rex-ID LoRA implementation')
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 80_000_000_000:
        raise ValueError('Training, including smoke training, requires >=80GB GPU memory')
    prepared = folder / 'prepared'; report = json.loads((prepared / 'report.json').read_text())
    if report['identity']['config'] != config:
        raise ValueError('Frozen training data/config mismatch')
    name = ('smoke_' if args.smoke else '') + f'train_{args.method}.jsonl'
    data = prepared / name
    if sha256(data) != report['output_sha256'][name]:
        raise ValueError('Training manifest checksum mismatch')
    if sha256(root / config['val_rows']) != report['identity']['input_sha256']['val_rows']:
        raise ValueError('Original validation data changed')
    order = 'text_first' if args.method == 'm37' else 'image_first'
    output = root / ('sft_ablation_smoke' if args.smoke else 'sft_ablation_outputs') / args.method
    if (output / 'trainer_state.json').exists():
        raise ValueError('Output already has training state; preserve it rather than restarting over it')
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'method': args.method, 'input_order': order, 'config': config, 'smoke': args.smoke,
                'train_sha256': sha256(data), 'preparation_report_sha256': sha256(prepared / 'report.json'),
                'base_model': config['model'], 'initialization': 'fresh Qwen3-VL-2B + fresh LoRA; no previous SFT/RL adapter loaded',
                'gpu': torch.cuda.get_device_name(0), 'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory}
    (output / 'sft_ablation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    snapshot = output / 'source'; snapshot.mkdir(exist_ok=True)
    for source in list(folder.glob('*.py')) + list(folder.glob('*.sbatch')) + [folder / 'config.json', folder / 'README.md']:
        if source.exists() and not (snapshot / source.name).exists():shutil.copy2(source, snapshot / source.name)
    set_seed(config['seed'])  # Set BEFORE creating LoRA, not only inside Trainer.
    original_trainer = baseline.Trainer

    class VariantTrainer(original_trainer):
        def __init__(self, *positional, **kwargs):
            training_args = kwargs['args']
            training_args.save_total_limit = None  # Retain every checkpoint; never delete previous outputs.
            if args.smoke:
                training_args.max_steps = 2; training_args.dataloader_num_workers = 0
                training_args.eval_strategy = 'no'; training_args.save_strategy = 'no'; training_args.logging_steps = 1
            processor = kwargs['processing_class']
            callbacks = list(kwargs.pop('callbacks', []))
            callbacks.append(Protocol(root, metadata, processor, args.smoke)); kwargs['callbacks'] = callbacks
            digest = hashlib.sha256()
            for name, parameter in kwargs['model'].named_parameters():
                if parameter.requires_grad:
                    digest.update(name.encode()); digest.update(parameter.detach().cpu().view(torch.uint8).numpy().tobytes())
            metadata['initial_trainable_parameters_sha256'] = digest.hexdigest()
            (output / 'sft_ablation.json').write_text(json.dumps(metadata, indent=2) + '\n')
            print('SFT_ABLATION_PROTOCOL', json.dumps(metadata), flush=True)
            super().__init__(*positional, **kwargs)

    baseline.Trainer = VariantTrainer
    baseline.CocoRexCollator = lambda processor, max_length: Collator(processor, max_length, order)
    settings = config['train']
    sys.argv = [str(folder / 'baseline_train.py'), '--model', str(root / config['model']),
        '--tokenizer-dir', str(root / config['tokenizer']), '--train-data', str(data),
        '--val-data', str(prepared / 'smoke_val.jsonl' if args.smoke else root / config['val_rows']),
        '--output-dir', str(output), '--train-batch-size', str(settings['per_device_batch']),
        '--eval-batch-size', '4', '--grad-accum', str(settings['gradient_accumulation']),
        '--epochs', str(settings['epochs']), '--learning-rate', str(settings['learning_rate']),
        '--eval-steps', str(settings['eval_loss_steps']), '--save-steps', str(settings['save_steps']),
        '--max-length', str(settings['max_length'])]
    baseline.main()
    state = json.loads((output / 'trainer_state.json').read_text())
    if not args.smoke and state['global_step'] != 14786:
        raise ValueError(f'Unexpected training update budget: {state["global_step"]}')
    if args.smoke:
        print('REAL_GPU_SMOKE_COMPLETE', args.method, state['global_step'],
              'peak_gpu_gib', torch.cuda.max_memory_allocated()/1024**3, flush=True)


if __name__ == '__main__':
    main()
