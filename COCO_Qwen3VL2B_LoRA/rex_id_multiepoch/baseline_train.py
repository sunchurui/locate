#!/usr/bin/env python3
"""Qwen3-VL LoRA adaptation of Rex-Omni's Hugging Face Trainer SFT entrypoint.

The answer serializer is invoked separately from the original Rex-Omni source.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments


class CocoRexDataset(Dataset):
    def __init__(self, path):
        with open(path, encoding='utf-8') as f:
            self.rows = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class CocoRexCollator:
    def __init__(self, processor, max_length):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        ids, labels, pixels, grids = [], [], [], []
        for example in examples:
            with Image.open(example['image']) as source:
                image = source.convert('RGB')
            messages = [{'role': 'user', 'content': [{'type': 'image'},
                                                       {'type': 'text', 'text': example['prompt']}]}]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full = prompt + example['answer'] + '<|im_end|>\n'
            prompt_input = self.processor(text=[prompt], images=[image], return_tensors='pt')
            full_input = self.processor(text=[full], images=[image], return_tensors='pt')
            prompt_ids, full_ids = prompt_input['input_ids'][0], full_input['input_ids'][0]
            if not torch.equal(full_ids[:len(prompt_ids)], prompt_ids):
                raise ValueError(f'prompt token mismatch for image {example["image_id"]}')
            if len(full_ids) > self.max_length:
                raise ValueError(f'image {example["image_id"]} has {len(full_ids)} tokens, limit {self.max_length}')
            target = full_ids.clone()
            target[:len(prompt_ids)] = -100
            ids.append(full_ids)
            labels.append(target)
            pixels.append(full_input['pixel_values'])
            grids.append(full_input['image_grid_thw'])
        input_ids = pad_sequence(ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return {'input_ids': input_ids,
                'attention_mask': input_ids.ne(self.tokenizer.pad_token_id).long(),
                'labels': pad_sequence(labels, batch_first=True, padding_value=-100),
                'pixel_values': torch.cat(pixels, dim=0), 'image_grid_thw': torch.cat(grids, dim=0)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--tokenizer-dir', required=True)
    p.add_argument('--train-data', required=True)
    p.add_argument('--val-data', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--train-batch-size', type=int, default=4)
    p.add_argument('--eval-batch-size', type=int, default=4)
    p.add_argument('--grad-accum', type=int, default=2)
    p.add_argument('--epochs', type=float, default=1)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--eval-steps', type=int, default=1000)
    p.add_argument('--save-steps', type=int, default=1000)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--resume-from-checkpoint')
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.tokenizer_dir, trust_remote_code=True, local_files_only=True)
    tokenizer = processor.tokenizer
    processor.image_processor.max_pixels = 1024 * 28 * 28
    processor.image_processor.min_pixels = 16 * 28 * 28
    if hasattr(processor.image_processor, 'size'):
        processor.image_processor.size['longest_edge'] = 1024 * 28 * 28
        processor.image_processor.size['shortest_edge'] = 16 * 28 * 28
    coordinate_tokens = [f'<{i}>' for i in range(1000)]
    coord_ids = [tokenizer.convert_tokens_to_ids(t) for t in coordinate_tokens]
    if coord_ids != list(range(150643, 151643)):
        raise ValueError('coordinate tokens do not match the Rex-Omni ID range')
    for token, token_id in zip(coordinate_tokens, coord_ids):
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f'coordinate is not a single token: {token}')
    processor.save_pretrained(args.output_dir)
    with open(Path(args.output_dir) / 'coordinate_token_ids.json', 'w', encoding='utf-8') as f:
        json.dump({'tokens': coordinate_tokens, 'ids': coord_ids, 'base_model': args.model}, f)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    if model.get_input_embeddings().num_embeddings != 151936:
        raise ValueError('unexpected Qwen3-VL embedding shape; cannot safely reuse Rex IDs')
    model.config.use_cache = False
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
        target_modules=r'.*language_model\.layers\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)',
        trainable_token_indices=coord_ids, ensure_weight_tying=True)
    model = get_peft_model(model, lora_config)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if not any('lora_' in name for name in trainable) or not any('trainable_tokens_delta' in name for name in trainable):
        raise RuntimeError('both LoRA weights and coordinate-token rows must be trainable')
    if 'TrainableTokens' not in type(model.get_output_embeddings()).__name__:
        raise RuntimeError('output head is not tied to the trainable coordinate tokens')
    model.print_trainable_parameters()
    print('Coordinate token IDs:', coord_ids[0], 'to', coord_ids[-1], flush=True)
    train_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size, per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.learning_rate,
        lr_scheduler_type='cosine', warmup_ratio=0.03, weight_decay=0.01,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='steps', eval_steps=args.eval_steps, save_strategy='steps', save_steps=args.save_steps,
        save_total_limit=3, logging_steps=20, dataloader_num_workers=4, remove_unused_columns=False,
        report_to=[], seed=42, data_seed=42, max_grad_norm=1.0)
    trainer = Trainer(model=model, args=train_args, processing_class=processor,
                      data_collator=CocoRexCollator(processor, args.max_length),
                      train_dataset=CocoRexDataset(args.train_data), eval_dataset=CocoRexDataset(args.val_data))
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    trainer.save_state()
    processor.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
