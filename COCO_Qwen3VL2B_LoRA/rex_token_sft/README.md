# Rex-Omni coordinate-token SFT on Qwen3-VL-2B

This experiment adapts the [Rex-Omni SFT implementation](https://github.com/IDEA-Research/Rex-Omni/tree/6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e/finetuning) to Qwen3-VL-2B with LoRA. Data conversion calls the upstream `GroundingTaskFn` directly for 0–999 coordinate quantization, per-category box ordering, and object/box marker serialization. The training entrypoint follows upstream's Hugging Face `Trainer` flow while replacing its Qwen2.5-VL and full-parameter assumptions with Qwen3-VL, PEFT LoRA and a Qwen3 image collator. Upstream default validation is disabled; this adaptation adds validation loss every 1,000 steps.

Unlike Rex-Omni, which repurposes existing vocabulary entries, this adaptation adds 1,000 coordinate tokens `<0>`–`<999>` to Qwen3-VL's tokenizer to preserve its existing vocabulary. PEFT trains the new token rows in the tied input/output embeddings as well as rank-16, alpha-32 LoRA adapters on language projection layers; vision layers stay frozen. The resulting adapter requires the saved tokenizer and resized base embeddings when loading. The training entrypoint checks that the coordinate tokens are single IDs and both LoRA and coordinate rows are trainable.

The experiment uses all 118,287 train2017 images and the 5,000 val2017 images, excludes crowd and invalid boxes, and retains empty images. The output prompt asks for all COCO categories and Rex-Omni's object and box markers. It starts from the original Qwen3-VL 2B model, in a separate output directory from the JSON LoRA baseline.

On another Linux machine, install the dependencies in the main README plus PEFT 0.20.0 and a Transformers release supporting Qwen3-VL. Clone the source at its recorded commit:

```bash
mkdir -p references logs
git clone https://github.com/IDEA-Research/Rex-Omni.git references/Rex-Omni
git -C references/Rex-Omni checkout 6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e
python rex_token_sft/prepare_rex_sft.py --rex-source references/Rex-Omni \
  --instances data/coco2017/annotations/instances_train2017.json \
  --image-dir data/coco2017/train2017 --output data/coco2017/annotations/train_rex_token.jsonl
python rex_token_sft/prepare_rex_sft.py --rex-source references/Rex-Omni \
  --instances data/coco2017/annotations/instances_val2017.json \
  --image-dir data/coco2017/val2017 --output data/coco2017/annotations/val_rex_token.jsonl
sbatch rex_token_sft/train_rex_lora.sbatch
```

The Slurm defaults target one A100 80 GB, BF16, microbatch 4, gradient accumulation 2, one epoch, learning rate 1e-4, cosine decay and 3% warmup. Model, environment, experiment root and batch sizes can be overridden with environment variables in the job script; modify the GPU GRES for a different cluster. Set `RESUME_FROM_CHECKPOINT` to a full Trainer checkpoint to resume an interrupted job. Checkpoints, tokenizer, coordinate-token IDs and training state are stored under `rex_token_sft_outputs/`.

After the final adapter is saved, `sbatch rex_token_sft/evaluate_rex_coco.sbatch` evaluates all 5,000 validation images with the same COCO evaluator as the JSON baseline, parsing Rex-Omni token sequences instead of JSON. Set `ADAPTER` to a checkpoint directory to evaluate another saved checkpoint. Generated-order scores remain an AP caveat. Neither GPU fit nor the new end-to-end training run has been verified yet; inspect Slurm logs for startup and memory errors.
