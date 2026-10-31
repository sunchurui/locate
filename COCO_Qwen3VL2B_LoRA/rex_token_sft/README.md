# Rex-Omni coordinate-token SFT on Qwen3-VL-2B

These experiments adapt the [Rex-Omni SFT implementation](https://github.com/IDEA-Research/Rex-Omni/tree/6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e/finetuning) to Qwen3-VL-2B with LoRA. Data conversion calls the upstream `GroundingTaskFn` directly for 0–999 coordinate quantization, per-category box ordering, and object/box marker serialization. The training entrypoint follows upstream's Hugging Face `Trainer` flow while replacing its Qwen2.5-VL and full-parameter assumptions with Qwen3-VL, PEFT LoRA and a Qwen3 image collator. Upstream default validation is disabled; this adaptation adds validation loss every 1,000 steps.

Unlike Rex-Omni, which repurposes existing vocabulary entries, this adaptation adds 1,000 coordinate tokens `<0>`–`<999>` to Qwen3-VL's tokenizer to preserve its existing vocabulary. PEFT trains the new token rows in the tied input/output embeddings as well as rank-16, alpha-32 LoRA adapters on language projection layers; vision layers stay frozen. The resulting adapter requires the saved tokenizer and resized base embeddings when loading. The training entrypoint checks that the coordinate tokens are single IDs and both LoRA and coordinate rows are trainable.

## Reused Rex-Omni token IDs

The separate `rex_id` experiment reassigns IDs **150643–151642** to `<0>`–`<999>`, matching [Rex-Omni's released tokenizer](https://huggingface.co/IDEA-Research/Rex-Omni/blob/c1d245b31bdc8159293a55573913afcea396a15b/tokenizer_config.json). It retains the Qwen3 model's 151,936 embedding rows, starts from Qwen3's weights in those rows, and trains those 1,000 rows through PEFT alongside the same LoRA configuration. The original base-vocabulary symbols at those IDs and their BPE merges are removed; image, chat and box markers are preserved. Consequently the remapping affects the tokenizer's handling of those ordinary symbols and must remain attached to this checkpoint. It does not import Rex-Omni model weights.

```bash
python rex_token_sft/prepare_rex_id_tokenizer.py --model model --output rex_id_tokenizer
sbatch rex_token_sft/train_rex_id_lora.sbatch
# After a successful training job:
sbatch rex_token_sft/evaluate_rex_id_coco.sbatch
```

The mapping generator refuses to overwrite an existing output, validates all 1,000 token IDs with the tokenizer library, and preserves the original structural token IDs. This version uses the same COCO data, output serialization and default batch/optimizer settings as the added-token experiment, but writes to `rex_id_sft_outputs/` and `evaluation/rex_id_val2017/`. It remains an adaptation to Qwen3-VL 2B and LoRA, not the original Rex-Omni pretrained model or its original full-parameter training.

`submit_gpu_candidates.py` can queue alternatives for one existing Rex-ID training job on A100 PCIe 80GB, H100 80GB HBM3 and A800 80GB. All use one GPU, the same microbatch/accumulation settings, and separate output directories. It records job IDs in `rex_id_gpu_candidates.json` and refuses duplicate submission when that registry exists. There is no automatic lock or mutual exclusion in the training code: an operator or periodic monitor must retain the first healthy running candidate and cancel the other candidate training and evaluation jobs. The current experiment has a ten-minute monitor for that purpose. The independently running added-token comparison is outside this candidate set.

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
