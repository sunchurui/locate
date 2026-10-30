# COCO 2017 + Qwen3-VL-2B LoRA baseline

A reproducible single-GPU supervised fine-tuning (SFT) and GRPO workflow for COCO 2017 object detection using `Qwen/Qwen3-VL-2B-Instruct` and ms-swift.

## Dataset

- COCO 2017 train: 118,287 images; validation: 5,000 images.
- The test split is not downloaded or used.
- `iscrowd=1` annotations and invalid boxes are excluded.
- The prepared data uses ms-swift's Qwen3-VL grounding format. Pixel-space COCO boxes are normalized to the 0-1000 coordinate range by ms-swift.
- The training archive is downloaded from the PAI/COCO2017 ModelScope dataset and verified by SHA-256. Validation images and annotations are downloaded from the QingStor mirror and verified by the official COCO SHA-1 values.

## Training configuration

- Model: `Qwen/Qwen3-VL-2B-Instruct`; model weights are downloaded separately and are not stored in this repository.
- Framework: ms-swift 4.6.0.
- One A100 80 GB, BF16, one epoch.
- LoRA rank 16, alpha 32, all linear layers.
- Per-device training batch 4, gradient accumulation 2 (effective batch 8); evaluation batch 4.
- Learning rate 1e-4, cosine schedule, 3% warmup, gradient checkpointing.
- Validation loss every 1,000 optimizer steps; save and select the best validation-loss checkpoint.
- Image token cap 1024; maximum sequence length 4096.

The Slurm job runs data preparation, SFT, validation-loss evaluation, then greedy generation on COCO val and `pycocotools.COCOeval` for AP@[.50:.95], AP50, AP75, size-specific AP, and AR. The detection score uses generated object order because the target format has no calibrated confidence value; this caveat is written to the metrics JSON.

For a direct full-set comparison, `evaluate_base_full.sbatch` evaluates the unmodified base model on all 5,000 COCO validation images, writes predictions and metrics under `evaluation/base_val2017_full`, and requests one 16 GB V100. It uses batched inference (default batch size 8, reduced automatically if GPU memory runs low). Change the typed GRES request to a GPU available on another cluster. If full LoRA validation generation is interrupted, `resume_coco_eval.sbatch` first evaluates the base model on the same first 1,600 validation images, then resumes the LoRA predictions across all allocated GPUs, merges the complete 5,000-image output, and computes the COCO metrics. Inference uses BF16 on Ampere-class GPUs and FP16 on older GPUs. The evaluator automatically uses every GPU allocated to the job.

## Setup on a new Linux/HPC machine

The setup expects a CUDA-enabled PyTorch installation available to `BASE_PYTHON` (for example, a site's Anaconda environment), a CUDA-capable GPU, Bash, `unzip`, and Slurm. Set `BASE_PYTHON` if the site's Python is not on `PATH`.

```bash
cd coco_qwen3vl2b_lora
export BASE_PYTHON=/path/to/python-with-cuda-torch  # omit if python3 is already configured
bash setup_env.sh
bash download_model.sh
mkdir -p logs
sbatch download_coco.sbatch
```

After the data job completes:

```bash
sbatch train_and_evaluate.sbatch
```

The default data paths are relative to the repository. `EXPERIMENT_ROOT` can override the project root. `TRAIN_BATCH_SIZE`, `EVAL_BATCH_SIZE`, `GRAD_ACCUM`, and `NUM_EPOCHS` can override the default training batch and epoch settings for a different GPU.

## Reinforcement learning

`train_grpo.sbatch` starts from the SFT LoRA adapter, using that same adapter as the reference policy. Set `SFT_ADAPTER` to your checkpoint path on another machine; `MODEL_DIR`, `VENV_PATH`, `OUTPUT_DIR` and `EXPERIMENT_ROOT` can also be overridden. Change the Slurm GPU type as needed and submit from the experiment root with `sbatch train_grpo.sbatch`. To continue a saved run, set `RESUME_FROM_CHECKPOINT` to its GRPO checkpoint before submission. RL training is not started by the preparation step.

The training subset contains 5,000 COCO train2017 images; the validation subset contains 256 val2017 images. Both use seed 42. For each split, rank every official image ID by the SHA-256 digest of the UTF-8 string `coco2017:{split}:seed=42:image_id={id}`, breaking ties by numeric ID, take the first 5,000 or 256, and store the selected IDs in ascending order. The committed `rl_train_manifest.json` and `rl_val_manifest.json` record the IDs and SHA-256 of the exact source annotation file. `prepare_rl_subset.py` checks an existing manifest against the source before generating machine-specific image paths. Annotations with `iscrowd=1` and invalid boxes are excluded; boxes are clipped and normalized to 0–1000. All selected images are retained, including those with no eligible objects.

The GRPO configuration uses four sampled completions per prompt, one epoch, learning rate 1e-5, KL coefficient 0.01, and a single A100 80 GB. Training batch is one completion per microbatch with four accumulation steps; validation uses four completions per batch. It logs validation reward every 100 optimizer steps and saves checkpoints at the same interval. The sample-based validation reward is not COCO AP; run the separate COCO evaluator for the final metrics. Each completion is scored by strict JSON validity plus a class-aware greedy matching of boxes at IoU >= 0.5: `0.1 + 0.7 * F1 + 0.2 * mean_matched_IoU`; invalid JSON scores zero, and a correct empty prediction for an empty image scores one. Invalid detections count as false positives. The 768-token completion cap may truncate dense scenes. Full training time and GPU memory use have not yet been measured.

## Files

- `setup_env.sh`: installs pinned training/evaluation Python packages into a project-local virtual environment while reusing the site's CUDA PyTorch.
- `download_model.sh`: downloads the base model from ModelScope.
- `download_coco.sbatch`, `finish_coco_modelscope.sh`, and the two range downloaders: resumable dataset download, checksum validation, extraction, and image-count checks.
- `prepare_coco_jsonl.py`: converts COCO instance annotations to Qwen3-VL grounding JSONL.
- `prepare_rl_subset.py`, `rl_train_manifest.json`, `rl_val_manifest.json`: deterministically reproduce GRPO subsets and create local JSONL files.
- `coco_grpo_reward.py`, `train_grpo.sbatch`, `compat/run_swift_rlhf.py`: detection reward, LoRA GRPO job, and compatibility launcher.
- `train_and_evaluate.sbatch`: one-GPU LoRA SFT and end-to-end validation.
- `evaluate_coco.py`: resumable detection generation and COCO metric calculation, with optional image sharding.
- `evaluate_base_full.sbatch`: full 5,000-image COCO validation for the unmodified Qwen3-VL-2B base model on one available 16 GB V100.
- `resume_coco_eval.sbatch`, `split_coco_predictions.py`, and `merge_coco_predictions.py`: resume interrupted validation generation across available GPUs and calculate metrics on the merged predictions.
- `download_coco.sh`: sequential mirror fallback with official SHA-1 verification.

Large files (COCO archives/images/annotations, model weights, checkpoints, generated predictions, metrics, logs, virtual environments, and caches) are excluded by `.gitignore`.
