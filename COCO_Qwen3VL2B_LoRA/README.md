# COCO 2017 + Qwen3-VL-2B LoRA baseline

A reproducible single-GPU supervised fine-tuning (SFT) baseline for COCO 2017 object detection using `Qwen/Qwen3-VL-2B-Instruct` and ms-swift. The current workflow covers LoRA SFT and COCO validation; it does not yet implement the separate reinforcement-learning stage.

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
- Image token cap 1024; maximum sequence length 4096.

The Slurm job runs data preparation, SFT, validation-loss evaluation, then greedy generation on COCO val and `pycocotools.COCOeval` for AP@[.50:.95], AP50, AP75, size-specific AP, and AR. The detection score uses generated object order because the target format has no calibrated confidence value; this caveat is written to the metrics JSON.

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

## Files

- `setup_env.sh`: installs pinned training/evaluation Python packages into a project-local virtual environment while reusing the site's CUDA PyTorch.
- `download_model.sh`: downloads the base model from ModelScope.
- `download_coco.sbatch`, `finish_coco_modelscope.sh`, and the two range downloaders: resumable dataset download, checksum validation, extraction, and image-count checks.
- `prepare_coco_jsonl.py`: converts COCO instance annotations to Qwen3-VL grounding JSONL.
- `train_and_evaluate.sbatch`: one-GPU LoRA SFT and end-to-end validation.
- `evaluate_coco.py`: generates detections and calculates COCO metrics.
- `download_coco.sh`: sequential mirror fallback with official SHA-1 verification.

Large files (COCO archives/images/annotations, model weights, checkpoints, generated predictions, metrics, logs, virtual environments, and caches) are excluded by `.gitignore`.
