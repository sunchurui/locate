# COCO 2017 + Qwen3-VL-2B LoRA baseline

A reproducible single-GPU supervised fine-tuning (SFT) baseline and fixed reinforcement-learning subset for COCO 2017 object detection using `Qwen/Qwen3-VL-2B-Instruct` and ms-swift. A GroundingPI-inspired adaptation of Rex-Omni's RL code is in progress and is not yet a runnable training recipe.

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

The training subset contains 10,000 COCO train2017 images; the validation subset contains 256 val2017 images. Both use seed 42. Rank every official image ID by the SHA-256 digest of the UTF-8 string `coco2017:{split}:seed=42:image_id={id}`, breaking ties by numeric ID, take the first 10,000 or 256, and store the selected IDs in ascending order. The 10,000-image subset contains all images from the earlier 5,000-image selection. `rl_train_10000_manifest.json` and `rl_val_manifest.json` record the IDs and SHA-256 of the exact source annotation files. The older 5,000-image manifest is retained for provenance.

`prepare_rl_subset.py` checks an existing manifest against the source before generating machine-specific paths and annotations. For example, to reproduce the training JSONL from the experiment root:

```bash
python prepare_rl_subset.py --instances data/coco2017/annotations/instances_train2017.json \
  --image-dir data/coco2017/train2017 --split train2017 --count 10000 --seed 42 \
  --manifest rl_train_10000_manifest.json --output data/coco2017/annotations/train_rl_10000.jsonl
```

Annotations with `iscrowd=1` and invalid boxes are excluded; boxes are clipped and normalized to 0–1000. All selected images are retained, including those with no eligible objects. This JSONL is a dataset preparation output, not a GroundingPI training cache.

The intended RL method follows [GroundingPI section 7.6](https://arxiv.org/html/2609.39601): initialize from SFT with a frozen SFT reference; use GRPO with eight completions per prompt, beta 0.02, clip epsilon 0.2, and separately normalized set-completeness and strict-localization rewards weighted 0.7 and 0.3. Freeze the vision encoder and projector. The paper uses learning rate 5e-7, a constant schedule without warmup, weight decay 0.01, seed/data seed 20260812, and 256 GPUs. Its 256-GPU full-language update is not the requested Qwen3-VL 2B LoRA setup; batch size and memory requirements must be adapted and verified when an appropriate implementation is available. Validation during training and final COCO AP evaluation remain requirements.

GroundingPI's [published VLM configuration](https://github.com/groundingpi/GroundingPI/blob/main/configs/train/vlm.yaml) uses `swift_module: sft`, and its [training parameter validation](https://github.com/groundingpi/GroundingPI/blob/main/train/release_parameters.py) only accepts that VLM module. The official RL entrypoint and the exact two grounding reward functions have not been located. `rex_groundingpi_adaptation/` contains an incomplete adaptation of Rex-Omni's official GRPO trainer and a two-component reward plugin, with explicit approximations and integration gaps. The earlier independently written GRPO job and reward have been withdrawn from the repository; their HPC copies are preserved under `legacy_previous_grpo/`. Do not start RL training from either prototype until the integration checks are complete.

## Files

- `setup_env.sh`: installs pinned training/evaluation Python packages into a project-local virtual environment while reusing the site's CUDA PyTorch.
- `download_model.sh`: downloads the base model from ModelScope.
- `download_coco.sbatch`, `finish_coco_modelscope.sh`, and the two range downloaders: resumable dataset download, checksum validation, extraction, and image-count checks.
- `prepare_coco_jsonl.py`: converts COCO instance annotations to Qwen3-VL grounding JSONL.
- `prepare_rl_subset.py`, `rl_train_10000_manifest.json`, `rl_val_manifest.json`: deterministically reproduce RL subsets and create local JSONL files.
- `rex_groundingpi_adaptation/`: incomplete Rex-Omni GRPO trainer patch and GroundingPI-inspired rewards; see its README for limitations.
- `train_and_evaluate.sbatch`: one-GPU LoRA SFT and end-to-end validation.
- `rex_token_sft/`: a separate Qwen3-VL 2B LoRA experiment using Rex-Omni's coordinate-token output and COCO evaluation; see its README.
- `evaluate_coco.py`: resumable detection generation and COCO metric calculation, with optional image sharding.
- `evaluate_base_full.sbatch`: full 5,000-image COCO validation for the unmodified Qwen3-VL-2B base model on one available 16 GB V100.
- `resume_coco_eval.sbatch`, `split_coco_predictions.py`, and `merge_coco_predictions.py`: resume interrupted validation generation across available GPUs and calculate metrics on the merged predictions.
- `download_coco.sh`: sequential mirror fallback with official SHA-1 verification.

Large files (COCO archives/images/annotations, model weights, checkpoints, generated predictions, metrics, logs, virtual environments, and caches) are excluded by `.gitignore`.
