# RL migration clarification — 2026-11-07

The normalized response-local object-credit implementation is in
`rex_id_object_credit/normalization.py` with `config_mcr_norm.json`. The
paired epoch3 initialization is in `rex_id_epoch3_rl/`: both policy and
reference start from `rex_id_sft_outputs_5epoch_b64/checkpoint-5547`.
Job207730 was cancelled before starting; there is no trained normalized RL
checkpoint. Job207096 was a CPU-only replay verification, not RL training.

Two completed old-base RL checkpoint-313 directories are backed up locally:
`old_global` and `old_object_credit`. They start from the one-epoch SFT, and
the object-credit version is v1 without response-local normalization. Their
adapter weights are not the epoch3 SFT base and must not be used to initialize
the new paired comparison. Reuse checkpoint-5547 plus Qwen3-VL-2B-Instruct.

The files under `rex_groundingpi_adaptation/` include an older veRL prototype
for provenance; the runnable normalized recipe uses ms-swift under
`rex_id_object_credit/` and `rex_id_epoch3_rl/`. The progress watcher scripts
are preserved as source only, not started by this publication.

The project instructions now require the pinned original-author evaluator;
the archived PROJECT_MEMORY.md / EVALUATION_PROTOCOL.md and compatibility
entrypoints describe a different historical micro aggregation. Before any new
evaluation, resolve entrypoints against current user instructions and preserve
all 80-class outputs; do not label the micro reducer as official. Existing
results and raw predictions remain untouched.

No training or evaluation was launched by this migration. GPU training requires
at least 80GB VRAM. Model weights, checkpoints, datasets and credentials are
never committed to this repository. See PLATFORM_MIGRATION.md for setup.
