> HISTORICAL METRIC SNAPSHOT. Superseded by [micro_current.md](micro_current.md); do not use these scores for current comparisons.

> Confirmed upstream metric issue: see ../audits/REX_F1_ROOT_CAUSE_20261102.md. These are unmodified author scores, not global endpoint F1.

# Rex-Omni official evaluator replay

Updated: 2026-11-02T18:24:06.185514+08:00.
Unmodified author FastEval and safe_mean/f1. Primary retains all outputs of our all80-class prompt. GT-positive-only reference is separate. Values in percent.

| Source | Images | Coverage | P mIoU | R mIoU | F1 mIoU | F1@.50 | F1@.95 | Positive-only F1 mIoU |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| evaluation/base_val2017_first1600/predictions.jsonl | 1600 | partial | 44.15 | 23.98 | 31.08 | 39.47 | 13.58 | 31.47 |
| evaluation/base_val2017_full/predictions.jsonl | 5000 | complete | 37.44 | 22.21 | 27.88 | 35.43 | 11.24 | 28.52 |
| evaluation/progress_val256/global16_step100/predictions.jsonl | 256 | complete | 56.12 | 44.44 | 49.60 | 64.66 | 26.19 | 50.97 |
| evaluation/progress_val256/global16_step200/predictions.jsonl | 256 | complete | 56.03 | 44.97 | 49.89 | 65.63 | 24.62 | 51.23 |
| evaluation/progress_val256/global16_step300/predictions.jsonl | 256 | complete | 57.37 | 44.98 | 50.43 | 66.24 | 25.74 | 51.75 |
| evaluation/progress_val256/object_credit_v1_step100/predictions.jsonl | 256 | complete | 56.23 | 44.26 | 49.53 | 64.35 | 25.88 | 50.84 |
| evaluation/progress_val256/object_credit_v1_step200/predictions.jsonl | 256 | complete | 56.62 | 44.55 | 49.86 | 65.20 | 25.13 | 51.25 |
| evaluation/progress_val256/object_credit_v1_step300/predictions.jsonl | 256 | complete | 57.70 | 44.79 | 50.43 | 66.72 | 25.01 | 51.70 |
| evaluation/progress_val256/sft/predictions.jsonl | 256 | complete | 55.54 | 44.07 | 49.15 | 64.20 | 25.22 | 50.69 |
| evaluation/rex_id_sft_outputs_5epoch_b64/epoch1_val256/predictions.jsonl | 256 | complete | 57.59 | 40.50 | 47.55 | 65.16 | 25.84 | 48.92 |
| evaluation/rex_id_sft_outputs_5epoch_b64/epoch2_val256/predictions.jsonl | 256 | complete | 55.31 | 44.36 | 49.23 | 67.25 | 26.62 | 50.69 |
| evaluation/rex_id_sft_outputs_5epoch_b64/epoch3_val256/predictions.jsonl | 256 | complete | 59.40 | 45.89 | 51.78 | 69.52 | 27.51 | 53.57 |
| evaluation/rex_id_sft_outputs_5epoch_b64/epoch4_val256/predictions.jsonl | 256 | complete | 58.96 | 45.80 | 51.55 | 67.90 | 28.15 | 53.13 |
| evaluation/rex_id_sft_outputs_5epoch_b64/epoch5_val256/predictions.jsonl | 256 | complete | 57.83 | 46.54 | 51.58 | 68.85 | 27.69 | 53.24 |
| evaluation/rex_id_val2017_a800_80/predictions.jsonl | 5000 | complete | 43.99 | 39.89 | 41.84 | 57.29 | 12.81 | 43.46 |
| evaluation/rex_id_val2017_a800_80/predictions.jsonl#fixed256 | 256 | complete | 55.87 | 44.63 | 49.62 | 64.90 | 25.06 | 51.04 |
| evaluation/rex_token_val2017/predictions.jsonl | 5000 | complete | 43.83 | 40.52 | 42.11 | 57.26 | 13.49 | 43.60 |
| evaluation/sft_ablation/m34/predictions.jsonl | 5000 | complete | 39.42 | 40.60 | 40.00 | 54.81 | 11.98 | 41.88 |
| evaluation/sft_ablation/m36/predictions.jsonl | 5000 | complete | 39.24 | 40.80 | 40.01 | 54.70 | 11.80 | 41.97 |
| evaluation/sft_ablation/m37/predictions.jsonl | 5000 | complete | 45.94 | 39.66 | 42.57 | 58.40 | 12.86 | 44.21 |
| evaluation/val2017/predictions.jsonl | 5000 | complete | 46.93 | 41.89 | 44.27 | 59.60 | 14.50 | 46.33 |
| rex_id_grpo_outputs_global16_v3/coco_val2017/predictions.jsonl | 5000 | complete | 48.91 | 41.33 | 44.80 | 61.72 | 13.39 | 46.17 |
| rex_id_grpo_outputs_object_credit_v1/coco_val2017/predictions.jsonl | 5000 | complete | 48.77 | 41.30 | 44.72 | 61.50 | 13.38 | 46.14 |
| sft_ablation_outputs/m34/progress_eval/step-10000/predictions.jsonl | 256 | complete | 51.43 | 42.31 | 46.43 | 61.44 | 24.40 | 47.44 |
| sft_ablation_outputs/m34/progress_eval/step-5000/predictions.jsonl | 256 | complete | 53.72 | 37.22 | 43.97 | 58.40 | 20.39 | 44.76 |
| sft_ablation_outputs/m36/progress_eval/step-10000/predictions.jsonl | 256 | complete | 51.36 | 43.18 | 46.91 | 63.10 | 24.44 | 49.13 |
| sft_ablation_outputs/m36/progress_eval/step-5000/predictions.jsonl | 256 | complete | 51.15 | 39.47 | 44.56 | 60.94 | 23.27 | 46.63 |
| sft_ablation_outputs/m37/progress_eval/step-10000/predictions.jsonl | 256 | complete | 59.67 | 41.34 | 48.84 | 65.21 | 25.31 | 49.83 |
| sft_ablation_outputs/m37/progress_eval/step-5000/predictions.jsonl | 256 | complete | 54.33 | 38.37 | 44.97 | 62.18 | 22.75 | 46.68 |
| sft_ablation_outputs/m38/progress_eval/step-5000/predictions.jsonl | 256 | complete | 54.81 | 40.06 | 46.29 | 61.31 | 27.27 | 47.50 |
| inference/m06/baseline | 256 | complete | 56.03 | 43.87 | 49.21 | 64.39 | 26.90 | 50.88 |
| inference/m06/best16_confidence | 256 | complete | 56.96 | 40.93 | 47.63 | 65.16 | 23.00 | 49.01 |
| inference/m06/best16_order | 256 | complete | 56.96 | 40.93 | 47.63 | 65.16 | 23.00 | 49.01 |
| inference/m06/best1_confidence | 256 | complete | 53.73 | 39.53 | 45.55 | 64.11 | 19.78 | 47.28 |
| inference/m06/best1_order | 256 | complete | 53.73 | 39.53 | 45.55 | 64.11 | 19.78 | 47.28 |
| inference/m06/best4_confidence | 256 | complete | 53.67 | 39.63 | 45.59 | 63.73 | 22.70 | 47.29 |
| inference/m06/best4_order | 256 | complete | 53.67 | 39.63 | 45.59 | 63.73 | 22.70 | 47.29 |
| inference/m06/best8_confidence | 256 | complete | 56.16 | 40.20 | 46.86 | 64.76 | 21.65 | 48.54 |
| inference/m06/best8_order | 256 | complete | 56.16 | 40.20 | 46.86 | 64.76 | 21.65 | 48.54 |
| inference/m06/greedy_confidence | 256 | complete | 56.03 | 43.87 | 49.21 | 64.39 | 26.90 | 50.88 |
| inference/m06/sample0_order | 256 | complete | 53.73 | 39.53 | 45.55 | 64.11 | 19.78 | 47.28 |
| inference/m44/baseline | 256 | complete | 55.53 | 44.79 | 49.58 | 64.24 | 25.21 | 51.19 |
| inference/m44/zoom_always | 256 | complete | 50.99 | 39.28 | 44.38 | 59.74 | 19.88 | 45.89 |
| inference/m44/zoom_consistency | 256 | complete | 51.10 | 40.02 | 44.88 | 60.54 | 19.92 | 46.25 |
| inference/m45/baseline | 256 | complete | 56.03 | 43.87 | 49.21 | 64.39 | 26.90 | 50.88 |
| inference/m45/t0.5_member | 256 | complete | 29.86 | 48.81 | 37.05 | 48.47 | 21.40 | 43.77 |
| inference/m45/t0.5_sample0 | 256 | complete | 55.41 | 44.73 | 49.50 | 66.81 | 22.83 | 51.04 |
| inference/m45/t0.5_vote | 256 | complete | 26.88 | 37.52 | 31.32 | 42.97 | 14.93 | 36.78 |
| inference/m45/t0.7_member | 256 | complete | 24.22 | 48.24 | 32.25 | 41.63 | 20.40 | 40.87 |
| inference/m45/t0.7_sample0 | 256 | complete | 55.72 | 43.25 | 48.70 | 66.99 | 25.30 | 50.38 |
| inference/m45/t0.7_vote | 256 | complete | 21.25 | 34.82 | 26.39 | 36.08 | 8.01 | 32.98 |
| inference/m46/baseline | 256 | complete | 56.03 | 43.87 | 49.21 | 64.39 | 26.90 | 50.88 |
| inference/m46/mcc | 256 | complete | 56.84 | 43.61 | 49.36 | 64.66 | 26.16 | 50.96 |
