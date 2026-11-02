> Current metric: rex_matching_global_endpoint_micro_v1. Scores in percent. Saved-prediction replay; partial generation is explicitly marked.

# Global endpoint micro P/R/F1

Current project metric: original Rex matching with pooled TP/FP/FN, no PR reduction or category macro averaging. Values in percent. Historical author scores are not current results.

| Source | Images | Coverage | P mIoU | R mIoU | F1 mIoU | F1@.50 | F1@.95 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| evaluation/base_val2017_first1600/predictions.jsonl | 1600 | partial | 25.30 | 18.51 | 21.38 | 29.04 | 6.86 |
| evaluation/base_val2017_full/predictions.jsonl | 5000 | complete | 24.39 | 18.00 | 20.71 | 28.28 | 6.42 |
| evaluation/epoch3_rl/base_full/predictions.jsonl | 1248 | partial | 41.13 | 37.96 | 39.48 | 56.62 | 10.81 |
| evaluation/progress_val256/global16_step100/predictions.jsonl | 256 | complete | 38.57 | 36.57 | 37.54 | 54.29 | 9.74 |
| evaluation/progress_val256/global16_step200/predictions.jsonl | 256 | complete | 40.24 | 37.43 | 38.78 | 56.30 | 10.01 |
| evaluation/progress_val256/global16_step300/predictions.jsonl | 256 | complete | 41.62 | 37.04 | 39.19 | 57.16 | 10.10 |
| evaluation/progress_val256/object_credit_v1_step100/predictions.jsonl | 256 | complete | 37.81 | 36.94 | 37.37 | 54.07 | 9.28 |
| evaluation/progress_val256/object_credit_v1_step200/predictions.jsonl | 256 | complete | 39.80 | 37.19 | 38.45 | 55.99 | 9.71 |
| evaluation/progress_val256/object_credit_v1_step300/predictions.jsonl | 256 | complete | 42.28 | 36.92 | 39.42 | 57.59 | 9.79 |
| evaluation/progress_val256/sft/predictions.jsonl | 256 | complete | 35.06 | 35.61 | 35.33 | 51.56 | 9.25 |
| evaluation/rex_id_sft_outputs_5epoch_b64/epoch1_val256/predictions.jsonl | 256 | complete | 38.83 | 31.63 | 34.86 | 53.28 | 8.06 |
| evaluation/rex_id_sft_outputs_5epoch_b64/epoch2_val256/predictions.jsonl | 256 | complete | 34.55 | 35.02 | 34.79 | 52.08 | 8.78 |
| evaluation/rex_id_sft_outputs_5epoch_b64/epoch3_val256/predictions.jsonl | 256 | complete | 41.91 | 38.61 | 40.19 | 58.98 | 10.40 |
| evaluation/rex_id_sft_outputs_5epoch_b64/epoch4_val256/predictions.jsonl | 256 | complete | 37.49 | 37.57 | 37.53 | 54.99 | 9.99 |
| evaluation/rex_id_sft_outputs_5epoch_b64/epoch5_val256/predictions.jsonl | 256 | complete | 39.45 | 38.46 | 38.95 | 57.13 | 10.76 |
| evaluation/rex_id_sft_outputs_5epoch_b64_full/checkpoint-9245/predictions.jsonl | 960 | partial | 39.72 | 38.61 | 39.16 | 55.65 | 11.51 |
| evaluation/rex_id_val2017_a800_80/predictions.jsonl | 5000 | complete | 36.63 | 34.46 | 35.51 | 50.64 | 9.72 |
| evaluation/rex_id_val2017_a800_80/predictions.jsonl#fixed256 | 256 | complete | 35.70 | 35.26 | 35.48 | 52.03 | 8.93 |
| evaluation/rex_token_val2017/predictions.jsonl | 5000 | complete | 36.11 | 35.23 | 35.67 | 50.77 | 9.82 |
| evaluation/sft_ablation/m34/predictions.jsonl | 5000 | complete | 28.49 | 35.46 | 31.60 | 45.65 | 8.37 |
| evaluation/sft_ablation/m36/predictions.jsonl | 5000 | complete | 29.62 | 35.80 | 32.42 | 46.66 | 8.38 |
| evaluation/sft_ablation/m37/predictions.jsonl | 5000 | complete | 38.01 | 34.42 | 36.13 | 51.69 | 9.86 |
| evaluation/val2017/predictions.jsonl | 5000 | complete | 41.59 | 38.83 | 40.16 | 56.58 | 11.72 |
| rex_id_grpo_outputs_global16_v3/coco_val2017/predictions.jsonl | 5000 | complete | 44.04 | 37.09 | 40.27 | 57.74 | 10.79 |
| rex_id_grpo_outputs_object_credit_v1/coco_val2017/predictions.jsonl | 5000 | complete | 43.91 | 37.04 | 40.18 | 57.55 | 10.76 |
| sft_ablation_outputs/m34/progress_eval/step-10000/predictions.jsonl | 256 | complete | 27.72 | 34.38 | 30.69 | 46.11 | 7.87 |
| sft_ablation_outputs/m34/progress_eval/step-5000/predictions.jsonl | 256 | complete | 25.22 | 28.58 | 26.79 | 40.93 | 5.77 |
| sft_ablation_outputs/m36/progress_eval/step-10000/predictions.jsonl | 256 | complete | 27.22 | 34.21 | 30.32 | 45.37 | 7.18 |
| sft_ablation_outputs/m36/progress_eval/step-5000/predictions.jsonl | 256 | complete | 30.15 | 30.85 | 30.49 | 46.74 | 6.89 |
| sft_ablation_outputs/m37/progress_eval/step-10000/predictions.jsonl | 256 | complete | 37.74 | 33.11 | 35.27 | 52.27 | 9.52 |
| sft_ablation_outputs/m37/progress_eval/step-5000/predictions.jsonl | 256 | complete | 31.83 | 29.57 | 30.66 | 47.19 | 6.88 |
| sft_ablation_outputs/m38/progress_eval/step-5000/predictions.jsonl | 256 | complete | 33.21 | 32.07 | 32.63 | 49.18 | 7.29 |
| inference/m06/baseline | 256 | complete | 34.88 | 36.10 | 35.48 | 51.89 | 9.37 |
| inference/m06/best16_confidence | 256 | complete | 41.90 | 33.48 | 37.22 | 56.70 | 7.37 |
| inference/m06/best16_order | 256 | complete | 41.90 | 33.48 | 37.22 | 56.70 | 7.37 |
| inference/m06/best1_confidence | 256 | complete | 38.55 | 33.86 | 36.06 | 56.00 | 6.87 |
| inference/m06/best1_order | 256 | complete | 38.55 | 33.86 | 36.06 | 56.00 | 6.87 |
| inference/m06/best4_confidence | 256 | complete | 40.25 | 33.35 | 36.48 | 56.55 | 7.18 |
| inference/m06/best4_order | 256 | complete | 40.25 | 33.35 | 36.48 | 56.55 | 7.18 |
| inference/m06/best8_confidence | 256 | complete | 41.07 | 33.23 | 36.74 | 56.03 | 7.31 |
| inference/m06/best8_order | 256 | complete | 41.07 | 33.23 | 36.74 | 56.03 | 7.31 |
| inference/m06/greedy_confidence | 256 | complete | 34.88 | 36.10 | 35.48 | 51.89 | 9.37 |
| inference/m06/sample0_order | 256 | complete | 38.55 | 33.86 | 36.06 | 56.00 | 6.87 |
| inference/m44/baseline | 256 | complete | 34.89 | 35.95 | 35.41 | 51.29 | 9.12 |
| inference/m44/zoom_always | 256 | complete | 31.70 | 32.70 | 32.19 | 48.83 | 5.71 |
| inference/m44/zoom_consistency | 256 | complete | 31.35 | 32.34 | 31.83 | 48.89 | 5.28 |
| inference/m45/baseline | 256 | complete | 34.88 | 36.10 | 35.48 | 51.89 | 9.37 |
| inference/m45/t0.5_member | 256 | complete | 7.30 | 42.75 | 12.48 | 20.67 | 2.34 |
| inference/m45/t0.5_sample0 | 256 | complete | 39.95 | 36.70 | 38.26 | 57.55 | 8.31 |
| inference/m45/t0.5_vote | 256 | complete | 5.18 | 30.19 | 8.84 | 16.11 | 1.13 |
| inference/m45/t0.7_member | 256 | complete | 5.94 | 42.42 | 10.43 | 18.19 | 1.65 |
| inference/m45/t0.7_sample0 | 256 | complete | 39.98 | 35.58 | 37.65 | 56.86 | 8.32 |
| inference/m45/t0.7_vote | 256 | complete | 3.95 | 28.12 | 6.93 | 13.60 | 0.54 |
| inference/m46/baseline | 256 | complete | 34.88 | 36.10 | 35.48 | 51.89 | 9.37 |
| inference/m46/mcc | 256 | complete | 35.77 | 35.08 | 35.42 | 51.82 | 9.33 |
