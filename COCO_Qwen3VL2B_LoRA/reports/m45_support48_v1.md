# M45 additional 48/64 support threshold

rex_matching_global_endpoint_micro_v1

Post-hoc threshold extension requested after inspecting 16/32 results; exploratory on the same 256 images.

Same frozen 64 generations per image, T=.5, member box, NMS IoU=.7. No new inference.

| Variant | Boxes | P mIoU | R mIoU | F1 mIoU | F1@.50 | F1@.95 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2020 | 34.88 | 36.10 | 35.48 | 51.89 | 9.37 |
| t0.5_member_s16_nms70 | 1754 | 42.68 | 37.99 | 40.20 | 63.01 | 8.05 |
| t0.5_member_s32_nms70 | 1273 | 53.46 | 35.25 | 42.49 | 64.38 | 9.11 |
| t0.5_member_s48_nms70 | 1030 | 60.58 | 32.86 | 42.61 | 61.76 | 9.65 |
