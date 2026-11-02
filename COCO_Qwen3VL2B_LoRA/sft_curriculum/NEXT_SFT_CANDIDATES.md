# Further SFT candidates after M30

These are proposals, NOT submitted experiments. User authorized M30 and M45
selection repair only in this turn. Keep COCO train/val, Qwen3-VL-2B, Rex original
IDs and current pooled-count evaluation. Changes require separate matched runs;
do not combine several ideas before measuring individual effects.

| Candidate | Concrete change | Provenance / caveat |
| --- | --- | --- |
| Image-balanced NTP loss | Average supervised-token CE within each image, then average across images, rather than letting long answers dominate the batch-token average | New engineering ablation, not assigned a paper method number; verify current Trainer token normalization first |
| Language+visual-projector LoRA | Compare language-only baseline against adding LoRA on the visual merger/projector, keeping the vision backbone frozen | New parameter-scope ablation; maintain original coordinate-ID rows and matched data/update budget |
| Actual-resolution / multi-scale SFT | Control actual processed grid size, not just increase max_pixels; transform geometry consistently | Related to resolution factor of WeVisOne stage2, isolated adaptation. COCO often lies below max_pixels; increasing only the cap may be a no-op |
| M47 verified absent-category queries | Mix queried absent/nearby classes with regular all80-category detection, using valid Rex empty output | Ferret uses object-existence QA; Rex-empty-answer variant is an adaptation. COCO lacks complete attribute labels; start class-only. Annotation absence alone is not a complete visual absence check; include crowd presence in candidate rejection |
| M40 COCO image stitching | Stitch training images and transform every GT without dropping targets | ScaleCUA source is GUI; natural-image adaptation. Control processed image budget and instance count; not equivalent to M38 shrinking/padding |
| M50 token-to-object attention supervision | Map each box's answer span to its own GT-derived patch map, add KL term to NTP | Direct Visual Grounding evaluated single-box REC/other tasks. Multi-box adaptation needs explicit token/GT alignment and80GB memory preflight; author repo does not establish a ready-made Qwen3-VL multi-box trainer |
| Class-balanced whole-image sampling | Sample whole images with class-frequency-derived weights, retaining every GT | New sampling ablation distinct from M36 spatial balancing; may favor rare classes at the expense of pooled micro F1; report per-class diagnostics separately |

Sources checked2026-11-02:

- https://arxiv.org/html/2610.31329v1#S4.SS1
- https://arxiv.org/html/2310.07704#S4.SS3
- https://arxiv.org/html/2509.15221v2
- https://arxiv.org/html/2511.12738
- https://github.com/ParsaKhani/visual-grounding-datasets
- https://github.com/facebookresearch/detr/blob/main/datasets/coco.py (reference
  implementation for geometry-consistent flip/resize augmentation, another simple
  candidate if needed; do not imply it is an original item in M01-M50).

M01 basic x-order, M04 direct output and M39 normalized coordinates are already
largely present. M17 belongs to another agent and should not be duplicated.
M48 is multi-task mixture sampling; it does not directly apply to this one-task
COCO experiment unless the task mixture is intentionally expanded.
