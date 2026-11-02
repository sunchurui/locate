# Fixed256 versus full5000 AP audit (2026-11-02)

CPU-only Slurm job207033 completed successfully in1m27s. No model inference,
GPU allocation, training update or change to existing predictions was involved.

The seed42 manifest is exactly reconstructed by sorting all5000 official
val2017 image IDs by SHA256 of `coco2017:val2017:seed=42:image_id={id}`,
taking256 and sorting selected IDs numerically. Annotation SHA256 matches.
Selection uses no scores, labels, model outputs or image difficulty.

| Prediction source | Images scored | AP points |
| --- | ---: | ---: |
| Saved full-validation baseline predictions |5000|29.5431|
| Exact same saved predictions, fixed subset only |256|35.4602|
| Separately generated historical progress baseline |256|35.2209|
| Inference-ablation paired baseline |256|35.4404|
| M45 temperature0.5 sampled-member adaptation |256|39.9375|

The same-prediction subset/full difference is5.9171 points, with no regeneration.
The separately generated inference baseline differs from the subset extracted
from full predictions by only-0.0197 AP, although individual detections differ.
This does not establish that hardware/parser differences never matter.

Fifty additional256-image subsets use the identical selection rule with
seeds1000..1049. Scoring the same full-validation predictions gives mean35.7833,
median35.5245, minimum31.8780 and maximum40.0701 AP.26/50 subsets score at least
as high as the fixed seed42 subset. These are distinct, possibly overlapping
subsets, not confidence intervals for M45's gain.

The fixed subset has78 noncrowd-GT categories, missing hair drier and toothbrush;
nine present categories have fewer than5 instances. Full5000 has all80, and no
category has fewer than5 noncrowd instances. Restricting full5000 to those78
categories raises AP only to29.8442, so dropping two categories accounts for
only about0.30 points of the observed difference. Example class APs from the
same predictions: snowboard25.1291 on full5000 versus80.0 on the fixed subset
with one GT instance; scissors41.1296 versus90.0 with two GT instances.

Mean objects/image is7.2670 versus7.2266; small-object share42.0091% versus
42.5946%. The coarse object-count/size distributions are similar.

COCO AP pools and ranks detections by category, interpolates precision/recall
and then averages across valid categories/IoUs; it is not an image-wise mean.
On this prediction set, small-subset AP is consistently optimistic. This
supports a finite-sample explanation, not a claim that all models/datasets
have the same bias or that every cause has been quantitatively isolated.

Use256 images for paired screening/progress checks. Do not extrapolate M45's
39.94 to full-val AP or assume its+4.50 gain transfers to5000 images. Full-val
evaluation is required for the final claim. Raw audit: runtime
`results/subset_audit_20261102/audit.json`; reproducible entry: `audit_subset.py`.
