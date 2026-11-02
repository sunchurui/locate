# Actual validation evidence, 2026-11-01

- All118287 original training answers exactly match the pinned Rex formatter.
- Each variant has118287 training records. M34 retains all instances of queried
  categories; M36 retains complete original records; M37 leaves all targets and
  pixels unchanged; M38 retains every original category/instance after geometry.
- Original and replacement unmodified collators produce bitwise equal input IDs,
  labels, attention masks, image pixels and grids on a real4-image batch.
- All four methods pass real image/processor batches. Actual chat-template
  expansion verifies M37 text-first and the other three image-first.
- An actual image overlay verifies M38 transformed GT against resized/padded
  image geometry. This can be regenerated with `audit.py`.
- M36 original center-distribution KL-to-uniform0.232273, weighted expectation
  0.164468, realized resample0.166223. There are118287 draws and72839 distinct
  images; repetition is intentional. Every drawn image keeps all its GT.
- M34 job206035 completed two real optimizer steps on NVIDIA A10080GB PCIe.
  Smoke losses11.6288/11.5993; gradient norms16.7643/17.0330, all gradients and
  final trainable parameters finite. Short8-example smoke peak7.872GiB does not
  estimate full-training peak on longer examples.
- Smoke and formal M34 initialize trainable tensors with the same SHA256:
  `53e2abac60e4091d008362ba926f6144b2b2a311ca6671d3045d4202fd03c074`.
  Formal M34 passed160/14786 updates at the last recorded check with finite
  logged loss/gradient norm. This is startup validation, not an efficacy result.
- M37/M38/M36 were queued at this check; their two-step GPU smoke checks run
  inside their own >=80GB allocation before formal training. GPU validation is
  not yet claimed for those three methods.

Source/data checksums and real-batch shapes are in `data_report.json` and
`data_audit.json`. Runtime paths and jobs are in the experiment's `jobs.json`.
Training/evaluation are independent per method; no M17 code/job is changed.

## 2026-11-02 status and recovery

- M34 (206035), M37 (206037), M36 (206041) completed14786 updates/one epoch.
  Wall times:6h51m59s,4h33m03s,4h01m26s respectively.
- Their step10000 fixed256-image APs are33.3128,34.1845,33.3480.
  These are intermediate checkpoints, not final/full5000-image scores.
  The historical final SFT on the same256-image manifest scores35.2209.
  Unequal training stages must not be interpreted as final ablation effects.
- Original M38 (206039) reached approximately6000 updates. Its last complete
  checkpoint is5000, with fixed256 AP32.3543. Multiprocessing could not create
  `/tmp/pymp-*` due to node-local disk exhaustion, then the job timed out at12h.
  All original checkpoints are retained.
- Standalone evals206036/206038/206042 failed before generation because of an
  unused access to a missing top-level `Qwen3VLConfig.use_cache` attribute.
  Removed that access. Corrected full5000 eval jobs:206993/206994/206995.
- M38 fresh retry207004 uses the same model, data, seed, learning settings and
  installed packages; A100PCIe80, isolated output `m38_retry_20261102`, home-backed
  per-job TMPDIR and0 dataloader workers. Original GPU smoke is retained and
  not repeated. Dependent full evaluation207005 loads the retry's own adapter.
  Earlier pending retry206996/evaluation206997 were cancelled before execution.
  SCIR requires a typed GPU request and rejected an attempt to request an
  untyped GPU restricted to known80GB nodes; no such job was created.
  Corrected M34/M37/M36 evaluation requests have8h limits for backfill.
- Runtime registries preserve failed IDs/reasons as history. The stuck original
  M38 evaluation206040 was cancelled; no model/data/checkpoint was deleted.
  No other experiment, including M17 and five-epoch training, was modified.
