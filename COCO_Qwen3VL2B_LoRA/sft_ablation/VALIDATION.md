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
