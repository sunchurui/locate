# A800 80GB execution-batch comparison, 2026-10-31

The original single-GPU run used microbatch 1, accumulation 256, generation
chunk 16. Its first six optimizer updates averaged 198.25 seconds and its
reported CUDA allocation peak was 8.51 GiB. Live device usage was about 10GB.

Two short real-GRPO benchmarks used another A800 80GB on gpu20 while the
original training continued. Each alternative performed two updates from the
same SFT adapter and fixed dataset order. Saving, validation and completion
logging were disabled for these timing runs; startup is excluded below.

| Profile | Train microbatch | Accumulation | Generation chunk | Mean update seconds | Peak allocation GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1 | 256 | 16 | 198.25 (first 6) | 8.51 |
| batch8 | 8 | 32 | 64 | 84.20 | 12.97 |
| batch16 | 16 | 16 | 64 | 83.53 | 19.85 |
| batch32 | 32 | 8 | 128 | 83.42 | 31.76 |
| batch64 | 64 | 4 | 128 | 83.29 | 68.93 |

Benchmark jobs: 203987 (batch8/batch16), 204002 (batch32/batch64). Raw logs,
args and phase timings remain under `speed_benchmarks/JOBID_PROFILE` on HPC.
For batch64's first update, generation took 45.10 seconds, reward scoring
0.11 seconds and encoding plus old/reference forward took 19.27 seconds;
the full update took 85.03 seconds. Reward, gradient norm and KL were finite,
and all profiles passed the 32-image x 8-answer batch guard.

The user requested at least 60GB utilization, so **batch64 is the selected
default**, with a measured 68.93 GiB allocation peak and roughly 2.4x speedup
over baseline. Bigger than batch8 did not give a meaningful additional speed
gain on these two updates; allocation alone is not a throughput measure.
These are short-run timings, not a claim about every possible image/output
length. The same 256-response update, complete-batch GDPO, optimizer and
reward definitions are preserved. Sampling/dropout/reduction order can differ
between execution batches; exact tensor equality is not claimed.

Formal fast training uses a fresh output folder and the same original-ID SFT
initialization, with `PERFORMANCE_PROFILE=batch64` and `SAVE_FIRST_STEP=1`.
It saves a checkpoint after the first update and then follows the standard
50-update cadence. The old baseline logs remain preserved; no model/data files
are deleted. Default maximum training remains 313 updates, with val256 rewards
every 100 updates and complete COCO evaluation after successful training.
