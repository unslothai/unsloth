# SentenceTransformer encoder unpadding

`FastSentenceTransformer.from_pretrained(..., use_unpadding="auto")` skips padded
tokens inside supported BERT/RoBERTa encoders during eager CUDA training. The
default is `"auto"`; set `use_unpadding=False` to keep the ordinary padded path
without wrapper overhead on short batches. `"auto"` requires at least 8,192 padded token
slots per encoder call (`batch_size * padded_sequence_length`) and at least one
padding token. `True` also packs smaller eligible batches.

The automatic cutoff is conservative, not a universal hardware crossover. In
MiniLM calibration on an RTX 5090 under WSL2, smaller batches were launch-bound:
packing saved activation memory but could reduce throughput. Long sequences and
larger batches benefited, including a measured case with only 1% padding. Use
`True` when the smaller-batch memory saving matters more than latency, and measure
on your own hardware. Even `"auto"` has small-batch wrapper overhead when enabled,
so use `False` to avoid that overhead. The PR reports every final case, including regressions.
These comparisons measure compaction plus shared FlashAttention against upstream
SDPA; they do not isolate padding removal from the attention backend change.

## Execution and compatibility

The optimization gathers the **original embedding output**, preserving token
types and positional embeddings, then uses bidirectional FlashAttention or xFormers through
Unsloth's existing attention dispatcher. It restores the rectangular output
before the original mean pooling. It does not replace LayerNorm, change weights,
introduce a custom loss, or implement a separate backend detector.

Supported pipelines are an exact Transformers BERT or RoBERTa encoder with
absolute positions, immediately followed by mean-only SentenceTransformer
Pooling and optional Dense/Normalize modules. Transformers 5's attention
interface, CUDA and a working FlashAttention or xFormers packed backend are required.
xFormers uses a noncausal `BlockDiagonalMask` through the same dispatcher used for
LLM packing; causal LLM masks must not be used for these encoders. FlashAttention
retains priority when available. FP16/BF16 execution is required, with BF16 limited
to hardware and operators that support it. CUDA
autocast may retain FP32 normalization/residual activations; those remain FP32.
LoRA and gradient checkpointing keep their existing implementations.

Evaluation, pure FP32 without autocast, Transformers 4, no supported packed backend,
other architectures, custom token-processing modules, requested hidden states
or attention weights, routed keyword mask overrides, feed-forward chunking, unsupported masks and head
dimensions above 256 remain padded. Upstream automatic compilation is unchanged;
compiled whole models and compiled inner encoders use the padded path, including
across graph breaks. Unsloth restores its own original forwards and SDPA backend
before applying its compilation helper, avoiding wrapper overhead on that path.
This does not replace native unpadding in other architectures.

An xFormers installation without bidirectional block masks also stays padded.
The xFormers path conservatively requires head dimensions divisible by eight.
The historical RTX 5090 FlashAttention measurements above do not measure xFormers
performance or calibrate its automatic cutoff. Verify actual packed xFormers calls
and its selected forward/backward operators when benchmarking without FlashAttention.

Attended token embeddings and sentence embeddings are the numerical contract.
Padded encoder positions are zero during packed training. Dropout draws differ
between packed and padded shapes, so numerical comparisons disable dropout;
separate tests exercise dropout-enabled backward.

Standard full-model checkpoints load in stock SentenceTransformers without this
patch. LoRA adapters retain the ordinary base-model dependency: load the base
with stock SentenceTransformers and the adapter with stock PEFT, or merge and
save a standalone model. In the tested Transformers 5.5/ST 6.1 environment,
directly loading an adapter-only Unsloth directory through SentenceTransformer
has an existing upstream `config.json`/missing-full-weights issue; it also fails
on unchanged main and is not fixed here. No final LayerNorm is removed.

## Reproduce a paired benchmark

Use two checkouts, the same environment and local checkpoint, and no concurrent
GPU work or substantial CPU workload. Keep downloads, caches and outputs on your
data drive. `prepare.py` accepts an explicit storage root and pins both sources:

- Model: `sentence-transformers/all-MiniLM-L6-v2` at
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Data: `sentence-transformers/stsb` at
  `ab7a5ac0e35aa22088bdcf23e7fd99b220e53308`; select training pairs with score
  at least 0.7, yielding 2,063 pairs. The parquet SHA-256 and row counts are checked.

```sh
# Choose existing candidate/base checkouts and a directory on your data drive.
WORK=/data/unsloth-unpadding
CANDIDATE=/data/unsloth-candidate
BASELINE=/data/unsloth-main
mkdir -p "$WORK/cache" "$WORK/tmp" "$WORK/results"
export HF_HOME="$WORK/cache/huggingface" TORCH_HOME="$WORK/cache/torch"
export TRITON_CACHE_DIR="$WORK/cache/triton" TMPDIR="$WORK/tmp"
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/inductor"
BENCH="$CANDIDATE/benchmarks/sentence_transformer/benchmark.py"

python "$CANDIDATE/benchmarks/sentence_transformer/prepare.py" \
  --root "$WORK" --download-model
python "$BENCH" --self-test

python "$BENCH" --mode baseline --repo "$BASELINE" \
  --model-path "$WORK/models/all-MiniLM-L6-v2" \
  --max-length 512 --batch-size 16 --padding-fraction 0.75 \
  --output "$WORK/results/main.json"
python "$BENCH" --mode candidate-auto --repo "$CANDIDATE" \
  --model-path "$WORK/models/all-MiniLM-L6-v2" \
  --max-length 512 --batch-size 16 --padding-fraction 0.75 \
  --output "$WORK/results/auto.json"
```

Repeat the identical command with `--mode candidate-off` for an implementation-off
control and `--mode candidate-force` for the explicit memory-oriented policy.
For real tokenization, add `--pairs-json "$WORK/fixtures/stsb-positive-pairs.json"`.
Add `--peft` for rank-8 LoRA, `--compile` for the existing upstream compilation
helper, or `--profile-only` for a separate three-step CPU/CUDA profiler trace.
Timing and profiling must use separate runs. Use a fresh output path for each run.

The measured environment is PyTorch 2.11.0+cu130, Transformers 5.5.0,
SentenceTransformers 6.1.0, PEFT 0.18.1, Triton 3.6.0 and FlashAttention 2.8.1,
on a 32 GB RTX 5090 under WSL2 Linux. FlashAttention binary compatibility must be
verified in your environment; its mere presence does not prove kernel execution.

## Protocol and matrix

Each arm reloads and reseeds the same checkpoint for three repeats: 10 warmup
steps and 30 measured steps per repeat. The workload is stock
MultipleNegativesRankingLoss plus AdamW; one step encodes two batches, so
sequences/sec is twice the pair batch size divided by step time. LoRA targets
`query,key,value,dense`, with rank 8 and alpha 16. BF16 is the default.

CUDA events measure forward/loss, backward and optimizer time. CUDA-synchronized
wall time covers gradient clearing, forward/loss, backward and optimizer, while
excluding event creation and untimed loss checks. Peak allocated and reserved memory
are reset after warmup. Tokenization and host-to-device transfer are recorded
separately, outside resident training-step timing. Real batches exclude sentences
repeated across different pairs and discard incomplete batches consistently across arms.

Wall timing uses `CLOCK_MONOTONIC_RAW` where available, with `perf_counter` as
the portable fallback, and records the selected clock. A per-step check rejects
non-finite/non-positive clocks or CUDA event time exceeding its enclosing wall
window by more than 1% plus 0.05 ms. Retain failed-run logs and rerun the entire
affected arm. On the measured WSL host, the adjusted monotonic clock sometimes
underreported elapsed time by about 9%; raw monotonic, CUDA events and an external
Windows timer agreed. Earlier adjusted-clock timings are not accepted results.
Do not interpret the wall/event difference as an isolated CPU-overhead measurement.

Every forward receives fresh feature dictionaries because SentenceTransformer
adds outputs to them. Exact fixture, initial-state, model-file and source hashes
are recorded. An untimed post-measurement probe checks actual FlashAttention
dispatch and automatic fallback. Compiled runs require an OptimizedModule **and**
successful Dynamo graphs, AOT backend compilations and generated Inductor kernels
after warmup, before timing. Counters reset for each repeat; failed compilations
and graph breaks are recorded. This proves partially compiled execution, not
fullgraph coverage. A wrapper alone can silently fall back under upstream's
compiler error-suppression policy. In WSL, inaccessible inherited Windows PATH
entries can make compiler executable probes fail; use a Linux-only process PATH
and retain any error evidence instead of accepting fallback timings as compiled.
Installation flags alone are not accepted as activation proof.
Keep raw per-step arrays and report medians across the three repeat medians.

Run all three policies (main/off/auto) for the following matrix:

| Pair batch / padded length | Padding or workload |
| --- | --- |
| 16 / 32, 32 / 128, 16 / 512 | 5%, 35%, 75% each |
| 8 / 128, 64 / 128 | 50% |
| 16 / 256, 31 / 256, 32 / 256 | 25%, including below/at the automatic cutoff |
| 16 / 512 | 0%, 1% |
| 32 / 128, LoRA | 5%, 35%, 75% |
| 32 / up to 128, real STSB | Full training and LoRA |
| 32 / 128, compiled | 50%, full training and LoRA |

Also report forced packing for 32/128/75% full training and LoRA, and real STSB.
Do not describe memory-only wins as throughput wins. The initial profiling found
launch overhead and synchronization significant for small shapes. Known-size
indexing and in-place repadding were tested independently and together. The
combination's apparent fixed-order benefit did not hold consistently in an
alternating-order confirmation. The original indexing and repadding remain.
Experimental in-place repadding saved less than 1% allocated memory without
reducing reserved memory, so it was not retained.
Zero-padding automatic fallback retained a measured 2–4% dispatch cost; choose
`False` when the workload never benefits from compaction. Short-batch fallback
timings were noisy, so retain both the full matrix and supplemental controls.

## Correctness checks

The focused tests compare BERT/RoBERTa outputs, attended tokens, losses, every
gradient and actual updates, including FP32 weights under FP16/BF16 autocast,
frozen/unfrozen LoRA bases, checkpoint replay, padding patterns, fallback paths,
compiled execution and fresh-process stock loading. Shared attention tests check
bidirectional sequence isolation while retaining the causal default.

Pretrained mixed-precision checks also use FP32 controls at the same rounded
weights. Near-zero contrastive losses can produce large relative gradient errors
in *both* AMP implementations; retain such diagnostics rather than discarding
them. Dtype-aware accuracy and storage-rounding bounds are used, not bitwise
identity. A passing short training check is not a convergence-quality claim.

```sh
python -m pytest -o addopts= -q \
  tests/python/test_sentence_transformer_unpadding.py \
  tests/utils/test_attention_dispatch_bidirectional.py \
  tests/utils/test_varlen_int32_overflow_guard.py \
  tests/utils/test_packing.py tests/utils/test_attention_masks.py
```

Clearing repository `addopts` is intentional: its normal marker selection excludes
some real-GPU/slow tests. Real FlashAttention cases are skipped explicitly when
the backend is unavailable; simulated dispatch tests are not GPU performance proof.


## xFormers without FlashAttention: Colab T4

The xFormers extension was measured against the earlier PR head `83b568338`,
which required FlashAttention for encoder unpadding. It uses the existing shared
xFormers dispatcher. On a Colab Tesla T4, PyTorch 2.11.0+cu128, xFormers 0.0.35,
Transformers 5.5.0 and SentenceTransformers 6.1.0, real packed forward/backward
used CUTLASS with noncausal block masks. `flash_attn` was absent. The original PR
stayed padded; both arms used the same underlying CUTLASS attention family.

Matched MiniLM training used FP32 weights, FP16 autocast and GradScaler(128),
AdamW at 2e-5, 16 sentence pairs, zero dropout, and no compilation/checkpointing.
Each row has six balanced paired repeats, fresh model copies, six warmup steps,
and three synchronized windows of eight complete training steps per arm.
Timings reuse a fixed real-text batch; they are not complete-epoch throughput.

| Workload | Original PR step ms | Packed xFormers ms | Median paired throughput change | Peak allocated MiB, before/after | Peak reserved MiB, before/after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short, low padding | 41.13 | 48.84 | -15.8% | 447.7 / 447.7 | 500 / 498 |
| Short, fixed padding to 128 | 44.97 | 49.40 | -8.0% | 690.5 / 447.8 | 738 / 514 |
| Long, low padding | 50.38 | 51.80 | -2.8% | 746.0 / 730.6 | 778 / 768 |
| Long, high padding, full training | 93.94 | 53.10 | +77.0% | 1207.7 / 638.1 | 1252 / 688 |
| Long, high padding, LoRA | 92.50 | 63.92 | +45.5% | 926.1 / 458.9 | 962 / 516 |

Short inputs are STS-B pairs with 12-16 tokens; fixed padding is a controlled
stress case. Long inputs are SciFact title/abstract pairs. The high-padding
batch has 3,141 useful tokens in 8,960 slots across both sentence branches.
Conservative 96.875% paired-median throughput intervals are +18.7% to +84.9%
(full) and +11.6% to +51.4% (LoRA). Both clear a 5% throughput gate. Other rows
do not qualify for speed; use automatic mode or disable packing for short inputs.
These measurements use forced packing and do not calibrate a universal cutoff.

Real MiniLM embedding relative-L2 error was 0.063%; parameter/input gradient
errors were approximately 0.26%. First AdamW update errors were 5.46% (full)
and 5.41% (LoRA), accepted with a 6% update bound; they are not exact parity.
The original stricter 5% update gate failed. Short training checks do not establish
long-run quality equivalence. Tiny BERT/RoBERTa tests additionally cover FP16,
FP32 autocast, LoRA, dropout checkpoint replay, fallback and stock save/load.
T4 native BF16 and real FlashAttention cases are skipped. No speed claim is made
for other GPUs, other architectures, compilation or multi-GPU training.

A subsequent matched automatic-versus-forced experiment on the same long/high
fixture left the short title branch padded. Six paired repeats did not establish
an additional full-training speed gain: +2.75%, interval -21.89% to +35.52%.
LoRA improved a further 7.59%, interval +6.88% to +34.73%, with median step time
52.82 to 49.01 ms. This increased peak allocated memory by 32.2 MiB and reserved
memory by 24 MiB versus forced packing. Full training used an additional
42.8 MiB allocated and 40 MiB reserved. These are separate comparisons;
do not multiply their ratios into the original-PR comparison above.
