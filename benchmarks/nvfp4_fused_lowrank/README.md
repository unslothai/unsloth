# NVFP4 + low-rank correction: portable benchmark

Throwaway benchmark, not intended to merge. It exists so the kernels can be checked on hardware I
do not have in the test box: RTX 5090, DGX Spark, RTX Spark laptops, and Windows or WSL rather than
Linux.

`bench_fused_lowrank_kernel.py` is self-contained. It imports nothing from this directory, carries
its own copies of both Triton kernels and a pure-torch quantiser fallback, and runs a guarded
per-device preflight before any real work. Copy the single file anywhere and run it.

## What it measures

SVDQuant computes `y = alpha * (Q_A(x) Q_W(R)^T) + (x L1^T) L2^T`. The rank-r branch is cheap in
FLOPs and expensive in memory: unfused it re-reads the M x K activation and does a read-modify-write
of the M x N output, so its cost tracks bandwidth rather than arithmetic.

The fused form appends the branch to the same NVFP4 GEMM as 64 extra contracted columns
(`x' = [x | t]`, `W' = [W | s*L2]`), so the vendor kernel computes both terms in one launch. That
needs no custom GEMM, which is the point: it runs wherever a vendor NVFP4 GEMM runs.

## Running it

```
python bench_fused_lowrank_kernel.py --selftest
python bench_fused_lowrank_kernel.py --window-check --shapes zimage --ranks 16,32,64,128
python bench_fused_lowrank_kernel.py --shapes small --ranks 32
```

Do not use `--quick` for numbers. It is noisy enough to trip the consistency guard on 3 of 4 rows
even on an idle B200.

`--dry-run` compiles for a target arch with no GPU present, which is what the compile matrix below
was produced with.

## Prerequisite on sm_120 (5090, RTX 6000 PRO), and the error that misleads you

flashinfer JITs `compute_120f` and needs nvcc >= 12.9. Under 12.8 it logs the real cause at INFO
level, `SM 12.x requires CUDA >= 12.9`, swallows it, hands an empty arch list to its own check, and
raises this instead:

```
RuntimeError: FlashInfer requires GPUs with sm75 or higher
```

The card is fine. It reads the toolchain nvcc, not the GPU and not torch, so `pip install
nvidia-cuda-nvcc-cu12` does not help (that wheel has ptxas, headers and libnvvm but no `nvcc`
binary) and neither does a cu130 torch wheel. What works:

```
apt-get install -y cuda-nvcc-12-9 cuda-cudart-dev-12-9 libcublas-dev-12-9 cuda-cccl-12-9 \
    cuda-nvrtc-dev-12-9 cuda-crt-12-9 libcurand-dev-12-9 libcusparse-dev-12-9 \
    libcusolver-dev-12-9 libnvjitlink-dev-12-9
export CUDA_HOME=/usr/local/cuda-12.9 CUDA_PATH=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:$PATH
pip install flashinfer-python==0.6.18.post1
```

`cuda-nvcc-12-9` alone is not enough; each missing dev package surfaces as a different and equally
misleading error. This matters because the fused path is a way of using the vendor GEMM, and no
fallback recovers its speed.

## What I expect on each machine

| target | hardware | GEMM | quantiser | assemble | status |
|---|---|---|---|---|---|
| sm_100 | B200 | ok | ok | ok | measured |
| sm_103 | B300 | ok | ok | ok | compiled only |
| sm_120 | RTX 5090, RTX 6000 PRO | ok | ok | ok | measured on RTX PRO 6000 |
| sm_121 | DGX Spark, RTX Spark | fails | ok | ok | compiled only |

The Triton GEMM does not compile on sm_121: triton 3.7.1's `TritonGPUAccelerateMatmul` asserts on
the e2m1 `tl.dot_scaled` before ptxas runs. That matters less than it looks, since the fused path
needs no custom GEMM.

So the falsifiable prediction on a **Spark**: `--selftest` and `--window-check` pass, the `triton_*`
arms skip with that compile error, and `kaug` works if and only if flashinfer ships an NVFP4 GEMM
for sm_121. If any of that is wrong I would like to know which part.

On **Windows** expect the bf16 and fp8 arms only, since there is no Triton wheel and no flashinfer.
That path is rehearsed and should degrade rather than error out. No Windows or WSL run exists yet,
which is the main thing this PR is asking for.

## Results so far

On sm_120, graph-timed with clocks warmed and max drift 1.95%, fusion won on all 8 shapes at 1.35x
to 14.05x, including two narrowing layers that lose on B200. That reversal is why the shipped
`N >= K` dispatch rule wants checking on more sm_120 parts before anyone retunes it: it scores 35 of
48 on sm_120 against 47 of 48 for `N >= 0.25K`, and retuning a shipped constant on 8 shapes of a
single card is how the previous crossover figure got published and then withdrawn.

`--selftest` passes 13 of 13 on B200 and 8 of 8 on sm_120.

One caveat: the pure-torch fallback quantiser is faithful but not bit-exact against flashinfer. It
differs on 0.02 to 0.11% of nibbles, every one exactly on an e2m1 round-to-nearest-even tie, moving
one code step, never flipping sign, identical reconstruction rms. The rounding rule is correct,
checked against all seven exact midpoints. The Triton quantiser, the shipped path, is bit-exact.

## Multi-GPU warning

flashinfer's cutlass FP4 GEMM takes its stream from the tensor but installs no `CUDADeviceGuard`.
Calling `mm_fp4` or `nvfp4_quantize` while the current device is not the tensors' device launches
onto a stream from another context: the kernel hangs, the process becomes unkillable, and the card
ends in "GPU requires reset". This cost three cards in one afternoon. Every entry point here enters
the tensor's own device first and the preflight runs a tiny guarded `mm_fp4` per visible device
before any real work.
