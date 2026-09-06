# Two-GPU layer split for diffusion and video diffusion

Throwaway benchmark. Not wired into the package, nothing imports it, delete the branch when the
numbers are in.

## Why

Users with two smaller cards cannot fit a large DiT on one. This splits the repeated transformer
blocks across both, blocks `0..k-1` on the first card and `k..n-1` on the second, and reports
whether that is free, correct and idempotent.

Measured already on B200 pairs: **31 of 31 placements bit-identical on z-image** across eager,
compiled and CUDA-graphed, at balanced, 25/75 and 75/25 ratios, through both boundary styles,
including the NVFP4 arm; **10 of 10 on Wan2.2-TI2V-5B**. Idempotent over five
split/unsplit/re-split cycles with no hook or shim growth.

What this script adds is **your hardware**, which we do not have: RTX 5090, DGX Spark, RTX Spark
laptops, and Windows rather than Linux.

## Run it

```bash
pip install torch diffusers transformers accelerate safetensors pillow

python bench_two_gpu_split.py --repo Tongyi-MAI/Z-Image-Turbo --steps 9 --size 1024 \
    --devices 0,1 --reps 5 --out split_5090.json
```

Add `--compile` for the deployment path. Compile time is reported in its own column and never
folded into the per-render seconds, because moving a block to another card puts the device in the
Dynamo guards and forces a genuine one-off recompile that a single-GPU run does not pay.

One card only, or want the single-GPU baseline alone, is fine: pass `--devices 0`.

## What matters in the output

| field | read it as |
|---|---|
| `bit_identical` | `max abs(single-GPU image - split image)`. 0 means the split changed nothing. |
| `P2P` | whether the driver copies card to card directly. **False is normal** for GeForce over PCIe and is not a failure; the driver stages through a host buffer, and that path is timed separately. |
| `boundary` | bytes and milliseconds per forward crossing the link. Compare against `s/render` divided by steps: that ratio is what the split costs. |
| `compile` | its own column, never inside `s/render`. |

**Two cards of different models are not expected to be bit-identical.** A 5090 paired with a Spark
have different SM counts, which changes the tile and split-k choice inside cuBLAS and therefore the
reduction order. That is a real result, not a bug, and the script reports the pixel count rather
than hiding it behind a tolerance.

## Portability

No `torch.distributed`, no NCCL, no `os.fork`, no signals, no Unix-only calls, nothing written
outside `--out`. Plain per-module `.to(device)` and an explicit copy of the tensors that cross.
Windows, WSL and Linux, any two CUDA devices, with or without peer-to-peer.

The environment block in the JSON records platform, torch, CUDA, driver, and per-device name,
compute capability and SM count, so a report is self-describing.

## One trap, if you extend this to custom kernels

flashinfer's cutlass FP4 GEMM takes its stream from the tensor and installs **no
`CUDADeviceGuard`**. A layer on card B with the current device left on card A never returns, the
process becomes unkillable, and the card ends in "GPU requires reset". This cost three cards in one
afternoon here. ATen operators get a guard from the dispatcher, which is why bf16 and fp8 split
correctly on tensor placement alone and only the 4-bit path wedges.

The script sets the device at every crossing, folded away under `torch.compiler.is_compiling()`
since inductor's generated wrapper already opens a device guard for the graph's device.

Related: `torch.cuda.set_stream` silently sets the current device.

## Notes for the Blackwell family

On sm_120 (RTX 5090, RTX 6000 PRO) flashinfer's NVFP4 kernels need a **CUDA >= 12.9 nvcc**. Under a
12.8 toolchain they refuse to build and the error names the wrong cause, reporting `FlashInfer
requires GPUs with sm75 or higher` while the real message, `SM 12.x requires CUDA >= 12.9`, is
logged and swallowed. It reads the toolchain, not the GPU, so upgrading torch to a cu130 wheel does
not help while `nvcc` is 12.8.

None of that affects this script unless your own layers call flashinfer: the bf16 and fp8 rows need
only torch.
