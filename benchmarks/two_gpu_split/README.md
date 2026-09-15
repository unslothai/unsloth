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

## What each machine can and cannot run

This script needs **two CUDA devices visible to one process**. That constrains where it applies:

| machine | two-GPU split | what it can run |
|---|---|---|
| 1x RTX 5090 | **no**, single card | `--devices 0`: single-GPU baseline, sm_120 kernels, Windows |
| 1x DGX Spark | **no**, single GB10 | `--devices 0`: single-GPU baseline, sm_121 kernels |
| 2x DGX Spark | **no**, see below | nothing here; needs a multi-node variant |
| 2 cards in one box | yes | the full matrix |

**Two DGX Sparks are two nodes, not two GPUs.** They link over ConnectX-7 200GbE with RoCE and are
driven by NCCL across a network, not by `.to(device)` inside one process. This script deliberately
uses no `torch.distributed` and no NCCL, so it cannot address that pair at all.

An earlier version of this section argued a cross-node split was also the wrong shape on bandwidth.
**That argument was wrong, and it was corrected by measurement on a two-Spark pair.** The corrected
numbers, from the PR discussion:

| path | earlier claim | measured on a GB10 pair |
|---|---|---|
| direct device-to-device, same box, 64 MiB | 582 GiB/s | 103.3 GiB/s (the 582 was a B200 NVLink pair) |
| host-staged, same box, 64 MiB | 26 GiB/s | 27.5 GiB/s |
| two Sparks, raw RDMA, one rail | | 13.98 GB/s |
| two Sparks, raw RDMA, both rails concurrent | | 24.5 GB/s |
| two Sparks, NCCL busbw | about 12 GiB/s | 20.31 GB/s = 18.91 GiB/s |

Three things in the earlier version do not survive. The 12 GiB/s figure was the single-rail number,
so the pair was costed as if one cable were connected; NCCL bandwidth is 1.69x that. The claim that
both 200G ports share two PCIe Gen5 x4 lanes is false on this hardware: the two devices sit in
separate PCI domains, `0000:01:00.0` and `0002:01:00.0`, each with its own 32.0 GT/s x4 link, and
aggregation measures 1.75x raw and 2.00x under NCCL. And the "50x" compared two link bandwidths
rather than link time against compute time, and used the NVLink figure as the in-box denominator
when a GeForce pair over PCIe without P2P is host-staged; the honest ratio is 27.5 / 18.91 = 1.45x.

**What a DiT boundary actually costs**, counted from the tensors the boundary block is called with,
both crossings, against measured per-forward compute on a GB10:

| | Z-Image-Turbo 1024, 9 steps | Wan2.2-TI2V-5B 1280x704x121, 100 forwards |
|---|---|---|
| bytes per render | 0.60 GB | 35.2 GB |
| compute per render | 14.4 s | 1180 s |
| boundary at 12 GiB/s | 46.8 ms = **0.32%** | 2.73 s = **0.23%** |
| boundary at 20.31 GB/s | 29.7 ms = 0.21% | 1.73 s = 0.15% |

The bandwidth argument fails at the earlier number, before any correction: a DiT crossing is large
in bytes and tiny against the FLOPs of the stack it separates.

**What does still hold for two Sparks, for reasons that are not about bandwidth.** For a single
render there is no capacity problem to solve (peak allocation is 21.67 GiB for the whole Z-Image
pipeline and 12.1 GiB for the Wan transformer, against 121.7 GiB on one Spark), and a layer split is
strictly sequential, so it buys 1.0x because step t+1 consumes step t. Where a two-node split would
pay is a **batch of renders**: images in a batch are independent, so a two-stage schedule over the
batch dimension has real in-flight work, and its steady-state ceiling is the boundary cost above.
That should approach 2x aggregate throughput. Not built here.

## sm_121: the ptxas trap

`--compile` on a Spark fails out of the box:

```
ptxas fatal : Value 'sm_121a' is not defined for option 'gpu-name'
```

The triton bundled inside the torch cu130 wheel (3.5.1 with torch 2.9.1) ships a CUDA 12.8.93 ptxas
that predates sm_121a. The fix needs no apt and no system CUDA, because the same torch wheel already
carries a newer assembler:

```bash
export TRITON_PTXAS_PATH=<venv>/lib/python3.12/site-packages/torch/bin/ptxas   # CUDA 13.0.48
```

`nvidia-cuda-nvcc-cu13` is not an alternative; it is a placeholder sdist with no aarch64 wheel. torch
also warns that capability 12.1 exceeds its stated 12.0 maximum, and runs anyway. This is the sm_121
analogue of the sm_120 nvcc trap documented in the NVFP4 benchmark PR.

## Results reported so far

**RTX 6000 Ada (sm_89) + RTX 3090 (sm_86), no P2P, 3090 on PCIe 3.0 x4.** Z-Image-Turbo 1024, 9
steps, balanced split at block 13 of 30:

| configuration | p50 s | vs single | GPU 0 GiB | GPU 1 GiB |
|---|---|---|---|---|
| single | 12.765 | 1.000x | 19.29 | 0 |
| split | 14.817 | 1.161x | 13.63 | 5.91 |
| accel_dispatch | 15.118 | 1.184x | 13.51 | 5.79 |
| cpu_offload | 18.046 | 1.414x | | |
| seq_offload | 33.076 | 2.591x | | |

5.66 GiB saved on the primary card for 16% more time, faster than `accel_dispatch`, 1.22x faster than
CPU offload and 2.23x faster than sequential offload. Boundary transfers were 2.85% of the render even
on a saturated PCIe 3.0 x4 link at 2.6 GiB/s. Moving fewer blocks to the slower card reduced the
slowdown (ratio 0.75: 1.070x, 2.85 GiB on GPU 1), so on a mismatched pair the memory-balanced default
is not the fastest split. Five split/undo cycles left no hooks, shims or buffers behind and reproduced
the baseline bit-identically.

**Mixed cards diverge, and by how much depends on the model.** The split matched `accel_dispatch`
byte for byte, so the divergence is between the GPUs, not from the split. On Wan2.2 it was a
one-level difference in 233k of 63.7M values. On 9-step Z-Image-Turbo it was a visibly different
image, PSNR 25.90 dB, and it grows with steps: 37.5 dB at 1 step, 33.2 at 2, 30.5 at 4, 25.9 at 9.
Running entirely on one card versus entirely on the other differs *more* (22.35 dB) than the split
does. Users with mixed cards should expect this, and a few-step image model is where it shows.

**The hook boundary recompiles slowly.** With `--compile` the shim and hook styles produce identical
pixels, but the hook is traced into the compiled block, so changing placement forces a full block
recompile: 42.7 s against 1.1 s for the shim. Prefer the shim under compile.

**The planner splits only the largest container**, so `noise_refiner` and `context_refiner` (1.33 GiB
together on Z-Image) stay on GPU 0 in every configuration.

**DGX Spark, GB10, sm_121a, `--devices 0`.** Z-Image-Turbo 1024, 9 steps, single card: 34.29 s eager,
25.01 s with regional compile (1.371x, 26.1 s wall and 13.3 s in dynamo, kept in its own column),
peak 21.67 GiB. CPU offload 104.0 s, sequential offload 80.9 s. With one device the `bit_identical`,
boundary and P2P columns are trivially true, zero and empty respectively, and mean nothing.

Two reporting bugs found in the first round are fixed in this revision: the `card0 GiB` / `card1 GiB`
columns read a process-lifetime peak without resetting between configurations, so an offload row
could report a peak on a card it never used (the JSON's `allocated_gib` was always correct); and the
no-P2P note printed under `--devices 0` where no split row had run.

## Run it

```bash
pip install torch diffusers transformers accelerate safetensors pillow

# two cards in one box, the full matrix
python bench_two_gpu_split.py --repo Tongyi-MAI/Z-Image-Turbo --steps 9 --size 1024 \
    --devices 0,1 --reps 5 --out split.json

# single card: baseline, kernel portability and Windows check
python bench_two_gpu_split.py --repo Tongyi-MAI/Z-Image-Turbo --steps 9 --size 1024 \
    --devices 0 --reps 5 --out single_5090.json
```

Add `--compile` for the deployment path. Compile time is reported in its own column and never
folded into the per-render seconds, because moving a block to another card puts the device in the
Dynamo guards and forces a genuine one-off recompile that a single-GPU run does not pay.

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
