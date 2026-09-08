# ROCm on RDNA2 APUs (Steam Deck / gfx1033)

Measured findings for AMD RDNA2 integrated GPUs, recorded so the routing decisions in
`install_llama_prebuilt.py` (`has_amd_gpu_without_rocm`, the Linux Vulkan branches) are not
"fixed" back into ROCm by someone who reasonably assumes ROCm must be the fast path.

Hardware: Valve Steam Deck LCD, AMD Van Gogh, `gfx1033`, 8 CU RDNA2, 16 GB LPDDR5
shared, 1 GB default UMA carveout. SteamOS 3 (immutable, no system ROCm).

## Summary

| | verdict |
|---|---|
| GGUF inference via **Vulkan** | **works, fastest** — the shipped default |
| GGUF inference via ROCm | works, but slower than CPU |
| **PyTorch training** via ROCm | **numerically wrong — do not enable** |
| PyTorch training via CPU | correct |

## 1. Inference: Vulkan wins by a wide margin

`llama-bench`, Qwen2.5-0.5B-Instruct Q4_0, `-p 128 -n 64 -ngl 99 -r 2`, one session:

| backend | prompt t/s | generation t/s |
|---|---|---|
| **Vulkan (RADV)** | **1444.80 ± 8.03** | **112.81 ± 18.32** |
| CPU (`-ngl 0`) | 753.08 ± 43.65 | 49.76 ± 2.15 |
| ROCm `gfx103X` prebuilt, via `HSA_OVERRIDE_GFX_VERSION=10.3.0` | 802.06 ± 431.34 | 17.51 ± 1.89 |

ROCm is ~6x slower than Vulkan and ~3x slower than plain CPU, with very unstable prompt
throughput.

**Building llama.cpp from source for the real arch does not help.** Built from the same
source commit with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1033` against AMD's native
gfx1033 `rocm-sdk` wheels (device reports `gfx1033 (0x1033)`, no `HSA_OVERRIDE`), all
three measured in one session:

| backend | prompt t/s | generation t/s |
|---|---|---|
| native `gfx1033` (self-built) | 864.60 ± 163.37 | 12.46 ± 0.34 |
| Vulkan (RADV) | 1268.85 ± 1.12 | 62.16 ± 9.99 |
| CPU | 686.43 ± 9.76 | 32.16 ± 0.42 |

Root cause: **108 of 142 gfx1033 rocBLAS Tensile files are named `fallback`** (76%).
AMD ships the arch but never tuned GEMM kernels for it, so "native but untuned" loses to
"foreign but tuned" (`gfx1030`), and both lose to Vulkan. Recompiling cannot synthesise a
tuning library; that needs Tensile's tuning pipeline, which wants far more RAM than the
device has.

Memory is *not* the limitation, so raising the BIOS UMA carveout does not change this:
llama.cpp's ROCm backend reports 8192 MiB (4478 free) via UMA detection and Vulkan
reports 9216 MiB (7361 free), both well past the 1 GB carveout.

## 2. Training: ROCm produces wrong results

Identical seed, init and data; 60 SGD steps on a 64→64→1 MLP with MSE loss:

| stack | CPU final loss | GPU final loss |
|---|---|---|
| rocm7.2 / torch 2.11 (`HSA_OVERRIDE` gfx1030) | 0.1148 | 17.4674 |
| rocm7.1 / torch 2.10 (`HSA_OVERRIDE` gfx1030) | 0.1148 | **-0.7765 → nan** |
| native gfx1033, TheRock torch 2.13 + ROCm 7.15 | 0.1148 | **nan** |

A negative MSE loss is not attainable, and `torch.autograd.gradcheck` fails in float64
on all three ("backward is not reentrant" on the native build — the same inputs and
`grad_output` produce different gradients between runs). The runtime also raises
`HSA_STATUS_ERROR_EXCEPTION` (code `0x1016`) and hangs the GPU queue under load.

Forward math is fine — full-module forward matched CPU to `4.2e-07` and loss to seven
significant figures — so this is specifically the backward pass. Three independent ROCm
versions fail identically, including AMD's own native gfx1033 build, so it is the
hardware/driver rather than a packaging or version problem.

Forward-only inference through PyTorch also disagreed with CPU on 1 of 40 fp32 runs, so
it is not dependable either. CPU is the correct default for the training stack here.

## 3. Two traps when debugging this hardware

* **Without `HSA_OVERRIDE_GFX_VERSION` set, GPU ops segfault** rather than degrading —
  `torch.randn(256, 256, device="cuda") @ itself` dumps core. The device still
  enumerates and `torch.cuda.is_available()` still returns `True`, so nothing warns
  first.
* On the `HSA_OVERRIDE` stacks, **`.item()` can return the previous reduction's value**:
  after `x.sum().item()`, a following `x.max().item()` on the same tensor returns the
  sum. `.cpu().item()` and `float()` are safe. This silently fabricates plausible-looking
  numbers and is easy to mistake for a compute bug. AMD's native gfx1033 build fixes this
  one; the training failures above survive it.

## 4. Why the Vulkan routing covers AMD generally

The Linux Vulkan branch accepts any AMD GPU with no usable ROCm, not just this part.
ROCm supports a short list of AMD hardware — APUs least of all — and many distros ship
Mesa/RADV without ROCm ever being installed, so that population is large and was getting
a CPU-only binary on a GPU host.

Widening it is safe because **the Vulkan bundle is a superset of the CPU bundle**: it
ships the same `libggml-cpu-*.so` variants alongside `libggml-vulkan.so`. With no usable
Vulkan device it enumerates zero devices and runs on the CPU backend — verified by
running `--list-devices` with the Vulkan ICD removed, which lists nothing and exits 0. So
a host that turns out to have no working Vulkan lands exactly where it does today, and
one that does gets its GPU. The numbers above are what justify preferring Vulkan when it
is available.

The probe only runs when there is no usable NVIDIA and no ROCm, so a host where ROCm
works keeps the ROCm branches; the flag cannot divert it. Such a host also resolves to the
`cpu` torch index — `get_torch_index_url` falls back to CPU when an AMD GPU is present
with no ROCm — so it ends up with the CPU training stack plus Vulkan inference, which is
the combination verified end to end on this hardware.
