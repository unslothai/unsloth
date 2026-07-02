## Summary

A selectable **attention backend** for the Studio diffusion transformer, via the diffusers
`set_attention_backend` dispatcher. Stacked on the Phase 9 pre-quantized loading pass (#6700).

Attention is memory-bandwidth bound, so swapping in a better SDPA kernel is a real end-to-end
win that is **orthogonal to the linear-weight quantisation** (it speeds the QK/PV matmuls
torchao never touches) and composes with torch.compile.

Measured on a B200 (Z-Image-Turbo, 1024px / 8 steps, dense bf16 + regional compile = today's
profile), LPIPS vs the default backend:

| attention | sec | vs default | LPIPS |
| --- | --- | --- | --- |
| default SDPA (`native`) | 0.686 | 1.00x | reference |
| **cuDNN fused (`_native_cudnn`)** | **0.584** | **1.18x** | 0.004 |

cuDNN's fused attention is **1.18x end-to-end, near-lossless** (LPIPS 0.004 is below the
compile/quant noise floor). It is exact (not quantized) and broadly available on NVIDIA.

## What it does

New load flag `attention_backend` (`auto | native | cudnn | flash | flash3 | flash4 | sage |
xformers | aiter`, default `auto`). Resolved per device, set on `pipe.transformer` **before
compile** (compile traces attention):

- **`auto`** picks the best *exact* backend: `_native_cudnn` on NVIDIA CUDA **when a speed
  profile is active** (so `speed_mode=off` stays bit-identical), `native` SDPA elsewhere
  (AMD / Intel / Apple / CPU, which the dispatcher already routes). The GGUF default path
  (which defaults `speed_mode=default`) therefore picks up the cuDNN win automatically.
- An **explicit** backend is honored verbatim: `cudnn` / `flash` / `flash3` (Hopper) /
  `flash4` (SM100) are exact; `sage` is INT8 attention (a small quality cost, consumer
  friendly); `xformers` / `aiter` are memory-efficient (NVIDIA) / AMD ROCm.
- An **unavailable** kernel (missing package / wrong arch) is caught at set time and the load
  falls back to the diffusers default rather than failing.

## Design

- New `core/inference/diffusion_attention.py` mirrors the sibling modules (pure functions,
  torch/diffusers imported lazily, best-effort, hermetic tests): `normalize_attention_backend`,
  `select_attention_backend(target, requested, *, speed_active)` (the per-device policy),
  `apply_attention_backend(pipe, backend)` (set + graceful fallback).
- `diffusion.py` `load_pipeline` selects + applies the backend right before
  `apply_speed_optims` (so it precedes compile); `attention_backend` threads through
  `begin_load` / `_LoadState` / `status()` like the other load knobs, and the engaged backend
  is reported in status. Orthogonal to the transformer/text-encoder quant and to GGUF vs dense.
- Flag surface: `DiffusionLoadRequest.attention_backend` (Literal) + the route forward +
  `DiffusionStatusResponse.attention_backend`.

## Tests

CPU-only, hermetic (`_is_cuda_nvidia` monkeypatched; a fake transformer records / raises on
`set_attention_backend`):
- new `tests/test_diffusion_attention.py` -- normalisation + alias map, the select policy
  (auto -> cuDNN on NVIDIA when speed active; native when off / off-NVIDIA; explicit honored;
  `native` -> no-op), and apply (sets the backend, falls back to the default on an unavailable
  kernel, handles a transformer without the method).
- extended `tests/test_diffusion_routes.py` -- `attention_backend` threads through to
  `begin_load`, and an invalid value is a 422.

GPU verification: `scripts/perf_levers_probe.py` (the table above; also measured that
SageAttention v1.0.6 lacks Blackwell kernels and FlashAttention `*_hub` need the `kernels`
package -> both correctly fall back, while the lossless inductor autotune flags were
~neutral on this B200/bf16 and are deferred), plus a backend-function smoke (auto ->
`_native_cudnn`, finite image).

## Scope / notes

- `off` stays bit-identical: `auto` only upgrades attention when a speed profile is active.
- cuDNN attention needs a recent cuDNN; the dispatcher validates at set time and falls back to
  `native` otherwise, so there is no regression on older stacks.
- `sage` / `flash*` are wired and validated by the fallback path, but not GPU-verified here
  (sage's PyPI build has no SM100 kernels; the `*_hub` flash variants need `kernels`, which
  conflicts with the diffusers huggingface-hub pin in this env) -> flagged, opt-in.
