## Summary

Make the transformer `auto` quant pick **int8 on consumer / workstation GPUs**, while
data-center parts keep fp8. Stacked on the Phase 10 attention pass (#6701).

Consumer Blackwell / Ada / Ampere (and workstation RTX) GPUs **halve** fp8 (and fp16/bf16)
tensor-core throughput when the matmul accumulates in FP32, while **int8 runs at full rate**
(its int32 accumulate is not nerfed). So on a consumer card the int8 tensor cores are the
faster path, not fp8.

Public benchmarks back this directly. SDNQ's per-GPU matmul numbers (int8 via
`torch._int_mm` vs fp8 via `torch._scaled_mm`):

| GPU | bf16 | int8 | fp8 | int8 vs bf16 |
| --- | --- | --- | --- | --- |
| RTX 3090 (Ampere) | 76 | **184** | 0 (no fp8 HW) | 2.4x |
| RTX 4090 (Ada) | 164 | **359** | 83 rowwise / 269 TW | 2.2x |
| RTX 5090 (Blackwell) | 232 | **471** | 433 | 2.0x |
| RTX PRO 6000 (data-center) | 390 | 499 | **659** | 1.3x |

int8 wins (or is the only option) on every consumer part; fp8 only wins on the data-center
chip. This matches the consumer FP8-accumulate nerf exactly.

## What it does

When `transformer_quant=auto`, the per-arch ladder is reordered to put int8 first on a
consumer / workstation GPU, detected by the existing `_is_consumer_gpu` name heuristic
(GeForce / TITAN / workstation / unknown -> consumer; recognised data-center tokens like
B200 / H100 / A100 / L40 -> not). Data-center HBM parts are unchanged and keep fp8 first.

It is a pure reorder of schemes that already exist (`_prefer_consumer_scheme`): no new flags,
no new kernels. The smoke probe still gates each scheme, and an explicit `transformer_quant`
(e.g. forcing `fp8`) is still honored verbatim.

## Design

- `diffusion_transformer_quant.py`: `select_transformer_quant_scheme` now walks
  `_prefer_consumer_scheme(tier, device)` -- which moves int8 to the front when
  `_is_consumer_gpu(device)` -- instead of the raw data-center tier. The `_AUTO_LADDER`
  comment documents that it is the data-center order, reordered per GPU class at selection.

## Tests / verification

- Hermetic CPU tests: consumer Blackwell (RTX 5090) and Ada (RTX 4090) and a workstation /
  unknown name -> int8; data-center Ada (L40S), Hopper (H100), Blackwell (B200) -> fp8;
  int8-unavailable on consumer falls back to the rest of the tier (fp8). The shared torch
  stub now carries a device name so the selection sees a GPU class.
- GPU non-regression on a B200: `_is_consumer_gpu("cuda")` is False, the reorder is a no-op,
  and `auto` still resolves to fp8 -- the data-center path is unchanged.

## Notes

- The consumer speedup itself is from the cited public benchmarks + the NVIDIA accumulate
  specs (this CI box is a data-center B200, so only the non-regression is measured here).
- int8 is also the broadest-compatibility scheme (Ampere+ and, via the same `torch._int_mm`
  core, the most portable to AMD / Intel), so this also nudges the widest set of consumer
  hardware onto a working fast path by default.
