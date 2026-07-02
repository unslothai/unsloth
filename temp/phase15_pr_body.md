### Summary

Lets the pre-quantized-checkpoint builder produce a **working int8** checkpoint, not just fp8.

The fast transformer_quant path can either quantise the dense bf16 transformer on the GPU at load (~2x the GGUF load VRAM, full bf16 download) or load a checkpoint that was quantised once ahead of time (`scripts/build_prequant_checkpoint.py` -> `diffusion_prequant.py`), which drops the transformer GPU load peak and download ~2x. That builder already accepted `--scheme int8`, but it applied the dense quant filter **without** the int8-only M=1 exclusion the runtime path got in #6716, so a built int8 checkpoint baked the AdaLN-modulation / conditioning-embedder projections as int8 and crashed at the first denoise step on Flux / Qwen:

```
RuntimeError: torch._int_mm: self.size(0) needs to be greater than 16, but got 1
```

### Root cause

int8 dynamic quant runs through `torch._int_mm`, which requires activation rows M > 16. A DiT's modulation / timestep / guidance / pooled-text projections run once from the `[batch, dim]` conditioning vector (M = batch = 1). The runtime path (#6716) excludes them from the int8 filter; the offline builder did not, so an int8 checkpoint diverged from the runtime model (and crashed where the runtime path is correct). fp8 / nvfp4 / mxfp8 use `scaled_mm` (no M limit), which is why fp8 prequant already worked.

### Fix

Factor the scheme -> exclusion decision into one shared `exclude_tokens_for_scheme(scheme)` in `diffusion_transformer_quant.py`, used by **both** the runtime quantise path and the offline builder, so the two can never drift -- an int8 checkpoint built ahead of time now skips exactly the layers the runtime path skips (the module's "offline == runtime, LPIPS-0" invariant). `build_prequant_checkpoint.py` applies it; for fp8 / fp4 / mx the helper returns `()`, so nothing changes for them.

Result: int8 prequant produces a working checkpoint on every supported model, giving **int8** -- the consumer-preferred scheme (consumer cards halve fp8 FP32-accumulate throughput; int8 runs full-rate) -- the same ~2x load-VRAM and download reduction fp8 already had. This directly lowers the peak memory to *load* the fast path on the hardware that needs it most.

### Changes

- `diffusion_transformer_quant.py`: add `exclude_tokens_for_scheme()`; `quantize_transformer` now calls it (behaviour-identical refactor of the inline int8 check).
- `scripts/build_prequant_checkpoint.py`: pass `exclude_name_tokens = exclude_tokens_for_scheme(scheme)` into the filter.
- Hermetic test that the shared helper returns the modulation/embedder tokens for int8 and `()` for fp8 / nvfp4 / mxfp8.

### Testing

`python -m pytest tests/test_diffusion_transformer_quant.py tests/test_diffusion_prequant.py -q` -> 43 passed. The runtime int8 path is unchanged (same tokens, now via the shared helper).

Stacked on #6716 (Phase 14). Base branch `diffusion-phase14-int8-modulation`.
