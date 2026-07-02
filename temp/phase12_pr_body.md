### Summary

Adds opt-in step caching (First-Block-Cache) for the diffusion transformer. Across denoise steps a DiT's output settles, so once the first block's residual barely changes the remaining blocks are skipped and their cached output reused. diffusers ships this natively (`FirstBlockCacheConfig` + `transformer.enable_cache`, with the standalone `apply_first_block_cache` hook as a fallback).

This is the next lever in the Studio diffusion efficiency stack, targeting denoise-time speed on many-step models.

### Measured (Flux.1-dev, 28 steps, 1024px, one B200)

Reference is the no-cache compiled output (`scripts/fbcache_flux_probe.py`):

| mode | latency | speedup | LPIPS vs no-cache |
| --- | --- | --- | --- |
| compile baseline | 2.83s | 1.00x | ref |
| fbcache (threshold 0.08) | 2.03s | 1.40x | ~0.08 |

~1.4x on top of `torch.compile` at a small, well-bounded quality cost.

### Design

- OFF by default and a per-load opt-in. The win scales with step count, so it is for many-step models (Flux / Qwen-Image) and pointless for few-step distilled models (e.g. Z-Image-Turbo at ~8 steps), where a single skipped step is a large fraction of the trajectory.
- Composes with regional compile only with `fullgraph=False` (the cache's per-step decision is a `torch.compiler.disable` graph break), which the speed layer now switches to automatically when a cache is engaged.
- Best-effort: a model whose block signature the hook does not recognise is caught and the load proceeds uncached.
- The residual threshold auto-raises for a quantised transformer (0.08 -> 0.12), which shifts the residual distribution, per ParaAttention's fp8 guidance. An explicit `transformer_cache_threshold` overrides.

### Changes

- new `core/inference/diffusion_cache.py`: `normalize_transformer_cache` + `apply_step_cache` (enable_cache / apply_first_block_cache fallback; lazy diffusers import).
- `diffusion_speed.py`: `apply_speed_optims` takes `cache_active`; compile drops `fullgraph` when a cache is engaged.
- `diffusion.py`: `apply_step_cache` runs before compile; `transformer_cache` / `transformer_cache_threshold` thread through `begin_load` -> `load_pipeline`, and the engaged mode is reported in `status()`.
- `models/inference.py` + `routes/inference.py`: `transformer_cache` (`off | fbcache`) and `transformer_cache_threshold` request fields; engaged mode in the status response.
- hermetic tests for normalisation, the enable_cache / hook-fallback paths, threshold selection, and best-effort failure handling, plus route threading + validation (422 on a bad enum / out-of-range threshold).
- `scripts/fbcache_flux_probe.py`: the Flux validation probe (latency / speedup / VRAM / LPIPS vs the compiled no-cache baseline).

### Compatibility

Default behaviour is unchanged: the request field defaults to `null` so nothing engages unless a caller opts in. No change to the GGUF build, the dense fast path, the placement/offload order, or any other family.

### Testing

`python -m pytest tests/ -q -k diffusion` -> 230 passed. The Flux speedup/quality numbers above are from `scripts/fbcache_flux_probe.py`.

Stacked on #6702 (Phase 11). Base branch `diffusion-phase11-consumer-int8`.
