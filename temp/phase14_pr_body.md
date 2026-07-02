### Summary

Fixes the opt-in dense **int8** transformer quant path, which crashed on FLUX.1 and Qwen-Image with:

```
RuntimeError: torch._int_mm: self.size(0) needs to be greater than 16, but got 1
```

Found while benchmarking the dense quant path across all supported models: fp8 worked everywhere, but int8 failed on every Flux.1 and Qwen variant (worked on Z-Image and FLUX.2-klein-4B).

### Root cause

int8 dynamic quant runs through `torch._int_mm`, which requires the activation row count **M > 16**. A DiT's AdaLN **modulation** projections and its **conditioning embedders** are computed once from the `[batch, dim]` conditioning vector (M = batch = 1), not per token, so they hit `_int_mm` at M=1 and crash. Examples (in -> out):

- FLUX.1: `transformer_blocks.*.norm1.linear` 3072 -> 18432, `norm1_context.linear`, `norm_out.linear`, `time_text_embed.*`
- Qwen-Image: `transformer_blocks.*.img_mod.1` / `txt_mod.1` 3072 -> 18432, `time_text_embed.timestep_embedder.*`
- FLUX.2-klein: `double_stream_modulation_*.linear`, `single_stream_modulation.linear`

These have large feature dims, so the existing `min_features` filter did not exclude them. (Z-Image / klein-4B happened not to hit an M=1 int8 matmul, which is why they worked.)

### Fix

The int8 filter now additionally skips any `nn.Linear` whose fully-qualified name matches a modulation / conditioning-embedder token: `norm`, `_mod`, `modulation`, `timestep_embed`, `guidance_embed`, `time_text_embed`, `pooled`. These run at M=1 once per block and are a negligible share of the FLOPs, so int8 keeps the full speedup on the attention / FFN layers (M = sequence length). 

- The exclusion is **int8-only**: fp8 / nvfp4 / mxfp8 use `scaled_mm`, which has no M>16 limit and quantises these layers fine.
- Sequence embedders (`context_embedder` / `x_embedder` / `txt_in`, M = seq) are deliberately **not** excluded. Note `context_embedder` contains the substring `text_embed`, which is why the token is the specific `time_text_embed`, not `text_embed`.

### Measured (B200, 1024px, transformer_quant=int8 + speed=default)

int8 now runs on every supported model and is the **fastest** dense path on Flux/Qwen (int8 runs full-rate vs fp8's FP32-accumulate):

| model | eager GGUF | int8 (this PR) | speedup | fp8 (for ref) |
| --- | --- | --- | --- | --- |
| FLUX.1-dev (28 step) | 9.62s | 1.98s | 4.86x | 2.15s |
| Qwen-Image (20 step) | 10.39s | 1.87s | 5.57x | 2.09s |
| Qwen-Image-2512 | 10.20s | 1.84s | 5.55x | 2.09s |
| FLUX.1-schnell (4 step) | 1.46s | 0.41s | 3.59x | 0.44s |

Z-Image and FLUX.2-klein-4B (already working) are unchanged.

### Changes

- `diffusion_transformer_quant.py`: add `_INT8_EXCLUDE_NAME_TOKENS`; `make_filter_fn` takes `exclude_name_tokens`; `quantize_transformer` passes it for int8 only.
- Hermetic test that the int8 filter excludes the modulation / embedder linears and keeps the attention / FFN / sequence-embedder linears (fp8 keeps them).
- `scripts/int8_linear_probe.py`: the meta-device probe used to enumerate each transformer's Linear layers and derive the exclusion list.

### Testing

`python -m pytest tests/ -q -k diffusion` -> 231 passed. The GPU numbers above are from `scripts/diffusion_bench.py --transformer-quant int8`.

Stacked on #6703 (Phase 12). Base branch `diffusion-phase12-fbcache`.
