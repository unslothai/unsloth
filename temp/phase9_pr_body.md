## Summary

Pre-quantized transformer loading for the Studio diffusion fast path: load an
already-quantized transformer checkpoint instead of materialising the dense bf16 and
quantising it on the GPU. Stacked on the Phase 8 fast-transformer pass (#6694).

The Phase 8 `transformer_quant` mode is fast, but its one cost is load memory + download:
it loads the **dense bf16** transformer onto the GPU and torchao-`quantize_`s it in place,
so the load peak is ~2x GGUF's and it pulls the full bf16 weights. Pre-quantizing fixes
both. Quantise once offline, then at runtime build the transformer skeleton on the **meta**
device (`accelerate.init_empty_weights`) and `load_state_dict(assign=True)` the quantized
weights, so the dense bf16 never touches the GPU.

Measured on a B200 (Z-Image-Turbo, fp8, 1024px / 8 steps), through the real loader:

| path | GPU load peak (full pipeline) | on-disk | output |
| --- | --- | --- | --- |
| runtime (dense `from_pretrained` -> `quantize_transformer`) | 21.2 GB | ~12 GB bf16 | reference |
| **pre-quantized (`load_prequantized_transformer`)** | **14.6 GB** | **6.28 GB** | **LPIPS 0.0** |

The fast mode's load peak drops to essentially GGUF's 13.4 GB, the download halves, and the
output is bit-identical, because it is the exact same torchao config + `min_features` filter
the runtime path uses, applied ahead of time (the isolated transformer load peak is 12.9 ->
6.3 GB; the 14.6 GB above includes the resident text encoder + VAE both paths load).

## What it does

`_load_dense_quant_pipeline` now tries a pre-quantized source first, then falls back:

1. **pre-quantized** -- if a checkpoint is configured for the resolved scheme (an explicit
   `transformer_prequant_path`, or the family's hosted repo), `load_prequantized_transformer`
   builds the skeleton on meta and assigns the quantized state dict in, then places it;
2. **dense + quantise** (the Phase 8 path, unchanged) if no checkpoint is available or its
   load fails;
3. **GGUF** if quantisation itself is unsupported.

So with nothing configured the behaviour is exactly Phase 8. Hosting of checkpoints is
deferred: `DiffusionFamily.prequant_repos` ships empty and the new request field defaults
null, so this is inert until a checkpoint is built/configured.

## Design

- New `core/inference/diffusion_prequant.py` mirrors the sibling quant modules (pure
  functions, torch / accelerate / huggingface_hub imported lazily, best-effort, hermetic CPU
  tests). `resolve_prequant_source` (priority: explicit path -> family repo -> none) and
  `load_prequantized_transformer` (meta-init + `load_state_dict(assign=True)` + metadata
  validation + place + the same `_unsloth_runtime_quant` marker). Any missing / mismatched /
  unreadable checkpoint returns None and the caller falls back -- the default cannot regress.
- The checkpoint is `{"format", "metadata", "state_dict"}` saved with `torch.save`. torchao
  weight subclasses are not safetensors-serializable, so loading uses `weights_only=False`;
  only a configured first-party family repo or an explicit local path reaches that, which is
  the trust signal (no arbitrary remote pickle). The `PrequantSource.kind` enum leaves room
  for a diffusers-native `TorchAoConfig` artifact later.
- `scripts/build_prequant_checkpoint.py` builds (and optionally uploads) a checkpoint,
  importing the runtime quant factory (`_make_quant_config` / `make_filter_fn`) so the
  offline artifact is identical to on-the-fly quantisation (the LPIPS-0 invariant).
- Flag surface: `DiffusionLoadRequest.transformer_prequant_path`, forwarded through the
  route / `begin_load` / `load_pipeline` exactly like `transformer_quant`. `status()` already
  reports the engaged scheme, so no status-model change.

## Tests

CPU-only, hermetic (torch / accelerate / huggingface_hub stubbed via `sys.modules`):
- new `tests/test_diffusion_prequant.py` -- the resolver (path override wins / family repo by
  scheme / wrong scheme / nothing configured), and the loader (meta-init + `assign=True`,
  never the dense `from_pretrained`, sets the marker; format / scheme / base mismatch, a
  raising `torch.load`, and a missing file each return None).
- extended `tests/test_diffusion_backend.py` -- the pre-quant branch engages (no dense load,
  no `quantize_transformer`), and a failed pre-quant load falls back to the dense path.
- extended `tests/test_diffusion_routes.py` -- `transformer_prequant_path` threads through to
  `begin_load`.

GPU verification: `scripts/build_prequant_checkpoint.py` then
`scripts/verify_prequant_backend.py` (the table above), and `scripts/prequant_probe.py` (the
original meta-init + assign measurement).

## Scope / notes

- Default unchanged: families ship `prequant_repos=()` and the request field defaults null.
- The VRAM preflight still estimates the dense size, so the resident gate stays conservative
  (a box that fits the pre-quant but not the dense load still falls back to GGUF). A follow-up
  can lower the estimate when a pre-quant source resolves.
- GPU-verified on Z-Image / fp8; Flux / Qwen and int8 are wired but unverified -> flagged.
