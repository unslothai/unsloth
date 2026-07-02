## Summary

A re-review of the diffusion stack (#6675 / #6679 / #6680) focused on performance, then a speed pass that preserves accuracy. The review surfaced one real accuracy bug and one dead-on-arrival speed path; this fixes both and adds the lossless / near-lossless wins, all measured on a B200. Stacked on #6680.

The headline: regional `torch.compile` was gated off for the GGUF transformer, and since the backend is GGUF-only that made it dead on every shipping model. It actually compiles and runs **2.2x faster** now, at a quality delta far below the quantisation noise floor, so it is safe to use.

## Correctness fixes

- **TF32 global-state leak.** `speed_mode=max` flipped `torch.backends.*.allow_tf32` process-wide and never restored them, so a later `off` load silently inherited TF32 and was no longer bit-identical. Added `snapshot_backend_flags` / `restore_backend_flags` (TF32 + cudnn.benchmark), captured before the speed layer runs and restored on unload. Verified: load `max` then `off` is now byte-identical (PSNR inf) to a fresh `off`.
- **sd-cli could hang past its timeout.** `_run()` blocked in `for line in stdout` and only checked the timeout after EOF, so a child stuck in model load / GPU init with no output ignored the timeout. Drained stdout on a reader thread with a wall-clock deadline; added a silent-hang regression test.

## Speed: diffusers path (near-lossless)

- **Regional `torch.compile` on the GGUF transformer.** The `is_gguf` gate (and Z-Image's `supports_torch_compile=False`) were stale: `compile_repeated_blocks` compiles and runs ~2.2x faster on torch 2.9.1 / diffusers 0.38 (the per-op dequant stays eager, the rest of the block compiles). Gate relaxed.
- **cudnn.benchmark** added to the `default` tier (autotunes the fixed-shape VAE convs).
- **`torch.inference_mode()`** around the pipeline call (strictly faster than the internal `no_grad`, numerically identical).
- **Default profile.** A GGUF model with no explicit `speed_mode` now resolves to `default` (`resolve_speed_mode`), since compile's perturbation sits below the quant noise floor and so does not reduce quality versus the dense reference. Dense models stay `off` / bit-identical, and an explicit value (including `"off"`) is always honored, so the byte-identical path is one flag away and remains the regression reference.

## Memory path

- **VAE tiling** (not bit-identical above 1MP) is now restricted to the `model` / `sequential` / CPU tiers. The `balanced` (group) tier keeps exact slicing only, so it is now **bit-identical** to the resident image (verified PSNR inf) and slightly faster.
- **Group offload** adds `non_blocking` + `record_stream` on the CUDA stream path to overlap each block's H2D copy with compute (lossless; gated on the installed diffusers signature so older versions still work).

## Native (sd.cpp) path

- **`native_speed_flags`**: a first-class speed knob. `default` adds `--diffusion-fa` (a near-lossless CUDA win that was previously only added on offload tiers); `max` also adds `--diffusion-conv-direct`. conv-direct stays opt-in because it measured **+45% on CUDA** here, so it is never auto-on. The engine `generate()` merges it, de-duped against the offload flags.

## Measured (single B200, Z-Image-Turbo Q4_K_M, 1024x1024, 8 steps)

| config | median/gen | vs off | accuracy |
| --- | --- | --- | --- |
| `off` (bit-identical reference) | 1.80 s | reference | reference |
| `default` (auto for GGUF) | **0.82 s** | **+54.7%** | PSNR 37.7 dB vs eager (Q4-vs-bf16 is ~21 dB, so below the noise floor) |
| `balanced` (group offload) | 2.19 s | n/a | PSNR inf (bit-identical, tiling now off) |
| `off` after a `max` load | 1.83 s | n/a | PSNR inf vs fresh `off` (TF32 leak fixed) |

`scripts/perf_verify.py` reproduces all of the above end to end through the real backend; `scripts/compile_probe.py` is the eager-vs-compiled GGUF probe.

## Tests

184 passing (was 166). New coverage: the backend-flag snapshot/restore, GGUF compile eligibility + the `resolve_speed_mode` GGUF auto-default, the `balanced` tiling/slicing split + group `non_blocking`/`record_stream`, `native_speed_flags` + the engine de-dup, and the sd-cli silent-hang timeout. `scripts/diffusion_bench.py` gains `--speed-mode` so the tiers are benchmarkable.
