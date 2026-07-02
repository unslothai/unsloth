## Summary

Phase 6 extends Phase 4's native stable-diffusion.cpp engine (#6679) from text-to-image to the wider feature surface: image-to-image, inpaint, edit, LoRA, and ESRGAN upscale. stable-diffusion.cpp already supports all of these through the binary, so this is pure command-builder additions plus one new engine method. The text-to-image path is byte-for-byte unchanged (the new fields all default to off).

Stacked on #6679; I will retarget the base as the lower phases land.

## What it does

- Image conditioning (`sd_cpp_args.py`). `SdCppGenParams` gains `init_img` + `strength` (img2img), `mask` (inpaint), `ref_images` (FLUX-Kontext / Qwen-Image-Edit editing, emitted as repeated `--ref-image`), and `lora_dir` + `lora_apply_mode`. Individual LoRAs are selected with sd.cpp's own `<lora:name:weight>` tags inside the prompt, so no prompt rewriting is needed.
- Upscale mode. A new `SdCppUpscaleParams` + `build_sd_cpp_upscale_command` drive sd-cli's ESRGAN run mode (input image + ESRGAN model, no prompt or text encoders).
- Engine (`sd_cpp_engine.py`). The subprocess runner is factored into a shared `_run()` so `generate()` (now carrying the conditioning flags) and the new `upscale()` reuse the same streaming / error / output-check path.
- Smoke harness. `scripts/sd_cpp_smoke.py` gains `--task {txt2img,img2img,upscale}` with `--init-img` / `--strength` / `--upscale-model` / `--upscale-repeats`.

## Verification (single B200 box, through SdCppEngine)

| task | detail | time |
| --- | --- | --- |
| img2img | Z-Image-Turbo Q4_K, init image at strength 0.6, prompt re-themed to autumn | 4.8 s |
| upscale | RealESRGAN_x4plus_anime_6B, 512x512 to 2048x2048 (4x) | 2.7 s |

Both produce coherent images: the img2img run preserves the source structure while applying the new prompt, and the upscale run quadruples each dimension. inpaint / edit / LoRA are verified at the command-construction level (the real-model runs need specific edit checkpoints / LoRA files).

## Tests

CPU only, subprocess and filesystem stubbed, from `studio/backend`:

```
python -m pytest tests/test_sd_cpp_args.py tests/test_sd_cpp_engine.py
```

10 new tests across the img2img / inpaint / edit / LoRA flag construction, the upscale builder and its validation, and the engine's img2img + upscale paths. Full diffusion suite 176 passing.

## Out of scope (later)

Video (`vid_gen`, Wan / LTX), and wiring these tasks through the route / backend and the diffusers path. This PR lands them on the native engine as a self-contained, tested unit.
