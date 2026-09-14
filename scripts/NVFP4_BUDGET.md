# NVFP4 time budget harness

Four scripts that answer "where does a render's GPU time actually go on the shipped path", and one
that answers "does compiling the VAE decode change the picture". They drive the real Studio
`DiffusionBackend`, so what they measure is the shipped loader, the shipped regional compile, the
shipped attention pin and the shipped manual CUDA graph, not a re-implementation. All need one real
CUDA GPU and a downloadable model; none of them run in CPU CI.

| script | what it produces |
|---|---|
| `nvfp4_budget_profile.py` | one cell: p50 wall, GPU busy (union of device intervals), host idle, and the per-bucket kernel table for one family / arm / resolution / graphs setting |
| `nvfp4_budget_attention_ab.py` | SDPA backends A/B in one process, paired by seed, with LPIPS and latent deltas against cuDNN |
| `nvfp4_budget_vae_numerics.py` | eager vs `torch.compile`d VAE decode on the SAME captured latent: compile wall, steady decode, max-abs, LPIPS |
| `nvfp4_budget_summarise.py` | the cell JSONs -> one Markdown report; reads only and recomputes nothing |

Buckets are assigned by kernel NAME plus the phase WINDOW the kernel lands in, first match wins,
because a CUDA-graph replay carries no correlation id back to a host op. `text_encoder` and
`vae_decode` are window overrides, so a GEMM inside the text encoder is text-encoder time and is not
double counted as fp8. Bucket shares are of GPU busy, so they do not sum to 100 when kernels
overlap.

`--attention shipped` forces nothing. Studio pins cuDNN SDPA on every denoiser whenever a speed tier
is active and then resets the process-wide diffusers backend to native, so the recorded
`attention_engaged` reads `native` while the kernels that actually run are cuDNN: the kernel names
in the report are the evidence, not the flag.

## Commands

Set the picks once. `MODEL` is a repo id or a local pick directory, `BASE` is the dense base repo a
GGUF pick dequantises against, and `CKPT` is an NVFP4 prequant checkpoint (omit both `--prequant-path`
and `--nvfp4-backend` for the fp8 arm).

```bash
export OUT=outputs/nvfp4_budget
export ZIMG=unsloth/Z-Image-Turbo-GGUF   ZIMG_GGUF=z_image_turbo-Q4_K.gguf   ZIMG_STEPS=9
export FLUX=unsloth/FLUX.1-schnell-GGUF  FLUX_GGUF=flux1-schnell-Q4_K_M.gguf FLUX_STEPS=4
export QWEN=unsloth/Qwen-Image-GGUF      QWEN_GGUF=qwen-image-Q4_K_M.gguf    QWEN_STEPS=20
```

One image cell, fp8 arm, both graph arms (`{graphs}` in `--out` is required by `--graphs both`):

```bash
python3 scripts/nvfp4_budget_profile.py \
  --family z-image --model "$ZIMG" --gguf-filename "$ZIMG_GGUF" --base-repo Tongyi-MAI/Z-Image-Turbo \
  --arm fp8 --resolution 1024 --steps "$ZIMG_STEPS" --graphs both --attention shipped \
  --warmups 3 --timed 7 --profiled 2 \
  --out "$OUT/z-image_1024_fp8_graphs{graphs}_shipped.json"
```

The same cell on the NVFP4 arm adds the checkpoint and the kernel backend:

```bash
python3 scripts/nvfp4_budget_profile.py \
  --family z-image --model "$ZIMG" --gguf-filename "$ZIMG_GGUF" --base-repo Tongyi-MAI/Z-Image-Turbo \
  --arm nvfp4 --prequant-path Z-Image-Turbo-NVFP4.pt --nvfp4-backend flashinfer \
  --resolution 1024 --steps "$ZIMG_STEPS" --graphs both --attention shipped \
  --out "$OUT/z-image_1024_nvfp4_graphs{graphs}_shipped.json"
```

Video is the same script with `--backend video`, a `WxH` resolution and `--frames`. The video loader
resolves the family's hosted prequant repo itself, so `--prequant-path` is ignored there:

```bash
python3 scripts/nvfp4_budget_profile.py --backend video \
  --family wan2.2-ti2v-5b --model Wan-AI/Wan2.2-TI2V-5B-Diffusers --arm nvfp4 \
  --resolution 1280x704 --frames 121 --steps 50 --graphs on --attention shipped \
  --out "$OUT/wan2.2-ti2v-5b_1280x704x121_nvfp4_graphson_shipped.json"
```

An eager reference for a family whose repeated block compiles (`TORCHDYNAMO_DISABLE=1` is not the
shipped configuration, it is the reference the shipped one is measured against), and the
whole-transformer compile arm:

```bash
TORCHDYNAMO_DISABLE=1 python3 scripts/nvfp4_budget_profile.py ... --out "$OUT/flux.1_1024_fp8_graphson_shipped_nodynamo.json"
python3 scripts/nvfp4_budget_profile.py ... --compile whole --out "$OUT/z-image_1024_nvfp4_wholecompile_graphson_shipped.json"
python3 scripts/nvfp4_budget_profile.py ... --compile off --out "$OUT/z-image_1024_nvfp4_nocompile_graphson_shipped.json"
```

`--compile off` loads the shipped `eager` tier: every lossless optimisation of `default`
(channels-last VAE, cudnn.benchmark, fp16 accumulation, the attention pin) and no torch.compile,
which is the compile-free control the other two arms are read against. The tier each run actually
loaded is in the JSON as `speed_mode_loaded`, beside the requested `compile_mode`.

VAE decode compile, before and after, at 1024 with graphs on. `UNSLOTH_DIFFUSION_COMPILE_VAE=0` is
the U-Net-only behaviour the switch replaced; `auto` is the default:

```bash
for FAM_ARM in "z-image nvfp4" "flux.1 nvfp4" "qwen-image fp8"; do
  set -- $FAM_ARM
  for VAE in 0 auto; do
    UNSLOTH_DIFFUSION_COMPILE_VAE=$VAE python3 scripts/nvfp4_budget_profile.py \
      --family "$1" --arm "$2" --resolution 1024 --graphs on --attention shipped \
      --model ... --steps ... --tag "${1}_1024_${2}_graphson_shipped_vae$VAE" \
      --out "outputs/nvfp4_vae_compile/${1}_1024_${2}_graphson_shipped_vae$VAE.json"
  done
done
```

Eager vs compiled decode on the same captured latent, which is where the LPIPS number comes from:

```bash
python3 scripts/nvfp4_budget_vae_numerics.py \
  --family z-image --model "$ZIMG" --gguf-filename "$ZIMG_GGUF" --base-repo Tongyi-MAI/Z-Image-Turbo \
  --arm nvfp4 --prequant-path Z-Image-Turbo-NVFP4.pt --resolution 1024 --steps "$ZIMG_STEPS" \
  --png-dir outputs/nvfp4_vae_compile/png --out outputs/nvfp4_vae_compile/numerics_z-image_nvfp4.json
```

Attention backends, one process, seven rotations. Every switch costs a `set_attention_backend` on
every denoiser, a `cuda_graph.reset_all` and one warm render to re-capture on the new kernel, and
that warm render is never timed:

```bash
python3 scripts/nvfp4_budget_attention_ab.py \
  --family z-image --model "$ZIMG" --gguf-filename "$ZIMG_GGUF" --base-repo Tongyi-MAI/Z-Image-Turbo \
  --arm nvfp4 --prequant-path Z-Image-Turbo-NVFP4.pt --resolution 1024 --steps "$ZIMG_STEPS" \
  --graphs on --backends cudnn,flash,efficient --rotations 7 \
  --out "$OUT/attn_ab_z-image_1024_nvfp4_graphson.json"
```

The report, once the cells are on disk. `--notes` splices a prose file in after the index table,
which is where a run's caveats belong:

```bash
python3 scripts/nvfp4_budget_summarise.py --results-dir "$OUT" --out "$OUT/image_budget.md" \
  --title "NVFP4 image time budget" --notes "$OUT/notes.md"
```

## Reading a cell honestly

- `clean` is true only when the contention bookends before and after the cell both read clean. A
  cell that is not clean shared the card with another CUDA context and its wall is not comparable.
- Wall is the median of the unprofiled renders. The profiled renders are for attribution only and
  carry their own overhead, reported as `profiler_overhead_ratio`.
- The phase hooks synchronize three times per render. That cost is measured
  (`phase_sync_overhead_s`) on hooked-but-unprofiled renders rather than assumed negligible.
- Two cells of the same configuration can differ in wall while agreeing on GPU busy to half a
  percent. That difference is host time and is reported as `host_idle_s`, never smoothed away.
- `neutralise_bitsandbytes` and `patch_hub_compat` are compatibility shims for a broken-but-present
  bitsandbytes and for huggingface_hub < 1.0 against a diffusers that imports hub 1.x names. Both
  are no-ops when the environment is healthy, both only flip in-process flags, and both are recorded
  in every JSON so a reader can see whether they fired.
