# Changelog

Release notes for Unsloth and Unsloth Studio.

Unsloth Studio reads this file to show release notes inside the "New Unsloth
version" update popup. Edit it here and the popup picks the change up on the
next update check, with no release or rebuild required.

## Format

Every release is a level-2 heading whose first token is the version, optionally
followed by a date:

```md
## 2026.7.6 - 2026-07-22
```

`## [2026.7.6] - 2026-07-22` and `## v2026.7.6` also work. Everything under a
heading, up to the next level-2 heading, is that release's notes and renders as
Markdown in the popup.

Notes are matched to one exact version. When Studio offers an update to
`2026.7.6` it renders the `2026.7.6` section and nothing else. If that section
is missing, the popup links out to the online changelog rather than showing
notes from an unrelated release, so a new version needs its own section here
before its notes can appear.

Keep the newest release at the top. Lead each bullet with the change itself:
the collapsed popup highlights the first sentence and dims the rest.
`## Unreleased` is ignored by the popup, so it is safe to stage notes there and
rename the heading at release time.

<!-- Add new releases directly below this line. -->

## Unreleased

## 2026.7.5

### What's Changed

- AMD support is here. Train, run RL, chat with and deploy 500+ models on
  Radeon, Instinct, Ryzen and data center GPUs across Windows, WSL and Linux,
  up to 2x faster with 70% less VRAM and no accuracy loss.
- Intel XPU support lands in Studio, so Arc and Data Center GPUs run chat and
  training alongside the NVIDIA, AMD and Apple paths.
- Local speech to text dictation runs fully offline, with slim Whisper bundles
  and a picker for custom models.
- DoRA training is available in Studio, selectable next to LoRA and full
  fine-tuning in the training tab.
- The update popup previews release notes inline, pulled from this file and
  matched to the exact version being offered.

### AMD, 23 July update

Our AMD collaboration, custom Triton kernels and math algorithms bring local
training and inference to AMD hardware. The 23 July update builds on the
[AMD release](https://github.com/unslothai/unsloth/releases/tag/v0.1.501-beta):

- RDNA2 and Gorgon Halo are supported, and the installer no longer fails to
  detect GPUs on Strix Halo and other AMD cards.
- RDNA4 handling is better, and HIP and ROCm failures are caught and fixed
  automatically instead of stopping the install.
- Unified memory safetensors loading is 2x faster, with much faster gradient
  checkpointing on unified memory devices.
- Voice dictation through whisper.cpp has preliminary support.
- Rollback environments left by installs no longer eat 5GB of disk. They are
  cleaned up automatically.

Optimized ROCm builds cover GGUF and safetensors inference, and ROCm
compatibility is improved for MI300X and MI325X. Full guide:
[unsloth.ai/docs/basics/amd](https://unsloth.ai/docs/basics/amd).

### Running larger models

- Automatic GPU placement, or pick exactly which GPUs and layers to use.
- Move MoE expert layers into system memory so larger models fit.
- Split a model across several GPUs, or use tensor parallelism.
- Hardware settings are saved per model and quant.

### Also in this release

- Remote access with `unsloth studio --secure` over free HTTPS via Cloudflare.
- Web search reads PDF papers and manuals, and parallel tool calls, reasoning
  output and tool retries are more reliable.
- The model download location is configurable, so weights can live on a second
  drive instead of the default cache.
- Stalled Hugging Face XET downloads retry over standard HTTP, and existing
  GGUF files are reused instead of downloaded again.
