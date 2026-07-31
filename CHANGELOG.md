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

## 2026.7.6

### What's Changed

- Kimi K3 runs locally with Unsloth Dynamic GGUFs. Moonshot AI's 2.8T
  parameter MoE has 104B active parameters, native vision and a 1M context
  window.
- Parallel chat keeps several conversations generating at once, so a new chat
  no longer interrupts the answer already streaming.
- Deep Research turns a local model into a full research workflow that plans,
  searches, organizes evidence and writes a cited report.
- AMD support covers more GPUs, with more reliable ROCm inference and
  training on Windows, WSL and Linux.
- Intel XPU brings local chat and training to Intel Arc and Data Center GPUs,
  and DoRA is selectable alongside LoRA and full fine-tuning.

### Kimi K3

Kimi K3 loads with thinking on, and low, high and max reasoning efforts are
all supported. Multi-GPU setups are detected automatically and expert layers
can be offloaded to system memory.

It is a very large model, so plan hardware accordingly:

- `UD-IQ1_S` is 595GB on disk.
- `UD-Q4_K_XL` is 1.51TB on disk.
- `UD-Q8_K_XL` is 1.56TB on disk, for lossless inference.

Read the [Kimi K3 guide](https://unsloth.ai/docs/models/kimi-k3) and learn
more about
[Unsloth Dynamic 2.0 GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs).

### Parallel chat

- 4 llama-server slots by default, adjustable in the web UI.
- Tools, uploads, self-healing and agents stay isolated between chats.
- Stop one chat without interrupting others or restarting the server.
- The slot count is reduced automatically when memory is limited.
- Reloading a model still stops active chats after confirmation.

### Deep Research

- Review and edit the plan before research begins.
- Follow progress and collected sources while it works.
- Resume or cancel without losing completed research.
- Allow or block websites to control source selection.
- Sources and citations stay saved with each run.
- Unsupported claims, contradictions and unresolved gaps are checked before
  writing, and recommendations without direct evidence are marked as testable
  inferences.

One run can be active per chat, and Deep Research currently works with local
models. Optional full-page grounding reads top search results into temporary
RAG before synthesis; enable it with `UNSLOTH_RESEARCH_AUTO_SCRAPE=1`. It is
off by default because it adds scraping time and needs extra context.

### Improved AMD support

This builds on the initial
[AMD release and setup guide](https://unsloth.ai/docs/basics/amd):

- Detection is better for RDNA2, Radeon, Ryzen, Strix Halo and workstation
  GPUs.
- MI50 and Radeon VII support 16-bit LoRA and full fine-tuning on Linux.
- 4-bit NaNs, library conflicts and long RDNA startup stalls are fixed.
- Unsupported Windows HIP GPUs can fall back to Vulkan.
- Vulkan devices show their real names and can be selected individually.
- Clean Windows installs no longer require Winget or developer tools.

### Also in this release

- Large exports can use every visible GPU, avoiding GPU 0 memory limits.
- MLX gains better VLM and LoRA support, and the inference sidecar is
  activated before detection.
- Incorrect `flash_attention_2` model output is fixed.
- Unsloth and its saving utilities now work without bitsandbytes.
- `.json` datasets and the Qwen3.5 / Qwen3.6 MoE and GRPO notebook setup are
  fixed.
- Hub download folders are easier to find and open.
- `unsloth start <agent> --as-subagent` lets Claude Code, Codex, OpenCode and
  Pi delegate tasks to a local Unsloth model.

## 2026.7.5

### What's Changed

- AMD support is here. Train, run RL, chat with and deploy 500+ models on
  Radeon, Instinct, Ryzen and data center GPUs across Windows, WSL and Linux,
  up to 2x faster with 70% less VRAM and no accuracy loss.
- Local speech to text dictation runs fully offline, with slim Whisper bundles
  and a picker for custom models.
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
