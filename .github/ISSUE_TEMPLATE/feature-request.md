---
name: Feature Request
about: New features, model support, ideas
title: "[Feature]"
labels: feature request
assignees: ''
type: Feature

---

# Feature request

> **Before submitting:** Search existing issues and feature requests first. Please describe the problem you want solved—not only the proposed implementation. Never share API keys, Hugging Face tokens, passwords, private prompts, datasets, or other sensitive information.

## Product area

**Where would this feature be used?**
- [ ] Unsloth Desktop (Tauri app)
- [ ] Unsloth Studio web UI (usually launched with `unsloth studio`)
- [ ] Unsloth Core (Python package or notebook)
- [ ] Unsloth CLI
- [ ] More than one of the above

**Which part of Unsloth does this affect?**
- [ ] Installation, startup, updates, or settings
- [ ] Model Hub, downloads, or local model management
- [ ] New model, model family, or architecture support
- [ ] Hardware backend, accelerator, performance, or memory usage
- [ ] Training or fine-tuning
- [ ] Chat or text inference
- [ ] Image or video generation and diffusion
- [ ] Audio, speech-to-text, or text-to-speech
- [ ] Datasets or Data Recipes
- [ ] Export, quantization, or GGUF
- [ ] Local serving or the OpenAI-compatible API
- [ ] Agents, tools, MCP, search, or RAG
- [ ] User interface, accessibility, or localization
- [ ] Python API, CLI, or third-party integration
- [ ] Other

## Problem to solve

**What problem or limitation are you experiencing?**
Explain the use case, who it affects, and why the current behavior is insufficient.

**What are you trying to accomplish?**
A concrete example or workflow is especially helpful.

## Proposed feature

**What would you like Unsloth to do?**

**What would the ideal workflow look like?**
1.
2.
3.

**Which parts are essential, and which are optional?**

## Platform details

Only complete the section relevant to your request.

<details>
<summary><strong>Unsloth Desktop (Tauri app)</strong></summary>

**Operating system(s):**
- [ ] Windows
- [ ] Linux
- [ ] macOS

**Is this specific to the desktop shell, or should it be part of shared Studio?**
Desktop-specific examples include native file dialogs, clipboard or drag-and-drop handling, the system tray, notifications, launch at login, app updates, window behavior, and backend process lifecycle.

**Should the same workflow also work in the Studio web UI?**

</details>

<details>
<summary><strong>Unsloth Studio web UI</strong></summary>

**How do you run or access Studio?**
Examples: local browser via `unsloth studio`, remote machine, Docker, Colab, or a Cloudflare tunnel.

**Is this a shared Studio UI/backend feature, or browser-specific behavior?**
Browser-specific examples include uploads, downloads, browser clipboard behavior, responsive layout, and accessibility.

**Should the same workflow also work in Unsloth Desktop?**

</details>

<details>
<summary><strong>Unsloth Core (Python package or notebook)</strong></summary>

**Preferred Python API, if any:**

```python
# Show how you would like to use the feature.
```

**Relevant model or model family:**

**Relevant workflow:**
Examples: model loading, training, inference, saving, export, quantization, or deployment.

**Must this remain compatible with an existing public API or workflow?**

</details>

<details>
<summary><strong>Unsloth CLI</strong></summary>

**Relevant command or proposed command:**
Existing command areas include `train`, `inference`, `chat`, `export`, `studio`, `run`, and `start`.

```console
# Show the command or flags you would like to use.
```

**Should this also be available through the Python API, Studio, or both?**

**Must existing command behavior or output remain compatible?**

</details>


<details>
<summary><strong>New model or hardware support</strong></summary>

**Model ID or link:**

**Model architecture and task, if known:**
Examples: text generation, vision-language, embedding, image/video diffusion, speech-to-text, or text-to-speech.

**Where should the model work?**
Examples: training, chat/inference, serving, export, Studio, Core, or CLI.

**Target hardware and backend:**
Examples: NVIDIA CUDA, AMD ROCm, Intel XPU, Apple MLX, Vulkan, or CPU.

**Reference implementation or upstream support:**
Link relevant model cards, papers, Transformers support, or working code.

</details>

## Alternatives considered

**How do you handle this today?**
Include workarounds, scripts, other tools, or related Unsloth features you have tried.

**Are there other approaches that would also solve the problem?**

## Examples and additional context

Add mockups, screenshots, links, sample code, related issues, or any other details that clarify the request.

**Would you be willing to help test this feature?**
- [ ] Yes
- [ ] No
- [ ] Maybe
