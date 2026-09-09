---
name: Bug / Issue
about: Bug / Issue
title: "[Bug] Please fill in your issue title here."
labels: bug, feature request
assignees: ''

---

---
name: Unsloth Studio Bug
about: Report a problem with the Unsloth Studio desktop app or web UI
title: "[Unsloth Bug] "
labels: bug
assignees: ""
---

<!--
Search existing issues before submitting. Please do not remove the questions.
Never post API keys, Hugging Face tokens, passwords, cookies, private prompts,
datasets, or other sensitive information.
-->

## Environment

**Where are you using Unsloth?**

- [ ] Unsloth desktop application
- [ ] Unsloth web UI (`unsloth studio`)
- [ ] Unsloth CLI
- [ ] Python package or notebook
- [ ] Colab or Kaggle

**Operating system and version:**

**GPU model(s) and accelerator backend:**
<!-- Examples: NVIDIA CUDA, AMD ROCm, Intel XPU, Apple MLX, CPU. -->

**Versions:**
<!--
For Unsloth, copy the Unsloth, package, desktop, and llama.cpp versions from
Settings → About when available.
-->

## What happened?

**Steps to reproduce:**

1.
2.
3.

**Expected behavior:**

**Actual behavior:**

**Model and operation involved:**
<!--
Include the model ID or filename and whether this involved installation,
startup, download, training, inference, GGUF, diffusion, or export.
-->

## Diagnostics and logs

<!--
Remove API keys, Hugging Face tokens, passwords, cookies, private prompts,
local paths you do not want to disclose, and other sensitive information.
Do not upload your entire ~/.unsloth/studio directory: it can contain auth
state, databases, chats, datasets, and models.
-->

### Desktop application

Click **Copy Diagnostics** on the error, startup, or update screen and paste the
result below:

```text
PASTE COPY DIAGNOSTICS HERE
```

If **Copy Diagnostics** is unavailable, attach the newest relevant files from:

- Windows: `%USERPROFILE%\.unsloth\studio\`
- Linux/macOS: `~/.unsloth/studio/`

Useful files include:

- `tauri.log` and, if relevant, `tauri.log.1`
- `logs/install-*.log`, `logs/update-*.log`, `logs/repair-*.log`, or `logs/backend-*.log`
- The newest `logs/server/server-*.log`

### Unsloth web UI

Attach the newest relevant items:

- Linux/macOS/Windows: `~/.unsloth/studio/logs/server/server-*.log`
  (use `%USERPROFILE%\.unsloth\studio\logs\server\` on Windows)
- Linux/macOS shortcut launches only: `~/.local/share/unsloth/studio.log`
- Terminal output if Unsloth was launched from a terminal
- Browser Console errors for browser-only problems

If you configured `UNSLOTH_STUDIO_HOME` or `STUDIO_HOME`, look under
`<CUSTOM_STUDIO_HOME>/logs/` instead.

### Model-specific logs, if applicable

- GGUF/llama.cpp: newest `~/.unsloth/studio/logs/llama-server/*.log`
- Diffusion GGUF: newest `~/.unsloth/studio/logs/diffusion-server/*.log`
- Training resume/checkpoint bug: relevant `trainer_state.json`

### Python package or notebook, if applicable

Paste the complete traceback and minimal reproduction:

```python
# Remove all tokens and private data.
```

Also include Python, Unsloth, unsloth_zoo, PyTorch, Transformers, and TRL
versions, plus `nvidia-smi` output when applicable.

## Additional context

<!-- Add screenshots or anything else that may help. -->
