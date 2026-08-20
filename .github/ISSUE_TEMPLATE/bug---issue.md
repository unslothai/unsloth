---
name: Bug / Issue
about: Bug / Issue
title: "[Bug] Please fill in your issue title here."
labels: bug, feature request
assignees: ''
type: Bug

---

# Bug report

> **Before submitting:** Search existing issues and update Unsloth if possible. Please answer every relevant question below. Never share API keys, Hugging Face tokens, passwords, cookies, private prompts, datasets, or other sensitive information.

## Environment

**Where are you using Unsloth?**
- [ ] Studio desktop application
- [ ] Studio web UI (`unsloth studio`)
- [ ] Unsloth CLI
- [ ] Python package or notebook
- [ ] Colab or Kaggle

**Operating system and version:**

**GPU model(s) and accelerator backend:**
Examples: NVIDIA CUDA, AMD ROCm, Intel XPU, Apple MLX, or CPU.

**Versions:**
For Studio, copy the Unsloth, package, desktop, and llama.cpp versions from **Settings → About**, when available.

## What happened?

**Steps to reproduce:**
1.
2.
3.

**Expected behavior:**

**Actual behavior:**

**Model and operation involved:**
Include the model ID or filename and the operation, such as installation, startup, download, training, inference, GGUF, diffusion, or export.

## Diagnostics and logs

> **Privacy reminder:** Remove tokens, passwords, cookies, private prompts, sensitive local paths, and other private information. Do not upload your entire `~/.unsloth/studio` directory—it can contain authentication state, databases, chats, datasets, and models.

Only include the section relevant to your setup.

<details>
<summary><strong>Studio desktop application</strong></summary>

Click **Copy Diagnostics** on the error, startup, or update screen and paste the result below:

```text
Paste diagnostics here
```

If **Copy Diagnostics** is unavailable, attach the newest relevant files from:
- Windows: `%USERPROFILE%\.unsloth\studio\`
- Linux/macOS: `~/.unsloth/studio/`

Useful files include:
- `tauri.log` and, if relevant, `tauri.log.1`
- `logs/install-*.log`, `logs/update-*.log`, `logs/repair-*.log`, or `logs/backend-*.log`
- The newest `logs/server/server-*.log`

</details>

<details>
<summary><strong>Studio web UI</strong></summary>

Attach the newest relevant items:
- Server log: `~/.unsloth/studio/logs/server/server-*.log`
  - Windows folder: `%USERPROFILE%\.unsloth\studio\logs\server\`
- Linux/macOS shortcut launches only: `~/.local/share/unsloth/studio.log`
- Terminal output, if Studio was launched from a terminal
- Browser Console errors for browser-only problems

If you configured `UNSLOTH_STUDIO_HOME` or `STUDIO_HOME`, use `<CUSTOM_STUDIO_HOME>/logs/` instead.

</details>

<details>
<summary><strong>Model-specific logs</strong></summary>

Include these only when applicable:
- GGUF/llama.cpp: newest `~/.unsloth/studio/logs/llama-server/*.log`
- Diffusion GGUF: newest `~/.unsloth/studio/logs/diffusion-server/*.log`
- Training resume/checkpoint bug: relevant `trainer_state.json`

</details>

<details>
<summary><strong>Python package or notebook</strong></summary>

Paste the complete traceback and a minimal reproduction:

```python
# Remove all tokens and private data.
```

Also include the Python, Unsloth, `unsloth_zoo`, PyTorch, Transformers, and TRL versions, plus `nvidia-smi` output when applicable.

</details>

## Additional context

Add screenshots or any other information that may help.⏎
