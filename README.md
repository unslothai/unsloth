<h1 align="center" style="margin:0;">
  <a href="https://unsloth.ai/docs"><picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20white%20text.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20black%20text.png">
    <img alt="Unsloth logo" src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20black%20text.png" height="80" style="max-width:100%;">
  </picture></a>
</h1>
<h3 align="center" style="margin: 0; margin-top: 0;">
Unsloth Studio lets you run and train models locally.
</h3>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-unsloth-news">News</a> •
  <a href="#-install">Quickstart</a> •
  <a href="#-free-notebooks">Notebooks</a> •
  <a href="https://unsloth.ai/docs">Documentation</a>
</p>
<br>
<a href="https://unsloth.ai/docs/new/studio">
<img alt="unsloth studio ui homepage" src="https://github.com/user-attachments/assets/53ae17a9-d975-44ef-9686-efb4ebd0454d" style="max-width: 100%; margin-bottom: 0;"></a>

## ⚡ Get started

#### macOS, Linux, WSL:
```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```
#### Windows:
```powershell
irm https://unsloth.ai/install.ps1 | iex
```
#### Community:

- [Discord](https://discord.gg/unsloth)
- [𝕏 (Twitter)](https://x.com/UnslothAI)
- [Reddit](https://reddit.com/r/unsloth)

## ⭐ Features
Unsloth Studio (Beta) lets you run and train text, [audio](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning), [embedding](https://unsloth.ai/docs/new/embedding-finetuning), [vision](https://unsloth.ai/docs/basics/vision-fine-tuning) models on Windows, Linux and macOS.

### Inference
* **Search + download + run models** including GGUF, LoRA adapters, safetensors
* **Export models**: [Save or export](https://unsloth.ai/docs/new/studio/export) models to GGUF, 16-bit safetensors and other formats.
* **Tool calling**: Support for [self-healing tool calling](https://unsloth.ai/docs/new/studio/chat#auto-healing-tool-calling) and web search
* **[Code execution](https://unsloth.ai/docs/new/studio/chat#code-execution)**: lets LLMs test code in Claude artifacts and sandbox environments
* **[API inference endpoint](https://unsloth.ai/docs/basics/api)**: Deploy and run local LLMs in Claude Code, Codex tools with Unsloth
* [Auto set inference settings](https://unsloth.ai/docs/new/studio/chat#auto-parameter-tuning) and customize chat templates.
* We work directly with teams behind [gpt-oss](https://docs.unsloth.ai/new/gpt-oss-how-to-run-and-fine-tune#unsloth-fixes-for-gpt-oss), [Qwen3](https://www.reddit.com/r/LocalLLaMA/comments/1kaodxu/qwen3_unsloth_dynamic_ggufs_128k_context_bug_fixes/), [Llama 4](https://github.com/ggml-org/llama.cpp/pull/12889), [Mistral](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B/discussions/18), [Gemma 1-3](https://news.ycombinator.com/item?id=39671146), and [Phi-4](https://unsloth.ai/blog/phi4), where we’ve fixed bugs that improve model accuracy.
* Chat with images, audio, PDFs, code, DOCX and more. [Connect API providers](https://unsloth.ai/docs/integrations/connections) (OpenAI, Anthropic) or servers (vLLM, Ollama).
* [**Compare any two models**](https://unsloth.ai/docs/new/studio/chat#model-arena) side by side with the same prompt.
* **OpenAI/Anthropic-compatible APIs**: Serve local models through `/v1/chat/completions`, `/v1/responses` and `/v1/messages`.
* **Connect local models to agents**: Use `unsloth start` with Claude Code, Codex, Hermes and more.
* **Web/PDF search** can read PDF papers, manuals and other PDF results.
* **GGUF hardware controls**: Choose GPUs/layers, offload MoE experts, use multi-GPU or Tensor Parallelism.
* The opt-in **MCP control endpoint** lets AI clients manage models, training, recipes and exports.
### Training
* Train and RL **500+ models** up to **2x faster** with **70% less VRAM**; MoE up to **12x faster**.
* Train and run RL on [AMD GPUs](https://unsloth.ai/docs/basics/amd) across Windows, WSL and Linux.
* **Data Recipes**: [Auto-create datasets](https://unsloth.ai/docs/new/studio/data-recipe) from **PDF, CSV, DOCX** etc. Edit data in a visual-node workflow.
* **[Reinforcement Learning](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)** uses **80% less VRAM** for GRPO, FP8 and vision RL, with 7x longer contexts.
* [**Long-context training**](https://unsloth.ai/docs/new/3x-faster-training-packing): **3x faster**, 30% less VRAM and 500K+ context.
* Supports LoRA/QLoRA, full fine-tuning, RL, pretraining, 4-bit, 16-bit and FP8.
* Custom Triton and mathematical **kernels** built with PyTorch and Hugging Face.
* **Observability**: Monitor training live, track loss and GPU usage and customize graphs.
* [Multi-GPU](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth) training is supported, with major improvements coming soon.

## 🚀 Unsloth Start

[Unsloth Start](https://unsloth.ai/docs/integrations/unsloth-start) connects [Claude Code](https://unsloth.ai/docs/basics/claude-code), [Codex](https://unsloth.ai/docs/basics/codex) and other agents to local models with one command.

Start Unsloth, load a model, open your project folder, then run:

```bash
unsloth start claude
```

Replace `claude` with any supported agent:

| Agent | Command |
| --- | --- |
| Claude Code | `unsloth start claude` |
| OpenAI Codex | `unsloth start codex` |
| Hermes Agent | `unsloth start hermes` |
| OpenClaw | `unsloth start openclaw` |
| OpenCode | `unsloth start opencode` |
| Pi Coding Agent | `unsloth start pi` |

## 📥 Install
Unsloth can be used in two ways: through **[Unsloth Studio](https://unsloth.ai/docs/new/studio/)**, the web UI, or through **Unsloth Core**, the code-based version. Each has different requirements.

### Unsloth Studio (web UI)
Unsloth Studio (Beta) works on **Windows, Linux, WSL** and **macOS**.

* **CPU:** Supported for Chat and Data Recipes currently
* **NVIDIA:** Training works on RTX 30/40/50, Blackwell, DGX Spark, Station and more
* **macOS:** Training, MLX and GGUF inference are ALL supported.
* **AMD:** Training, RL, chat and deployment work on Windows, WSL and Linux. [Read the AMD guide](https://unsloth.ai/docs/basics/amd).
* **Vulkan:** GGUF inference is supported on [compatible GPUs, including Intel GPUs](https://github.com/unslothai/unsloth/pull/5819).
* **Multi-GPU:** Available now, with a major upgrade on the way

#### macOS, Linux, WSL:
```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```
Use the same command to update.

#### Windows:
```powershell
irm https://unsloth.ai/install.ps1 | iex
```
Use the same command to update.

#### Launch
```bash
unsloth studio -p 8888
```
For LAN or cloud access, add `-H 0.0.0.0` (raw port only; add `--cloudflare` for a public URL). By default, Unsloth is accessible only locally.

To reach Unsloth over HTTPS, use `unsloth studio --secure`. Unsloth stays bound to localhost and is reached only through a free Cloudflare tunnel, which publishes it at a public `https://*.trycloudflare.com` URL (it fails closed if the tunnel can't start, so the raw port is never exposed). This makes Unsloth reachable from the internet, so anyone with the link and API key can use it and run code: keep your API key private (see Remote access below).

#### Docker
Use our [Docker image](https://hub.docker.com/r/unsloth/unsloth) ```unsloth/unsloth``` container. Run:
```bash
docker run -d -e JUPYTER_PASSWORD="mypassword" \
  -p 8888:8888 -p 8000:8000 -p 2222:22 \
  -v $(pwd)/work:/workspace/work \
  --gpus all \
  unsloth/unsloth
  ```

#### Developer, Nightly, Uninstall
To see developer, nightly and uninstallation etc. instructions, see [advanced installation](#-advanced-installation).

### Unsloth Core (code-based)
#### Linux, WSL:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv unsloth_env --python 3.13
source unsloth_env/bin/activate
uv pip install unsloth --torch-backend=auto
```
#### Windows:
```powershell
winget install -e --id Python.Python.3.13
winget install --id=astral-sh.uv  -e
uv venv unsloth_env --python 3.13
.\unsloth_env\Scripts\activate
uv pip install unsloth --torch-backend=auto
```
For Windows, `pip install unsloth` works only if you have PyTorch installed. Read our [Windows Guide](https://unsloth.ai/docs/get-started/install/windows-installation).
You can use the same Docker image as Unsloth Studio.

#### AMD, Intel:
For RTX 50x, B200, 6000 GPUs: `uv pip install unsloth --torch-backend=auto`. Read our guides for: [Blackwell](https://unsloth.ai/docs/blog/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth) and [DGX Spark](https://unsloth.ai/docs/blog/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth). <br>
To install Unsloth on **AMD** and **Intel** GPUs, follow our [AMD Guide](https://unsloth.ai/docs/basics/amd) and [Intel Guide](https://unsloth.ai/docs/get-started/install/intel).

## 📒 Free Notebooks

Train for free with our notebooks. You can use our new [free Unsloth Studio notebook](https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb) to run and train models for free in a web UI.
Read our [guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide). Add dataset, run, then deploy your trained model.

| Model | Free Notebooks | Performance | Memory use |
|-----------|---------|--------|----------|
| **Gemma 4 (E2B)**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E2B)-Vision.ipynb)               | 1.5x faster | 50% less |
| **Qwen3.5 (4B)**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)_Vision.ipynb)               | 1.5x faster | 60% less |
| **gpt-oss (20B)**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb)               | 2x faster | 70% less |
| **Qwen3.5 GSPO**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)_Vision_GRPO.ipynb)               | 2x faster | 70% less |
| **gpt-oss (20B): GRPO**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb)               | 2x faster | 80% less |
| **Qwen3: Advanced GRPO**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)               | 2x faster | 70% less |
| **embeddinggemma (300M)**    | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/EmbeddingGemma_(300M).ipynb)               | 2x faster | 20% less |
| **Mistral Ministral 3 (3B)**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Ministral_3_VL_(3B)_Vision.ipynb)               | 1.5x faster | 60% less |
| **Llama 3.1 (8B) Alpaca**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb)               | 2x faster | 70% less |
| **Llama 3.2 Conversational**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb)               | 2x faster | 70% less |
| **Orpheus-TTS (3B)**     | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb)               | 1.5x faster | 50% less |

- See all our notebooks for: [Kaggle](https://github.com/unslothai/notebooks?tab=readme-ov-file#-kaggle-notebooks), [GRPO](https://unsloth.ai/docs/get-started/unsloth-notebooks#grpo-reasoning-rl-notebooks), [TTS](https://unsloth.ai/docs/get-started/unsloth-notebooks#text-to-speech-tts-notebooks), [embedding](https://unsloth.ai/docs/new/embedding-finetuning) & [Vision](https://unsloth.ai/docs/get-started/unsloth-notebooks#vision-multimodal-notebooks)
- See [all our models](https://unsloth.ai/docs/get-started/unsloth-model-catalog) and [all our notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks)
- See detailed documentation for Unsloth [here](https://unsloth.ai/docs)

## 🦥 Unsloth News
- **AMD training**: Train, run RL, chat and deploy on AMD GPUs across Windows, WSL and Linux. [Guide](https://unsloth.ai/docs/basics/amd)
- **GGUF hardware controls**: Choose GPU/layer placement, offload MoE experts and use multi-GPU or Tensor Parallelism. [#6414](https://github.com/unslothai/unsloth/pull/6414)
- **Local models for any agent**: Use `unsloth start` with Claude Code, Codex, Hermes, OpenCode, OpenClaw, Pi and more through Unsloth's OpenAI- and Anthropic-compatible APIs. [Guide](https://unsloth.ai/docs/basics/api)
- **MCP control endpoint**: Let compatible clients manage models, training, recipes, checkpoints and exports. [#7191](https://github.com/unslothai/unsloth/pull/7191)
- **Local inference reliability**: Resume long chats faster, recover stalled downloads and reuse existing GGUF files. [#7204](https://github.com/unslothai/unsloth/pull/7204) • [#6858](https://github.com/unslothai/unsloth/pull/6858) • [#7209](https://github.com/unslothai/unsloth/pull/7209)
- **New models**: [Qwen-AgentWorld](https://huggingface.co/unsloth/Qwen-AgentWorld-35B-A3B-GGUF), [Ornith](https://huggingface.co/unsloth/models?search=ornith), [Kimi K2.7 Code](https://unsloth.ai/docs/models/kimi-k2.7-code) and [MiniMax M3](https://unsloth.ai/docs/models/minimax-m3)
- **GLM-5.2**: Run Z.ai's 744B-parameter, 1M-context open model locally with Unsloth Dynamic GGUFs. [Guide](https://unsloth.ai/docs/models/glm-5.2)
- **DeepSeek-V4**: Run DeepSeek-V4-Flash locally with corrected multi-turn and tool-calling behavior. [Guide](https://unsloth.ai/docs/models/deepseek-v4)
- **DiffusionGemma**: Run and fine-tune Google's diffusion language model with 1.8x faster inference in Unsloth Studio. [Guide](https://unsloth.ai/docs/models/diffusiongemma)
- **Qwen3.6**: Run and train Qwen3.6 with MTP for 1.4-2.2x faster inference and NVFP4 quants for supported GPUs. [Guide](https://unsloth.ai/docs/models/qwen3.6)
- **Gemma 4**: Run and train Gemma 4 text, image and audio models with QAT, MTP, GGUF and MLX support. [Guide](https://unsloth.ai/docs/models/gemma-4)
- **MCP servers**: Connect local models to files, apps, databases and external tools through Model Context Protocol. [Guide](https://unsloth.ai/docs/basics/mcp)
- **Connections**: Mix local models with API providers (OpenAI, Anthropic) or servers (vLLM, Ollama) in the same interface. [Guide](https://unsloth.ai/docs/integrations/connections)
- **Introducing Unsloth Studio**: our new web UI for running and training LLMs. [Blog](https://unsloth.ai/docs/new/studio)
- Train **MoE LLMs 12x faster** with 35% less VRAM - DeepSeek, GLM, Qwen and gpt-oss. [Blog](https://unsloth.ai/docs/new/faster-moe)
- **Embedding models**: Unsloth now supports ~1.8-3.3x faster embedding fine-tuning. [Blog](https://unsloth.ai/docs/new/embedding-finetuning) • [Notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks#embedding-models)
- New **7x longer context RL** vs. all other setups, via our new batching algorithms. [Blog](https://unsloth.ai/docs/new/grpo-long-context)
- New RoPE & MLP **Triton Kernels** & **Padding Free + Packing**: 3x faster training & 30% less VRAM. [Blog](https://unsloth.ai/docs/new/3x-faster-training-packing)
- **500K Context**: Training a 20B model with >500K context is now possible on an 80GB GPU. [Blog](https://unsloth.ai/docs/blog/500k-context-length-fine-tuning)
- **FP8 & Vision RL**: You can now do FP8 & VLM GRPO on consumer GPUs. [FP8 Blog](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning) • [Vision RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/vision-reinforcement-learning-vlm-rl)

## 📥 Advanced Installation
The below advanced instructions are for Unsloth Studio. For Unsloth Core advanced installation, [view our docs](https://unsloth.ai/docs/get-started/install/pip-install#advanced-pip-installation).
#### Developer / Nightly / Experimental installs: macOS, Linux, WSL:
The developer install builds from the `main` branch, which is the latest (nightly) source.
```bash
git clone https://github.com/unslothai/unsloth
cd unsloth
./install.sh --local
unsloth studio -p 8888
```
To install into an isolated location (its own virtual env, `auth/`, `studio.db`, cache and llama.cpp build), set `UNSLOTH_STUDIO_HOME` and pass it again at launch:
```bash
UNSLOTH_STUDIO_HOME="$PWD/.studio" ./install.sh --local
UNSLOTH_STUDIO_HOME="$PWD/.studio" unsloth studio -p 8888
```
Then to update :
```bash
cd unsloth && git pull
./install.sh --local
unsloth studio -p 8888
```

#### Developer / Nightly / Experimental installs: Windows PowerShell:
The developer install builds from the `main` branch, which is the latest (nightly) source.
```powershell
git clone https://github.com/unslothai/unsloth.git
cd unsloth
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install.ps1 --local
unsloth studio -p 8888
```
To install into an isolated location (its own virtual env, `auth/`, `studio.db`, cache and llama.cpp build), set `UNSLOTH_STUDIO_HOME` and pass it again at launch:
```powershell
$env:UNSLOTH_STUDIO_HOME="$PWD\.studio"; .\install.ps1 --local
$env:UNSLOTH_STUDIO_HOME="$PWD\.studio"; unsloth studio -p 8888
```
Then to update :
```powershell
cd unsloth; git pull
.\install.ps1 --local
unsloth studio -p 8888
```

#### Remote access: `--secure` (HTTPS tunnel) vs raw port
By default `unsloth studio` binds to `127.0.0.1` (this machine only). To reach it from another device, pick one of:

- `--secure` (recommended): serve **only** through a free Cloudflare HTTPS link. Unsloth stays bound to localhost and the tunnel provides the public URL; it fails closed (does not start) if the tunnel can't come up, so the raw port is never exposed.
```bash
unsloth studio --secure -p 8888
```
- `-H 0.0.0.0`: bind the raw port on all network interfaces, reachable from anywhere on the network (subject to your firewall). It does not create a public internet URL; add `--cloudflare` to also publish an internet-reachable `https://*.trycloudflare.com` link even behind a firewall. Only use this on a network you trust.
```bash
unsloth studio -H 0.0.0.0 -p 8888
```
The Cloudflare tunnel is **off by default**: `-H 0.0.0.0` exposes the raw port only, not a public internet URL. Pair the wildcard bind with `--cloudflare` (`unsloth studio -H 0.0.0.0 --cloudflare`) to also publish a public `https://*.trycloudflare.com` link, or prefer `--secure` (above), which keeps the raw port private. `--cloudflare` has no effect on a loopback bind.

The first time Unsloth is published on a public URL (`--secure` or `--cloudflare`) with the auto-generated admin password still in place, it asks for a new admin password in the terminal (masked input with confirmation) before the public link goes up. Without an attached terminal it warns instead and keeps the bootstrap deadline: Unsloth shuts down after `UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT` (default 1 hour) unless the password is changed in the web UI.

For headless setups that cannot answer that prompt, set the initial admin password non-interactively with `--password` (only takes effect when no password is set yet; if one already exists it is a hard error, so rotate later with `unsloth studio reset-password`):

```bash
unsloth studio --secure --password 'your-strong-password'        # visible in `ps`/history
UNSLOTH_STUDIO_PASSWORD='your-strong-password' unsloth studio --secure   # via env var
printf '%s\n' 'your-strong-password' | unsloth studio --secure --password -   # via stdin
```

A literal `--password VALUE` is visible in the process list and shell history, so prefer the `UNSLOTH_STUDIO_PASSWORD` env var or `--password -` (stdin) for automation. This applies to any launch (public or a headless `-H 0.0.0.0` bind), and the password is set in the parent before the server binds, so it never reaches a re-executed child process.

Server-side tools (web search, Python and terminal code execution) run as your user and are on by default. Anyone who can reach the server with the API key can run code on this machine, so keep your API key private and pass `--disable-tools` when exposing Unsloth.

#### Advanced launch options
Installer options can be passed as environment variables. On macOS, Linux and WSL place the variable after the pipe so the shell passes it to `sh`; on Windows set it with `$env:` before piping to `iex`.

Skip PyTorch (GGUF-only mode):
```bash
curl -fsSL https://unsloth.ai/install.sh | UNSLOTH_NO_TORCH=1 sh
```
```powershell
$env:UNSLOTH_NO_TORCH=1; irm https://unsloth.ai/install.ps1 | iex
```

Skip the post-install prompt that starts Unsloth (useful for automated installs):
```bash
curl -fsSL https://unsloth.ai/install.sh | UNSLOTH_SKIP_AUTOSTART=1 sh
```
```powershell
$env:UNSLOTH_SKIP_AUTOSTART=1; irm https://unsloth.ai/install.ps1 | iex
```

Pin the Python version:
```bash
curl -fsSL https://unsloth.ai/install.sh | UNSLOTH_PYTHON=3.12 sh
```
```powershell
$env:UNSLOTH_PYTHON='3.12'; irm https://unsloth.ai/install.ps1 | iex
```

Install to a custom location with `UNSLOTH_STUDIO_HOME`:
```bash
curl -fsSL https://unsloth.ai/install.sh | UNSLOTH_STUDIO_HOME=/abs/path sh
```
```powershell
$env:UNSLOTH_STUDIO_HOME='C:\path'; irm https://unsloth.ai/install.ps1 | iex
```

On macOS, the installer defaults to the system certificate store (`UV_SYSTEM_CERTS=1`) so uv trusts the CAs in your Keychain, needed behind TLS-inspecting proxies (Cisco Umbrella, Zscaler, etc.). Opt out with:
```bash
curl -fsSL https://unsloth.ai/install.sh | UV_SYSTEM_CERTS=0 sh
```

Point the frontend build at a corporate npm mirror/proxy with `UNSLOTH_NPM_REGISTRY` (for the developer install behind a firewall that blocks `registry.npmjs.org`):
```bash
UNSLOTH_NPM_REGISTRY=https://artifactory.example.com/api/npm/npm/ ./install.sh --local
```
```powershell
$env:UNSLOTH_NPM_REGISTRY='https://artifactory.example.com/api/npm/npm/'; .\install.ps1 --local
```
It is threaded as `--registry` into the Unsloth frontend `npm`/`bun` installs; the supply-chain locks (7-day `min-release-age`, exact version pins) stay in force.

Cap Unsloth's native CPU thread pools on high-core hosts: `UNSLOTH_CPU_THREADS=8 unsloth studio -p 8888`.

#### Uninstall
The recommended way to fully remove Unsloth Studio is the matching uninstall script for your OS. It stops any running servers, removes the install dir, the launcher data dir, the desktop shortcut, and any platform-specific entries (macOS `.app` bundle + Launch Services on Mac; Start Menu, `HKCU\Software\Unsloth` registry key and user `PATH` entries on Windows):

* ​ **MacOS, WSL, Linux:** `curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.sh | sh`
* ​ **Windows (PowerShell):** `irm https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.ps1 | iex`

If you only want to drop the install dir and keep the launcher/shortcut for a later reinstall, you can instead run `rm -rf ~/.unsloth/studio` (Mac/Linux/WSL) or `Remove-Item -Recurse -Force "$HOME\.unsloth\studio"` (Windows). The model cache at `~/.cache/huggingface` is not touched by any of these.

For more info, [see our docs](https://unsloth.ai/docs/new/studio/install#uninstall).

#### Deleting model files

You can delete old model files either from the bin icon in model search or by removing the relevant cached model folder from the default Hugging Face cache directory. By default, HF uses:

* ​ **MacOS, Linux, WSL:** `~/.cache/huggingface/hub/`
* ​ **Windows:** `%USERPROFILE%\.cache\huggingface\hub\`

## 💚 Community and Links
| Type                                                                                                                                      | Links                                                                          |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| <img width="16" src="https://cdn.prod.website-files.com/6257adef93867e50d84d30e2/66e3d80db9971f10a9757c99_Symbol.svg" />  **Discord**                       | [Join Discord server](https://discord.com/invite/unsloth)                          |
| <img width="15" src="https://redditinc.com/hs-fs/hubfs/Reddit%20Inc/Brand/Reddit_Logo.png" />  **r/unsloth Reddit**                       | [Join Reddit community](https://reddit.com/r/unsloth)                          |
| 📚 **Documentation & Wiki**                                                                                                               | [Read Our Docs](https://unsloth.ai/docs)                                       |
| <img width="13" src="https://upload.wikimedia.org/wikipedia/commons/0/09/X_(formerly_Twitter)_logo_late_2025.svg" />  **Twitter (aka X)** | [Follow us on X](https://twitter.com/unslothai)                                |
| 🔮 **Our Models**                                                                                                                         | [Unsloth Catalog](https://unsloth.ai/docs/get-started/unsloth-model-catalog)   |
| ✍️ **Blog**                                                                                                                               | [Read our Blogs](https://unsloth.ai/blog)                                      |

### Citation

You can cite the Unsloth repo as follows:
```bibtex
@software{unsloth,
  author = {Daniel Han, Michael Han and Unsloth team},
  title = {Unsloth},
  url = {https://github.com/unslothai/unsloth},
  year = {2023}
}
```
If you trained a model with 🦥Unsloth, you can use this cool sticker!   <img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/made with unsloth.png" width="200" align="center" />

### License
Unsloth uses a dual-licensing model of Apache 2.0 and AGPL-3.0. The core Unsloth package remains licensed under **[Apache 2.0](https://github.com/unslothai/unsloth?tab=Apache-2.0-1-ov-file)**, while certain optional components, such as the Unsloth Studio UI are licensed under the open-source license **[AGPL-3.0](https://github.com/unslothai/unsloth?tab=AGPL-3.0-2-ov-file)**.

This structure helps support ongoing Unsloth development while keeping the project open source and enabling the broader ecosystem to continue growing.

### Thank You to
- The [llama.cpp library](https://github.com/ggml-org/llama.cpp) that lets users run and save models with Unsloth
- The Hugging Face team and their libraries: [transformers](https://github.com/huggingface/transformers) and [TRL](https://github.com/huggingface/trl)
- The Pytorch and [Torch AO](https://github.com/unslothai/unsloth/pull/3391) team for their contributions
- NVIDIA for their [NeMo DataDesigner](https://github.com/NVIDIA-NeMo/DataDesigner) library and their contributions
- And of course for every single person who has contributed or has used Unsloth!
