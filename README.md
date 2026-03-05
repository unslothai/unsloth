<p align="center">
  <img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20white%20text.png" alt="Unsloth Studio" width="400"/>
</p>

<h3 align="center">🦥 Unsloth Studio</h3>

<p align="center">
  A modern, full-stack web interface for fine-tuning, managing, and chatting with large language models — locally or in the cloud.
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#project-structure">Project Structure</a>
</p>

---

## Features

| Area | Capabilities |
|---|---|
| **Training** | Configure and launch LoRA / QLoRA fine-tuning jobs with real-time SSE progress streaming, live loss charts, and one-click stop / resume |
| **Model Management** | Browse, load, and manage Hugging Face hub models and local checkpoints |
| **Inference** | Interactive chat playground for testing fine-tuned models |
| **Dataset Tools** | Upload, preview, and prepare datasets (JSON, CSV, Parquet, PDF, DOCX) |
| **Export** | Export & push trained adapters to the Hugging Face Hub |
| **Auth** | Token-based authentication with JWT access / refresh flow and first-time setup token |

## Quick Start

### Prerequisites

| Requirement | Linux / WSL | Windows |
|---|---|---|
| **GPU** | NVIDIA GPU with working driver | NVIDIA GPU with working driver |
| **Python** | 3.11 – 3.13 | 3.11 – 3.13 |
| **Git** | Pre-installed on most distros | Auto-installed by setup script (via `winget`) |
| **CMake** | Pre-installed or `sudo apt install cmake` | Auto-installed by setup script (via `winget`) |
| **C++ compiler** | `build-essential` (auto-detected) | Visual Studio Build Tools 2022 (auto-installed by setup script) |
| **CUDA Toolkit** | Optional — setup auto-detects `nvcc` | Auto-installed by setup script (version matched to driver) |

> [!NOTE]
> On **WSL**, the setup script will also run `sudo apt-get install build-essential cmake curl git libcurl4-openssl-dev` so that GGUF export works in non-interactive subprocesses. You may be prompted for your password during setup.

---

### Linux / Windows WSL

```bash
# 1. Clone the repo
git clone https://github.com/unslothai/unsloth-studio.git
cd unsloth-studio

# 2. Run setup (installs Node, builds frontend, creates .venv, builds llama.cpp)
bash setup.sh

# 3. Open a new terminal (or source your shell rc), then launch:
unsloth-studio -H 0.0.0.0 -p 8000
```

<details>
<summary><b>What does <code>setup.sh</code> do?</b></summary>

1. Installs **Node.js ≥ 20** via nvm (if needed)
2. Runs `npm install && npm run build` for the React frontend
3. Detects the best **Python 3.11 – 3.13** on your system and creates a `.venv`
4. Installs all Python dependencies (unsloth, PyTorch with CUDA, triton kernels, etc.)
5. On **WSL**: pre-installs build dependencies via `apt-get`
6. Clones and builds **llama.cpp** at `~/.unsloth/llama.cpp` (GPU-accelerated if CUDA is found)
7. Registers `unsloth-studio` and `unsloth-ui` shell aliases in your shell rc (bash, zsh, fish, or ksh)

</details>

---

### Windows (Native)

> [!IMPORTANT]
> Requires an **NVIDIA GPU** — CPU-only machines are not supported on Windows.

```powershell
# 1. Clone the repo
git clone https://github.com/unslothai/unsloth-studio.git
cd unsloth-studio

# 2. Run setup (Right-click → "Run with PowerShell", or from a terminal):
.\setup.bat
# Or directly:
powershell -ExecutionPolicy Bypass -File setup.ps1
```

After setup completes, **open a new terminal** and run:

```powershell
# PowerShell
unsloth-studio -H 0.0.0.0 -p 8000

# Or cmd.exe
unsloth-studio -H 0.0.0.0 -p 8000
```

<details>
<summary><b>What does <code>setup.ps1</code> do?</b></summary>

1. Enables **Windows Long Paths** (required for deep dependency trees — prompts for UAC)
2. Auto-installs missing system tools via `winget`: **Git**, **CMake**, **Visual Studio Build Tools 2022**, **CUDA Toolkit** (version-matched to your driver), **Node.js LTS**, **Python 3.12**, **OpenSSL dev**
3. Builds the React frontend (`npm install && npm run build`)
4. Creates a `.venv` and installs all Python dependencies (including CUDA-enabled PyTorch from the official index)
5. Sets `TORCHINDUCTOR_CACHE_DIR=C:\tc` to avoid Windows MAX_PATH issues with Triton
6. Clones and builds **llama.cpp** at `%USERPROFILE%\.unsloth\llama.cpp` with CUDA + Visual Studio
7. Registers `unsloth-studio` and `unsloth-ui` commands in both PowerShell profile and `cmd.exe` (via batch files on PATH)

</details>

---

### Google Colab

The setup script auto-detects Colab and installs everything into the existing system Python (no venv):

```python
!bash setup.sh
```

---

### Launching the Studio

After setup on any platform, the command is the same:

```bash
unsloth-studio -H 0.0.0.0 -p 8000
```

| Flag | Description |
|---|---|
| `-H` / `--host` | Bind address (`0.0.0.0` for all interfaces, `127.0.0.1` for local only) |
| `-p` / `--port` | Port number (default: `8000`) |

On **first launch**, a one-time setup token is printed to the console. Open the URL shown in your browser and use this token to create your admin account.

> [!TIP]
> This repo is in active development. After pulling new changes, **always re-run the setup script** (`bash setup.sh` or `.\setup.bat`) to pick up dependency and build updates.

## API Reference

All endpoints require a valid JWT `Authorization: Bearer <token>` header (except `/api/auth/*` and `/api/health`).

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/system` | System info (GPU, CPU, memory) |
| `POST` | `/api/auth/signup` | Create account (requires setup token on first run) |
| `POST` | `/api/auth/login` | Login and receive JWT tokens |
| `POST` | `/api/auth/refresh` | Refresh an expired access token |
| `GET` | `/api/auth/status` | Check if auth is initialized |
| `POST` | `/api/train/start` | Start a training job |
| `POST` | `/api/train/stop` | Stop a running training job |
| `POST` | `/api/train/reset` | Reset training state |
| `GET` | `/api/train/status` | Get current training status |
| `GET` | `/api/train/metrics` | Get training metrics (loss, LR, steps) |
| `GET` | `/api/train/stream` | SSE stream of real-time training progress |
| `GET` | `/api/models/` | List available models |
| `POST` | `/api/inference/chat` | Send a chat message for inference |
| `GET` | `/api/datasets/` | List / manage datasets |

> Full interactive docs are available at `/docs` (Swagger UI) and `/redoc` when the server is running.

## CLI Commands

The Unsloth CLI (`cli.py`) provides the following commands:

```
Usage: cli.py [COMMAND]

Commands:
  train             Fine-tune a model
  inference         Run inference on a trained model
  export            Export a trained adapter
  list-checkpoints  List saved checkpoints
  ui                Launch the Unsloth Studio web UI
  studio            Launch the studio (alias)
```

## Project Structure

```
new-ui-prototype/
├── cli.py                     # CLI entry point
├── cli/                       # Typer CLI commands
│   └── commands/
│       ├── train.py
│       ├── inference.py
│       ├── export.py
│       ├── ui.py
│       └── studio.py
├── setup.sh                   # Bootstrap script (Linux / WSL / Colab)
├── setup.ps1                  # Bootstrap script (Windows native)
├── setup.bat                  # Wrapper to launch setup.ps1 via double-click
├── install_python_stack.py    # Cross-platform Python dependency installer
└── studio/
    ├── backend/
    │   ├── main.py            # FastAPI app & middleware
    │   ├── run.py             # Server launcher (uvicorn)
    │   ├── auth/              # Auth storage & JWT logic
    │   ├── routes/            # API route handlers
    │   │   ├── training.py
    │   │   ├── models.py
    │   │   ├── inference.py
    │   │   ├── datasets.py
    │   │   └── auth.py
    │   ├── models/            # Pydantic request/response schemas
    │   ├── core/              # Training engine & config
    │   ├── utils/             # Hardware detection, helpers
    │   └── requirements.txt
    ├── frontend/
    │   ├── src/
    │   │   ├── features/      # Feature modules
    │   │   │   ├── auth/      # Login / signup flow
    │   │   │   ├── training/  # Training config & monitoring
    │   │   │   ├── studio/    # Main studio workspace
    │   │   │   ├── chat/      # Inference chat UI
    │   │   │   ├── export/    # Model export flow
    │   │   │   └── onboarding/# Onboarding wizard
    │   │   ├── components/    # Shared UI components (shadcn)
    │   │   ├── hooks/         # Custom React hooks
    │   │   ├── stores/        # Zustand state stores
    │   │   └── types/         # TypeScript type definitions
    │   ├── package.json
    │   └── vite.config.ts
    └── tests/                 # Backend test suite
```

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.html).

Copyright © 2026 Unsloth AI.
