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

### One-command setup

```bash
bash setup.sh
```

This script will:
1. Install **Node.js ≥ 20** via nvm (if needed)
2. Build the frontend to `studio/frontend/dist`
3. Create a Python virtual environment and install all dependencies (including `unsloth`)
4. Register a convenient `unsloth-ui` shell alias

### Launch the studio

```bash
# After setup, open a new terminal (or source ~/.bashrc), then inside your working directory:
unsloth-ui -H 0.0.0.0 -p 8000
```

On **first launch**, a one-time setup token is printed to the console. Use it in the browser to create your admin account.

As this repo is in continuous development, please make sure to run the setup.sh file everytime you pull new changes from the repo.

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
├── setup.sh                   # One-command bootstrap script
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
