# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

"""
Main FastAPI application for Unsloth UI Backend
"""

import os

# Suppress annoying C-level dependency warnings globally
os.environ["PYTHONWARNINGS"] = "ignore"

import secrets
import shutil
import warnings
from contextlib import asynccontextmanager

# Suppress annoying dependency warnings in production
if os.getenv("ENVIRONMENT_TYPE", "production") == "production":
    warnings.filterwarnings("ignore")
    # Alternatively, you can be more specific:
    # warnings.filterwarnings("ignore", category=DeprecationWarning)
    # warnings.filterwarnings("ignore", module="triton.*")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, Response
from pathlib import Path
from datetime import datetime

# Import routers
from routes import (
    auth_router,
    data_recipe_router,
    datasets_router,
    export_router,
    inference_router,
    models_router,
    training_router,
)
from auth import storage
from utils.hardware import detect_hardware, get_device, DeviceType
import utils.hardware.hardware as _hw_module

from utils.cache_cleanup import clear_unsloth_compiled_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: detect hardware, print setup token if needed. Shutdown: clean up compiled cache."""
    # Clean up any stale compiled cache from previous runs
    clear_unsloth_compiled_cache()

    # Remove stale .venv_overlay from previous versions — no longer used.
    # Version switching now uses .venv_t5/ (pre-installed by setup.sh).
    overlay_dir = Path(__file__).resolve().parent.parent.parent / ".venv_overlay"
    if overlay_dir.is_dir():
        shutil.rmtree(overlay_dir, ignore_errors = True)

    # Detect hardware first — sets DEVICE global used everywhere
    detect_hardware()

    # Disable flex attention on Blackwell+ GPUs (sm_120 and above)
    if get_device() == DeviceType.CUDA:
        import torch

        props = torch.cuda.get_device_properties(0)
        sm_version = props.major * 10 + props.minor
        if sm_version >= 120:
            os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "0"
            import structlog
            from loggers import get_logger

            get_logger(__name__).info(
                f"GPU sm_{sm_version} detected — setting UNSLOTH_FLEX_ATTENTION=0"
            )

    # Pre-cache the helper GGUF model for LLM-assisted dataset detection.
    # Runs in a background thread so it doesn't block server startup.
    import threading

    def _precache():
        try:
            from utils.datasets.llm_assist import precache_helper_gguf

            precache_helper_gguf()
        except Exception:
            pass  # non-critical

    threading.Thread(target = _precache, daemon = True).start()

    if not storage.is_initialized():
        setup_token = secrets.token_urlsafe(32)
        storage.save_setup_token(setup_token)
        print("\n" + "=" * 60)
        print("FIRST-TIME SETUP REQUIRED")
        print("Use this one-time setup token to create your admin account:\n")
        print(f"    {setup_token}\n")
        print("This token can only be used once.")
        print("=" * 60 + "\n")
    yield
    # Cleanup
    _hw_module.DEVICE = None
    clear_unsloth_compiled_cache()


# Create FastAPI app
app = FastAPI(
    title = "Unsloth UI Backend",
    version = "1.0.0",
    description = "Backend API for Unsloth UI - Training and Model Management",
    lifespan = lifespan,
)

# Initialize structured logging
from loggers.config import LogConfig
from loggers.handlers import LoggingMiddleware

logger = LogConfig.setup_logging(
    service_name = "unsloth-studio-backend",
    env = os.getenv("ENVIRONMENT_TYPE", "production"),
)

app.add_middleware(LoggingMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],  # In production, specify allowed origins
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# ============ Register API Routes ============

# Register routers
app.include_router(auth_router, prefix = "/api/auth", tags = ["auth"])
app.include_router(training_router, prefix = "/api/train", tags = ["training"])
app.include_router(models_router, prefix = "/api/models", tags = ["models"])
app.include_router(inference_router, prefix = "/api/inference", tags = ["inference"])

# OpenAI-compatible endpoints: mount the same inference router at /v1
# so external tools (Open WebUI, SillyTavern, etc.) can use the
# standard /v1/chat/completions path.
app.include_router(inference_router, prefix = "/v1", tags = ["openai-compat"])
app.include_router(datasets_router, prefix = "/api/datasets", tags = ["datasets"])
app.include_router(data_recipe_router, prefix = "/api/data-recipe", tags = ["data-recipe"])
app.include_router(export_router, prefix = "/api/export", tags = ["export"])


# ============ Health and System Endpoints ============


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Unsloth UI Backend",
    }


@app.get("/api/system")
async def get_system_info():
    """Get system information"""
    import platform
    import psutil
    from utils.hardware import get_device, get_gpu_memory_info, DeviceType

    # GPU Info — uses the hardware module (works on CUDA, MPS, CPU)
    mem_info = get_gpu_memory_info()
    gpu_info = {"available": mem_info.get("available", False), "devices": []}

    if mem_info.get("available"):
        gpu_info["devices"].append(
            {
                "index": mem_info.get("device", 0),
                "name": mem_info.get("device_name", "Unknown"),
                "memory_total_gb": round(mem_info.get("total_gb", 0), 2),
            }
        )

    # CPU & Memory
    memory = psutil.virtual_memory()

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "device_backend": get_device().value,
        "cpu_count": psutil.cpu_count(),
        "memory": {
            "total_gb": round(memory.total / 1e9, 2),
            "available_gb": round(memory.available / 1e9, 2),
            "percent_used": memory.percent,
        },
        "gpu": gpu_info,
    }


@app.get("/api/system/hardware")
async def get_hardware_info():
    """Return GPU name, total VRAM, and key ML package versions."""
    from utils.hardware import get_gpu_summary, get_package_versions

    return {
        "gpu": get_gpu_summary(),
        "versions": get_package_versions(),
    }


# ============ Serve Frontend (Optional) ============


def setup_frontend(app: FastAPI, build_path: Path):
    """Mount frontend static files (optional)"""
    if not build_path.exists():
        return False

    # Mount assets
    assets_dir = build_path / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory = assets_dir), name = "assets")

    @app.get("/")
    async def serve_root():
        content = (build_path / "index.html").read_bytes()
        return Response(
            content = content,
            media_type = "text/html",
            headers = {"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        if full_path.startswith("api"):
            return {"error": "API endpoint not found"}

        file_path = (build_path / full_path).resolve()

        # Block path traversal — ensure resolved path stays inside build_path
        if not str(file_path).startswith(str(build_path.resolve())):
            return Response(status_code = 403)

        if file_path.is_file():
            return FileResponse(file_path)

        # Serve index.html as bytes — avoids Content-Length mismatch
        content = (build_path / "index.html").read_bytes()
        return Response(
            content = content,
            media_type = "text/html",
            headers = {"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    return True
