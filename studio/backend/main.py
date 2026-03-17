# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Main FastAPI application for Unsloth UI Backend
"""

import os

# Suppress annoying C-level dependency warnings globally
os.environ["PYTHONWARNINGS"] = "ignore"

import shutil
import sys
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
    """Startup: detect hardware, seed default admin if needed. Shutdown: clean up compiled cache."""
    # Clean up any stale compiled cache from previous runs
    clear_unsloth_compiled_cache()

    # Remove stale .venv_overlay from previous versions — no longer used.
    # Version switching now uses .venv_t5/ (pre-installed by setup.sh).
    overlay_dir = Path(__file__).resolve().parent.parent.parent / ".venv_overlay"
    if overlay_dir.is_dir():
        shutil.rmtree(overlay_dir, ignore_errors = True)

    # Detect hardware first — sets DEVICE global used everywhere
    detect_hardware()

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

    # Build llama.cpp in the background if the binary is missing.
    # Studio starts immediately; GGUF features wait for the build to finish.
    from core.inference.llama_cpp_builder import get_llama_cpp_builder

    get_llama_cpp_builder().check_and_build()

    if storage.ensure_default_admin():
        bootstrap_pw = storage.get_bootstrap_password()
        app.state.bootstrap_password = bootstrap_pw
        print("\n" + "=" * 60)
        print("DEFAULT ADMIN ACCOUNT CREATED")
        print(
            "Sign in with the seeded credentials and change the password immediately:\n"
        )
        print(f"    username: {storage.DEFAULT_ADMIN_USERNAME}")
        print(f"    password: {bootstrap_pw}\n")
        print("=" * 60 + "\n")
    else:
        app.state.bootstrap_password = storage.get_bootstrap_password()
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
    platform_map = {"darwin": "mac", "win32": "windows", "linux": "linux"}
    device_type = platform_map.get(sys.platform, sys.platform)

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Unsloth UI Backend",
        "device_type": device_type,
    }


@app.get("/api/system")
async def get_system_info():
    """Get system information"""
    import platform
    import subprocess
    import psutil
    from utils.hardware import get_device, get_gpu_memory_info, DeviceType

    # GPU Info — query nvidia-smi for physical GPUs, filtered by
    # CUDA_VISIBLE_DEVICES when set (the frontend uses this for GGUF
    # fit estimation and llama-server respects CVD too).
    import os

    gpu_info: dict = {"available": False, "devices": []}

    device = get_device()
    if device == DeviceType.CUDA:
        # Parse CUDA_VISIBLE_DEVICES allowlist
        allowed_indices = None
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None and cvd.strip():
            try:
                allowed_indices = set(int(x.strip()) for x in cvd.split(","))
            except ValueError:
                pass  # Non-numeric (e.g. GPU-uuid), show all

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output = True,
                text = True,
                timeout = 10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 3:
                        idx = int(parts[0])
                        if allowed_indices is not None and idx not in allowed_indices:
                            continue
                        gpu_info["devices"].append(
                            {
                                "index": idx,
                                "name": parts[1],
                                "memory_total_gb": round(int(parts[2]) / 1024, 2),
                            }
                        )
                gpu_info["available"] = len(gpu_info["devices"]) > 0
        except Exception:
            pass

    # Fallback to torch-based single-GPU detection
    if not gpu_info["available"]:
        mem_info = get_gpu_memory_info()
        if mem_info.get("available"):
            gpu_info["available"] = True
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


def _strip_crossorigin(html_bytes: bytes) -> bytes:
    """Remove ``crossorigin`` attributes from script/link tags.

    Vite adds ``crossorigin`` by default which forces CORS mode on font
    subresource loads.  When Studio is served over plain HTTP, Firefox
    HTTPS-Only Mode does not exempt CORS font requests -- causing all
    @font-face downloads to fail silently.  Stripping the attribute
    makes them regular same-origin fetches that work on any protocol.
    """
    import re as _re

    html = html_bytes.decode("utf-8")
    html = _re.sub(r'\s+crossorigin(?:="[^"]*")?', "", html)
    return html.encode("utf-8")


def _inject_bootstrap(html_bytes: bytes, app: FastAPI) -> bytes:
    """Inject bootstrap credentials into HTML when password change is required.

    The script tag is only injected while the default admin account still
    has ``must_change_password=True``.  Once the user changes the password
    the HTML is served clean — no credentials leak.
    """
    import json as _json

    if not storage.requires_password_change(storage.DEFAULT_ADMIN_USERNAME):
        return html_bytes

    bootstrap_pw = getattr(app.state, "bootstrap_password", None)
    if not bootstrap_pw:
        return html_bytes

    payload = _json.dumps(
        {
            "username": storage.DEFAULT_ADMIN_USERNAME,
            "password": bootstrap_pw,
        }
    )
    tag = f"<script>window.__UNSLOTH_BOOTSTRAP__={payload}</script>"
    html = html_bytes.decode("utf-8")
    html = html.replace("</head>", f"{tag}</head>", 1)
    return html.encode("utf-8")


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
        content = _strip_crossorigin(content)
        content = _inject_bootstrap(content, app)
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
        if not file_path.is_relative_to(build_path.resolve()):
            return Response(status_code = 403)

        if file_path.is_file():
            return FileResponse(file_path)

        # Serve index.html as bytes — avoids Content-Length mismatch
        content = (build_path / "index.html").read_bytes()
        content = _strip_crossorigin(content)
        content = _inject_bootstrap(content, app)
        return Response(
            content = content,
            media_type = "text/html",
            headers = {"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    return True
