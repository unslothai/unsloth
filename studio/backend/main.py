"""
Main FastAPI application for Unsloth UI Backend
"""
import secrets
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from datetime import datetime

# Import routers
from routes import training_router, models_router, inference_router, datasets_router, auth_router, export_router
from auth import storage
from utils.hardware import detect_hardware
import utils.hardware.hardware as _hw_module

UNSLOTH_CACHE_DIR = Path(__file__).parent / "unsloth_compiled_cache"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: detect hardware, print setup token if needed. Shutdown: clean up compiled cache."""
    # Detect hardware first — sets DEVICE global used everywhere
    detect_hardware()

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
    shutil.rmtree(UNSLOTH_CACHE_DIR, ignore_errors=True)


# Create FastAPI app
app = FastAPI(
    title="Unsloth UI Backend",
    version="1.0.0",
    description="Backend API for Unsloth UI - Training and Model Management",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ Register API Routes ============

# Register routers
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(training_router, prefix="/api/train", tags=["training"])
app.include_router(models_router, prefix="/api/models", tags=["models"])
app.include_router(inference_router, prefix="/api/inference", tags=["inference"])
app.include_router(datasets_router, prefix="/api/datasets", tags=["datasets"])
app.include_router(export_router, prefix="/api/export", tags=["export"])


# ============ Health and System Endpoints ============

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Unsloth UI Backend"
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
        gpu_info["devices"].append({
            "index": mem_info.get("device", 0),
            "name": mem_info.get("device_name", "Unknown"),
            "memory_total_gb": round(mem_info.get("total_gb", 0), 2),
        })

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
    if build_path.exists():
        # Mount assets
        assets_dir = build_path / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/")
        async def serve_root():
            return FileResponse(build_path / "index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

        @app.get("/{full_path:path}")
        async def serve_frontend(full_path: str):
            if full_path.startswith("api"):
                return {"error": "API endpoint not found"}

            file_path = build_path / full_path
            if file_path.is_file():
                return FileResponse(file_path)

            return FileResponse(build_path / "index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

        return True
    return False

