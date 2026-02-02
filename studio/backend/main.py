"""
Main FastAPI application for Unsloth UI Backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from datetime import datetime

# Import routers
from routes import training_router, models_router

# Create FastAPI app
app = FastAPI(
    title="Unsloth UI Backend",
    version="1.0.0",
    description="Backend API for Unsloth UI - Training and Model Management"
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
app.include_router(training_router, prefix="/api/train", tags=["training"])
app.include_router(models_router, prefix="/api/models", tags=["models"])


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
    import torch
    import platform
    import psutil

    # GPU Info
    gpu_info = {"available": False, "devices": []}
    if torch.cuda.is_available():
        gpu_info["available"] = True
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info["devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "memory_total_gb": round(props.total_memory / 1e9, 2),
                }
            )

    # CPU & Memory
    memory = psutil.virtual_memory()

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory": {
            "total_gb": round(memory.total / 1e9, 2),
            "available_gb": round(memory.available / 1e9, 2),
            "percent_used": memory.percent,
        },
        "gpu": gpu_info,
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
            return FileResponse(build_path / "index.html")

        @app.get("/{full_path:path}")
        async def serve_frontend(full_path: str):
            if full_path.startswith("api"):
                return {"error": "API endpoint not found"}

            file_path = build_path / full_path
            if file_path.is_file():
                return FileResponse(file_path)

            return FileResponse(build_path / "index.html")

        return True
    return False

