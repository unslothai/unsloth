"""
API Routes
"""

from routes.training import router as training_router
from routes.models import router as models_router
from routes.inference import router as inference_router
from routes.datasets import router as datasets_router
from routes.auth import router as auth_router
from routes.export import router as export_router

__all__ = [
    "training_router",
    "models_router",
    "inference_router",
    "datasets_router",
    "auth_router",
    "export_router",
]