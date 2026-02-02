"""
API Routes
"""

from routes.training import router as training_router
from routes.models import router as models_router

__all__ = ["training_router", "models_router"]