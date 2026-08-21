# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
API Routes
"""

from routes.training import router as training_router
from routes.models import router as models_router
from routes.inference import router as inference_router
from routes.inference import studio_router as inference_studio_router
from routes.video import router as video_router
from routes.datasets import router as datasets_router
from routes.auth import router as auth_router
from routes.data_recipe import router as data_recipe_router
from routes.export import router as export_router
from routes.training_history import router as training_history_router
from routes.chat_history import router as chat_history_router
from routes.cursor_import import router as cursor_import_router
from routes.claude_import import router as claude_import_router
from routes.providers import router as providers_router

from routes.openai_codex_auth import router as openai_codex_auth_router
from routes.mcp_servers import router as mcp_servers_router
from routes.rag import router as rag_router
from routes.research_runs import router as research_runs_router
from routes.youtube import router as youtube_router

__all__ = [
    "training_router",
    "models_router",
    "inference_router",
    "inference_studio_router",
    "video_router",
    "datasets_router",
    "auth_router",
    "data_recipe_router",
    "export_router",
    "training_history_router",
    "chat_history_router",
    "cursor_import_router",
    "claude_import_router",
    "providers_router",
    "openai_codex_auth_router",
    "mcp_servers_router",
    "rag_router",
    "research_runs_router",
    "youtube_router",
]

# Bind the re-export so the import-hoist verifier counts it as used.
_ = (rag_router, video_router, research_runs_router, youtube_router)
