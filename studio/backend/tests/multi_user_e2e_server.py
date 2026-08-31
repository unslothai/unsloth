# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small production-route host for the multi-user Playwright CI scenario.

It deliberately mounts the real auth/chat routers and storage code without importing
the GPU application graph. That keeps the browser regression lane CPU-only while the
regular Backend CI still exercises these routes inside the full backend test suite.
"""

from __future__ import annotations

import os
import secrets
import sys
import types
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from auth import storage as auth_storage

# routes/__init__.py intentionally assembles the entire production API and therefore
# imports the ML stack. This focused CPU runner needs two real leaf routers only.
routes_path = Path(__file__).resolve().parents[1] / "routes"
routes_package = types.ModuleType("routes")
routes_package.__path__ = [str(routes_path)]
sys.modules["routes"] = routes_package
from routes import auth, chat_history


FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

if not auth_storage.is_initialized():
    auth_storage.create_initial_user(
        "unsloth",
        os.environ.get("STUDIO_E2E_ADMIN_PASSWORD", "owner-e2e-password"),
        secrets.token_urlsafe(64),
        is_admin = True,
    )

app = FastAPI()
app.include_router(auth.router, prefix = "/api/auth")
app.include_router(chat_history.router, prefix = "/api/chat")


@app.get("/api/models/list")
def empty_model_list():
    """Keep the focused browser harness visually honest without loading the ML graph."""
    return {"models": [], "default_models": []}


@app.get("/api/inference/status")
def idle_inference_status():
    return {
        "active_model": None,
        "is_vision": False,
        "loading": [],
        "loaded": [],
    }


@app.get("/{requested_path:path}")
def serve_frontend(requested_path: str):
    """Serve built assets when present and index.html for client-side routes."""
    if requested_path == "api" or requested_path.startswith("api/"):
        raise HTTPException(status_code = 404, detail = "Route is outside this focused harness")
    requested = (FRONTEND_DIST / requested_path).resolve()
    if requested.is_relative_to(FRONTEND_DIST.resolve()) and requested.is_file():
        return FileResponse(requested)
    index = FRONTEND_DIST / "index.html"
    if not index.is_file():
        raise HTTPException(status_code = 503, detail = "Frontend has not been built")
    return FileResponse(index)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "127.0.0.1", port = 8767)
