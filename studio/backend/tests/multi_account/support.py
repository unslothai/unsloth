# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small real-router application shared by integration tests and the perf runner."""

from fastapi import Depends, FastAPI

from auth import authentication
from routes import auth, chat_history


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(auth.router, prefix = "/api/auth")
    app.include_router(chat_history.router, prefix = "/api/chat")

    @app.get("/account-probe")
    async def account_probe(subject: str = Depends(authentication.get_current_subject)):
        return {"subject": subject}

    return app


def bearer(username: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {authentication.create_access_token(subject = username)}"}
