# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Queued token previews must not hold the shared default executor that other routes use."""

import asyncio
import threading

import core.training.token_preview as token_preview
import routes.models as models_routes
import routes.training as training
from models.training import TokenPreviewRequest
from utils.account_context import current_account_id


def test_queued_token_previews_leave_the_default_executor_free(monkeypatch, tmp_path):
    release = threading.Event()
    seen_accounts = []

    def slow_preview(
        key,
        _config,
        request,
        _record_load = None,
    ):
        seen_accounts.append(current_account_id())
        release.wait(30)
        return {"text": request["text"]}

    async def allowed(_model_name, _hf_token):
        return None

    monkeypatch.setattr(token_preview, "run_token_preview", slow_preview)
    monkeypatch.setattr(models_routes, "_require_model_access_or_caller_token", allowed)
    monkeypatch.setattr(training, "account_hf_token", lambda token: token)
    monkeypatch.setattr(training, "account_path", lambda *a, **k: None)

    async def scenario():
        previews = [
            asyncio.create_task(
                training.token_preview(
                    TokenPreviewRequest(model_name = str(tmp_path), text = f"row {i}"),
                    hf_token = None,
                    current_subject = "u",
                )
            )
            for i in range(64)
        ]
        await asyncio.sleep(0.5)
        try:
            other = await asyncio.wait_for(asyncio.to_thread(lambda: "ok"), timeout = 5)
        finally:
            release.set()
            answers = await asyncio.gather(*previews)
        return other, answers

    other, answers = asyncio.run(scenario())
    assert other == "ok"
    assert [a["text"] for a in answers] == [f"row {i}" for i in range(64)]
    # The account context still reaches the worker call.
    assert seen_accounts and all(a == seen_accounts[0] for a in seen_accounts)
