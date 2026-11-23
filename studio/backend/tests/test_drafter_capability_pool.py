# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Queued drafter checks must not hold the shared default executor that other routes use."""

import asyncio
import threading

import core.training.drafter as drafter
import routes.training as training


def test_queued_drafter_checks_leave_the_default_executor_free(monkeypatch):
    release = threading.Event()

    def slow_probe(model_name, _hf_token, _key):
        release.wait(30)
        return {"model_name": model_name, "supported": False}

    monkeypatch.setattr(drafter, "_probe_capability", slow_probe)
    monkeypatch.setattr(training, "managed_account", lambda: None)

    async def scenario():
        checks = [
            asyncio.create_task(
                training.get_drafter_capability(
                    model_name = f"org/model-{i}",
                    current_subject = "u",
                    via_api_key = False,
                    hf_token = None,
                )
            )
            for i in range(64)
        ]
        await asyncio.sleep(0.5)
        try:
            other = await asyncio.wait_for(asyncio.to_thread(lambda: "ok"), timeout = 5)
        finally:
            release.set()
            answers = await asyncio.gather(*checks)
        return other, answers

    other, answers = asyncio.run(scenario())
    assert other == "ok"
    assert [a["model_name"] for a in answers] == [f"org/model-{i}" for i in range(64)]
