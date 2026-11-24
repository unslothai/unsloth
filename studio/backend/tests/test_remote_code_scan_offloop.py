# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The remote-code scan must not run on the event loop, and a managed account that may not
use remote code must not make Studio download and scan a repo's Python first."""

import asyncio
import time

import pytest
from fastapi import HTTPException

import routes.models as models_routes
import utils.security as security
import utils.security.consent as consent


def test_a_slow_scan_does_not_freeze_the_event_loop(monkeypatch, tmp_path):
    def slow_target(*_args, **_kwargs):
        time.sleep(1.0)
        raise HTTPException(status_code = 418, detail = "stop here")

    monkeypatch.setattr(security, "load_scan_target", slow_target)
    monkeypatch.setattr(models_routes.account_access, "managed_account", lambda: None)

    async def scenario():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        tick_task = asyncio.create_task(ticker())
        await asyncio.sleep(0)
        with pytest.raises(HTTPException) as raised:
            await models_routes.scan_model_remote_code(
                model_name = str(tmp_path),
                hf_token = None,
                prefer_local_cache = False,
                model_local_path = None,
                model_snapshot_path = None,
                model_snapshot_repo_id = None,
                allow_ambient_token = False,
                current_subject = "unsloth",
            )
        tick_task.cancel()
        return ticks, raised.value.status_code

    ticks, status = asyncio.run(scenario())
    assert status == 418
    assert ticks >= 20, ticks


def test_a_refused_managed_account_skips_the_code_download(monkeypatch):
    def never(*_args, **_kwargs):
        raise AssertionError("downloaded the repo's Python for an account that is refused anyway")

    monkeypatch.setattr(consent, "_config_has_auto_map", lambda *_a, **_k: True)
    monkeypatch.setattr(consent, "repo_remote_code_files", never)
    monkeypatch.setattr(consent, "managed_remote_code_refused", lambda: True)

    decision = consent.evaluate_remote_code_consent_for_targets(
        ["someone/huge-code-repo"], None, trust_remote_code = True
    )
    assert decision.blocked and decision.has_remote_code
    assert decision.block_cause == consent.MANAGED_REFUSED
    assert decision.approvable is False
    assert decision.reason == consent.MANAGED_REMOTE_CODE_REFUSAL
