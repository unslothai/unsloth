# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The single-flight auto-download slot names its repo and quant only to the account that started it."""

import asyncio
import time

import pytest

from core.inference import openai_auto_download as auto
from utils.account_context import AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")
PRIVATE = "alice/private-checkpoint-GGUF"


@pytest.fixture
def alice_downloading(monkeypatch):
    monkeypatch.setattr(
        auto,
        "_active",
        auto._Active(
            repo_id = PRIVATE, variant = "Q4_K_M", started_at = time.time(), account_id = ALICE.account_id
        ),
    )

    async def downloadable(repo_id, hf_token):
        return True

    monkeypatch.setattr(auto, "_is_downloadable_model", downloadable)
    monkeypatch.setattr(auto, "_is_not_servable", lambda repo_id, hf_token: False)


def _refusal(account):
    return run_as(account, lambda: asyncio.run(auto.maybe_auto_download("bob/public-GGUF:Q4_K_M")))


def test_another_account_gets_a_generic_busy_answer(alice_downloading):
    refusal = _refusal(BOB)
    print(f"bob sees: {refusal.message}")
    assert refusal is not None and refusal.code == "model_download_busy"
    assert "alice" not in refusal.message and "Q4_K_M'" not in refusal.message.split("Retry")[0]
    assert "bob/public-GGUF:Q4_K_M" in refusal.message


def test_the_downloading_account_still_sees_its_own_repo(alice_downloading):
    refusal = _refusal(ALICE)
    assert refusal is not None and PRIVATE in refusal.message
