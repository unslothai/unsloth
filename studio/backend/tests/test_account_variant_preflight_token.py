# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The GGUF variant preflight must accept a caller token that proves Hub access.

A managed account picking a quantization for a private or gated repo it has never
downloaded holds no grant yet -- the grant is only recorded once a download finishes --
so the grant/public check alone turns the whole select-then-download flow into a 404.
The cache-only direction (no token) still needs the grant.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from auth import policy
from hub.services.models import account_access as access
from hub.services.models import gguf_variants
from hub.utils import hf_tokens
from utils.account_context import AccountContext, arun_as

ALICE = AccountContext("a" * 32, "alice")

TOKEN = "hf_alice_token"
REPO = "Org/PrivateGGUF"


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(access, "_public_repos", {})
    hf_tokens.reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    monkeypatch.setattr(hf_tokens, "_probe_repo_access", lambda *_a, **_k: True)
    # No cached copy anywhere: the answer has to come from the Hub listing below.
    monkeypatch.setattr(gguf_variants, "select_gguf_cache_snapshot", lambda *_a, **_k: None)
    monkeypatch.setattr(gguf_variants, "_quants_from_state", lambda *_a, **_k: None)
    monkeypatch.setattr(
        gguf_variants,
        "list_gguf_variants",
        lambda *_a, **_k: (
            [
                SimpleNamespace(
                    filename = "model-Q4_K_M.gguf",
                    quant = "Q4_K_M",
                    display_label = "Q4_K_M",
                    size_bytes = 1234,
                    shard_count = 0,
                )
            ],
            False,
            [],
        ),
    )
    yield
    hf_tokens.reset_repo_access_cache()


def _hub(monkeypatch, calls: list) -> None:
    """A private repo: anonymous asks are refused, the caller's token is accepted."""

    def repo_info(repo, **kwargs):
        calls.append(kwargs.get("token"))
        if not isinstance(kwargs.get("token"), str):
            error = Exception("private repo")
            error.response = SimpleNamespace(status_code = 404)
            raise error
        return SimpleNamespace(private = True, gated = False)

    monkeypatch.setattr(access, "HfApi", lambda: SimpleNamespace(repo_info = repo_info))


def _variants(hf_token):
    return asyncio.run(
        arun_as(ALICE, gguf_variants.get_gguf_variants_answer(REPO, hf_token = hf_token))
    )


def test_a_caller_token_authorizes_the_variant_preflight_before_any_download(monkeypatch):
    calls: list = []
    _hub(monkeypatch, calls)

    answer = _variants(TOKEN)

    assert [v.quant for v in answer.response.variants] == ["Q4_K_M"]
    assert TOKEN in calls, "the supplied token was never offered to the Hub"


def test_a_cache_only_request_without_a_token_still_needs_the_grant(monkeypatch):
    calls: list = []
    _hub(monkeypatch, calls)

    with pytest.raises(HTTPException) as excinfo:
        _variants(None)

    assert excinfo.value.status_code == 404


def test_a_recorded_grant_still_answers_without_asking_the_hub(monkeypatch):
    calls: list = []
    _hub(monkeypatch, calls)

    asyncio.run(arun_as(ALICE, asyncio.to_thread(access.record_model_grant, REPO)))
    answer = _variants(None)

    assert [v.quant for v in answer.response.variants] == ["Q4_K_M"]


def test_an_unreachable_hub_does_not_turn_a_token_into_access(monkeypatch):
    def repo_info(repo, **kwargs):
        raise OSError("Hub unavailable")

    monkeypatch.setattr(access, "HfApi", lambda: SimpleNamespace(repo_info = repo_info))

    with pytest.raises(HTTPException) as excinfo:
        _variants(TOKEN)

    assert excinfo.value.status_code == 404
