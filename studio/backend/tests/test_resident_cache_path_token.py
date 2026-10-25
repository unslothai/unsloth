# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A caller's own Hub token unlocks a resident loaded from a Hub cache path, as for a repo id."""

import asyncio
from types import SimpleNamespace

import pytest

from .test_account_inference_isolation import isolated, ALICE  # noqa: F401 - autouse fixture
from utils.account_context import arun_as
from hub.dependencies import HUB_HF_TOKEN_HEADER
from hub.services.models import account_access as access
from routes import inference


def _setup(
    monkeypatch,
    tmp_path,
    resident_kind,
    auto_switch,
    in_cache = True,
):
    from utils import hf_cache_settings, openai_auto_switch_settings as settings
    from core.inference import local_model_resolver

    cache = tmp_path / "hub"
    cache.mkdir()
    # Outside the cache: a folder an account could name like a cache entry.
    root = cache if in_cache else tmp_path / "account-models"
    snapshot = root / "models--org--private" / "snapshots" / "revision"
    snapshot.mkdir(parents = True)
    gguf = snapshot / "weights.gguf"
    gguf.touch()
    resident = {"repo_id": "org/private", "snapshot": str(snapshot), "gguf": str(gguf)}[
        resident_kind
    ]
    is_gguf = resident_kind == "gguf"
    backend = SimpleNamespace(
        active_model_name = None if is_gguf else resident, _openai_advertised_id = "org/private"
    )
    llama = SimpleNamespace(
        is_loaded = is_gguf, model_identifier = resident, _openai_advertised_id = "org/private"
    )
    monkeypatch.setattr(inference, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(inference, "_peek_inference_backend", lambda: backend)
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(inference, "_resident_npu_model", lambda: None)
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [cache])
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: auto_switch)
    monkeypatch.setattr(settings, "get_openai_auto_download_enabled", lambda: True)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: False)
    monkeypatch.setattr(local_model_resolver, "warm_index_soon", lambda: None)
    monkeypatch.setattr(inference, "_claim_slot_for_non_preview", lambda request: None)

    async def catalog():
        return []

    monkeypatch.setattr(inference, "_cached_local_catalog", catalog)
    probes = []

    class Hub:
        def repo_info(self, repo_id, *, repo_type, token, timeout):
            assert repo_id == "org/private" and token == "caller-owned-token"
            probes.append(repo_id)
            return SimpleNamespace(gated = True)

        def auth_check(self, repo_id, *, repo_type, token):
            assert repo_id == "org/private" and token == "caller-owned-token"

    monkeypatch.setattr(access, "HfApi", Hub)
    request = SimpleNamespace(
        scope = {},
        state = SimpleNamespace(),
        headers = {HUB_HF_TOKEN_HEADER: "caller-owned-token"},
        url = SimpleNamespace(path = "/v1/chat/completions"),
    )
    return request, probes


@pytest.mark.parametrize("resident_kind", ["repo_id", "snapshot", "gguf"])
@pytest.mark.parametrize("auto_switch", [False, True])
def test_authorized_hub_token_can_use_the_matching_resident(
    monkeypatch, tmp_path, resident_kind, auto_switch
):
    request, probes = _setup(monkeypatch, tmp_path, resident_kind, auto_switch)
    assert inference._loaded_identity_satisfies("org/private")
    for _ in range(2):
        asyncio.run(
            arun_as(ALICE, inference._maybe_auto_switch_model("org/private", request, "alice"))
        )
    assert probes


@pytest.mark.parametrize("resident_kind", ["snapshot", "gguf"])
def test_a_cache_named_folder_outside_the_hub_cache_is_still_refused(
    monkeypatch, tmp_path, resident_kind
):
    from fastapi import HTTPException

    request, _ = _setup(monkeypatch, tmp_path, resident_kind, False, in_cache = False)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(
            arun_as(ALICE, inference._maybe_auto_switch_model("org/private", request, "alice"))
        )
    assert refused.value.status_code == 404
