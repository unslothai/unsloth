# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from auth import authentication, policy
from core.inference import llama_keepwarm, local_model_resolver
from hub.services.models import account_access as access
from routes import inference, models
from utils import openai_auto_switch_settings
from utils.account_context import OWNER, AccountContext, arun_as

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(access, "_resident_accounts", {})
    monkeypatch.setattr(inference, "_managed_catalogs", {})
    monkeypatch.setattr(inference, "_CATALOG_CACHE", {"at": 0.0, "models": []})
    monkeypatch.setattr(inference, "_ADVERTISED_CACHE", {"at": None, "paths": {}})
    # A first-time private repo is not public to an anonymous probe.
    monkeypatch.setattr(access, "repo_is_public", lambda *a, **k: False)
    yield


def _request():
    return SimpleNamespace(
        headers = {"X-Unsloth-HF-Token": "alice-token"},
        scope = {},
        state = SimpleNamespace(),
        url = SimpleNamespace(path = "/v1/chat/completions"),
    )


@pytest.fixture
def downloads(monkeypatch):
    """Uncached, unresolvable private repo; record whether auto-download is reached."""
    monkeypatch.setattr(openai_auto_switch_settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(
        openai_auto_switch_settings, "get_openai_auto_download_enabled", lambda: True
    )
    monkeypatch.setattr(openai_auto_switch_settings, "idle_unload_is_configured", lambda: False)

    # The Hub answers this private repo only for the caller's own token.
    class _Hub:
        def repo_info(
            self,
            repo_id,
            repo_type = "model",
            token = None,
            timeout = None,
        ):
            if token != "alice-token":
                raise RuntimeError("401")
            return SimpleNamespace(private = True, gated = False)

    monkeypatch.setattr(access, "HfApi", _Hub)
    monkeypatch.setattr(inference, "_loaded_identity_satisfies", lambda *a, **k: False)
    monkeypatch.setattr(inference, "_claim_slot_for_non_preview", lambda *a, **k: None)
    monkeypatch.setattr(
        local_model_resolver, "resolve_trusted_cached_local_gguf", lambda *a, **k: None
    )
    monkeypatch.setattr(local_model_resolver, "resolve_local_gguf", lambda *a, **k: None)
    monkeypatch.setattr(llama_keepwarm, "get_last_unloaded_model", lambda: None)
    monkeypatch.setattr(
        authentication, "request_admitted_without_credential", lambda *a, **k: False
    )
    monkeypatch.setattr(inference, "_classified_catalog", lambda rows: rows)
    monkeypatch.setattr(models, "collect_local_models", lambda path: [])
    seen = []

    async def record(requested_model, fastapi_request, **kwargs):
        seen.append((requested_model, inference._auto_download_hf_token(fastapi_request)))

    monkeypatch.setattr(inference, "_maybe_auto_download_model", record)
    return seen


def test_the_owner_reaches_auto_download_for_an_uncached_private_repo(downloads):
    asyncio.run(
        arun_as(OWNER, inference._maybe_auto_switch_model("org/private", _request(), "owner"))
    )
    assert downloads == [("org/private", "alice-token")]


def test_a_managed_caller_with_a_token_reaches_auto_download_for_an_uncached_private_repo(
    downloads,
):
    """The pre-check must not 404 an unresolved Hub reference before the caller's own
    token can authorize the download."""
    asyncio.run(
        arun_as(ALICE, inference._maybe_auto_switch_model("org/private", _request(), "alice"))
    )
    assert downloads == [("org/private", "alice-token")]


def test_a_managed_caller_still_cannot_reach_another_accounts_cached_private_model(
    monkeypatch, downloads, tmp_path
):
    """A cached foreign private model stays a 404 and never triggers a download."""
    monkeypatch.setattr(access, "model_visible", lambda *a, **k: False)
    monkeypatch.setattr(inference, "_own_local_model_for_alias", lambda alias: None)
    cached = tmp_path / "hub" / "models--org--private" / "snapshots" / "abc"
    cached.mkdir(parents = True)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            arun_as(ALICE, inference._maybe_auto_switch_model(str(cached), _request(), "alice"))
        )
    assert exc.value.status_code == 404
    assert downloads == []


def test_a_managed_caller_without_a_usable_token_is_still_refused(monkeypatch, downloads):
    """A wrong token proves nothing, so the refusal stands and no download starts."""
    request = _request()
    request.headers = {"X-Unsloth-HF-Token": "bob-token"}
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            arun_as(ALICE, inference._maybe_auto_switch_model("org/private", request, "alice"))
        )
    assert exc.value.status_code == 404
    assert downloads == []


def test_a_managed_caller_that_sends_no_token_is_still_refused(monkeypatch, downloads):
    request = _request()
    request.headers = {}
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            arun_as(ALICE, inference._maybe_auto_switch_model("org/private", request, "alice"))
        )
    assert exc.value.status_code == 404
    assert downloads == []
