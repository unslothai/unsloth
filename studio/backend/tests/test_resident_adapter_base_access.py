# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chatting with, or joining, another account's resident LoRA must clear the adapter's base too:
the base is the model that answers."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import routes.inference as routes


def _resident(monkeypatch, base):
    backend = SimpleNamespace(
        active_model_name = "public/adapter",
        models = {"public/adapter": {"is_lora": True, "base_model": base}},
    )
    monkeypatch.setattr(routes, "_peek_inference_backend", lambda: backend)


def _deny_private(reference, *args, **kwargs):
    if reference.startswith("acme/private"):
        raise HTTPException(status_code = 404, detail = "Model not found")


def test_a_refused_base_refuses_the_resident_adapter(monkeypatch):
    _resident(monkeypatch, "acme/private-base")
    monkeypatch.setattr(routes.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(routes.account_access, "require_model_access", _deny_private)
    with pytest.raises(HTTPException) as err:
        routes._require_loaded_base_access()
    assert err.value.status_code == 404


def test_reachable_bases_and_the_owner_pass(monkeypatch):
    monkeypatch.setattr(routes.account_access, "require_model_access", _deny_private)
    monkeypatch.setattr(routes.account_access, "managed_account", lambda: True)
    _resident(monkeypatch, "unsloth/Llama-3.2-1B-Instruct")
    routes._require_loaded_base_access()
    _resident(monkeypatch, None)
    routes._require_loaded_base_access()
    monkeypatch.setattr(routes.account_access, "managed_account", lambda: False)
    _resident(monkeypatch, "acme/private-base")
    routes._require_loaded_base_access()


def test_both_reuse_paths_check_the_base():
    source = inspect.getsource(routes)
    shortcut = source.split("async def _refuse_foreign_resident", 1)[1].split(
        "\n    async def ", 1
    )[0]
    assert "_require_loaded_base_access" in shortcut
    joined = source.split('account_access.join_resident("chat")')
    assert any("_require_loaded_base_access" in before[-300:] for before in joined[:-1])


def test_the_loaded_entry_records_an_adapters_base():
    from core.inference import orchestrator
    assert '["base_model"]' in inspect.getsource(orchestrator.InferenceOrchestrator.load_model)
