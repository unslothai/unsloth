# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A connection the owner shared sends the owner's saved key, so a managed account may not pick where it
goes or which headers go with it, even when the "is a key saved" read fails."""

from models.providers import ProviderModelsRequest
from routes import providers as providers_route
from storage import providers_db

_SHARED = {
    "id": "shared",
    "provider_type": "custom",
    "display_name": "Team gateway",
    "base_url": "https://gateway.example/v1",
    "api_type": "chat_completions",
    "is_enabled": True,
    "extra_headers": {"x-team": "owner"},
    "shared_by_owner": True,
}


def _as_managed(monkeypatch, *, key_saved: bool):
    monkeypatch.setattr(providers_db, "get_own_provider", lambda _id: None)
    monkeypatch.setattr(providers_db, "get_provider", lambda _id: dict(_SHARED))
    monkeypatch.setattr(
        providers_route.credential_secrets, "provider_api_key_saved", lambda _id: key_saved
    )


def test_shared_connection_keeps_the_saved_endpoint_when_the_key_read_fails(monkeypatch):
    # provider_api_key_saved() answers False when the owner's store could not be read.
    _as_managed(monkeypatch, key_saved = False)
    payload = ProviderModelsRequest(
        provider_type = "custom", provider_id = "shared", base_url = "https://attacker.example/v1"
    )
    assert (
        providers_route._bind_saved_provider_target(payload).base_url
        == "https://gateway.example/v1"
    )


def test_shared_connection_sends_only_the_saved_headers(monkeypatch):
    _as_managed(monkeypatch, key_saved = True)
    payload = ProviderModelsRequest(
        provider_type = "custom",
        provider_id = "shared",
        base_url = "https://gateway.example/v1",
        extra_headers = {"x-portkey-custom-host": "https://attacker.example"},
    )
    bound = providers_route._bind_saved_provider_target(payload)
    assert providers_route._request_extra_headers(bound, via_api_key = False) == {"x-team": "owner"}


def test_own_connection_still_uses_the_edited_url_and_headers(monkeypatch):
    own = dict(_SHARED, id = "mine", shared_by_owner = False)
    monkeypatch.setattr(providers_db, "get_own_provider", lambda _id: dict(own))
    monkeypatch.setattr(providers_db, "get_provider", lambda _id: dict(own))
    monkeypatch.setattr(
        providers_route.credential_secrets, "provider_api_key_saved", lambda _id: False
    )
    payload = ProviderModelsRequest(
        provider_type = "custom",
        provider_id = "mine",
        base_url = "https://edited.example/v1",
        extra_headers = {"x-try": "1"},
    )
    bound = providers_route._bind_saved_provider_target(payload)
    assert bound.base_url == "https://edited.example/v1"
    assert providers_route._request_extra_headers(bound, via_api_key = False) == {"x-try": "1"}
