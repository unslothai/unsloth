# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account that names a model is never served by another account's
resident model under that name.

With auto-switch off (or an unknown name) the OpenAI routes fall through to
whatever is resident. For the owner that is the drop-in behaviour; for a managed
account it would run its prompt through a model it cannot even see on the
status routes. The switch helper therefore refuses at its exit unless the
resident model is the caller's own or answers to the requested name.
"""

import asyncio

import pytest
from fastapi import HTTPException

import routes.inference as inference
from hub.services.models import account_access


class _Request:
    def __init__(self):
        self.scope = {}
        self.state = type("State", (), {})()


@pytest.fixture
def no_switch(monkeypatch):
    """Auto-switch off, so the body falls through to the resident model."""
    import utils.openai_auto_switch_settings as settings

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: False)
    monkeypatch.setattr(account_access, "require_model_access", lambda requested: None)
    monkeypatch.setattr(inference, "_loaded_slot_ident", lambda: "/alice/private.gguf")
    monkeypatch.setattr(inference, "_claim_slot_for_non_preview", lambda request: None)


def _run(requested, *, managed, hidden, satisfies, monkeypatch):
    monkeypatch.setattr(account_access, "managed_account", lambda: managed)
    monkeypatch.setattr(account_access, "resident_hidden", lambda modality, reference = None: hidden)
    monkeypatch.setattr(inference, "_loaded_identity_satisfies", lambda requested: satisfies)
    return asyncio.run(inference._maybe_auto_switch_model(requested, _Request(), "bob"))


def test_managed_caller_naming_another_model_is_refused_when_the_resident_is_foreign(
    no_switch, monkeypatch
):
    with pytest.raises(HTTPException) as info:
        _run(
            "review/public-model",
            managed = True,
            hidden = True,
            satisfies = False,
            monkeypatch = monkeypatch,
        )
    assert info.value.status_code == 404
    assert info.value.detail == "Model not found"


def test_managed_caller_is_served_when_the_resident_answers_to_the_name(no_switch, monkeypatch):
    assert (
        _run(
            "review/public-model",
            managed = True,
            hidden = True,
            satisfies = True,
            monkeypatch = monkeypatch,
        )
        is None
    )


def test_managed_caller_is_served_by_its_own_resident(no_switch, monkeypatch):
    assert (
        _run("anything", managed = True, hidden = False, satisfies = False, monkeypatch = monkeypatch)
        is None
    )


def test_owner_keeps_the_fall_through(no_switch, monkeypatch):
    assert (
        _run(
            "review/public-model",
            managed = False,
            hidden = True,
            satisfies = False,
            monkeypatch = monkeypatch,
        )
        is None
    )


def test_omitted_model_against_a_foreign_resident_is_refused_as_before(no_switch, monkeypatch):
    with pytest.raises(HTTPException) as info:
        _run(None, managed = True, hidden = True, satisfies = False, monkeypatch = monkeypatch)
    assert info.value.status_code == 404


def test_a_raw_body_request_without_a_model_is_served_by_the_accounts_own_resident(
    no_switch, monkeypatch
):
    """The raw-body routes pass a reload-only sentinel for an omitted model; a
    managed account is not sent to look that sentinel up as a model name."""
    assert (
        _run(
            inference._RELOAD_ONLY_MODEL,
            managed = True,
            hidden = False,
            satisfies = False,
            monkeypatch = monkeypatch,
        )
        is None
    )
    with pytest.raises(HTTPException) as info:
        _run(
            inference._RELOAD_ONLY_MODEL,
            managed = True,
            hidden = True,
            satisfies = False,
            monkeypatch = monkeypatch,
        )
    assert info.value.status_code == 404
