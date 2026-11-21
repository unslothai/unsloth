# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which credential an export's Hub push may fall back to when the token box is empty."""

import pytest

from hub.utils.hf_tokens import AmbientAuthorizedToken
from routes import export as export_routes


@pytest.mark.parametrize("raw", [None, "", "  "])
def test_managed_account_push_without_a_token_never_gets_the_operator_login(monkeypatch, raw):
    monkeypatch.setattr(export_routes.account_access, "managed_account", lambda: True)

    assert export_routes._resolve_export_hf_token(raw, push_to_hub = True) is False


def test_managed_account_keeps_its_own_token_and_ambient_reads(monkeypatch):
    monkeypatch.setattr(export_routes.account_access, "managed_account", lambda: True)

    token = export_routes._resolve_export_hf_token("hf_own", push_to_hub = True)
    assert token == "hf_own" and isinstance(token, AmbientAuthorizedToken)
    assert export_routes._resolve_export_hf_token(None, push_to_hub = False) is None


def test_owner_push_without_a_token_may_use_the_cli_login(monkeypatch):
    monkeypatch.setattr(export_routes.account_access, "managed_account", lambda: False)

    assert export_routes._resolve_export_hf_token(None, push_to_hub = True) is None
