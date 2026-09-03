# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seams between the external-provider route and the shared Studio tool loop.

Both are one-line policy decisions in ``_proxy_to_external_provider`` that no
loop test can reach: the tool-call budget it hands the loop, and whether a
durable Deep Research hop may use the saved connection its run was created with.
"""

import ast
import pathlib

import pytest

from routes.inference import _request_is_internal_workflow


_ROUTE_SOURCE = pathlib.Path(__file__).resolve().parents[1] / "routes" / "inference.py"


class _Headers:
    def __init__(self, authorization = None):
        self._value = authorization

    def get(self, name):
        return self._value if name.lower() == "authorization" else None


class _Request:
    def __init__(self, authorization = None):
        self.headers = _Headers(authorization)


def test_a_zero_tool_call_budget_is_not_rewritten_to_the_default():
    """0 documents "disabled"; ``or 25`` turned it into a 25-call budget."""
    tree = ast.parse(_ROUTE_SOURCE.read_text(encoding = "utf-8"))
    budgets = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword)
        and node.arg == "max_calls"
        and isinstance(node.value, ast.BoolOp)
    ]
    assert budgets == [], "max_calls must use an `is not None` fallback, not `or`"


def test_a_session_request_may_use_a_saved_connection():
    assert _request_is_internal_workflow(_Request(None)) is False


@pytest.mark.parametrize(
    "authorization",
    ["Bearer not-an-unsloth-key", "Basic sk-unsloth-abc", "", "Bearer"],
)
def test_a_bearer_that_is_not_an_unsloth_key_is_never_internal(authorization):
    assert _request_is_internal_workflow(_Request(authorization)) is False


def test_an_sk_unsloth_bearer_is_internal_only_when_storage_says_so(monkeypatch):
    """The prefix is public, so a caller could send one; storage decides."""
    from auth.authentication import API_KEY_PREFIX
    from routes import inference as route_mod

    token = f"{API_KEY_PREFIX}deadbeef"
    request = _Request(f"Bearer {token}")

    monkeypatch.setattr(route_mod.auth_storage, "is_internal_api_key", lambda raw: False)
    assert _request_is_internal_workflow(request) is False

    monkeypatch.setattr(route_mod.auth_storage, "is_internal_api_key", lambda raw: raw == token)
    assert _request_is_internal_workflow(request) is True


def test_a_failing_storage_probe_withholds_saved_credentials(monkeypatch):
    from auth.authentication import API_KEY_PREFIX
    from routes import inference as route_mod

    def _boom(raw):
        raise RuntimeError("db is gone")

    monkeypatch.setattr(route_mod.auth_storage, "is_internal_api_key", _boom)
    assert _request_is_internal_workflow(_Request(f"Bearer {API_KEY_PREFIX}x")) is False
