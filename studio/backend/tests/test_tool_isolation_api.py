# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.inference import tool_isolation as isolation
from models.inference import (
    AnthropicMessagesRequest,
    ChatCompletionRequest,
    ChatCountTokensRequest,
    ResponsesRequest,
)
from routes import inference as inference_route


def _capability(
    *,
    generation: str = "probe-1",
    qualified: bool = False,
    available: bool | None = None,
):
    if available is None:
        available = qualified
    return isolation.ToolIsolationCapability(
        environment = "wsl2",
        backend = "bubblewrap",
        protection_state = "preview" if qualified else "unavailable",
        profile_id = "bubblewrap-v1",
        probe_generation = generation,
        environment_fingerprint = "fingerprint-1",
        reason = "live probe did not qualify" if not qualified else "",
        remediation = "Use Limited mode for this session" if not qualified else "",
        retryable = True,
        qualified = qualified,
        available = available,
    )


def _client(*, via_api_key: bool) -> TestClient:
    app = FastAPI()
    app.include_router(inference_route.studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "actor-a"
    app.dependency_overrides[authenticated_via_api_key] = lambda: via_api_key
    return TestClient(app)


@pytest.mark.parametrize(
    "payload",
    [
        ChatCompletionRequest(messages = []),
        ResponsesRequest(input = "hello"),
        AnthropicMessagesRequest(messages = []),
    ],
)
def test_request_families_default_to_required_os_isolation(payload):
    assert payload.tool_execution_mode == "os_isolation_required"
    assert payload.limited_grant is None
    assert payload.tool_ui_session_id is None
    assert payload.bypass_permissions is False


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ChatCompletionRequest(messages = [], tool_execution_mode = None),
        lambda: ResponsesRequest(input = "hello", tool_execution_mode = None),
        lambda: AnthropicMessagesRequest(messages = [], tool_execution_mode = None),
    ],
)
def test_explicit_null_execution_mode_normalizes_to_required(factory):
    assert factory().tool_execution_mode == "os_isolation_required"


def test_token_count_requests_declare_and_normalize_the_execution_mode():
    assert ChatCountTokensRequest(messages = []).tool_execution_mode == "os_isolation_required"
    assert (
        ChatCountTokensRequest(messages = [], tool_execution_mode = "limited").tool_execution_mode
        == "limited"
    )
    full = ChatCountTokensRequest(messages = [], tool_execution_mode = "full")
    assert full.permission_mode == "full"
    assert full.bypass_permissions is True


@pytest.mark.parametrize(
    "factory",
    [
        lambda **values: ChatCompletionRequest(messages = [], **values),
        lambda **values: ResponsesRequest(input = "hello", **values),
        lambda **values: AnthropicMessagesRequest(messages = [], **values),
    ],
)
@pytest.mark.parametrize("legacy", ["permission_mode", "bypass_permissions"])
def test_legacy_full_permissions_normalize_to_full_execution(factory, legacy):
    values = {legacy: "full" if legacy == "permission_mode" else True}
    request = factory(**values)
    assert request.tool_execution_mode == "full"
    assert request.permission_mode == "full"
    assert request.bypass_permissions is True


@pytest.mark.parametrize(
    "factory",
    [
        lambda **values: ChatCompletionRequest(messages = [], **values),
        lambda **values: ResponsesRequest(input = "hello", **values),
        lambda **values: AnthropicMessagesRequest(messages = [], **values),
    ],
)
def test_limited_mode_does_not_bypass_approval_or_permissions(factory):
    request = factory(
        tool_execution_mode = "limited",
        limited_grant = "opaque",
        tool_ui_session_id = "page-a",
    )
    assert request.tool_execution_mode == "limited"
    assert request.bypass_permissions is False
    assert request.permission_mode is None


def test_responses_translation_preserves_isolation_fields():
    payload = ResponsesRequest(
        input = "hello",
        tool_execution_mode = "limited",
        limited_grant = "opaque",
        tool_ui_session_id = "page-a",
        permission_mode = "ask",
    )
    translated = inference_route._build_chat_request(payload, [], False)
    assert translated.tool_execution_mode == "limited"
    assert translated.limited_grant == "opaque"
    assert translated.tool_ui_session_id == "page-a"
    assert translated.permission_mode == "ask"
    assert translated.bypass_permissions is False
    assert translated.tool_network_policy == "deny"


@pytest.mark.parametrize(
    "factory",
    [
        lambda **values: ChatCompletionRequest(messages = [], **values),
        lambda **values: ChatCountTokensRequest(messages = [], **values),
        lambda **values: ResponsesRequest(input = "hello", **values),
        lambda **values: AnthropicMessagesRequest(messages = [], **values),
    ],
)
def test_network_policy_defaults_to_deny_and_rejects_unknown_values(factory):
    # A client that predates the field, or sends an explicit null, gets no network.
    assert factory().tool_network_policy == "deny"
    assert factory(tool_network_policy = None).tool_network_policy == "deny"
    assert factory(tool_network_policy = "allowlist").tool_network_policy == "allowlist"
    with pytest.raises(ValueError):
        factory(tool_network_policy = "open")
    # The policy is a separate axis: it never widens the execution mode or permissions.
    request = factory(tool_network_policy = "allowlist")
    assert request.tool_execution_mode == "os_isolation_required"
    # The token-count request leaves an unset bypass as None; the rest default to False.
    assert not request.bypass_permissions


def test_responses_translation_preserves_network_policy():
    payload = ResponsesRequest(input = "hello", tool_network_policy = "allowlist")
    translated = inference_route._build_chat_request(payload, [], False)
    assert translated.tool_network_policy == "allowlist"


def test_requested_network_allowlist_is_gated_on_mode_and_capability(monkeypatch):
    calls: list[bool] = []

    def _snapshot(*, force: bool):
        calls.append(force)
        capability = _capability(qualified = True)
        return isolation.ToolIsolationCapability(
            **{
                **asdict(capability),
                "network_policies": ("deny", "allowlist"),
                "network_allowlist": ("pypi.org", "huggingface.co"),
            }
        )

    monkeypatch.setattr(inference_route, "tool_isolation_capability_snapshot", _snapshot)
    allow = ChatCompletionRequest(messages = [], tool_network_policy = "allowlist")
    assert inference_route._requested_network_allowlist(allow) == ["pypi.org", "huggingface.co"]
    # The description helper never forces a live probe.
    assert calls == [False]
    assert inference_route._requested_network_allowlist(ChatCompletionRequest(messages = [])) is None
    assert (
        inference_route._requested_network_allowlist(
            ChatCompletionRequest(
                messages = [], tool_network_policy = "allowlist", tool_execution_mode = "limited"
            )
        )
        is None
    )
    assert (
        inference_route._requested_network_allowlist(
            ChatCompletionRequest(
                messages = [], tool_network_policy = "allowlist", bypass_permissions = True
            )
        )
        is None
    )

    def _deny_only(*, force: bool):
        return _capability(qualified = True)

    monkeypatch.setattr(inference_route, "tool_isolation_capability_snapshot", _deny_only)
    assert inference_route._requested_network_allowlist(allow) is None

    def _broken(*, force: bool):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(inference_route, "tool_isolation_capability_snapshot", _broken)
    assert inference_route._requested_network_allowlist(allow) is None


def test_capability_shape_tolerates_backends_without_the_new_fields():
    # A backend snapshot that predates the network proxy and the restricted token.
    legacy = {
        "environment": "native_linux",
        "backend": "linux-bubblewrap",
        "protection_state": "protected",
        "profile_id": "linux-bubblewrap-v2",
        "probe_generation": "probe-1",
        "environment_fingerprint": "fp",
        "reason": "",
        "remediation": "",
        "retryable": False,
        "qualified": True,
        "available": True,
        "limitations": (),
    }
    shaped = isolation._shape_capability(legacy)
    assert shaped.network_policies == ("deny",)
    assert shaped.network_allowlist == ()
    assert shaped.limited_backend is None
    assert shaped.limited_profile_id is None
    assert shaped.limited_limitations == ()
    # And one that publishes them, as a plain mapping or as attributes.
    enriched = {
        **legacy,
        "network_policies": ["deny", "allowlist"],
        "network_allowlist": ["pypi.org"],
        "limited_backend": "windows-restricted-token",
        "limited_profile_id": "windows-restricted-token-write-isolation-v1",
        "limited_limitations": ["user_profile_readable"],
    }
    shaped = isolation._shape_capability(enriched)
    assert shaped.network_policies == ("deny", "allowlist")
    assert shaped.network_allowlist == ("pypi.org",)
    assert shaped.limited_backend == "windows-restricted-token"
    assert shaped.limited_profile_id == "windows-restricted-token-write-isolation-v1"
    assert shaped.limited_limitations == ("user_profile_readable",)
    assert isolation._shape_capability(SimpleNamespace(**enriched)) == shaped


def test_store_binds_grant_to_actor_ui_session_generation_and_mode():
    store = isolation.LimitedGrantStore(ttl_seconds = 60, max_entries = 8)
    issued = store.issue(
        current_subject = "actor-a",
        tool_ui_session_id = "page-a",
        probe_generation = "probe-1",
    )

    validated = store.validate(
        issued.token,
        current_subject = "actor-a",
        tool_ui_session_id = "page-a",
        probe_generation = "probe-1",
        requested_mode = "limited",
    )
    assert validated.current_subject == "actor-a"
    assert validated.tool_ui_session_id == "page-a"
    assert validated.probe_generation == "probe-1"

    mismatches = [
        dict(current_subject = "actor-b", tool_ui_session_id = "page-a", probe_generation = "probe-1"),
        dict(current_subject = "actor-a", tool_ui_session_id = "page-b", probe_generation = "probe-1"),
        dict(current_subject = "actor-a", tool_ui_session_id = "page-a", probe_generation = "probe-2"),
    ]
    for values in mismatches:
        with pytest.raises(isolation.LimitedGrantError):
            store.validate(issued.token, requested_mode = "limited", **values)

    with pytest.raises(isolation.LimitedGrantError, match = "only Limited mode"):
        store.validate(
            issued.token,
            current_subject = "actor-a",
            tool_ui_session_id = "page-a",
            probe_generation = "probe-1",
            requested_mode = "full",
        )


def test_forged_and_expired_grants_fail_without_leaking_token(monkeypatch):
    clock = {"monotonic": 10.0, "wall": 1_000.0}
    monkeypatch.setattr(isolation.time, "monotonic", lambda: clock["monotonic"])
    monkeypatch.setattr(isolation.time, "time", lambda: clock["wall"])
    store = isolation.LimitedGrantStore(ttl_seconds = 1, max_entries = 8)
    issued = store.issue(
        current_subject = "actor-a",
        tool_ui_session_id = "page-a",
        probe_generation = "probe-1",
    )

    forged = f"{issued.token}forged"
    with pytest.raises(isolation.LimitedGrantError) as forged_error:
        store.validate(
            forged,
            current_subject = "actor-a",
            tool_ui_session_id = "page-a",
            probe_generation = "probe-1",
            requested_mode = "limited",
        )
    assert forged not in str(forged_error.value)
    assert issued.token not in str(forged_error.value)

    clock["monotonic"] = 12.0
    with pytest.raises(isolation.LimitedGrantError) as expired_error:
        store.validate(
            issued.token,
            current_subject = "actor-a",
            tool_ui_session_id = "page-a",
            probe_generation = "probe-1",
            requested_mode = "limited",
        )
    assert expired_error.value.code == "EXPIRED_LIMITED_GRANT"


def test_store_uses_constant_time_comparison_and_bounded_cleanup(monkeypatch):
    comparisons: list[tuple[bytes, bytes]] = []
    original = isolation.hmac.compare_digest

    def _compare(left: bytes, right: bytes) -> bool:
        comparisons.append((left, right))
        return original(left, right)

    monkeypatch.setattr(isolation.hmac, "compare_digest", _compare)
    store = isolation.LimitedGrantStore(ttl_seconds = 60, max_entries = 2)
    first = store.issue(
        current_subject = "actor-a", tool_ui_session_id = "page-a", probe_generation = "probe-1"
    )
    second = store.issue(
        current_subject = "actor-a", tool_ui_session_id = "page-a", probe_generation = "probe-1"
    )
    third = store.issue(
        current_subject = "actor-a", tool_ui_session_id = "page-a", probe_generation = "probe-1"
    )
    assert len(store._records) == 2
    with pytest.raises(isolation.LimitedGrantError):
        store.validate(
            first.token,
            current_subject = "actor-a",
            tool_ui_session_id = "page-a",
            probe_generation = "probe-1",
            requested_mode = "limited",
        )
    store.validate(
        second.token,
        current_subject = "actor-a",
        tool_ui_session_id = "page-a",
        probe_generation = "probe-1",
        requested_mode = "limited",
    )
    store.validate(
        third.token,
        current_subject = "actor-a",
        tool_ui_session_id = "page-a",
        probe_generation = "probe-1",
        requested_mode = "limited",
    )
    assert len(comparisons) == 3


def test_capability_endpoint_is_ui_only_and_advisory(monkeypatch):
    calls: list[bool] = []

    def _snapshot(*, force: bool):
        calls.append(force)
        return _capability()

    monkeypatch.setattr(inference_route, "tool_isolation_capability_snapshot", _snapshot)
    with _client(via_api_key = False) as client:
        response = client.get("/api/inference/tool-isolation/capability")
    assert response.status_code == 200
    expected = asdict(_capability())
    expected["limitations"] = []
    expected["network_policies"] = list(expected["network_policies"])
    expected["network_allowlist"] = list(expected["network_allowlist"])
    expected["limited_limitations"] = list(expected.get("limited_limitations") or [])
    assert response.json() == expected
    assert response.json()["network_policies"] == ["deny"]
    assert calls == [True]

    with _client(via_api_key = True) as client:
        response = client.get("/api/inference/tool-isolation/capability")
    assert response.status_code == 403
    # The refusal talks about this action, not about MCP servers.
    assert response.json()["detail"] == (
        "This action can only be performed from the Unsloth UI, not with an API key."
    )
    assert "MCP" not in response.json()["detail"]
    assert calls == [True]


def test_grant_endpoint_reprobes_and_rejects_stale_generation(monkeypatch):
    calls: list[bool] = []

    def _snapshot(*, force: bool):
        calls.append(force)
        return _capability(generation = "probe-2")

    monkeypatch.setattr(inference_route, "tool_isolation_capability_snapshot", _snapshot)
    with _client(via_api_key = False) as client:
        response = client.post(
            "/api/inference/tool-isolation/limited-grant",
            json = {"ui_session_id": "page-a", "probe_generation": "probe-1"},
        )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "CAPABILITY_CHANGED"
    assert calls == [True]


def test_grant_endpoint_does_not_downgrade_an_available_preview_backend(monkeypatch):
    monkeypatch.setattr(
        inference_route,
        "tool_isolation_capability_snapshot",
        lambda *, force: _capability(qualified = False, available = True),
    )

    with _client(via_api_key = False) as client:
        response = client.post(
            "/api/inference/tool-isolation/limited-grant",
            json = {"ui_session_id": "page-a", "probe_generation": "probe-1"},
        )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "OS_ISOLATION_AVAILABLE"


def test_grant_endpoint_issues_opaque_session_grant_and_is_ui_only(monkeypatch):
    monkeypatch.setattr(
        inference_route, "tool_isolation_capability_snapshot", lambda *, force: _capability()
    )
    with _client(via_api_key = False) as client:
        response = client.post(
            "/api/inference/tool-isolation/limited-grant",
            json = {"ui_session_id": "page-a", "probe_generation": "probe-1"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["probe_generation"] == "probe-1"
    assert body["grant"]
    assert body["expires_at"].endswith("+00:00")

    validated = isolation.validate_limited_grant(
        body["grant"],
        current_subject = "actor-a",
        tool_ui_session_id = "page-a",
        probe_generation = "probe-1",
        requested_mode = "limited",
    )
    assert validated.probe_generation == "probe-1"

    with _client(via_api_key = True) as client:
        forbidden = client.post(
            "/api/inference/tool-isolation/limited-grant",
            json = {"ui_session_id": "page-a", "probe_generation": "probe-1"},
        )
    assert forbidden.status_code == 403
    assert "MCP" not in forbidden.json()["detail"]
