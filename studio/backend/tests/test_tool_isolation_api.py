# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
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

    async def recheck():
        return "actor-a"

    app.dependency_overrides[inference_route._isolation_auth_recheck] = lambda: recheck
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
    # Limited cannot enforce an allowlist, so the combination is refused at the edge
    # (422) instead of failing every tool call after the generation was spent.
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            messages = [], tool_network_policy = "allowlist", tool_execution_mode = "limited"
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

    def _unavailable(*, force: bool):
        return isolation.ToolIsolationCapability(
            **{
                **asdict(_capability()),
                "network_policies": ("deny", "allowlist"),
                "network_allowlist": ("pypi.org",),
            }
        )

    monkeypatch.setattr(inference_route, "tool_isolation_capability_snapshot", _unavailable)
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


@pytest.mark.parametrize("environment", ["linux", "colab"])
def test_diagnostic_and_disclosure_survive_http_response(monkeypatch, environment):
    from dataclasses import replace
    from core.inference.srt_diagnostics import limited_disclosure

    diagnostic = {"code": "dependency_missing", "stage": "dependency", "dependency": "bubblewrap"}
    snapshot = replace(
        _capability(),
        environment = environment,
        reason_code = "dependency_missing",
        diagnostic = diagnostic,
        limited_disclosure = limited_disclosure(environment),
    )
    calls = []

    def read(*, force):
        calls.append(force)
        return snapshot

    monkeypatch.setattr(inference_route, "tool_isolation_capability_snapshot", read)
    with _client(via_api_key = False) as client:
        response = client.get("/api/inference/tool-isolation/capability")
    assert response.status_code == 200
    body = response.json()
    assert body["diagnostic"] == diagnostic
    assert body["limited_disclosure"] == limited_disclosure(environment)
    assert body["available"] is False
    assert calls == [False]


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
    assert calls == [False]

    with _client(via_api_key = True) as client:
        response = client.get("/api/inference/tool-isolation/capability")
    assert response.status_code == 403
    # The refusal talks about this action, not about MCP servers.
    assert response.json()["detail"] == (
        "This action can only be performed from the Unsloth UI, not with an API key."
    )
    assert "MCP" not in response.json()["detail"]
    assert calls == [False]


def test_grant_endpoint_uses_fresh_cache_and_rejects_stale_generation(monkeypatch):
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
    assert calls == [False]


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


@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("success", 200),
        ("ineligible", 409),
        ("probe_failed", 409),
        ("changed", 409),
        ("api_key", 403),
    ],
)
def test_nested_consent_http_boundary(monkeypatch, scenario, expected):
    from dataclasses import replace
    from unittest.mock import Mock
    from core.inference import srt_nested, srt_probe
    from core.inference.srt_diagnostics import ProbeReason

    snapshot = replace(
        _capability(),
        nested_eligible = scenario != "ineligible",
        nested_profile_id = srt_nested.NESTED_PROFILE,
    )
    snapshots = [
        snapshot,
        replace(snapshot, probe_generation = "changed") if scenario == "changed" else snapshot,
    ]
    read = Mock(side_effect = snapshots)
    monkeypatch.setattr(inference_route, "tool_isolation_capability_snapshot", read)
    probe = Mock(return_value = (scenario != "probe_failed", ProbeReason("probe_timeout")))
    monkeypatch.setattr(srt_probe, "probe", probe)
    store = srt_nested.NestedGrantStore()
    issue = Mock(wraps = store.issue)
    monkeypatch.setattr(store, "issue", issue)
    monkeypatch.setattr(srt_nested, "NESTED_GRANTS", store)
    with _client(via_api_key = scenario == "api_key") as client:
        response = client.post(
            "/api/inference/tool-isolation/nested-grant",
            json = {"ui_session_id": "page-a", "probe_generation": "probe-1"},
        )
    assert response.status_code == expected, response.text
    if scenario == "success":
        store.validate(
            response.json()["grant"],
            current_subject = "actor-a",
            tool_ui_session_id = "page-a",
            probe_generation = "probe-1",
        )
        assert read.call_count == 2
        assert probe.call_args.kwargs == {"force": True, "isolation_variant": "nested"}
    else:
        issue.assert_not_called()
    if scenario in ("ineligible", "api_key"):
        probe.assert_not_called()
    if scenario == "probe_failed":
        assert response.json()["detail"]["diagnostic"]["code"] == "probe_timeout"


def test_limited_timeout_never_issues_grant(monkeypatch):
    from dataclasses import replace

    monkeypatch.setattr(
        inference_route,
        "tool_isolation_capability_snapshot",
        lambda **kw: replace(_capability(), reason_code = "probe_timeout"),
    )
    monkeypatch.setattr(
        inference_route, "issue_limited_grant", lambda **kw: pytest.fail("timeout issued a grant")
    )
    with _client(via_api_key = False) as client:
        response = client.post(
            "/api/inference/tool-isolation/limited-grant",
            json = {"ui_session_id": "page", "probe_generation": "probe-1"},
        )
    assert response.status_code == 503


@pytest.mark.parametrize("status", [401, 499])
def test_limited_revalidates_auth_and_disconnect_after_check(monkeypatch, status):
    from fastapi import HTTPException

    monkeypatch.setattr(
        inference_route, "tool_isolation_capability_snapshot", lambda **kw: _capability()
    )
    monkeypatch.setattr(
        inference_route,
        "issue_limited_grant",
        lambda **kw: pytest.fail("revoked request issued a grant"),
    )

    async def revoked():
        raise HTTPException(status_code = status, detail = "Request no longer valid")

    with _client(via_api_key = False) as client:
        client.app.dependency_overrides[inference_route._isolation_auth_recheck] = lambda: revoked
        response = client.post(
            "/api/inference/tool-isolation/limited-grant",
            json = {"ui_session_id": "page", "probe_generation": "probe-1"},
        )
    assert response.status_code == status


def test_explicit_capability_recheck_forces_probe(monkeypatch):
    calls = []
    monkeypatch.setattr(
        inference_route,
        "tool_isolation_capability_snapshot",
        lambda **kw: calls.append(kw["force"]) or _capability(),
    )
    with _client(via_api_key = False) as client:
        assert (
            client.get("/api/inference/tool-isolation/capability?refresh=true").status_code == 200
        )
    assert calls == [True]


@pytest.mark.parametrize(
    "via_api_key,host,body,status",
    [
        (True, "127.0.0.1", {"confirm": True}, 403),
        (False, "203.0.113.1", {"confirm": True}, 403),
        (False, "127.0.0.1", {"confirm": False}, 400),
        (False, "127.0.0.1", {"confirm": 1}, 400),
        (False, "127.0.0.1", {"confirm": True, "command": "bad"}, 400),
        (False, "127.0.0.1", {"confirm": True}, 200),
    ],
)
def test_windows_setup_requires_local_explicit_ui_consent(
    monkeypatch, via_api_key, host, body, status
):
    from core.inference import srt_setup

    calls = []
    monkeypatch.setattr(
        srt_setup,
        "install_windows_sandbox",
        lambda **kw: calls.append(kw) or {"status": "installed", "message": "fixture"},
    )
    client = _client(via_api_key = via_api_key)
    with TestClient(client.app, client = (host, 1234)) as http:
        response = http.post("/api/inference/tool-isolation/windows-setup", json = body)
    assert response.status_code == status
    assert len(calls) == int(status == 200)


def test_configuration_change_during_final_auth_wait_rejects_grant(monkeypatch):
    from core.inference import os_sandbox

    identity = ["before"]
    monkeypatch.setattr(os_sandbox, "_runtime_identity", lambda: identity[0])
    monkeypatch.setattr(
        inference_route, "tool_isolation_capability_snapshot", lambda **kw: _capability()
    )
    monkeypatch.setattr(
        inference_route,
        "issue_limited_grant",
        lambda **kw: pytest.fail("stale configuration issued grant"),
    )

    async def recheck():
        identity[0] = "after"
        return "actor-a"

    with _client(via_api_key = False) as client:
        client.app.dependency_overrides[inference_route._isolation_auth_recheck] = lambda: recheck
        response = client.post(
            "/api/inference/tool-isolation/limited-grant",
            json = {"ui_session_id": "page", "probe_generation": "probe-1"},
        )
    assert response.status_code == 409
