# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Gate on permission_mode not confirm_tool_calls; internal alone cannot spend every saved credential."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from fastapi import HTTPException


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class FakeExternalClient:
    last: dict = {}

    def __init__(self, **kwargs):
        FakeExternalClient.last = {"ctor": kwargs, "passthrough": None}
        self.provider_type = kwargs.get("provider_type")

    def stream_chat_completion(self, **kwargs):
        FakeExternalClient.last["passthrough"] = kwargs

        async def gen():
            yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield "data: [DONE]\n\n"

        return gen()

    async def close(self):
        return None


def _request(authorization = None):
    async def is_disconnected():
        return False

    headers = {"X-Unsloth-Events": "1"}
    if authorization:
        headers["authorization"] = authorization
    return SimpleNamespace(
        # the confirm gate can only ask over these frames
        headers = headers,
        state = SimpleNamespace(skip_api_monitor = True),
        is_disconnected = is_disconnected,
    )


@pytest.fixture(autouse = True)
def _clean_policy():
    from state.tool_policy import reset_tool_policy

    reset_tool_policy()
    yield
    reset_tool_policy()


def _mint_internal(monkeypatch, inf, name):
    """Present a key the store reports as internal and minted under *name*."""
    monkeypatch.setattr(inf.auth_storage, "is_internal_api_key", lambda raw: True)
    if isinstance(name, Exception) or callable(name):
        monkeypatch.setattr(inf.auth_storage, "internal_api_key_name", name)
    else:
        monkeypatch.setattr(inf.auth_storage, "internal_api_key_name", lambda raw: name)


def _install(
    monkeypatch,
    provider_type = "openai",
    record_keys = None,
):
    from core.inference.providers import get_base_url
    from routes import inference as inf

    monkeypatch.setattr(
        inf.providers_db,
        "get_provider",
        lambda _pid: {
            "id": _pid,
            "provider_type": provider_type,
            "base_url": get_base_url(provider_type) or "http://127.0.0.1:8080/v1",
            "display_name": "Saved connection",
            "is_enabled": True,
        },
    )

    def _resolve(
        provider_id,
        encrypted,
        *,
        allow_saved_key = True,
    ):
        if record_keys is not None:
            record_keys.append(allow_saved_key)
        return "k"

    monkeypatch.setattr(inf, "resolve_provider_api_key_or_400", _resolve)
    monkeypatch.setattr(inf, "ExternalProviderClient", FakeExternalClient)
    return inf


def _payload(**overrides):
    from models.inference import ChatCompletionRequest

    base = dict(
        messages = [{"role": "user", "content": "what is 2+2?"}],
        provider_id = "saved-1",
        external_model = "gpt-5.4",
        stream = True,
    )
    base.update(overrides)
    return ChatCompletionRequest(**base)


def _run(
    inf,
    payload,
    request = None,
):
    async def go():
        resp = await inf._proxy_to_external_provider(
            payload, request or _request(), current_subject = "t"
        )
        return [chunk async for chunk in resp.body_iterator]

    return _drive(go())


def test_non_streaming_ask_mode_is_rejected_like_the_local_routes(monkeypatch):
    """A non-streaming request with permission_mode ask must 400, as the local routes do."""
    inf = _install(monkeypatch, "openai")

    payload = _payload(stream = False, enable_tools = True, permission_mode = "ask")
    assert payload.confirm_tool_calls is None, "the flag is omitted; only the mode asks"
    assert inf._confirm_gate_needs_stream(payload) is True

    with pytest.raises(HTTPException) as excinfo:
        _drive(inf._proxy_to_external_provider(payload, _request(), current_subject = "t"))
    assert excinfo.value.status_code == 400


def test_an_explicit_confirm_flag_still_401s_the_streaming_hosted_only_request(monkeypatch):
    """An explicit confirm_tool_calls still 401s hosted-only streams, which never enter the loop."""
    inf = _install(monkeypatch, "openai")
    payload = _payload(
        enable_tools = True,
        enabled_tools = ["web_search"],
        confirm_tool_calls = True,
    )
    with pytest.raises(HTTPException) as excinfo:
        _run(inf, payload)
    assert excinfo.value.status_code == 400


def _capture_loop(monkeypatch, inf):
    """Replace the shared loop with a stub that records the ToolLoopRun it got."""
    entered: dict = {}

    def _loop(*a, **k):
        entered.update(k)

        async def gen():
            yield "data: [DONE]\n\n"

        return gen()

    monkeypatch.setattr(inf, "stream_with_studio_tools", _loop)
    return entered


def test_a_plain_streaming_ask_request_still_reaches_the_loop(monkeypatch):
    """The gate CAN prompt over SSE, so streaming ask must not be rejected."""
    inf = _install(monkeypatch, "openai")
    entered = _capture_loop(monkeypatch, inf)
    _run(inf, _payload(enable_tools = True, permission_mode = "ask"))
    assert entered["policy"].confirm_calls is True


_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4"
    "z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="
)


def test_an_mcp_image_is_refused_when_the_tool_loop_will_not_run(monkeypatch):
    """The direct proxy forwards no MCP image, so accepting one would drop it silently."""
    inf = _install(monkeypatch, "openai")
    monkeypatch.setattr(inf, "provider_model_runs_local_tools", lambda *a: False)
    payload = _payload(enable_tools = True, mcp_enabled = True, mcp_image = _PNG)
    with pytest.raises(HTTPException) as excinfo:
        _run(inf, payload)
    assert excinfo.value.status_code == 400 and "MCP" in excinfo.value.detail


def test_an_mcp_image_is_refused_when_the_catalog_has_no_mapped_tool(monkeypatch):
    from core.inference import tools as tools_mod

    inf = _install(monkeypatch, "openai")
    entered = _capture_loop(monkeypatch, inf)
    monkeypatch.setattr(tools_mod, "mcp_catalog_takes_image", lambda names: False)
    payload = _payload(enable_tools = True, permission_mode = "ask", mcp_enabled = True, mcp_image = _PNG)
    with pytest.raises(HTTPException) as excinfo:
        _run(inf, payload)
    assert excinfo.value.status_code == 400 and not entered


def test_an_mcp_image_reaches_the_tool_loop(monkeypatch):
    from core.inference import tools as tools_mod

    inf = _install(monkeypatch, "openai")
    monkeypatch.setattr(tools_mod, "mcp_catalog_takes_image", lambda names: True)
    entered = _capture_loop(monkeypatch, inf)
    _run(
        inf,
        _payload(enable_tools = True, permission_mode = "ask", mcp_enabled = True, mcp_image = _PNG),
    )
    assert entered["mcp_image"].mime == "image/png"


def test_a_non_streaming_request_without_any_confirm_intent_still_proxies(monkeypatch):
    """off/full never prompt, so they keep the legacy non-streaming passthrough."""
    inf = _install(monkeypatch, "openai")
    payload = _payload(stream = False, enable_tools = True, permission_mode = "off")
    _run(inf, payload)
    assert FakeExternalClient.last["passthrough"] is not None


def test_the_external_loop_is_told_which_model_the_usage_belongs_to(monkeypatch):
    """The loop's synthetic usage chunk must carry the real model, or tokens cannot be priced."""
    inf = _install(monkeypatch, "openai")
    entered = _capture_loop(monkeypatch, inf)
    _run(inf, _payload(enable_tools = True, external_model = "gpt-5.4"))
    assert entered["run"].model == "gpt-5.4"


def test_the_codex_loop_keeps_its_model_and_nudge_policy(monkeypatch):
    """Codex must carry its model and nudge setting through the shared loop's usage and policy."""
    from core.inference import openai_codex_tool_loop as loop_mod

    entered: dict = {}

    def _loop(*a, **k):
        entered.update(k)

        async def gen():
            yield "data: [DONE]\n\n"

        return gen()

    monkeypatch.setattr(loop_mod, "stream_with_studio_tools", _loop)
    run = loop_mod.CodexRunContext(
        provider_id = "saved-1",
        thread_id = None,
        session_id = None,
        messages = [{"role": "user", "content": "hi"}],
        model = "gpt-5.4-codex",
        reasoning_effort = None,
        response_format = None,
        tool_choice = None,
        continue_final_message = False,
    )
    policy = loop_mod.CodexToolPolicy(
        tools = [],
        max_calls = 1,
        timeout = 1,
        permission_mode = "auto",
        confirm_calls = False,
        bypass_permissions = False,
        rag_scope = None,
        nudge_tool_calls = False,
    )
    loop_mod.stream_codex_with_studio_tools(
        object(), run = run, policy = policy, cancel_event = threading.Event()
    )
    assert entered["run"].model == "gpt-5.4-codex"
    assert entered["policy"].nudge_tool_calls is False


def test_the_usage_chunk_falls_back_only_when_no_model_is_known():
    """`"external"` is the last resort, so it must not be what a real run reports."""
    from core.inference.studio_tool_loop import ToolLoopRun
    assert ToolLoopRun(messages = []).model is None


def test_a_data_recipe_key_cannot_spend_a_saved_cloud_credential(monkeypatch):
    """A data-recipe key must not spend saved cloud credentials; recipes run user-authored code."""
    from auth.authentication import API_KEY_PREFIX
    from routes import inference as inf

    seen: list[bool] = []
    inf = _install(monkeypatch, "openai", record_keys = seen)
    token = f"{API_KEY_PREFIX}deadbeefdeadbeef"
    _mint_internal(monkeypatch, inf, "data-recipe workflow")
    _run(inf, _payload(), request = _request(f"Bearer {token}"))
    assert seen == [False], "a recipe key must not unlock the saved connection"


def test_the_deep_research_key_keeps_its_saved_connection(monkeypatch):
    """The durable Deep Research hop is the caller the exception exists for."""
    from auth.authentication import API_KEY_PREFIX
    from routes import inference as inf

    seen: list[bool] = []
    inf = _install(monkeypatch, "openai", record_keys = seen)
    token = f"{API_KEY_PREFIX}deadbeefdeadbeef"
    _mint_internal(monkeypatch, inf, inf.auth_storage.DEEP_RESEARCH_WORKFLOW_KEY_NAME)
    _run(inf, _payload(), request = _request(f"Bearer {token}"))
    assert seen == [True]


def test_a_storage_failure_withholds_the_saved_connection(monkeypatch):
    """Fail closed: an unreadable key store must not hand out a credential."""
    from auth.authentication import API_KEY_PREFIX
    from routes import inference as inf

    seen: list[bool] = []
    inf = _install(monkeypatch, "openai", record_keys = seen)

    def _boom(_raw):
        raise RuntimeError("database is locked")

    _mint_internal(monkeypatch, inf, _boom)
    _run(inf, _payload(), request = _request(f"Bearer {API_KEY_PREFIX}deadbeefdeadbeef"))
    assert seen == [False]


def test_a_third_party_key_never_unlocks_a_saved_connection(monkeypatch):
    """The pre-existing rule: someone using Unsloth as an API server brings a key."""
    from auth.authentication import API_KEY_PREFIX
    from routes import inference as inf

    seen: list[bool] = []
    inf = _install(monkeypatch, "openai", record_keys = seen)
    monkeypatch.setattr(inf.auth_storage, "is_internal_api_key", lambda raw: False)
    _run(inf, _payload(), request = _request(f"Bearer {API_KEY_PREFIX}deadbeefdeadbeef"))
    assert seen == [False]


START = "Coding agents (unsloth start)"


@pytest.mark.parametrize(
    "record, allowed",
    [
        ({"role": "owner", "key_name": START, "key_purpose": "unsloth_start"}, True),
        ({"role": "owner", "key_name": "my script", "key_purpose": None}, False),
        # Named like the start key before the name was reserved, but never stamped.
        ({"role": "owner", "key_name": START, "key_purpose": None}, False),
        ({"role": "member", "key_name": START, "key_purpose": "unsloth_start"}, False),
    ],
)
def test_only_the_owners_unsloth_start_key_unlocks_a_saved_connection(monkeypatch, record, allowed):
    from auth.authentication import API_KEY_PREFIX
    from routes import inference as inf

    seen: list[bool] = []
    inf = _install(monkeypatch, "openai", record_keys = seen)
    monkeypatch.setattr(inf.auth_storage, "is_internal_api_key", lambda raw: False)
    monkeypatch.setattr(
        inf.auth_storage, "validate_api_key_account", lambda raw, touch = True: (record, "s")
    )
    _run(inf, _payload(), request = _request(f"Bearer {API_KEY_PREFIX}deadbeefdeadbeef"))
    assert seen == [allowed]


def test_a_keyless_caller_never_unlocks_a_saved_connection(monkeypatch):
    from auth import authentication
    from routes import inference as inf

    seen: list[bool] = []
    inf = _install(monkeypatch, "openai", record_keys = seen)
    monkeypatch.setattr(authentication, "admitted_without_session", lambda request: True)
    _run(inf, _payload(), request = _request())
    assert seen == [False]


def test_an_interactive_session_still_uses_its_saved_connection(monkeypatch):
    """Unsloth's own chat sends a session JWT and no API key at all."""
    from routes import inference as inf

    seen: list[bool] = []
    inf = _install(monkeypatch, "openai", record_keys = seen)
    _run(inf, _payload(), request = _request())
    assert seen == [True]


def test_the_external_disconnect_watcher_is_awaited_after_cancel():
    """Await the external disconnect watcher after cancel(), or asyncio logs an unretrieved exception."""
    import ast
    import pathlib

    source = (pathlib.Path(__file__).resolve().parents[1] / "routes" / "inference.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(source)
    cancels = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "cancel"
        and isinstance(node.value, ast.Name)
        and node.value.id == "disconnect_task"
    ]
    gathers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "disconnect_task"
    ]
    assert len(gathers) >= len(cancels) * 3, "each disconnect_task.cancel() needs a gather"


def test_transport_cancellation_is_wired_through_the_loop():
    """The transport must observe the flag /inference/cancel actually sets."""
    from core.inference.external_tool_transport import OAICompatTransport

    class _Stalling:
        provider_type = "custom"

        def __init__(self):
            self.torn_down = False
            self.released = asyncio.Event()

        async def stream_chat_completion(self, **_kwargs):
            try:
                yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
                await self.released.wait()
            finally:
                self.torn_down = True

    async def scenario():
        client = _Stalling()
        cancel_event = threading.Event()
        seen: list[str] = []

        async def consume():
            async for line in OAICompatTransport(client, model = "m").stream(
                messages = [{"role": "user", "content": "hi"}],
                tools = None,
                tool_choice = "auto",
                cancel_event = cancel_event,
            ):
                seen.append(line)

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0.1)
        assert seen and not client.torn_down
        cancel_event.set()
        await asyncio.wait_for(task, timeout = 5.0)
        assert client.torn_down

    _drive(scenario())
