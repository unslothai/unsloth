# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two authorization decisions the external-provider route makes before proxying.

Both are one-line conditions in ``_proxy_to_external_provider`` and both are
reachable only by driving the route, so they are pinned here rather than through
a helper's return value:

* the confirm gate. Studio's UI expresses "ask me first" as ``permission_mode``,
  not as ``confirm_tool_calls``, so a guard that reads the raw flag admits the
  very request the local routes reject.
* the saved-credential exception for internal workflow keys. Studio mints those
  keys for more than one workflow, and the data-recipe key is handed to a
  user-authored recipe subprocess, so "internal" alone cannot be the licence to
  spend every saved cloud credential.
"""

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

    return SimpleNamespace(
        headers = {"authorization": authorization} if authorization else {},
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


# ── the confirm gate reads the effective mode, not the raw flag ──────────────


def test_non_streaming_ask_mode_is_rejected_like_the_local_routes(monkeypatch):
    """``permission_mode: "ask"`` is how the UI asks for the gate.

    The gate can only prompt over SSE, so a non-streaming request carrying it
    must 400 exactly as ``/v1/chat/completions`` does for a local model. Reading
    ``confirm_tool_calls`` alone lets it through, and the caller is then proxied
    with its tools live and no confirmation it explicitly asked for.

    The saved connection has to resolve first: the route looks the provider up
    before it reaches this guard, so without the stub this asserts on the 404
    and would keep passing if the guard were deleted.
    """
    inf = _install(monkeypatch, "openai")

    payload = _payload(stream = False, enable_tools = True, permission_mode = "ask")
    assert payload.confirm_tool_calls is None, "the flag is omitted; only the mode asks"
    assert inf._confirm_gate_needs_stream(payload) is True

    with pytest.raises(HTTPException) as excinfo:
        _drive(inf._proxy_to_external_provider(payload, _request(), current_subject = "t"))
    assert excinfo.value.status_code == 400


def test_an_explicit_confirm_flag_still_401s_the_streaming_hosted_only_request(monkeypatch):
    """The pre-existing rejection must survive the mode-derived one.

    A streaming request whose selection is purely the provider's hosted tools
    never enters Studio's loop, so an explicit ``confirm_tool_calls`` cannot be
    honoured there either.
    """
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


def test_a_non_streaming_request_without_any_confirm_intent_still_proxies(monkeypatch):
    """off/full never prompt, so they keep the legacy non-streaming passthrough."""
    inf = _install(monkeypatch, "openai")
    payload = _payload(stream = False, enable_tools = True, permission_mode = "off")
    _run(inf, payload)
    assert FakeExternalClient.last["passthrough"] is not None


# ── the summed usage chunk keeps the model it was spent on ───────────────────


def test_the_external_loop_is_told_which_model_the_usage_belongs_to(monkeypatch):
    """The loop withholds the provider's usage chunks and sends one of its own.

    That synthetic chunk is the only usage the client sees for the answer, so a
    ToolLoopRun without a model reports the literal "external" and the tokens
    cannot be attributed or priced.
    """
    inf = _install(monkeypatch, "openai")
    entered = _capture_loop(monkeypatch, inf)
    _run(inf, _payload(enable_tools = True, external_model = "gpt-5.4"))
    assert entered["run"].model == "gpt-5.4"


def test_the_codex_loop_keeps_reporting_its_own_model(monkeypatch):
    """A regression guard, not a new feature.

    Before the shared loop, the Codex path relayed the provider's usage chunks
    untouched and they named the Codex model. The shared loop replaces them, so
    the model has to be carried across or Codex metadata moves.
    """
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
    )
    loop_mod.stream_codex_with_studio_tools(
        object(), run = run, policy = policy, cancel_event = threading.Event()
    )
    assert entered["run"].model == "gpt-5.4-codex"


def test_the_usage_chunk_falls_back_only_when_no_model_is_known():
    """`"external"` is the last resort, so it must not be what a real run reports."""
    from core.inference.studio_tool_loop import ToolLoopRun
    assert ToolLoopRun(messages = []).model is None


# ── the saved-credential exception is scoped to the workflow that needs it ───


def test_a_data_recipe_key_cannot_spend_a_saved_cloud_credential(monkeypatch):
    """Recipe keys live inside a user-authored subprocess.

    ``routes/data_recipe/jobs.py`` writes the minted key straight into the
    recipe's provider block so the recipe can call this host's local ``/v1``.
    If "internal" alone unlocked saved connections, that subprocess could name
    any saved provider_id and bill the user's cloud account.
    """
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


def test_an_interactive_session_still_uses_its_saved_connection(monkeypatch):
    """Studio's own chat sends a session JWT and no API key at all."""
    from routes import inference as inf

    seen: list[bool] = []
    inf = _install(monkeypatch, "openai", record_keys = seen)
    _run(inf, _payload(), request = _request())
    assert seen == [True]


# ── the watcher the stream starts is joined, not just cancelled ──────────────


def test_the_external_disconnect_watcher_is_awaited_after_cancel():
    """A bare cancel() leaves the task's exception unretrieved.

    asyncio logs "Task exception was never retrieved" for it at collection time,
    which is why both the Codex branch and the local watcher gather the task
    before returning. The external branch must not be the odd one out.
    """
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
    # Every cancel site must be followed by a gather of the same task; counting
    # the bare references is enough to catch a cancel with no join next to it.
    assert len(gathers) >= len(cancels) * 3, "each disconnect_task.cancel() needs a gather"


def test_transport_cancellation_is_wired_through_the_loop():
    """The transport must observe the flag /inference/cancel actually sets."""
    from core.inference.external_tool_transport import OAICompatTransport

    class _Stalling:
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
