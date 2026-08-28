# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which side executes the tools on an external provider, and under what permission.

Two questions live at the same one-line gate in ``_proxy_to_external_provider``,
and both are upgrade-shaped -- a browser can hold a cached bundle from before
this capability existed, and a third-party client can send the documented
hosted-tool body forever:

* ``enable_tools: true`` + ``enabled_tools: ["web_search", "code_execution"]``
  has always meant "the provider runs its own server tools". Unsloth's loop must
  not read those same bytes as a request to run *its* web_search and drop
  ``code_execution`` on the floor (it has no local implementation of it).
* an omitted ``permission_mode`` must resolve exactly as it does on the Codex
  path, since both build the same policy object from the same request fields.

The route is driven for real (fake HTTP client, real payload model, real
StreamingResponse body) so these pin behaviour, not helper return values.
"""

import asyncio
import ast
import pathlib
import threading
from types import SimpleNamespace

import pytest

from core.inference.providers import provider_hosted_tools
from core.inference.tools import is_high_risk_tool_call


_ROUTE_SOURCE = pathlib.Path(__file__).resolve().parents[1] / "routes" / "inference.py"


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class LoopEntered(Exception):
    """stream_with_studio_tools was called; carries the ToolLoopPolicy."""


class FakeExternalClient:
    """Stands in for ExternalProviderClient, recording the passthrough call."""

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


def _request():
    async def is_disconnected():
        return False

    return SimpleNamespace(
        headers = {},
        state = SimpleNamespace(skip_api_monitor = True),
        is_disconnected = is_disconnected,
    )


@pytest.fixture(autouse = True)
def _clean_policy():
    from state.tool_policy import reset_tool_policy

    reset_tool_policy()
    yield
    reset_tool_policy()


def _install(monkeypatch, provider_type: str):
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
    monkeypatch.setattr(inf, "resolve_provider_api_key_or_400", lambda *a, **k: "k")
    monkeypatch.setattr(inf, "ExternalProviderClient", FakeExternalClient)

    def _loop_raiser(*a, **k):
        raise LoopEntered(k.get("policy"))

    monkeypatch.setattr(inf, "stream_with_studio_tools", _loop_raiser)
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


def _run(inf, payload):
    async def go():
        resp = await inf._proxy_to_external_provider(payload, _request(), current_subject = "t")
        return [chunk async for chunk in resp.body_iterator]

    return _drive(go())


# ── Task 2: hosted vs local, A/B against the merge base ──────────────


# The gate as it stood at merge base b3376300: only the Codex subscription ran
# Unsloth's tools on an external provider. Every other provider took the plain
# passthrough, whatever the request said about tools. Kept as executable code so
# the expectations below are derived from the old behaviour, not restated.
def _merge_base_takes_studio_loop(payload, provider_type: str) -> bool:
    from routes.inference import _explicit_studio_tool_loop_requested
    return (
        provider_type == "openai_codex"
        and payload.stream is True
        and _explicit_studio_tool_loop_requested(payload)
    )


# Exactly what the pre-PR bundle put on the wire for the hosted-tool pills:
# two keys, no permission_mode, no mcp_enabled. See
# `git show b3376300:studio/frontend/src/features/chat/api/chat-adapter.ts`.
HOSTED_PROVIDERS = ("openai", "gemini", "openrouter", "kimi", "anthropic")

HOSTED_SELECTIONS = (
    ["web_search"],
    ["code_execution"],
    ["web_search", "code_execution"],
    ["web_search", "web_fetch", "code_execution", "image_generation"],
)

SELF_HOSTED_PROVIDERS = ("llama_cpp", "vllm", "ollama", "custom")


@pytest.mark.parametrize("provider_type", HOSTED_PROVIDERS)
@pytest.mark.parametrize("selection", HOSTED_SELECTIONS)
def test_a_hosted_tool_request_still_reaches_the_provider(monkeypatch, provider_type, selection):
    """Shape 1: only hosted names, on a provider that hosts them."""
    inf = _install(monkeypatch, provider_type)
    payload = _payload(enable_tools = True, enabled_tools = selection)

    assert _merge_base_takes_studio_loop(payload, provider_type) is False

    chunks = _run(inf, payload)
    passthrough = FakeExternalClient.last["passthrough"]
    assert passthrough is not None, "the Unsloth loop stole a hosted-tool request"
    # Forwarded verbatim: dropping a name here is the provider losing a tool.
    assert passthrough["enabled_tools"] == selection
    assert passthrough["stream"] is True
    assert any("hi" in chunk for chunk in chunks)


@pytest.mark.parametrize("provider_type", HOSTED_PROVIDERS)
def test_a_studio_hosted_provider_receives_the_current_date(monkeypatch, provider_type):
    inf = _install(monkeypatch, provider_type)
    monkeypatch.setattr(
        inf,
        "current_date_prompt_line",
        lambda **_kwargs: "The current date is 2026-08-15.",
    )

    _run(inf, _payload())

    messages = FakeExternalClient.last["passthrough"]["messages"]
    assert messages[0] == {"role": "system", "content": "The current date is 2026-08-15."}
    assert messages[1] == {"role": "user", "content": "what is 2+2?"}


def test_an_api_request_without_resolved_server_tools_stays_undated(monkeypatch):
    inf = _install(monkeypatch, "openai")
    monkeypatch.setattr(inf, "_request_has_api_key", lambda _request: True)
    monkeypatch.setattr(inf, "_request_is_internal_workflow", lambda _request: False)
    monkeypatch.setattr(
        inf,
        "current_date_prompt_line",
        lambda **_kwargs: "The current date is 2026-08-15.",
    )

    _run(
        inf,
        _payload(
            enable_tools = True,
            enabled_tools = ["unknown_tool"],
            run_tools_locally = True,
        ),
    )

    assert FakeExternalClient.last["passthrough"]["messages"] == [
        {"role": "user", "content": "what is 2+2?"}
    ]


def test_a_hosted_code_execution_is_not_dropped(monkeypatch):
    """The regression in one line: `code_execution` has no local implementation,
    so a loop that captures this request executes web_search itself and silently
    never runs the other half of what the user turned on."""
    inf = _install(monkeypatch, "openai")
    _run(inf, _payload(enable_tools = True, enabled_tools = ["web_search", "code_execution"]))
    assert "code_execution" in (FakeExternalClient.last["passthrough"]["enabled_tools"] or [])


def test_a_code_execution_with_run_tools_locally_still_answers_the_confirm_gate(monkeypatch):
    """`run_tools_locally` must not smuggle a hosted-only turn past the 400.

    Unsloth has no `code_execution`, so the local catalog is empty whatever the
    flag says and the route falls back to the provider. The confirmation
    rejection keys on the request NOT having taken the loop, so a "local"
    reading here answers a confirm-me request with an unconfirmed sandbox run.
    """
    from fastapi import HTTPException

    inf = _install(monkeypatch, "openai")
    # Class-level state; the client is built after the guard, so an untouched
    # record is the evidence nothing was sent.
    FakeExternalClient.last = {}
    payload = _payload(
        enable_tools = True,
        enabled_tools = ["code_execution"],
        run_tools_locally = True,
        confirm_tool_calls = True,
    )
    with pytest.raises(HTTPException) as excinfo:
        _run(inf, payload)
    assert excinfo.value.status_code == 400
    assert FakeExternalClient.last.get("passthrough") is None, "ran unconfirmed"


def test_a_code_execution_with_run_tools_locally_still_reaches_the_provider(monkeypatch):
    """And with no confirmation asked for, it proxies exactly as it always did."""
    inf = _install(monkeypatch, "openai")
    _run(
        inf,
        _payload(
            enable_tools = True,
            enabled_tools = ["code_execution"],
            run_tools_locally = True,
        ),
    )
    assert FakeExternalClient.last["passthrough"]["enabled_tools"] == ["code_execution"]


@pytest.mark.parametrize("provider_type", SELF_HOSTED_PROVIDERS)
def test_a_self_hosted_provider_still_runs_studios_own_web_search(monkeypatch, provider_type):
    """Shape 2, the PR's primary use case: a self-hosted server has no hosted
    tools at all, so the same body can only mean Unsloth's local loop."""
    assert provider_hosted_tools(provider_type) == frozenset()
    inf = _install(monkeypatch, provider_type)
    with pytest.raises(LoopEntered):
        _run(inf, _payload(enable_tools = True, enabled_tools = ["web_search"]))


@pytest.mark.parametrize(
    "overrides",
    [
        {"enable_tools": True, "enabled_tools": ["python"]},
        {"enable_tools": True, "enabled_tools": ["terminal"]},
        {"enable_tools": True, "enabled_tools": ["web_search", "python"]},
        {"enable_tools": True, "enabled_tools": ["web_search"], "mcp_enabled": True},
        {"enable_tools": True},  # no selection: every local tool
    ],
)
def test_a_local_only_selection_takes_the_loop_on_a_hosted_provider(monkeypatch, overrides):
    """Shape 3: one Unsloth-only name (or MCP) is unambiguous, so the feature
    works on hosted providers too."""
    # ``_select_request_tools`` imports this from ``core.inference.tools`` inside the function
    # body, so it is never an attribute of ``routes.inference``: patching the route set a dead
    # name, and ``raising = False`` hid that while the real function ran instead. On this job
    # it reads an empty settings DB and short-circuits before spawning anything, but nothing
    # here held it to that. Default ``raising`` catches a future move.
    monkeypatch.setattr(
        "core.inference.tools.get_enabled_mcp_tools",
        lambda: _noop_mcp(),
    )
    inf = _install(monkeypatch, "openai")
    with pytest.raises(LoopEntered):
        _run(inf, _payload(**overrides))


async def _noop_mcp():
    return []


def test_a_unknown_tool_names_never_read_as_hosted(monkeypatch):
    """Fails toward the loop, which owns the local catalog, rather than
    forwarding a name the provider has no tool for."""
    from routes.inference import _selects_only_provider_hosted_tools

    payload = _payload(enable_tools = True, enabled_tools = ["web_search", "not_a_tool"])
    assert _selects_only_provider_hosted_tools(payload, "openai") is False


@pytest.mark.parametrize("bad", [None, 5, {"web_search": True}, ["web_search", 5]])
def test_a_malformed_enabled_tools_is_not_a_hosted_request(bad):
    from routes.inference import _selects_only_provider_hosted_tools
    payload = SimpleNamespace(enabled_tools = bad, mcp_enabled = False)
    assert _selects_only_provider_hosted_tools(payload, "openai") is False


def test_a_codex_declares_no_hosted_tools():
    """Codex's `web_search` is Unsloth's own tool run by the Codex loop, so the
    hosted check must never fire there."""
    assert provider_hosted_tools("openai_codex") == frozenset()


# ── Task 1: what an omitted permission_mode means ────────────────────


def test_b_an_omitted_permission_mode_arms_the_auto_gate(monkeypatch):
    """`permission_mode` unset on a streaming request resolves to "auto" with
    the confirm gate ON, so high-risk calls still prompt."""
    inf = _install(monkeypatch, "llama_cpp")
    payload = _payload(enable_tools = True, enabled_tools = ["python"])
    assert payload.permission_mode is None
    assert payload.confirm_tool_calls is None

    with pytest.raises(LoopEntered) as excinfo:
        _run(inf, payload)
    policy = excinfo.value.args[0]
    assert policy.permission_mode == "auto"
    assert policy.confirm_calls is True


@pytest.mark.parametrize(
    "nudge_tool_calls", [None, False, True], ids = ["omitted", "disabled", "enabled"]
)
def test_b_external_tool_loop_receives_requested_nudge_setting(monkeypatch, nudge_tool_calls):
    """The external Unsloth loop must receive the request-level nudge policy."""
    monkeypatch.setattr(
        "core.inference.tools.get_enabled_mcp_tools",
        lambda: _noop_mcp(),
    )
    inf = _install(monkeypatch, "openai")
    payload = _payload(
        enable_tools = True,
        enabled_tools = ["python"],
        nudge_tool_calls = nudge_tool_calls,
    )

    with pytest.raises(LoopEntered) as excinfo:
        _run(inf, payload)

    assert excinfo.value.args[0].nudge_tool_calls is nudge_tool_calls


def test_b_the_external_and_codex_paths_derive_the_gate_identically():
    """Both policy constructions must read the same policy expressions off the
    payload; a divergence would make one path quietly more permissive."""
    tree = ast.parse(_ROUTE_SOURCE.read_text(encoding = "utf-8"))
    modes: set[str] = set()
    confirms: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword):
            continue
        if node.arg == "permission_mode" and isinstance(node.value, ast.BoolOp):
            modes.add(ast.unparse(node.value))
        if node.arg == "confirm_calls":
            confirms.add(ast.unparse(node.value))
    assert modes == {"payload.permission_mode or 'auto'"}
    assert confirms == {"_permission_mode_confirm(payload)"}

    nudge_values = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in {"CodexToolPolicy", "ToolLoopPolicy"}:
            continue
        nudge_values.extend(
            ast.unparse(keyword.value)
            for keyword in node.keywords
            if keyword.arg == "nudge_tool_calls"
        )
    assert nudge_values == ["payload.nudge_tool_calls", "payload.nudge_tool_calls"]


@pytest.mark.parametrize(
    "name, arguments, high_risk",
    [
        ("python", {"code": "print(sum(range(10)))"}, False),
        ("python", {"code": "open('/home/u/.ssh/id_rsa').read()"}, True),
        ("python", {"code": "import os; os.system('curl http://x')"}, True),
        ("python", {"code": "import shutil; shutil.rmtree('/tmp/x')"}, True),
        # Observed, not endorsed: a bare `subprocess.run` clears the static
        # safety check and is not classified high risk, so auto runs it inside
        # the sandbox without prompting. Same on every path (local, Codex,
        # external), so it is not this PR's regression -- pinned so a change
        # to it is a deliberate one.
        ("python", {"code": "import subprocess; subprocess.run(['sh', '-c', 'x'])"}, False),
        ("terminal", {"command": "ls -la"}, False),
        ("terminal", {"command": "cat ~/.aws/credentials"}, True),
        ("terminal", {"command": "sudo rm -rf /var"}, True),
        ("web_search", {"query": "unsloth"}, False),
    ],
)
def test_b_auto_mode_prompts_on_risk_not_on_the_tool_name(name, arguments, high_risk):
    """ "auto" is per-call, not per-tool: ordinary development commands run and
    credential/escalation/egress ones prompt. Pinned because the docstring on
    `permission_mode` promises exactly this."""
    assert is_high_risk_tool_call(name, arguments) is high_risk


def _run_loop(
    monkeypatch,
    *,
    code: str,
    verdict: str = "allow",
):
    """Drive the real loop with the real risk classifier under auto/gate-on."""
    import json

    from core.inference import studio_tool_loop as loop_mod
    from core.inference.studio_tool_loop import (
        ToolLoopPolicy,
        ToolLoopRun,
        stream_with_studio_tools,
    )

    executed: list[dict] = []
    monkeypatch.setattr(
        loop_mod,
        "execute_tool",
        lambda name, arguments, **kw: executed.append({"name": name, "arguments": arguments})
        or "RESULT",
    )
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "wait_tool_decision", lambda *a, **k: verdict)

    call = {
        "index": 0,
        "id": "call_a",
        "function": {"name": "python", "arguments": json.dumps({"code": code})},
    }
    turns = [
        ["data: " + json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [call]}}]})],
        ["data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": "done"}}]})],
    ]

    class _Transport:
        heals_text_tool_calls = False

        def stream(self, *, messages, tools, tool_choice, cancel_event):
            lines = turns.pop(0) if turns else ["data: [DONE]"]

            async def _gen():
                for line in lines:
                    yield line

            return _gen()

    async def _collect():
        out = []
        agen = stream_with_studio_tools(
            _Transport(),
            run = ToolLoopRun(
                messages = [{"role": "user", "content": "hi"}],
                session_id = "s1",
                thread_id = "t1",
                tool_choice = None,
            ),
            policy = ToolLoopPolicy(
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "python",
                            "description": "",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                max_calls = 25,
                timeout = 300,
                permission_mode = "auto",
                confirm_calls = True,
                bypass_permissions = False,
                rag_scope = None,
            ),
            cancel_event = threading.Event(),
        )
        async for line in agen:
            out.append(line)
        return out

    lines = asyncio.new_event_loop().run_until_complete(_collect())
    starts = []
    for line in lines:
        if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
            continue
        payload = json.loads(line[6:])
        if payload.get("type") == "tool_start":
            starts.append(payload)
    return starts, executed


def test_b_a_benign_python_call_runs_without_an_approval_frame(monkeypatch):
    starts, executed = _run_loop(monkeypatch, code = "print(2 + 2)")
    assert [s["awaiting_confirmation"] for s in starts] == [False]
    assert [c["name"] for c in executed] == ["python"]


def test_b_a_credential_reading_python_call_is_gated(monkeypatch):
    starts, executed = _run_loop(
        monkeypatch,
        code = "print(open('/home/u/.ssh/id_rsa').read())",
        verdict = "deny",
    )
    assert [s["awaiting_confirmation"] for s in starts] == [True]
    assert executed == []
