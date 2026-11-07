# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import sys
import threading
import types

import pytest

from core.inference import claude_agent
from core.inference import studio_tool_loop as loop_mod
from core.inference.claude_agent_transport import ClaudeAgentTransport, close_after
from core.inference.external_provider import ExternalProviderClient
from core.inference.providers import get_connectable_provider_info, list_available_providers
from core.inference.studio_tool_loop import ToolLoopPolicy, ToolLoopRun, stream_with_studio_tools

PNG = "data:image/png;base64,iVBORw0KGgo="
MODELS = [
    {"value": "default"},
    {
        "value": "opus",
        "resolvedModel": "claude-opus-5-5",
        "displayName": "Opus 5.5",
        "supportsEffort": True,
        "supportedEffortLevels": ["low", "medium", "high", "xhigh", "max"],
        "supportsThinkingOff": False,
    },
    {
        "value": "claude-sonnet-4-6",
        "resolvedModel": "claude-sonnet-4-6",
        "displayName": "Sonnet 4.6",
        "supportsEffort": True,
        "supportedEffortLevels": ["low", "medium", "high", "max"],
        "supportsThinkingOff": True,
    },
    {
        "value": "claude-haiku-4-5-20251001",
        "resolvedModel": "claude-haiku-4-5-20251001",
        "displayName": "Haiku 4.5",
        "supportsThinkingOff": True,
    },
]
INFO = {
    "claude_code_version": "2.1.318",
    "account": {"subscriptionType": "Claude Ultra", "apiProvider": "firstParty"},
    "models": MODELS,
}


class _Message:
    def __init__(self, **fields):
        self.__dict__.update(fields)


class _ToolUse:
    def __init__(self, call_id, name, arguments):
        self.id, self.name, self.input = call_id, name, arguments


def _fake_sdk(turns, server_info, connect_error, windows):
    sdk = types.ModuleType("claude_agent_sdk")
    sdk.clients = []
    turns = [list(turn) for turn in turns]
    end = object()

    class ClaudeSDKError(Exception):
        pass

    for name in (
        "ClaudeAgentOptions",
        "StreamEvent",
        "AssistantMessage",
        "ResultMessage",
        "UserMessage",
        "RateLimitEvent",
        "ToolResultBlock",
        "HookMatcher",
        "ToolAnnotations",
    ):
        setattr(sdk, name, type(name, (_Message,), {}))
    sdk.ClaudeSDKError = ClaudeSDKError
    sdk.ToolUseBlock = _ToolUse

    def tool(
        name,
        description,
        input_schema,
        annotations = None,
    ):
        return lambda handler: types.SimpleNamespace(
            name = name,
            description = description,
            input_schema = input_schema,
            annotations = annotations,
            handler = handler,
        )

    sdk.tool = tool
    sdk.create_sdk_mcp_server = lambda name, tools: {"type": "sdk", "name": name, "tools": tools}

    class ClaudeSDKClient:
        def __init__(self, options):
            self.options = options
            self.sent = []
            self.results = {}
            self.interrupted = False
            self.disconnected = False
            self.queue = asyncio.Queue()
            self.player = None
            self.model = getattr(options, "model", None)
            sdk.clients.append(self)

        async def connect(self):
            if connect_error:
                raise ClaudeSDKError(connect_error)

        async def get_server_info(self):
            return server_info

        async def set_model(self, model):
            self.model = model

        async def get_context_usage(self):
            if self.model not in (windows or {}):
                raise ClaudeSDKError("no usage")
            return {"rawMaxTokens": windows[self.model], "maxTokens": 1}

        async def query(self, prompt):
            async for message in prompt:
                self.sent.append(message)
            self.player = asyncio.ensure_future(self._play(turns.pop(0) if turns else []))

        async def _play(self, turn):
            # Like Claude Code: a tool call blocks the turn until its handler returns.
            for step in turn:
                if step[0] == "wait":
                    await asyncio.Event().wait()
                if step[0] != "call":
                    self.queue.put_nowait(getattr(sdk, step[0])(**step[1]))
                    continue
                _, call_id, name, arguments = step
                for matcher in (self.options.hooks or {}).get("PreToolUse", []):
                    for hook in matcher.hooks:
                        await hook(
                            {
                                "hook_event_name": "PreToolUse",
                                "tool_name": name,
                                "tool_input": arguments,
                                "tool_use_id": call_id,
                            },
                            call_id,
                            None,
                        )
                tools = self.options.mcp_servers["studio"]["tools"]
                handler = next(t.handler for t in tools if f"mcp__studio__{t.name}" == name)
                result = await handler(arguments)
                self.results[call_id] = result
                self.queue.put_nowait(
                    sdk.UserMessage(
                        content = [sdk.ToolResultBlock(tool_use_id = call_id, **result)],
                        parent_tool_use_id = None,
                    )
                )

        async def receive_messages(self):
            while True:
                message = await self.queue.get()
                if message is end:
                    return
                yield message

        async def interrupt(self):
            self.interrupted = True

        async def disconnect(self):
            self.disconnected = True
            if self.player is not None:
                self.player.cancel()
            self.queue.put_nowait(end)

    sdk.ClaudeSDKClient = ClaudeSDKClient
    return sdk


def _event(event):
    return ("StreamEvent", {"event": event, "parent_tool_use_id": None})


def _text(text):
    return _event({"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}})


def _assistant(*blocks, error = None):
    return (
        "AssistantMessage",
        {"content": list(blocks), "error": error, "parent_tool_use_id": None},
    )


def _result(**fields):
    base = {
        "subtype": "success",
        "is_error": False,
        "result": "",
        "errors": None,
        "stop_reason": "end_turn",
        "api_error_status": None,
    }
    return ("ResultMessage", {**base, **fields})


def _tool_turn(calls, text = ""):
    steps = [_event({"type": "message_start", "message": {"usage": {"input_tokens": 9}}})]
    if text:
        steps.append(_text(text))
    steps += [_assistant(_ToolUse(i, f"mcp__studio__{name}", args)) for i, name, args in calls]
    steps.append(_event({"type": "message_delta", "usage": {"output_tokens": 4}}))
    steps.append(_event({"type": "message_stop"}))
    steps += [("call", i, f"mcp__studio__{name}", args) for i, name, args in calls]
    return steps


@pytest.fixture(autouse = True)
def _fresh_catalog(monkeypatch):
    monkeypatch.setattr(claude_agent, "_catalog", {"version": None, "models": {}})


def _install(
    monkeypatch,
    tmp_path,
    turns = (),
    cli = True,
    server_info = None,
    connect_error = None,
    windows = None,
):
    sdk = _fake_sdk(turns, server_info, connect_error, windows)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if cli:
        claude = bin_dir / "claude"
        claude.write_text("#!/bin/sh\n")
        claude.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    return sdk


def _stream(
    messages,
    model = "sonnet",
    **kwargs,
):
    async def run():
        client = ExternalProviderClient("claude_agent", "", "")
        return [
            line
            async for line in client.stream_chat_completion(
                messages = messages, model = model, **kwargs
            )
        ]

    return asyncio.run(run())


def _payloads(lines):
    return [
        json.loads(line[len("data: ") :])
        for line in lines
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def _content(lines):
    return "".join(
        p["choices"][0]["delta"].get("content") or "" for p in _payloads(lines) if p.get("choices")
    )


def _finishes(lines):
    return [p["choices"][0]["finish_reason"] for p in _payloads(lines) if p.get("choices")]


def _error(lines):
    return json.loads(lines[-1][len("data: ") :])["error"]


def _tool(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


TOOLS = [_tool("get_time"), _tool("get_weather")]


def _claude_agent_row():
    return next(r for r in list_available_providers() if r["provider_type"] == "claude_agent")


def test_claude_agent_is_offered_while_off_but_connectable_only_when_opted_in(monkeypatch):
    monkeypatch.setenv("UNSLOTH_CLAUDE_AGENT", "0")
    assert _claude_agent_row()["available_but_off"] is True
    assert get_connectable_provider_info("claude_agent") is None
    assert all(
        not r["available_but_off"]
        for r in list_available_providers()
        if r["provider_type"] != "claude_agent"
    )

    monkeypatch.setenv("UNSLOTH_CLAUDE_AGENT", "1")
    row = _claude_agent_row()
    assert row["available_but_off"] is False
    assert row["display_name"] == "Claude Agent"
    assert row["model_list_mode"] == "curated"
    assert row["supports_studio_tools"] is True
    assert row["supports_tool_calling"] is True
    assert row["supports_vision"] is True
    assert get_connectable_provider_info("claude_agent") is not None


def test_reply_streams_as_openai_chunks(monkeypatch, tmp_path):
    sdk = _install(
        monkeypatch,
        tmp_path,
        turns = [
            [
                _event({"type": "message_start", "message": {"usage": {"input_tokens": 12}}}),
                _event(
                    {
                        "type": "content_block_delta",
                        "delta": {"type": "thinking_delta", "thinking": "Short."},
                    }
                ),
                _text("Hel"),
                _text("lo"),
                _event({"type": "message_delta", "usage": {"output_tokens": 5}}),
                _assistant(),
                _result(),
            ]
        ],
    )
    lines = _stream([{"role": "user", "content": "Say hello"}])

    assert lines[-1] == "data: [DONE]"
    assert _content(lines) == "<think>Short.</think>Hello"
    assert _finishes(lines)[-1] == "stop"
    usage = _payloads(lines)[-1]["usage"]
    assert usage["prompt_tokens"] == 12
    assert usage["completion_tokens"] == 5
    (client,) = sdk.clients
    assert client.sent[0]["message"] == {
        "role": "user",
        "content": [{"type": "text", "text": "Say hello"}],
    }
    assert client.disconnected is True


def test_output_token_limit_ends_the_reply_as_length(monkeypatch, tmp_path):
    _install(
        monkeypatch,
        tmp_path,
        turns = [
            [
                _text("Once"),
                _assistant(error = "max_output_tokens"),
                _result(
                    is_error = True,
                    result = "API Error: Claude's response exceeded the 30 output token maximum.",
                ),
            ]
        ],
    )
    lines = _stream([{"role": "user", "content": "Write a story"}], max_tokens = 30)

    assert lines[-1] == "data: [DONE]"
    assert not any("error" in p for p in _payloads(lines))
    assert _finishes(lines)[-1] == "length"


def test_options_isolate_claude_code_and_pass_model_and_effort(monkeypatch, tmp_path):
    sdk = _install(monkeypatch, tmp_path, turns = [[_result()]])
    _stream(
        [{"role": "user", "content": "hi"}],
        temperature = 0.2,
        max_tokens = 2048,
        reasoning_effort = "high",
    )
    options = sdk.clients[-1].options

    assert options.cli_path == str(tmp_path / "bin" / "claude")
    assert options.model == "sonnet"
    assert options.effort == "high"
    assert options.tools == []
    assert options.setting_sources == []
    assert options.strict_mcp_config is True
    assert json.loads(options.settings) == {"disableAllHooks": True}
    assert options.verbatim_prompts is True
    assert options.include_partial_messages is True
    assert options.system_prompt["type"] == "preset"
    assert options.system_prompt["preset"] == "claude_code"
    assert "Unsloth Studio" in options.system_prompt["append"]
    assert options.env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    assert options.env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert "DISABLE_AUTO_COMPACT" not in options.env
    assert options.env["CLAUDE_CODE_SIMPLE"] == "0"
    assert options.env["ENABLE_TOOL_SEARCH"] == "false"
    assert options.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "2048"
    assert not hasattr(options, "temperature")
    assert not hasattr(options, "mcp_servers")


def test_user_turn_is_only_the_newest_message(monkeypatch, tmp_path):
    sdk = _install(monkeypatch, tmp_path, turns = [[_result()]])
    _stream(
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "My name is Ada."},
            {"role": "assistant", "content": "Hi Ada."},
            {"role": "user", "content": [{"type": "text", "text": "What is my name?"}]},
        ]
    )
    client = sdk.clients[-1]
    assert client.sent[0]["message"]["content"] == [{"type": "text", "text": "What is my name?"}]
    append = client.options.system_prompt["append"]
    assert "You are terse." in append
    assert append.index("<user>\nMy name is Ada.\n</user>") < append.index(
        "<assistant>\nHi Ada.\n</assistant>"
    )
    assert "What is my name?" not in append


def test_newest_turn_sends_images_as_content_blocks(monkeypatch, tmp_path):
    sdk = _install(monkeypatch, tmp_path, turns = [[_result()]])
    _stream(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these."},
                    {"type": "image_url", "image_url": {"url": PNG}},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
                    {"type": "image_url", "image_url": {"url": "data:image/bmp;base64,Qk0="}},
                ],
            }
        ]
    )
    content = sdk.clients[-1].sent[0]["message"]["content"]
    assert content[0] == {"type": "text", "text": "Compare these."}
    assert content[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
    }
    assert content[2] == {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/cat.jpg"},
    }
    assert content[3]["type"] == "text" and "left out" in content[3]["text"]


def test_an_image_only_turn_is_sent_and_answered(monkeypatch, tmp_path):
    sdk = _install(monkeypatch, tmp_path, turns = [[_text("Red."), _result()]])
    lines = _stream(
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": PNG}}]}]
    )

    assert _content(lines) == "Red."
    assert sdk.clients[-1].sent[0]["message"]["content"][0]["type"] == "image"


def test_earlier_images_become_markers_and_ride_in_the_new_turn(monkeypatch, tmp_path):
    sdk = _install(monkeypatch, tmp_path, turns = [[_result()]])
    _stream(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Keep this."},
                    {"type": "image_url", "image_url": {"url": PNG}},
                ],
            },
            {"role": "assistant", "content": "Kept."},
            {"role": "user", "content": "What did I send?"},
        ]
    )
    append = sdk.clients[-1].options.system_prompt["append"]
    assert "<user>\nKeep this.\n[image]\n</user>" in append
    assert "iVBORw0KGgo" not in append
    # A follow-up starts a new session, so the image must still reach Claude.
    content = sdk.clients[-1].sent[0]["message"]["content"]
    assert content[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
    }
    assert content[-1] == {"type": "text", "text": "What did I send?"}


@pytest.mark.parametrize(
    ("model", "requested", "sent"),
    [
        ("opus", "xhigh", "xhigh"),
        ("claude-sonnet-4-6", "xhigh", "high"),
        ("claude-sonnet-4-6", "max", "max"),
        ("claude-haiku-4-5-20251001", "high", None),
        ("unknown-model", "low", "low"),
        ("opus", "extreme", None),
    ],
)
def test_effort_is_clamped_to_what_the_model_supports(
    monkeypatch, tmp_path, model, requested, sent
):
    sdk = _install(monkeypatch, tmp_path, turns = [[_result()]], server_info = INFO)
    _stream([{"role": "user", "content": "hi"}], model = model, reasoning_effort = requested)

    assert sdk.clients[-1].options.effort == sent


@pytest.mark.parametrize(
    ("model", "thinking"),
    [
        ("claude-sonnet-4-6", {"type": "disabled"}),
        ("opus", {"type": "adaptive", "display": "summarized"}),
    ],
)
def test_thinking_turns_off_only_where_the_model_allows_it(monkeypatch, tmp_path, model, thinking):
    sdk = _install(monkeypatch, tmp_path, turns = [[_result()]], server_info = INFO)
    _stream([{"role": "user", "content": "hi"}], model = model, enable_thinking = False)

    probe, chat = sdk.clients
    assert probe.sent == []
    assert chat.options.thinking == thinking


def test_context_window_is_discovered_and_cached(monkeypatch, tmp_path):
    sdk = _install(
        monkeypatch,
        tmp_path,
        server_info = INFO,
        windows = {"opus": 1_000_000, "claude-sonnet-4-6": 200_000},
    )

    async def run():
        client = ExternalProviderClient("claude_agent", "", "")
        first = await client.served_context_window("opus")
        second = await client.served_context_window("claude-opus-5-5")
        return first, second

    assert asyncio.run(run()) == (1_000_000, 1_000_000)
    (probe,) = sdk.clients
    assert probe.sent == []


def test_context_window_falls_back_when_discovery_fails(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, server_info = INFO, windows = {})
    window = asyncio.run(claude_agent.claude_agent_context_window("opus"))

    assert window == claude_agent.FALLBACK_CONTEXT_WINDOW


def test_signed_out_error_tells_the_user_how_to_sign_in(monkeypatch, tmp_path):
    _install(
        monkeypatch,
        tmp_path,
        turns = [
            [
                _assistant(error = "authentication_failed"),
                _result(is_error = True, result = "Not logged in · Please run /login"),
            ]
        ],
    )
    error = _error(_stream([{"role": "user", "content": "hi"}]))

    assert error["code"] == "401"
    assert error["provider"] == "claude_agent"
    assert "Run `claude` in a terminal and sign in." in error["message"]
    assert "/login" not in error["message"]


@pytest.mark.parametrize(
    ("steps", "code", "expected"),
    [
        (
            [
                (
                    "RateLimitEvent",
                    {
                        "rate_limit_info": types.SimpleNamespace(
                            status = "rejected", rate_limit_type = "five_hour", resets_at = 0
                        )
                    },
                ),
                _assistant(error = "rate_limit"),
                _result(is_error = True, result = "You've hit your limit"),
            ],
            "429",
            "Your Claude usage limit was reached (five hour limit). "
            "It resets at 1970-01-01 00:00 UTC.",
        ),
        (
            [
                _assistant(error = "rate_limit"),
                _result(is_error = True, result = "Rate limited", api_error_status = 429),
            ],
            "429",
            "Your Claude usage limit was reached. Claude Code says: Rate limited",
        ),
        (
            [
                _assistant(error = "billing_error"),
                _result(is_error = True, result = "Credit balance is too low"),
            ],
            "402",
            "Credit balance is too low. Check your Claude plan and extra usage settings.",
        ),
        (
            [_assistant(error = "server_error"), _result(is_error = True, api_error_status = 529)],
            "529",
            "Anthropic is overloaded and Claude Code ran out of retries. Try again shortly.",
        ),
        (
            [_assistant(error = "server_error"), _result(is_error = True, result = "boom")],
            "502",
            "Anthropic returned HTTP 502 and Claude Code ran out of retries. Try again shortly.",
        ),
        (
            [
                _assistant(error = "invalid_request"),
                _result(is_error = True, result = "Prompt is too long", api_error_status = 400),
            ],
            "400",
            "Prompt is too long",
        ),
        (
            [_result(is_error = True, result = "OAuth token has expired")],
            "401",
            "Claude Code is signed out or its login expired. "
            "Run `claude` in a terminal and sign in.",
        ),
    ],
)
def test_errors_map_to_clear_messages(monkeypatch, tmp_path, steps, code, expected):
    _install(monkeypatch, tmp_path, turns = [steps])
    error = _error(_stream([{"role": "user", "content": "hi"}]))

    assert error["code"] == code
    assert error["message"] == expected


def test_missing_claude_cli_is_reported_without_starting_anything(monkeypatch, tmp_path):
    sdk = _install(monkeypatch, tmp_path, turns = [[_result()]], cli = False)
    lines = _stream([{"role": "user", "content": "hi"}])

    assert "Install Claude Code" in _error(lines)["message"]
    assert sdk.clients == []


def test_status_reports_plan_version_and_model_objects(monkeypatch, tmp_path):
    sdk = _install(
        monkeypatch,
        tmp_path,
        server_info = {**INFO, "account": {"subscriptionType": "max", "apiProvider": "firstParty"}},
        windows = {"opus": 1_000_000, "claude-sonnet-4-6": 200_000},
    )
    status = asyncio.run(claude_agent.claude_agent_status())

    assert status == {
        "ready": True,
        "version": "2.1.318",
        "account": "Claude Max",
        "message": None,
        "problem": None,
        "models": [
            {
                "id": "opus",
                "display_name": "Opus 5.5",
                "alias": True,
                "efforts": ["low", "medium", "high", "xhigh", "max"],
                "thinking_off": False,
                "context_window": 1_000_000,
            },
            {
                "id": "claude-sonnet-4-6",
                "display_name": "Sonnet 4.6",
                "alias": False,
                "efforts": ["low", "medium", "high", "max"],
                "thinking_off": True,
                "context_window": None,
            },
            {
                "id": "claude-haiku-4-5-20251001",
                "display_name": "Haiku 4.5",
                "alias": False,
                "efforts": [],
                "thinking_off": True,
                "context_window": None,
            },
        ],
    }
    (probe,) = sdk.clients
    assert probe.sent == []
    assert probe.disconnected is True
    assert asyncio.run(claude_agent.claude_agent_models()) == status["models"]
    assert asyncio.run(claude_agent.claude_agent_context_window("claude-sonnet-4-6")) == 200_000


@pytest.mark.parametrize(
    ("kwargs", "message", "problem"),
    [
        (
            {"server_info": {"account": {"tokenSource": "none"}, "models": []}},
            "Run `claude` in a terminal and sign in.",
            "signed_out",
        ),
        ({"connect_error": "boom"}, "Claude Code did not start: boom", "failed"),
        ({"cli": False}, claude_agent.CLI_MISSING, "cli_missing"),
    ],
)
def test_status_explains_what_is_missing(monkeypatch, tmp_path, kwargs, message, problem):
    _install(monkeypatch, tmp_path, **kwargs)
    status = asyncio.run(claude_agent.claude_agent_status())

    assert status["ready"] is False
    assert status["message"] == message
    assert status["problem"] == problem


def test_windows_runs_the_exe_behind_an_npm_shim(monkeypatch, tmp_path):
    # claude-agent-sdk refuses .cmd/.bat CLIs on Windows.
    npm = tmp_path / "npm"
    exe = npm / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    exe.parent.mkdir(parents = True)
    exe.write_bytes(b"MZ")
    monkeypatch.setattr(claude_agent.sys, "platform", "win32")
    monkeypatch.setattr(claude_agent.shutil, "which", lambda name: str(npm / "claude.CMD"))
    monkeypatch.setattr(claude_agent.Path, "home", lambda: tmp_path / "home")
    assert claude_agent.find_cli() == str(exe)
    exe.unlink()
    assert claude_agent.find_cli() == str(npm / "claude.CMD")
    native = tmp_path / "home" / ".local" / "bin" / "claude.exe"
    native.parent.mkdir(parents = True)
    native.write_bytes(b"MZ")
    assert claude_agent.find_cli() == str(native)


def _cli_prints(tmp_path, text):
    (tmp_path / "bin" / "claude").write_text(f"#!/bin/sh\necho '{text}'\n")


@pytest.mark.skipif(sys.platform == "win32", reason = "fake CLI is a shell script")
def test_cli_older_than_verbatim_prompts_is_refused_before_it_starts(monkeypatch, tmp_path):
    sdk = _install(monkeypatch, tmp_path, turns = [[_result()]])
    _cli_prints(tmp_path, "2.1.247 (Claude Code)")

    status = asyncio.run(claude_agent.claude_agent_status())
    lines = _stream([{"role": "user", "content": "/cost"}])

    assert status["ready"] is False
    assert status["problem"] == "failed"
    assert status["version"] == "2.1.247"
    assert "claude update" in status["message"]
    assert "2.1.247 is too old" in _error(lines)["message"]
    assert sdk.clients == []


@pytest.mark.skipif(sys.platform == "win32", reason = "fake CLI is a shell script")
def test_cli_version_comes_from_the_cli_when_initialize_omits_it(monkeypatch, tmp_path):
    info = {key: value for key, value in INFO.items() if key != "claude_code_version"}
    _install(monkeypatch, tmp_path, server_info = info)
    _cli_prints(tmp_path, "2.1.248 (Claude Code)")

    status = asyncio.run(claude_agent.claude_agent_status())

    assert status["ready"] is True
    assert status["version"] == "2.1.248"


def test_status_without_the_sdk_names_the_install_command(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    status = asyncio.run(claude_agent.claude_agent_status())

    assert status["ready"] is False
    assert "claude-agent-sdk" in status["message"]
    assert status["problem"] == "sdk_missing"


def _calls(lines):
    return [
        call
        for p in _payloads(lines)
        if p.get("choices")
        for call in p["choices"][0]["delta"].get("tool_calls") or []
    ]


async def _round(
    transport,
    messages,
    tools = TOOLS,
    tool_choice = "auto",
    cancel_event = None,
):
    return [
        line
        async for line in transport.stream(
            messages = messages,
            tools = tools,
            tool_choice = tool_choice,
            cancel_event = cancel_event or threading.Event(),
        )
    ]


def _replay(messages, lines, *results):
    calls = _calls(lines)
    replayed = messages + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": c["id"], "type": "function", "function": c["function"]} for c in calls
            ],
        }
    ]
    for call, result in zip(calls, results):
        replayed.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    return replayed


def test_tool_round_trip_keeps_one_session(monkeypatch, tmp_path):
    sdk = _install(
        monkeypatch,
        tmp_path,
        turns = [
            _tool_turn([("toolu_1", "get_time", {"city": "Paris"})], text = "Checking.")
            + [_text("It is noon in Paris."), _result()]
        ],
    )
    messages = [{"role": "user", "content": "Time in Paris?"}]

    async def run():
        transport = ClaudeAgentTransport(model = "haiku")
        first = await _round(transport, messages)
        second = await _round(transport, _replay(messages, first, "12:00"))
        await transport.aclose()
        return first, second

    first, second = asyncio.run(run())
    assert _content(first) == "Checking."
    assert _calls(first) == [
        {
            "index": 0,
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "get_time", "arguments": '{"city": "Paris"}'},
        }
    ]
    assert _finishes(first)[-1] == "tool_calls"
    assert _content(second) == "It is noon in Paris."
    assert _finishes(second)[-1] == "stop"
    (client,) = sdk.clients
    assert client.results["toolu_1"] == {
        "content": [{"type": "text", "text": "12:00"}],
        "is_error": False,
    }
    options = client.options
    assert options.tools == []
    assert options.allowed_tools == ["mcp__studio__*"]
    assert [t.name for t in options.mcp_servers["studio"]["tools"]] == ["get_time", "get_weather"]
    assert "mcp__studio__" in options.system_prompt["append"]
    assert len(client.sent) == 1
    assert client.disconnected is True


def test_two_calls_in_one_turn_and_a_denied_one(monkeypatch, tmp_path):
    sdk = _install(
        monkeypatch,
        tmp_path,
        turns = [
            _tool_turn(
                [
                    ("toolu_1", "get_time", {"city": "Tokyo"}),
                    ("toolu_2", "get_weather", {"city": "Lima"}),
                ]
            )
            + [_text("Done."), _result()]
        ],
    )
    messages = [{"role": "user", "content": "Time and weather?"}]

    async def run():
        transport = ClaudeAgentTransport(model = "haiku")
        first = await _round(transport, messages)
        second = await _round(
            transport,
            _replay(messages, first, "17:42", "The user declined to run this tool call."),
        )
        await transport.aclose()
        return first, second

    first, second = asyncio.run(run())
    assert [(c["index"], c["id"], c["function"]["name"]) for c in _calls(first)] == [
        (0, "toolu_1", "get_time"),
        (1, "toolu_2", "get_weather"),
    ]
    assert _content(second) == "Done."
    results = sdk.clients[0].results
    assert results["toolu_1"]["is_error"] is False
    assert results["toolu_2"] == {
        "content": [{"type": "text", "text": "The user declined to run this tool call."}],
        "is_error": True,
    }


def test_tool_result_images_join_the_last_result(monkeypatch, tmp_path):
    sdk = _install(
        monkeypatch,
        tmp_path,
        turns = [_tool_turn([("toolu_1", "get_time", {"city": "Oslo"})]) + [_result()]],
    )
    messages = [{"role": "user", "content": "Screenshot?"}]

    async def run():
        transport = ClaudeAgentTransport(model = "haiku")
        first = await _round(transport, messages)
        replayed = _replay(messages, first, "Took one.") + [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image from the tool call above."},
                    {"type": "image_url", "image_url": {"url": PNG}},
                ],
            }
        ]
        await _round(transport, replayed)
        await transport.aclose()

    asyncio.run(run())
    assert sdk.clients[0].results["toolu_1"]["content"] == [
        {"type": "text", "text": "Took one."},
        {"type": "text", "text": "Image from the tool call above."},
        {"type": "image", "data": "iVBORw0KGgo=", "mimeType": "image/png"},
    ]


def test_tool_names_round_trip_through_the_mcp_prefix(monkeypatch, tmp_path):
    long_name = "mcp__my-server__" + "x" * 60
    sdk = _install(monkeypatch, tmp_path)
    transport = ClaudeAgentTransport(model = "haiku")
    server = transport._server(sdk, [_tool("get.time"), _tool(long_name)])
    short = [tool.name for tool in server["tools"]]

    assert all(len(f"mcp__studio__{name}") <= 64 for name in short)
    assert transport._names == {
        f"mcp__studio__{short[0]}": "get.time",
        f"mcp__studio__{short[1]}": long_name,
    }


def test_cancel_mid_reply_interrupts_and_closes(monkeypatch, tmp_path):
    sdk = _install(monkeypatch, tmp_path, turns = [[_text("Thinking about"), ("wait",)]])

    async def run():
        cancel = threading.Event()
        transport = ClaudeAgentTransport(model = "haiku")
        lines = []
        async for line in transport.stream(
            messages = [{"role": "user", "content": "hi"}],
            tools = None,
            tool_choice = None,
            cancel_event = cancel,
        ):
            lines.append(line)
            cancel.set()
        return lines

    lines = asyncio.run(run())
    assert _content(lines) == "Thinking about"
    client = sdk.clients[0]
    assert client.interrupted is True
    assert client.disconnected is True


def test_closing_while_a_tool_waits_cancels_it_and_ends_claude(monkeypatch, tmp_path):
    sdk = _install(
        monkeypatch,
        tmp_path,
        turns = [_tool_turn([("toolu_1", "get_time", {"city": "Oslo"})]) + [("wait",)]],
    )

    async def run():
        transport = ClaudeAgentTransport(model = "haiku")
        await _round(transport, [{"role": "user", "content": "Time?"}])
        await asyncio.sleep(0.01)
        waiting = transport._calls["toolu_1"]
        await transport.aclose()
        return waiting.result()

    assert asyncio.run(run())["is_error"] is True
    client = sdk.clients[0]
    assert client.interrupted is True
    assert client.disconnected is True


def test_withdrawn_tools_are_answered_inside_the_session(monkeypatch, tmp_path):
    sdk = _install(
        monkeypatch,
        tmp_path,
        turns = [
            _tool_turn([("toolu_1", "get_time", {"city": "Rome"})])
            + _tool_turn([("toolu_2", "get_time", {"city": "Oslo"})])
            + [_text("Rome is at noon."), _result()]
        ],
    )
    messages = [{"role": "user", "content": "Time?"}]

    async def run():
        transport = ClaudeAgentTransport(model = "haiku")
        first = await _round(transport, messages)
        second = await _round(
            transport, _replay(messages, first, "12:00"), tools = None, tool_choice = "none"
        )
        await transport.aclose()
        return second

    second = asyncio.run(run())
    assert _calls(second) == []
    assert _content(second) == "Rome is at noon."
    assert sdk.clients[0].results["toolu_2"]["is_error"] is True


def test_tool_choice_none_offers_no_tools(monkeypatch, tmp_path):
    sdk = _install(monkeypatch, tmp_path, turns = [[_result()]])

    async def run():
        transport = ClaudeAgentTransport(model = "haiku")
        await _round(transport, [{"role": "user", "content": "hi"}], tool_choice = "none")
        await transport.aclose()

    asyncio.run(run())
    assert not hasattr(sdk.clients[0].options, "mcp_servers")


def test_invalid_arguments_are_left_to_claude_code(monkeypatch, tmp_path):
    # The SDK answers a call that fails its schema itself, so its handler never runs.
    turn = [s for s in _tool_turn([("toolu_1", "get_time", {"town": "Rome"})]) if s[0] != "call"]
    _install(monkeypatch, tmp_path, turns = [turn + [_text("Sorry."), _result()]])

    async def run():
        transport = ClaudeAgentTransport(model = "haiku")
        lines = await _round(transport, [{"role": "user", "content": "Time?"}])
        await transport.aclose()
        return lines

    lines = asyncio.run(run())
    assert _calls(lines) == []
    assert _content(lines) == "Sorry."


@pytest.fixture
def executed(monkeypatch):
    calls = []

    def _execute(name, arguments, **kwargs):
        calls.append((name, arguments))
        return f"RESULT<{name}:{arguments['city']}>"

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: False)
    return calls


def _loop(transport, max_calls = 25):
    async def run():
        lines = close_after(
            stream_with_studio_tools(
                transport,
                run = ToolLoopRun(
                    messages = [{"role": "user", "content": "Time in Paris and Rome?"}],
                    session_id = "s1",
                    thread_id = "t1",
                    model = "haiku",
                ),
                policy = ToolLoopPolicy(
                    tools = TOOLS,
                    max_calls = max_calls,
                    timeout = 300,
                    permission_mode = "off",
                    confirm_calls = False,
                    bypass_permissions = False,
                    rag_scope = None,
                ),
                cancel_event = threading.Event(),
            ),
            transport,
        )
        return [line async for line in lines]

    return asyncio.run(run())


def test_studio_loop_runs_claude_tool_calls(monkeypatch, tmp_path, executed):
    sdk = _install(
        monkeypatch,
        tmp_path,
        turns = [
            _tool_turn([("toolu_1", "get_time", {"city": "Paris"})])
            + _tool_turn([("toolu_2", "get_time", {"city": "Rome"})])
            + [_text("Both are at noon."), _result()]
        ],
    )
    lines = _loop(ClaudeAgentTransport(model = "haiku"))

    assert executed == [("get_time", {"city": "Paris"}), ("get_time", {"city": "Rome"})]
    ended = [p["tool_call_id"] for p in _payloads(lines) if p.get("type") == "tool_end"]
    assert ended == ["toolu_1", "toolu_2"]
    assert "Both are at noon." in _content(lines)
    client = sdk.clients[0]
    assert client.results["toolu_2"]["content"][0]["text"].startswith("RESULT<get_time:Rome>")
    assert client.disconnected is True


def test_studio_loop_budget_ends_with_an_error_result(monkeypatch, tmp_path, executed):
    sdk = _install(
        monkeypatch,
        tmp_path,
        turns = [
            _tool_turn([("toolu_1", "get_time", {"city": "Paris"})])
            + _tool_turn([("toolu_2", "get_time", {"city": "Rome"})])
            + [_text("Only Paris."), _result()]
        ],
    )
    lines = _loop(ClaudeAgentTransport(model = "haiku"), max_calls = 1)

    assert executed == [("get_time", {"city": "Paris"})]
    assert "Only Paris." in _content(lines)
    client = sdk.clients[0]
    assert client.results["toolu_2"]["is_error"] is True
    assert client.disconnected is True
