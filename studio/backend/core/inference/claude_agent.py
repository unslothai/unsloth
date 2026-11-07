# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Claude Agent connection: the user's own `claude` CLI, driven through the public Claude Agent SDK."""

import asyncio
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

from utils.paths.storage_roots import cache_root
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

PROVIDER_TYPE = "claude_agent"
SDK_MISSING = (
    "Install the Claude Agent SDK in the Unsloth environment: "
    f'uv pip install --python "{sys.executable}" --no-binary claude-agent-sdk "claude-agent-sdk==0.2.178"'
)
CLI_MISSING = "Install Claude Code, then run `claude` in a terminal and sign in."
SIGN_IN = "Run `claude` in a terminal and sign in."
# verbatim_prompts needs it: older CLIs only log a warning, then run "/" commands typed in chat.
MIN_CLI_VERSION = (2, 1, 248)
CLI_OUTDATED = (
    "Claude Code {} is too old for Studio. Run `claude update` to get 2.1.248 or later, "
    "then check again."
)
TOOL_SERVER = "studio"
TOOL_PREFIX = f"mcp__{TOOL_SERVER}__"
# Used when Claude Code cannot report a model's window; every current Claude model has at least this.
FALLBACK_CONTEXT_WINDOW = 200_000
_DISCOVERY_BUDGET_S = 20.0
_RUNTIME_NOTE = (
    "<runtime_info>You are running in Unsloth Studio through the Claude Agent SDK. "
    "No tools are available in this chat. Mention this only when asked.</runtime_info>"
)
_RUNTIME_NOTE_TOOLS = (
    "<runtime_info>You are running in Unsloth Studio through the Claude Agent SDK. "
    "Your only tools are Unsloth Studio's, named mcp__studio__*; Studio may ask the user "
    "before one runs. Mention this only when asked.</runtime_info>"
)
_ISOLATION_ENV = {
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_ENABLE_AWAY_SUMMARY": "0",
    "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "1",
    "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "1",
    "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK": "1",
    # Deferred tools need ToolSearch, which is off with the other built-ins.
    "ENABLE_TOOL_SEARCH": "false",
    # Studio already caps tool results.
    "MAX_MCP_OUTPUT_TOKENS": "200000",
    # Simple mode skips the subscription login.
    "CLAUDE_CODE_SIMPLE": "0",
}
EFFORTS = ("low", "medium", "high", "xhigh", "max")
_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
_SIGNED_OUT = ("not logged in", "/login", "oauth token has expired", "invalid api key")

_catalog: dict[str, Any] = {"version": None, "models": {}}
_cli_versions: dict[tuple[str, float], Optional[str]] = {}


def _read_cli_version(cli: str) -> Optional[str]:
    try:
        key = (cli, os.path.getmtime(cli))
    except OSError:
        key = (cli, 0.0)
    if key not in _cli_versions:
        try:
            out = subprocess.run(
                [cli, "-v"],
                capture_output = True,
                text = True,
                timeout = 15,
                **windows_hidden_subprocess_kwargs(),
            ).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        match = re.match(r"\s*(\d+\.\d+\.\d+)", out or "")
        _cli_versions[key] = match.group(1) if match else None
    return _cli_versions[key]


async def cli_version(cli: str) -> Optional[str]:
    # Initialize reports claude_code_version only on newer CLIs, so ask `claude -v`.
    return await asyncio.to_thread(_read_cli_version, cli)


def outdated_message(version: Optional[str]) -> Optional[str]:
    if version and tuple(int(part) for part in version.split(".")) < MIN_CLI_VERSION:
        return CLI_OUTDATED.format(version)
    return None


def _sdk():
    try:
        return importlib.import_module("claude_agent_sdk")
    except ImportError:
        return None


def _options(sdk, cli: str, **extra):
    workdir = cache_root() / "claude-agent"
    workdir.mkdir(parents = True, exist_ok = True)
    return sdk.ClaudeAgentOptions(
        cli_path = cli,
        cwd = str(workdir),
        env = dict(_ISOLATION_ENV),
        setting_sources = [],
        settings = json.dumps({"disableAllHooks": True}),
        strict_mcp_config = True,
        tools = [],
        permission_mode = "dontAsk",
        extra_args = {"no-session-persistence": None},
        **extra,
    )


def remember_models(info: dict[str, Any]) -> None:
    version = info.get("claude_code_version")
    previous = _catalog["models"] if version == _catalog["version"] else {}
    models = {}
    for model in info.get("models") or []:
        if not isinstance(model, dict) or model.get("value") in (None, "default"):
            continue
        entry = dict(model)
        entry["context_window"] = (previous.get(model["value"]) or {}).get("context_window")
        models[model["value"]] = entry
    _catalog["version"] = version
    _catalog["models"] = models


def _model_entry(model: str) -> Optional[dict[str, Any]]:
    models = _catalog["models"]
    return models.get(model) or next(
        (entry for entry in models.values() if entry.get("resolvedModel") == model), None
    )


def _model_object(entry: dict[str, Any]) -> dict[str, Any]:
    efforts = entry.get("supportedEffortLevels") if entry.get("supportsEffort") else None
    return {
        "id": entry["value"],
        "display_name": entry.get("displayName") or entry["value"],
        # An alias like "opus" follows the newest model; a full id is pinned.
        "alias": entry.get("resolvedModel") not in (None, entry["value"]),
        "efforts": [effort for effort in efforts or [] if effort in EFFORTS],
        "thinking_off": bool(entry.get("supportsThinkingOff")),
        "context_window": entry.get("context_window"),
    }


def clamp_effort(model: str, effort: Optional[str]) -> Optional[str]:
    if effort not in EFFORTS:
        return None
    entry = _model_entry(model)
    if entry is None:
        return effort
    levels = _model_object(entry)["efforts"]
    if not levels:
        return None
    lower = [level for level in EFFORTS[: EFFORTS.index(effort) + 1] if level in levels]
    return lower[-1] if lower else levels[0]


def thinking_config(model: str, enable_thinking: Optional[bool]) -> dict[str, Any]:
    entry = _model_entry(model)
    if enable_thinking is False and entry is not None and entry.get("supportsThinkingOff"):
        return {"type": "disabled"}
    return {"type": "adaptive", "display": "summarized"}


async def _discover_windows(client, entries: list[dict[str, Any]]) -> None:
    deadline = time.monotonic() + _DISCOVERY_BUDGET_S
    for entry in entries:
        try:
            await asyncio.wait_for(
                client.set_model(entry["value"]), timeout = deadline - time.monotonic()
            )
            usage = await asyncio.wait_for(
                client.get_context_usage(), timeout = deadline - time.monotonic()
            )
        except asyncio.TimeoutError:
            return
        except Exception:
            continue
        # rawMaxTokens is the model's window; maxTokens can be Claude Code's compaction limit.
        window = usage.get("rawMaxTokens") if isinstance(usage, dict) else None
        if isinstance(window, int) and window > 0:
            entry["context_window"] = window


async def _probe(
    sdk,
    cli: str,
    discover: Optional[list[str]] = None,
) -> dict[str, Any]:
    client = sdk.ClaudeSDKClient(_options(sdk, cli))
    try:
        await asyncio.wait_for(client.connect(), timeout = 30)
        info = await client.get_server_info() or {}
        remember_models(info)
        if discover is not None:
            await _discover_windows(
                client,
                [
                    entry
                    for value, entry in _catalog["models"].items()
                    if entry.get("context_window") is None
                    and (value in discover if discover else entry.get("resolvedModel") != value)
                ],
            )
        return info
    finally:
        await client.disconnect()


async def ensure_catalog(sdk, cli: str) -> None:
    if not _catalog["models"]:
        try:
            await _probe(sdk, cli)
        except Exception:
            pass


def _account_label(account: dict[str, Any]) -> Optional[str]:
    plan = account.get("subscriptionType")
    if isinstance(plan, str) and plan:
        plan = plan.replace("_", " ").strip()
        return plan if plan.lower().startswith("claude") else f"Claude {plan.title()}"
    if account.get("apiKeySource"):
        return "API key"
    provider = account.get("apiProvider")
    return provider if isinstance(provider, str) and provider != "firstParty" else None


async def claude_agent_status() -> dict[str, Any]:
    """Starts `claude` without sending a prompt, so no model call is made."""
    status: dict[str, Any] = {
        "ready": False,
        "version": None,
        "account": None,
        "models": [],
        "message": None,
        "problem": None,
    }
    sdk = _sdk()
    if sdk is None:
        status.update(message = SDK_MISSING, problem = "sdk_missing")
        return status
    cli = shutil.which("claude")
    if cli is None:
        status.update(message = CLI_MISSING, problem = "cli_missing")
        return status
    version = await cli_version(cli)
    outdated = outdated_message(version)
    if outdated:
        status.update(version = version, message = outdated, problem = "failed")
        return status
    try:
        info = await _probe(sdk, cli, discover = [])
    except (sdk.ClaudeSDKError, asyncio.TimeoutError) as exc:
        status.update(message = f"Claude Code did not start: {exc or 'timed out'}", problem = "failed")
        return status
    account = info.get("account") or {}
    status["version"] = info.get("claude_code_version") or version
    status["models"] = [_model_object(entry) for entry in _catalog["models"].values()]
    status["ready"] = account.get("tokenSource", "none") != "none" or bool(
        account.get("apiKeySource") or account.get("subscriptionType")
    )
    if status["ready"]:
        status["account"] = _account_label(account) or "Claude account"
    else:
        status.update(message = SIGN_IN, problem = "signed_out")
    return status


async def claude_agent_models() -> list[dict[str, Any]]:
    if not _catalog["models"]:
        await claude_agent_status()
    return [_model_object(entry) for entry in _catalog["models"].values()]


async def claude_agent_context_window(model: str) -> int:
    entry = _model_entry(model)
    if entry is None or entry.get("context_window") is None:
        sdk = _sdk()
        cli = shutil.which("claude")
        if sdk is not None and cli is not None:
            try:
                await _probe(sdk, cli, discover = [model])
            except Exception:
                pass
        entry = _model_entry(model)
    return (entry or {}).get("context_window") or FALLBACK_CONTEXT_WINDOW


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def image_url(part: Any) -> Optional[str]:
    if not isinstance(part, dict) or part.get("type") != "image_url":
        return None
    image = part.get("image_url")
    url = image.get("url") if isinstance(image, dict) else image
    return url if isinstance(url, str) else ""


def image_block(url: str) -> dict[str, Any]:
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = header[5:].split(";")[0].strip().lower()
        if media_type in _IMAGE_TYPES and data and header.endswith(";base64"):
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
        return {
            "type": "text",
            "text": "[An attached image was left out: Claude takes JPEG, PNG, GIF and WebP images.]",
        }
    if url.startswith(("https://", "http://")):
        return {"type": "image", "source": {"type": "url", "url": url}}
    return {"type": "text", "text": "[An attached image was left out: its URL is not supported.]"}


def _history_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    parts = [content] if isinstance(content, str) else []
    for part in content if isinstance(content, list) else []:
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
            parts.append(part["text"])
        elif image_url(part) is not None:
            # Earlier images cannot ride in the system prompt.
            parts.append("[image]")
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) else None
        if isinstance(function, dict):
            arguments = function.get("arguments") or "{}"
            parts.append(f'<tool_call name="{function.get("name")}">{arguments}</tool_call>')
    text = "\n".join(part for part in parts if part.strip())
    if message.get("role") == "tool" and text:
        return f'<tool_result name="{message.get("name") or "tool"}">\n{text}\n</tool_result>'
    return text


def _user_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    blocks = []
    for part in content if isinstance(content, list) else []:
        if isinstance(part, dict) and part.get("type") == "text":
            if isinstance(part.get("text"), str) and part["text"].strip():
                blocks.append({"type": "text", "text": part["text"]})
        elif (url := image_url(part)) is not None:
            blocks.append(image_block(url))
    return blocks


def build_prompt(
    messages: list[dict[str, Any]], tools: bool = False
) -> tuple[str, Optional[list[dict[str, Any]]]]:
    """The SDK takes one user turn, so the system text and earlier turns go to the appended system prompt."""
    system = [_RUNTIME_NOTE_TOOLS if tools else _RUNTIME_NOTE]
    turns = []
    for message in messages:
        role = message.get("role")
        if role in ("system", "developer"):
            text = _text(message.get("content"))
            if text.strip():
                system.append(text)
        elif role in ("user", "assistant", "tool"):
            turns.append(message)
    last = next((i for i in range(len(turns) - 1, -1, -1) if turns[i]["role"] == "user"), None)
    # A trailing plain assistant turn asks for a continuation, which Claude Code cannot do.
    if last is None or (turns[-1]["role"] == "assistant" and not turns[-1].get("tool_calls")):
        return "\n\n".join(system), None
    blocks = _user_blocks(turns[last].get("content"))
    for message in turns[last + 1 :]:
        text = _history_text(message)
        if text:
            blocks.append({"type": "text", "text": text})
    if not blocks:
        return "\n\n".join(system), None
    history = []
    for message in turns[:last]:
        text = _history_text(message)
        if text:
            role = message["role"]
            history.append(text if role == "tool" else f"<{role}>\n{text}\n</{role}>")
    if history:
        body = "\n".join(history)
        system.append(
            "<conversation_history>\nThe earlier turns of this conversation, oldest first:\n"
            f"{body}\n</conversation_history>"
        )
    return "\n\n".join(system), blocks


def failure(
    result: Any,
    assistant_error: Optional[str],
    rate_limit: Any = None,
) -> tuple[int, str]:
    text = (result.result or "\n".join(result.errors or [])).strip()
    status = result.api_error_status
    if assistant_error == "authentication_failed" or any(s in text.lower() for s in _SIGNED_OUT):
        return 401, f"Claude Code is signed out or its login expired. {SIGN_IN}"
    rejected = rate_limit if getattr(rate_limit, "status", None) == "rejected" else None
    if assistant_error == "rate_limit" or rejected is not None or status == 429:
        window = getattr(rejected, "rate_limit_type", None)
        message = "Your Claude usage limit was reached" + (
            f" ({window.replace('_', ' ')} limit)." if window else "."
        )
        resets_at = getattr(rejected, "resets_at", None)
        if isinstance(resets_at, (int, float)):
            reset = datetime.fromtimestamp(resets_at, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            message += f" It resets at {reset}."
        elif text:
            message += f" Claude Code says: {text}"
        return 429, message
    if assistant_error == "billing_error":
        lead = text.rstrip(".") if text else "Claude Code reported a billing problem"
        return 402, f"{lead}. Check your Claude plan and extra usage settings."
    if assistant_error == "server_error" or (status or 0) >= 500:
        status = status if (status or 0) >= 500 else 502
        reason = "Anthropic is overloaded" if status == 529 else f"Anthropic returned HTTP {status}"
        return status, f"{reason} and Claude Code ran out of retries. Try again shortly."
    if assistant_error == "invalid_request":
        return status or 400, text or "Claude rejected the request."
    return status or 502, text or "Claude Code could not finish its reply."


async def stream_claude_agent(
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
) -> AsyncGenerator[str, None]:
    from core.inference.claude_agent_transport import ClaudeAgentTransport
    transport = ClaudeAgentTransport(
        model = model,
        max_tokens = max_tokens,
        reasoning_effort = reasoning_effort,
        enable_thinking = enable_thinking,
    )
    try:
        async for line in transport.stream(
            messages = messages, tools = None, tool_choice = None, cancel_event = None
        ):
            yield line
    finally:
        await transport.aclose()
