# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the model is TOLD about its environment under Full access.

permission_mode='full' folds to bypass_permissions=True, which the tool loops
pass on as disable_sandbox=True: the static analysis, the command blocklist and
the rlimit pre-exec are all skipped and absolute host paths resolve. The
python/terminal schemas used to be module constants describing the sandboxed
run regardless, and the tool nudge never mentioned the mode at all, so the model
was told it was isolated from a machine it could in fact read. Asked "are you
able to see the files on my laptop", it answered "no, I operate in a sandboxed
environment" without ever calling a tool.

These tests pin the two halves of the fix: the schemas swap under Full access,
and the nudge states the mode so the model checks instead of guessing. Every
other mode keeps the sandboxed wording verbatim.
"""

import asyncio
import sys

import pytest

from core.inference import tools
from core.inference.tools import (
    ALL_TOOLS,
    PYTHON_TOOL,
    PYTHON_TOOL_FULL_ACCESS,
    TERMINAL_TOOL,
    TERMINAL_TOOL_FULL_ACCESS,
    apply_full_access_tool_descriptions,
)
from models.inference import ChatCompletionRequest, ChatCountTokensRequest
from routes.inference import (
    _TOOL_FULL_ACCESS_TIP,
    _build_tool_action_nudge,
    _select_request_tools,
)


def _desc(tool: dict) -> str:
    return tool["function"]["description"]


def _named(tools: list[dict], name: str) -> dict:
    return next(t for t in tools if t["function"]["name"] == name)


# ── Schemas ───────────────────────────────────────────────────────────


def test_sandboxed_descriptions_are_unchanged():
    """The default pair is what every existing importer gets."""
    assert "in a sandbox" in _desc(PYTHON_TOOL)
    assert "do not exist" in _desc(PYTHON_TOOL) or "Windows" in _desc(PYTHON_TOOL)
    assert "return stdout/stderr" in _desc(TERMINAL_TOOL)


@pytest.mark.parametrize(
    "tool",
    [PYTHON_TOOL_FULL_ACCESS, TERMINAL_TOOL_FULL_ACCESS],
    ids = ["python", "terminal"],
)
def test_full_access_descriptions_drop_the_isolation_claim(tool):
    description = _desc(tool)
    assert "in a sandbox" not in description
    # The one claim that is outright false with the sandbox off.
    assert "do not exist" not in description
    assert "sandbox is disabled" in description
    assert "machine running Unsloth Studio" in description
    # The remote modes (--secure / -H 0.0.0.0, README) put the tools on the host
    # serving Studio, not on the device the user is looking at, so the prompt
    # must not claim the two are the same.
    assert "user's own machine" not in description
    # The workdir really is still the per-session dir in bypass mode
    # (_build_bypass_env repoints HOME/TMPDIR/TEMP/TMP at it), so the relative
    # path advice and the download-link note both have to survive.
    assert "persists for this conversation" in description
    assert "download link" in description


def test_full_access_schemas_keep_name_and_parameters():
    """Only the description changes: a differing name or schema would break the
    dispatcher and every caller that matches on them."""
    for sandboxed, full in (
        (PYTHON_TOOL, PYTHON_TOOL_FULL_ACCESS),
        (TERMINAL_TOOL, TERMINAL_TOOL_FULL_ACCESS),
    ):
        assert full["type"] == sandboxed["type"]
        assert full["function"]["name"] == sandboxed["function"]["name"]
        assert full["function"]["parameters"] == sandboxed["function"]["parameters"]


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_the_substitutions_land_on_every_platform(monkeypatch, platform):
    """The module constants are built once for the host platform, so a Linux
    runner would never exercise the Windows branch. Rebuild the note per
    platform and re-derive, which is also the guard against a rewording of
    _build_sandbox_paths_note silently turning the substitutions into no-ops:
    the sandboxed markers would survive into the result below."""
    monkeypatch.setattr(sys, "platform", platform)
    sandboxed = "Execute Python code in a sandbox and return stdout/stderr." + (
        tools._build_sandbox_paths_note()
    )
    full = tools._to_full_access(sandboxed)

    assert full != sandboxed
    assert "in a sandbox" not in full
    assert "do not exist" not in full
    assert "sandbox is disabled" in full
    assert "do resolve" in full
    assert "user's own machine" not in full
    # _build_bypass_env keeps _SANDBOX_SITE_DIR on PYTHONPATH, so sitecustomize
    # still heals these onto the workdir under Full access. A blanket "absolute
    # paths resolve" would have the model report a write that went elsewhere.
    if platform != "win32":
        assert "/mnt/data" in full
        assert "redirected into the working directory" in full
    # True in both modes, so untouched.
    assert "persists for this conversation" in full
    assert "download link" in full
    if platform == "win32":
        assert "You are on Windows" in full


def test_python_full_access_description_still_omits_the_shell():
    """Same reason as the sandboxed one: naming a shell there points a model at
    subprocess/os.system instead of the terminal tool."""
    assert "shell" not in _desc(PYTHON_TOOL_FULL_ACCESS).lower()


def test_terminal_full_access_keeps_the_shell_note():
    """The shell note is platform-derived and applies in either mode; dropping
    it on Windows brings back the cmd/bash confusion it exists to prevent."""
    for marker in ("cmd, not bash", "bash (Git for Windows)"):
        assert (marker in _desc(TERMINAL_TOOL)) == (marker in _desc(TERMINAL_TOOL_FULL_ACCESS))


def test_swap_leaves_other_tools_alone_and_does_not_mutate():
    before = list(ALL_TOOLS)
    swapped = apply_full_access_tool_descriptions(list(ALL_TOOLS))
    assert _named(swapped, "python") is PYTHON_TOOL_FULL_ACCESS
    assert _named(swapped, "terminal") is TERMINAL_TOOL_FULL_ACCESS
    for name in ("web_search", "render_html", "search_knowledge_base"):
        assert _named(swapped, name) is _named(ALL_TOOLS, name)
    # The module global is shared across requests, so the swap must not touch it.
    assert ALL_TOOLS == before
    assert _desc(_named(ALL_TOOLS, "python")) == _desc(PYTHON_TOOL)


def test_swap_is_a_no_op_without_the_sandboxed_builtins():
    tools = [t for t in ALL_TOOLS if t["function"]["name"] == "web_search"]
    assert apply_full_access_tool_descriptions(tools) is tools
    assert apply_full_access_tool_descriptions([]) == []


# ── Request-level selection ───────────────────────────────────────────


def _select(**payload_kwargs) -> list[dict]:
    payload = ChatCompletionRequest(
        model = "test-model",
        messages = [{"role": "user", "content": "hi"}],
        enable_tools = True,
        enabled_tools = ["python", "terminal", "web_search"],
        stream = True,
        **payload_kwargs,
    )
    return asyncio.run(_select_request_tools(payload, tools_on = True, mcp_allowed = False))


@pytest.mark.parametrize("mode", ["ask", "auto", "off"])
def test_non_full_modes_keep_the_sandboxed_schemas(mode):
    tools = _select(permission_mode = mode)
    assert _desc(_named(tools, "python")) == _desc(PYTHON_TOOL)
    assert _desc(_named(tools, "terminal")) == _desc(TERMINAL_TOOL)


def test_omitted_mode_keeps_the_sandboxed_schemas():
    tools = _select()
    assert _desc(_named(tools, "python")) == _desc(PYTHON_TOOL)


@pytest.mark.parametrize(
    "payload_kwargs",
    [{"permission_mode": "full"}, {"bypass_permissions": True}],
    ids = ["permission_mode", "legacy_bypass_flag"],
)
def test_full_access_selection_swaps_the_schemas(payload_kwargs):
    """Both spellings fold to bypass_permissions=True, so both must swap."""
    tools = _select(**payload_kwargs)
    assert _desc(_named(tools, "python")) == _desc(PYTHON_TOOL_FULL_ACCESS)
    assert _desc(_named(tools, "terminal")) == _desc(TERMINAL_TOOL_FULL_ACCESS)
    assert _named(tools, "web_search") is _named(ALL_TOOLS, "web_search")


# ── Nudge ─────────────────────────────────────────────────────────────

_CODE_TOOLS = [PYTHON_TOOL, TERMINAL_TOOL]
_WEB_ONLY = [t for t in ALL_TOOLS if t["function"]["name"] == "web_search"]


def test_nudge_is_unchanged_without_full_access():
    plain = _build_tool_action_nudge(tools = _CODE_TOOLS, model_name = "test-8B")
    assert "sandbox" not in plain
    assert "code execution" in plain
    assert plain == _build_tool_action_nudge(
        tools = _CODE_TOOLS, model_name = "test-8B", full_access = False
    )


def test_nudge_states_the_environment_under_full_access():
    nudge = _build_tool_action_nudge(tools = _CODE_TOOLS, model_name = "test-8B", full_access = True)
    assert "machine running Unsloth Studio" in nudge
    assert "code sandbox and the approval prompts disabled" in nudge
    # Scoped to the two local tools: execute_tool passes disable_sandbox to
    # python/terminal only, web_search is a network call, and an MCP tool may run
    # on a remote server, so an unqualified "tool calls run here" is wrong when
    # any of those are enabled alongside.
    assert nudge.count("The python and terminal tools run on") == 1
    # Studio can be served remotely, so the tools' host is not necessarily the
    # device in front of the user.
    assert "not always the device the user is viewing this on" in nudge
    # The actual reported failure: the model asserted isolation instead of
    # checking, so the nudge has to redirect that guess to a tool call.
    assert "check with a tool call" in nudge


def test_full_access_only_returns_the_sentence_alone():
    """The Codex studio-tools path has never carried the general tool nudge, so
    it takes the Full access sentence without the date or the base guidance."""
    only = _build_tool_action_nudge(
        tools = _CODE_TOOLS, model_name = "test-8B", full_access = True, full_access_only = True
    )
    assert only == _TOOL_FULL_ACCESS_TIP
    assert "The current date is" not in only
    assert "Tools are available when they materially improve" not in only


@pytest.mark.parametrize(
    "kwargs",
    [{"full_access": False}, {"full_access": True, "tools": _WEB_ONLY}],
    ids = ["not_full_access", "no_code_tool"],
)
def test_full_access_only_is_empty_when_it_does_not_apply(kwargs):
    tools = kwargs.pop("tools", _CODE_TOOLS)
    assert (
        _build_tool_action_nudge(tools = tools, model_name = "test-8B", full_access_only = True, **kwargs)
        == ""
    )


def test_full_access_tip_needs_a_code_tool():
    """web_search alone runs nothing locally, so the sandbox sentence would be
    noise (and false)."""
    nudge = _build_tool_action_nudge(tools = _WEB_ONLY, model_name = "test-8B", full_access = True)
    assert "machine running Unsloth Studio" not in nudge
    assert nudge == _build_tool_action_nudge(tools = _WEB_ONLY, model_name = "test-8B")


def test_full_access_tip_needs_tools_at_all():
    assert _build_tool_action_nudge(tools = [], model_name = "test-8B", full_access = True) == ""


# ── Token count parity ────────────────────────────────────────────────


def _count_request(**kwargs) -> ChatCountTokensRequest:
    return ChatCountTokensRequest(
        model = "test-model",
        messages = [{"role": "user", "content": "hi"}],
        enable_tools = True,
        enabled_tools = ["python", "terminal"],
        **kwargs,
    )


def test_count_request_reads_the_flag_when_omitted():
    """The count route reaches for payload.bypass_permissions unconditionally,
    so the field has to exist rather than arrive via extra='allow'."""
    assert _count_request().bypass_permissions is None


@pytest.mark.parametrize(
    "kwargs",
    [{"permission_mode": "full"}, {"bypass_permissions": True}],
    ids = ["permission_mode", "legacy_bypass_flag"],
)
def test_count_request_folds_full_access(kwargs):
    request = _count_request(**kwargs)
    assert request.bypass_permissions is True
    assert request.permission_mode == "full"


@pytest.mark.parametrize("mode", ["ask", "auto", "off"])
def test_count_request_leaves_other_modes_alone(mode):
    assert _count_request(permission_mode = mode).bypass_permissions is None


def test_count_request_selection_matches_the_completion():
    """The whole point of carrying the flag: the counted tool list is the one
    the completion will render."""
    counted = asyncio.run(
        _select_request_tools(
            _count_request(permission_mode = "full"), tools_on = True, mcp_allowed = False
        )
    )
    assert _desc(_named(counted, "python")) == _desc(PYTHON_TOOL_FULL_ACCESS)
