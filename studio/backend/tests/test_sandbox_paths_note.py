# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Accuracy guards for the sandbox-paths note appended to the python/terminal
tool descriptions (#7242).

The note must not misdirect the model with claims that are false for the real
sandbox: (1) the sandbox is not network-isolated -- the python tool should fetch
from public hosts (github.com / huggingface.co / pypi.org), so it must not claim
it "cannot reach ... remote hosts" at all, nor overstate the host check as a hard
wall that makes private/arbitrary hosts impossible; (2) a project sandbox is a
shared per-project directory, so a new thread can inherit files from prior
threads -- the note must not claim the workdir "starts empty" or "persists only
for this conversation".
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.tools import (
    PYTHON_TOOL,
    PYTHON_TOOL_BYPASS,
    TERMINAL_TOOL,
    TERMINAL_TOOL_BYPASS,
    _SANDBOX_PATHS_NOTE,
    _SANDBOX_PATHS_NOTE_BYPASS,
    _bash_exec,
    apply_bypass_tool_notes,
)


def test_note_does_not_claim_full_network_isolation():
    lowered = _SANDBOX_PATHS_NOTE.lower()
    # The old absolute claim is false for the python tool (it does reach egress).
    assert "cannot reach other machines or remote hosts" not in lowered
    # It must not overstate the host check as an enforced hard boundary:
    # _sandbox_preexec leaves networking on and the AST check only inspects
    # literal hosts, so a dynamically built request to a private host still runs.
    assert "only from a fixed allowlist" not in lowered
    assert "arbitrary addresses" not in lowered
    # It should still steer to public sources so the model fetches valid URLs.
    assert "public sources" in lowered
    assert "github.com" in lowered and "huggingface.co" in lowered
    assert "private" in lowered


def test_note_does_not_claim_project_sandbox_starts_empty():
    lowered = _SANDBOX_PATHS_NOTE.lower()
    # Project sandboxes are shared across threads, so these are inaccurate.
    assert "starts empty" not in lowered
    assert "persists only for this conversation" not in lowered
    # It should acknowledge that project files may already be present.
    assert "project" in lowered


def test_note_does_not_claim_local_files_are_inaccessible():
    lowered = _SANDBOX_PATHS_NOTE.lower()
    # On a locally hosted Studio the child runs on the host with no filesystem
    # isolation on this branch (Landlock is a separate change), and cat is an
    # auto-safe terminal command, so an exact local path is readable; the note
    # must not claim otherwise. It should frame the workdir as the default work
    # location instead.
    assert "cannot see the user's own computer" not in lowered
    assert "default location for your work" in lowered


def test_note_is_appended_to_both_tool_descriptions():
    assert PYTHON_TOOL["function"]["description"].endswith(_SANDBOX_PATHS_NOTE)
    assert TERMINAL_TOOL["function"]["description"].endswith(_SANDBOX_PATHS_NOTE)


def test_note_scopes_network_block_to_the_terminal_not_all_shell_commands():
    # The terminal leaves the network namespace intact and does not block git/pip,
    # so the note must not claim shell network is fully blocked; it names the
    # commands that are blocked (curl / wget) and attributes the host allowlist to
    # the python tool.
    lowered = _SANDBOX_PATHS_NOTE.lower()
    assert "shell network commands are blocked" not in lowered
    assert "curl" in lowered and "wget" in lowered
    assert "python tool is intended to fetch" in lowered


def test_note_distinguishes_attachments_from_sandbox_uploads():
    # Chat/project document uploads land in the RAG store, not the sandbox workdir,
    # so the note must not tell the model that uploaded files appear via `ls`.
    lowered = _SANDBOX_PATHS_NOTE.lower()
    assert "was uploaded" not in lowered
    assert "or that is uploaded" not in lowered
    assert "retrieved separately" in lowered
    assert "attach" in lowered


def test_blocked_network_command_message_gates_code_fallback_on_tool_availability():
    # A blocked curl/wget must not tell the model the whole sandbox is offline, and
    # must not name a specific tool (e.g. python) as the remedy: when only the
    # terminal is enabled the python tool is absent from the schema, so an
    # unconditional "fetch it from Python code" instruction invites an invalid tool
    # call. The fallback is gated on a code-execution tool being enabled this turn.
    msg = _bash_exec("curl https://raw.githubusercontent.com/foo/bar/main/x.py").lower()
    assert "blocked command" in msg
    assert "cannot reach other machines or remote hosts" not in msg
    assert "from python code instead" not in msg
    assert "if a code-execution tool is enabled this turn" in msg
    assert "github.com" in msg


def test_blocked_network_command_message_scopes_claim_to_the_command():
    # The block is by command name; the terminal keeps its network namespace and
    # does not block git/pip, so `git clone http://<private-host>/repo` can still
    # reach a private host. The message must not assert the destination itself is
    # unreachable, and must attribute the block to the command by name.
    msg = _bash_exec("wget http://10.0.0.5/internal/repo.tar.gz").lower()
    assert "by name" in msg
    assert "not reachable" not in msg
    assert "are not reachable" not in msg
    assert "hosts are not reachable" not in msg


def test_blocked_network_command_message_does_not_recommend_chat_upload():
    # Chat attachments land in the RAG store, not the sandbox workdir, so telling
    # the user to upload the file to chat is a dead end. The message must instead
    # point at an accessible path or placing the file in the working directory.
    msg = _bash_exec("curl https://raw.githubusercontent.com/foo/bar/main/x.py").lower()
    assert "upload" not in msg
    assert "working directory" in msg or "path the sandbox can read" in msg


def test_bypass_note_drops_the_curl_wget_allowlist_restriction():
    # Under Bypass Permissions _python_exec/_bash_exec skip the safety analysis and
    # the curl/wget blocklist, so egress is not limited to the allowlist and those
    # downloads work. The bypass note must not tell the model they are blocked.
    lowered = _SANDBOX_PATHS_NOTE_BYPASS.lower()
    assert "curl" not in lowered and "wget" not in lowered
    assert "allowlist" not in lowered
    assert "internet access is limited" not in lowered
    # It keeps the workdir-default framing and the "not a copy of the host" guard.
    assert "default location for your work" in lowered
    assert "do not assume files elsewhere on the host are already here" in lowered
    # The bypass note is a strict prefix+suffix of the default note (only the
    # network sentence is removed), so the rest of the guidance is unchanged.
    assert _SANDBOX_PATHS_NOTE_BYPASS != _SANDBOX_PATHS_NOTE
    assert "curl" in _SANDBOX_PATHS_NOTE.lower()


def test_bypass_tool_variants_use_the_bypass_note():
    assert PYTHON_TOOL_BYPASS["function"]["description"].endswith(_SANDBOX_PATHS_NOTE_BYPASS)
    assert TERMINAL_TOOL_BYPASS["function"]["description"].endswith(_SANDBOX_PATHS_NOTE_BYPASS)
    # Same tool names/parameters as the default variants; only the note differs.
    assert PYTHON_TOOL_BYPASS["function"]["name"] == PYTHON_TOOL["function"]["name"]
    assert TERMINAL_TOOL_BYPASS["function"]["name"] == TERMINAL_TOOL["function"]["name"]
    assert PYTHON_TOOL_BYPASS["function"]["parameters"] == PYTHON_TOOL["function"]["parameters"]


def test_apply_bypass_tool_notes_swaps_only_python_and_terminal():
    tools = [
        {"function": {"name": "web_search", "description": "search"}},
        PYTHON_TOOL,
        TERMINAL_TOOL,
    ]
    swapped = apply_bypass_tool_notes(tools)
    by_name = {t["function"]["name"]: t for t in swapped}
    assert "curl" not in by_name["python"]["function"]["description"].lower()
    assert "curl" not in by_name["terminal"]["function"]["description"].lower()
    # Unrelated tools pass through unchanged (same object).
    assert by_name["web_search"] is tools[0]
    # The shared module globals are not mutated by the swap.
    assert "curl" in PYTHON_TOOL["function"]["description"].lower()
    # A tool list without python/terminal is returned unchanged (same object).
    plain = [{"function": {"name": "web_search", "description": "search"}}]
    assert apply_bypass_tool_notes(plain) is plain


def test_bypass_code_execution_nudge_drops_the_limited_internet_claim():
    from routes.inference import _TOOL_CODE_TIP, _TOOL_CODE_TIP_BYPASS

    lowered = _TOOL_CODE_TIP_BYPASS.lower()
    assert "internet access is limited" not in lowered
    # Keeps the workdir-default framing and the exact-path guidance.
    assert "default" in lowered and "location for your work" in lowered
    assert "give an exact path" in lowered
    # The default nudge still carries the restriction for sandboxed sessions.
    assert "internet access is limited" in _TOOL_CODE_TIP.lower()


def test_code_execution_nudge_does_not_deny_local_file_access():
    # On this no-Landlock branch the child runs on the host with only cwd set, so an
    # exact local path the user supplies is readable. The code-execution nudge must
    # frame the workdir as the default work location, not assert the user's own
    # computer is inaccessible, and it must still allow an exact path.
    from routes.inference import _TOOL_CODE_TIP

    lowered = _TOOL_CODE_TIP.lower()
    assert "cannot access the user's own computer" not in lowered
    assert "default" in lowered and "location for your work" in lowered
    assert "give an exact path" in lowered


def test_sandbox_disabled_treats_permission_mode_full_as_unsandboxed():
    # Both agent loops fold permission_mode "full" into bypass_permissions=True and
    # pass disable_sandbox=bypass_permissions at execution, so "full" runs python /
    # terminal unsandboxed (skips _check_code_safety and the curl/wget blocklist).
    # The tool notes and action nudge key off the effective flag so they always
    # match what executes, even if the model-layer fold is ever refactored away.
    from types import SimpleNamespace

    from routes.inference import _sandbox_disabled

    # Explicit bypass -> unsandboxed.
    assert _sandbox_disabled(SimpleNamespace(bypass_permissions=True, permission_mode="ask")) is True
    # "full" even without bypass set on the object -> still unsandboxed (decoupled
    # from the model-layer fold, so the notes never overclaim a live restriction).
    assert _sandbox_disabled(SimpleNamespace(bypass_permissions=False, permission_mode="full")) is True
    # A genuinely sandboxed request keeps the restrictive notes.
    assert _sandbox_disabled(SimpleNamespace(bypass_permissions=False, permission_mode="ask")) is False
    # Missing attributes must not raise (defensive getattr).
    assert _sandbox_disabled(SimpleNamespace()) is False
