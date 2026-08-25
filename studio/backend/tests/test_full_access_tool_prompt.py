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
import json
import os
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
    _append_to_codex_instructions,
    _build_tool_action_nudge,
    _full_access_tip,
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
    assert "wherever Unsloth Studio is running" in description
    # Docker is a documented deployment, where only mounted paths are visible,
    # so the reach is the Studio process's, not a whole machine's.
    assert "container with only some paths mounted" in description
    # The remote modes (--secure / -H 0.0.0.0, README) put the tools on the host
    # serving Studio, not on the device the user is looking at, so the prompt
    # must not claim the two are the same.
    assert "user's own machine" not in description
    # The workdir really is still the per-session dir in bypass mode
    # (_build_bypass_env repoints HOME at it and TMPDIR/TEMP/TMP just inside it),
    # so the relative path advice and the download-link note both have to survive.
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
@pytest.mark.parametrize("tool_name", ["python", "terminal"])
def test_the_substitutions_land_on_every_platform(monkeypatch, platform, tool_name):
    """The module constants are built once for the host platform, so a Linux
    runner would never exercise the Windows branch. Rebuild the note per
    platform and re-derive, which is also the guard against a rewording of
    _build_sandbox_paths_note silently turning the substitutions into no-ops:
    the sandboxed markers would survive into the result below."""
    monkeypatch.setattr(sys, "platform", platform)
    sandboxed = "Execute Python code in a sandbox and return stdout/stderr." + (
        tools._build_sandbox_paths_note()
    )
    full = tools._to_full_access(sandboxed, tool_name)

    assert full != sandboxed
    assert "in a sandbox" not in full
    assert "do not exist" not in full
    assert "sandbox is disabled" in full
    assert "do resolve" in full
    assert "user's own machine" not in full
    assert "wherever Unsloth Studio is running" in full
    # _build_bypass_env keeps _SANDBOX_SITE_DIR on PYTHONPATH, so sitecustomize
    # still heals these onto the workdir under Full access. A blanket "absolute
    # paths resolve" would have the model report a write that went elsewhere.
    # The clause is per tool on BOTH platforms: the split is the shim, not the OS.
    # sitecustomize is a CPython startup hook, so it patches python (and any
    # python the terminal launches) wherever it runs, while a plain shell gets
    # nothing. Measured: parent exists -> real path; parent missing -> <cwd>/base,
    # unless that name is taken, where it raises.
    if tool_name == "python":
        assert "absolute paths under a directory that exists do resolve" in full
        # Two branches, measured: an absent convention prefix keeps the SUFFIX
        # (/mnt/data/reports/out.csv -> ./reports/out.csv) and overwrites an
        # existing file; any other missing parent keeps only the base name and
        # raises when that name is taken. Describing one as both was wrong.
        assert "the rest of the path is kept relative to the working directory" in full
        assert "replacing any file already sitting there" in full
        assert "only the base name is kept" in full
        # Measured: a DIFFERENT invented path with the same basename raises, but
        # rewriting the SAME invented path is permitted via the remap sidecar.
        assert "fails outright if that name is taken by an unrelated file" in full
        assert "rewriting the same absolute path just replaces" in full
        # Only open/io.open/os.open and the mkdir family are wrapped. Measured:
        # os.rename and os.symlink raise, and shutil.copy writes the rewritten
        # file through open and then raises in copymode.
        # Measured: os.makedirs under a missing parent OUTSIDE the convention
        # prefixes targets the REAL host path, because _makedirs calls _remap only
        # and never the generic fallback, so the two rewrites do NOT cover the same
        # APIs. Inside a prefix _remap still rewrites, so the clause is scoped:
        # makedirs("/mnt/data/reports") with no /mnt/data created ./reports.
        assert "The convention rewrite covers open() and the mkdir calls" in full
        assert "the other covers open() alone" in full
        assert "os.makedirs under a missing parent outside those prefixes" in full
        assert "is not rewritten and attempts the real host path" in full
        assert "missing absolute parent is not rewritten at all" not in full
        assert "shutil.copy can write the rewritten file and still raise" in full
    else:
        assert "absolute paths do resolve as the shell resolves them" in full
        # _build_bypass_env sets PYTHONPATH for the terminal subprocess too, so
        # python launched from a shell command carries the same shim.
        assert "Python you launch from here is the exception" in full
        assert "gets the same rewrites" in full
    # No categorical claim about the convention paths in either: on a host where
    # /mnt/data is a real mount the shim never shadows it, so "not real" is wrong,
    # and the parent-directory rule already covers the absent case.
    # /mnt/data may be named, but only inside the conditional describing what
    # happens while it is ABSENT; a real mount is never shadowed, so no
    # categorical "not real" claim may survive.
    assert "not real there" not in full
    if tool_name == "python":
        assert "when the directory does not exist" in full
    # True in both modes, so untouched.
    assert "persists for this conversation" in full
    assert "download link" in full
    if platform == "win32":
        assert "You are on Windows" in full


def test_python_full_access_description_still_omits_the_shell():
    """Same reason as the sandboxed one: naming a shell there points a model at
    subprocess/os.system instead of the terminal tool."""
    assert "shell" not in _desc(PYTHON_TOOL_FULL_ACCESS).lower()


def test_full_access_drops_the_local_desktop_promise(monkeypatch):
    """The Git Bash branch of the shell note says a detached program opens a
    window on the user's desktop, which only holds while Studio is local. The
    Full access text now says it may be remote or containerized, so the two
    would contradict each other."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\bash.exe")
    note = tools._build_terminal_shell_note()
    assert "opens a window on the user's desktop" in note
    full = tools._to_full_access("X." + note, "terminal")
    assert "opens a window on the user's desktop" not in full
    assert "on that machine's desktop" in full
    # The shell-selection guidance itself has to survive.
    assert "The shell is bash (Git for Windows)" in full


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
    assert "where Unsloth Studio is running" in nudge
    assert "code sandbox and the approval prompts disabled" in nudge
    # Containerized Studio sees only its mounts, so the claim is scoped to what
    # the process can reach rather than to the machine.
    assert "whatever that process can reach" in nudge
    assert "container that mounts only some" in nudge
    # Scoped to the two local tools: execute_tool passes disable_sandbox to
    # python/terminal only, web_search is a network call, and an MCP tool may run
    # on a remote server, so an unqualified "tool calls run here" is wrong when
    # any of those are enabled alongside.
    assert nudge.count("The python and terminal tools run where") == 1


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [
        (["python"], "The python tool runs where"),
        (["terminal"], "The terminal tool runs where"),
        (["python", "terminal"], "The python and terminal tools run where"),
        # Order comes from _LOCAL_CODE_TOOLS, not from the caller's list.
        (["terminal", "python"], "The python and terminal tools run where"),
    ],
    ids = ["python_only", "terminal_only", "both", "reversed"],
)
def test_the_tip_names_only_the_selected_code_tools(enabled, expected):
    """enabled_tools=["python"] leaves terminal out of the request's schemas, so
    naming it would advertise a tool the loop would refuse to run."""
    tools = [t for t in ALL_TOOLS if t["function"]["name"] in enabled]
    nudge = _build_tool_action_nudge(tools = tools, model_name = "test-8B", full_access = True)
    assert expected in nudge
    for absent in {"python", "terminal"} - set(enabled):
        assert f"The {absent} tool runs where" not in nudge
        assert f"and {absent} tools run where" not in nudge
    # Studio can be served remotely, so the tools' host is not necessarily the
    # device in front of the user.
    assert "not necessarily the device the user is viewing this on" in nudge
    # The actual reported failure: the model asserted isolation instead of
    # checking, so the nudge has to redirect that guess to a tool call.
    assert "check with a tool call" in nudge


def test_full_access_only_returns_the_sentence_alone():
    """The Codex studio-tools path has never carried the general tool nudge, so
    it takes the Full access sentence without the date or the base guidance."""
    only = _build_tool_action_nudge(
        tools = _CODE_TOOLS, model_name = "test-8B", full_access = True, full_access_only = True
    )
    assert only == _full_access_tip(["python", "terminal"])
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
    assert "where Unsloth Studio is running" not in nudge
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


# ── Codex studio-tools instructions ───────────────────────────────────


def test_codex_instructions_skip_a_developer_message():
    """_responses_input folds only `system` turns into the Responses
    instructions and drops every other role bar user/assistant/tool, so a nudge
    appended to a `developer` turn would never reach the model. `developer` is an
    accepted ChatMessage role, so this shape is reachable."""
    messages = [
        {"role": "developer", "content": "house style"},
        {"role": "user", "content": "hi"},
    ]
    out = _append_to_codex_instructions(messages, "NUDGE")
    assert out[0] == {"role": "system", "content": "NUDGE"}
    assert out[1] == messages[0]
    assert messages[0]["content"] == "house style"


def test_codex_instructions_extend_an_existing_system_message():
    messages = [{"role": "system", "content": "base"}, {"role": "user", "content": "hi"}]
    out = _append_to_codex_instructions(messages, "NUDGE")
    assert out[0]["content"] == "base\n\nNUDGE"
    assert len(out) == 2
    assert messages[0]["content"] == "base"


def test_codex_instructions_are_a_no_op_without_an_addition():
    messages = [{"role": "user", "content": "hi"}]
    assert _append_to_codex_instructions(messages, "") is messages


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


def _sandbox_site_dir():
    from pathlib import Path
    return Path(tools.__file__).resolve().parent / "sandbox_site"


# hasattr, not the win32 marker above: a marker's argument is evaluated when the decorator
# is applied, so os.geteuid() runs at import on a platform that has no geteuid and takes the
# whole module down at collection, every test in it, not just this one.
@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX directory modes")
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason = "root ignores a mode-500 directory",
)
def test_the_mkdir_clause_promises_an_attempt_not_a_created_directory(tmp_path):
    """The unrewritten mkdir path is an attempt, and the clause may not promise more.

    Full access keeps _SANDBOX_SITE_DIR on PYTHONPATH, so ``os.makedirs`` is
    wrapped -- but ``_makedirs`` calls ``_remap`` alone, so outside a convention
    prefix nothing is rewritten and the real host path is what the syscall gets.
    A process that cannot create that path (a non-root write under a root-owned
    directory, a read-only mount) raises instead, creating nothing anywhere. The
    subprocess below is that measurement, not an assumption about os.makedirs:
    the same shim the model runs under, a mode-500 parent, and a check that
    neither the host path nor a workdir fallback appeared.

    The clause is held to the standard the rest of this comment block already
    keeps -- see the convention-prefix branch, which names /mnt/data only inside
    a conditional and never categorically.
    """
    import subprocess

    workdir = tmp_path / "work"
    readonly = tmp_path / "readonly"
    workdir.mkdir()
    readonly.mkdir()
    readonly.chmod(0o500)
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json, os, sys\n"
                "target = sys.argv[1] + '/missing/child'\n"
                "try:\n"
                "    os.makedirs(target)\n"
                "    outcome = 'created'\n"
                "except OSError as exc:\n"
                "    outcome = type(exc).__name__\n"
                "print(json.dumps({'outcome': outcome, 'host': os.path.exists(target),\n"
                "                  'workdir': sorted(os.listdir('.'))}))\n",
                str(readonly),
            ],
            cwd = workdir,
            env = {**os.environ, "PYTHONPATH": str(_sandbox_site_dir())},
            capture_output = True,
            text = True,
            timeout = 120,
        )
    finally:
        readonly.chmod(0o700)
    assert probe.returncode == 0, probe.stderr
    measured = json.loads(probe.stdout.strip().splitlines()[-1])
    assert measured["outcome"] == "PermissionError", measured
    assert measured["host"] is False, "nothing was created on the host"
    assert measured["workdir"] == [], "and nothing fell back into the working directory"

    full = tools._to_full_access(
        "Execute Python code in a sandbox and return stdout/stderr."
        + tools._build_sandbox_paths_note(),
        "python",
    )
    assert (
        "really does create it" not in full
    ), "the clause promises a directory the filesystem may refuse to create"
    assert "outside those prefixes is not rewritten" in full
    assert "attempts the real host path" in full


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX directory modes")
@pytest.mark.skipif(os.path.exists("/mnt/data"), reason = "a real mount is never shadowed")
def test_the_mkdir_clause_is_scoped_to_parents_outside_the_convention_prefixes(tmp_path):
    """Inside a convention prefix, makedirs IS rewritten, so the clause cannot be flat.

    ``_makedirs`` calls ``_remap``, and ``_remap``'s convention branch keeps the
    suffix under the working directory before it reaches the generic fallback. So
    ``/mnt/data/reports`` has a missing absolute parent and still lands in the
    workdir. A clause saying makedirs under a missing absolute parent is not
    rewritten at all would have the model report a host path for a directory
    sitting in its working directory, which is the one thing the closing sentence
    asks it not to do.
    """
    import subprocess

    workdir = tmp_path / "work"
    workdir.mkdir()
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, os\n"
            "target = '/mnt/data/reports'\n"
            "try:\n"
            "    os.makedirs(target)\n"
            "    outcome = 'created'\n"
            "except OSError as exc:\n"
            "    outcome = type(exc).__name__\n"
            "print(json.dumps({'outcome': outcome, 'host': os.path.exists(target),\n"
            "                  'workdir': sorted(os.listdir('.'))}))\n",
        ],
        cwd = workdir,
        env = {**os.environ, "PYTHONPATH": str(_sandbox_site_dir())},
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert probe.returncode == 0, probe.stderr
    measured = json.loads(probe.stdout.strip().splitlines()[-1])
    assert measured["outcome"] == "created", measured
    assert measured["host"] is False, "the absent prefix is not created on the host"
    assert measured["workdir"] == ["reports"], measured
