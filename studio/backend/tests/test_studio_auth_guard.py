# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio's own credentials are not tool input.

The CLI caches the raw API key at ``$STUDIO_HOME/auth/.cli_api_key_<stem>_<digest>`` so a relaunch
reuses it instead of minting another. Tool subprocesses run as the backend's OS user, so that file's
0600 mode is no boundary against them: a provider that emits ``cat <auth dir>/.cli_api_key_*`` would
get a live bearer replayed into its next inference request, and that key authenticates the terminal
and python tools in turn.

Two independent guards, so neither one has to be complete:
  - the executors refuse a call that names the auth directory, in every permission mode (Bypass
    Permissions included, unlike the command blocklist);
  - an API key or desktop secret that reaches a result by any other route is masked on the
    model-bound copy only, leaving the tool card the user is looking at untouched. The remaining
    credentials in that directory (bootstrap password, llama stream key, auth.db) have no
    recognisable shape to mask, so for those the executor refusal is the guard.
"""

from __future__ import annotations

import os

import pytest

from core.inference import tools
from core.inference.mcp_client import MCP_TOOL_PREFIX
from core.inference.tool_loop_controller import redact_studio_credentials, strip_result_for_model
from core.inference.tools import is_high_risk_tool_call

_SESSION = "studio-auth-guard-session"
# Shaped like the real thing (auth/storage.py: API_KEY_PREFIX + secrets.token_hex(16)), but minted here.
_FAKE_KEY = "sk-unsloth-" + "a1b2c3d4" * 4

_CREDENTIAL_COMMANDS = (
    "cat ~/.unsloth/studio/auth/.cli_api_key_cli_99bb88401742",
    "ls -a ~/.unsloth/studio/auth",
    "cat /home/u/.unsloth/studio/auth/.bootstrap_password",
    "sqlite3 /home/u/.unsloth/studio/auth/auth.db .dump",
    # `unsloth start` caches the coding-agent keys here, in the same directory.
    "cat /home/u/.unsloth/studio/auth/agent_api_key.json",
    "cat agent_api_key.json",
)
_CREDENTIAL_CODE = (
    "from pathlib import Path\n"
    "print(Path('/home/u/.unsloth/studio/auth/.cli_api_key_cli_99bb88401742').read_text())",
    "print(open('/home/u/.unsloth/studio/auth/.desktop_secret').read())",
    "import os\nprint(os.listdir('/home/u/.unsloth/studio/auth'))",
)


@pytest.fixture(autouse = True)
def _no_subprocess(monkeypatch):
    """A refusal must happen before anything is spawned, so make a spawn an outright failure."""

    def _boom(*args, **kwargs):
        raise AssertionError("a blocked call reached the subprocess layer")

    monkeypatch.setattr(tools.subprocess, "Popen", _boom)
    monkeypatch.setattr(tools.subprocess, "run", _boom)


@pytest.mark.parametrize("command", _CREDENTIAL_COMMANDS)
@pytest.mark.parametrize("disable_sandbox", [False, True])
def test_terminal_refuses_studio_credentials(command, disable_sandbox):
    result = tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = disable_sandbox)
    assert result == tools._STUDIO_CREDENTIAL_BLOCKED


@pytest.mark.parametrize("code", _CREDENTIAL_CODE)
@pytest.mark.parametrize("disable_sandbox", [False, True])
def test_python_refuses_studio_credentials(code, disable_sandbox):
    result = tools._python_exec(code, None, 30, _SESSION, disable_sandbox = disable_sandbox)
    assert result == tools._STUDIO_CREDENTIAL_BLOCKED


def test_refusal_names_no_path_and_no_value():
    result = tools._bash_exec(_CREDENTIAL_COMMANDS[0], None, 30, _SESSION)
    assert ".cli_api_key" not in result
    assert "auth" in result  # it still says what it refused, in prose


@pytest.mark.parametrize(
    "command",
    ["grep -rn auth src/", "cat src/auth.py", "ls -la", "python -m pytest tests/ -q"],
)
def test_ordinary_commands_are_not_blocked(command):
    # Not asserting they run (the fixture forbids spawning); only that the credential guard, which
    # returns before anything else, lets them past.
    assert not tools._references_studio_credential(command)
    assert tools._STUDIO_CREDENTIAL_BLOCKED != command


def test_a_custom_studio_home_is_covered(monkeypatch, tmp_path):
    auth_dir = tmp_path / "studio-home" / "auth"
    auth_dir.mkdir(parents = True)
    key_file = auth_dir / ".cli_api_key_cli_99bb88401742"
    key_file.write_text(_FAKE_KEY + "\n", encoding = "utf-8")

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio-home"))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        # The basename is enough on its own; the resolved root covers a directory listing, which
        # carries no Studio-specific filename at all.
        assert tools._references_studio_credential(f"cat {key_file}")
        assert tools._references_studio_credential(f"ls -a {auth_dir}")
        assert tools._bash_exec(f"cat {key_file}", None, 30, _SESSION) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        )
    finally:
        tools._studio_auth_markers_cache = None


def test_the_auth_dir_named_relative_to_a_cd_into_the_studio_root(monkeypatch, tmp_path):
    # `cd <studio home> && ls -a auth` carries no absolute auth path and no Studio-specific
    # basename, so it is only recognisable through the root the same command entered.
    import utils.paths.storage_roots as roots

    home = tmp_path / "studio-home"
    monkeypatch.setattr(roots, "auth_root", lambda: home / "auth")
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        assert tools._references_studio_credential(f"cd {home} && ls -a auth")
        assert tools._references_studio_credential(f'cd "{home}"; ls auth/')
        # ...while merely naming the studio root is ordinary work: reading your own Studio logs,
        # or grepping them for the word, must not turn into a refusal.
        assert not tools._references_studio_credential(f"grep -rn auth {home}/logs/studio.log")
        assert not tools._references_studio_credential(f"ls {home}/models && grep -c auth app.log")
        assert not tools._references_studio_credential(f"cd {home} && ls models")
    finally:
        tools._studio_auth_markers_cache = None


def test_mcp_call_at_the_auth_dir_is_refused():
    name = f"{MCP_TOOL_PREFIX}fs__read_file"
    args = {"path": "~/.unsloth/studio/auth/.cli_api_key_cli_99bb88401742"}
    assert tools._mcp_arguments_reference_studio_credential(args) is True
    # Nested and list-shaped arguments reach the same answer.
    assert (
        tools._mcp_arguments_reference_studio_credential(
            {"files": [{"path": "/home/u/.unsloth/studio/auth/.bootstrap_password"}]}
        )
        is True
    )
    # Prose stays prose: an issue body that mentions the file is text, not a read.
    assert (
        tools._mcp_arguments_reference_studio_credential(
            {"body": "the key is cached in .cli_api_key_cli_99bb88401742"}
        )
        is False
    )
    # And auto mode would have paused on it even before the refusal.
    assert is_high_risk_tool_call(name, args) is True


def test_auto_mode_prompts_on_studio_credential_reads():
    assert is_high_risk_tool_call("terminal", {"command": _CREDENTIAL_COMMANDS[0]}) is True
    assert is_high_risk_tool_call("python", {"code": _CREDENTIAL_CODE[0]}) is True


def test_a_key_in_a_result_never_reaches_the_model():
    result = f"UNSLOTH_API_KEY={_FAKE_KEY}\nok\n"
    masked = strip_result_for_model(result, "terminal")
    assert _FAKE_KEY not in masked
    assert masked == "UNSLOTH_API_KEY=[redacted]\nok\n"


def test_redaction_is_idempotent_and_leaves_other_output_alone():
    once = redact_studio_credentials(f"a {_FAKE_KEY} b {_FAKE_KEY}")
    assert redact_studio_credentials(once) == once
    assert _FAKE_KEY not in once
    plain = "model sk-unsloth loaded; 12 files; sk-openai-style-name; desktop-app starts"
    assert redact_studio_credentials(plain) == plain
    # The desktop credential (`desktop-` + token_urlsafe(48)) goes the same way.
    desktop = "desktop-" + "Ab3_-x9Z" * 8
    assert desktop not in redact_studio_credentials(f"secret={desktop}\n")
    # A key cut in half by the window fit is still credential material, so it is masked too, while
    # prose about the prefix itself reads through.
    assert redact_studio_credentials(_FAKE_KEY[:24]) == "[redacted]"
    assert redact_studio_credentials("keys look like sk-unsloth-<hex>") == (
        "keys look like sk-unsloth-<hex>"
    )


def test_the_envelope_split_still_sees_a_suffix_only_strip():
    # _split_frontend_suffix subtracts the stripped body from the text, so masking has to stay off
    # that path or the envelope-aware truncation silently degrades to cutting the whole result.
    body = f"line with {_FAKE_KEY}"
    suffix = '\n__FILES__:[{"name": "a.txt", "size": 3}]'
    kept, envelope = tools._split_frontend_suffix(body + suffix, "terminal")
    assert kept == body
    assert envelope == suffix


def test_edit_file_cannot_reach_the_auth_dir_even_with_the_sandbox_off():
    result = tools._edit_file(
        {
            "path": "~/.unsloth/studio/auth/.bootstrap_password",
            "edits": [{"old_string": "a", "new_string": "b"}],
        },
        session_id = _SESSION,
        disable_sandbox = True,
    )
    assert result == tools._STUDIO_CREDENTIAL_BLOCKED


def test_the_user_facing_tool_card_is_masked_too():
    # Originally the mask was on the model path only, so the card kept the raw text. That is the
    # replay hole: the frontend PERSISTS this payload and serializes the stored value back into a
    # role="tool" message on the user's next turn, so the key was hidden from the continuation and
    # then sent to the provider one turn later. The card shows the user their own key, so masking
    # it costs them nothing and closes the stored copy as well.
    from core.inference.tool_loop_controller import ToolCallCompletion, ToolCallDecision

    decision = ToolCallDecision(
        action = "execute",
        tool_name = "terminal",
        arguments = {"command": "echo $UNSLOTH_API_KEY"},
        tool_call_id = "call-1",
    )
    completion = ToolCallCompletion(
        decision = decision, result = _FAKE_KEY, is_error = False, executed = True
    )
    assert _FAKE_KEY not in completion.tool_end_payload()["result"]
    assert _FAKE_KEY not in completion.tool_end_event()["result"]
    assert _FAKE_KEY not in completion.tool_message()["content"]


def test_the_cli_cache_path_still_matches_what_the_guard_blocks():
    # Ties the guard to the CLI's actual filename, so renaming one without the other fails here.
    import importlib.util

    spec = importlib.util.find_spec("unsloth_cli.commands.studio")
    if spec is None or not spec.origin or not os.path.isfile(spec.origin):
        pytest.skip("unsloth_cli is not importable from the backend test env")
    with open(spec.origin, encoding = "utf-8") as f:
        source = f.read()
    assert 'CLI_API_KEY_FILE_PREFIX = ".cli_api_key_"' in source
    assert 'BOOTSTRAP_PASSWORD_FILE = ".bootstrap_password"' in source


def test_tool_end_payload_masks_a_leaked_studio_key():
    """The frontend persists this payload and replays it into a role="tool" message on the user's
    next turn, so redacting only the in-memory continuation hides the key this turn and sends it
    the next one."""
    from core.inference.tool_loop_controller import ToolLoopController

    key = "sk-unsloth-" + "A" * 48
    controller = ToolLoopController(tools = [{"type": "function", "function": {"name": "terminal"}}])
    decision = controller.prepare_call(
        {"id": "call_0", "function": {"name": "terminal", "arguments": '{"command":"env"}'}}
    )
    completion = controller.record_result(decision, f"UNSLOTH_API_KEY={key}")

    assert key not in completion.tool_end_payload()["result"]
    assert key not in completion.tool_end_event()["result"]
    assert key not in completion.model_message()["content"]
