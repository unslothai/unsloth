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
    "cd ~/.unsloth/studio/auth && cat agent_api_key.json",
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
    [
        "grep -rn auth src/",
        "cat src/auth.py",
        "ls -la",
        "python -m pytest tests/ -q",
        # A bare filename is a string, not a read: `print('agent_api_key.json')` was refused.
        "echo agent_api_key.json",
        "grep -rn agent_api_key.json src/",
    ],
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
        # The root has to END where it matched, or continue into `auth`. A path that merely STARTS
        # with it is a different directory: without the boundary both of these were refused in
        # every permission mode.
        assert not tools._references_studio_credential(f"cd {home}/models && grep auth README")
        assert not tools._references_studio_credential(f"cd {home}-backup && ls auth")
        # ...and the boundary must not cost the real thing.
        assert tools._references_studio_credential(f"cd {home}/auth && ls")
    finally:
        tools._studio_auth_markers_cache = None


def test_equivalent_spellings_of_the_auth_path_are_all_covered(monkeypatch, tmp_path):
    # The OS opens every one of these as the same file, so matching only the tidiest spelling left
    # the others walking straight past the guard.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        for spelling in (
            f"{home}/auth/auth.db",
            f"{home}/bin/../auth/auth.db",
            f"{home}/./auth/auth.db",
            f"{home}//auth//auth.db",
            f"{home}/a/b/../../auth/auth.db",
            str(home).replace("/", "\\") + "\\.\\auth\\auth.db",
            # Mixed separators with a doubled slash, which is what a Windows runner produces. The
            # canonical form collapsed `//` only in the slash-normalised candidate, built from the
            # raw text, so this spelling matched neither until the collapse moved into the fold.
            str(home).replace("/", "\\") + "//auth//auth.db",
        ):
            assert tools._references_studio_credential(f"cat {spelling}"), spelling
        # Cancelling `..` must not invent a match that was never there.
        assert not tools._references_studio_credential(f"cat {home}/auth-backup/../notes.txt")
    finally:
        tools._studio_auth_markers_cache = None


def test_the_environment_variable_spelling_of_the_studio_home(monkeypatch, tmp_path):
    # `cat $STUDIO_HOME/auth/auth.db` reaches the same file as the resolved path: both studio-home
    # variables survive into the tool subprocess environment, so the shell expands this for real.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        for spelling in (
            'cat "$STUDIO_HOME/auth/auth.db"',
            "cat ${UNSLOTH_STUDIO_HOME}/auth/auth.db",
            "ls -a $UNSLOTH_STUDIO_HOME/auth",
            r"type %UNSLOTH_STUDIO_HOME%\auth\.desktop_secret",
            r"Get-Content $env:STUDIO_HOME\auth\auth.db",
            'cd "$STUDIO_HOME" && ls auth',
            'sqlite3 "$UNSLOTH_STUDIO_HOME"/auth/auth.db "select jwt_secret from auth_user"',
        ):
            assert tools._references_studio_credential(spelling), spelling
        # Naming the variable without going into the auth directory stays ordinary work.
        assert not tools._references_studio_credential("export STUDIO_HOME=/tmp/studio")
        assert not tools._references_studio_credential("ls $STUDIO_HOME/models")
        assert not tools._references_studio_credential("grep -rn auth $STUDIO_HOME/logs/studio.log")
    finally:
        tools._studio_auth_markers_cache = None


def test_a_studio_home_whose_name_contains_a_space(monkeypatch, tmp_path):
    # macOS installs under "Application Support" put a space in the path, and a shell spells that
    # with a backslash escape. Read as a separator it split the directory name and slipped past.
    home = tmp_path / "Studio Data"
    (home / "auth").mkdir(parents = True)
    escaped = str(home).replace(" ", "\\ ")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        # A shell concatenates adjacent fragments, so the quote sits INSIDE the path here and no
        # marker could span it until the candidate was dequoted.
        assert tools._references_studio_credential(f'sqlite3 "{home}"/auth/auth.db "select 1"')
        assert tools._references_studio_credential(f"cat '{home}'/auth/.desktop_secret")
        assert not tools._references_studio_credential(f'cat "{home}"/models/notes.txt')
        assert tools._references_studio_credential(f"cat {escaped}/auth/auth.db")
        assert tools._references_studio_credential(f"cd {escaped} && ls -a auth")
        assert tools._references_studio_credential(f'cat "{home}/auth/auth.db"')
        assert not tools._references_studio_credential(f"cat {escaped}/authors/notes.txt")
    finally:
        tools._studio_auth_markers_cache = None


def test_traversal_out_of_the_sandbox_into_the_auth_dir(monkeypatch, tmp_path):
    # The tool cwd is <studio home>/sandbox/<session>, a sibling of the auth directory, so
    # `open('../../auth/auth.db')` reads the protected database while naming neither the directory
    # nor a credential basename. Bypass Permissions skips the blocklist, so only this guard is left.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "auth" / "auth.db").write_text("not a real database", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        for call, args in (
            (tools._bash_exec, "cat ../../auth/auth.db"),
            (tools._python_exec, "print(open('../../auth/auth.db').read())"),
        ):
            assert call(args, None, 30, _SESSION, disable_sandbox = True) == (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), args
        # Traversal that lands anywhere else is ordinary work: it is not refused. (The suite blocks
        # real subprocesses, so the refusal string is what is checked, not the output.)
        for ordinary in ("cat ../notes.txt", "ls ../../models", "cp ../a.txt ../../b.txt"):
            assert (
                tools._bash_exec(
                    ordinary,
                    None,
                    30,
                    _SESSION,
                    disable_sandbox = True,
                )
                != tools._STUDIO_CREDENTIAL_BLOCKED
            ), ordinary
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
    # The token is hex, so a name that only shares the prefix is left alone. On the alphanumeric
    # pattern this repository's own `sk-unsloth-internal-workflow` came back as
    # `[redacted]-workflow` in source listings and test output.
    for not_a_key in (
        "container name sk-unsloth-internal-workflow here",
        "sk-unsloth-zzzzzzzzzz",
    ):
        assert redact_studio_credentials(not_a_key) == not_a_key


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


def _stream_chunks(chunks):
    """Feed `chunks` through the live-stream masking exactly as the generator does."""
    from core.inference.tool_loop_controller import redact_studio_credentials
    from core.inference.tool_stream_exec import _hold_back_partial_secret

    carry = ""
    out = []
    for chunk in chunks:
        combined = carry + chunk
        split = _hold_back_partial_secret(combined)
        out.append(redact_studio_credentials(combined[:split]))
        carry = combined[split:]
    out.append(redact_studio_credentials(carry))
    return "".join(out)


def test_a_key_is_masked_in_the_live_stream_however_it_is_chunked():
    # tool_end is not the first thing the card paints. The live tool_output chunks are, so a key
    # that is only masked at the end is still rendered in full and persisted that way. Found by
    # screenshotting the card rather than by a unit test: the tool_end payload said [redacted]
    # while the output block above it showed the key.
    key = "sk-unsloth-0123456789abcdef0123456789abcdef"
    for chunks in (
        [key],
        [key[:15], key[15:]],  # split mid-token
        ["sk-unslo", key[8:]],  # split mid-PREFIX, which no per-chunk regex can see
        ["out ", key[:20], key[20:], " tail"],
        [("a" * 500) + key],
    ):
        rendered = _stream_chunks(chunks)
        assert key not in rendered, chunks
        assert "[redacted]" in rendered, chunks

    # Held-back text that turns out to be ordinary must still be emitted, not swallowed.
    assert _stream_chunks(["hello ", "sk-unslo"]) == "hello sk-unslo"
    assert _stream_chunks(["plain output\n"]) == "plain output\n"


def test_the_auth_path_assembled_through_a_shell_variable(monkeypatch, tmp_path):
    # Bypass Permissions keeps STUDIO_HOME in the child env, so the shell resolves
    # `r=$STUDIO_HOME; sqlite3 "$r/auth/auth.db"` to the protected database while the literal text
    # names neither the directory nor a credential basename. auth.db carries no `sk-unsloth-`
    # prefix either, so the result redactor cannot mask the JWT secret on the way back out.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    sandbox = home / "sandbox" / "sess1"
    sandbox.mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setenv("STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        for command in (
            'r=$STUDIO_HOME; sqlite3 "$r/auth/auth.db" "select jwt_secret from auth_user"',
            "r=$STUDIO_HOME; cat $r/auth/auth.db",
            f"h={home}; cat $h/auth/auth.db",
            'sqlite3 "$STUDIO_HOME/auth/auth.db"',
        ):
            assert tools._references_studio_credential_here(command, str(sandbox)), command

        # Naming the studio home is ordinary work, and so is any other assignment.
        for command in (
            "r=$STUDIO_HOME; ls $r/models",
            "d=/tmp; cat $d/notes.txt",
            "echo $HOME",
            "git commit -m fix",
        ):
            assert not tools._references_studio_credential_here(command, str(sandbox)), command
    finally:
        tools._studio_auth_markers_cache = None


def test_a_python_path_built_in_pieces_is_still_the_auth_dir(monkeypatch, tmp_path):
    # `os.path.join("..", "..", "auth", "auth.db")` names the protected database in pieces, so the
    # text scan sees four ordinary strings and no path at all, and Bypass Permissions has already
    # skipped the code analyzer. auth.db carries no prefix the result mask can key on either.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        for code in (
            'import os\nprint(open(os.path.join("..", "..", "auth", "auth.db")).read())',
            'from pathlib import Path\nprint((Path("..") / ".." / "auth" / "auth.db").read_text())',
            # as_posix(), not str(): a Windows path inside a python string literal turns its
            # separators into escapes (\U, \A, \t), so the snippet stops being the path it names.
            f'import os\nprint(open(os.path.join("{home.as_posix()}", "auth", "auth.db")).read())',
        ):
            assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), code
        # A path built the same way that lands anywhere else is ordinary work.
        for code in (
            'import os\nprint(os.path.join("..", "data", "x.csv"))',
            'import os\nprint(os.path.join("src", "auth", "views.py"))',
            'print("auth")',
        ):
            assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) != (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), code
    finally:
        tools._studio_auth_markers_cache = None


def test_edit_file_is_checked_on_the_resolved_target(monkeypatch, tmp_path):
    # The raw argument can be relative, and the resolve joins it to the sandbox workdir, a sibling
    # of the auth directory. Bypass Permissions also lifts containment, and the receipt echoes a
    # window of the file back, so an edit there is a read.
    home = tmp_path / "studio-home"
    agent_dir = home / "auth" / "agents" / "opencode"
    agent_dir.mkdir(parents = True)
    (agent_dir / "opencode.json").write_text("{}", encoding = "utf-8")
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        assert (
            tools._edit_file(
                {
                    "path": "../../auth/agents/opencode/opencode.json",
                    "edits": [{"old_string": "{}", "new_string": "{ }"}],
                },
                _SESSION,
                disable_sandbox = True,
            )
            == tools._STUDIO_CREDENTIAL_BLOCKED
        )
        # An ordinary relative edit in the sandbox still goes through.
        assert "Created" in tools._edit_file(
            {"path": "notes.txt", "edits": [{"old_string": "", "new_string": "hi"}]},
            _SESSION,
            disable_sandbox = True,
        )
    finally:
        tools._studio_auth_markers_cache = None


def test_a_cd_earlier_in_the_command_moves_what_a_relative_path_means(monkeypatch, tmp_path):
    # `cd ../..` from the session sandbox lands on the Studio root, and the `auth/auth.db` that
    # follows is then the protected database under a name that matches nothing on its own. Resolving
    # the traversal and the later relative path INDEPENDENTLY against the original cwd misses it:
    # neither half reaches the auth directory, only the composition does.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    sandbox = home / "sandbox" / "sess1"
    sandbox.mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        for command in (
            "cd ../..; python -c 'import sqlite3; sqlite3.connect(\"auth/auth.db\")'",
            "cd ../.. && cat auth/auth.db",
            "cd ../../ && cat auth/.cli_api_key_cli_99bb88401742",
            f"cd {home} && cat auth/auth.db",
        ):
            assert tools._references_studio_credential_here(command, str(sandbox)), command

        # Moving around is ordinary, and an `auth` directory the user owns DEEPER in the sandbox is
        # still theirs: after `cd subdir` that path is sandbox/subdir/auth, not Studio's.
        for command in (
            "cd ../.. && ls models",
            "cd ../.. && cat README.md",
            "cd subdir && cat auth/config.json",
            "cd build && make",
        ):
            assert not tools._references_studio_credential_here(command, str(sandbox)), command
    finally:
        tools._studio_auth_markers_cache = None


def test_a_chain_of_cds_that_ends_inside_the_auth_dir(monkeypatch, tmp_path):
    # `cd ../..; cd auth; sqlite3 auth.db` never writes a path with a separator in it, so every
    # token in it reads as an ordinary filename and only the walked directory itself gives it away.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        for command in (
            'cd ../..; cd auth; sqlite3 auth.db "select jwt_secret from auth_user"',
            "cd ../.. && cd auth && cat .cli_api_key_cli_1",
        ):
            assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), command
        for ordinary in ("cd ../..; cd models; ls", "cd .. && cd data && cat x.csv"):
            assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), ordinary
    finally:
        tools._studio_auth_markers_cache = None


def test_traversal_through_the_proc_cwd_symlink(monkeypatch, tmp_path):
    # The kernel resolves /proc/self/cwd to the session sandbox BEFORE applying the `..` that
    # follows, so a lexical normpath reads this as /proc/self/auth/auth.db and misses the file the
    # call actually opens.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        for code in (
            "print(open('/proc/self/cwd/../../auth/auth.db').read())",
            "import sqlite3\nprint(sqlite3.connect('/proc/self/cwd/../../auth/auth.db'))",
        ):
            assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), code
        assert (
            tools._bash_exec(
                "cat /proc/self/cwd/../../auth/.desktop_secret",
                None,
                30,
                _SESSION,
                disable_sandbox = True,
            )
            == tools._STUDIO_CREDENTIAL_BLOCKED
        )
        # Reading procfs itself is ordinary work.
        for ordinary in ("cat /proc/self/status", "ls /proc/self/cwd"):
            assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), ordinary
    finally:
        tools._studio_auth_markers_cache = None


def test_padding_a_command_with_cds_does_not_spend_the_walk(monkeypatch, tmp_path):
    # The walk counted `cd` commands, so eight no-op ones filled the budget and the two that
    # entered the auth directory after them were never looked at.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        padded = ("cd .; " * 8) + 'cd ../..; cd auth; sqlite3 auth.db "select 1"'
        assert tools._bash_exec(padded, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        )
        distinct = (
            "cd a; cd b; cd c; cd d; cd e; cd f; cd g; cd h; "
            "cd ../..; cd ../../..; cd auth; cat .cli_api_key_cli_1"
        )
        assert tools._bash_exec(distinct, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        )
        assert (
            tools._bash_exec(
                "cd .; cd models; ls",
                None,
                30,
                _SESSION,
                disable_sandbox = True,
            )
            != tools._STUDIO_CREDENTIAL_BLOCKED
        )
    finally:
        tools._studio_auth_markers_cache = None


def test_the_shell_pid_spelling_of_the_proc_cwd_symlink(monkeypatch, tmp_path):
    # `$$` is the shell's own PID, expanded before the path is opened, so it names the same symlink
    # a literal number does.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        assert (
            tools._bash_exec(
                "cat /proc/$$/cwd/../../auth/auth.db",
                None,
                30,
                _SESSION,
                disable_sandbox = True,
            )
            == tools._STUDIO_CREDENTIAL_BLOCKED
        )
        assert (
            tools._python_exec(
                "print(open('/proc/$$/cwd/../../auth/auth.db').read())",
                None,
                30,
                _SESSION,
                disable_sandbox = True,
            )
            == tools._STUDIO_CREDENTIAL_BLOCKED
        )
        assert (
            tools._bash_exec(
                "echo $$",
                None,
                30,
                _SESSION,
                disable_sandbox = True,
            )
            != tools._STUDIO_CREDENTIAL_BLOCKED
        )
    finally:
        tools._studio_auth_markers_cache = None


def test_a_python_path_rooted_in_the_studio_home_variable(monkeypatch, tmp_path):
    # `os.environ["UNSLOTH_STUDIO_HOME"] + "/auth/auth.db"` folds to a dynamic piece plus the rest.
    # Bypass mode keeps that variable in the child env, so the dynamic piece has a known value and
    # substituting it is not a guess.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setenv("STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        for code in (
            'import os, sqlite3\np = os.environ["UNSLOTH_STUDIO_HOME"] + "/auth/auth.db"\n'
            "print(sqlite3.connect(p))",
            'import os\nprint(open(os.getenv("STUDIO_HOME") + "/auth/.desktop_secret").read())',
        ):
            assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), code
        # The same variable pointed anywhere else, and an unrelated variable, stay ordinary work.
        for code in (
            'import os\nprint(os.environ["UNSLOTH_STUDIO_HOME"] + "/models")',
            'import os\nprint(os.environ.get("HOME") + "/auth")',
        ):
            assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) != (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), code
    finally:
        tools._studio_auth_markers_cache = None


def test_cmd_exe_spells_the_directory_change_in_any_case(monkeypatch, tmp_path):
    # On a Windows host without a trusted bash the shell is `cmd /c`, whose built-ins are
    # case-insensitive, so `CD ..\..` moves the working directory exactly as `cd ../..` does.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        for command in (
            'CD ..\\.. & CD auth & sqlite3 auth.db "select jwt_secret from auth_user"',
            "Cd ../..; cd auth; cat .cli_api_key_cli_1",
            "CD ../.. && cat auth/auth.db",
        ):
            assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), command
        for ordinary in ("CD ../.. && ls models", "Cd src & type README.md"):
            assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), ordinary
    finally:
        tools._studio_auth_markers_cache = None


def test_the_studio_home_variable_is_read_in_any_case(monkeypatch, tmp_path):
    # `os.environ` upper-cases every key on Windows (`os._createenviron` sets `encodekey = str.upper`
    # for `nt`), so a lower-case lookup returns the same studio root the upper-case spelling does.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        for var in ("unsloth_studio_home", "Unsloth_Studio_Home", "studio_home"):
            code = (
                "import os, sqlite3\n"
                'print(sqlite3.connect(os.environ["%s"] + "/auth/auth.db"))' % var
            )
            assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), var
        ordinary = 'import os\nprint(open(os.environ["data_dir"] + "/notes.txt").read())'
        assert tools._python_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        )
    finally:
        tools._studio_auth_markers_cache = None


def test_a_wildcard_that_expands_to_the_auth_directory(monkeypatch, tmp_path):
    # The shell expands `a?th` to `auth` before the command opens anything, so comparing the literal
    # text alone let the database through under a name that matches no marker.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        for command in (
            'sqlite3 ../../a?th/auth.db "select jwt_secret from auth_user"',
            "cat ../../aut[h]/auth.db",
            "cat ../../a[a-z]th/auth.db",
            "cat ../../au*/auth.db",
            "cat ../../auth/auth.d?",
            "cat ../../a?th/.cli_api_key_cli_1",
        ):
            assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), command
        # A segment that is nothing but a wildcard names the auth directory only in the sense that
        # listing its parent does, so it stays ordinary work.
        for ordinary in (
            "ls ../../*",
            "ls ../../models/*.gguf",
            "grep -r auth src/*.py",
            "ls data[0].csv",
        ):
            assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), ordinary
    finally:
        tools._studio_auth_markers_cache = None


def test_home_is_the_workdir_under_bypass_permissions(monkeypatch, tmp_path):
    # `_build_bypass_env` repoints HOME at the tool workdir, so `$HOME/../..` is the sandbox walked
    # two levels up, which is the auth directory's parent. Joining the literal token under the
    # workdir read it as `<workdir>/$HOME/...` and missed.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        for command in (
            'sqlite3 "$HOME/../../auth/auth.db" "select jwt_secret from auth_user"',
            "cat ${HOME}/../../auth/auth.db",
            "cat ~/../../auth/auth.db",
            "cat $HOME/../../auth/.cli_api_key_cli_1",
        ):
            assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), command
        for ordinary in (
            "cat $HOME/notes.md",
            "ls ~/models",
            "cat $HOME/../../models/m.gguf",
            "echo $HOMEBREW_PREFIX",
            "cat backup~/x.txt",
        ):
            assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), ordinary
    finally:
        tools._studio_auth_markers_cache = None


def test_pushd_and_a_python_chdir_move_the_directory_too(monkeypatch, tmp_path):
    # `pushd DIR` makes DIR the working directory exactly as `cd` does, and `os.chdir('../..')`
    # moves every path AFTER it in the same snippet. Checked independently, the move reaches only
    # the studio root and the `auth/auth.db` that follows still looks like it is in the sandbox.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        for command in (
            'pushd ../..; sqlite3 auth/auth.db "select jwt_secret from auth_user"',
            "pushd ../.. && cat auth/.desktop_secret",
            # cmd.exe environment names are case-insensitive, so %home% expands like %HOME%.
            r"type %home%\..\..\auth\auth.db",
        ):
            assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), command
        for code in (
            "import os, sqlite3\nos.chdir('../..')\nprint(sqlite3.connect('auth/auth.db'))",
            "import os\nos.chdir('..')\nos.chdir('..')\nprint(open('auth/auth.db').read())",
            "import os\nos.chdir('../../auth')\nprint(open('.cli_api_key_cli_1').read())",
        ):
            assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), code
        # Moving anywhere else, and an `auth` that is a string rather than a path, stay ordinary.
        for ordinary in ("pushd ../models; ls", "popd"):
            assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), ordinary
        for code in (
            "import os\nos.chdir('../data')\nprint(open('x.csv').read())",
            "print(open('auth/auth.db').read())",
            "print('authentication helper')",
            "x = 'auth'\nprint(x)",
        ):
            assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) != (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), code
    finally:
        tools._studio_auth_markers_cache = None


def test_a_studio_home_variable_pointing_elsewhere_is_not_ours(monkeypatch, tmp_path):
    # `STUDIO_HOME` is a generic name another application can own. With the spelling registered
    # unconditionally, a command naming that application's directory was refused in every permission
    # mode even though it never came near this install.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setenv("STUDIO_HOME", str(tmp_path / "other-app"))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        assert not tools._references_studio_credential("cat $STUDIO_HOME/auth/auth.db")
        assert tools._references_studio_credential("cat $UNSLOTH_STUDIO_HOME/auth/auth.db")
    finally:
        tools._studio_auth_markers_cache = None


def test_an_unset_studio_home_variable_stays_registered(monkeypatch, tmp_path):
    # The child cannot expand it either, so the spelling reaches nothing and dropping it would only
    # widen the guard for no gain.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        assert tools._references_studio_credential("cat $STUDIO_HOME/auth/auth.db")
    finally:
        tools._studio_auth_markers_cache = None


def test_a_differently_cased_studio_home_value_is_still_ours(monkeypatch, tmp_path):
    # Windows paths are case-insensitive, so a value spelled with a different drive-letter case is
    # the same directory. Comparing it exactly dropped every spelling of the variable.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setenv("STUDIO_HOME", str(home).upper())
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        assert tools._references_studio_credential("cat $UNSLOTH_STUDIO_HOME/auth/auth.db")
        if os.path.normcase("A") == os.path.normcase("a"):  # a case-insensitive filesystem
            assert tools._references_studio_credential(r"type %STUDIO_HOME%\auth\auth.db")
    finally:
        tools._studio_auth_markers_cache = None


def test_a_studio_home_variable_pointing_elsewhere_is_not_ours(monkeypatch, tmp_path):
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setenv("STUDIO_HOME", str(tmp_path / "another-app"))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        assert not tools._references_studio_credential("cat $STUDIO_HOME/auth/auth.db")
        assert tools._references_studio_credential("cat $UNSLOTH_STUDIO_HOME/auth/auth.db")
    finally:
        tools._studio_auth_markers_cache = None


def test_a_symlinked_studio_home_is_still_ours(monkeypatch, tmp_path):
    # `studio_root()` resolves aliases, so a STUDIO_HOME that is a symlink to the configured root
    # compared unequal to it and every spelling of the variable was dropped from the guard.
    real = tmp_path / "real-home"
    (real / "auth").mkdir(parents = True)
    alias = tmp_path / "alias-home"
    try:
        alias.symlink_to(real, target_is_directory = True)
    except (OSError, NotImplementedError):  # no symlink privilege on this host
        pytest.skip("symlinks unavailable")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(alias))
    monkeypatch.setenv("STUDIO_HOME", str(alias))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        assert tools._references_studio_credential(
            'sqlite3 "$UNSLOTH_STUDIO_HOME/auth/auth.db" "select 1"'
        )
        assert tools._references_studio_credential("cat $STUDIO_HOME/auth/.desktop_secret")
    finally:
        tools._studio_auth_markers_cache = None


def test_a_failed_directory_change_leaves_the_cwd_where_it_was(monkeypatch, tmp_path):
    # `cd missing` fails and the shell carries on from where it was, so the NEXT `cd ../..` starts
    # at the real sandbox and reaches the studio root. Assuming every change succeeds resolved the
    # rest against a directory the command was never in. Python catching an OSError is the same.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        for command in (
            "cd definitely-missing; cd ../..; cat auth/auth.db",
            'cd nope || true; cd ../.. ; sqlite3 auth/auth.db "select 1"',
        ):
            assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), command
        for code in (
            "import os\ntry:\n    os.chdir('definitely-missing')\nexcept OSError:\n    pass\n"
            "os.chdir('../..')\nprint(open('auth/auth.db').read())",
            "import os, sqlite3\ntry:\n    os.chdir('nope')\nexcept Exception:\n    pass\n"
            "os.chdir('../..')\nprint(sqlite3.connect('auth/auth.db'))",
        ):
            assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), code
        for ordinary in ("cd missing; ls", "cd ../data && cat x.csv"):
            assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), ordinary
        assert (
            tools._python_exec(
                "import os\nos.chdir('sub')\nprint(open('auth.db').read())",
                None,
                30,
                _SESSION,
                disable_sandbox = True,
            )
            != tools._STUDIO_CREDENTIAL_BLOCKED
        )
    finally:
        tools._studio_auth_markers_cache = None


def test_cd_options_padding_and_a_keyword_chdir(monkeypatch, tmp_path):
    # `cd [-L|[-P [-e]] [-@]] [dir]` is what bash documents, so an option is not the target;
    # padding with failing `cd`s must not spend the state budget that holds the real sandbox; and
    # `os.chdir(path=...)` is the same call as the positional one.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        padded = "; ".join(f"cd missing{i}" for i in range(7)) + "; cd ../..; cat auth/auth.db"
        for command in (
            "cd -P ../..; cat auth/auth.db",
            'cd -L -e ../.. && sqlite3 auth/auth.db "select 1"',
            padded,
        ):
            assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), command
        assert (
            tools._python_exec(
                "import os\nos.chdir(path='../..')\nprint(open('auth/auth.db').read())",
                None,
                30,
                _SESSION,
                disable_sandbox = True,
            )
            == tools._STUDIO_CREDENTIAL_BLOCKED
        )
        for ordinary in ("cd -P ../models; ls", "cd -; ls"):
            assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), ordinary
        assert (
            tools._python_exec(
                "import os\nos.chdir(path='../data')\nprint(1)",
                None,
                30,
                _SESSION,
                disable_sandbox = True,
            )
            != tools._STUDIO_CREDENTIAL_BLOCKED
        )
    finally:
        tools._studio_auth_markers_cache = None


def test_a_glob_is_read_per_token_not_per_command(monkeypatch, tmp_path):
    # `sqlite3 <home>/a?th/auth.db` starts its segments at `sqlite3 <home>`, so comparing from
    # segment zero could never line up with an absolute marker; the glob pass reads each path-shaped
    # token on its own now.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        for command in (
            f"sqlite3 {home}/a?th/auth.db 'select jwt_secret from auth_user'",
            f"cat {home}/au*/.desktop_secret",
            f"cat {home}/a[u]th/auth.db",
            "sqlite3 ../../a?th/auth.db 'select 1'",
        ):
            assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), command
        # Listing a parent, or globbing anywhere else, is ordinary work.
        for ordinary in (
            f"ls {home}/*",
            f"ls {home}/models/*.gguf",
            f"cat {home}/logs/*.log",
            "grep -rn 'auth' src/*.py",
        ):
            assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), ordinary
    finally:
        tools._studio_auth_markers_cache = None
