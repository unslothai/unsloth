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


@pytest.fixture
def studio_home(monkeypatch, tmp_path):
    """A studio home with `auth/` and this session's sandbox, and the marker cache reset around it.

    Every guard test needs the same four lines of setup and the same teardown, so they live here.
    """
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        yield home
    finally:
        tools._studio_auth_markers_cache = None


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


def test_an_mcp_call_at_the_auth_dir_never_reaches_the_server(monkeypatch):
    # Through `execute_tool`, so the refusal is asserted where it has to happen: BEFORE dispatch.
    # Deleting the refusal from that function left every test here passing, because the test above
    # checks the predicate and the classifier and never the dispatch.
    name = f"{MCP_TOOL_PREFIX}fs__read_file"
    reached = []

    def _never(*args, **kwargs):
        reached.append(args)
        raise AssertionError("the MCP server was reached for a credential path")

    for attribute in ("_mcp_server_for_tool", "_call_mcp_tool", "call_mcp_tool"):
        if hasattr(tools, attribute):
            monkeypatch.setattr(tools, attribute, _never, raising = False)
    result = tools.execute_tool(
        name, {"path": "~/.unsloth/studio/auth/.cli_api_key_cli_99bb88401742"}, _SESSION
    )
    assert result == tools._STUDIO_CREDENTIAL_BLOCKED
    assert not reached


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


def test_the_real_stream_generator_masks_a_key_across_chunk_boundaries():
    # Driven through `stream_tool_execution` itself, not through a copy of its masking. A reviewer
    # deleted the redaction from both of that generator's `_masked` return paths and all 94 tests
    # here still passed, because the test below re-implements the loop instead of running it.
    from core.inference.tool_stream_exec import stream_tool_execution
    key = "sk-unsloth-0123456789abcdef0123456789abcdef"
    for chunks in (
        [key],
        [key[:15], key[15:]],
        ["sk-unslo", key[8:]],
        ["out ", key[:20], key[20:], " tail"],
    ):

        def invoke(emit, _chunks = chunks):
            for chunk in _chunks:
                emit(chunk)
            return "".join(_chunks)

        generator = stream_tool_execution(invoke, tool_name = "terminal", tool_call_id = "c1")
        streamed = []
        try:
            while True:
                event = next(generator)
                if event.get("type") == "tool_output":
                    streamed.append(event.get("text") or "")
        except StopIteration as stop:
            returned = stop.value
        rendered = "".join(streamed)
        assert key not in rendered, chunks
        assert "[redacted]" in rendered, chunks
        # The RETURNED result is what the card and the model get; it is masked by the loop
        # controller, so this asserts only that the generator hands back what the tool produced.
        assert isinstance(returned, str)


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

    # A terminator anywhere after the token ends it, so a chunk that carries on past a key is
    # emitted at once. Read only at the ends, `<key> done` looked open and held this chunk and
    # every chunk after it back until the tool finished, which starved the stream of output.
    from core.inference.tool_stream_exec import _hold_back_partial_secret

    for text in (
        key + " done",
        "out " + key + " more text here",
        key + "\nnext line",
    ):
        assert _hold_back_partial_secret(text) == len(text), text
    # A token still open at the end of the chunk is still held back.
    assert _hold_back_partial_secret("out " + key) == 4


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


def test_a_python_path_built_in_pieces_is_still_the_auth_dir(studio_home):
    home = studio_home
    # `os.path.join("..", "..", "auth", "auth.db")` names the protected database in pieces, so the
    # text scan sees four ordinary strings and no path at all, and Bypass Permissions has already
    # skipped the code analyzer. auth.db carries no prefix the result mask can key on either.
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


def test_a_chain_of_cds_that_ends_inside_the_auth_dir(studio_home):
    home = studio_home
    # `cd ../..; cd auth; sqlite3 auth.db` never writes a path with a separator in it, so every
    # token in it reads as an ordinary filename and only the walked directory itself gives it away.
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


def test_traversal_through_the_proc_cwd_symlink(studio_home):
    home = studio_home
    # The kernel resolves /proc/self/cwd to the session sandbox BEFORE applying the `..` that
    # follows, so a lexical normpath reads this as /proc/self/auth/auth.db and misses the file the
    # call actually opens.
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


def test_padding_a_command_with_cds_does_not_spend_the_walk(studio_home):
    home = studio_home
    # The walk counted `cd` commands, so eight no-op ones filled the budget and the two that
    # entered the auth directory after them were never looked at.
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


def test_the_shell_pid_spelling_of_the_proc_cwd_symlink(studio_home):
    home = studio_home
    # `$$` is the shell's own PID, expanded before the path is opened, so it names the same symlink
    # a literal number does.
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


def test_a_python_path_rooted_in_the_studio_home_variable(studio_home, monkeypatch):
    home = studio_home
    # `os.environ["UNSLOTH_STUDIO_HOME"] + "/auth/auth.db"` folds to a dynamic piece plus the rest.
    # Bypass mode keeps that variable in the child env, so the dynamic piece has a known value and
    # substituting it is not a guess.
    monkeypatch.setenv("STUDIO_HOME", str(home))
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


def test_cmd_exe_spells_the_directory_change_in_any_case(studio_home):
    home = studio_home
    # On a Windows host without a trusted bash the shell is `cmd /c`, whose built-ins are
    # case-insensitive, so `CD ..\..` moves the working directory exactly as `cd ../..` does.
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


def test_the_studio_home_variable_is_read_in_any_case(studio_home):
    home = studio_home
    # `os.environ` upper-cases every key on Windows (`os._createenviron` sets `encodekey = str.upper`
    # for `nt`), so a lower-case lookup returns the same studio root the upper-case spelling does.
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


def test_a_wildcard_that_expands_to_the_auth_directory(studio_home):
    home = studio_home
    # The shell expands `a?th` to `auth` before the command opens anything, so comparing the literal
    # text alone let the database through under a name that matches no marker.
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


def test_home_is_the_workdir_under_bypass_permissions(studio_home):
    home = studio_home
    # `_build_bypass_env` repoints HOME at the tool workdir, so `$HOME/../..` is the sandbox walked
    # two levels up, which is the auth directory's parent. Joining the literal token under the
    # workdir read it as `<workdir>/$HOME/...` and missed.
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


def test_a_backslash_escape_inside_a_relative_path(studio_home):
    home = studio_home
    # On POSIX bash drops the backslash and opens `auth/auth.db`, but the join read it as a
    # separator and resolved `au/th/auth.db`, which names nothing. Both spellings are tried now,
    # because the same character IS the separator on Windows.
    for command in (
        "cd ../..; cat au\\th/auth.db",
        "cd ../..; cat auth/au\\th.db",
        "cat ../../au\\th/auth.db",
    ):
        assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), command
    for ordinary in (
        "cd ../..; cat models\\sub\\file.txt",
        "cat notes\\ file.txt",
        "cd ../..; cat models/m.gguf",
    ):
        assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary


def test_a_quoted_bracket_is_not_a_subshell(studio_home):
    home = studio_home
    # `echo '('; cd ../..; echo ')'` opens no subshell at all. Counted as syntax, the quoted
    # brackets ended a subshell that was never entered and dropped the real `cd` between them.
    for command in (
        "echo '('; cd ../..; echo ')'; strings auth/auth.db",
        'echo "("; cd ../..; cat auth/auth.db',
    ):
        assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), command
    # A real subshell still ends there, and quoted brackets around ordinary work are ordinary.
    for ordinary in (
        "(cd ../..; ls models); cat auth/config.json",
        "echo '('; ls; echo ')'; cat auth/config.json",
    ):
        assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary


def test_a_child_process_runs_from_the_directory_it_is_handed(studio_home):
    home = studio_home
    # `subprocess.run([...], cwd = "../..")` moves nothing in this process, so the walk never
    # moved and each literal argument was tested against the sandbox instead of against the
    # directory the child actually opens them from.
    for code in (
        'import subprocess\nsubprocess.run(["strings", "auth/auth.db"], cwd = "../..")',
        'import subprocess\nsubprocess.run("cat auth/.desktop_secret", shell = True, cwd = "../..")',
        'import subprocess\nsubprocess.run(["ls"], cwd = "../../auth")',
        'import subprocess\nfrom pathlib import Path\nsubprocess.run(["strings", "auth/auth.db"], cwd = Path.cwd().parents[1])',
    ):
        assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), code
    for code in (
        'import subprocess\nsubprocess.run(["ls", "auth"], cwd = "/tmp/project")',
        'import subprocess\nsubprocess.run(["ls", "models"], cwd = "../..")',
        'import subprocess\nsubprocess.run(["cat", "auth/config.json"], cwd = ".")',
    ):
        assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), code


def test_a_subshell_cd_does_not_outlive_the_subshell(studio_home):
    home = studio_home
    # `(cd ../..; ls models); cat auth/config.json` runs the `cat` where it started, because a
    # subshell's directory dies with it. Carried past the bracket, the move refused a project's
    # own auth/ read. Inside the subshell the move is real and still counts.
    for ordinary in (
        "(cd ../..; ls models); cat auth/config.json",
        "(cd ../..; ls models)\ncat auth/notes.txt",
    ):
        assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary
    for command in (
        "(cd ../..; cat auth/auth.db)",
        "x=$(cd ../.. && cat auth/auth.db)",
        "(cd ../../auth; ls)",
        "(cd ../..); cd ../..; cat auth/auth.db",
    ):
        assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), command


def test_only_a_module_that_owns_the_process_directory_changes_it(studio_home):
    home = studio_home
    # `ftp.chdir('../..')` moves a remote directory, not this process's, so the local read that
    # follows never leaves the sandbox. Matched on the method name alone, it was refused.
    assert (
        tools._python_exec(
            'import ftplib\nftp = ftplib.FTP("h")\nftp.chdir("../..")\nprint(open("auth/config.json").read())',
            None,
            30,
            _SESSION,
            disable_sandbox = True,
        )
        != tools._STUDIO_CREDENTIAL_BLOCKED
    )
    # An ALIAS of somebody else's chdir is not one either.
    assert (
        tools._python_exec(
            'import ftplib\nftp = ftplib.FTP("h")\nmove = ftp.chdir\nmove("../..")\nprint(open("auth/config.json").read())',
            None,
            30,
            _SESSION,
            disable_sandbox = True,
        )
        != tools._STUDIO_CREDENTIAL_BLOCKED
    )
    for code in (
        'import os\nos.chdir("../..")\nprint(open("auth/auth.db", "rb").read())',
        'import os as o\no.chdir("../..")\nprint(open("auth/auth.db", "rb").read())',
        'from os import chdir\nchdir("../..")\nprint(open("auth/auth.db", "rb").read())',
        'import contextlib\nwith contextlib.chdir("../.."):\n    print(open("auth/auth.db", "rb").read())',
        'import os\nmove = os.chdir\nmove("../..")\nprint(open("auth/auth.db", "rb").read())',
        'import os as o\ngo = o.chdir\ngo("../..")\nprint(open("auth/auth.db", "rb").read())',
        'from contextlib import chdir as cd\nwith cd("../.."):\n    print(open("auth/auth.db", "rb").read())',
        'import os\nmove = os.chdir\nagain = move\nagain("../..")\nprint(open("auth/auth.db", "rb").read())',
    ):
        assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), code


def test_a_cd_in_a_conditional_position_moves_the_directory(studio_home):
    home = studio_home
    # `if cd ../..; then ...; fi` runs the move as the condition itself, and a brace group is just
    # another command position. Only `then`/`do`/`else` were recognised, so those moves went
    # unrecorded and the auth database that followed was resolved against the sandbox.
    root = str(home)
    for command in (
        "if cd ../..; then sqlite3 auth/auth.db 'select jwt_secret from auth_user'; fi",
        "while cd ../..; do cat auth/auth.db; done",
        "until cd ../..; do cat auth/.desktop_secret; done",
        "{ cd ../..; cat auth/auth.db; }",
        "if false; then :; elif cd ../..; then cat auth/auth.db; fi",
        f"if cd {root}; then cat auth/auth.db; fi",
    ):
        assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), command
    for ordinary in (
        "if cd ../..; then cat models/m.gguf; fi",
        "echo 'if cd ../..'; grep auth README",
        "if true; then cat auth.py; fi",
    ):
        assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary


def test_a_chdir_to_a_parent_walk_moves_the_directory(studio_home):
    home = studio_home
    # `os.chdir(Path.cwd().parents[1])` is an ordinary move into the studio root, but the fold
    # writes the walk as one marker, so the move was ignored and the `auth/auth.db` that followed
    # was still resolved against the sandbox.
    for code in (
        'import os, sqlite3\nfrom pathlib import Path\nos.chdir(Path.cwd().parents[1])\nprint(sqlite3.connect("auth/auth.db"))',
        'import os\nfrom pathlib import Path\nos.chdir(Path.cwd().parent.parent)\nprint(open("auth/.desktop_secret").read())',
        'import os\nfrom pathlib import Path\nos.chdir(Path.cwd().parents[1] / "auth")\nprint(open("auth.db", "rb").read())',
    ):
        assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), code
    # The same move, then ordinary work, and a walk from a literal base that is not the sandbox.
    for code in (
        'import os\nfrom pathlib import Path\nos.chdir(Path.cwd().parents[1])\nprint(open("models/m.gguf", "rb").read(4))',
        'import os\nfrom pathlib import Path\nroot = Path("/tmp/project")\nos.chdir(root.parent)\nprint(open("auth/auth.db", "rb").read(4))',
    ):
        assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), code


def test_a_quoted_cd_into_the_studio_root_is_not_a_move(studio_home):
    home = studio_home
    # `echo 'cd <root>'; grep auth README` prints the text and searches a project. Matched wherever
    # it appeared, the quoted text read as a move into the studio root and refused ordinary work.
    root = str(home)
    for ordinary in (
        f"echo 'cd {root}'; grep auth README",
        f"echo cd {root}; cat auth.py",
        f"grep -rn 'cd {root}' src/; grep auth README",
        f"cd {root}/models && grep auth README",
        f"cd {root}-backup && ls auth",
    ):
        assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary
    # A real move into the root, in every position a shell runs one, still counts.
    for command in (
        f"cd {root} && cat auth/auth.db",
        f"cd {root}; ls auth",
        f"cd '{root}'/auth && ls",
        f"builtin cd {root} && cat auth/auth.db",
        f"if true; then cd {root}; cat auth/auth.db; fi",
        f"pushd {root} && cat auth/auth.db",
    ):
        assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), command


def test_shell_punctuation_ends_a_credential_name(studio_home):
    home = studio_home
    # A name ends where the shell ends a word, so `cat /tmp/.bootstrap_password; echo done` names
    # the file. Accepting only whitespace or a quote as the boundary matched none of these.
    for command in (
        "cat /tmp/.bootstrap_password; echo done",
        "cat /tmp/llama_api_key;",
        "(cat /tmp/.desktop_secret)",
        "cat /tmp/.cli_api_key_cli_1;echo x",
        "cat /opt/agent_api_key.json|head",
        "cat /tmp/.bootstrap_password&",
    ):
        assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), command
    # The same punctuation after ordinary work is still ordinary work.
    for ordinary in (
        "cat notes.txt; echo done",
        "grep -rn auth src/;",
        "cat auth.py;",
        'python -c "import auth"',
    ):
        assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary


def test_a_builtin_wrapped_cd_moves_the_directory(studio_home):
    home = studio_home
    # `builtin cd ..` and `command cd ..` run the same shell builtin with the same argument, so
    # everything after them is relative to the new directory. Matched on the bare name only, the
    # move went unrecorded and the auth database that followed still looked like the sandbox's.
    for command in (
        'builtin cd ../..; sqlite3 auth/auth.db "select jwt_secret from auth_user"',
        "command cd ../.. && cat auth/.desktop_secret",
        "builtin cd ../..; cat auth/.cli_api_key_cli_1",
    ):
        assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), command
    for ordinary in (
        "echo builtin cd ../..",
        "builtin cd ../..; cat models/m.gguf",
        "command ls auth/",
    ):
        assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary


def test_a_parent_walk_from_a_named_literal_base_is_that_base(studio_home):
    home = studio_home
    # A name holding a literal path is not the working directory, so `root = Path('/tmp/project')`
    # walked up twice is `/`, not the studio root, and a project's own auth/ must still be readable.
    for code in (
        'from pathlib import Path\nroot = Path("/tmp/project")\nprint(root.parent.parent / "auth" / "auth.db")',
        'from pathlib import Path\nroot = Path("/srv/app")\nprint(root.parent / "auth" / "auth.db")',
    ):
        assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), code
    # A name bound to the working directory, or rebound so its value is unknown, still counts.
    for code in (
        'from pathlib import Path\nroot = Path.cwd()\nprint((root.parent.parent / "auth" / "auth.db").read_bytes())',
        'from pathlib import Path\nprint((Path.cwd().parents[1] / "auth" / "auth.db").read_bytes())',
    ):
        assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), code


def test_pwd_is_the_workdir_under_bypass_permissions(studio_home):
    home = studio_home
    # Bash rewrites PWD on every `cd`, so under bypass it names the tool workdir and `$PWD/../..`
    # is the auth directory's parent. Left as a literal segment it read as `<workdir>/$PWD/...`.
    for command in (
        'sqlite3 "$PWD/../../auth/auth.db" "select jwt_secret from auth_user"',
        "cat ${PWD}/../../auth/.desktop_secret",
        "cat $PWD/../../auth/.cli_api_key_cli_1",
    ):
        assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), command
    # Python reads the same directory by name or by call.
    for code in (
        'import os, sqlite3\nprint(sqlite3.connect(os.environ["PWD"] + "/../../auth/auth.db"))',
        'import os\nprint(open(os.getcwd() + "/../../auth/auth.db").read())',
        'from pathlib import Path\nprint((Path.cwd() / ".." / ".." / "auth" / "auth.db").read_bytes())',
    ):
        assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), code
    # Ordinary work names the working directory constantly and must still run.
    for ordinary in (
        "echo $PWD",
        "ls $PWD/../models",
        "cat $PWD/notes.md",
        "echo $PWDX",
    ):
        assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary
    assert (
        tools._python_exec(
            'import os\nprint(os.environ["PWD"] + "/data")',
            None,
            30,
            _SESSION,
            disable_sandbox = True,
        )
        != tools._STUDIO_CREDENTIAL_BLOCKED
    )


def test_pushd_and_a_python_chdir_move_the_directory_too(studio_home):
    home = studio_home
    # `pushd DIR` makes DIR the working directory exactly as `cd` does, and `os.chdir('../..')`
    # moves every path AFTER it in the same snippet. Checked independently, the move reaches only
    # the studio root and the `auth/auth.db` that follows still looks like it is in the sandbox.
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


def test_a_studio_home_variable_pointing_elsewhere_is_not_ours(studio_home, monkeypatch, tmp_path):
    home = studio_home
    # `STUDIO_HOME` is a generic name another application can own. With the spelling registered
    # unconditionally, a command naming that application's directory was refused in every permission
    # mode even though it never came near this install.
    monkeypatch.setenv("STUDIO_HOME", str(tmp_path / "other-app"))
    assert not tools._references_studio_credential("cat $STUDIO_HOME/auth/auth.db")
    assert tools._references_studio_credential("cat $UNSLOTH_STUDIO_HOME/auth/auth.db")


def test_an_unset_studio_home_variable_stays_registered(studio_home):
    home = studio_home
    # The child cannot expand it either, so the spelling reaches nothing and dropping it would only
    # widen the guard for no gain.
    assert tools._references_studio_credential("cat $STUDIO_HOME/auth/auth.db")


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


def test_a_failed_directory_change_leaves_the_cwd_where_it_was(studio_home):
    home = studio_home
    # `cd missing` fails and the shell carries on from where it was, so the NEXT `cd ../..` starts
    # at the real sandbox and reaches the studio root. Assuming every change succeeds resolved the
    # rest against a directory the command was never in. Python catching an OSError is the same.
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


def test_cd_options_padding_and_a_keyword_chdir(studio_home):
    home = studio_home
    # `cd [-L|[-P [-e]] [-@]] [dir]` is what bash documents, so an option is not the target;
    # padding with failing `cd`s must not spend the state budget that holds the real sandbox; and
    # `os.chdir(path=...)` is the same call as the positional one.
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


def test_a_glob_is_read_per_token_not_per_command(studio_home):
    home = studio_home
    # `sqlite3 <home>/a?th/auth.db` starts its segments at `sqlite3 <home>`, so comparing from
    # segment zero could never line up with an absolute marker; the glob pass reads each path-shaped
    # token on its own now.
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


def test_a_pushd_to_an_absolute_home_and_paths_before_the_cd(studio_home):
    home = studio_home
    # Two halves of the same walk: the workdir has to be resolved for a command that MOVES the
    # directory even when it carries no `..`, and a relative path written BEFORE the move opens
    # from the old directory, so it must not be resolved against the new one.
    for command in (
        f'pushd {home}; sqlite3 auth/auth.db "select jwt_secret from auth_user"',
        f"cd {home} && cat auth/.desktop_secret",
    ):
        assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), command
    for ordinary in (
        "cat auth/auth.db; cd ../..",
        "cat auth/auth.db",
        f"pushd {home}/models; ls",
    ):
        assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary


def test_a_pathlib_parent_walk_and_the_option_terminator(studio_home):
    home = studio_home
    # `Path.cwd().parents[1] / "auth" / "auth.db"` walks up from the sandbox without writing a `..`,
    # and bash accepts `--` as the end of options, so it is not the target of the `cd`.
    for code in (
        "import sqlite3\nfrom pathlib import Path\n"
        'print(sqlite3.connect(Path.cwd().parents[1] / "auth" / "auth.db"))',
        "from pathlib import Path\n"
        'print((Path.cwd().parent.parent / "auth" / ".desktop_secret").read_text())',
    ):
        assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), code
    assert (
        tools._bash_exec(
            "cd -- ../..; cat auth/auth.db",
            None,
            30,
            _SESSION,
            disable_sandbox = True,
        )
        == tools._STUDIO_CREDENTIAL_BLOCKED
    )
    # A parent walk that lands anywhere else stays ordinary work.
    for code in (
        'from pathlib import Path\nprint(Path.cwd().parent / "data" / "x.csv")',
        'from pathlib import Path\nprint(Path("x.txt").parent)',
        'from pathlib import Path\nprint((Path.cwd().parents[1] / "models" / "m.gguf").exists())',
        # The sandbox is <home>/sandbox/<session>, so parents[1] is the home and parents[3] is
        # two levels above it, which is not this install.
        'from pathlib import Path\nprint((Path.cwd().parents[3] / "auth" / "auth.db"))',
    ):
        assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), code
    assert (
        tools._bash_exec(
            "cd -- ../models; ls",
            None,
            30,
            _SESSION,
            disable_sandbox = True,
        )
        != tools._STUDIO_CREDENTIAL_BLOCKED
    )


def test_command_positions_explicit_bases_and_chdir_aliases(studio_home):
    home = studio_home
    for command in (
        "cd ../..; cat auth/auth.db",
        # An indented command line moves the shell exactly as an unindented one does.
        "  cd ../..; cat auth/auth.db",
        "x=1 && cd ../.. && cat auth/auth.db",
        "if true; then cd ../..; cat auth/auth.db; fi",
    ):
        assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), command
    # ...and nowhere else: `cd` as an ARGUMENT changes no directory, and reading a project's own
    # auth/ after one was refused in every mode.
    for ordinary in (
        "echo cd ../..; cat auth/config.json",
        'echo "cd ../.."',
        "grep -rn cd ../../src",
    ):
        assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary
    # An alias of os.chdir moves the directory exactly as the name does.
    for code in (
        "from os import chdir as move\nmove('../..')\nprint(open('auth/auth.db').read())",
        "import os\ngo = os.chdir\ngo('../..')\nprint(open('auth/auth.db').read())",
    ):
        assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), code
    # A parent walk from an EXPLICIT base is that base's parent, not the sandbox's.
    assert (
        tools._python_exec(
            "from pathlib import Path\nprint(Path('/tmp').parent / 'auth' / 'auth.db')",
            None,
            30,
            _SESSION,
            disable_sandbox = True,
        )
        != tools._STUDIO_CREDENTIAL_BLOCKED
    )


def test_a_generic_cwd_keyword_is_not_a_child_process(studio_home):
    home = studio_home
    # `cwd` is an ordinary keyword name. Reading it as process semantics on any call refused
    # ordinary code whose function may never touch the path it was handed.
    for ordinary in (
        'describe("auth/config.json", cwd = "../..")',
        'import subprocess as sp\nsp.run(["ls"], cwd = "build")',
    ):
        assert tools._python_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary
    for code in (
        'import subprocess\nsubprocess.run(["cat", "auth/auth.db"], cwd = "../..")',
        'from subprocess import run\nrun(["cat", "auth/auth.db"], cwd = "../..")',
        # An import alias leaves the call spelled `sp.run(...)`.
        'import subprocess as sp\nsp.run(["cat", "auth/auth.db"], cwd = "../..")',
        'import asyncio as aio\naio.create_subprocess_exec("cat", "auth/auth.db", cwd = "../..")',
        # A renamed from-import leaves the call under a name no fixed table holds.
        'from subprocess import run as launch\nlaunch(["cat", "auth/auth.db"], cwd = "../..")',
    ):
        assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), code


def test_a_context_manager_chdir_restores_on_exit(studio_home):
    home = studio_home
    # `contextlib.chdir` moves only while its body runs, so a read AFTER the block is back in the
    # sandbox and must not be resolved against the studio root.
    after = (
        "import contextlib\n"
        'with contextlib.chdir("../.."):\n'
        "    pass\n"
        'print(open("auth/config.json").read())'
    )
    assert tools._python_exec(after, None, 30, _SESSION, disable_sandbox = True) != (
        tools._STUDIO_CREDENTIAL_BLOCKED
    )
    inside = (
        "import contextlib\n"
        'with contextlib.chdir("../.."):\n'
        '    print(open("auth/auth.db", "rb").read())'
    )
    assert tools._python_exec(inside, None, 30, _SESSION, disable_sandbox = True) == (
        tools._STUDIO_CREDENTIAL_BLOCKED
    )
    permanent = 'import os\nos.chdir("../..")\nprint(open("auth/auth.db", "rb").read())'
    assert tools._python_exec(permanent, None, 30, _SESSION, disable_sandbox = True) == (
        tools._STUDIO_CREDENTIAL_BLOCKED
    )


def test_a_shell_variable_supplying_the_cd_target(studio_home):
    home = studio_home
    # `d=../..; cd "$d"` moves the directory every later relative path opens from. Handing the
    # UNexpanded text to the cwd walk read `$d` as a directory name, so the walk never moved.
    command = 'd=../..; cd "$d"; sqlite3 auth/auth.db "select jwt_secret from auth_user"'
    assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
        tools._STUDIO_CREDENTIAL_BLOCKED
    )
    assert (
        tools._bash_exec('d=build; cd "$d"; make', None, 30, _SESSION, disable_sandbox = True)
        != tools._STUDIO_CREDENTIAL_BLOCKED
    )


def test_a_snippets_own_run_is_not_a_child_process(studio_home):
    home = studio_home
    # `run`, `call` and `check_output` are ordinary function names. Treating a snippet's own
    # definition as a process launch refused ordinary code in every permission mode.
    ordinary = 'def run(*args, **kwargs):\n    pass\nrun(["auth/config.json"], cwd = "../..")'
    assert tools._python_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
        tools._STUDIO_CREDENTIAL_BLOCKED
    )
    imported = 'from subprocess import run\nrun(["cat", "auth/auth.db"], cwd = "../..")'
    assert tools._python_exec(imported, None, 30, _SESSION, disable_sandbox = True) == (
        tools._STUDIO_CREDENTIAL_BLOCKED
    )


def test_a_command_substitution_inside_double_quotes_is_a_subshell(studio_home):
    home = studio_home
    # Substitution stays ACTIVE inside double quotes, so `"$(cd ../..; pwd)"` opens a real subshell
    # whose move dies with it. Skipping both brackets made the inner `cd` look like it lasted
    # through the rest of the command, and the project's own auth/config.json was refused.
    ordinary = 'echo "$(cd ../..; pwd)"; cat auth/config.json'
    assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
        tools._STUDIO_CREDENTIAL_BLOCKED
    )
    # Inside the substitution the move is real, and a literal bracket in single quotes is not
    # syntax at all, so neither loses the credential read.
    for blocked in (
        'echo "$(cd ../..; cat auth/auth.db)"',
        "echo '('; cd ../..; echo ')'; cat auth/auth.db",
    ):
        assert tools._bash_exec(blocked, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), blocked


def test_consecutive_cds_inside_one_subshell(studio_home):
    home = studio_home
    # A subshell's move has to reach the NEXT command inside the same subshell. Dropping it outright
    # resolved the second `cd` from the outer sandbox and missed the database entirely.
    blocked = '(cd ../..; cd auth; sqlite3 auth.db "select jwt_secret from auth_user")'
    assert tools._bash_exec(blocked, None, 30, _SESSION, disable_sandbox = True) == (
        tools._STUDIO_CREDENTIAL_BLOCKED
    )
    # ...and must not reach past the closing bracket. Entering the directory and leaving
    # without reading anything is not a credential read.
    for ordinary in (
        "(cd ../..; cd auth); cat config.json",
        "(cd ../..; ls models); cat auth/config.json",
    ):
        assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary


def test_a_chdir_in_an_uncalled_helper_does_not_move_the_walk(studio_home):
    home = studio_home
    # A `chdir` inside a function body only moves anything if that function runs.
    ordinary = (
        "import os\n"
        "def helper():\n"
        '    os.chdir("../..")\n'
        'print(open("auth/config.json").read())'
    )
    assert tools._python_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
        tools._STUDIO_CREDENTIAL_BLOCKED
    )
    # A body whose name IS called stays live, and a branch is not a scope.
    for blocked in (
        'import os\ndef helper():\n    os.chdir("../..")\nhelper()\n'
        'print(open("auth/auth.db", "rb").read())',
        'import os\nif x:\n    os.chdir("../..")\nprint(open("auth/auth.db", "rb").read())',
    ):
        assert tools._python_exec(blocked, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), blocked


def test_a_wildcard_that_carries_no_literal_hint(studio_home):
    home = studio_home
    # `../../a?th/.b*` expands onto `auth/.bootstrap_password` while containing none of the literal
    # hints, so the prefilter was returning before the glob analysis that exists for this could run.
    for command in ("cat ../../a?th/.b*", "cat ../../a?th/*"):
        assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), command
    for ordinary in ("ls *.py", "ls ../../m*/", "ls ../../models/*.gguf", "grep -r x src/*.py"):
        assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary


def test_fchdir_is_a_move_with_an_unknown_destination(studio_home):
    home = studio_home
    # `os.fchdir(fd)` names where it goes by descriptor, so there is no path to fold. The move is
    # real, so the studio root joins the live directories and a later relative credential path is
    # resolved from there as well as from the sandbox.
    blocked = (
        "import os\n"
        'fd = os.open("../..", os.O_RDONLY)\n'
        "os.fchdir(fd)\n"
        'print(open("auth/auth.db", "rb").read())'
    )
    assert tools._python_exec(blocked, None, 30, _SESSION, disable_sandbox = True) == (
        tools._STUDIO_CREDENTIAL_BLOCKED
    )
    ordinary = (
        "import os\n"
        'fd = os.open("data", os.O_RDONLY)\n'
        "os.fchdir(fd)\n"
        'print(open("notes.txt").read())'
    )
    assert tools._python_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
        tools._STUDIO_CREDENTIAL_BLOCKED
    )


def test_a_snippets_own_chdir_is_not_a_move(studio_home):
    home = studio_home
    # A bare `chdir` only moves the walk once an import binds it. A snippet's own definition is an
    # ordinary function, and a call to it was resolving later paths under the studio root.
    ordinary = 'def chdir(path):\n    pass\nchdir("../..")\nprint(open("auth/config.json").read())'
    assert tools._python_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
        tools._STUDIO_CREDENTIAL_BLOCKED
    )
    for blocked in (
        'from os import chdir\nchdir("../..")\nprint(open("auth/auth.db", "rb").read())',
        'from os import chdir as move\nmove("../..")\nprint(open("auth/auth.db", "rb").read())',
        'import os\nmove = os.chdir\nmove("../..")\nprint(open("auth/auth.db", "rb").read())',
    ):
        assert tools._python_exec(blocked, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), blocked


def test_a_reserved_word_before_cd_still_moves_the_directory(studio_home):
    home = studio_home
    # `!` and `time` are reserved words rather than commands, so the shell runs the `cd` builtin
    # straight after them and the move lands. Accepting only a command position that started at a
    # separator, the guard read `! cd ../..; cat auth/auth.db` as a command that never moved.
    for command in (
        "! cd ../..; cat auth/auth.db",
        "time cd ../..; cat auth/.desktop_secret",
        # Bash accepts assignment words before a special builtin, so these move too.
        "X=1 cd ../..; cat auth/auth.db",
        'LC_ALL=C TZ=UTC cd ../..; sqlite3 auth/auth.db "select jwt_secret from auth_user"',
        "time -p cd ../..; cat auth/.cli_api_key_cli_1",
        'nohup cd ../..; sqlite3 auth/auth.db "select jwt_secret from auth_user"',
    ):
        assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), command
    for ordinary in (
        "! true; cat auth/config.json",
        "X=1 cat models/m.gguf",
        "echo X=1 cd ../..",
        "time cat models/m.gguf",
        "echo time cd ../..",
    ):
        assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary


def test_a_cd_in_a_function_nothing_calls_does_not_move_the_directory(studio_home):
    home = studio_home
    # Defining a function does not run it, so `helper() { cd ../..; }; cat auth/config.json` reads
    # the project's own auth file from the unchanged sandbox. Carried out of the body regardless,
    # the move turned an ordinary read into a refusal in every permission mode.
    for ordinary in (
        "helper() { cd ../..; }; cat auth/config.json",
        "function helper { cd ../..; }; cat auth/auth.db",
        "helper () { cd ../..; helper; }; cat auth/auth.db",
    ):
        assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary
    # Invoking it, or moving outside any function at all, is the same move it always was.
    for command in (
        "helper() { cd ../..; }; helper; cat auth/auth.db",
        "function helper { cd ../..; }; helper && cat auth/.desktop_secret",
        "helper() { echo hi; }; cd ../..; cat auth/auth.db",
        "{ cd ../..; cat auth/auth.db; }",
    ):
        assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), command


def test_a_singleton_class_and_an_escaped_builtin_are_the_literals_they_expand_to(
    studio_home, monkeypatch, tmp_path
):
    home = studio_home
    # `[a][u][t][h]` expands to exactly `auth`, and `c\d` is the `cd` builtin with its backslash
    # removed. Both were folded away before the guard read them: the classes collapsed to wildcards,
    # and a segment of nothing but wildcards is deliberately not matched, while the escaped name
    # matched no command position at all.
    for command in (
        'sqlite3 ../../[a][u][t][h]/auth.db "select jwt_secret from auth_user"',
        "cat ../../[a][u][t][h]/.desktop_secret",
        'c\\d ../..; sqlite3 auth/auth.db "select jwt_secret from auth_user"',
        "\\c\\d ../..; cat auth/.desktop_secret",
    ):
        assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), command
    # A broad class still names nothing in particular, and a singleton elsewhere is ordinary.
    for ordinary in (
        "ls ../../[a-z]*",
        "cat data/[a]/notes.txt",
        "cat notes.txt",
    ):
        assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary


def test_a_chdir_to_the_studio_home_variable_moves_there(studio_home):
    home = studio_home
    # `os.chdir(os.environ["UNSLOTH_STUDIO_HOME"])` names the root without spelling it, and bypass
    # keeps that variable in the child, so the move is real. The fold has no value for a subscript,
    # which left the walk sitting in the sandbox and the later relative path looking ordinary.
    for code in (
        'import os\nos.chdir(os.environ["UNSLOTH_STUDIO_HOME"])\nprint(open("auth/auth.db").read())',
        # Windows upper-cases every key `os.environ` is handed, so the lower-case spelling
        # resolves to the same directory there.
        'import os\nos.chdir(os.environ["unsloth_studio_home"])\nprint(open("auth/.desktop_secret").read())',
        'import os\nos.chdir(os.getenv("STUDIO_HOME"))\nprint(open("auth/auth.db").read())',
    ):
        assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), code
    for ordinary in (
        'import os\nos.chdir("data")\nprint(open("notes.txt").read())',
        'import os\nos.chdir(os.environ["HOME"])\nprint(open("notes.txt").read())',
    ):
        assert tools._python_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary


def test_a_child_cwd_and_a_directory_descriptor_both_carry_the_studio_root(studio_home):
    home = studio_home
    # Two ways to reach the auth directory without ever naming it in one path. The child's `cwd`
    # can be the studio home variable, which the fold has no value for, and `dir_fd` joins a
    # descriptor opened on `../..` to a later relative path, which the kernel combines and a scan
    # reading the two calls separately does not.
    for code in (
        "import subprocess, os\n"
        'subprocess.run(["cat", "auth/auth.db"], cwd = os.environ["UNSLOTH_STUDIO_HOME"])',
        "import subprocess, os\n"
        'subprocess.run(["cat", "auth/.desktop_secret"], cwd = os.getenv("STUDIO_HOME"))',
        'import os\nroot = os.open("../..", os.O_RDONLY)\n'
        'fd = os.open("auth/auth.db", os.O_RDONLY, dir_fd = root)',
        'import os\nroot = os.open("../..", os.O_RDONLY)\n'
        'fd = os.open("auth/.desktop_secret", os.O_RDONLY, dir_fd = root)',
    ):
        assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), code
    for ordinary in (
        'import subprocess, os\nsubprocess.run(["ls"], cwd = os.environ["UNSLOTH_STUDIO_HOME"])',
        'import os\nd = os.open("data", os.O_RDONLY)\n'
        'fd = os.open("notes.txt", os.O_RDONLY, dir_fd = d)',
        # The project's OWN auth file, opened relative to the sandbox it sits in.
        'import os\nd = os.open(".", os.O_RDONLY)\n'
        'fd = os.open("auth/config.json", os.O_RDONLY, dir_fd = d)',
    ):
        assert tools._python_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary


def test_a_class_body_runs_at_definition_but_its_methods_do_not(studio_home):
    home = studio_home
    # Python executes a class body when the class statement runs, whether or not anything
    # instantiates it, so `class C: os.chdir("../..")` moves the process. Grouped with the deferred
    # bodies, that move was dropped and the database that followed resolved from the sandbox.
    for code in (
        'import os, sqlite3\nclass C:\n    os.chdir("../..")\n'
        'print(sqlite3.connect("auth/auth.db"))',
        'import os\nclass C:\n    os.chdir("../..")\nprint(open("auth/.desktop_secret").read())',
        # An invoked method is a real move for the same reason it always was.
        'import os\nclass C:\n    def m(self):\n        os.chdir("../..")\n'
        'C().m()\nprint(open("auth/auth.db").read())',
    ):
        assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), code
    for ordinary in (
        # A method nothing calls is still deferred, and so is a plain function.
        'import os\nclass C:\n    def m(self):\n        os.chdir("../..")\n'
        'print(open("auth/config.json").read())',
        'import os\ndef f():\n    os.chdir("../..")\nprint(open("auth/config.json").read())',
        'import os\nclass C:\n    x = 1\nprint(open("notes.txt").read())',
    ):
        assert tools._python_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary


def test_an_invoked_lambda_moves_and_so_does_the_platform_os_module(studio_home):
    home = studio_home
    # `(lambda: os.chdir("../.."))()` runs its body right there, and `posix.chdir` is the same
    # process primitive `os.chdir` is, under the platform module `os` is built on. Both were read as
    # no move at all, so the database that followed resolved from the sandbox.
    for code in (
        'import os, sqlite3\n(lambda: os.chdir("../.."))()\n'
        'print(sqlite3.connect("auth/auth.db"))',
        'import posix\nposix.chdir("../..")\nprint(open("auth/auth.db").read())',
        'import posix as p\np.chdir("../..")\nprint(open("auth/.desktop_secret").read())',
        # Bound to a name and then called: the same body, one step further out.
        'import os\nmove = lambda: os.chdir("../..")\nmove()\n'
        'print(open("auth/auth.db").read())',
    ):
        assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), code
    for ordinary in (
        # Defined and never called, which is what the inert rule is for.
        'import os\nmove = lambda: os.chdir("../..")\nprint(open("auth/config.json").read())',
        # A remote directory change leaves the local one alone.
        'import ftplib\nftp = ftplib.FTP()\nftp.chdir("../..")\n'
        'print(open("auth/config.json").read())',
    ):
        assert tools._python_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary


def test_ordinary_directory_work_is_not_read_as_a_move_to_the_studio_root(monkeypatch, tmp_path):
    # Four false refusals, all of them ordinary work: a `&&` chain only reaches its second `cd` when
    # the first succeeded, `cd -` and `popd` go BACK, a directory whose path merely embeds the
    # studio path is not the studio directory, and a case-sensitive filesystem tells `Auth` from
    # `auth`. Each read below is the project's OWN auth file, not Studio's.
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    # A case-INSENSITIVE filesystem (Windows, macOS) already has this directory: `Auth` and `auth`
    # are the same one there, which is exactly why the case check below only runs elsewhere.
    if tools._CASE_SENSITIVE_PATHS:
        (home / "Auth").mkdir()
    (home / "sandbox" / _SESSION / "subdir").mkdir(parents = True)
    (tmp_path / "backup").mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        ordinary = [
            "cd subdir && cd ../.. && cat auth/config.json",
            "cd ../..; cd - >/dev/null; cat auth/config.json",
            "pushd ../.. >/dev/null; popd >/dev/null; cat auth/config.json",
            f"cat /mnt/backup{home}/auth/config.json",
        ]
        if tools._CASE_SENSITIVE_PATHS:
            ordinary.append(f"cat {home}/Auth/notes.txt")
        for command in ordinary:
            assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) != (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), command
        # The moves themselves still land where they always did.
        for command in (
            "cd ../..; cat auth/auth.db",
            "cd ../.. && cat auth/auth.db",
            "cd ../..; cd -; cd ../..; cat auth/auth.db",
            "cd -- ../..; cat auth/auth.db",
            f"cat {home}/auth/auth.db",
        ):
            assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
                tools._STUDIO_CREDENTIAL_BLOCKED
            ), command
    finally:
        tools._studio_auth_markers_cache = None


def test_enumerating_the_studio_root_for_a_credential_name_is_refused(studio_home):
    home = studio_home
    # `find "$UNSLOTH_STUDIO_HOME" -name auth.db -exec cat {} \;` hands the directory to a tool that
    # walks it, so no single token of the command is a path to the database. Naming the ROOT and a
    # credential BASENAME is the shape, however they are joined.
    for command in (
        'find "$UNSLOTH_STUDIO_HOME" -name auth.db -exec cat {} \;',
        f'find {home} -name ".desktop_secret" -exec cat {{}} \;',
        'grep -r sk-unsloth "$UNSLOTH_STUDIO_HOME" --include=auth.db',
    ):
        assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), command
    for ordinary in (
        'find . -name "*.py" -exec wc -l {} \;',
        "find data -name notes.txt",
        # A command that BINDS the generic variable means its own directory by it.
        'STUDIO_HOME=/opt/app cat "$STUDIO_HOME/auth/config.json"',
    ):
        assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary


def test_an_uppercase_studio_root_is_reconstructed_with_its_case(monkeypatch, tmp_path):
    # The root was rebuilt from the FOLDED marker, so a path with an uppercase character in it came
    # back lowercase; the case-sensitive check then rejected the guard's own synthesized path and a
    # move to the real root went through.
    home = tmp_path / "Studio-Home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        assert tools._studio_home_for_guard() == str(home)
        code = 'import os\nos.chdir(os.environ["UNSLOTH_STUDIO_HOME"])\nopen("auth/auth.db")'
        assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        )
        assert (
            tools._python_exec('open("notes.txt")', None, 30, _SESSION, disable_sandbox = True)
            != tools._STUDIO_CREDENTIAL_BLOCKED
        )
    finally:
        tools._studio_auth_markers_cache = None


def test_a_foreign_studio_home_variable_is_not_this_installs_root(
    studio_home, monkeypatch, tmp_path
):
    home = studio_home
    # `STUDIO_HOME` is a generic name another application can own. Registered unconditionally, the
    # recursive-root check refused a walk of that application's tree in every permission mode.
    other = tmp_path / "other-app"
    other.mkdir()
    monkeypatch.setenv("STUDIO_HOME", str(other))
    assert (
        tools._bash_exec(
            'find "$STUDIO_HOME" -type f -exec cat {} +',
            None,
            30,
            _SESSION,
            disable_sandbox = True,
        )
        != tools._STUDIO_CREDENTIAL_BLOCKED
    )
    assert (
        tools._bash_exec(
            'find "$UNSLOTH_STUDIO_HOME" -type f -exec cat {} +',
            None,
            30,
            _SESSION,
            disable_sandbox = True,
        )
        == tools._STUDIO_CREDENTIAL_BLOCKED
    )


def test_a_definition_time_call_is_not_inert(studio_home):
    home = studio_home
    # Default arguments and decorators run when the function is DEFINED, so the move happens even
    # though nothing calls `f`. Only the BODY of an uncalled function is inert.
    read = "print(open('auth/auth.db','rb').read())"
    for code in (
        f"import os\ndef f(x = os.chdir('../..')): pass\n{read}",
        f"import os\n@os.chdir('../..')\ndef f(): pass\n{read}",
    ):
        assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), code
    # A call in the body of a function nothing invokes still moves nothing.
    inert = f"import os\ndef f():\n    os.chdir('../..')\n{read}"
    assert tools._python_exec(inert, None, 30, _SESSION, disable_sandbox = True) != (
        tools._STUDIO_CREDENTIAL_BLOCKED
    )


def test_a_shell_local_studio_home_wins_over_the_backend_one(studio_home):
    home = studio_home
    # An assignment that stands as a command of its own rebinds what FOLLOWS it, so a command that
    # rebinds the variable before using it names its own directory. Two things it does not cover: a
    # PREFIX assignment, which POSIX expands the rest of its own command line before applying, and
    # an assignment after a use, which rebinds nothing for that use.
    for ordinary in (
        'UNSLOTH_STUDIO_HOME=/tmp/project; cat "$UNSLOTH_STUDIO_HOME/auth/config.json"',
        'export UNSLOTH_STUDIO_HOME=/tmp/p\ncat "$UNSLOTH_STUDIO_HOME/auth/config.json"',
        "UNSLOTH_STUDIO_HOME=/tmp/a; UNSLOTH_STUDIO_HOME=/tmp/b;"
        ' cat "$UNSLOTH_STUDIO_HOME/auth/config.json"',
    ):
        assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary
    for refused in (
        'cat "$UNSLOTH_STUDIO_HOME/auth/auth.db"; UNSLOTH_STUDIO_HOME=/tmp/project',
        # The LAST assignment is the one the expansion applies, so a read between two of them
        # still happens under the real root.
        'UNSLOTH_STUDIO_HOME=$UNSLOTH_STUDIO_HOME; sqlite3 "$UNSLOTH_STUDIO_HOME/auth/auth.db"'
        " .dump; UNSLOTH_STUDIO_HOME=/tmp",
        # A PREFIX assignment does not govern its own command's expansion: the shell builds the
        # word list first, so this still opens the INHERITED home's database.
        'UNSLOTH_STUDIO_HOME=/tmp sqlite3 "$UNSLOTH_STUDIO_HOME/auth/auth.db" .dump',
        'UNSLOTH_STUDIO_HOME=/tmp/project cat "$UNSLOTH_STUDIO_HOME/auth/auth.db"',
        f'UNSLOTH_STUDIO_HOME={home}; cat "$UNSLOTH_STUDIO_HOME/auth/auth.db"',
        # A quoted assignment is DATA that `echo` prints, so the later expansion still uses the
        # inherited home and the database read is a real one.
        'echo " UNSLOTH_STUDIO_HOME=/tmp;"; sqlite3 "$UNSLOTH_STUDIO_HOME/auth/auth.db" .dump',
        # An assignment inside `( ... )` binds only the subshell, so the outer command still
        # expands the inherited home.
        '(UNSLOTH_STUDIO_HOME=/tmp); sqlite3 "$UNSLOTH_STUDIO_HOME/auth/auth.db" .dump',
        # A backslash escapes the quote it precedes, so the region is still open and the
        # assignment inside it is still data.
        'echo "x \\" UNSLOTH_STUDIO_HOME=/tmp;"; sqlite3 "$UNSLOTH_STUDIO_HOME/auth/auth.db" .dump',
    ):
        assert tools._bash_exec(refused, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), refused


def test_a_path_walked_up_from_getcwd_is_resolved(studio_home):
    home = studio_home
    # `os.path.dirname(os.path.dirname(os.getcwd()))` walks out of the sandbox without writing a
    # `..`, so the workdir was never handed to the analyzer that resolves `getcwd` itself.
    code = (
        "import os, sqlite3\nroot = os.path.dirname(os.path.dirname(os.getcwd()))\n"
        'sqlite3.connect(os.path.join(root, "auth", "auth.db"))'
    )
    assert tools._python_exec(code, None, 30, _SESSION, disable_sandbox = True) == (
        tools._STUDIO_CREDENTIAL_BLOCKED
    )
    assert (
        tools._python_exec(
            "import os\nprint(os.getcwd())", None, 30, _SESSION, disable_sandbox = True
        )
        != tools._STUDIO_CREDENTIAL_BLOCKED
    )


def test_a_foreign_studio_home_value_is_not_read_as_this_installs_root(
    studio_home, monkeypatch, tmp_path
):
    home = studio_home
    # Bypass keeps a foreign `STUDIO_HOME` in the child, so python that reads it names the other
    # application's directory. The variable NAME alone was enough to refuse the call.
    other = tmp_path / "other-app"
    other.mkdir()
    monkeypatch.setenv("STUDIO_HOME", str(other))
    for ordinary in (
        'import os\nopen(os.environ["STUDIO_HOME"] + "/auth/config.json").read()',
        'import os\nopen(os.getenv("STUDIO_HOME") + "/auth/config.json").read()',
    ):
        assert tools._python_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary
    refused = 'import os\nopen(os.environ["UNSLOTH_STUDIO_HOME"] + "/auth/auth.db").read()'
    assert tools._python_exec(refused, None, 30, _SESSION, disable_sandbox = True) == (
        tools._STUDIO_CREDENTIAL_BLOCKED
    )


def test_a_foreign_environment_variable_is_not_the_studio_home(studio_home):
    home = studio_home
    # A dynamic path piece was attributed to the studio root whenever the snippet mentioned the
    # variable ANYWHERE, so an unrelated project directory read from a different variable was
    # refused. The expression that built the path is what decides it.
    ordinary = (
        'import os\nprint(os.environ["UNSLOTH_STUDIO_HOME"])\n'
        'project = os.environ["PROJECT_HOME"]\nopen(project + "/auth/config.json").read()'
    )
    assert tools._python_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
        tools._STUDIO_CREDENTIAL_BLOCKED
    )
    for refused in (
        'import os\nroot = os.environ["UNSLOTH_STUDIO_HOME"]\nopen(root + "/auth/auth.db").read()',
        'import os\nopen(os.environ["UNSLOTH_STUDIO_HOME"] + "/auth/auth.db").read()',
        'import os\nroot = os.getenv("UNSLOTH_STUDIO_HOME")\nopen(root + "/auth/auth.db").read()',
    ):
        assert tools._python_exec(refused, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), refused


def test_windows_spellings_of_the_root_are_read_as_paths(monkeypatch, tmp_path):
    # Platform independent, because the marker comparison is: a backslash path must not be mangled
    # by escape processing when the words are split, and a drive-qualified marker embedded inside
    # ANOTHER directory is a different path, exactly as a POSIX one is.
    home = r"c:\users\runneradmin\appdata\local\temp\studio-home"
    marker = home + r"\auth"
    assert tools._marker_is_a_path_segment("cat " + home + r"\auth\auth.db", marker)
    assert not tools._marker_is_a_path_segment(
        r"cat /mnt/backup" + home + r"\auth\config.json", marker
    )
    assert tools._quoted_words(r'find "C:\Users\me\Unsloth Studio" -type f') == [
        "find",
        r"C:\Users\me\Unsloth Studio",
        "-type",
        "f",
    ]


def test_an_escaped_space_in_the_root_still_matches(monkeypatch):
    # The default root has a space in it, so `cp -a /tmp/Studio\\ Home .` is the ordinary POSIX
    # spelling of the root itself and has to fold to the same directory as the quoted form.
    root = "/tmp/unsloth/studio home"
    monkeypatch.setattr(tools, "_studio_root_spellings", lambda: [root])
    for named in (
        "cp -a /tmp/unsloth/studio\\ home ./leak",
        'cp -a "/tmp/unsloth/studio home" ./leak',
    ):
        assert tools._names_the_studio_root_itself(named), named
    assert not tools._names_the_studio_root_itself(
        "cp -a /tmp/unsloth/studio\\ home/projects/p ./d"
    )


def test_the_root_spellings_cache_follows_a_changed_home(studio_home, tmp_path, monkeypatch):
    # The spellings are memoized per command, so a home that changes has to invalidate them or the
    # guard keeps answering for the previous install.
    assert tools._references_studio_credential_here(f'cp -a "{studio_home}" ./leak', None)
    other = tmp_path / "other-home"
    (other / "auth").mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(other))
    tools._studio_auth_markers_cache = None
    assert tools._references_studio_credential_here(f'cp -a "{other}" ./leak', None)
    assert not tools._references_studio_credential_here(f'cp -a "{studio_home}" ./leak', None)


def test_a_windows_root_is_matched_through_the_same_fold(monkeypatch):
    # `_folded_word` turns a backslash root into forward slashes, so the spellings it is compared
    # against have to be folded the same way. This shape only occurs on Windows, so it is driven
    # through the spellings directly and runs on every host.
    root = r"C:\Users\runneradmin\AppData\Local\Temp\pytest-0\test_x0\studio-home"
    monkeypatch.setattr(tools, "_studio_root_spellings", lambda: [root.lower()])
    for named in (
        f"find {root} -type f -exec cat {{}} +",
        f'tar -czf b.tgz "{root}"',
        f"cp -a {root}\\. ./copy",
    ):
        assert tools._names_the_studio_root_itself(named), named
    for ordinary in (f"find {root}\\projects\\p -type f", "find . -type f"):
        assert not tools._names_the_studio_root_itself(ordinary), ordinary


def test_a_windows_root_survives_python_literal_escaping(monkeypatch):
    # A python literal doubles the separators, so the cheap prefilter needs that spelling too.
    root = r"C:\Users\runner\Temp\studio-home"
    monkeypatch.setattr(tools, "_studio_home_lowered_cache", None)
    monkeypatch.setattr(tools, "_studio_home_for_guard", lambda: root)
    monkeypatch.setattr(
        tools,
        "_studio_auth_dir_markers",
        lambda: (((root + "/auth", "", root + "/auth"),), (), None),
    )
    code = "import shutil\nshutil.copytree(%r, './leak')" % root
    assert any(
        spelling in code.lower() for spelling in tools._studio_home_spellings_lowered()
    ), code
    # The same doubled spelling has to satisfy the root-naming test the guard gates on.
    monkeypatch.setattr(tools, "_studio_root_spellings", lambda: [root.lower()])
    assert tools._text_names_the_studio_root(code), code


def test_a_recursive_read_of_the_studio_root_is_refused(studio_home):
    home = studio_home
    # `find "$UNSLOTH_STUDIO_HOME" -type f -exec cat {} +` names no credential and no auth segment,
    # but the child inherits the variable and emits `auth/auth.db` and `.bootstrap_password`. Only
    # the forms that READ or COPY the tree count: a listing names files without opening them, and
    # work inside a project under the root is ordinary.
    for command in (
        'find "$UNSLOTH_STUDIO_HOME" -type f -exec cat {} +',
        'find "$UNSLOTH_STUDIO_HOME" -type f | xargs cat',
        f"find {home} -type f -exec cat {{}} +",
        f'tar -czf backup.tgz "{home}"',
        f'grep -r sk-unsloth "{home}"',
        f'cp -a "{home}" ./copy',
        f'rsync -av "{home}/" ./copy',
        # `7z a` recurses into a directory with no flag at all.
        f'7z a out.7z "{home}"',
        # The walker behind a wrapper, whose option value must not read as the command.
        'env -u FOO tar -czf b.tgz "$UNSLOTH_STUDIO_HOME"',
        'timeout 5 tar -czf b.tgz "$UNSLOTH_STUDIO_HOME"',
        # `<root>/.` is the root itself: `cp --help` defines `-a` as recursive.
        'cp -a "$UNSLOTH_STUDIO_HOME"/. ./copy',
        'cp -a "$UNSLOTH_STUDIO_HOME"/./ ./copy',
        f'zip -r out.zip "{home}"',
    ):
        assert tools._bash_exec(command, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), command
    for ordinary in (
        f'find "{home}" -type f',
        f'ls -la "{home}"',
        f'cp -r ./out "{home}/projects/p/sandbox/"',
        f'grep -r TODO "{home}/projects/p"',
        "grep -r TODO .",
        "tar -czf out.tgz .",
        "7z a out.7z src",
        f'du -sh "{home}"',
        # A walker's NAME as data is not a walk: `echo` is what runs here.
        'echo tar "$UNSLOTH_STUDIO_HOME"',
        'echo "rsync is a tool" "$UNSLOTH_STUDIO_HOME"',
        'printf "%s" "$UNSLOTH_STUDIO_HOME"',
        "cp -a ./src/. ./copy",
        'cp -a "$UNSLOTH_STUDIO_HOME/projects/p/." ./copy',
    ):
        assert tools._bash_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary


def test_a_recursive_python_copy_of_the_studio_root_is_refused(studio_home):
    # `shutil.copytree(<root>, "/tmp/leak")` carries `auth/` with it and names no credential, and
    # the copy is then an ordinary file nothing guards.
    for refused in (
        'import shutil, os\nshutil.copytree(os.environ["UNSLOTH_STUDIO_HOME"], "./leak")',
        # The constructor wraps the name of the source rather than replacing it.
        "import shutil, os\nfrom pathlib import Path\n"
        'shutil.copytree(Path(os.environ["UNSLOTH_STUDIO_HOME"]), "./leak")',
        # A recursive WALK of the root reaches the same files a copy of it does.
        "import os\nfrom pathlib import Path\n"
        'for p in Path(os.environ["UNSLOTH_STUDIO_HOME"]).rglob("*"):\n    print(p.read_text())',
        'import os\nfor r, d, f in os.walk(os.environ["UNSLOTH_STUDIO_HOME"]):\n    print(f)',
        # One binding between the root and the walker resolves to the same directory.
        'import os\nroot = os.environ["UNSLOTH_STUDIO_HOME"]\n'
        "for r, d, f in os.walk(root):\n    print(f)",
        'import shutil, os\nshutil.copytree(src = os.environ["UNSLOTH_STUDIO_HOME"], dst = "./l")',
        'import shutil, os\nshutil.make_archive("b", "zip", os.environ["UNSLOTH_STUDIO_HOME"])',
        f'import shutil\nshutil.copytree({str(studio_home)!r}, "./leak")',
    ):
        assert tools._python_exec(refused, None, 30, _SESSION, disable_sandbox = True) == (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), refused
    for ordinary in (
        'import shutil\nshutil.copytree("./src", "./dst")',
        'import shutil\nshutil.make_archive("b", "zip", "./src")',
        # A SHALLOW listing returns the root's own entry names and opens nothing inside `auth`,
        # exactly as the terminal side's plain root listing does.
        'import os\nprint(os.listdir(os.environ["UNSLOTH_STUDIO_HOME"]))',
        "import os\nfrom pathlib import Path\n"
        'print(list(Path(os.environ["UNSLOTH_STUDIO_HOME"]).iterdir()))',
        f'import shutil\nshutil.copytree({str(studio_home / "projects" / "p")!r}, "./dst")',
    ):
        assert tools._python_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
            tools._STUDIO_CREDENTIAL_BLOCKED
        ), ordinary


def test_a_parent_walk_from_path_home_lands_in_the_studio_root(studio_home):
    # Both environment builders set HOME to the session workdir, so `Path.home()` is `Path.cwd()`.
    refused = 'from pathlib import Path\nprint(open(Path.home().parents[1] / "auth" / "auth.db", "rb").read())'
    assert tools._python_exec(refused, None, 30, _SESSION, disable_sandbox = True) == (
        tools._STUDIO_CREDENTIAL_BLOCKED
    )
    ordinary = 'from pathlib import Path\nprint(open(Path.home() / "notes.txt").read())'
    assert tools._python_exec(ordinary, None, 30, _SESSION, disable_sandbox = True) != (
        tools._STUDIO_CREDENTIAL_BLOCKED
    )


def test_two_open_secrets_in_one_chunk_hold_from_the_first(studio_home):
    # `rfind` walks right to left and the loop stopped at the first hit, so a chunk holding two
    # open tokens emitted the earlier partial key.
    from core.inference.tool_stream_exec import _hold_back_partial_secret
    assert _hold_back_partial_secret("sk-unsloth-abcdsk-unsloth-") == 0
    assert _hold_back_partial_secret("sk-unsloth-0123 done") == len("sk-unsloth-0123 done")


def test_a_snippet_the_parser_cannot_hold_is_a_decision(studio_home):
    # `ast.parse` raises RecursionError on ~10k chained operators. Caught here, the guard returns a
    # decision; uncaught, the tool raised an internal exception instead of the interpreter's own
    # error. The credential hint is what gets the snippet this far at all.
    huge = "x = " + "+".join(["1"] * 20000) + "\n# auth\n"
    workdir = str(studio_home / "sandbox" / _SESSION)
    assert tools._python_builds_a_credential_path(huge, workdir) is False


def test_a_long_run_of_escapes_is_a_decision_rather_than_a_crash(studio_home):
    home = studio_home
    # Unescaping recursed once per backslash, so 550 of them raised RecursionError out of the guard,
    # before the command was classified at all.
    workdir = str(home / "sandbox" / _SESSION)
    assert tools._references_studio_credential_here("echo " + "\\" * 1100 + "a", workdir) is (False)
    # The one level it needs still works.
    assert (
        tools._references_studio_credential_here(
            'c\\d ../..; sqlite3 auth/auth.db "select jwt_secret from auth_user"', workdir
        )
        is True
    )


def test_a_bare_cd_is_a_move_back_to_the_sandbox(studio_home):
    home = studio_home
    # Both sandbox environments set HOME to the tool workdir, so a bare `cd` returns there and the
    # commands after it open from the sandbox again. Reading it as no move at all kept the studio
    # root live and refused `cd ../..; cd; cat auth/config.json`, which reads the project's own file.
    workdir = str(home / "sandbox" / _SESSION)
    for ordinary in (
        "cd ../..; cd; cat auth/config.json",
        "cd ../..; cd --; ls -a auth",
        "cd; cat notes.txt",
        "cd ../..; cd; cd auth; cat auth.db",
    ):
        assert tools._references_studio_credential_here(ordinary, workdir) is False, ordinary
    # ...and the return must not cost the real reads. A bare `cd` inside a subshell moves only
    # that subshell, and `cd -` after it goes back to the root.
    for refused in (
        "cd ../..; cat auth/auth.db",
        "cd ../..; (cd); cat auth/auth.db",
        "cd ../..; cd; cd -; cat auth/auth.db",
        "cd ../..; cd; cat ../../auth/auth.db",
    ):
        assert tools._references_studio_credential_here(refused, workdir) is True, refused


def test_a_return_reopens_the_directory_it_lands_in(studio_home):
    home = studio_home
    # `cd -` and `popd` land somewhere, and that is where every later relative path opens from.
    # Restoring the state without re-opening its span left nothing covering the read, so
    # `cd ../..; cd sandbox; cd -; cat auth/auth.db` walked back to the root unnoticed.
    workdir = str(home / "sandbox" / _SESSION)
    for refused in (
        "cd ../..; cd sandbox; cd -; cat auth/auth.db",
        "cd ../..; pushd /tmp; popd; cat auth/auth.db",
    ):
        assert tools._references_studio_credential_here(refused, workdir) is True, refused
    # A return that lands back in the sandbox is ordinary work.
    for ordinary in (
        "cd sub; cd -; cat auth/config.json",
        "pushd sub; popd; ls -a auth",
    ):
        assert tools._references_studio_credential_here(ordinary, workdir) is False, ordinary


def test_every_home_variable_is_checked_before_the_expansion(studio_home):
    home = studio_home
    # The expansion rewrites all of the studio-home names from their last assignment, so one name
    # that is only assigned must not authorize rewriting another whose assignment comes AFTER its
    # use: the shell expands that use to the INHERITED home, which is this install's own.
    workdir = str(home / "sandbox" / _SESSION)
    assert (
        tools._references_studio_credential_here(
            'STUDIO_HOME=/tmp/p; cat "$UNSLOTH_STUDIO_HOME/auth/auth.db"; '
            "UNSLOTH_STUDIO_HOME=/tmp/p",
            workdir,
        )
        is True
    )
    # A genuine rebind before the use still names the caller's own directory, not this install's.
    assert (
        tools._references_studio_credential_here(
            'UNSLOTH_STUDIO_HOME=/tmp/project; cat "$UNSLOTH_STUDIO_HOME/auth/config.json"',
            workdir,
        )
        is False
    )
