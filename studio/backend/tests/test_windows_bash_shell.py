# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The terminal tool must run bash on Windows, not cmd.

Models write bash for a shell tool, and every other platform runs bash. ``cmd /c``
executes only the first line of a multi-line command, leaves single quotes in
the argument, and does not understand bash quoting, so a correct script
half-executes and reports success. These run on every OS by faking the platform,
because studio-backend-ci is Linux-only.
"""

import os
import sys
import time
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import tools


@pytest.fixture(autouse = True)
def _clear_bash_cache():
    # Bound before the test so a monkeypatched _windows_bash (a plain lambda,
    # with no cache_clear) does not break teardown.
    cached = tools._windows_bash
    cached.cache_clear()
    yield
    cached.cache_clear()


def _fake_trusted_root(monkeypatch, root):
    """Point the Program Files trust check at ``root``.

    _windows_program_roots goes through SHGetKnownFolderPath, absent off
    Windows. The roots are faked here rather than %ProgramFiles%, which the
    resolver deliberately does not read (a caller could relocate the boundary).
    """
    monkeypatch.setattr(tools, "_windows_program_roots", lambda: [str(root)])


def test_posix_shell_is_unchanged(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert tools._get_shell_cmd("echo hi") == ["bash", "-c", "echo hi"]


def test_windows_uses_bash_when_present(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert tools._get_shell_cmd("echo hi") == [
        r"C:\Program Files\Git\bin\bash.exe",
        "-c",
        "echo hi",
    ]


def test_windows_falls_back_to_cmd_without_bash(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert tools._get_shell_cmd("echo hi") == ["cmd", "/c", "echo hi"]


def test_prefers_git_for_windows_over_path(monkeypatch, tmp_path):
    git_bash = tmp_path / "Git" / "bin" / "bash.exe"
    git_bash.parent.mkdir(parents = True)
    git_bash.write_text("")
    _fake_trusted_root(monkeypatch, tmp_path)
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(tools.shutil, "which", lambda _name: r"C:\somewhere\else\bash.exe")
    assert tools._windows_bash() == str(git_bash)


def test_untrusted_path_bash_is_rejected(monkeypatch, tmp_path):
    # A bash in a user-writable dir (Scoop, a per-user Git, a project checkout)
    # would run every sandboxed command, defeating the PATH/PATHEXT hardening in
    # _build_safe_env: the command passed the blocklist, then the shell itself is
    # attacker-controlled. cmd is worse to write for but stays trusted.
    scoop_bash = tmp_path / "scoop" / "shims" / "bash.exe"
    scoop_bash.parent.mkdir(parents = True)
    scoop_bash.write_text("")
    _fake_trusted_root(monkeypatch, tmp_path / "Program Files")
    monkeypatch.setattr(os, "environ", {"PATH": str(scoop_bash.parent)})
    monkeypatch.setattr(tools.shutil, "which", lambda _name: str(scoop_bash))
    assert tools._windows_bash() is None


def test_trusted_path_bash_behind_an_untrusted_shim_is_found(monkeypatch, tmp_path):
    # shutil.which stops at the first hit, so a user shim early on PATH used to
    # decide the answer for a host that does have a trusted bash.
    shim = tmp_path / "shims" / "bash.exe"
    shim.parent.mkdir(parents = True)
    shim.write_text("")
    trusted_dir = tmp_path / "Program Files" / "Git" / "bin"
    trusted_dir.mkdir(parents = True)
    (trusted_dir / "bash.exe").write_text("")
    _fake_trusted_root(monkeypatch, tmp_path / "Program Files")
    monkeypatch.setattr(
        os, "environ", {"PATH": os.pathsep.join([str(shim.parent), str(trusted_dir)])}
    )
    monkeypatch.setattr(tools.shutil, "which", lambda _name: str(shim))
    assert tools._windows_bash() == str(trusted_dir / "bash.exe")


@pytest.mark.parametrize(
    "wsl_path",
    [
        r"C:\Windows\System32\bash.exe",
        r"C:\Users\me\AppData\Local\Microsoft\WindowsApps\bash.exe",
        "C:/Windows/System32/bash.exe",
    ],
)
def test_wsl_launcher_is_rejected(monkeypatch, wsl_path):
    # WSL's bash runs in a different filesystem, so the sandbox workdir would not
    # apply. Falling back to cmd is worse but stays inside the sandbox. The dir
    # is trusted here so the marker is the only thing that can reject these.
    monkeypatch.setattr(tools, "_is_trusted_windows_program_dir", lambda _dir: True)
    assert tools._is_trusted_windows_bash(wsl_path) is False


def test_windowsapps_under_program_files_is_still_rejected(monkeypatch, tmp_path):
    # A store package installs under Program Files, so the trust check alone
    # would accept the WSL shim there.
    store_bash = tmp_path / "WindowsApps" / "bash.exe"
    store_bash.parent.mkdir(parents = True)
    store_bash.write_text("")
    _fake_trusted_root(monkeypatch, tmp_path)
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(tools.shutil, "which", lambda _name: str(store_bash))
    assert tools._windows_bash() is None


def test_no_bash_anywhere_returns_none(monkeypatch):
    monkeypatch.setattr(tools, "_windows_program_roots", lambda: [])
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(tools.shutil, "which", lambda _name: None)
    assert tools._windows_bash() is None


def test_blocklist_lexes_bash_syntax_when_the_shell_is_bash(monkeypatch):
    # The scan is keyed to the shell, not the OS: the non-posix lexer never
    # splits on `;`, so a blocked command after a control-flow keyword stayed in
    # argument position and `if true; then rm -rf x; fi` really ran under bash.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert "rm" in tools._find_blocked_commands("if true; then rm -rf x; fi")
    assert "curl" in tools._find_blocked_commands("for i in 1; do curl http://x; done")


@pytest.mark.parametrize("command", ["'rm' -rf x", '"rm" -rf x', "echo hi; 'rm' -rf x"])
def test_blocklist_sees_through_quoting_when_the_shell_is_bash(monkeypatch, command):
    # The non-posix lexer keeps the quote marks and _token_basename strips only
    # meta-chars, so `'rm'` never matched `rm`. Harmless in front of cmd, where
    # it is a literal unknown program; under bash it really deletes.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert "rm" in tools._find_blocked_commands(command)


def test_blocklist_still_catches_the_plain_form_under_cmd(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert "rm" in tools._find_blocked_commands("rm -rf x")


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows shell behaviour")
def test_multiline_script_runs_every_line_on_windows():
    # The regression itself: under cmd /c only the first line ran, so the loop
    # body and the redirect were silently dropped.
    if tools._windows_bash() is None:
        pytest.skip("no native Win32 bash on this host")
    script = "\n".join(
        [
            "value=unsloth",
            "for i in 1 2 3; do",
            '  echo "line $i $value"',
            "done",
        ]
    )
    out = tools._bash_exec(script)
    for expected in ("line 1 unsloth", "line 2 unsloth", "line 3 unsloth"):
        assert expected in out, out


def test_paths_note_names_the_real_platform():
    # A note that only cites /mnt/data and /tmp/outputs reads as "you are on
    # Linux", and models then decline to launch Windows programs that do exist.
    note = tools._SANDBOX_PATHS_NOTE
    if sys.platform == "win32":
        assert "Windows" in note
        assert "/mnt/data" not in note
        assert "/tmp/outputs" not in note
    else:
        assert "/mnt/data" in note


def test_shell_note_names_the_shell_that_will_run(monkeypatch):
    # Telling a model it has bash on a host that fell back to cmd reintroduces
    # the multi-line half-execution this note exists to prevent.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert "The shell is bash" in tools._build_terminal_shell_note()
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    cmd_note = tools._build_terminal_shell_note()
    assert "cmd, not bash" in cmd_note
    assert "one command per call" in cmd_note


def test_posix_shell_note_is_empty(monkeypatch):
    # The POSIX descriptions are unchanged by this PR.
    monkeypatch.setattr(sys, "platform", "linux")
    assert tools._build_terminal_shell_note() == ""


@pytest.mark.parametrize("bash", [r"C:\Program Files\Git\bin\bash.exe", None])
def test_notes_never_recommend_a_blocked_program(monkeypatch, bash):
    # powershell/pwsh are in _BLOCKED_COMMANDS_WIN, so a sandboxed user who
    # follows the prompt gets a hard block instead of a command. `cmd /c start`
    # is not blocked, which makes naming it worse: it advertises the gap.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    text = (tools._build_sandbox_paths_note() + tools._build_terminal_shell_note()).lower()
    for program in tools._BLOCKED_COMMANDS_WIN:
        assert program not in text
    assert "start-process" not in text
    assert "/c start" not in text


def test_terminal_tool_description_is_the_two_notes_appended():
    # _TERMINAL_SHELL_NOTE is "" off Windows, so `note in description` proves
    # nothing about the shell half there. Compare the whole string instead, which
    # at least pins the composition, and cover the note's own content below.
    description = tools.TERMINAL_TOOL["function"]["description"]
    assert description == (
        "Execute a terminal command and return stdout/stderr."
        + tools._SANDBOX_PATHS_NOTE
        + tools._TERMINAL_SHELL_NOTE
    )


@pytest.mark.parametrize("bash", [r"C:\Program Files\Git\bin\bash.exe", None])
def test_windows_shell_note_is_never_empty(monkeypatch, bash):
    # The half the test above cannot reach on a Linux runner: whichever shell the
    # resolver lands on, the terminal description gains a sentence naming it.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    note = tools._build_terminal_shell_note()
    assert note.startswith(" The shell is ")
    assert note not in tools._build_sandbox_paths_note()


def test_python_tool_description_omits_the_shell():
    # The shell note on the python description would point a model at
    # subprocess/os.system as a way around the terminal blocklist, and none of
    # it applies to the python sandbox anyway.
    description = tools.PYTHON_TOOL["function"]["description"]
    assert tools._SANDBOX_PATHS_NOTE in description
    assert "shell" not in description.lower()
    if sys.platform == "win32":
        assert tools._TERMINAL_SHELL_NOTE not in description


def test_notes_say_where_commands_run(monkeypatch):
    # Without this, models decline to launch a window they believe the user
    # cannot see, and hand back manual instructions instead.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert "user's own machine" in tools._build_sandbox_paths_note()
    assert "opens a window on the user's desktop" in tools._build_terminal_shell_note()


@pytest.fixture
def _windows_blocklist(monkeypatch):
    # _BLOCKED_COMMANDS resolves the Windows names at import, so patching
    # sys.platform is too late; fake the resolved set instead. Kept alongside
    # windows_terminal, which also fixes which lexer runs; main's tests below
    # take this one.
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )


@pytest.fixture(params = ["git-bash", "cmd-fallback"])
def windows_terminal(request, monkeypatch):
    """A Windows host, in both shell configurations _get_shell_cmd produces.

    Faking sys.platform, which is all the rest of this file needs, cannot reach
    _BLOCKED_COMMANDS: it folds in _BLOCKED_COMMANDS_WIN at import, so on the
    Linux runner powershell/pwsh are not blocked names and every assertion below
    would pass or fail for the wrong reason. Both shells are parameters because
    with no trusted bash the blocklist lexes with shlex(posix = False), which
    keeps the quote marks cmd screening exists for.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        tools,
        "_windows_bash",
        (lambda: r"C:\Program Files\Git\bin\bash.exe")
        if request.param == "git-bash"
        else (lambda: None),
    )
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )


@pytest.fixture
def windows_cmd_only(monkeypatch):
    """A Windows host with no trusted bash, so cmd does the parsing.

    The blocklist is arranged exactly as ``windows_terminal`` does it, for the
    reason that fixture gives: powershell is only a blocked name once
    _BLOCKED_COMMANDS_WIN is folded in.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )


@pytest.fixture
def windows_git_bash_only(monkeypatch):
    """The same host with a trusted Git Bash, so bash does the parsing."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )


@pytest.mark.parametrize(
    "command",
    [
        "cmd /c powershell -Command ls",
        "cmd //c powershell -Command ls",
        "cmd //k powershell -Command ls",
        'cmd /c "powershell -Command ls"',
        'cmd //c start "" powershell -Command ls',
        'cmd //c start /b "" pwsh -Command ls',
        'cmd //c start //min "" powershell -Command ls',
        r'cmd //c start /d C:/tmp "" powershell -Command ls',
        # start quotes the shell it launches too, and the leading quote hid that
        # token from the shell-name lookup, so nothing recursed into the tail.
        'cmd //c start "" "cmd" /c powershell -Command ls',
        # A quoted first argument is START's window title whatever it holds, so
        # the program is one token further on than the `""` idiom implies. A
        # title with a space stays provable after posix lexing drops the marks,
        # because nothing else survives as one token.
        'cmd //c start "my window" pwsh -Command ls',
        # /s strips the outer pair off the whole payload, leaving the lexer an
        # empty first token with the program behind it.
        'cmd /s /c ""powershell" -Command ls"',
        # A program path holding spaces is ONE quoted word, so re-lexing it
        # reads `C:\Program` and leaves the executable in argument position.
        r'cmd /c "C:\Program Files\PowerShell\7\pwsh.exe" -Command ls',
        # START reaches the same shape behind its title.
        r'cmd //c start "" "C:\Program Files\PowerShell\7\pwsh.exe" -Command ls',
        # cmd searches PATHEXT, so the suffix is optional...
        r'cmd /c "C:\Program Files\PowerShell\7\pwsh" -Command ls',
        r'cmd //c start "" "C:\Program Files\PowerShell\7\pwsh" -Command ls',
        # ...and it expands %VAR% before reading the path.
        r'cmd /c "%ProgramFiles%\PowerShell\7\pwsh.exe" -Command ls',
        r'cmd /c "%ProgramFiles%\PowerShell\7\pwsh" -Command ls',
    ],
)
def test_cmd_shellout_is_screened_through_mangled_switches(windows_terminal, command):
    # Git Bash turns a lone /c into a path, so a model writes //c. That spelling
    # skipped the nested scan, making `cmd //c powershell` reachable where
    # `cmd /c powershell` was blocked, and `start` launches its argument too.
    # The quoted spellings reach the scan with their quotes on (_unwrap_quotes).
    # windows_terminal fakes the resolved blocklist the way #7934 did, and adds
    # the shell, since which lexer runs is what decides half of these cases.
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd //c start "" bash -c "bash /c/x.sh"',
        "cmd //c start wt",
        "cmd //c dir",
        "start notepad",
        # Skipping the title must not start blocking an ordinary argument.
        "cmd //c start notepad readme.txt",
        'cmd //c start "job" notepad',
    ],
)
def test_detached_windows_stay_launchable(windows_terminal, command):
    # `start` is the only route to a window on the user's desktop, which the
    # terminal description promises, so screening must not blanket-block cmd.
    # #7936 added this with its own blocklist fixture; windows_terminal fakes the
    # same set AND fixes which lexer runs, so the two were folded together rather
    # than left as one name defined twice, where the second silently won.
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        """bash -c '""rm -rf x""'""",
        """bash -c '""curl http://x""'""",
        """sh -c '""sudo rm -rf /""'""",
        """cmd /c '""rm -rf x""'""",
    ],
)
def test_doubled_quotes_do_not_hide_the_nested_command(command):
    # The two quote marks belong to DIFFERENT spans, so posix shlex hands the
    # recursion `rm -rf x` and blocks it. Unwrapping a pair there would leave
    # one fully quoted span, lexing to the single word `rm -rf x`, which is no
    # blocked name and which the regex cannot reach behind a quote. bash runs
    # the payload for real, so that path must stay screened.
    assert tools._find_blocked_commands(command)


def test_cmd_runs_only_double_quotes(monkeypatch):
    # cmd has no single-quote syntax: it looks for a program literally named
    # `'powershell`. Blocking these would refuse a line cmd cannot execute, so
    # unwrapping is limited to `"`. Pinned to the cmd lexer because under bash
    # the same spelling really does reach powershell, and is blocked.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )
    assert not tools._find_blocked_commands("cmd /c 'powershell -Command ls'")
    assert not tools._find_blocked_commands("cmd //c start '' powershell -Command ls")
    assert tools._find_blocked_commands('cmd /c "powershell -Command ls"')
    # The path recovery follows the same rule, or it reconstructs a program out
    # of a path cmd would look for literally, single quotes and all.
    assert not tools._find_blocked_commands(
        r"cmd /c 'C:\Program Files\PowerShell\7\pwsh.exe' -Command ls"
    )
    assert tools._find_blocked_commands(
        r'cmd /c "C:\Program Files\PowerShell\7\pwsh.exe" -Command ls'
    )


def test_a_drive_path_after_start_is_the_program_not_a_switch(windows_terminal):
    # Git Bash hands `/c/Windows/...` to cmd as a C: executable path, but every
    # documented START option is one word after the slash, so skipping anything
    # slash-prefixed walked past the program onto its arguments.
    assert tools._find_blocked_commands(
        'cmd //c start "" /c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -Command ls'
    )


def test_the_s_switch_payload_keeps_a_spaced_program_path(monkeypatch):
    # /s strips the outer pair off the whole payload, so the program path is
    # split across tokens with a stray mark on each. Pinned to the cmd lexer:
    # posix shlex reads the backslashes as escapes and mangles the path.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )
    assert tools._find_blocked_commands(
        r'cmd /s /c ""C:\Program Files\PowerShell\7\pwsh.exe" -Command ls"'
    )
    # A path named in passing is still an argument, not a program.
    assert not tools._find_blocked_commands(r'cmd /c "ls /usr/bin/rm"')


def test_a_full_path_still_names_the_nested_shell(windows_terminal):
    # os.path.basename leaves a backslash path whole off Windows, so
    # `C:\Windows\System32\cmd.exe` matched no shell and the second /c payload
    # was never read. Asserting the program, not merely that something was
    # blocked: the previous revision reported the POSIX source builtin here,
    # which would have kept this test green for the wrong reason.
    assert "powershell" in tools._find_blocked_commands(
        r'cmd /c "C:\Windows\System32\cmd.exe" /c powershell -Command ls'
    )


def test_git_bash_hands_cmd_its_own_quotes(monkeypatch):
    # Under Git Bash the outer posix lexer strips only the shell's quoting, so
    # cmd's own pair survives into the payload and cmd unquotes and runs it.
    # Scanned in ADDITION to the quoted form, never instead: the two spans of
    # `""rm -rf x""` are what keep that one from collapsing to a single word.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )
    assert tools._find_blocked_commands("""cmd //c '"powershell -Command ls"'""")
    assert "rm" in tools._find_blocked_commands("""cmd /c '""rm -rf x""'""")


@pytest.mark.parametrize(
    "command",
    [
        r'cmd //c start "" C:\tmp\image.png',
        r'cmd //c start "job" C:\tmp\readme.txt',
        r'cmd //c start "" "C:\Users\me\My Documents\report.docx"',
    ],
)
def test_start_still_opens_a_document(windows_terminal, command):
    # File association is what START is for. Re-scanning the target as a command
    # line reads the extension dot as the POSIX `.` builtin, which turned every
    # absolute Windows path into a hard block. A bare word is a program name.
    assert not tools._find_blocked_commands(command)


def test_a_quoted_command_line_after_start_is_still_lexed():
    # The other half: quoting can hand the whole line over as one token, and an
    # unbalanced quote drops the scan onto split(), whose tokens keep their
    # marks. Both have to be lexed again or the program is never read.
    assert "rm" in tools._find_blocked_commands('start "" "rm -rf x"')
    assert "curl" in tools._find_blocked_commands('start ""curl http://x"')


@pytest.mark.parametrize(
    "command",
    [
        r'cmd /c "echo C:\tmp\pwsh.exe"',
        r'cmd /c "dir C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"',
        r'cmd /c "ls /usr/bin/rm"',
    ],
)
def test_a_path_named_in_passing_is_not_a_program(windows_terminal, command):
    # The split-path recovery reads the program a payload OPENS with. Scanning
    # every word for an executable suffix instead turns printing or listing a
    # PowerShell path into a hard block.
    assert not tools._find_blocked_commands(command)


def test_cmd_reads_a_one_word_start_title(monkeypatch):
    # Only the cmd lexer keeps the marks that prove `"job"` is a title rather
    # than the program, so this spelling is screened there and not under bash.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )
    assert tools._find_blocked_commands('cmd //c start "job" powershell -Command ls')
    assert tools._find_blocked_commands('cmd //c start /b "job" powershell -Command ls')


def test_an_unquoted_start_program_keeps_its_arguments(monkeypatch):
    # Posix lexing dropped the marks, so `notepad` could have been a title. It
    # is taken as the program: `start notepad <file>` is the ordinary shape, and
    # guessing the other way blocks the argument rather than a launched program.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )
    assert not tools._find_blocked_commands("cmd //c start notepad powershell -Command ls")
    # A space could only have come from quoting, so that one is still a title.
    assert tools._find_blocked_commands('cmd //c start "my window" pwsh -Command ls')


@pytest.mark.parametrize(
    "command",
    [
        r'cmd //c start "" "C:\Users\me\My Documents\powershell.docx"',
        r'start "" "C:\Users\me\My Reports\rm.docx"',
        r'cmd /c start "" "C:\Users\me\My Notes\curl.pdf"',
    ],
)
def test_a_document_named_after_a_shell_is_still_a_document(windows_terminal, command):
    # Dropping the .exe requirement so PATHEXT spellings resolve also accepted
    # every other suffix, so a file called powershell.docx read as the program.
    # cmd hands a non-executable suffix to its file association, never runs it.
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd //c start "my window" /min powershell -Command ls',
        'cmd //c start "my window" /b ssh a@b',
        'start "my window" /min pwsh -Command ls',
    ],
)
def test_switches_may_follow_the_title(windows_terminal, command):
    # START takes its title first and its switches after, so the walk has to
    # keep alternating rather than stop at the title. Only the cmd lexer kept
    # the marks that prove one, so this went unread under Git Bash.
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'echo start "" powershell',
        'echo start "job" powershell -Command ls',
        'cmd //c echo start "" powershell',
    ],
)
def test_a_printed_start_is_not_a_launched_one(windows_terminal, command):
    # The word has to sit where the shell would RUN it. Screening it anywhere
    # turns printing the line into a hard block, and this scan cannot borrow the
    # main loop's command position: the outer shell runs only token 0 of
    # `cmd //c start "" powershell`, which is the case the walk exists for.
    assert not tools._find_blocked_commands(command)


def test_a_quoted_command_line_is_not_only_a_title(monkeypatch):
    # Under bash a title is provable only by its whitespace, and a quoted
    # command line looks identical: `start "ssh a@b"` has no program after it
    # because that WAS the program. Screened as well as skipped over.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )
    assert "ssh" in tools._find_blocked_commands('cmd //c start "ssh a@b"')
    assert "rm" in tools._find_blocked_commands("cmd //c start 'sudo rm -rf /'")


@pytest.mark.parametrize(
    "command",
    [
        'ls\nstart "" powershell -Command ls',
        'echo hi\nstart "" ssh a@b',
        'FOO=1 start "" powershell -Command ls',
        'env start "" powershell -Command ls',
        'nohup start "" powershell -Command ls',
        'echo x | xargs start "" powershell -Command ls',
    ],
)
def test_a_launched_start_is_read_wherever_the_shell_runs_it(windows_terminal, command):
    # Command position is what the main scan already decides, wrappers and
    # assignment prefixes and all, so the walk reuses that verdict. Reading only
    # the token before START instead missed every one of these, each of which
    # really launches. A newline separates commands but lexes as whitespace,
    # leaving no token to look back at, so it is recovered from the raw text.
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    ['time start "" powershell -Command ls', "time powershell -Command ls"],
)
def test_a_posix_builtin_wrapper_is_read_only_under_bash(windows_git_bash_only, command):
    # `time` runs the command behind it under bash. Under cmd it is `time [/t]`,
    # which sets the clock, so nothing behind it is launched there and the cmd
    # lane must not read it as a wrapper.
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        "cmd /c time powershell",
        "cmd /c command powershell",
        "cmd /c exec powershell",
        "cmd /c builtin powershell",
    ],
)
def test_a_posix_builtin_wrapper_is_not_read_in_a_cmd_payload(windows_terminal, command):
    # The other half. A `/c` payload is cmd's line whatever lexed the outer one,
    # so bash's own builtins do not run anything inside it.
    assert not tools._find_blocked_commands(command)


def test_a_newline_inside_quotes_does_not_start_a_command(windows_terminal):
    # Only a newline the quoting left bare ends a command. One inside a quoted
    # word is data the command receives, and treating it as a separator would
    # read printed text as a launch.
    assert not tools._find_blocked_commands("echo \"hi\nstart '' powershell\"")


def test_a_continued_line_is_one_command(windows_terminal):
    # A backslash before the newline joins the two lines, so what follows is one
    # more argument rather than a new command. Verified against bash: the second
    # line is printed, not launched. Only a newline the quoting left bare counts,
    # and an escaped one is quoted like any other character.
    assert not tools._find_blocked_commands('echo hi \\\nstart "" powershell')
    assert not tools._find_blocked_commands('echo one two \\\nthree start "" powershell')


@pytest.mark.parametrize(
    "command",
    [
        '\nstart "" powershell -Command ls',
        'echo hi\n\n\nstart "" powershell -Command ls',
        'echo hi\r\nstart "" powershell -Command ls',
    ],
)
def test_blank_and_leading_lines_still_open_a_command(windows_terminal, command):
    # The recovery counts marks rather than measuring gaps, so a run of newlines,
    # a leading one and a CRLF pair all open exactly one command between them.
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd //c env start "" powershell -Command ls',
        'cmd //c FOO=1 start "" powershell -Command ls',
        'echo hi\nFOO=1 start "" powershell -Command ls',
    ],
)
def test_a_prefix_may_stand_between_a_boundary_and_start(windows_terminal, command):
    # A newline and cmd's own /c each open a command position, and a wrapper or
    # assignment prefix may sit in it before START does. Blessing only the first
    # word after the boundary left every one of these launches unscreened, so
    # both feed the same prefix walk an ordinary command position uses.
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'echo "cmd" /c powershell',
        "echo C:\\Windows\\System32\\cmd.exe /c powershell",
        'echo /c start "" powershell',
        'echo /k start "" powershell',
    ],
)
def test_a_shell_named_in_passing_opens_no_payload(windows_terminal, command):
    # Only a cmd the shell RUNS hands anything to a /c. Recovering the name by
    # unquoting and normalising any previous word, and taking any /c-shaped token
    # as a handoff, turned printing these lines into a hard block.
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        "echo ';' start \"\" powershell",
        'echo "then" start "" powershell',
        "echo '|' start \"\" powershell",
    ],
)
def test_a_quoted_separator_before_start_is_still_data(windows_terminal, command):
    # The posix lexer splits real separators into tokens of their own and records
    # which of them the quoting only made look that way. Reading the previous
    # token instead second-guessed it, and a printed `';'` or `"then"` read as
    # though a command followed it.
    assert not tools._find_blocked_commands(command)


def test_a_quoted_cmd_payload_is_not_a_program_name(windows_terminal):
    # Marking what /c hands over as a command position also offered the whole
    # quoted line to the blocklist as one word, and os.path.basename read its
    # last path segment: listing a directory came back as the rm builtin.
    assert not tools._find_blocked_commands('cmd /c "ls /usr/bin/rm"')
    assert not tools._find_blocked_commands('cmd /c "dir C:\\tmp"')


@pytest.mark.parametrize(
    "command",
    [
        '"C:\\Windows\\System32\\cmd.exe" /c start "" powershell -Command ls',
        'C:/Windows/System32/cmd.exe /c start "" powershell -Command ls',
        '"C:\\Windows\\System32\\cmd.exe" /c env start "" powershell -Command ls',
    ],
)
def test_a_full_path_cmd_still_opens_its_payload(windows_terminal, command):
    # A shell can be spelled as a full path, and os.path.basename leaves a
    # backslash path whole off Windows, so the /c opened no command position and
    # the START behind it stayed in argument position. Normalised the way the
    # nested-shell lookup already normalises the same spelling.
    assert tools._find_blocked_commands(command)


def test_bash_eats_an_unquoted_backslash_path(monkeypatch):
    # Unquoted, those backslashes are bash escapes and the program name arrives
    # as C:WindowsSystem32cmd.exe, which launches nothing. Checked against bash
    # rather than assumed. Quoting is what carries the path through, and that
    # spelling is covered above.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )
    assert "powershell" not in tools._find_blocked_commands(
        'C:\\Windows\\System32\\cmd.exe /c start "" powershell'
    )


def test_a_bash_keyword_boundary_belongs_to_bash(monkeypatch):
    # `;` and `then` are bash syntax. Under Git Bash they open a command and the
    # START behind them launches; cmd has neither, so the same line only prints.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert "rm" in tools._find_blocked_commands('if x; then start "" rm -rf x')
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert not tools._find_blocked_commands('cmd /c echo hi; start "" powershell')


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c echo hi; start "" powershell',
        'cmd /c echo hi ^& start "" powershell',
        'cmd /c echo hi ^^^& start "" powershell',
    ],
)
def test_cmd_operators_decide_a_cmd_boundary(monkeypatch, command):
    # cmd's control operators are &, &&, ||, | and parentheses. A `;` is a plain
    # argument separator there, and a caret hands the operator after it over as
    # text, so each of these prints its whole line and launches nothing.
    # Asked of the cmd lexer alone: bash has neither rule, and the same lines
    # really do run START there, which the next test pins.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c echo hi; start "" powershell',
        'cmd /c echo hi ^& start "" powershell',
    ],
)
def test_bash_has_neither_rule(monkeypatch, command):
    # Under bash a `;` separates and a `^` is an ordinary character, so the `&`
    # behind it is still an operator and START really launches. The caret rule
    # must not leak across to the lexer it does not belong to.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )
    assert tools._find_blocked_commands(command)


def test_a_live_cmd_operator_still_opens_a_command(monkeypatch):
    # The other side of it: a bare `&`, and an escaped caret followed by a live
    # `&`, both really do start a second command.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )
    assert tools._find_blocked_commands('cmd /c echo hi& start "" powershell')
    assert tools._find_blocked_commands('cmd /c echo hi ^^& start "" powershell')


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c start "" https://example.com/powershell',
        'cmd /c start "title" "https://example.com/powershell"',
        'cmd /c start "" http://x/rm',
    ],
)
def test_start_browses_a_url_rather_than_running_it(windows_terminal, command):
    # START hands a URL to the default browser, the same association route as the
    # document targets. Reading its last path segment as the program refused the
    # link outright.
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        "find . -exec bash -c 'rm -rf x' ;",
        "fd -x bash -c 'rm -rf x'",
        "find . -exec sh -c 'curl http://x' ;",
        "find . -exec env bash -c 'rm -rf x' ;",
    ],
)
def test_an_exec_launched_shell_still_has_its_payload_read(monkeypatch, command):
    # find and fd run their -exec child themselves, so it is a command position
    # the main walk never reaches: that walk sees `find` and stops. Requiring the
    # nested-shell name to be at command position without publishing these first
    # meant the `-c` payload went unread, and `rm -rf x` really deletes.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd //c start "" env bash -c "rm -rf x"',
        'cmd //c start "" env cmd /c powershell -Command ls',
    ],
)
def test_start_follows_the_prefix_it_launches(windows_terminal, command):
    # A wrapper is a command in its own right and a step on the way to one, the
    # same shape `-exec` already models. Naming only `env` left the shell it
    # forwards to out of command position, so its payload was never read.
    assert tools._find_blocked_commands(command)


def test_an_empty_pair_is_read_by_where_it_sits(monkeypatch):
    # `""cmd` closes a zero-length span glued to the program, and the shell drops
    # the marks before anything sees argv, so cmd really runs. `"" rm` is a
    # different line: a genuine empty argument, and the wrapper in front of it
    # tries to run nothing at all. Only the whitespace between them tells the two
    # apart, so the raw text decides rather than the token alone.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )
    assert "powershell" in tools._find_blocked_commands('""cmd /c powershell""')
    assert "powershell" in tools._find_blocked_commands('cmd //c start "" ""cmd /c powershell""')
    assert not tools._find_blocked_commands('xargs "" rm')
    assert not tools._find_blocked_commands('find . -exec "" rm ;')


def test_an_earlier_pair_does_not_vouch_for_a_later_one(monkeypatch):
    # Gluing is a fact about one position, so it is asked positionally. Searching
    # the line for the two spellings joined let the `""rm` echoes here vouch for
    # the `"" rm` after them, which is a separate command that runs nothing.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )
    assert not tools._find_blocked_commands('echo ""rm && xargs "" rm')
    assert "powershell" in tools._find_blocked_commands('echo hi && ""cmd /c powershell""')


@pytest.mark.parametrize(
    "command",
    [
        "cmd /c powershell&echo ok",
        "cmd /c powershell|more",
        "cmd /c powershell&&echo ok",
    ],
)
def test_a_glued_cmd_operator_hides_a_second_command(windows_terminal, command):
    # cmd needs no whitespace around & or |, so a payload can carry a whole
    # second command with nothing separating the words. Screened as one program
    # name, `powershell&echo` matched nothing and the launch went unread.
    assert "powershell" in tools._find_blocked_commands(command)


def test_the_operator_test_reads_carets():
    # An odd run of carets hands the operator over as text; an even run is
    # escaped carets in front of a live one. Asserted on the helper rather than
    # end to end, because the regex backstop matches a blocked word after any
    # `&` whether or not a caret escaped it, and does so on main too. That is a
    # separate false positive, called out in the PR as not fixed here.
    assert tools._has_bare_cmd_operator("powershell&echo ok")
    assert tools._has_bare_cmd_operator("a^^&powershell")
    assert not tools._has_bare_cmd_operator("a^&powershell")
    assert not tools._has_bare_cmd_operator("plain text")


def test_a_caret_continues_a_cmd_line(monkeypatch):
    # A caret at the end of a cmd line continues it, the way a backslash does
    # under bash, so START is one more argument of the echo and only prints.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )
    assert not tools._find_blocked_commands('cmd /c echo hi ^\nstart "" powershell')
    # An even run is escaped carets, so the newline still separates.
    assert tools._find_blocked_commands('cmd /c echo hi ^^\nstart "" powershell')


@pytest.mark.parametrize(
    "command",
    [
        'cmd //c start "" "C:\\tmp\\rm report.docx"',
        'cmd //c start "" "C:\\my docs\\curl notes.pdf"',
    ],
)
def test_a_document_path_with_spaces_is_still_a_document(windows_terminal, command):
    # A quoted document path holds spaces like any other, and re-lexing it as a
    # command line read `rm report.docx` as the rm builtin. Its suffix is not one
    # cmd executes, so START opens it through its association.
    assert not tools._find_blocked_commands(command)


def test_a_quoted_command_line_after_start_is_not_a_document(windows_terminal):
    # The other side: no path and no suffix cmd would associate, so this really
    # is a command line and must still be lexed.
    assert "rm" in tools._find_blocked_commands('start "" "rm -rf x"')
    assert "rm" in tools._find_blocked_commands('start "" "rm -rf x.txt"')


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c "C:\\powershell scripts\\notepad.exe"',
        'start "" "C:\\powershell scripts\\notepad.exe"',
        'cmd /c "C:\\rm backups\\notepad.exe"',
    ],
)
def test_a_folder_name_is_not_the_program(windows_terminal, command):
    # A program path holds spaces like any other, so re-lexing one read its
    # directory components as words and reported the folder the executable sits
    # in. The path shape and an executable suffix say to read the program off the
    # path instead of scanning the whole thing as a shell line.
    assert not tools._find_blocked_commands(command)


def test_a_program_path_is_still_read(windows_terminal):
    # The other side: the same shape, and this one really is a blocked shell.
    assert "pwsh" in tools._find_blocked_commands(
        'cmd /c "C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command ls'
    )
    # And a real command line handed over in quotes is not a path, so it is lexed.
    assert "rm" in tools._find_blocked_commands('start "" "rm -rf x"')


def test_cmd_has_no_semicolon_separator(monkeypatch):
    # The main scan treats `;` as a separator for bash and records what follows
    # as a command. cmd never opens a command there, so trusting that set read
    # `echo ; start "" powershell` as a launch when it only prints.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert not tools._find_blocked_commands('echo ; start "" powershell')
    assert not tools._find_blocked_commands('cmd /c echo hi; start "" powershell')
    # Under bash the same line really does open a command.
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert "powershell" in tools._find_blocked_commands('echo ; start "" powershell')


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c "C:\\tools\\rm.exe -rf x"',
        'start "" "C:\\tools\\rm.exe -rf x"',
        'cmd /c "C:\\tools\\pwsh.exe -Command ls"',
    ],
)
def test_a_program_path_with_arguments_is_still_read(windows_terminal, command):
    # os.path.splitext keeps everything after the last dot, so this path plus its
    # arguments came back with a suffix of `.exe -rf x`. Calling that a document
    # meant START skipped the line rather than screening it. A real suffix
    # carries no whitespace.
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c "C:\\tools\\notepad.exe -x y"',
        'start "" "C:\\tools\\notepad.exe -x y"',
    ],
)
def test_arguments_after_a_program_path_are_not_a_command(windows_terminal, command):
    # The other half: a path followed by its own arguments is still a program.
    # Re-lexing it as a command line matched the POSIX source builtin on the
    # extension dot and refused an ordinary launch.
    assert not tools._find_blocked_commands(command)


def test_an_operator_still_wins_over_the_path_shape(windows_terminal):
    # An operator means a second command follows whatever the first looks like.
    assert "rm" in tools._find_blocked_commands('cmd /c "C:\\tools\\notepad.exe&rm -rf x"')


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c "C:\\tools.v2\\pwsh"',
        'start "" "C:\\tools.v2\\pwsh"',
    ],
)
def test_a_dotted_directory_does_not_supply_the_extension(windows_terminal, command):
    # os.path.splitext does not split a backslash off Windows, so splitting the
    # whole path let the DIRECTORY supply the suffix: `C:\tools.v2\pwsh` came
    # back with `.v2\pwsh`, no executable suffix, and the shell went unreported.
    # The basename comes first now, everywhere a suffix is read.
    assert "pwsh" in tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c "C:\\powershell scripts\\notepad.exe -x"',
        'start "" "C:\\powershell scripts\\notepad.exe -x"',
    ],
)
def test_a_spaced_path_with_arguments_is_recovered_whole(windows_terminal, command):
    # Reading only the first whitespace-delimited word missed that the directory
    # itself holds a space, so the payload fell through to a full re-lex and the
    # folder `powershell scripts` was reported for a line that runs notepad.
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize("depth", [1, 2, 3, 8])
def test_a_launcher_and_a_shell_hand_command_position_to_each_other(windows_terminal, depth):
    # Each scan opens command positions the other reads, so reading each list
    # once, in a fixed order, stopped at the first handover: the second `start`
    # sat in a payload the shell scan had not published yet, its own target was
    # never published in turn, and the `rm` behind it went unread.
    command = 'start "" cmd /c ' * depth + "rm -rf x"
    assert "rm" in tools._find_blocked_commands(command)


def test_nesting_does_not_decide_how_long_the_scan_takes(windows_terminal):
    # Every nested scan runs on a SUFFIX of the same text and the token walk
    # reaches those suffixes too, so each was scanned once per path that arrived
    # at it: 2**n for n wrappers, which is 19 seconds at 17 of them and does not
    # finish at 30. A command that refuses to be screened is a command that runs.
    start = time.monotonic()
    tools._find_blocked_commands('start "" ' * 40 + "echo hi")
    assert time.monotonic() - start < 5


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c "C:\\tools.v2\\notepad -x"',
        'cmd /c "C:\\tools.v2\\notepad"',
    ],
)
def test_pathext_makes_the_suffix_optional_on_a_launched_path(windows_terminal, command):
    # PATHEXT is why `C:\tools\notepad` runs notepad.exe, so a path that never
    # carries a suffix is still a program. Reading it as anything else sent the
    # payload through a full re-lex, where the dotted directory matched the
    # POSIX source builtin and a legitimate launch was refused.
    assert not tools._find_blocked_commands(command)


def test_echo_open_paren_prints_rather_than_groups(monkeypatch):
    # cmd's `echo(` form prints what follows it. Reading the glued `(` as the
    # grouping operator opened a command position the shell never opens, and
    # refused a line that only prints its argument. A cmd shape: under bash the
    # `(` is that shell's own operator and origin/main reads the line the same
    # way, so only the cmd lexer is asserted here.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert not tools._find_blocked_commands('cmd /c echo( start "" powershell')


@pytest.mark.parametrize(
    "command",
    ['cmd /c ( start "" powershell )', '(start "" powershell)'],
)
def test_real_cmd_grouping_still_opens_a_command(windows_terminal, command):
    # An operator GLUED to the word in front of START is a separate, pre-existing
    # gap (`echo hi&(start ""...` reaches the cmd lexer as one token `hi&(start`),
    # so it is not asserted here.
    assert "powershell" in tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c "C:\\tools\\notepad rm"',
        'cmd /c "C:\\tools\\notepad powershell"',
        'start "" "C:\\tools\\notepad rm"',
    ],
)
def test_a_word_without_a_separator_does_not_continue_a_path(windows_terminal, command):
    # `C:\Program Files\PowerShell\7\pwsh` and `C:\tools\notepad rm` are the same
    # list of words: a path holding a space, and a path followed by a file to
    # open. Joining both read the second as a program named rm and refused a line
    # that runs notepad. Only a word carrying a separator continues the path.
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c "C:\\Program Files\\PowerShell\\7\\pwsh"',
        'cmd /c "C:\\tools\\rm -rf x"',
        'cmd /c "C:\\tools\\rm"',
        'cmd /c "C:\\powershell scripts\\pwsh.exe -x"',
    ],
)
def test_a_spaced_or_suffixless_program_path_is_still_read(windows_terminal, command):
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c "C:\\tmp\\curl http://x"',
        'start "" "C:\\tmp\\curl http://x"',
        'cmd /c "C:\\tmp\\curl C:\\out\\x"',
    ],
)
def test_an_argument_is_not_a_path_continuation(windows_terminal, command):
    # What continues a path holding spaces is a RELATIVE fragment. A word that
    # opens a path of its own, or names a URL, is an argument however many
    # separators it carries, and reading those as continuations took the
    # basename off the ARGUMENT: `C:\tmp\curl http://x` came back as `x`.
    assert "curl" in tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        "start \"\" sed '1e rm -f victim' input",
        'cmd //c start "" sed "1e rm -f victim" input',
    ],
)
def test_a_sed_that_start_launches_is_read_as_sed(windows_terminal, command):
    # The `e` scan reads its own list, built while the main walk ran, so a sed
    # START launches was never read as one even though the launch itself was
    # published. GNU sed's `e` shells out, so that script really deletes.
    assert "rm" in tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    ["%COMSPEC% /c powershell", 'cmd /c start "" %COMSPEC% /c powershell'],
)
def test_comspec_names_cmd(windows_terminal, command):
    # cmd expands %COMSPEC% before running anything and it names cmd.exe, so
    # this is a shell invocation spelled the long way. Comparing the literal
    # token to the shell names left the payload behind its /c unread.
    assert "powershell" in tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        "cmd /c echo hi& cmd /c powershell",
        'cmd /c start "" https://x/?a=1& cmd /c powershell',
    ],
)
def test_a_glued_operator_opens_a_command_for_the_shell_lookup(windows_terminal, command):
    # The cmd lexer never splits an operator off the word it is glued to, so
    # `hi&` keeps its `&` and the main scan stays in argument position for the
    # rest of the line. The START gate already reads that off the previous
    # token; the shell lookups did not, so the second shell was refused a
    # command position and its payload went unread. Both ask one predicate now.
    assert "powershell" in tools._find_blocked_commands(command)


def test_a_padded_wrapper_chain_behind_start_fails_closed(windows_terminal):
    # The wrapper chain outran the hop budget, so the command START finally runs
    # was never reached. Blocking the chain's own first word is what `-exec`
    # already does; failing open instead just tells an author how long to make
    # the padding.
    assert tools._find_blocked_commands('start "" ' + "env " * 40 + "rm -rf x")


def test_a_wrapper_chain_within_the_budget_still_names_the_child(windows_terminal):
    assert "rm" in tools._find_blocked_commands('start "" env env rm -rf x')


@pytest.mark.parametrize(
    "command",
    [
        'cmd //c call start "" powershell -Command ls',
        'cmd /c call start "" powershell',
    ],
)
def test_call_forwards_inside_a_cmd_payload(windows_terminal, command):
    # What follows a `/c` is cmd's to parse even when the outer shell is bash,
    # so CALL forwards there on either lexer. The main walk recorded only
    # `call`, so the position gate refused the START behind it.
    assert "powershell" in tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    ['call start "" powershell', "cmd /c call powershell", "call rm -rf x"],
)
def test_call_forwards_when_cmd_is_the_shell(windows_cmd_only, command):
    # CALL re-parses the rest of the line and runs it, so its target is a
    # command position exactly as a wrapper's child is.
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    ["call curl http://x", "call rm -rf x"],
)
def test_call_is_not_a_prefix_under_bash(windows_git_bash_only, command):
    # bash has no CALL builtin, so this runs a user program named call and the
    # rest is its ARGUMENTS. Treating it as a prefix everywhere blocked a line
    # bash would not have run the blocked name from.
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    ["call build.bat", "call node app.js", "call :build", 'echo call start "" powershell'],
)
def test_call_does_not_invent_a_command(windows_terminal, command):
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c echo ok&&start "" cmd /c powershell -Command ls',
        'cmd /c echo hi&start "" powershell',
        'echo hi&start "" powershell',
        'cmd /c echo hi|start "" powershell',
    ],
)
def test_start_after_a_glued_operator_still_launches(windows_cmd_only, command):
    # cmd needs no whitespace around its operators, so this really launches, and
    # that lexer hands the whole thing back as one token `ok&&start`.
    assert "powershell" in tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c echo "a&start b"',
        'cmd /c echo hi^&start "" powershell',
        'cmd /c restart "" powershell',
        "cmd /c kickstart --now",
    ],
)
def test_a_glued_operator_has_to_be_live(windows_cmd_only, command):
    # Inside a quoted span it is text, an odd run of carets escapes it, and a
    # word merely ENDING in start is not one: `restart` and `kickstart` launch
    # nothing.
    assert not tools._find_blocked_commands(command)


def test_glue_survives_an_earlier_quoted_argument(windows_cmd_only):
    # The owner map was built by splitting the whole line on whitespace, so it
    # gave up as soon as any EARLIER argument held quoted whitespace and the
    # glue was lost on a pair much further along. Read off token offsets now.
    assert "powershell" in tools._find_blocked_commands('echo "a b" && ""cmd /c powershell""')


@pytest.mark.parametrize(
    "command",
    ['cmd /c "C:\\rm.exe dir\\notepad"', 'start "" "C:\\tools\\pwsh.exe scripts\\job.ps1"'],
)
def test_an_executable_suffix_ends_the_path(windows_terminal, command):
    # CreateProcess resolves a path holding spaces by trying its prefixes
    # SHORTEST first, so `C:\tools\pwsh.exe scripts\job.ps1` runs pwsh with a
    # script argument rather than naming one long path. Carrying the path on
    # through that argument read `job.ps1` as the program, whose suffix is not
    # executable, and let the shell behind it through.
    #
    # A directory really named `rm.exe dir` is the same list of words, so it is
    # refused here. That is the safe direction of an ambiguity the command line
    # cannot resolve, and it is what the shell would try first.
    assert tools._find_blocked_commands(command)


def test_a_real_executable_still_ends_the_path(windows_terminal):
    assert "rm" in tools._find_blocked_commands('cmd /c "C:\\tools\\rm.exe"')


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c "C:\\Program Files (x86)\\PowerShell\\7\\pwsh.exe"',
        'start "" "C:\\Program Files (x86)\\PowerShell\\7\\pwsh.exe"',
    ],
)
def test_a_space_only_fragment_still_continues_a_path(windows_terminal, command):
    # `Files` carries no separator of its own, and stopping there left `Program`
    # as the program while the shell at the end of the path went unreported.
    # A word like that continues the path when a LATER one still carries it.
    assert "pwsh" in tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c "C:\\tools\\pwsh.exe scripts\\job.ps1"',
        'start "" "C:\\tools\\pwsh.exe scripts\\job.ps1"',
    ],
)
def test_an_executable_is_not_extended_into_its_arguments(windows_terminal, command):
    # Carrying the path on through a relative ARGUMENT read `job.ps1` as the
    # program; its suffix is not executable, so the launch was reported as a
    # document and the shell in front of it went unscreened.
    assert "pwsh" in tools._find_blocked_commands(command)


def test_a_document_name_may_still_hold_a_space(windows_terminal):
    # The other side of that: nothing along `C:\tmp\rm report.docx` is
    # executable, so the whole target is one file and its own suffix decides.
    assert not tools._find_blocked_commands('cmd //c start "" "C:\\tmp\\rm report.docx"')


def test_the_newline_marker_is_not_attacker_controlled(windows_terminal):
    # The stand-in for a newline during the second lex used to be one fixed
    # control character, so including it disabled newline recovery for the whole
    # line: a boundary an author could delete by typing it.
    assert "powershell" in tools._find_blocked_commands('echo \x01\nstart "" powershell')
    assert "powershell" in tools._find_blocked_commands('echo \x01\x02\x03\nstart "" powershell')


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c "C:\\tools\\rm scripts\\x"',
        'cmd /c "C:\\tools\\pwsh scripts\\job.ps1"',
        # START is not in this list on purpose. `cmd /c` runs a COMMAND LINE, so
        # its prefixes are candidate programs; START takes a single target, which
        # is why its arguments go outside the quotes, so a fully quoted target
        # naming no executable reads as one file. With a suffix present there is
        # no ambiguity and `start "" "C:\tools\pwsh.exe scripts\job.ps1"` is
        # screened, which the test above pins.
    ],
)
def test_the_shortest_prefix_is_screened_when_nothing_ends_the_path(windows_terminal, command):
    # No executable suffix anywhere says where this path ENDS, and CreateProcess
    # tries the shortest prefix first, so `C:\tools\rm scripts\x` runs rm on a
    # file rather than naming one long path. Joining it all read the argument as
    # the program and let the blocked one in front of it through.
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c "C:\\powershell scripts\\notepad.exe -x"',
        'cmd /c "C:\\tools\\notepad rm"',
    ],
)
def test_a_directory_named_like_a_command_is_not_the_program(windows_terminal, command):
    # The other side: when the path DOES end in an executable, that is what runs,
    # so the folder it sits in is not screened as the program.
    assert not tools._find_blocked_commands(command)


def test_the_newline_marker_cannot_be_exhausted(windows_terminal):
    # Any fixed set of markers is one an author can include in full, and doing
    # that used to disable newline recovery for the whole line. Searched now, so
    # a command holding every previous candidate is still read.
    every_old_marker = "".join(chr(code) for code in (1, 2, 3, 4, 5, 6, 14, 15))
    command = f'echo {every_old_marker}\nstart "" powershell'
    assert "powershell" in tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        "cmd /c echo hi&call powershell",
        "cmd /c echo hi & call powershell",
        'call "powershell" -Command ls',
        'cmd /c call "powershell" -Command ls',
    ],
)
def test_call_is_read_through_glue_and_quoting(windows_cmd_only, command):
    # cmd needs no whitespace around its operators, and it strips its own quote
    # marks before resolving the program, so all four of these run PowerShell.
    assert "powershell" in tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    ['cmd /c echo "a&call b"', "cmd /c echo hi^&call powershell", "echo call powershell"],
)
def test_call_glue_has_to_be_live(windows_cmd_only, command):
    assert not tools._find_blocked_commands(command)


def test_newline_scanning_does_not_reread_the_line(windows_cmd_only):
    # Counting the caret run by copying the prefix and stripping it re-read the
    # whole line per newline, which is quadratic in the command.
    script = "\n".join("echo line %d ^" % index for index in range(20000))
    started = time.monotonic()
    tools._find_blocked_commands(script)
    assert time.monotonic() - started < 5


@pytest.mark.parametrize(
    "command",
    ['cmd //c "call powershell"', 'cmd /c "call rm -rf x"'],
)
def test_a_quoted_cmd_payload_is_parsed_as_cmd(windows_terminal, command):
    # What follows a `/c` reaches cmd as a command string even where bash is the
    # outer shell, so CALL forwards inside it. Re-lexing the payload with the
    # outer lexer left it unread there while the unquoted spelling was caught.
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize("command", ["find . -exec call powershell ;", "fd . -x call powershell"])
def test_call_is_not_a_prefix_for_an_exec_child(windows_terminal, command):
    # find and fd run their -exec child themselves; cmd never re-parses it, so
    # CALL is not a prefix there and the word behind it is that child's argument.
    assert not tools._find_blocked_commands(command)


def test_an_exec_child_still_steps_over_a_real_wrapper(windows_terminal):
    assert "powershell" in tools._find_blocked_commands("find . -exec env powershell ;")


def test_a_start_title_is_not_the_command(windows_terminal):
    # START documents its first quoted argument as the WINDOW TITLE with the
    # command after it, so this runs notepad and screening the title refused the
    # line for the title's sake.
    assert not tools._find_blocked_commands(
        'cmd //c start "C:\\Program Files\\PowerShell\\7\\pwsh.exe" notepad'
    )


def test_a_quoted_command_line_with_nothing_behind_it_is_still_screened(windows_git_bash_only):
    # The reason the title is screened at all: posix proves a title only by its
    # whitespace, so a quoted COMMAND LINE looks the same when nothing follows.
    # Only under bash. cmd really does take that first quoted word as the title
    # and run nothing, which is why `start "" "path"` is the documented idiom.
    assert "pwsh" in tools._find_blocked_commands(
        'cmd //c start "C:\\Program Files\\PowerShell\\7\\pwsh.exe"'
    )


@pytest.mark.parametrize(
    "command",
    [
        'start "powershell -Command ls"',
        'start "powershell -Command ls" /min',
        'start /d C:\\dir "powershell -Command ls" ; rm -rf x',
        'start "powershell -Command ls"\nrm -rf x',
    ],
)
def test_a_quoted_command_line_is_screened_when_no_command_follows(windows_git_bash_only, command):
    # What follows the quoted word decides whether it WAS a title. An option, a
    # separator, a newline, or nothing at all means no command came after it, so
    # the quoted word is the command line START runs.
    assert "powershell" in tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd //c start "C:\\Program Files\\PowerShell\\7\\pwsh.exe" -notepad.exe',
        'start "powershell -Command ls" -x y',
    ],
)
def test_a_hyphen_target_is_a_program_not_a_start_option(windows_git_bash_only, command):
    # Every option START documents is slash-prefixed, so a `-` opens none of
    # them. A hyphenated word behind the quoted one is the program START runs,
    # which makes the quoted word its window title and nothing to screen.
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        "notepad start rm -rf x",
        "dir report.docx start powershell -Command ls",
    ],
)
def test_start_only_launches_where_a_command_may_begin(windows_terminal, command):
    # START is a launcher at a command position and an ordinary argument
    # anywhere else. A program already named ahead of it receives these words,
    # so nothing behind the START runs. Randomised differential fuzzing against
    # the previous scan reduced to this one family, so it is pinned here.
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "prefix",
    ["env", "nohup", "time", "xargs", "FOO=1", "true &&", "true ;", "echo hi |"],
)
def test_start_still_launches_behind_a_real_prefix(windows_git_bash_only, prefix):
    # The other half, and the reason the rule above is safe: everywhere a
    # command really does begin, START is read as one.
    assert "rm" in tools._find_blocked_commands(f"{prefix} start rm -rf x")


@pytest.mark.parametrize("command", ["call bash -c pwsh", "call cmd /c powershell"])
def test_a_call_wrapper_only_forwards_where_cmd_parses(command):
    # CALL is a cmd builtin, so the shell behind it opens a payload only on the
    # cmd lane. Under bash `call` is an ordinary program name and everything
    # after it is its arguments, which run nothing.
    def screen(posix):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(sys, "platform", "win32")
            patch.setattr(tools, "_shell_is_posix", lambda: posix)
            patch.setattr(
                tools,
                "_BLOCKED_COMMANDS",
                tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
            )
            return tools._find_blocked_commands(command)

    assert not screen(True)
    assert screen(False)


@pytest.mark.parametrize(
    "command",
    ['cmd //c "xargs call powershell"', 'cmd //c "env call powershell"'],
)
def test_call_stops_at_an_external_wrapper(windows_terminal, command):
    # xargs and env run their child themselves rather than handing the rest of
    # the line back to cmd, so cmd's own builtins no longer apply behind one.
    # `xargs [OPTION]... COMMAND [INITIAL-ARGS]...` looks for a program named
    # call, finds none, and never reaches PowerShell.
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    ['cmd //c "call xargs powershell"', 'cmd //c "xargs powershell"'],
)
def test_a_wrapper_still_runs_its_own_child(windows_terminal, command):
    # The other half. CALL re-parses the line under cmd, so a wrapper behind one
    # is still cmd's, and a wrapper with a real child still launches it.
    assert "powershell" in tools._find_blocked_commands(command)


def test_screening_stays_linear_in_escaped_operators(monkeypatch):
    # Counting the carets in front of each operator by re-measuring the text
    # from its start copied a longer prefix every time, so a payload made of
    # escaped operators cost time in the square of its length.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )
    assert tools._has_bare_cmd_operator("echo hi&start")
    assert not tools._has_bare_cmd_operator("echo hi^&start")
    assert tools._has_bare_cmd_operator("echo hi^^&start")
    assert not tools._has_bare_cmd_operator("echo hi^^^&start")

    def cost(count):
        text = 'cmd /c "echo ' + "^&" * count + '"'
        start = time.perf_counter()
        tools._find_blocked_commands(text)
        return time.perf_counter() - start

    cost(2000)  # warm the caches the walk builds
    small, large = cost(4000), cost(16000)
    # Quadratic would be about 16x for four times the input.
    assert large < small * 10, f"{small = } {large = }"


def test_screening_stays_linear_in_path_fragments(windows_terminal):
    # A quoted path made of many separator-less fragments asked the same
    # lookahead once per fragment, and every call rescanned what was left.
    assert "pwsh" in tools._find_blocked_commands('cmd /c "C:\\Program a a dir\\pwsh.exe"')

    def cost(count):
        text = 'cmd /c "C:\\Program ' + "a " * count + 'dir\\pwsh.exe"'
        start = time.perf_counter()
        tools._find_blocked_commands(text)
        return time.perf_counter() - start

    cost(500)
    small, large = cost(1000), cost(4000)
    assert large < small * 10, f"{small = } {large = }"


@pytest.mark.parametrize(
    "command",
    [
        "cmd /c echo ok&cmd /c powershell",
        "cmd /c echo ok&&cmd /c powershell",
        "cmd /c echo ok|cmd /c powershell",
    ],
)
def test_a_shell_behind_a_glued_operator_opens_its_payload(windows_cmd_only, command):
    # cmd needs no whitespace around its separators, so the second `cmd /c`
    # really runs. Reading the whole token left that shell unrecognised and its
    # payload unscreened.
    assert "powershell" in tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        "cmd /c echo hi& call powershell",
        "cmd /c echo hi&call powershell",
        "cmd /c echo hi & call powershell",
    ],
)
def test_call_is_read_however_the_operator_is_spelled(windows_cmd_only, command):
    # The same boundary with the operator glued to the word in front, glued to
    # CALL, or standing alone. All three run PowerShell.
    assert "powershell" in tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c "C:\\Program -Files\\PowerShell\\7\\pwsh.exe" -Command ls',
        'cmd /c "C:\\Program - Files\\PowerShell\\7\\pwsh.exe" -Command ls',
    ],
)
def test_a_hyphen_may_open_a_directory_name(windows_terminal, command):
    # Windows allows a directory name to open with a hyphen, so a fragment
    # carrying one is still part of the quoted path. Breaking there stopped the
    # reconstruction at `C:\Program` and let the shell through unnamed.
    assert "pwsh" in tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    ['start "my title" /? powershell', 'cmd /c start "job" /? powershell'],
)
def test_start_help_launches_nothing(windows_terminal, command):
    # `/?` displays START's help and returns, so the word behind it is never
    # launched and screening it refused a line that only prints usage.
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    ["( call rm -rf x", ") call rm -rf x", "cmd /c echo hi& call powershell"],
)
def test_a_call_boundary_only_forwards_where_cmd_parses(windows_git_bash_only, command):
    # The operator boundary that publishes CALL's child is cmd's, so it must not
    # apply to a line bash is reading: `( call rm` runs a subshell looking for a
    # program named call and never reaches rm. Found by randomised differential
    # fuzzing against the previous scan, which reported neither.
    assert not tools._find_blocked_commands(command)


def test_a_call_boundary_still_forwards_inside_a_quoted_payload():
    # And it does apply inside a `/c` payload even where bash is the outer
    # shell, because that payload is cmd's command line.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "platform", "win32")
        patch.setattr(tools, "_shell_is_posix", lambda: True)
        patch.setattr(
            tools,
            "_BLOCKED_COMMANDS",
            tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
        )
        assert "powershell" in tools._find_blocked_commands('cmd //c "echo hi& call powershell"')


@pytest.mark.parametrize(
    "command",
    [
        "cmd /c echo&call call powershell",
        "cmd /c echo & call call powershell",
        "cmd /c echo&call powershell",
    ],
)
def test_a_prefix_handed_over_by_an_operator_still_forwards(windows_cmd_only, command):
    # A wrapper spelled behind a glued operator is a command word like any
    # other, so it forwards to the word behind it, and CALL may forward to
    # another CALL. Screening the tail but dropping the prefix state left
    # PowerShell unread even though cmd reparses and runs it.
    assert "powershell" in tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command", ['cmd /c echo(start "" powershell', 'cmd /c rem(start "" powershell']
)
def test_a_glued_parenthesis_after_a_word_opens_no_group(windows_cmd_only, command):
    # `echo(` and `rem(` are the no-space forms of those commands and print what
    # follows. Only a `(` of its own, or one behind another operator, opens a
    # group, which is the distinction `_ends_with_cmd_operator` already drew.
    assert not tools._find_blocked_commands(command)


def test_a_real_group_still_opens_a_command(windows_cmd_only):
    # The other half: a `(` that is not glued behind a word does open one.
    assert "powershell" in tools._find_blocked_commands('cmd /c (start "" powershell')


@pytest.mark.parametrize(
    "command",
    [
        "cmd /c if 1==1 cmd /c powershell",
        "cmd /c if not 1==1 cmd /c powershell",
        "cmd /c if %a% equ 1 cmd /c powershell",
        "cmd /c if not %a% neq 1 cmd /c powershell",
    ],
)
def test_a_cmd_comparison_still_leaves_a_command_behind_it(windows_terminal, command):
    # cmd's IF takes a comparison glued (`1==1`) or spelled with an operator
    # (`%a% equ 1`), and runs the body when it holds. Reading the comparison as
    # the command word left that body in argument position.
    assert "powershell" in tools._find_blocked_commands(command)


def test_a_comparison_in_argument_position_is_not_a_condition(windows_terminal):
    # And only where a command may begin: `echo a==b rm` prints its arguments.
    assert not tools._find_blocked_commands("echo a==b rm")


@pytest.mark.parametrize(
    "command",
    [
        "1==1",
        "cmd /c if 1==1",
        'cmd //c "if 1==1 echo ok"',
        "if",
        "cmd /c if not",
        "cmd /c if %a% equ",
    ],
)
def test_a_comparison_at_the_end_of_the_line_does_not_raise(windows_terminal, command):
    # The screen answers a yes or no; an exception is neither, and whatever the
    # caller does with one it is not a refusal. A comparison with nothing behind
    # it read past the end of the token list.
    assert isinstance(tools._find_blocked_commands(command), set)


def test_a_quoted_cmd_payload_with_a_comparison_is_still_screened(windows_terminal):
    # And the payload behind it is still read.
    assert "rm" in tools._find_blocked_commands('cmd //c "if 1==1 rm -rf x"')


@pytest.mark.parametrize(
    "command",
    [
        "C:/a==b/powershell.exe -Command ls",
        "./a==b/rm -rf x",
        "cmd /c C:/a==b/powershell.exe -Command ls",
    ],
)
def test_an_equals_in_a_path_is_not_a_comparison(windows_terminal, command):
    # `==` reads as cmd's comparison only inside a condition an IF opened. A
    # path may hold one, and skipping that word passed over the program it
    # names while the regex backstop could not match it either.
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'start "" https://example.com/cmd /c powershell',
        'start "" report.docx /c powershell',
    ],
)
def test_a_browser_target_is_not_a_shell_word(windows_terminal, command):
    # START hands a URL to the browser and a document to whatever is registered
    # for it, so neither is a program. Publishing them as command positions let
    # the nested-shell pass read `https://example.com/cmd` as a shell.
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        "if C:/a==b/powershell.exe",
        "if ./a==b/rm -rf x",
        "| if ./a==b/rm -rf x",
        "if 1==1 rm -rf x",
    ],
)
def test_bash_if_opens_no_cmd_comparison(windows_git_bash_only, command):
    # Only cmd's IF takes a comparison. Under bash `if` is its own keyword and
    # the word behind it is a command, so `if C:/a==b/powershell.exe` runs one
    # and `if 1==1` looks for a program by that name. Found by randomised
    # differential fuzzing once the corpus learned to spell a comparison.
    assert tools._find_blocked_commands(command) or "1==1" in command


@pytest.mark.parametrize(
    "command",
    ["cmd /c if C:/a==b/powershell.exe -Command ls", "cmd /c if ./a==b/rm -rf x"],
)
def test_a_cmd_comparison_is_never_a_path(windows_terminal, command):
    # And even inside a real condition, cmd compares two operands and neither
    # carries a separator, so a word holding one is the program it looks like.
    assert tools._find_blocked_commands(command)
