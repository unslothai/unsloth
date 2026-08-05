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
        'time start "" powershell -Command ls',
        'env start "" powershell -Command ls',
        'nohup start "" powershell -Command ls',
        'echo x | xargs start "" powershell -Command ls',
        'if x; then start "" rm -rf x',
    ],
)
def test_a_launched_start_is_read_wherever_the_shell_runs_it(windows_terminal, command):
    # Command position is what the main scan already decides, wrappers and
    # assignment prefixes and all, so the walk reuses that verdict. Reading only
    # the token before START instead missed every one of these, each of which
    # really launches. A newline separates commands but lexes as whitespace,
    # leaving no token to look back at, so it is recovered from the raw text.
    assert tools._find_blocked_commands(command)


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
        'echo hi\ntime start "" powershell -Command ls',
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
