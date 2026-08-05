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
import subprocess
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


@pytest.mark.parametrize(
    "command",
    [
        "cmd /c powershell -Command ls",
        "cmd //c powershell -Command ls",
        "cmd //k powershell -Command ls",
        'cmd //c start "" powershell -Command ls',
        'cmd //c start /b "" pwsh -Command ls',
        'cmd //c start //min "" powershell -Command ls',
        r'cmd //c start /d C:/tmp "" powershell -Command ls',
    ],
)
@pytest.mark.parametrize("bash", [r"C:\Program Files\Git\bin\bash.exe", None])
def test_cmd_shellout_is_screened_through_mangled_switches(monkeypatch, bash, command):
    # Git Bash turns a lone /c into a path, so a model writes //c. That spelling
    # skipped the nested scan, making `cmd //c powershell` reachable where
    # `cmd /c powershell` was blocked, and `start` launches its argument too.
    #
    # Pinned to win32 like the rest of this file: powershell and pwsh are hard
    # blocks only there (off Windows they are a prompt, see test_permission_mode),
    # so on a Linux runner this asserted nothing about the nested scan at all.
    #
    # Both shells, because the presence of Git Bash decides which lexer runs and
    # the two disagree about quotes: without it the title in `start ""` arrives
    # as a literal `""` and every start form here went unscreened.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize("bash", [r"C:\Program Files\Git\bin\bash.exe", None])
@pytest.mark.parametrize(
    "command",
    [
        # A title cmd reads as one on either lexer: it holds whitespace, so bash
        # has to re-quote it, and it arrives quoted whoever built the line.
        'start "My Title" powershell -Command ls',
        'cmd /c start "a b" pwsh -c ls',
        'cmd //c start /min "the win" powershell -Command ls',
    ],
)
def test_start_screens_the_program_behind_a_window_title(monkeypatch, bash, command):
    # `start "title" prog` is the documented form. The title was read as the
    # program, so anything after it launched unscreened -- on both lexers, which
    # made this the wider of the two holes.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c start "t" pwsh -c ls',
        'cmd //c start /min "win" powershell -Command ls',
        'start "t" /wait /b pwsh -c ls',
    ],
)
def test_a_bare_quoted_title_is_only_a_title_under_the_cmd_lexer(monkeypatch, command):
    # Nothing rewrites the line without Git Bash, so cmd is handed the quotes
    # the user wrote and reads them as the window title.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert tools._find_blocked_commands(command)

    # With Git Bash it is not a title at all: bash removes the quotes and MSYS
    # re-quotes an argument only when it holds whitespace or is empty, so cmd
    # gets `start t pwsh -c ls`, which launches t and hands it pwsh as a
    # parameter. Screening the word behind it instead was a guess at how MSYS
    # rebuilds the line, and guessing cost two over-blocks before this;
    # test_git_bash_does_not_requote_a_bare_word checks the rule on Windows.
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize("bash", [r"C:\Program Files\Git\bin\bash.exe", None])
@pytest.mark.parametrize(
    "command",
    [
        "start notepad rm.txt",
        "start excel curl.csv",
        'start "" notepad rm.txt',
        # A title cmd still sees quoted on either lexer. main screens this one
        # as a program and reports `rm`; the unspaced `"rm.txt"` spelling stays
        # blocked under Git Bash exactly as on main, because bash drops the
        # quotes and cmd is handed a program named rm.txt.
        'start "rm x.txt" notepad',
    ],
)
def test_start_arguments_are_not_screened_as_commands(monkeypatch, bash, command):
    # An unquoted first token is the program and the rest are its arguments, so
    # only a quoted title moves the scan along. Screening an argument is not
    # harmless: the recursive scan re-anchors its command-boundary regex at the
    # start of the token, so `rm.txt` reported `rm` where the same word in the
    # same place on a full line does not match. A quoted title is data too, and
    # is not screened either.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd //c start "" bash -c "bash /c/x.sh"',
        "cmd //c start wt",
        "cmd //c dir",
        "start notepad",
    ],
)
@pytest.mark.parametrize("bash", [r"C:\Program Files\Git\bin\bash.exe", None])
def test_detached_windows_stay_launchable(monkeypatch, bash, command):
    # `start` is the only route to a window on the user's desktop, which the
    # terminal description promises, so screening must not blanket-block cmd.
    #
    # Pinned like the rest: unpinned this ran the POSIX lexer with the non-Windows
    # block set, so it never saw the posture it exists to protect.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize("bash", [r"C:\Program Files\Git\bin\bash.exe", None])
@pytest.mark.parametrize(
    "command",
    [
        # A separator before the start is the whole point: inferring the title
        # from a second lex of the whole line meant a `;` or a `#` glued onto
        # its neighbour stopped the two lining up, and the title was screened
        # in the program's place.
        'cd /c/proj; cmd //c start "" pwsh -Command ls',
        'echo done; cmd //c start "" powershell -Command ls',
        'cmd //c start "" pwsh -Command ls; echo ok',
        '(cmd //c start "" pwsh -Command ls)',
        'ls; cmd //c start "The Build" powershell -Command "Remove-Item x"',
        'start "" powershell -c ls # note',
        # Microsoft documents the switches after the title, not before it.
        'start "" /min powershell -c ls',
        'start "a b" /wait /b pwsh -c ls',
        'cmd /c start "" /min powershell -c ls',
    ],
)
def test_start_screening_survives_separators_and_switch_order(monkeypatch, bash, command):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        # Bash strips the user's quotes and re-quotes an argument only when it
        # has to, so cmd is handed `start powershell -c ls`: powershell is the
        # program, not a title. Reading the source quotes skipped it entirely.
        'cmd //c start "powershell" -c ls',
        "cmd //c start 'pwsh' -c ls",
        # A POSIX-only escape anywhere earlier in the line used to desync the
        # second lex the title came from, which then reported nothing and left
        # the program behind the title unscreened.
        r'echo a\ b; cmd //c start "" powershell -c ls',
        r'echo a\ b\ c && cmd //c start "" pwsh -Command ls',
        # Git Bash hands native programs their arguments in MSYS form, and every
        # real start switch is one word, so a slash-rooted program is not one.
        'cmd //c start "" /c/Windows/System32/powershell.exe -c ls',
        "cmd //c start /c/Windows/System32/powershell.exe -c ls",
    ],
)
def test_start_screening_reads_what_cmd_receives_not_the_source_quotes(monkeypatch, command):
    # These all run through Git Bash, the shell that rewrites the line between
    # the user and cmd. Screening the tokens the user typed rather than the ones
    # cmd parses left the blocked program launchable in each.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        "start 'title' rm.txt",
        "start 'notes' curl.csv",
    ],
)
def test_cmd_has_no_single_quote_syntax(monkeypatch, command):
    # Without Git Bash nothing rewrites the line and cmd documents the title as
    # "<Title>", so 'title' is a program literally named that and rm.txt is its
    # argument. Counting a single quote as a title moved the scan onto the
    # argument and hard-blocked a benign file for holding `rm`.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize("bash", [r"C:\Program Files\Git\bin\bash.exe", None])
@pytest.mark.parametrize(
    "command",
    [
        # A start that is not a launch. Reading every token named start walked
        # an ordinary argument into the title branch and refused a benign echo.
        'echo start "my title" powershell',
        "echo to start pwsh run this",
        'printf "%s" start powershell',
    ],
)
def test_start_is_only_screened_in_command_position(monkeypatch, bash, command):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        # cmd strips the quotes before it resolves the program, but the non-POSIX
        # lexer keeps them, so the scan looked for a program spelled with them
        # and the hard block was one quote away. Quoting is the ordinary way to
        # write a path holding a space.
        ('cmd /c start "" "powershell.exe" -c ls', True),
        (r'cmd /c start "" "C:\Windows\System32\powershell.exe"', True),
        # ...and the target is a program, not a command line. Rescanning it as
        # one read the path as a command position, where the blocklist's own `.`
        # matches the dot in any file that has one.
        (r'cmd /c start "" "C:\Users\me\report.txt"', False),
        (r'cmd /c start "" "C:\Program Files\app\notepad.exe"', False),
    ],
)
def test_start_target_is_matched_as_a_program_path(monkeypatch, command, blocked):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert bool(tools._find_blocked_commands(command)) is blocked


@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        # A POSIX-only escape desyncs the quote check, which then answers
        # "cannot tell". Screening both candidates on that answer brought the
        # unquoted-argument over-block back for any line holding one.
        (r"echo a\ b; start notepad rm.txt", False),
        (r"echo a\ b; start excel curl.csv", False),
        # The argument is a file that happens to be named like a blocked
        # command, so it is the one shape a name match alone still refuses.
        (r"echo a\ b; start notepad rm", False),
        (r"echo a\ b; start notepad curl", False),
        # An earlier quoted copy of the word used to be read as evidence that
        # this one was quoted too, because the search was over the whole line.
        (r'echo "notepad" a\ b; start notepad rm', False),
        # The program itself is still screened whatever else is on the line.
        (r'echo a\ b; cmd //c start "" powershell -c ls', True),
    ],
)
def test_desync_does_not_screen_unquoted_start_arguments(monkeypatch, command, blocked):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert bool(tools._find_blocked_commands(command)) is blocked


@pytest.mark.parametrize("bash", [r"C:\Program Files\Git\bin\bash.exe", None])
@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        # cmd keeps parsing past its own conditionals and operators, so taking
        # only the first word of its command line missed everything behind one.
        ('cmd //c if exist . start "" powershell -c ls', True),
        ('cmd //c dir & start "" powershell -c ls', True),
        ('cmd //c if not exist x.txt start "" pwsh -c ls', True),
        # ...but only the word cmd runs counts. Anywhere else the start is an
        # operand of the command that is running, whether that prints it or
        # reads it as a name, and neither launches anything.
        ('cmd //c echo start "my title" powershell', False),
        ('cmd //c set x=start "t" powershell', False),
        ('cmd /c dir start "" powershell', False),
        ('cmd /c type start "" pwsh', False),
        # `>` redirects into a file rather than beginning another command, so
        # these write output to one called start and launch nothing, including
        # where the redirection sits on the segment's own command word.
        ("cmd /c echo hi>start powershell", False),
        ('cmd /c x>start "" powershell', False),
    ],
)
def test_start_is_found_across_a_whole_cmd_command_line(monkeypatch, bash, command, blocked):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    assert bool(tools._find_blocked_commands(command)) is blocked


@pytest.mark.parametrize(
    "command",
    [
        # The cmd lexer takes no punctuation_chars, so an operator stays glued to
        # the name, where cmd itself ends the command word and runs the program.
        'start "" powershell& echo ok',
        'start "" powershell&echo ok',
        'start "" pwsh|more',
        'start "" powershell>out.txt',
    ],
)
def test_start_target_ends_at_a_cmd_operator(monkeypatch, command):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'start "dry run" rm',
        'start "backup" curl http://x',
        "start notepad rm",
    ],
)
@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_start_is_not_read_as_cmd_syntax_off_windows(monkeypatch, platform, command):
    # `start` is a cmd launcher. Off Windows it is whatever the user has by that
    # name, so its arguments are arguments and reading them as a title and a
    # program refused a benign line that main allows.
    monkeypatch.setattr(sys, "platform", platform)
    assert not tools._find_blocked_commands(command)


@pytest.mark.skipif(sys.platform != "win32", reason = "Git Bash argument passing")
def test_git_bash_does_not_requote_a_bare_word():
    # The screen reads `start "t" prog` as launching t, not prog, because MSYS
    # re-quotes an argument only when it holds whitespace or is empty. That rule
    # decides which token cmd runs, so it is checked against the real thing
    # rather than argued from the docs: ask cmd to echo the line it was handed.
    bash = tools._windows_bash()
    if bash is None:
        pytest.skip("no trusted Git Bash on this host")
    echoed = subprocess.run(
        [bash, "-c", 'cmd //c echo start "t" powershell'],
        capture_output = True,
        text = True,
        timeout = 60,
    ).stdout.strip()
    assert echoed == "start t powershell", echoed
    # ...and that it does re-quote once the word holds a space, which is what
    # makes the whitespace test the right one.
    echoed = subprocess.run(
        [bash, "-c", 'cmd //c echo start "a b" powershell'],
        capture_output = True,
        text = True,
        timeout = 60,
    ).stdout.strip()
    assert echoed == 'start "a b" powershell', echoed


@pytest.mark.parametrize("bash", [r"C:\Program Files\Git\bin\bash.exe", None])
@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        # cmd's `if` carries out the command after its condition, so the leading
        # word is not the command and the one behind the condition may be a
        # text command whose arguments are only printed.
        ('cmd /c if exist x echo start "my title" powershell', False),
        ('cmd /c if not defined V echo start "t" powershell', False),
        ('cmd /c if errorlevel 1 echo start "t" powershell', False),
        # Microsoft documents /i before the comparison. Missing it left `a==a`
        # looking like the command, which both refused a printed start and hid
        # a real one behind the same condition.
        ('cmd /c if /i a==a echo start "" powershell', False),
        # ...but a real launch behind a condition still counts.
        ('cmd /c if exist x start "" powershell -c ls', True),
        ('cmd /c if not exist x start "" pwsh -c ls', True),
        ('cmd /c if /i a==a start "" powershell', True),
        ('cmd //c if //i a==a start "" pwsh -c ls', True),
    ],
)
def test_cmd_if_advances_to_the_command_it_runs(monkeypatch, bash, command, blocked):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    assert bool(tools._find_blocked_commands(command)) is blocked


@pytest.mark.parametrize(
    "command",
    [
        # The cmd lexer glues a spaceless separator to the word before it, so
        # the start was not found as a word at all.
        'cmd /c dir&start "" powershell -c ls',
        'cmd /c echo ok&start "" pwsh -c ls',
        'cmd /c dir|start "" powershell -c ls',
    ],
)
def test_start_is_found_behind_a_glued_cmd_separator(monkeypatch, command):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        # A bash separator ends the cmd invocation rather than starting another
        # cmd command, so what follows belongs to bash and not to cmd.
        ('cmd //c echo ok; printf "%s" start "" powershell', False),
        ('cmd //c dir; ls start "" pwsh -c ls', False),
        ('cmd //c dir; echo start "my title" pwsh', False),
        # The cmd payload before the separator is still read.
        ('cmd //c start "" powershell -c ls; echo done', True),
        # An escaped separator is the opposite case: bash passes it through and
        # cmd reads it as its own, so it opens a segment rather than ending the
        # scan, and the launch behind it is real.
        (r'cmd //c echo ok \& start "" powershell -c ls', True),
        (r'cmd //c dir \& start "" pwsh -c ls', True),
    ],
)
def test_the_nested_cmd_scan_stops_at_an_outer_separator(monkeypatch, command, blocked):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert bool(tools._find_blocked_commands(command)) is blocked


@pytest.mark.parametrize(
    "command",
    [
        # Quoting is what makes a path holding an operator usable, so an
        # operator inside the quotes is part of the name, not cmd syntax.
        r'start "" "C:\A&B\powershell.exe"',
        r'start "" "C:\Program Files (x86)\A&B\pwsh.exe"',
    ],
)
def test_an_operator_inside_a_quoted_path_is_not_cmd_syntax(monkeypatch, command):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command", ["pwsh -Command ls", "powershell -c ls", "rmdir x", "runas /u:a b"]
)
def test_windows_only_names_are_not_hard_blocked_off_windows(monkeypatch, command):
    # The Windows set is a hard refusal; off Windows these stay a prompt instead
    # (tests/test_permission_mode.py). Nothing asserted that, so dropping the
    # platform gate entirely -- blocking them on Linux and macOS too -- passed
    # the whole suite.
    monkeypatch.setattr(sys, "platform", "linux")
    assert not tools._find_blocked_commands(command)
