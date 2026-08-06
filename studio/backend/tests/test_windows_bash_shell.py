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
        'cmd //c start "" pwsh -Command ls; echo ok',
        '(cmd //c start "" pwsh -Command ls)',
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
        'cd /c/proj; cmd //c start "" pwsh -Command ls',
        'echo done; cmd //c start "" powershell -Command ls',
        'ls; cmd //c start "The Build" powershell -Command "Remove-Item x"',
    ],
)
def test_a_semicolon_starts_a_command_for_bash_but_not_for_cmd(monkeypatch, command):
    # Microsoft lists cmd's command separators as `&`, `&&`, `||` and `|`, and
    # puts `;` with the characters that merely have to be quoted -- it is an
    # ARGUMENT delimiter. So the same line reads two ways: bash runs the cmd
    # behind the `;`, while cmd looks up a program named by the first word and
    # hands it everything else, launching nothing.
    # test_cmd_reads_a_semicolon_as_an_argument_delimiter checks that on Windows.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert tools._find_blocked_commands(command)
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert not tools._find_blocked_commands(command)


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


@pytest.mark.skipif(sys.platform != "win32", reason = "cmd parsing")
def test_cmd_reads_a_semicolon_as_an_argument_delimiter():
    # Whether `;` begins a command decides whether the words behind one are a
    # command line or a first program's arguments, so it is checked against cmd
    # itself. Microsoft lists only `&`, `&&`, `||` and `|` as command
    # separators: if `;` were one, the second word would run and print on its
    # own line. It is echo's argument instead, delimiter included.
    echoed = subprocess.run(
        ["cmd", "/c", "echo one; echo two"],
        capture_output = True,
        text = True,
        timeout = 60,
    ).stdout.strip()
    assert echoed == "one; echo two", echoed


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
    ("command", "blocked"),
    [
        # Microsoft: "&, | and parentheses are special characters that must be
        # preceded by the escape character ^ ... when you pass them as
        # arguments", so an escaped one is text and opens no command.
        ('cmd /c echo ok^&start "" powershell', False),
        ('cmd /c echo ok^|start "" pwsh', False),
        # ...and the caret is dropped, so it hides a name from a plain
        # comparison the way quoting once did.
        ('start "" power^shell -c ls', True),
        ('start "" pw^sh -c ls', True),
        ('cmd /c start "" po^wers^hell', True),
    ],
)
def test_cmd_carets_escape_and_are_dropped(monkeypatch, command, blocked):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert bool(tools._find_blocked_commands(command)) is blocked


@pytest.mark.parametrize(
    "command",
    [
        # `start <"title"> [switches] [<command>|<program> [<parameter>...]]`:
        # the program is optional, so a lone quoted operand is only the title
        # and cmd opens a window with that name rather than running it.
        'start "powershell"',
        'cmd /c start "rm"',
        'start /min "pwsh"',
    ],
)
def test_a_lone_quoted_operand_is_only_a_title(monkeypatch, command):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize("bash", [r"C:\Program Files\Git\bin\bash.exe", None])
@pytest.mark.parametrize(
    "command",
    [
        # The start reference: file types with a registered association,
        # "including URLs, which are automatically detected and opened in the
        # default browser". Its own example is `start "Bing" "https://..."`.
        'start "" https://example.com/curl',
        'start "Bing" "https://curl.com"',
        'start "" http://rm.example.com/rm',
        'cmd //c start "" https://example.com/powershell',
    ],
)
def test_a_url_target_goes_to_the_browser_not_a_program(monkeypatch, bash, command):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        # cmd reads an escaped `&` as its own separator, so a launch behind one
        # is real, but `;` is not cmd syntax at all and is only text to it.
        (r'cmd //c echo ok \& start "" powershell -c ls', True),
        (r'cmd //c echo ok \; start "" powershell -c ls', False),
        (r'cmd //c echo ok \| start "" pwsh -c ls', True),
    ],
)
def test_only_cmds_own_separators_open_a_cmd_segment(monkeypatch, command, blocked):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert bool(tools._find_blocked_commands(command)) is blocked


@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        # A conditional glued behind a separator still hands its command along.
        ('cmd /c echo&if exist . start "" powershell -c ls', True),
        ('cmd /c dir&if /i a==a start "" pwsh', True),
        ('cmd /c echo&if exist . echo start "" powershell', False),
        # POSIX wrappers are not cmd's: `time` is a builtin that shows the
        # clock, so it runs neither start nor what follows it.
        ('time start "" powershell', False),
        ('env start "" pwsh', False),
    ],
)
def test_cmd_grammar_is_not_read_with_posix_rules(monkeypatch, command, blocked):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert bool(tools._find_blocked_commands(command)) is blocked


@pytest.mark.parametrize(
    ("command", "blocked", "bash"),
    [
        # cmd begins a command at the line start, after a separator or a group
        # bracket, after an `if` condition or an `else`, and after the `do` of a
        # `for`. Each of these hid a launch behind one of those.
        ('cmd /c if 1 EQU 1 start "" powershell', True, None),
        ('cmd /c if /i a NEQ b start "" pwsh', True, None),
        ("cmd /c for %i in (x) do start powershell", True, None),
        ("cmd /c for /f %i in (x) do start pwsh", True, None),
        ('cmd /c if exist no_f echo ok else start "" powershell', True, None),
        ('cmd /c if exist x (start "" powershell)', True, None),
        ('cmd /c (start "" pwsh)', True, None),
        # Bash removes the escape before cmd reads the separator, so it can
        # arrive glued to the word in front of it.
        (r'cmd //c echo ok\&start "" powershell', True, r"C:\Program Files\Git\bin\bash.exe"),
        (r'cmd //c dir\|start "" pwsh', True, r"C:\Program Files\Git\bin\bash.exe"),
        # ...and the operands of those same forms are still not launches.
        ('cmd /c if 1 EQU 1 echo start "" powershell', False, None),
        ("cmd /c for %i in (start) do echo powershell", False, None),
        ('cmd /c if exist x echo start "" powershell', False, None),
    ],
)
def test_cmd_command_positions_are_followed_through_its_grammar(
    monkeypatch, command, blocked, bash
):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    assert bool(tools._find_blocked_commands(command)) is blocked


@pytest.mark.parametrize(
    ("command", "blocked", "bash"),
    [
        # `cmdextversion <n>` takes an operand like exist and defined do.
        ('cmd /c if cmdextversion 1 start "" powershell', True, None),
        ('cmd /c if not cmdextversion 2 start "" pwsh', True, None),
        # The caret escapes what follows it and is then dropped, so the two have
        # to be read together: in `ok^&start` the & is an argument, but in
        # `ok^^&start` the first caret escapes the second and the & still
        # separates, which a lookbehind for one caret cannot tell apart.
        ('cmd /c echo ok^&start "" powershell', False, None),
        ('cmd /c echo ok^^&start "" powershell', True, None),
        ('cmd /c echo ok^^^&start "" powershell', False, None),
        # `@` suppresses echoing without changing which command runs.
        ('cmd /c @start "" powershell', True, None),
        ('cmd /c @echo start "" powershell', False, None),
        # A leading redirection is bash's, consumed before cmd is handed its
        # arguments, so it holds the command position rather than taking it.
        ('cmd //c >nul start "" powershell', True, r"C:\Program Files\Git\bin\bash.exe"),
        ('cmd //c 2>nul start "" pwsh', True, r"C:\Program Files\Git\bin\bash.exe"),
        # ...and `else` or `do` only hand off where one of their statements is
        # open, or a printed word reads as a handoff.
        ('cmd /c echo else start "" powershell', False, None),
        ('cmd /c echo do start "" powershell', False, None),
        ('cmd /c if exist no_f echo ok else start "" powershell', True, None),
        # A separator closes the statement with the segment it was opened in,
        # so a printed `else` or `do` after one is not a handoff either.
        ('cmd /c if exist x start "" notepad & echo do start "" powershell', False, None),
        ('cmd /c if exist x start "" notepad & echo else start "" pwsh', False, None),
        ('cmd /c for %i in (x) do start notepad & echo do start "" pwsh', False, None),
    ],
)
def test_cmd_escapes_prefixes_and_control_words(monkeypatch, command, blocked, bash):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    assert bool(tools._find_blocked_commands(command)) is blocked


@pytest.mark.parametrize("bash", [r"C:\Program Files\Git\bin\bash.exe", None])
@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        # `call` reparses its target, so the word behind it is another command.
        ('cmd //c call start "" powershell', True),
        ('cmd //c call call start "" pwsh', True),
        ('cmd //c echo call start "" powershell', False),
    ],
)
def test_call_hands_the_command_along(monkeypatch, bash, command, blocked):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    assert bool(tools._find_blocked_commands(command)) is blocked


@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        # MSYS re-quotes an argument holding whitespace when it builds cmd's
        # command line, and quoting is what makes cmd's separators ordinary
        # data, so the `&` inside one of those is not a separator at all.
        ('cmd //c echo "x &start" "" powershell', False),
        ('cmd //c echo "a & start" "" pwsh', False),
        # ...while one in a word MSYS passes through bare still is.
        (r'cmd //c echo ok\&start "" powershell', True),
    ],
)
def test_a_separator_inside_a_requoted_argument_is_data(monkeypatch, command, blocked):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert bool(tools._find_blocked_commands(command)) is blocked


@pytest.mark.parametrize(
    ("command", "blocked", "platform"),
    [
        # Where a POSIX shell lexes, a group opener is its own token, so one
        # still attached to a word survived quoting and belongs to it: these
        # print their arguments and run none of them.
        ("echo '(bash' -c rm", False, "linux"),
        ("echo '(sh' -c curl", False, "linux"),
        # A real nested shell behind a group is still read there.
        ('x; (bash -c "rm -rf /")', True, "linux"),
        # ...but the cmd lexer keeps the opener glued, where it really is one.
        ('(cmd //c start "" pwsh -Command ls)', True, "win32"),
    ],
)
def test_a_quoted_group_opener_belongs_to_the_word(monkeypatch, command, blocked, platform):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert bool(tools._find_blocked_commands(command)) is blocked


@pytest.mark.parametrize(
    ("command", "blocked", "bash"),
    [
        # A bare redirection operator carries its target in the next word, and
        # that word was taking the command position the operator had kept.
        ('cmd //c > nul start "" powershell', True, r"C:\Program Files\Git\bin\bash.exe"),
        ('cmd //c 2> nul start "" pwsh', True, r"C:\Program Files\Git\bin\bash.exe"),
        ('cmd //c >nul start "" powershell', True, r"C:\Program Files\Git\bin\bash.exe"),
        # The set closer glues to `do` under the cmd lexer, and the body still
        # runs: cmd reads the bracket as its own syntax either way.
        ('cmd /c for %i in (x)do start "" powershell', True, None),
        ('cmd /c for %i in (x) do start "" pwsh', True, None),
        # An explicit extension is a batch file in the working tree, whose
        # arguments are its data; the launcher is the bare builtin.
        ("cmd /c start.cmd powershell", False, None),
        ("cmd /c start.bat rm", False, None),
        ('cmd /c start "" powershell', True, None),
    ],
)
def test_cmd_reads_its_own_brackets_targets_and_extensions(monkeypatch, command, blocked, bash):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    assert bool(tools._find_blocked_commands(command)) is blocked


@pytest.mark.parametrize("bash", [r"C:\Program Files\Git\bin\bash.exe", None])
@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        # `for /f ... in ('CMD')` runs CMD as a child and reads its output, so
        # the set is a command position the token scan sees only as data. The
        # escapes there are the outer parse's, so the child is handed a line
        # with them applied and the `^&` really does separate.
        ("""cmd /c for /f %i in ('echo x ^& start "" powershell') do echo %i""", True),
        ("""cmd /c for /f %i in ('start "" pwsh') do echo %i""", True),
        ("""cmd /c for /f "usebackq" %i in (`start "" powershell`) do echo %i""", True),
    ],
)
def test_for_f_runs_its_quoted_set_as_a_command(monkeypatch, bash, command, blocked):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    assert bool(tools._find_blocked_commands(command)) is blocked


def test_an_unquoted_for_set_is_data_under_the_cmd_lexer(monkeypatch):
    # Only a quoted set is a command; a plain one is the list to iterate. This
    # is the cmd lexer alone, because under Git Bash the parentheses are bash's
    # own and it would run the words inside them in a subshell.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert not tools._find_blocked_commands("cmd /c for /f %i in (start powershell) do echo %i")


@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        # A separator inside a group is between the branch's own commands, so
        # the `if` it belongs to is still open and its `else` still reopens.
        ('cmd /c if exist no_f (echo ok & echo done) else start "" powershell', True),
        ('cmd /c if exist no_f (echo a & echo b) else echo start "" pwsh', False),
        # ...while one outside a group closes the statement as before.
        ('cmd /c if exist x echo ok & echo do start "" powershell', False),
        # The cmd lexer keeps the quotes and splits `"a"=="a"`, whose second
        # half was taking the command position.
        ('cmd /c if "a"=="a" start "" powershell', True),
        ('cmd /c if /i "a" == "a" start "" pwsh', True),
        ('cmd /c if "a"=="a" echo start "" powershell', False),
        # cmd strips the quotes round its own name before running the payload.
        ('"cmd.exe" /c start "" powershell', True),
        ('"cmd" /c start "" pwsh', True),
        # The group opener comes off before the echo-suppression prefix.
        ('cmd /c (@start "" powershell)', True),
        ('cmd /c (@echo start "" powershell)', False),
    ],
)
def test_cmd_groups_quotes_and_prefixes_compose(monkeypatch, command, blocked):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert bool(tools._find_blocked_commands(command)) is blocked


@pytest.mark.parametrize("bash", [r"C:\Program Files\Git\bin\bash.exe", None])
@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        # The set is found by re-reading the text, which cannot tell a construct
        # from a quotation of one, so it only runs where cmd runs a `for`.
        ("""echo "for /f %i in ('rm -rf victim') do echo %i\"""", False),
        ("""cmd /c echo "for /f %i in ('start "" pwsh') do echo %i\"""", False),
        ("""cmd //c for /f %i in ('start "" pwsh') do echo %i""", True),
    ],
)
def test_a_quoted_for_expression_is_not_a_for(monkeypatch, bash, command, blocked):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    assert bool(tools._find_blocked_commands(command)) is blocked


def test_a_bare_for_is_only_cmds_where_cmd_is_the_shell(monkeypatch):
    # The substitution is cmd syntax. Under Git Bash the same text is bash's
    # own: the parentheses open a subshell and the quoted set is one word, so
    # there is no cmd construct to read.
    monkeypatch.setattr(sys, "platform", "win32")
    command = """for /f %i in ('start "" pwsh') do echo %i"""
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert tools._find_blocked_commands(command)
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        # Quoting is the other way cmd documents for passing its separators as
        # arguments, and the cmd lexer keeps the quotes, so these print a word.
        ('cmd /c echo "x&start" "" powershell', False),
        ('cmd /c echo "x|start" "" pwsh', False),
        # ...while the same text unquoted really does begin a command, and a
        # quoted path holding one is still resolved to its program.
        ('cmd /c echo x&start "" powershell', True),
        (r'cmd /c start "" "C:\A&B\powershell.exe"', True),
        ('start "" powershell&echo ok', True),
    ],
)
def test_quotes_protect_cmd_separators_under_the_cmd_lexer(monkeypatch, command, blocked):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert bool(tools._find_blocked_commands(command)) is blocked


@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        # A quoted bracket is argument data, so it neither opens nor closes the
        # branch: closing early on one let the `&` read as outside it and lost
        # the `else` that reopens the command position.
        ('cmd /c if exist no_f (echo ")" & echo done) else start "" powershell', True),
        ('cmd /c if exist no_f (echo "(" & echo done) else start "" pwsh', True),
        ('cmd /c if exist no_f (echo ok & echo done) else start "" powershell', True),
        # A `for` cmd runs somewhere on the line is not this `for`: the outer
        # one here is real while the inner expression is echo's data.
        (
            """cmd /c for %i in (x) do echo "for /f %j in ('start "" powershell') do echo %j\"""",
            False,
        ),
        ("""cmd /c for /f %i in ('start "" pwsh') do echo %i""", True),
    ],
)
def test_quoting_decides_what_is_cmd_syntax_in_place(monkeypatch, command, blocked):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert bool(tools._find_blocked_commands(command)) is blocked


@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        # Quoting protects a caret the same way it protects an operator, so the
        # name cmd looks up is a file really called `power^shell`. Applying the
        # escape to it hard-blocked the benign name as powershell.
        ('cmd /c start "" "power^shell"', False),
        ('cmd /c start "" power^shell', True),
        ('cmd /c start "" "powershell"', True),
        # cmd drops the marks wherever they sit when it resolves the program.
        ('cmd /c start "" pow"ers"hell', True),
        # ...but a quoted bracket is part of the filename, not cmd's grouping.
        ('cmd /c start "" "power)shell"', False),
        ('cmd /c if exist x (start "" powershell)', True),
    ],
)
def test_a_quoted_start_target_is_a_filename(monkeypatch, command, blocked):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert bool(tools._find_blocked_commands(command)) is blocked


def test_a_quoted_caret_reaches_cmd_unquoted_through_git_bash(monkeypatch):
    # The same line the other way round: bash removes the quotes and MSYS
    # re-quotes nothing without whitespace, so cmd is handed a real escape and
    # powershell does launch.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert "powershell" in tools._find_blocked_commands('cmd /c start "" "power^shell"')


@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        # A statement is consumed by its own handoff word, so a later one is an
        # operand again: these only print the words behind `echo`.
        ('cmd /c for %i in (x) do echo do start "" powershell', False),
        ('cmd /c if exist x (echo a) else echo else start "" powershell', False),
        # The real handoffs still count, including one statement's after
        # another's has been used up.
        ('cmd /c for %i in (x) do start "" powershell', True),
        ('cmd /c if exist x (echo a) else start "" powershell', True),
        ('cmd /c if exist x (for %i in (y) do echo a) else start "" powershell', True),
        ('cmd /c for %i in (x) do for %j in (y) do start "" powershell', True),
    ],
)
def test_a_for_or_if_hands_off_once(monkeypatch, command, blocked):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert bool(tools._find_blocked_commands(command)) is blocked


@pytest.mark.parametrize("bash", [r"C:\Program Files\Git\bin\bash.exe", None])
@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        # A `cmd /c` payload is a command line only where that cmd is itself
        # run. echo prints these words, so reading them as one refused a
        # message; the same words in command position really do launch.
        ('echo cmd /c start "" powershell', False),
        ('cmd /c cmd /c start "" powershell', True),
        # cmd launches a start's program, which is deliberately not a command
        # position (it is screened by name), so it is named separately.
        ('cmd /c start "" cmd /c start "" powershell', True),
        ('cmd /c echo start "" cmd /c start "" powershell', False),
    ],
)
def test_a_nested_cmd_payload_needs_a_cmd_that_runs(monkeypatch, bash, command, blocked):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    assert bool(tools._find_blocked_commands(command)) is blocked


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
