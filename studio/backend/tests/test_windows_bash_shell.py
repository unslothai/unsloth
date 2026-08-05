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


_WIN_BASH = r"C:\Program Files\Git\bin\bash.exe"


def _fake_windows_screening(monkeypatch, bash = _WIN_BASH):
    """Screen a command the way a Windows host would, on any runner.

    Faking sys.platform is not enough: _BLOCKED_COMMANDS is derived at import,
    so powershell/pwsh are absent off Windows and the assertions pass on nothing.
    The bash resolver is patched too, since the lexer branch keys off it.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    monkeypatch.setattr(
        tools,
        "_BLOCKED_COMMANDS",
        tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN,
    )


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


def test_windows_only_names_are_not_blocked_off_windows():
    # Why the tests below fake the blocklist as well as the platform:
    # _BLOCKED_COMMANDS is derived at import, so off Windows powershell is not
    # blocked and `assert _find_blocked_commands(...)` asserts the empty set.
    if sys.platform == "win32":
        pytest.skip("the win32 union is already live on this host")
    assert "powershell" in tools._BLOCKED_COMMANDS_WIN
    assert "powershell" not in tools._BLOCKED_COMMANDS


@pytest.mark.parametrize("bash", [_WIN_BASH, None], ids = ["bash", "cmd"])
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
def test_cmd_shellout_is_screened_through_mangled_switches(monkeypatch, command, bash):
    # Git Bash turns a lone /c into a path, so a model writes //c. That spelling
    # skipped the nested scan, making `cmd //c powershell` reachable where
    # `cmd /c powershell` was blocked, and `start` launches its argument too.
    # Screened under both shells: the lexer differs, the verdict must not.
    _fake_windows_screening(monkeypatch, bash = bash)
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize("bash", [_WIN_BASH, None], ids = ["bash", "cmd"])
@pytest.mark.parametrize(
    "command",
    [
        'cmd //c start "" bash -c "bash /c/x.sh"',
        "cmd //c start wt",
        "cmd //c dir",
        "start notepad",
    ],
)
def test_detached_windows_stay_launchable(monkeypatch, command, bash):
    # `start` is the only route to a window on the user's desktop, which the
    # terminal description promises, so screening must not blanket-block cmd.
    # Faked as Windows; off it none of these names are blocked anyway.
    _fake_windows_screening(monkeypatch, bash = bash)
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize("bash", [_WIN_BASH, None], ids = ["bash", "cmd"])
@pytest.mark.parametrize(
    "command",
    [
        # A SPACED title is the other spelling that puts the program one token
        # later, and it reads that way under either shell (see below for why a
        # title without spaces does not).
        'cmd //c start "My Window" powershell -Command ls',
        'cmd //c start /b "Build Step" pwsh -Command ls',
        # The cmd lexer keeps quote marks, so quoted names and payloads never
        # matched the nested-shell scan.
        '"cmd" /c powershell -Command ls',
        'cmd /c "powershell -Command ls"',
        'cmd /c "rm -rf x"',
        '"bash" -c "rm -rf x"',
    ],
)
def test_quoted_shellouts_are_screened(monkeypatch, command, bash):
    _fake_windows_screening(monkeypatch, bash = bash)
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command,blocked_under_cmd",
    [
        ('cmd //c start "MyWindow" powershell -Command ls', "powershell"),
        ('cmd //c start "explorer" .', "."),
        ('cmd //c start "code" .', "."),
    ],
)
def test_a_title_without_spaces_reads_differently_per_shell(
    monkeypatch, command, blocked_under_cmd
):
    # Same text, opposite correct answers. Under cmd the quotes survive on the
    # token, so the word really is a title and the NEXT one is the program.
    # Under bash the quotes are gone before exec and the MSYS runtime re-emits
    # a word with no space bare (cygwin winf.cc, linebuf::fromargv), so cmd
    # receives `start explorer .`, the word itself is the program, and the token
    # behind it is only an argument. Screening it there hard-blocked a
    # legitimate launch on the `.` source builtin.
    _fake_windows_screening(monkeypatch, bash = None)
    assert tools._find_blocked_commands(command) == {blocked_under_cmd}
    _fake_windows_screening(monkeypatch, bash = _WIN_BASH)
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize("bash", [_WIN_BASH, None], ids = ["bash", "cmd"])
@pytest.mark.parametrize(
    "command",
    [
        r'find . -exec env start "" powershell \;',
        r'find . -exec nice start "" powershell \;',
        r'find . -exec env -u FOO start "" powershell \;',
    ],
)
def test_a_wrapped_find_exec_start_is_still_reachable(monkeypatch, command, bash):
    # A command prefix forwards to its target, so the child of the -exec is
    # `start`, not the wrapper. Testing the word directly after the flag read
    # `env` and stopped, leaving the program start launches unscreened.
    _fake_windows_screening(monkeypatch, bash = bash)
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize("bash", [_WIN_BASH, None], ids = ["bash", "cmd"])
@pytest.mark.parametrize(
    "command",
    [
        # An UNQUOTED first argument is the program, so the token after it is
        # just an argument; screening it blocks `.` on `start explorer .`.
        "cmd //c start explorer .",
        "cmd //c start code .",
        r"cmd //c start explorer C:\Users\me\project",
        'cmd //c start "" wt -d .',
        "cmd //c start https://example.com",
        'cmd //c start /min "" myapp.exe --flag',
    ],
)
def test_start_arguments_are_not_read_as_commands(monkeypatch, command, bash):
    _fake_windows_screening(monkeypatch, bash = bash)
    assert not tools._find_blocked_commands(command)


def test_posix_start_scan_is_unaffected():
    # The start scan runs everywhere, so an over-block lands on Linux and macOS.
    assert not tools._find_blocked_commands("start code .")
    assert not tools._find_blocked_commands("start explorer .")


@pytest.mark.parametrize("bash", [_WIN_BASH, None], ids = ["bash", "cmd"])
@pytest.mark.parametrize(
    "command",
    [
        # cmd.exe has no single-quote syntax, and bash strips them before cmd
        # sees the word, so these reach it as the bare `start explorer .` the
        # test above already keeps launchable. Reading the quotes as a title
        # moved the program one token right and hard-blocked the `.` behind it
        # as the POSIX source builtin.
        "cmd //c start 'explorer' .",
        "cmd //c start 'code' .",
        # The same false title from a quote that belongs to another command.
        "echo 'explorer' && cmd //c start explorer .",
    ],
)
def test_single_quoted_start_targets_are_not_titles(monkeypatch, command, bash):
    _fake_windows_screening(monkeypatch, bash = bash)
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize("bash", [_WIN_BASH, None], ids = ["bash", "cmd"])
def test_double_quoted_start_titles_still_move_the_program_along(monkeypatch, bash):
    _fake_windows_screening(monkeypatch, bash = bash)
    assert tools._find_blocked_commands('cmd //c start "My Window" pwsh -Command ls')


def test_a_spaced_single_quoted_start_title_moves_the_program_under_bash(monkeypatch):
    # A title reaches cmd quoted when its CONTENT forces the MSYS runtime to
    # re-quote it, which a space does whichever quote style bash was handed; the
    # program is then the token behind it. cmd itself has no single-quote
    # syntax, so this spelling only reads as a title on the bash host.
    _fake_windows_screening(monkeypatch, bash = _WIN_BASH)
    assert tools._find_blocked_commands("cmd //c start 'My Window' powershell -Command ls")


@pytest.mark.parametrize("bash", [_WIN_BASH, None], ids = ["bash", "cmd"])
@pytest.mark.parametrize(
    "command",
    [
        # The quoted word belongs to the echo, not to start, so the head after
        # `start` is the program. Screening only the token behind a title would
        # read this as titled and let the program through unscreened.
        'echo "powershell" && cmd //c start powershell -Command ls',
        "echo 'rm' && cmd //c start rm -rf /",
    ],
)
def test_a_quote_elsewhere_cannot_hide_the_start_program(monkeypatch, command, bash):
    _fake_windows_screening(monkeypatch, bash = bash)
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize("bash", [_WIN_BASH, None], ids = ["bash", "cmd"])
@pytest.mark.parametrize(
    "command",
    [
        # cmd keeps the quotes around the string after /c when it names an
        # executable file (`cmd /?` rule 1) and never splits a quoted command
        # token on its spaces, and start reads its program argument the same
        # way, so all three really launch pwsh. Only re-lexing the payload split
        # it into `C:/Program` and `Files/...` and the program went unscreened.
        # The sandbox PATH carries System32 but not Program Files, so the full
        # path is the spelling left once the bare `pwsh` is not found.
        'cmd //c "C:/Program Files/PowerShell/7/pwsh.exe" -Command ls',
        'cmd //c start "" "C:/Program Files/PowerShell/7/pwsh.exe"',
        'cmd //c start "My Window" "C:/Program Files/PowerShell/7/pwsh.exe"',
    ],
)
def test_quoted_program_paths_with_spaces_are_screened(monkeypatch, command, bash):
    _fake_windows_screening(monkeypatch, bash = bash)
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize("bash", [_WIN_BASH, None], ids = ["bash", "cmd"])
@pytest.mark.parametrize(
    "command",
    [
        # Only a payload that is ENTIRELY one executable path reads as a
        # program, so an ordinary argument list that merely mentions one keeps
        # working: cmd splits these itself and runs the word in front.
        'cmd //c "C:/Program Files/Git/bin/git.exe" status',
        'cmd //c "C:/Program Files/Microsoft VS Code/Code.exe"',
        'cmd //c "type C:/logs/curl"',
        'cmd //c "echo C:/tools/curl.exe"',
        'cmd //c "copy build/curl.exe dist"',
    ],
)
def test_quoted_payloads_are_not_read_as_programs_by_mistake(monkeypatch, command, bash):
    _fake_windows_screening(monkeypatch, bash = bash)
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize("bash", [_WIN_BASH, None], ids = ["bash", "cmd"])
@pytest.mark.parametrize(
    "command",
    [
        # A `start` the shell never runs launches nothing, so the quoted word
        # behind it is not a title and the word behind THAT is not a program.
        # Reading them as one hard-refused a grep and an echo on a program that
        # never starts. The head is still screened, whatever position it is in.
        'echo start "title" powershell',
        'printf "%s" start "title" powershell',
        'grep -rn start "src/my dir" curl',
    ],
)
def test_a_start_in_argument_position_launches_nothing(monkeypatch, command, bash):
    _fake_windows_screening(monkeypatch, bash = bash)
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize("bash", [_WIN_BASH, None], ids = ["bash", "cmd"])
@pytest.mark.parametrize(
    "command",
    [
        # Every route by which the shell really reaches a start: `cmd /c` hands
        # control one token past a switch, which is no command position of its
        # own, and find/xargs run their child directly.
        'cmd //c start "My Window" pwsh -Command ls',
        'cmd /c start "" powershell -Command ls',
        r'find . -exec start "" powershell \;',
        'xargs start "" powershell',
        'start "" powershell',
    ],
)
def test_a_start_the_shell_runs_still_steps_past_its_title(monkeypatch, command, bash):
    _fake_windows_screening(monkeypatch, bash = bash)
    assert tools._find_blocked_commands(command)


def _fake_git_for_windows(monkeypatch, tmp_path):
    """A trusted Git for Windows layout under a faked Program Files root."""
    program_files = tmp_path / "Program Files"
    bin_dir = program_files / "Git" / "bin"
    usr_bin = program_files / "Git" / "usr" / "bin"
    bin_dir.mkdir(parents = True)
    usr_bin.mkdir(parents = True)
    (bin_dir / "bash.exe").write_text("")
    monkeypatch.setattr(sys, "platform", "win32")
    _fake_trusted_root(monkeypatch, program_files)
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(tools.shutil, "which", lambda _name: None)
    return bin_dir, usr_bin


def test_bash_userland_is_on_the_sandbox_path(monkeypatch, tmp_path):
    # _build_safe_env builds PATH from scratch and bash -c is non-login, so
    # nothing sources /etc/profile: without this ls / cat / grep are missing.
    bin_dir, usr_bin = _fake_git_for_windows(monkeypatch, tmp_path)
    entries = tools._build_safe_env(str(tmp_path / "work"))["PATH"].split(os.pathsep)
    assert os.path.realpath(bin_dir) in entries
    assert os.path.realpath(usr_bin) in entries
    # This server's interpreter stays pinned ahead of a Git-shipped one.
    assert entries[0] == os.path.dirname(sys.executable)


def test_bash_userland_outranks_the_system32_dos_twins(monkeypatch, tmp_path):
    # System32 ships find.exe and sort.exe, which are the DOS commands, not the
    # POSIX ones. Behind them a bare `find . -name '*.py'` in the bash this tool
    # advertises answered "FIND: Parameter format not correct".
    _, usr_bin = _fake_git_for_windows(monkeypatch, tmp_path)
    # Read as one string: os.pathsep is ':' on a Linux runner, which would split
    # the faked C:\Windows drive letter off into its own entry.
    path = tools._build_safe_env(str(tmp_path / "work"))["PATH"]
    system32 = os.path.join(r"C:\Windows", "System32")
    assert system32 in path
    assert path.index(os.path.realpath(usr_bin)) < path.index(system32)
    # ...and the interpreter still outranks the userland, so a Git-shipped
    # python.exe cannot shadow the environment this server runs in.
    assert path.index(os.path.dirname(sys.executable)) < path.index(os.path.realpath(usr_bin))


def test_usr_bin_is_found_from_either_install_layout(monkeypatch, tmp_path):
    # bash.exe ships at Git\bin and Git\usr\bin, so both parents must be probed.
    _, usr_bin = _fake_git_for_windows(monkeypatch, tmp_path)
    (usr_bin / "bash.exe").write_text("")
    monkeypatch.setattr(tools, "_windows_bash", lambda: str(usr_bin / "bash.exe"))
    assert os.path.realpath(usr_bin) in tools._windows_bash_userland_dirs()


def test_untrusted_bash_contributes_no_path_entries(monkeypatch, tmp_path):
    # Same boundary as the git PATH entry: a user-writable dir would let an
    # attacker drop ls.exe beside bash and have a bare name run it.
    shims = tmp_path / "scoop" / "shims"
    shims.mkdir(parents = True)
    (shims / "bash.exe").write_text("")
    monkeypatch.setattr(sys, "platform", "win32")
    _fake_trusted_root(monkeypatch, tmp_path / "Program Files")
    monkeypatch.setattr(os, "environ", {"PATH": str(shims)})
    monkeypatch.setattr(tools.shutil, "which", lambda _name: str(shims / "bash.exe"))
    assert tools._windows_bash_userland_dirs() == []
    entries = tools._build_safe_env(str(tmp_path / "work"))["PATH"].split(os.pathsep)
    assert str(shims) not in entries


def test_no_bash_leaves_the_windows_path_exactly_as_it_was(monkeypatch, tmp_path):
    # The cmd fallback host must see the pre-existing PATH, byte for byte.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(tools.shutil, "which", lambda _name: None)
    env = tools._build_safe_env(str(tmp_path / "work"))
    # os.path.join, so this reads the same on a Linux runner as on Windows.
    sysroot = r"C:\Windows"
    assert env["PATH"] == os.pathsep.join(
        [os.path.dirname(sys.executable), os.path.join(sysroot, "System32"), sysroot]
    )


def test_windows_repoints_temp_and_tmp_at_the_workdir(monkeypatch, tmp_path):
    # Windows reads TEMP/TMP, not TMPDIR, so a native program fell back to
    # GetTempPath and wrote outside the sandbox.
    _fake_git_for_windows(monkeypatch, tmp_path)
    workdir = str(tmp_path / "work")
    env = tools._build_safe_env(workdir)
    assert env["TMPDIR"] == env["TEMP"] == env["TMP"] == workdir


def test_posix_env_is_unchanged(monkeypatch, tmp_path):
    # TEMP/TMP are a Windows concern; POSIX keeps exactly the vars it had.
    monkeypatch.setattr(sys, "platform", "linux")
    env = tools._build_safe_env(str(tmp_path))
    assert "TEMP" not in env
    assert "TMP" not in env
    assert env["PATH"].split(os.pathsep)[-3:] == ["/usr/local/bin", "/usr/bin", "/bin"]


@pytest.mark.parametrize("bash", [_WIN_BASH, None], ids = ["bash", "cmd"])
@pytest.mark.parametrize(
    "command",
    [
        # cmd runs control flow of its own, so the program start launches can
        # sit behind a condition that bash grammar reads as plain arguments.
        r'cmd /c if exist C:\Windows start "" powershell -Command ls',
        r'cmd //c if exist C:\Windows start "" pwsh -Command ls',
        r'cmd /c if not exist C:\nope start "" powershell -Command ls',
    ],
)
def test_start_behind_cmd_control_flow_is_still_screened(monkeypatch, command, bash):
    _fake_windows_screening(monkeypatch, bash = bash)
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize("bash", [_WIN_BASH, None], ids = ["bash", "cmd"])
@pytest.mark.parametrize(
    "command",
    [
        # A shell name a program merely prints or searches for is data. Reading
        # it as a real invocation hard-blocked the payload behind it.
        'echo "cmd" /c powershell',
        "echo cmd /c powershell",
        'printf "%s" "bash" -c "rm -rf x"',
        'grep -rn cmd /c "src/my dir"',
    ],
)
def test_a_shell_name_in_argument_position_invokes_nothing(monkeypatch, command, bash):
    _fake_windows_screening(monkeypatch, bash = bash)
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize("bash", [_WIN_BASH, None], ids = ["bash", "cmd"])
@pytest.mark.parametrize(
    "command",
    [
        # Double-quoted: cmd has no single-quote syntax, so the single-quoted
        # spelling is only a real invocation on the host that has bash, and it
        # is covered by the bash-lexer cases elsewhere in this file.
        'bash -c "rm -rf x"',
        'env FOO=1 bash -c "rm -rf x"',
        r'find . -exec bash -c "rm -rf x" \;',
        '"cmd" /c powershell -Command ls',
        "cmd /c powershell -Command ls",
    ],
)
def test_a_shell_the_shell_runs_still_screens_its_payload(monkeypatch, command, bash):
    # The guard above must not cost the nested-shell scan its real cases.
    _fake_windows_screening(monkeypatch, bash = bash)
    assert tools._find_blocked_commands(command)
