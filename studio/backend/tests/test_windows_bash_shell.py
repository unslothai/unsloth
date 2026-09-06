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
    # Native Windows is intentionally unqualified for ordinary OS-sandboxed
    # execution. This regression owns the existing explicit bypass shell path.
    out = tools._bash_exec(script, disable_sandbox = True)
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
    # sys.platform is too late; fake the resolved set instead.
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
        'cmd //c start "" powershell -Command ls',
        'cmd //c start /b "" pwsh -Command ls',
        'cmd //c start //min "" powershell -Command ls',
        r'cmd //c start /d C:/tmp "" powershell -Command ls',
    ],
)
def test_cmd_shellout_is_screened_through_mangled_switches(command, _windows_blocklist):
    # Git Bash turns a lone /c into a path, so a model writes //c. That spelling
    # skipped the nested scan, making `cmd //c powershell` reachable where
    # `cmd /c powershell` was blocked, and `start` launches its argument too.
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd //c start "" bash -c "bash /c/x.sh"',
        "cmd //c start wt",
        "cmd //c dir",
        "start notepad",
    ],
)
def test_detached_windows_stay_launchable(command, _windows_blocklist):
    # `start` is the only route to a window on the user's desktop, which the
    # terminal description promises, so screening must not blanket-block cmd.
    # The blocklist is faked here too, or the Linux runner asserts this against
    # a set with no powershell in it and cannot see a blanket block at all.
    assert not tools._find_blocked_commands(command)


# Whether Git Bash resolves picks the lexer, and the cmd lexer keeps the quote
# marks the posix one strips. The tests above pin only the shell the Linux runner
# happens to pick, so these run each command through both.
_WINDOWS_SHELLS = [r"C:\Program Files\Git\bin\bash.exe", None]


def _screen_on_windows(monkeypatch, bash, command):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    return tools._find_blocked_commands(command)


@pytest.mark.parametrize("bash", _WINDOWS_SHELLS)
@pytest.mark.parametrize(
    "command",
    [
        # A quoted payload: the cmd lexer hands the recursion `"powershell`.
        'cmd /c "powershell -Command ls"',
        'start "My Title" powershell -Command ls',
        # Switches may follow the title as well as precede it.
        'cmd //c start "my window" /min powershell',
        # A value-style switch, which the old width heuristic did not recognise.
        "cmd /v:on /c powershell -Command ls",
        # Git Bash doubles the slash on the switches ahead of /c too.
        "cmd //v:on //c powershell -Command ls",
        # MSYS rewrites a POSIX path before cmd sees it, so the program arrives
        # with a leading slash. Skipping every /-word as a switch stepped over it.
        'cmd //c start "" /c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -Command ls',
        'start "" powershell -Command ls',
        'cmd //c env start "" powershell',
    ],
)
def test_windows_shellouts_are_screened_on_either_shell(
    monkeypatch, bash, command, _windows_blocklist
):
    assert _screen_on_windows(monkeypatch, bash, command)


@pytest.mark.parametrize("bash", _WINDOWS_SHELLS)
@pytest.mark.parametrize(
    "command",
    [
        'start "" notepad readme.txt',
        'start "Build" npm run build',
        'cmd /c "echo hello"',
        r'echo "C:\Windows\System32\cmd.exe"',
        # A document opened through its file association, what `start ""` is for.
        r'cmd //c start "" "C:\Users\me\My Documents\report.docx"',
        "npm start",
        "./start.sh",
        # `start` and `cmd` as data, not as programs. A shell name the shell
        # never runs is text, however it is spelled after it.
        "echo start notepad powershell",
        "grep start README powershell",
    ],
)
def test_ordinary_windows_commands_stay_runnable(monkeypatch, bash, command, _windows_blocklist):
    assert not _screen_on_windows(monkeypatch, bash, command)


@pytest.mark.parametrize(
    "command",
    [
        # The documented `start "title" prog` form. Only the cmd lexer can see
        # it: the posix lexer has stripped the marks, leaving a one-word title
        # indistinguishable from a program name, and guessing there would refuse
        # ordinary text like `echo start notepad powershell`.
        'cmd //c start "job" powershell -Command ls',
        'cmd /c start "t" pwsh -c ls',
    ],
)
def test_a_single_word_start_title_is_screened_under_the_cmd_lexer(
    monkeypatch, command, _windows_blocklist
):
    assert _screen_on_windows(monkeypatch, None, command)


@pytest.mark.parametrize("bash", _WINDOWS_SHELLS)
@pytest.mark.parametrize(
    "command",
    [
        # An assignment prefixes a command the shell still runs, and a shell
        # handed to another shell is still run. Deciding "is this token executed"
        # from the token before it missed all of these, which is a bypass and
        # strictly worse than the echo-data over-blocks it was meant to fix.
        'FOO=1 bash -c "rm -rf x"',
        'env FOO=1 bash -c "rm -rf x"',
        'FOO=1 start "" powershell',
        'cmd //c start "" cmd /c powershell',
        # A wrapper option's value sits between the wrapper and `start`, so the
        # token before `start` says nothing about whether it is executed.
        'cmd //c env -u FOO start "" powershell',
        'cmd //c nice -n 5 start "" powershell',
    ],
)
def test_a_prefixed_shell_is_still_screened(monkeypatch, bash, command, _windows_blocklist):
    # An assignment is not cmd syntax, so under the cmd lexer `FOO=1` is itself
    # the program name and the rest really is its arguments.
    if bash is None and not command.startswith("cmd "):
        pytest.skip("assignment prefixes are not cmd syntax")
    assert _screen_on_windows(monkeypatch, bash, command)


def test_a_program_path_is_not_read_as_a_cmd_switch(monkeypatch, _windows_blocklist):
    # Matched loosely, the pattern would find the `/b` of /bin/bash, skip the
    # token as a flag, and never reach the shell name behind it.
    assert not tools._CMD_SWITCH_RE.fullmatch("/bin/bash")
    assert _screen_on_windows(monkeypatch, _WINDOWS_SHELLS[0], '/bin/bash -c "rm -rf x"') == {"rm"}


def test_os_isolated_windows_launch_uses_cmd_even_with_bash(monkeypatch):
    # MSYS2 bash cannot start inside an AppContainer, so the isolated launch
    # runs cmd while every other mode keeps the bash the host has.
    monkeypatch.setattr(tools.sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\\Program Files\\Git\\bin\\bash.exe")
    assert tools._get_shell_cmd("echo hi")[0].endswith("bash.exe")
    assert tools._get_shell_cmd("echo hi", os_isolated = False)[0].endswith("bash.exe")
    assert tools._get_shell_cmd("echo hi", os_isolated = True) == ["cmd", "/c", "echo hi"]
    # The real launch hands cmd a batch file so every line of the command runs.
    # `call`, not the bare path: inside the container cmd's search for a command
    # named like a batch file is refused even when the file itself reads fine.
    assert tools._get_shell_cmd("echo hi", os_isolated = True, script_path = r"C:\w\studio_exec_a.cmd") == [
        "cmd", "/d", "/c", "call", r"C:\w\studio_exec_a.cmd",
    ]


def test_isolated_batch_script_carries_every_line_with_echo_off(tmp_path):
    path = tools._reserve_isolated_batch_script(str(tmp_path))
    assert os.path.basename(path).startswith("studio_exec_") and path.endswith(".cmd")
    # Reserving the name must not create the file: it is written only after the
    # sandbox is prepared, so that it inherits the container's ACE.
    assert not os.path.exists(path)
    # On Windows the file is created through a handle that denies writers and
    # that handle comes back for the caller to release; elsewhere an exclusive
    # create is enough and there is nothing to hold.
    handle = tools._write_isolated_batch_script("echo one\necho two\r\nexit /b 3", path)
    assert (handle is None) is (sys.platform != "win32")
    try:
        with open(path, "rb") as reader:
            body = reader.read()
        assert body == b"@echo off\r\necho one\r\necho two\r\nexit /b 3\r\n"
        # The name is never reused over an existing file.
        with pytest.raises(OSError) as excinfo:
            tools._write_isolated_batch_script("echo again", path)
        assert isinstance(excinfo.value, FileExistsError) or excinfo.value.winerror == 80
    finally:
        tools._release_batch_script(handle)


@pytest.mark.skipif(sys.platform != "win32", reason = "CreateFileW sharing is Windows only")
def test_the_isolated_batch_script_is_unwritable_while_the_launch_holds_it(tmp_path):
    # Creating and locking are one operation, so there is no window in which a
    # concurrent call in the same chat workdir can swap the script under cmd.
    path = tools._reserve_isolated_batch_script(str(tmp_path))
    handle = tools._write_isolated_batch_script("echo held", path)
    assert handle is not None
    try:
        # A reader (cmd) is admitted, every writer and the delete are refused.
        with open(path, "rb") as reader:
            assert reader.read() == b"@echo off\r\necho held\r\n"
        with pytest.raises(PermissionError):
            open(path, "wb").close()
        with pytest.raises(PermissionError):
            os.remove(path)
    finally:
        tools._release_batch_script(handle)
    # Released with the launch: the caller can clean the file up afterwards.
    os.remove(path)


def test_os_isolated_description_names_cmd_only_on_windows_with_bash(monkeypatch):
    specs = [dict(tools.TERMINAL_TOOL), dict(tools.PYTHON_TOOL)]
    monkeypatch.setattr(tools.sys, "platform", "linux")
    assert tools.apply_os_isolated_tool_descriptions(specs) is specs

    monkeypatch.setattr(tools.sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert tools.apply_os_isolated_tool_descriptions(specs) is specs

    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\\Git\\bin\\bash.exe")
    bash_tools = [
        {
            **tools.TERMINAL_TOOL,
            "function": {
                **tools.TERMINAL_TOOL["function"],
                "description": "Run a command." + tools._TERMINAL_BASH_NOTE,
            },
        },
        tools.PYTHON_TOOL,
    ]
    swapped = tools.apply_os_isolated_tool_descriptions(bash_tools)
    assert swapped is not bash_tools
    terminal = swapped[0]["function"]["description"]
    assert "bash (Git for Windows)" not in terminal
    assert "The shell is cmd, not bash" in terminal
    assert swapped[1] is tools.PYTHON_TOOL
    # The module constants are never mutated.
    assert "The shell is cmd" not in bash_tools[0]["function"]["description"]


def test_batch_script_lock_is_windows_only_and_never_raises(tmp_path, monkeypatch):
    # Creating the script and denying every writer is one operation, and it is
    # fail-closed: a host where the API refuses raises rather than running a
    # script anyone could swap. Releasing a handle never raises, since it runs
    # in the launch's finally.
    monkeypatch.setattr(tools.sys, "platform", "linux")
    tools._release_batch_script(None)
    tools._release_batch_script(object())
    monkeypatch.setattr(tools.sys, "platform", "win32")
    monkeypatch.setattr(tools, "_create_locked_batch_script", _raise_oserror)
    with pytest.raises(OSError):
        tools._write_isolated_batch_script("echo hi", str(tmp_path / "studio_exec_x.cmd"))
    assert not list(tmp_path.iterdir())
    tools._release_batch_script(object())


def _raise_oserror(*args, **kwargs):
    raise OSError("no WinDLL on this host")


def test_a_replaced_python_script_is_refused_before_anything_runs(tmp_path):
    # Both tool calls of one chat share a workdir the sandboxed process can
    # write, so a second call could swap the first call's script between the
    # write and the interpreter opening it, and the wrong code would run under
    # the first call's tier and grant.
    script = tmp_path / "studio_exec_victim.py"
    script.write_text("print('victim')", encoding = "utf-8")
    handle, identity = tools._seal_scratch_script(str(script))
    try:
        assert not tools._scratch_script_was_swapped(str(script), identity)
        # Rewriting the same bytes in place is not a swap.
        script.write_text("print('victim')", encoding = "utf-8")
        assert not tools._scratch_script_was_swapped(str(script), identity)
        if sys.platform == "win32":
            # Windows prevents it outright: the handle denies delete and write.
            assert handle is not None
            with pytest.raises(PermissionError):
                os.replace(str(tmp_path / "attacker.py"), str(script))
        else:
            # POSIX cannot prevent it, so it is detected: a replacement gets a
            # new inode and the launch is refused.
            assert identity is not None
            attacker = tmp_path / "attacker.py"
            attacker.write_text("print('attacker')", encoding = "utf-8")
            os.replace(str(attacker), str(script))
            assert tools._scratch_script_was_swapped(str(script), identity)
        # A script that vanished entirely is a swap too, never a silent pass.
    finally:
        tools._release_batch_script(handle)
    os.unlink(str(script))
    assert tools._scratch_script_was_swapped(str(script), identity)


def test_the_python_launch_checks_the_script_after_preparing_it(tmp_path):
    # The check has to sit between prepare_tool_launch and the spawn: earlier
    # and a swap during preparation is missed, later and the child is already
    # running someone else's code.
    source = Path(tools.__file__).read_text(encoding = "utf-8")
    body = source.split("def _python_exec(", 1)[1].split("def ", 1)[0]
    seal = body.index("_seal_scratch_script(")
    check = body.index("_scratch_script_was_swapped(")
    prepare = body.index("prepare_tool_launch(")
    spawn = body.index("popen_kwargs = dict(")
    assert seal < prepare < check < spawn
