# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Optional launchd user agent for Studio on macOS (CLI installs; the launchd twin of the systemd unit).

Runs on Linux too: `uname`, `id` and `launchctl` are faked on PATH.
"""

from __future__ import annotations

import os
import plistlib
import re
import shutil
import subprocess
import sys

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = REPO_ROOT / "install.sh"
LAUNCHD_INSTALL_SH = REPO_ROOT / "studio" / "launchd" / "install_user_agent.sh"
UNINSTALL_SH = REPO_ROOT / "scripts" / "uninstall.sh"
LABEL = "ai.unsloth.studio.server"
MARKER = "<!-- unsloth-studio-managed-launchd -->"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason = "launchd helper is a POSIX shell script",
)


def _fake_bin(tmp_path: Path, name: str, body: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok = True)
    exe = bin_dir / name
    exe.write_text(f"#!/bin/sh\n{body}\n")
    exe.chmod(0o755)
    return exe


def _fake_mac(tmp_path: Path, launchctl: str = "exit 0") -> Path:
    """Fakes a Mac; launchctl logs its args to launchctl.log, then runs `launchctl`."""
    log = tmp_path / "launchctl.log"
    _fake_bin(tmp_path, "uname", "echo Darwin")
    _fake_bin(tmp_path, "id", "echo 501")
    _fake_bin(tmp_path, "launchctl", f'echo "$*" >> "{log}"\n{launchctl}')
    return log


def _env(tmp_path: Path, **extra: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_LAUNCHD_")}
    env["HOME"] = str(tmp_path / "home")
    env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}"
    env.update(extra)
    return env


def _plist_path(tmp_path: Path) -> Path:
    return tmp_path / "home" / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _run_helper(
    tmp_path: Path,
    *args: str,
    exe: Path | None = None,
    **env: str,
):
    if exe is None:
        exe = _fake_bin(tmp_path, "unsloth", "exit 0")
    return subprocess.run(
        ["bash", str(LAUNCHD_INSTALL_SH), "--unsloth-exe", str(exe), *args],
        capture_output = True,
        text = True,
        env = _env(tmp_path, **env),
    )


def _write_agent(tmp_path: Path, *args: str, **env: str) -> dict:
    _fake_mac(tmp_path)
    r = _run_helper(tmp_path, *args, **env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(_plist_path(tmp_path))
    text = _plist_path(tmp_path).read_text()
    assert text.splitlines()[1] == MARKER
    assert "@@" not in text
    return plistlib.loads(text.encode())


def test_agent_defaults_to_loopback(tmp_path):
    agent = _write_agent(tmp_path, "--port", "9090")
    assert agent["Label"] == LABEL
    assert agent["ProgramArguments"][1:] == ["studio", "-H", "127.0.0.1", "-p", "9090"]
    assert agent["RunAtLoad"] is True
    assert agent["KeepAlive"] == {"SuccessfulExit": False}
    assert agent["ThrottleInterval"] == 10
    assert "UNSLOTH_STUDIO_HOME" not in agent["EnvironmentVariables"]


@pytest.mark.parametrize(
    ("args", "env"),
    [(("--host", "0.0.0.0"), {}), ((), {"UNSLOTH_LAUNCHD_HOST": "0.0.0.0"})],
)
def test_agent_lan_bind_is_opt_in(tmp_path, args, env):
    agent = _write_agent(tmp_path, *args, **env)
    assert agent["ProgramArguments"][2:] == ["-H", "0.0.0.0", "-p", "8888"]


def test_agent_path_reaches_the_venv_and_homebrew(tmp_path):
    # launchd's own PATH is /usr/bin:/bin:/usr/sbin:/sbin.
    path = _write_agent(tmp_path)["EnvironmentVariables"]["PATH"].split(":")
    assert path[0] == str((tmp_path / "bin").resolve())
    assert "/opt/homebrew/bin" in path
    assert str(tmp_path / "home" / ".local" / "bin") in path


def test_agent_path_keeps_the_installers_node(tmp_path):
    nvm_bin = tmp_path / "nvm" / "bin"
    nvm_bin.mkdir(parents = True)
    node = nvm_bin / "node"
    node.write_text("#!/bin/sh\n")
    node.chmod(0o755)
    _fake_mac(tmp_path)
    r = _run_helper(tmp_path, PATH = f"{tmp_path / 'bin'}:{nvm_bin}:/usr/bin:/bin")
    assert r.returncode == 0, r.stderr
    path = plistlib.loads(_plist_path(tmp_path).read_bytes())["EnvironmentVariables"]["PATH"]
    assert path.split(":")[1] == str(nvm_bin)


def test_agent_runs_inside_the_studio_home(tmp_path):
    agent = _write_agent(tmp_path)
    home = tmp_path / "home" / ".unsloth" / "studio"
    assert agent["WorkingDirectory"] == str(home)
    # launchd refuses to start a job whose working directory is missing.
    assert home.is_dir()


def test_agent_leaves_logging_to_studio(tmp_path):
    # Studio tees both streams to logs/server (0600, rotated); a launchd copy would never rotate
    # and would hold the startup API key in a world-readable file.
    agent = _write_agent(tmp_path)
    assert agent["StandardOutPath"] == "/dev/null"
    assert agent["StandardErrorPath"] == "/dev/null"


def test_agent_round_trips_xml_special_paths(tmp_path):
    odd = tmp_path / "a & <b> 'c' \"d\" 100% $HOME"
    odd.mkdir()
    exe = odd / "unsloth"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    _fake_mac(tmp_path)
    r = _run_helper(tmp_path, "--studio-home", str(odd / "home"), exe = exe)
    assert r.returncode == 0, r.stderr
    agent = plistlib.loads(_plist_path(tmp_path).read_bytes())
    assert agent["ProgramArguments"][0] == str(exe.resolve())
    assert agent["EnvironmentVariables"]["UNSLOTH_STUDIO_HOME"] == str((odd / "home").resolve())


def test_agent_has_no_stop_hook_that_stops_other_studios(tmp_path):
    # `unsloth studio stop` would also stop servers the user started by hand.
    agent = _write_agent(tmp_path)
    assert "stop" not in agent["ProgramArguments"]
    assert set(agent) == {
        "Label",
        "ProgramArguments",
        "EnvironmentVariables",
        "WorkingDirectory",
        "RunAtLoad",
        "KeepAlive",
        "ThrottleInterval",
        "ExitTimeOut",
        "ProcessType",
        "StandardOutPath",
        "StandardErrorPath",
    }


def test_start_enables_then_bootstraps(tmp_path):
    log = _fake_mac(tmp_path, '[ "$1" = print ] && [ "$2" != gui/501 ] && exit 113; exit 0')
    r = _run_helper(tmp_path, "--start")
    assert r.returncode == 0, r.stderr
    assert log.read_text().splitlines() == [
        "print gui/501",
        f"enable gui/501/{LABEL}",
        f"print gui/501/{LABEL}",
        f"bootstrap gui/501 {_plist_path(tmp_path)}",
    ]


def test_start_unloads_a_running_agent_first(tmp_path):
    loaded = tmp_path / "loaded"
    loaded.touch()
    log = _fake_mac(
        tmp_path,
        f'case "$1 $2" in "print gui/501/{LABEL}") [ -e "{loaded}" ] && exit 0; exit 113 ;; '
        f'"bootout "*) rm -f "{loaded}" ;; esac; exit 0',
    )
    r = _run_helper(tmp_path, "--start")
    assert r.returncode == 0, r.stderr
    calls = log.read_text().splitlines()
    assert calls.index(f"bootout gui/501/{LABEL}") < calls.index(
        f"bootstrap gui/501 {_plist_path(tmp_path)}"
    )


def test_failed_bootstrap_leaves_no_plist_to_start_at_login(tmp_path):
    _fake_mac(
        tmp_path,
        f'[ "$1 $2" = "print gui/501/{LABEL}" ] && exit 113; [ "$1" = bootstrap ] && exit 5; exit 0',
    )
    r = _run_helper(tmp_path, "--start")
    assert r.returncode == 1
    assert "bootstrap" in r.stderr and "next login" in r.stderr
    assert not _plist_path(tmp_path).exists()


def test_failed_enable_leaves_no_plist(tmp_path):
    _fake_mac(tmp_path, '[ "$1" = enable ] && exit 5; exit 0')
    r = _run_helper(tmp_path, "--enable")
    assert r.returncode == 1
    assert not _plist_path(tmp_path).exists()


def test_unload_wait_outlasts_the_agent_exit_timeout():
    # bootout can take up to ExitTimeOut for a Studio mid-shutdown; giving up sooner reports a false failure.
    helper = LAUNCHD_INSTALL_SH.read_text(encoding = "utf-8")
    template = (LAUNCHD_INSTALL_SH.parent / "ai.unsloth.studio.server.plist.in").read_text(
        encoding = "utf-8"
    )
    exit_timeout = int(
        re.search(r"<key>ExitTimeOut</key>\s*<integer>(\d+)</integer>", template).group(1)
    )
    wait = int(re.search(r'"\$_waited" -ge (\d+)', helper).group(1))
    assert wait > exit_timeout


def test_enable_without_start_does_not_load_now(tmp_path):
    log = _fake_mac(tmp_path)
    r = _run_helper(tmp_path, "--enable")
    assert r.returncode == 0, r.stderr
    assert log.read_text().splitlines() == ["print gui/501", f"enable gui/501/{LABEL}"]


def test_no_gui_session_fails_clearly(tmp_path):
    _fake_mac(tmp_path, "exit 113")
    r = _run_helper(tmp_path, "--start")
    assert r.returncode == 1
    assert "no GUI login session" in r.stderr
    assert not _plist_path(tmp_path).exists()


def test_helper_refuses_off_macos(tmp_path):
    _fake_mac(tmp_path)
    _fake_bin(tmp_path, "uname", "echo Linux")
    r = _run_helper(tmp_path)
    assert r.returncode == 1
    assert "macOS only" in r.stderr
    assert not _plist_path(tmp_path).exists()


@pytest.mark.parametrize(
    ("args", "env"),
    [
        (("--port", "0"), {}),
        (("--port", "80a"), {}),
        (("--host", ""), {}),
        (("--studio-home", "home\nKeepAlive"), {}),
        ((), {"UNSLOTH_LAUNCHD_HOST": "a\tb"}),
    ],
)
def test_helper_rejects_values_that_break_the_agent(tmp_path, args, env):
    _fake_mac(tmp_path)
    r = _run_helper(tmp_path, *args, **env)
    assert r.returncode == 2, r.stderr
    assert not _plist_path(tmp_path).exists()


def test_agent_goes_to_the_login_accounts_home_when_home_is_redirected(tmp_path):
    account = tmp_path / "account"
    _fake_bin(tmp_path, "dscl", f"echo 'NFSHomeDirectory: {account}'")
    _fake_mac(tmp_path)
    r = _run_helper(tmp_path)
    assert r.returncode == 0, r.stderr
    real = account / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    assert r.stdout.strip() == str(real)
    assert real.read_text().splitlines()[1] == MARKER
    assert not _plist_path(tmp_path).exists()
    agent = plistlib.loads(real.read_bytes())
    assert agent["EnvironmentVariables"]["HOME"] == str(tmp_path / "home")


def test_agent_keeps_the_login_home_unset_when_home_is_not_redirected(tmp_path):
    assert "HOME" not in _write_agent(tmp_path)["EnvironmentVariables"]


def test_helper_never_replaces_an_agent_the_user_wrote(tmp_path):
    plist = _plist_path(tmp_path)
    plist.parent.mkdir(parents = True)
    mine = f'<?xml version="1.0"?>\n<plist><!-- mentions {MARKER} later --></plist>\n'
    plist.write_text(mine)
    _fake_mac(tmp_path)
    r = _run_helper(tmp_path)
    assert r.returncode == 1
    assert "was not written by Unsloth" in r.stderr
    assert plist.read_text() == mine
    assert list(plist.parent.iterdir()) == [plist]


def test_helper_rewrites_its_own_agent(tmp_path):
    _write_agent(tmp_path, "--port", "9000")
    assert _write_agent(tmp_path, "--port", "9001")["ProgramArguments"][-1] == "9001"


_GATE = re.compile(
    r'^if \[ "\$_INSTALL_LAUNCHD" = true \]; then\n.*?^fi\n', re.DOTALL | re.MULTILINE
)


def _harness() -> str:
    source = INSTALL_SH.read_text(encoding = "utf-8")
    parts = ['step() { echo "STEP $*"; }\nsubstep() { :; }\n']
    for name in ("_resolve_launchd_install_script", "_install_launchd_user_agent"):
        m = re.search(rf"^{name}\(\) \{{.*?^\}}", source, flags = re.DOTALL | re.MULTILINE)
        assert m is not None, name
        parts.append(m.group(0))
    gate = _GATE.search(source)
    assert gate is not None
    parts.append(gate.group(0))
    parts.append('echo "STARTED=$_LAUNCHD_STARTED SKIP_AUTOSTART=$_SKIP_AUTOSTART"\n')
    return "\n".join(parts)


def _run_installer_tail(
    tmp_path: Path,
    cwd: Path | None = None,
    real_python: bool = False,
    **vars: str,
) -> str:
    script = tmp_path / "helper.sh"
    log = tmp_path / "helper.log"
    script.write_text(f'echo "$*" > "{log}"\n')
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents = True, exist_ok = True)
    py = venv / "bin" / "python"
    if real_python:
        py.symlink_to(sys.executable)
    else:
        py.write_text(f'#!/bin/sh\necho "{script}"\n')
        py.chmod(0o755)
    preset = {
        "OS": "macos",
        "_INSTALL_LAUNCHD": "false",
        "_LAUNCHD_STARTED": "false",
        "_SKIP_AUTOSTART": "false",
        "_REPO_IS_CHECKOUT": "0",
        "_REPO_ROOT": str(tmp_path),
        "VENV_DIR": str(venv),
        "STUDIO_HOME": str(tmp_path / "studio_home"),
        "_STUDIO_HOME_REDIRECT": "default",
        **vars,
    }
    assigns = "".join(f"{k}='{v}'\n" for k, v in preset.items())
    out = subprocess.run(
        ["sh", "-ec", assigns + _harness()],
        check = True,
        capture_output = True,
        text = True,
        env = _env(tmp_path),
        stdin = subprocess.DEVNULL,
        cwd = cwd,
    ).stdout
    return out + (log.read_text() if log.exists() else "")


def test_flag_off_changes_nothing(tmp_path):
    out = _run_installer_tail(tmp_path)
    assert out == "STARTED=false SKIP_AUTOSTART=false\n"


def test_opt_in_installs_and_skips_the_launch_prompt(tmp_path):
    out = _run_installer_tail(tmp_path, _INSTALL_LAUNCHD = "true")
    assert "STARTED=true SKIP_AUTOSTART=true" in out
    assert f"--unsloth-exe {tmp_path / 'venv' / 'bin' / 'unsloth'}" in out
    assert "--host 127.0.0.1 --port 8888 --enable --start" in out
    assert "--studio-home" not in out


@pytest.mark.parametrize("redirect", ["env", "home"])
def test_opt_in_passes_studio_home_when_redirected(tmp_path, redirect):
    out = _run_installer_tail(tmp_path, _INSTALL_LAUNCHD = "true", _STUDIO_HOME_REDIRECT = redirect)
    assert f"--studio-home {tmp_path / 'studio_home'}" in out


@pytest.mark.parametrize("os_name", ["linux", "wsl"])
def test_opt_in_off_macos_warns_and_keeps_launch_prompt(tmp_path, os_name):
    out = _run_installer_tail(tmp_path, _INSTALL_LAUNCHD = "true", OS = os_name)
    assert "macOS only" in out
    assert "STARTED=false SKIP_AUTOSTART=false" in out
    assert "--unsloth-exe" not in out


def test_opt_in_defers_to_the_desktop_apps_login_agent(tmp_path):
    desktop = tmp_path / "home" / "Library" / "LaunchAgents" / "Unsloth.plist"
    desktop.parent.mkdir(parents = True)
    desktop.write_text("<plist/>\n")
    out = _run_installer_tail(tmp_path, _INSTALL_LAUNCHD = "true")
    assert "Run at login is on; skipped" in out
    assert "STARTED=false SKIP_AUTOSTART=false" in out
    assert "--unsloth-exe" not in out


def test_opt_in_sees_the_desktop_agent_in_the_login_accounts_home(tmp_path):
    account = tmp_path / "account"
    desktop = account / "Library" / "LaunchAgents" / "Unsloth.plist"
    desktop.parent.mkdir(parents = True)
    desktop.write_text("<plist/>\n")
    _fake_bin(tmp_path, "dscl", f"echo 'NFSHomeDirectory: {account}'")
    out = _run_installer_tail(tmp_path, _INSTALL_LAUNCHD = "true")
    assert "Run at login is on; skipped" in out
    assert "--unsloth-exe" not in out


def test_piped_install_never_runs_a_helper_planted_in_cwd(tmp_path):
    planted = tmp_path / "studio" / "launchd" / "install_user_agent.sh"
    planted.parent.mkdir(parents = True)
    planted.write_text(f'touch "{tmp_path / "planted_ran"}"\n')
    assert "STARTED=true" in _run_installer_tail(tmp_path, _INSTALL_LAUNCHD = "true")
    assert not (tmp_path / "planted_ran").exists()


def test_package_lookup_ignores_a_studio_package_planted_in_cwd(tmp_path):
    # Real interpreter: `python -c` puts the cwd first on sys.path; the lookup must not.
    cwd = tmp_path / "cwd"
    (cwd / "studio" / "launchd").mkdir(parents = True)
    (cwd / "studio" / "__init__.py").write_text(f"open({str(tmp_path / 'imported')!r}, 'w')\n")
    (cwd / "studio" / "launchd" / "install_user_agent.sh").write_text(
        f'touch "{tmp_path / "planted_ran"}"\n'
    )
    _run_installer_tail(tmp_path, cwd = cwd, real_python = True, _INSTALL_LAUNCHD = "true")
    assert not (tmp_path / "imported").exists()
    assert not (tmp_path / "planted_ran").exists()


def test_installer_asks_no_launchd_question():
    source = INSTALL_SH.read_text(encoding = "utf-8")
    assert not re.search(r"printf[^\n]*launchd[^\n]*\?", source)
    assert _GATE.search(source).start() < source.index("Start Unsloth Studio now? [Y/n]")


def _uninstall_removal_script(uid: str = "501") -> str:
    source = UNINSTALL_SH.read_text(encoding = "utf-8")
    parts = [f"_os=Darwin\n_uid={uid}\n"]
    for name in ("_set_marker", "_remove_path", "_home_uid", "_run_as_home_owner"):
        m = re.search(rf"^{name}\(\) \{{.*?^\}}", source, flags = re.DOTALL | re.MULTILINE)
        parts.append(m.group(0))
    m = re.search(
        r"^    _remove_launchd_user_agent\(\) \{.*?^    \}",
        source,
        flags = re.DOTALL | re.MULTILINE,
    )
    assert m is not None
    parts.append(m.group(0) + "\n_remove_launchd_user_agent\n")
    return "\n".join(parts)


def _run_uninstall_removal(
    tmp_path: Path,
    plist_text: str | None,
    launchctl: str,
    caller_uid: str = "501",
) -> str:
    if plist_text is not None:
        plist = _plist_path(tmp_path)
        plist.parent.mkdir(parents = True)
        plist.write_text(plist_text)
    log = _fake_mac(tmp_path, launchctl)
    _fake_bin(tmp_path, "stat", "echo 501")
    _fake_bin(tmp_path, "id", f"echo {caller_uid}")
    _fake_bin(tmp_path, "sudo", "exit 0")
    r = subprocess.run(
        ["sh", "-c", _uninstall_removal_script(caller_uid)],
        capture_output = True,
        text = True,
        env = _env(tmp_path),
        check = True,
    )
    return r.stdout + r.stderr + (log.read_text() if log.exists() else "")


_OURS = f'<?xml version="1.0" encoding="UTF-8"?>\n{MARKER}\n<plist/>\n'


def test_uninstall_leaves_agents_it_did_not_write(tmp_path):
    text = f'<?xml version="1.0"?>\n<plist><!-- {MARKER} --></plist>\n'
    assert _run_uninstall_removal(tmp_path, text, "exit 0") == ""
    assert _plist_path(tmp_path).exists()


def test_uninstall_without_an_agent_does_nothing(tmp_path):
    assert _run_uninstall_removal(tmp_path, None, "exit 0") == ""


def test_uninstall_boots_out_then_removes(tmp_path):
    out = _run_uninstall_removal(tmp_path, _OURS, "exit 0")
    assert f"bootout gui/501/{LABEL}" in out
    assert "Removed launchd user agent" in out
    assert "could not unload" not in out
    assert not _plist_path(tmp_path).exists()


def test_uninstall_of_an_unloaded_agent_just_removes_it(tmp_path):
    out = _run_uninstall_removal(tmp_path, _OURS, '[ "$1" = print ] && exit 113; exit 0')
    assert "bootout" not in out
    assert "could not unload" not in out
    assert not _plist_path(tmp_path).exists()


def test_uninstall_falls_back_to_remove_when_bootout_fails(tmp_path):
    # A still-loaded job would be respawned by KeepAlive after the kill sweep.
    out = _run_uninstall_removal(tmp_path, _OURS, '[ "$1" = bootout ] && exit 5; exit 0')
    assert f"remove {LABEL}" in out
    assert "could not unload" not in out
    assert not _plist_path(tmp_path).exists()


def test_uninstall_still_removes_the_file_when_launchd_refuses(tmp_path):
    launchctl = 'case "$1" in bootout|remove) exit 5 ;; esac; exit 0'
    out = _run_uninstall_removal(tmp_path, _OURS, launchctl)
    assert "could not unload it from launchd" in out
    assert not _plist_path(tmp_path).exists()


def test_uninstall_under_sudo_unloads_the_home_owners_agent(tmp_path):
    out = _run_uninstall_removal(
        tmp_path, _OURS, '[ "$1" = bootout ] && exit 5; exit 0', caller_uid = "0"
    )
    assert f"print gui/501/{LABEL}" in out
    assert f"bootout gui/501/{LABEL}" in out
    assert f"asuser 501 sudo -u #501 launchctl remove {LABEL}" in out
    assert "gui/0/" not in out
    assert not _plist_path(tmp_path).exists()


def test_uninstall_finds_the_agent_in_the_login_accounts_home(tmp_path):
    account = tmp_path / "account"
    real = account / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    real.parent.mkdir(parents = True)
    real.write_text(_OURS)
    _fake_bin(tmp_path, "dscl", f"echo 'NFSHomeDirectory: {account}'")
    out = _run_uninstall_removal(tmp_path, None, "exit 0")
    assert f"bootout gui/501/{LABEL}" in out
    assert not real.exists()


def test_uninstall_under_sudo_looks_up_the_home_owners_account(tmp_path):
    account = tmp_path / "account"
    real = account / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    real.parent.mkdir(parents = True)
    real.write_text(_OURS)
    dscl_log = tmp_path / "dscl.log"
    _fake_bin(
        tmp_path,
        "dscl",
        f'echo "$*" >> "{dscl_log}"\n'
        f'[ "$3" = /Users/alice ] && echo "NFSHomeDirectory: {account}"; exit 0',
    )
    id_body = 'case "$*" in "-un 501") echo alice ;; -un) echo root ;; *) echo 0 ;; esac'
    source = _uninstall_removal_script("0").replace(
        "_remove_launchd_user_agent\n", f"id() {{ {id_body}; }}\n_remove_launchd_user_agent\n", 1
    )
    _fake_mac(tmp_path)
    _fake_bin(tmp_path, "stat", "echo 501")
    r = subprocess.run(
        ["sh", "-c", source], capture_output = True, text = True, env = _env(tmp_path), check = True
    )
    assert "/Users/alice" in dscl_log.read_text()
    assert "Removed launchd user agent" in r.stdout
    assert not real.exists()


def test_uninstall_unloads_the_agent_before_kill_sweep():
    source = UNINSTALL_SH.read_text(encoding = "utf-8")
    assert source.index("\n    _remove_launchd_user_agent\n") < source.index(
        "\n    _pkill_studio\n"
    )


def _real_gui_domain() -> bool:
    if sys.platform != "darwin" or not (
        os.environ.get("GITHUB_ACTIONS") == "true"
        or os.environ.get("UNSLOTH_TEST_REAL_LAUNCHD") == "1"
    ):
        return False
    r = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}"], capture_output = True)
    return r.returncode == 0


_FAKE_STUDIO = """
import http.server, signal, sys
port = int(sys.argv[sys.argv.index("-p") + 1])
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
print("serving", port, flush=True)
http.server.HTTPServer(("127.0.0.1", port), http.server.BaseHTTPRequestHandler).serve_forever()
"""


def _job_pid(target: str) -> int | None:
    r = subprocess.run(["launchctl", "print", target], capture_output = True, text = True)
    m = re.search(r"^\s*pid = (\d+)", r.stdout, flags = re.MULTILINE)
    return int(m.group(1)) if r.returncode == 0 and m else None


def _wait(pred, timeout: float):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = pred()
        if value:
            return value
        time.sleep(0.5)
    return pred()


def _port_open(port: int) -> bool:
    import socket
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


# Real launchd: disposable CI Macs only (or UNSLOTH_TEST_REAL_LAUNCHD=1), since it loads the real label.
@pytest.mark.skipif(
    not _real_gui_domain(), reason = "needs a macOS GUI launchd domain on a CI runner"
)
def test_real_launchd_restarts_crashes_only_and_uninstall_unloads(tmp_path):
    import signal
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = tmp_path / "fake_studio.py"
    server.write_text(_FAKE_STUDIO)
    exe = tmp_path / "venv" / "bin" / "unsloth"
    exe.parent.mkdir(parents = True)
    log = tmp_path / "fake_studio.log"
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{server}" "$@" >> "{log}" 2>&1\n')
    exe.chmod(0o755)
    env = {**os.environ, "HOME": str(tmp_path / "home")}
    target = f"gui/{os.getuid()}/{LABEL}"
    plist = None
    try:
        r = subprocess.run(
            [
                "bash",
                str(LAUNCHD_INSTALL_SH),
                "--unsloth-exe",
                str(exe),
                "--port",
                str(port),
                "--start",
            ],
            capture_output = True,
            text = True,
            env = env,
        )
        assert r.returncode == 0, r.stderr
        plist = Path(r.stdout.strip())
        first = _wait(lambda: _port_open(port) and _job_pid(target), 60)
        assert first, (log.read_text() if log.exists() else "no server log") + subprocess.run(
            ["launchctl", "print", target], capture_output = True, text = True
        ).stdout

        os.kill(first, signal.SIGKILL)
        second = _wait(
            lambda: (pid := _job_pid(target)) not in (None, first) and _port_open(port) and pid, 40
        )
        assert second, "KeepAlive did not restart the agent after a crash"

        os.kill(second, signal.SIGTERM)
        assert _wait(lambda: _job_pid(target) is None and not _port_open(port), 15)
        import time

        time.sleep(15)  # past ThrottleInterval: a clean exit must stay stopped
        assert _job_pid(target) is None and not _port_open(port)

        out = subprocess.run(
            ["sh", "-c", _uninstall_removal_script(str(os.getuid()))],
            capture_output = True,
            text = True,
            env = env,
            check = True,
        )
        assert "Removed launchd user agent" in out.stdout
        assert "could not unload" not in out.stderr
        assert not plist.exists()
        assert subprocess.run(["launchctl", "print", target], capture_output = True).returncode != 0
    finally:
        subprocess.run(["launchctl", "bootout", target], capture_output = True)
        if plist and plist.is_file() and plist.read_text().splitlines()[1:2] == [MARKER]:
            plist.unlink()
