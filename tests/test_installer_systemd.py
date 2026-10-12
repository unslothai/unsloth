# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Optional systemd user service for Studio (#9258)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = REPO_ROOT / "install.sh"
SYSTEMD_INSTALL_SH = REPO_ROOT / "studio" / "systemd" / "install_user_service.sh"
UNINSTALL_SH = REPO_ROOT / "scripts" / "uninstall.sh"

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("bash") is None,
    reason = "systemd user units are Linux only",
)


def _fake_bin(tmp_path: Path, name: str, body: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok = True)
    exe = bin_dir / name
    exe.write_text(f"#!/bin/sh\n{body}\n")
    exe.chmod(0o755)
    return exe


def _write_unit(
    tmp_path: Path,
    *args: str,
    env: dict | None = None,
) -> str:
    exe = _fake_bin(tmp_path, "unsloth", "exit 0")
    full_env = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_SYSTEMD_")}
    full_env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    full_env.update(env or {})
    out = subprocess.run(
        ["bash", str(SYSTEMD_INSTALL_SH), "--unsloth-exe", str(exe), *args],
        check = True,
        capture_output = True,
        text = True,
        env = full_env,
    ).stdout.strip()
    assert out == str(tmp_path / "config" / "systemd" / "user" / "unsloth-studio.service")
    return Path(out).read_text()


def test_unit_defaults_to_loopback(tmp_path):
    unit = _write_unit(tmp_path, "--port", "9090")
    assert 'studio -H "127.0.0.1" -p 9090' in unit
    assert "0.0.0.0" not in unit
    assert "unsloth-studio-managed-systemd" in unit
    assert "Restart=on-failure" in unit
    assert "Environment=" not in unit
    assert "@@" not in unit


@pytest.mark.parametrize(
    ("args", "env"),
    [(("--host", "0.0.0.0"), {}), ((), {"UNSLOTH_SYSTEMD_HOST": "0.0.0.0"})],
)
def test_unit_lan_bind_is_opt_in(tmp_path, args, env):
    assert 'studio -H "0.0.0.0" -p 8888' in _write_unit(tmp_path, *args, env = env)


def test_unit_has_no_execstop_that_stops_other_studios(tmp_path):
    # `unsloth studio stop` stops every server on the home, including ones the user started by hand.
    assert "ExecStop" not in _write_unit(tmp_path)


def test_unit_escapes_odd_paths(tmp_path):
    home = tmp_path / 'my "studio" 100% $HOME'
    home.mkdir()
    unit = _write_unit(tmp_path, "--studio-home", str(home))
    env_line = next(l for l in unit.splitlines() if l.startswith("Environment="))
    escaped = str(home.resolve()).replace('"', '\\"').replace("%", "%%")
    assert env_line == f'Environment="UNSLOTH_STUDIO_HOME={escaped}"'


def test_enable_start_drives_systemctl(tmp_path):
    log = tmp_path / "systemctl.log"
    _fake_bin(tmp_path, "systemctl", f'echo "$*" >> "{log}"')
    _write_unit(
        tmp_path,
        "--start",
        env = {"PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}"},
    )
    assert log.read_text().splitlines() == [
        "--user show-environment",
        "--user show-environment",
        "--user daemon-reload",
        "--user enable unsloth-studio.service",
        "--user restart unsloth-studio.service",
    ]


def test_enable_writes_where_the_user_manager_looks(tmp_path):
    # HOME redirected for the installer: the manager still reads its own HOME's config.
    mgr_home = tmp_path / "passwd_home"
    _fake_bin(
        tmp_path,
        "systemctl",
        f'[ "$2" = show-environment ] && printf "HOME={mgr_home}\\nLANG=C\\n"; exit 0',
    )
    exe = _fake_bin(tmp_path, "unsloth", "exit 0")
    out = subprocess.run(
        ["bash", str(SYSTEMD_INSTALL_SH), "--unsloth-exe", str(exe), "--enable"],
        check = True,
        capture_output = True,
        text = True,
        env = {
            **os.environ,
            "HOME": str(tmp_path / "redirected"),
            "XDG_CONFIG_HOME": str(tmp_path / "redirected" / ".config"),
            "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
        },
    ).stdout.strip()
    assert out == str(mgr_home / ".config" / "systemd" / "user" / "unsloth-studio.service")


_GATE = re.compile(
    r'^if \[ "\$_INSTALL_SYSTEMD" = true \]; then\n.*?^fi\n', re.DOTALL | re.MULTILINE
)


def _harness() -> str:
    source = INSTALL_SH.read_text(encoding = "utf-8")
    parts = ['step() { echo "STEP $*"; }\nsubstep() { :; }\n']
    for name in ("_resolve_systemd_install_script", "_install_systemd_user_service"):
        m = re.search(rf"^{name}\(\) \{{.*?^\}}", source, flags = re.DOTALL | re.MULTILINE)
        assert m is not None, name
        parts.append(m.group(0))
    gate = _GATE.search(source)
    assert gate is not None
    parts.append(gate.group(0))
    parts.append('echo "STARTED=$_SYSTEMD_STARTED SKIP_AUTOSTART=$_SKIP_AUTOSTART"\n')
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
        "OS": "linux",
        "_INSTALL_SYSTEMD": "false",
        "_SYSTEMD_STARTED": "false",
        "_SKIP_AUTOSTART": "false",
        "_REPO_IS_CHECKOUT": "0",
        "_REPO_ROOT": str(tmp_path),
        "VENV_DIR": str(venv),
        "STUDIO_HOME": str(tmp_path / "home"),
        "_STUDIO_HOME_REDIRECT": "default",
        **vars,
    }
    assigns = "".join(f"{k}='{v}'\n" for k, v in preset.items())
    env = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_SYSTEMD_")}
    env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{env['PATH']}"
    out = subprocess.run(
        ["sh", "-ec", assigns + _harness()],
        check = True,
        capture_output = True,
        text = True,
        env = env,
        stdin = subprocess.DEVNULL,
        cwd = cwd,
    ).stdout
    return out + (log.read_text() if log.exists() else "")


def test_flag_off_changes_nothing_even_with_a_user_bus(tmp_path):
    _fake_bin(tmp_path, "systemctl", "exit 0")
    out = _run_installer_tail(tmp_path)
    assert out == "STARTED=false SKIP_AUTOSTART=false\n"


def test_opt_in_installs_with_studio_home_for_env_redirect(tmp_path):
    out = _run_installer_tail(tmp_path, _INSTALL_SYSTEMD = "true", _STUDIO_HOME_REDIRECT = "env")
    assert "STARTED=true SKIP_AUTOSTART=true" in out
    assert f"--studio-home {tmp_path / 'home'}" in out
    assert "--host 127.0.0.1 --port 8888" in out


def test_opt_in_runs_on_wsl(tmp_path):
    # WSL2 distros can run systemd; the helper itself refuses when the user bus is missing.
    out = _run_installer_tail(tmp_path, _INSTALL_SYSTEMD = "true", OS = "wsl")
    assert "STARTED=true" in out
    assert "--unsloth-exe" in out


def test_opt_in_off_linux_warns_and_keeps_launch_prompt(tmp_path):
    out = _run_installer_tail(tmp_path, _INSTALL_SYSTEMD = "true", OS = "macos")
    assert "Linux only" in out
    assert "STARTED=false SKIP_AUTOSTART=false" in out
    assert "--unsloth-exe" not in out


def test_piped_install_never_runs_a_helper_planted_in_cwd(tmp_path):
    planted = tmp_path / "studio" / "systemd" / "install_user_service.sh"
    planted.parent.mkdir(parents = True)
    planted.write_text(f'touch "{tmp_path / "planted_ran"}"\n')
    assert "STARTED=true" in _run_installer_tail(tmp_path, _INSTALL_SYSTEMD = "true")
    assert not (tmp_path / "planted_ran").exists()


def test_package_lookup_ignores_a_studio_package_planted_in_cwd(tmp_path):
    # Real interpreter: `python -c` puts the cwd first on sys.path; the lookup must not.
    cwd = tmp_path / "cwd"
    (cwd / "studio" / "systemd").mkdir(parents = True)
    (cwd / "studio" / "__init__.py").write_text(f"open({str(tmp_path / 'imported')!r}, 'w')\n")
    (cwd / "studio" / "systemd" / "install_user_service.sh").write_text(
        f'touch "{tmp_path / "planted_ran"}"\n'
    )
    _run_installer_tail(tmp_path, cwd = cwd, real_python = True, _INSTALL_SYSTEMD = "true")
    assert not (tmp_path / "imported").exists()
    assert not (tmp_path / "planted_ran").exists()


@pytest.mark.parametrize("bad", ["home\nExecStartPre=/bin/true", 'a"b'])
def test_helper_rejects_values_that_break_the_unit(tmp_path, bad):
    exe_dir = tmp_path / bad if '"' in bad else tmp_path
    exe_dir.mkdir(exist_ok = True)
    exe = exe_dir / "unsloth"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    args = ["--unsloth-exe", str(exe)]
    if "\n" in bad:
        args += ["--studio-home", str(tmp_path / bad)]
    r = subprocess.run(
        ["bash", str(SYSTEMD_INSTALL_SH), *args],
        capture_output = True,
        text = True,
        env = {**os.environ, "XDG_CONFIG_HOME": str(tmp_path / "config")},
    )
    assert r.returncode == 2, r.stderr
    assert not (tmp_path / "config" / "systemd" / "user" / "unsloth-studio.service").exists()


def test_helper_never_replaces_a_unit_the_user_wrote(tmp_path):
    unit = tmp_path / "config" / "systemd" / "user" / "unsloth-studio.service"
    unit.parent.mkdir(parents = True)
    unit.write_text("[Service]\nExecStart=/opt/mine\n")
    with pytest.raises(subprocess.CalledProcessError):
        _write_unit(tmp_path)
    assert unit.read_text() == "[Service]\nExecStart=/opt/mine\n"
    assert list(unit.parent.iterdir()) == [unit]


def test_helper_ignores_relative_xdg_config_home(tmp_path):
    exe = _fake_bin(tmp_path, "unsloth", "exit 0")
    out = subprocess.run(
        ["bash", str(SYSTEMD_INSTALL_SH), "--unsloth-exe", str(exe)],
        check = True,
        capture_output = True,
        text = True,
        cwd = tmp_path,
        env = {**os.environ, "XDG_CONFIG_HOME": "rel", "HOME": str(tmp_path / "home")},
    ).stdout.strip()
    assert out == str(tmp_path / "home" / ".config" / "systemd" / "user" / "unsloth-studio.service")


def _run_uninstall_removal(tmp_path: Path, unit_text: str | None, systemctl: str) -> str:
    source = UNINSTALL_SH.read_text(encoding = "utf-8")
    parts = []
    for name in ("_set_marker", "_remove_path", "_xdg_dir"):
        m = re.search(rf"^{name}\(\) \{{.*?^\}}", source, flags = re.DOTALL | re.MULTILINE)
        parts.append(m.group(0))
    m = re.search(
        r"^    _remove_systemd_user_service\(\) \{.*?^    \}",
        source,
        flags = re.DOTALL | re.MULTILINE,
    )
    parts.append(m.group(0) + "\n_remove_systemd_user_service\n")
    d = tmp_path / "home" / ".config" / "systemd" / "user"
    (d / "default.target.wants").mkdir(parents = True)
    if unit_text is not None:
        (d / "unsloth-studio.service").write_text(unit_text)
        (d / "default.target.wants" / "unsloth-studio.service").symlink_to(
            d / "unsloth-studio.service"
        )
    log = tmp_path / "systemctl.log"
    _fake_bin(tmp_path, "systemctl", f'echo "$*" >> "{log}"\n{systemctl}')
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
    }
    env.pop("XDG_CONFIG_HOME", None)
    r = subprocess.run(
        ["sh", "-c", "\n".join(parts)], capture_output = True, text = True, env = env, check = True
    )
    return r.stdout + r.stderr + (log.read_text() if log.exists() else "")


def test_uninstall_leaves_units_it_did_not_write(tmp_path):
    text = "[Service]\n# mentions unsloth-studio-managed-systemd in passing\nExecStart=/opt/mine\n"
    assert _run_uninstall_removal(tmp_path, text, "exit 0") == ""
    assert (tmp_path / "home" / ".config" / "systemd" / "user" / "unsloth-studio.service").exists()


def test_uninstall_without_user_bus_still_drops_the_enable_link(tmp_path):
    out = _run_uninstall_removal(
        tmp_path, "# unsloth-studio-managed-systemd\n[Service]\n", "exit 1"
    )
    d = tmp_path / "home" / ".config" / "systemd" / "user"
    assert not (d / "unsloth-studio.service").exists()
    assert not os.path.lexists(d / "default.target.wants" / "unsloth-studio.service")
    assert "could not reach the systemd user manager" in out
    assert "disable" not in out


def test_installer_passes_studio_home_when_home_is_redirected(tmp_path):
    out = _run_installer_tail(tmp_path, _INSTALL_SYSTEMD = "true", _STUDIO_HOME_REDIRECT = "home")
    assert f"--studio-home {tmp_path / 'home'}" in out
    out = _run_installer_tail(tmp_path, _INSTALL_SYSTEMD = "true")
    assert "--studio-home" not in out


def test_uninstall_finds_the_unit_through_the_manager(tmp_path):
    other = tmp_path / "passwd_home" / ".config" / "systemd" / "user"
    other.mkdir(parents = True)
    (other / "unsloth-studio.service").write_text("# unsloth-studio-managed-systemd\n")
    out = _run_uninstall_removal(
        tmp_path,
        None,
        f'[ "$2" = show ] && echo "{other}/unsloth-studio.service"; exit 0',
    )
    assert "--user disable --now unsloth-studio.service" in out
    assert not (other / "unsloth-studio.service").exists()


def test_uninstall_finds_the_unit_under_the_passwd_home_without_a_bus(tmp_path):
    other = tmp_path / "passwd_home" / ".config" / "systemd" / "user"
    other.mkdir(parents = True)
    (other / "unsloth-studio.service").write_text("# unsloth-studio-managed-systemd\n")
    _fake_bin(tmp_path, "getent", f'echo "me:x:1:1::{tmp_path / "passwd_home"}:/bin/sh"')
    out = _run_uninstall_removal(tmp_path, None, "exit 1")
    assert not (other / "unsloth-studio.service").exists()
    assert "could not reach the systemd user manager" in out


def test_uninstall_with_user_bus_disables_then_removes(tmp_path):
    out = _run_uninstall_removal(
        tmp_path, "# unsloth-studio-managed-systemd\n[Service]\n", "exit 0"
    )
    assert "--user disable --now unsloth-studio.service" in out
    assert "could not reach" not in out
    assert not (
        tmp_path / "home" / ".config" / "systemd" / "user" / "unsloth-studio.service"
    ).exists()


def test_installer_asks_no_systemd_question():
    source = INSTALL_SH.read_text(encoding = "utf-8")
    assert not re.search(r"printf[^\n]*systemd[^\n]*\?", source)
    assert _GATE.search(source).start() < source.index("Start Unsloth Studio now? [Y/n]")


def test_uninstall_disables_service_before_kill_sweep():
    source = UNINSTALL_SH.read_text(encoding = "utf-8")
    assert source.index("\n    _remove_systemd_user_service\n") < source.index(
        "\n    _pkill_studio\n"
    )
    assert "systemctl --user disable --now unsloth-studio.service" in source
