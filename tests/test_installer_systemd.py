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
        "--user daemon-reload",
        "--user enable unsloth-studio.service",
        "--user restart unsloth-studio.service",
    ]


def _offer_harness() -> str:
    source = INSTALL_SH.read_text(encoding = "utf-8")
    funcs = []
    for name in (
        "_resolve_systemd_install_script",
        "_systemd_user_session_available",
        "_offer_systemd_user_service",
    ):
        m = re.search(rf"^{name}\(\) \{{.*?^\}}", source, flags = re.DOTALL | re.MULTILINE)
        assert m is not None, name
        funcs.append(m.group(0))
    stubs = 'step() { echo "STEP $*"; }\nsubstep() { :; }\n_can_read_tty() { return 0; }\n'
    return (
        stubs
        + "\n".join(funcs)
        + '\n_offer_systemd_user_service\necho "STARTED=$_SYSTEMD_STARTED"\n'
    )


def _run_offer(tmp_path: Path, **vars: str) -> str:
    script = tmp_path / "helper.sh"
    log = tmp_path / "helper.log"
    script.write_text(f'echo "$*" > "{log}"\n')
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents = True, exist_ok = True)
    py = venv / "bin" / "python"
    py.write_text(f'#!/bin/sh\necho "{script}"\n')
    py.chmod(0o755)
    preset = {
        "OS": "linux",
        "_SKIP_SYSTEMD": "false",
        "_INSTALL_SYSTEMD": "false",
        "_SYSTEMD_STARTED": "false",
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
        ["sh", "-c", assigns + _offer_harness()],
        check = True,
        capture_output = True,
        text = True,
        env = env,
        stdin = subprocess.DEVNULL,
    ).stdout
    return out + (log.read_text() if log.exists() else "")


def test_no_user_bus_asks_nothing_and_keeps_launch_prompt(tmp_path):
    _fake_bin(tmp_path, "systemctl", "exit 1")
    out = _run_offer(tmp_path)
    assert "STARTED=false" in out
    assert "--unsloth-exe" not in out


def test_opt_in_installs_with_studio_home_for_env_redirect(tmp_path):
    out = _run_offer(tmp_path, _INSTALL_SYSTEMD = "true", _STUDIO_HOME_REDIRECT = "env")
    assert "STARTED=true" in out
    assert f"--studio-home {tmp_path / 'home'}" in out
    assert "--host 127.0.0.1 --port 8888" in out


def test_piped_install_never_runs_a_helper_planted_in_cwd(tmp_path):
    planted = tmp_path / "studio" / "systemd" / "install_user_service.sh"
    planted.parent.mkdir(parents = True)
    planted.write_text(f'touch "{tmp_path / "planted_ran"}"\n')
    assert "STARTED=true" in _run_offer(tmp_path, _INSTALL_SYSTEMD = "true")
    assert not (tmp_path / "planted_ran").exists()


def test_launch_prompt_skipped_only_after_service_started():
    source = INSTALL_SH.read_text(encoding = "utf-8")
    offer = source.index("\n_offer_systemd_user_service\n")
    skip = source.index('[ "$_SYSTEMD_STARTED" = true ] && _SKIP_AUTOSTART=true')
    assert offer < skip < source.index("Start Unsloth Studio now? [Y/n]")


def test_uninstall_disables_service_before_kill_sweep():
    source = UNINSTALL_SH.read_text(encoding = "utf-8")
    assert source.index("\n    _remove_systemd_user_service\n") < source.index(
        "\n    _pkill_studio\n"
    )
    assert "systemctl --user disable --now unsloth-studio.service" in source
