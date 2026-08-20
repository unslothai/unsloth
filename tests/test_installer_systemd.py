"""Regression tests for optional systemd user service install (#9258)."""

from __future__ import annotations

import os
import re
import subprocess

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = REPO_ROOT / "install.sh"
SYSTEMD_INSTALL_SH = REPO_ROOT / "studio" / "systemd" / "install_user_service.sh"
UNINSTALL_SH = REPO_ROOT / "scripts" / "uninstall.sh"

TRUTHY_VALUES = ("1", "true", "TRUE", "yes", "YES", "on", "ON")
FALSEY_VALUES = ("", "0", "false", "no", "off", "anything-else")


def test_systemd_install_script_exists():
    assert SYSTEMD_INSTALL_SH.is_file()
    assert (REPO_ROOT / "studio" / "systemd" / "unsloth-studio.service.in").is_file()


def test_install_sh_documents_systemd_env_vars():
    source = INSTALL_SH.read_text(encoding = "utf-8")
    assert "UNSLOTH_SKIP_SYSTEMD" in source
    assert "UNSLOTH_INSTALL_SYSTEMD" in source
    assert "UNSLOTH_SYSTEMD_HOST" in source
    assert "UNSLOTH_SYSTEMD_PORT" in source


def test_systemd_defaults_to_loopback_not_all_interfaces():
    """Opting into systemd must not quietly expose Studio on every interface (#9308)."""
    install_source = INSTALL_SH.read_text(encoding = "utf-8")
    assert "UNSLOTH_SYSTEMD_HOST:-127.0.0.1" in install_source
    assert "UNSLOTH_SYSTEMD_HOST:-0.0.0.0" not in install_source

    script = SYSTEMD_INSTALL_SH.read_text(encoding = "utf-8")
    assert "UNSLOTH_SYSTEMD_HOST:-127.0.0.1" in script
    assert '_HOST="0.0.0.0"' not in script


@pytest.mark.parametrize(
    ("value", "expected"),
    [(value, "true") for value in TRUTHY_VALUES] + [(value, "false") for value in FALSEY_VALUES],
)
@pytest.mark.skipif(not Path("/bin/sh").exists(), reason = "POSIX shell is unavailable")
def test_posix_skip_systemd_value_parsing(value: str, expected: str):
    source = INSTALL_SH.read_text(encoding = "utf-8")
    parser = re.search(
        r'case "\$\{UNSLOTH_SKIP_SYSTEMD:-\}" in.*?esac',
        source,
        flags = re.DOTALL,
    )
    assert parser is not None
    env = os.environ.copy()
    env["UNSLOTH_SKIP_SYSTEMD"] = value
    result = subprocess.run(
        [
            "sh",
            "-c",
            f"_SKIP_SYSTEMD=false\n{parser.group(0)}\nprintf '%s' \"$_SKIP_SYSTEMD\"",
        ],
        check = True,
        capture_output = True,
        text = True,
        env = env,
    )
    assert result.stdout == expected


def test_install_sh_offers_systemd_before_autostart_prompt():
    source = INSTALL_SH.read_text(encoding = "utf-8")
    assert "_offer_systemd_user_service" in source
    assert source.index("_offer_systemd_user_service") < source.index(
        "Start Unsloth Studio now? [Y/n]"
    )
    assert (
        "Install a systemd user service for auto-start on boot and crash recovery? [y/N]" in source
    )


def test_uninstall_sh_removes_managed_systemd_unit():
    source = UNINSTALL_SH.read_text(encoding = "utf-8")
    assert "_remove_systemd_user_service" in source
    assert "unsloth-studio-managed-systemd" in source
    assert "systemctl --user disable --now unsloth-studio.service" in source
