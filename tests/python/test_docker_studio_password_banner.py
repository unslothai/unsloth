# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""`docker logs` must show how to sign in to Studio.

Studio writes its generated first-boot password to a file and prints only the
path, so a launcher banner saying "password below" left users with nothing to
type. The one-shot `studio-password` supervisord program prints the credential
once the file exists, or says why there is none. These run the shipped script
against a fake Studio home.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "docker" / "studio_password.sh"
LAUNCH = REPO_ROOT / "docker" / "studio_launch.sh"
SUPERVISORD = REPO_ROOT / "docker" / "supervisord.conf"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.studio"

behavioural = pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")


def _run(
    home: Path,
    *,
    env: dict | None = None,
    wait: str = "3",
) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_STUDIO")}
    e.update(UNSLOTH_STUDIO_HOME = str(home), UNSLOTH_STUDIO_PASSWORD_WAIT = wait)
    e.update(env or {})
    return subprocess.run(["bash", str(SCRIPT)], capture_output = True, text = True, env = e, timeout = 60)


@behavioural
def test_the_generated_password_is_printed_once_studio_writes_it(tmp_path: Path):
    auth = tmp_path / "auth"
    auth.mkdir()

    def _studio_writes_it_later():
        time.sleep(1.0)
        (auth / ".bootstrap_password").write_bytes(b"s3cret pass\n")

    threading.Thread(target = _studio_writes_it_later).start()
    res = _run(tmp_path)
    assert res.returncode == 0, res.stderr
    assert "username: unsloth" in res.stdout
    assert "password: s3cret pass" in res.stdout, res.stdout
    assert "60 min" in res.stdout, "the change-it-or-shut-down window is not explained"


@behavioural
def test_a_crlf_file_does_not_leak_the_cr_into_the_credential(tmp_path: Path):
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / ".bootstrap_password").write_bytes(b"abc\r\n")
    res = _run(tmp_path)
    assert "password: abc   (" in res.stdout, repr(res.stdout)


@behavioural
@pytest.mark.parametrize("value", ["0", "-5", " -1 "])
def test_a_disabled_timeout_drops_the_shutdown_note(tmp_path: Path, value: str):
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / ".bootstrap_password").write_bytes(b"abc\n")
    res = _run(tmp_path, env = {"UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT": value})
    assert res.stdout.rstrip().endswith("password: abc"), repr(res.stdout)


@behavioural
@pytest.mark.parametrize("value", ["abc", "", "  ", "1.5"])
def test_a_malformed_timeout_keeps_the_note_like_the_backend_does(tmp_path: Path, value: str):
    """bootstrap_timeout.py falls back to 3600 on a typo rather than disabling, so
    Studio still stops after an hour; the note must not vanish."""
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / ".bootstrap_password").write_bytes(b"abc\n")
    res = _run(tmp_path, env = {"UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT": value})
    assert "60 min" in res.stdout, repr(res.stdout)


@behavioural
def test_a_custom_timeout_is_reported_in_minutes(tmp_path: Path):
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / ".bootstrap_password").write_bytes(b"abc\n")
    res = _run(tmp_path, env = {"UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT": "+0900"})
    assert "15 min" in res.stdout, repr(res.stdout)


@behavioural
def test_a_supplied_password_is_never_echoed(tmp_path: Path):
    res = _run(tmp_path, env = {"UNSLOTH_STUDIO_PASSWORD": "hunter22hunter"})
    assert res.returncode == 0
    assert "hunter22hunter" not in res.stdout
    assert "from UNSLOTH_STUDIO_PASSWORD" in res.stdout


@behavioural
def test_an_already_changed_password_is_reported_not_waited_for_forever(tmp_path: Path):
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / "auth.db").write_bytes(b"")
    started = time.monotonic()
    res = _run(tmp_path, wait = "2")
    assert time.monotonic() - started < 30
    assert res.returncode == 0
    assert "already set on an earlier boot" in res.stdout, res.stdout
    assert "reset-password" in res.stdout


def _launcher_banner_block() -> str:
    """The banner decision, verbatim from studio_launch.sh."""
    source = LAUNCH.read_text(encoding = "utf-8")
    start = source.index("STUDIO_AUTH=")
    end = source.index('echo "Unsloth Studio  ->', start)
    return source[start:end]


def _banner(tmp_path: Path, *, env_password: str | None) -> tuple[str, str]:
    """(banner note, UNSLOTH_STUDIO_PASSWORD as supervisord would inherit it)."""
    script = (
        "set -euo pipefail\n"
        + _launcher_banner_block()
        + 'printf "%s\\n%s\\n" "$STUDIO_NOTE" "${UNSLOTH_STUDIO_PASSWORD:-<unset>}"\n'
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_STUDIO")}
    env["UNSLOTH_STUDIO_HOME"] = str(tmp_path)
    if env_password is not None:
        env["UNSLOTH_STUDIO_PASSWORD"] = env_password
    res = subprocess.run(
        ["bash", "-c", script], capture_output = True, text = True, env = env, timeout = 30
    )
    assert res.returncode == 0, res.stderr
    note, inherited = res.stdout.rstrip("\n").split("\n")
    return note, inherited


@behavioural
def test_the_initial_password_env_is_dropped_once_a_password_is_stored(tmp_path: Path):
    """`unsloth studio --password` exits 1 when a password is already set, so a
    container restarted with UNSLOTH_STUDIO_PASSWORD still in its environment
    left Studio in supervisord's FATAL state. After the first boot the launcher
    must not hand the variable to supervisord."""
    (tmp_path / "auth").mkdir()
    (tmp_path / "auth" / "auth.db").write_bytes(b"")
    note, inherited = _banner(tmp_path, env_password = "hunter22hunter")
    assert inherited == "<unset>"
    assert "set on an earlier boot" in note


@behavioural
def test_the_initial_password_env_is_kept_while_the_default_is_still_active(tmp_path: Path):
    """auth.db plus a bootstrap file means nobody changed the seeded password, and
    the CLI still accepts an initial password then."""
    (tmp_path / "auth").mkdir()
    (tmp_path / "auth" / "auth.db").write_bytes(b"")
    (tmp_path / "auth" / ".bootstrap_password").write_bytes(b"seeded\n")
    note, inherited = _banner(tmp_path, env_password = "hunter22hunter")
    assert inherited == "hunter22hunter"
    assert "from UNSLOTH_STUDIO_PASSWORD" in note


@behavioural
def test_a_fresh_home_passes_the_initial_password_through(tmp_path: Path):
    note, inherited = _banner(tmp_path, env_password = "hunter22hunter")
    assert inherited == "hunter22hunter"
    _, inherited = _banner(tmp_path, env_password = None)
    assert inherited == "<unset>"


def test_the_image_wires_the_printer_in():
    conf = SUPERVISORD.read_text(encoding = "utf-8")
    assert "[program:studio-password]" in conf
    block = conf.split("[program:studio-password]", 1)[1].split("[program:", 1)[0]
    assert "autorestart=false" in block and "stdout_logfile=/dev/stdout" in block
    assert "COPY studio_password.sh /usr/local/bin/unsloth-studio-password" in DOCKERFILE.read_text(
        encoding = "utf-8"
    )
    launch = LAUNCH.read_text(encoding = "utf-8")
    assert (
        "first-boot password below" not in launch
    ), "the banner promises what Studio no longer prints"
    assert "UNSLOTH_STUDIO_PASSWORD" in launch
