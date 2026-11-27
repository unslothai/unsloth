# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The owner's UNSLOTH_STUDIO_PASSWORD is staged in /run/unsloth for unsloth-studio-run.
A managed account's tool process runs as uid 0 with no capabilities and may read /run,
so the plaintext has to go once Studio has stored the password."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN = REPO_ROOT / "docker" / "studio_run.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("python3") is None,
    reason = "needs bash and python3",
)

_STORE = r"""
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute(
    "CREATE TABLE IF NOT EXISTS auth_user (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, "
    "password_salt TEXT NOT NULL, password_hash TEXT NOT NULL, jwt_secret TEXT NOT NULL, "
    "must_change_password INTEGER NOT NULL DEFAULT 0)"
)
conn.execute(
    "INSERT OR REPLACE INTO auth_user (username, password_salt, password_hash, jwt_secret, "
    "must_change_password) VALUES ('unsloth', 's', 'h', 'j', 0)"
)
conn.commit()
"""


def _store_admin(home: Path) -> None:
    (home / "auth").mkdir(parents = True, exist_ok = True)
    subprocess.run(["python3", "-c", _STORE, str(home / "auth" / "auth.db")], check = True)


def _run(
    home: Path,
    cli_body: str,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    bin_dir = home / "bin"
    bin_dir.mkdir(exist_ok = True)
    cli = bin_dir / "unsloth"
    cli.write_text("#!/usr/bin/env bash\n" + cli_body, encoding = "utf-8")
    cli.chmod(0o755)
    e = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_STUDIO")}
    e["UNSLOTH_STUDIO_HOME"] = str(home)
    e["UNSLOTH_STUDIO_INITIAL_PASSWORD_FILE"] = str(home / "initial")
    e.update(env or {})
    return subprocess.run(["bash", str(RUN)], capture_output = True, text = True, env = e, timeout = 60)


def _gone_within(path: Path, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not path.exists():
            return True
        time.sleep(0.2)
    return not path.exists()


def test_the_staged_password_is_removed_once_studio_stores_it(tmp_path: Path):
    initial = tmp_path / "initial"
    initial.write_text("hunter22hunter", encoding = "utf-8")
    store = f'mkdir -p "{tmp_path}/auth"\npython3 -c {_quote(_STORE)} "{tmp_path}/auth/auth.db"\n'
    res = _run(tmp_path, 'printf "%s\\n" "${UNSLOTH_STUDIO_PASSWORD:-<unset>}"\n' + store)
    assert res.stdout.strip() == "hunter22hunter", res.stdout + res.stderr
    assert _gone_within(initial, 15), "the owner's plaintext password stayed in the staging file"


def test_a_respawn_after_the_password_is_stored_removes_a_leftover_file(tmp_path: Path):
    initial = tmp_path / "initial"
    initial.write_text("hunter22hunter", encoding = "utf-8")
    _store_admin(tmp_path)
    res = _run(tmp_path, 'printf "%s\\n" "${UNSLOTH_STUDIO_PASSWORD:-<unset>}"\n')
    assert res.stdout.strip() == "<unset>", res.stdout + res.stderr
    assert not initial.exists()


def test_a_launch_that_never_stores_keeps_the_file_for_the_respawn(tmp_path: Path):
    """An invalid password or a crash before the store must still reach the next spawn."""
    initial = tmp_path / "initial"
    initial.write_text("hunter22hunter", encoding = "utf-8")
    res = _run(
        tmp_path,
        'printf "%s\\n" "${UNSLOTH_STUDIO_PASSWORD:-<unset>}"\nexit 1\n',
        env = {"UNSLOTH_STUDIO_PASSWORD_WAIT": "2"},
    )
    assert res.returncode == 1
    assert res.stdout.strip() == "hunter22hunter"
    time.sleep(3)
    assert initial.read_text(encoding = "utf-8") == "hunter22hunter"
    res = _run(
        tmp_path,
        'printf "%s\\n" "${UNSLOTH_STUDIO_PASSWORD:-<unset>}"\n',
        env = {"UNSLOTH_STUDIO_PASSWORD_WAIT": "1"},
    )
    assert res.stdout.strip() == "hunter22hunter"


def _quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"
