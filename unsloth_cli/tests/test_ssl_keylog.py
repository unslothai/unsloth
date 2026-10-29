# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An SSLKEYLOGFILE the process cannot write must not take down every HTTPS client."""

from __future__ import annotations

import os
import ssl
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli._ssl_keylog import drop_unwritable_ssl_keylog_file


def test_an_unwritable_key_log_is_dropped_and_tls_works(tmp_path, monkeypatch):
    # A directory stands in for the unopenable path; open(..., "a") fails on it everywhere.
    monkeypatch.setenv("SSLKEYLOGFILE", str(tmp_path))
    try:
        ssl.create_default_context()
        broken = False
    except OSError:
        broken = True
    assert broken, "the stand-in path must break ssl the way the real one does"
    assert drop_unwritable_ssl_keylog_file() is True
    assert "SSLKEYLOGFILE" not in os.environ
    ssl.create_default_context()


def test_a_writable_key_log_is_kept(tmp_path, monkeypatch):
    target = tmp_path / "keys.log"
    monkeypatch.setenv("SSLKEYLOGFILE", str(target))
    assert drop_unwritable_ssl_keylog_file() is False
    assert os.environ["SSLKEYLOGFILE"] == str(target)


def test_unset_is_left_alone(monkeypatch):
    monkeypatch.delenv("SSLKEYLOGFILE", raising = False)
    assert drop_unwritable_ssl_keylog_file() is False
    assert "SSLKEYLOGFILE" not in os.environ


def _run(code, env_path):
    env = dict(os.environ, SSLKEYLOGFILE = str(env_path), PYTHONPATH = str(_REPO_ROOT))
    return subprocess.run(
        [sys.executable, "-c", code], capture_output = True, text = True, env = env, timeout = 300
    )


def test_the_entry_point_clears_it_before_any_client_is_built(tmp_path):
    # The Desktop app starts the backend exactly this way (argv[0] = "unsloth", then import).
    code = (
        "import sys, os, ssl; sys.argv[0] = 'unsloth'; import unsloth_cli; "
        "ssl.create_default_context(); import httpx; httpx.AsyncClient(); "
        "print('KEYLOG', os.environ.get('SSLKEYLOGFILE'))"
    )
    result = _run(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "KEYLOG None" in result.stdout
    assert "ignoring SSLKEYLOGFILE" in result.stderr


def test_a_plain_import_leaves_the_host_environment_alone(tmp_path):
    result = _run(
        "import os, unsloth_cli; print('KEYLOG', os.environ.get('SSLKEYLOGFILE'))", tmp_path
    )
    assert result.returncode == 0, result.stderr
    assert f"KEYLOG {tmp_path}" in result.stdout
