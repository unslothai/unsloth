# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runtime sidecar repairs pass the pin in a requirements file: macOS XProtect kills a session
started from a pasted install whose child argv holds transformers==5.13.1 (#18699)."""

import os
import sys
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import utils.transformers_version as tv


def test_uv_and_pip_get_the_pin_from_a_file(tmp_path, monkeypatch):
    calls = []

    class _Result:
        returncode = 1
        stdout = ""

    def fake_run(cmd, **kwargs):
        if list(cmd[-2:]) == ["config", "list"]:
            return _Result()
        req = cmd[cmd.index("-r") + 1]
        calls.append((list(cmd), Path(req).read_text(encoding = "utf-8")))
        return _Result()

    monkeypatch.setattr(tv.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(tv.subprocess, "run", fake_run)
    monkeypatch.setenv("UV_OFFLINE", "0")
    pin = f"transformers=={tv.TRANSFORMERS_510_VERSION}"
    assert tv._install_to_dir(pin, str(tmp_path)) is False
    assert [cmd[0] for cmd, _ in calls] == ["uv", tv.sys.executable]
    for cmd, content in calls:
        assert not any(tv.TRANSFORMERS_510_VERSION in arg for arg in cmd)
        assert content == pin + "\n"
        assert not os.path.exists(cmd[cmd.index("-r") + 1])
