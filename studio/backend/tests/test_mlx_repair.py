# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MLX self-heal: on Apple Silicon with MLX missing, reinstall it by name on a
background thread (off the startup critical path). No-op elsewhere / when present
/ when disabled. Models on core.training.worker's runtime backend self-heal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.mlx_repair as mr  # noqa: E402


@pytest.fixture(autouse = True)
def _reset_attempt_guard(monkeypatch):
    monkeypatch.setattr(mr, "_attempted", False)
    monkeypatch.delenv(mr.DISABLE_ENV_VAR, raising = False)


def test_pip_cmd_targets_this_interpreter_with_mlx_packages(monkeypatch):
    monkeypatch.setattr(mr.shutil, "which", lambda _x: None)  # force pip path
    cmd = mr._pip_install_cmd("--upgrade", *mr.MLX_PACKAGES)
    assert cmd[0] == sys.executable
    assert cmd[1:4] == ["-m", "pip", "install"]
    for pkg in ("mlx", "mlx-lm", "mlx-vlm"):
        assert pkg in cmd


def test_uv_path_used_when_available(monkeypatch):
    monkeypatch.setattr(mr.shutil, "which", lambda x: "/usr/bin/uv" if x == "uv" else None)
    cmd = mr._pip_install_cmd(*mr.MLX_PACKAGES)
    assert cmd[:5] == ["uv", "pip", "install", "--python", sys.executable]


def test_no_op_off_apple_silicon(monkeypatch):
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: False)
    called = {"n": 0}
    monkeypatch.setattr(
        mr, "attempt_mlx_repair", lambda **_k: called.__setitem__("n", called["n"] + 1) or True
    )
    assert mr.start_mlx_autorepair_if_needed() is False
    assert called["n"] == 0


def test_no_op_when_mlx_present(monkeypatch):
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_available", lambda: True)
    started = mr.start_mlx_autorepair_if_needed()
    assert started is False


def test_disable_env_skips(monkeypatch):
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_available", lambda: False)
    monkeypatch.setenv(mr.DISABLE_ENV_VAR, "1")
    assert mr.start_mlx_autorepair_if_needed() is False


def test_apple_silicon_missing_mlx_starts_repair_and_redetects(monkeypatch):
    import threading

    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_available", lambda: False)

    repaired = {"called": False}

    def _fake_repair(**_kw):
        repaired["called"] = True
        return True

    redetected = {"called": False}

    # _run_repair_and_redetect imports utils.hardware.hardware lazily; stub repair
    # and capture that re-detection is invoked on success.
    monkeypatch.setattr(mr, "attempt_mlx_repair", _fake_repair)

    import utils.hardware.hardware as hw

    monkeypatch.setattr(hw, "detect_hardware", lambda: redetected.__setitem__("called", True))

    started = mr.start_mlx_autorepair_if_needed()
    assert started is True

    # Join the daemon thread deterministically.
    for thread in threading.enumerate():
        if thread.name == "mlx-autorepair":
            thread.join(timeout = 5)

    assert repaired["called"] is True
    assert redetected["called"] is True


def test_attempts_only_once_per_process(monkeypatch):
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_available", lambda: False)
    monkeypatch.setattr(mr, "attempt_mlx_repair", lambda **_k: False)

    first = mr.start_mlx_autorepair_if_needed()
    second = mr.start_mlx_autorepair_if_needed()
    assert first is True
    assert second is False  # guard prevents a second concurrent attempt
