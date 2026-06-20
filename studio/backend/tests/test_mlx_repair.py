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
    assert set(mr.MLX_PACKAGES) <= set(cmd)
    # Minimum versions are pinned so the resolver cannot backtrack to an old
    # mlx-vlm that imports but breaks VLM Train/Export.
    assert "mlx-vlm>=0.4.4" in cmd


def test_uv_path_used_when_available(monkeypatch):
    monkeypatch.setattr(mr.shutil, "which", lambda x: "/usr/bin/uv" if x == "uv" else None)
    cmd = mr._pip_install_cmd(*mr.MLX_PACKAGES)
    assert cmd[:5] == ["uv", "pip", "install", "--python", sys.executable]


def test_constraint_pins_installed_transformers(monkeypatch):
    transformers = pytest.importorskip("transformers")
    args, path = mr._transformers_constraint_args()
    try:
        assert args[:1] == ["--constraint"]
        assert args[1] == path
        assert Path(path).read_text().strip() == f"transformers=={transformers.__version__}"
    finally:
        if path:
            Path(path).unlink(missing_ok = True)


def test_repair_install_pins_transformers_and_cleans_up(monkeypatch):
    pytest.importorskip("transformers")
    captured = {}
    created_paths = []
    real_args = mr._transformers_constraint_args

    def _spy_args():
        args, path = real_args()
        if path:
            created_paths.append(path)
        return args, path

    monkeypatch.setattr(mr, "_transformers_constraint_args", _spy_args)

    class _Result:
        returncode = 0
        stdout = ""

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return _Result()

    monkeypatch.setattr(mr.subprocess, "run", _fake_run)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: True)

    assert mr.attempt_mlx_repair() is True
    cmd = captured["cmd"]
    # transformers is pinned via a constraint file so the mlx install cannot
    # upgrade it underneath Studio, and the temp constraint file is cleaned up.
    assert "--constraint" in cmd
    assert "--upgrade" in cmd
    for pkg in mr.MLX_PACKAGES:
        assert pkg in cmd
    assert created_paths and not Path(created_paths[0]).exists()
    # The install mirrors the main installer by relaxing the transformers pin via
    # UV_OVERRIDE so a current mlx-vlm can coexist with transformers==4.57.6.
    env = captured["env"]
    assert env is not None
    assert env.get("UV_OVERRIDE", "").endswith("overrides-darwin-arm64.txt")


def test_repair_rejects_inadequate_stack(monkeypatch):
    # A successful pip run that still leaves an old/missing mlx-vlm must NOT clear
    # chat-only: attempt_mlx_repair returns False so Train/Export stay disabled.
    class _Result:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(mr.subprocess, "run", lambda *a, **k: _Result())
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: False)
    assert mr.attempt_mlx_repair() is False


def test_stack_unavailable_without_mlx(monkeypatch):
    monkeypatch.setattr(mr, "mlx_available", lambda: False)
    assert mr.mlx_stack_available() is False


def test_no_op_off_apple_silicon(monkeypatch):
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: False)
    called = {"n": 0}
    monkeypatch.setattr(
        mr, "attempt_mlx_repair", lambda **_k: called.__setitem__("n", called["n"] + 1) or True
    )
    assert mr.start_mlx_autorepair_if_needed() is False
    assert called["n"] == 0


def test_no_op_when_mlx_stack_present(monkeypatch):
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: True)
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
