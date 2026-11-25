# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`studio update` keeps an Apple Silicon install's torch through the core package update.

Only a new install, or install.sh under UNSLOTH_TORCH_UPGRADE=1 (Desktop's manual repair), picks
an Apple Silicon torch. The with-deps unsloth / unsloth-zoo re-resolve must not move it: not back
under a released ceiling (2.13 to 2.12) and not up past a floor (2.11 to 2.13). Linux and Windows
keep their torch >= 2.13 scope, and Intel macOS resolves as before.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_INSTALL_SCRIPT = Path(__file__).resolve().parents[2] / "install_python_stack.py"
_MLX_OVERRIDES = (
    _INSTALL_SCRIPT.parent
    / "backend"
    / "requirements"
    / "single-env"
    / "overrides-darwin-arm64.txt"
)
_MAC_TRIO = ["torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0"]


@pytest.fixture
def mod(monkeypatch):
    sys.modules.pop("install_python_stack", None)
    monkeypatch.syspath_prepend(str(_INSTALL_SCRIPT.parent))
    import install_python_stack

    return install_python_stack


def _host(
    monkeypatch,
    mod,
    platform,
    machine,
    resident,
    trio = _MAC_TRIO,
):
    monkeypatch.setattr(mod.sys, "platform", platform)
    monkeypatch.setattr(mod.platform, "machine", lambda: machine)
    monkeypatch.setattr(mod, "_is_win_arm64_interpreter", lambda: False)
    monkeypatch.setattr(mod, "_restrict_to_current_user", lambda path: True)
    monkeypatch.setattr(mod, "_resident_torch_release", lambda: resident)
    monkeypatch.setattr(mod, "_resident_torch_trio_pins", lambda: list(trio))


@pytest.mark.parametrize("resident", ["2.11.0", "2.12.1", "2.13.0", "2.14.0"])
def test_apple_silicon_core_update_keeps_every_resident_torch(monkeypatch, mod, resident):
    trio = [f"torch=={resident}", "torchvision==0.28.0", "torchaudio==2.11.0"]
    _host(monkeypatch, mod, "darwin", "arm64", resident, trio)
    # What the module sets on Apple Silicon at import: the bundled MLX overrides file.
    monkeypatch.setenv("UV_OVERRIDE", str(_MLX_OVERRIDES))
    with mod._FreezeNewTorchForCoreUpdate() as freeze:
        assert mod._TORCH_FREEZE_ACTIVE is True
        value = mod.os.environ["UV_OVERRIDE"]
        assert value != str(_MLX_OVERRIDES)
        lines = Path(value).read_text(encoding = "utf-8").splitlines()
        assert lines[:3] == trio
        # The MLX overrides still apply: uv takes every override for a package, so they are folded in.
        mlx = _MLX_OVERRIDES.read_text(encoding = "utf-8").splitlines()
        assert lines[3:] == mlx
        assert any(line.startswith("transformers>=") for line in lines)
    assert mod.os.environ["UV_OVERRIDE"] == str(_MLX_OVERRIDES)
    assert mod._TORCH_FREEZE_ACTIVE is False
    assert not freeze._path.exists()


def test_apple_silicon_without_a_preset_override_restores_it_unset(monkeypatch, mod):
    _host(monkeypatch, mod, "darwin", "arm64", "2.13.0")
    monkeypatch.delenv("UV_OVERRIDE", raising = False)
    with mod._FreezeNewTorchForCoreUpdate():
        assert Path(mod.os.environ["UV_OVERRIDE"]).read_text(encoding = "utf-8").split() == _MAC_TRIO
    assert "UV_OVERRIDE" not in mod.os.environ


@pytest.mark.parametrize(
    "platform, machine, resident",
    [
        # Intel macOS: no torch past 2.2.2 on PyPI, resolves exactly as before.
        ("darwin", "x86_64", "2.2.2"),
        ("darwin", "x86_64", "2.13.0"),
        # Nothing to keep: no torch, or a nightly / local build that is not a plain release.
        ("darwin", "arm64", None),
        # Linux and Windows keep their >= 2.13 scope, whatever the machine reports.
        ("linux", "aarch64", "2.12.1"),
        ("linux", "arm64", "2.11.0"),
        ("win32", "arm64", "2.12.1"),
    ],
)
def test_every_other_host_resolves_as_before(
    monkeypatch, tmp_path, mod, platform, machine, resident
):
    _host(monkeypatch, mod, platform, machine, resident)
    inherited = tmp_path / "overrides.txt"
    inherited.write_text("numpy<3\n", encoding = "utf-8")
    monkeypatch.setenv("UV_OVERRIDE", str(inherited))
    with mod._FreezeNewTorchForCoreUpdate():
        assert mod._TORCH_FREEZE_ACTIVE is False
        assert mod.os.environ["UV_OVERRIDE"] == str(inherited)
    assert mod.os.environ["UV_OVERRIDE"] == str(inherited)


def test_apple_silicon_uv_failure_under_the_freeze_never_falls_back_to_pip(monkeypatch, mod):
    _host(monkeypatch, mod, "darwin", "arm64", "2.13.0")
    monkeypatch.delenv("UV_OVERRIDE", raising = False)
    monkeypatch.setattr(mod, "USE_UV", True)

    class _Failed:
        returncode = 1
        stdout = b"uv: no solution"

    calls = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _Failed()

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    with mod._FreezeNewTorchForCoreUpdate():
        with pytest.raises(SystemExit):
            mod._pip_install_once("Updating core packages", "--no-cache-dir", "unsloth")
    # uv ran once; pip, which ignores UV_OVERRIDE, never did.
    assert calls and all(cmd[0] == "uv" for cmd in calls)
