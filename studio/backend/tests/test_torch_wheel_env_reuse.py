# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An SSM load reads the torch build from the torch it already imported instead of a 1-3 s child probe."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from utils import ssm_runtime, wheel_utils


def test_loaded_torch_matches_the_child_probe():
    pytest.importorskip("torch")
    if wheel_utils.wheel_platform_tag() is None:
        pytest.skip("no wheel platform tag on this host")
    child = wheel_utils.probe_torch_wheel_env(timeout = 120, include_windows = True)
    reused = wheel_utils.probe_torch_wheel_env(
        timeout = 120, include_windows = True, reuse_loaded_torch = True
    )
    assert child is not None
    assert reused == child
    assert list(reused) == list(child)


def test_no_loaded_torch_falls_back_to_the_child(monkeypatch):
    monkeypatch.setattr(wheel_utils, "wheel_platform_tag", lambda: "linux_x86_64")
    monkeypatch.delitem(sys.modules, "torch", raising = False)
    calls = []

    def _run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode = 0, stdout = json.dumps({"cuda_version": "12.8"}))

    monkeypatch.setattr(subprocess, "run", _run)
    env = wheel_utils.probe_torch_wheel_env(timeout = 5, reuse_loaded_torch = True)
    assert env == {"cuda_version": "12.8", "platform_tag": "linux_x86_64"}
    assert len(calls) == 1


def test_windows_gate_still_applies_before_reuse(monkeypatch):
    monkeypatch.setattr(wheel_utils, "wheel_platform_tag", lambda: "win_amd64")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    assert wheel_utils.probe_torch_wheel_env(reuse_loaded_torch = True) is None


def test_ssm_torch_env_asks_for_the_loaded_torch(monkeypatch):
    seen = []

    def _probe(**kwargs):
        seen.append(kwargs)
        return {"cuda_version": "13.0", "cuda_major": "13"}

    monkeypatch.setattr(ssm_runtime, "_TORCH_ENV", {})
    monkeypatch.setattr(ssm_runtime, "probe_torch_wheel_env", _probe)
    assert ssm_runtime._torch_env() == {"cuda_version": "13.0", "cuda_major": "13"}
    assert ssm_runtime._torch_env() == {"cuda_version": "13.0", "cuda_major": "13"}
    assert seen == [{"timeout": 30, "include_windows": True, "reuse_loaded_torch": True}]
