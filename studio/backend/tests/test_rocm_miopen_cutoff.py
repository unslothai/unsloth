# SPDX-License-Identifier: AGPL-3.0-only
import os
import sys
from types import FunctionType, SimpleNamespace
from unittest.mock import Mock

import pytest

from utils.hardware import hardware as hw


@pytest.fixture(autouse = True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())


@pytest.mark.parametrize(
    "arches, expected",
    [
        (["gfx1151"], "1"),
        (["gfx1151:xnack-", "gfx1151"], "1"),
        (["gfx1100"], "1"),
        (["gfx1102"], "1"),
        (["gfx1150"], "1"),
        (["gfx1200"], "1"),
        (["gfx1201"], "1"),
        (["gfx1151", "gfx1100", "gfx1201"], "1"),
        (["gfx1030"], None),
        (["gfx90a"], None),
        (["gfx942:sramecc+:xnack-"], None),
        (["gfx950"], None),
        (["gfx1250"], None),
        (["gfx1201", "gfx942"], None),
        (["gfx1151", ""], None),
        ([], None),
    ],
)
def test_visible_device_scope(monkeypatch, arches, expected):
    monkeypatch.delenv("MIOPEN_SEARCH_CUTOFF", raising = False)
    cuda = SimpleNamespace(
        device_count = lambda: len(arches),
        get_device_properties = lambda i: SimpleNamespace(gcnArchName = arches[i]),
    )
    hw._configure_rocm_miopen(SimpleNamespace(cuda = cuda))
    assert os.environ.get("MIOPEN_SEARCH_CUTOFF") == expected


@pytest.mark.parametrize("value", ["0", "1", "", "custom"])
def test_explicit_override_is_preserved_without_probing(monkeypatch, value):
    monkeypatch.setenv("MIOPEN_SEARCH_CUTOFF", value)
    torch = Mock()
    hw._configure_rocm_miopen(torch)
    torch.cuda.device_count.assert_not_called()
    assert os.environ["MIOPEN_SEARCH_CUTOFF"] == value


def test_probe_failure_does_not_break_startup(monkeypatch):
    monkeypatch.delenv("MIOPEN_SEARCH_CUTOFF", raising = False)
    torch = Mock()
    torch.cuda.device_count.side_effect = RuntimeError("driver unavailable")
    hw._configure_rocm_miopen(torch)
    assert "MIOPEN_SEARCH_CUTOFF" not in os.environ


@pytest.mark.parametrize(
    "hip, version, enabled",
    [("7.13", "2.11", True), (None, "2.11+rocm7.13", True), (None, "2.11+cu130", False)],
)
def test_detection_configures_only_rocm(monkeypatch, hip, version, enabled):
    torch = SimpleNamespace(
        version = SimpleNamespace(hip = hip),
        __version__ = version,
        cuda = SimpleNamespace(
            is_available = lambda: True, get_device_properties = lambda i: SimpleNamespace(name = "GPU")
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(hw, "_has_torch", lambda: True)
    monkeypatch.setattr(hw, "_print_cuda_device_list", lambda _: None)
    configure = Mock()
    monkeypatch.setattr(hw, "_configure_rocm_miopen", configure)
    monkeypatch.delenv("UNSLOTH_FORCE_XPU", raising = False)
    monkeypatch.delenv("ZE_AFFINITY_MASK", raising = False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    detect = FunctionType(hw._detect_hardware_locked.__code__, vars(hw).copy())
    assert detect() == hw.DeviceType.CUDA
    assert configure.call_count == int(enabled)


def test_desktop_launch_imports_a_shell_override():
    from utils import desktop_shell_env as dse
    assert "MIOPEN_SEARCH_CUTOFF" in dse.ROCM_SHELL_ENV_ALLOWLIST
