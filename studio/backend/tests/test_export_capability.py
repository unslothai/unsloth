# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export is supported only on CUDA, XPU or MLX, and the backend must import without PyTorch."""

import ast
import builtins
from pathlib import Path

import pytest

import utils.hardware.hardware as hw

_BACKEND = Path(__file__).resolve().parent.parent


def _src(rel):
    return (_BACKEND / rel).read_text(encoding = "utf-8")


def _func_src(rel, name):
    src = _src(rel)
    node = next(
        n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.get_source_segment(src, node)


def _patch(monkeypatch, *, torch: bool, device, apple: bool):
    monkeypatch.setattr(hw, "_has_torch", lambda: torch)
    monkeypatch.setattr(hw, "get_device", lambda: device)
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: apple)


def test_cpu_with_torch_unsupported_no_accelerator(monkeypatch):
    _patch(monkeypatch, torch = True, device = hw.DeviceType.CPU, apple = False)
    # pin the verdict: it is cached module state an earlier test may have faked
    monkeypatch.setattr(hw, "current_chat_only_verdict", lambda: ("no_gpu", None))
    cap = hw.export_capability()
    assert cap["export_supported"] is False
    assert cap["export_unsupported_reason"] == "no_accelerator"
    assert "accelerator" in cap["export_unsupported_message"].lower()
    assert "PyTorch is not installed" not in cap["export_unsupported_message"]


def test_cuda_with_torch_supports_export(monkeypatch):
    _patch(monkeypatch, torch = True, device = hw.DeviceType.CUDA, apple = False)
    cap = hw.export_capability()
    assert cap["export_supported"] is True
    assert cap["export_unsupported_reason"] is None
    assert cap["export_unsupported_message"] is None


def test_xpu_with_torch_supports_export(monkeypatch):
    _patch(monkeypatch, torch = True, device = hw.DeviceType.XPU, apple = False)
    assert hw.export_capability()["export_supported"] is True


def test_mlx_without_torch_supports_export(monkeypatch):
    _patch(monkeypatch, torch = False, device = hw.DeviceType.MLX, apple = True)
    assert hw.export_capability()["export_supported"] is True


def test_no_torch_non_apple_reports_pytorch_missing(monkeypatch):
    _patch(monkeypatch, torch = False, device = hw.DeviceType.CPU, apple = False)
    # pin the verdict: it is cached module state an earlier test may have faked
    monkeypatch.setattr(hw, "current_chat_only_verdict", lambda: ("no_gpu", None))
    cap = hw.export_capability()
    assert cap["export_supported"] is False
    assert cap["export_unsupported_reason"] == "pytorch_not_installed"
    assert "PyTorch is not installed" in cap["export_unsupported_message"]


def test_apple_without_mlx_reports_mlx_unavailable(monkeypatch):
    for has_torch in (False, True):
        _patch(monkeypatch, torch = has_torch, device = hw.DeviceType.CPU, apple = True)
        cap = hw.export_capability()
        assert cap["export_supported"] is False
        assert cap["export_unsupported_reason"] == "mlx_unavailable"
        assert "MLX" in cap["export_unsupported_message"]


def test_apple_no_torch_install_reports_no_torch(monkeypatch):
    _patch(monkeypatch, torch = False, device = hw.DeviceType.CPU, apple = True)
    monkeypatch.setattr(hw, "current_chat_only_verdict", lambda: ("no_torch", None))
    cap = hw.export_capability()
    assert cap["export_supported"] is False
    assert cap["export_unsupported_reason"] == "no_torch"
    assert "--no-torch" in cap["export_unsupported_message"]
    assert "unsloth studio update" not in cap["export_unsupported_message"]


def test_export_backend_imports_without_torch(monkeypatch):
    """core/export/export.py must import on a --no-torch host (unsloth/torch blocked) and return a
    clean 'PyTorch is not installed' message from an export attempt, not crash at import."""
    import importlib
    import sys

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if top in {"torch", "unsloth"}:
            raise ImportError(f"simulated: {top} not installed")
        return real_import(name, *args, **kwargs)

    # drop preloaded copies so the guarded imports re-run under the block
    for m in [k for k in sys.modules if k.split(".")[0] in {"torch", "unsloth"}]:
        monkeypatch.delitem(sys.modules, m, raising = False)
    monkeypatch.delitem(sys.modules, "core.export.export", raising = False)
    monkeypatch.setattr(builtins, "__import__", blocking_import)

    mod = importlib.import_module("core.export.export")
    assert mod._IS_MLX is False
    assert mod.torch is None
    assert mod._export_runtime_available() is False

    be = mod.ExportBackend.__new__(mod.ExportBackend)
    be.current_model = None
    be.current_tokenizer = None
    be.is_peft = False
    be._audio_type = None
    ok, message, out = be.export_merged_model("/tmp/does-not-matter")
    assert ok is False
    assert "PyTorch is not installed" in message


def test_main_endpoints_expose_export_capability():
    m = _src("main.py")
    assert m.count("**export_capability()") >= 2
    assert '"/api/system/hardware"' in m and '"/api/system"' in m


def test_routes_guard_mutating_endpoints():
    r = _src("routes/export.py")
    assert "async def _ensure_export_supported()" in r
    assert r.count("await _ensure_export_supported()") >= 5


def test_export_methods_check_runtime():
    e = _src("core/export/export.py")
    assert "def _export_runtime_available()" in e
    assert e.count("_export_runtime_available()") >= 5
    assert "_PYTORCH_MISSING_MESSAGE" in e


def test_export_capability_reads_no_torch_helper():
    cap = _func_src("utils/hardware/hardware.py", "export_capability")
    assert "_has_torch()" in cap and "DeviceType.MLX" in cap and "is_apple_silicon()" in cap


def test_a_failed_detection_is_reported_as_such(monkeypatch):
    """A failed probe must report detection_failed, not no_accelerator, which misdirects the user."""
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CPU)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "detection_failed")

    cap = hw.export_capability()

    assert cap["export_supported"] is False
    assert cap["export_unsupported_reason"] == "detection_failed"
    assert "detection failed" in cap["export_unsupported_message"].lower()


def test_a_genuinely_cpu_only_host_still_says_no_accelerator(monkeypatch):
    """The new branch must not swallow the case it sits in front of."""
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CPU)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "no_gpu")
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(hw, "_has_torch", lambda: True)

    cap = hw.export_capability()

    assert cap["export_unsupported_reason"] == "no_accelerator"
