# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for torchao version selection and the Windows-ROCm export gate.

First half: the installer must pin the torchao version matching the installed torch (its cpp
kernels are built per torch release). Second half: torch.distributed is unsupported on Windows
ROCm, so torchao is import-stubbed and the portable FP8/INT8 export must be gated off there
(shared is_win32_rocm() helper) with a clear defensive error.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# install_python_stack.py lives at repo_root/studio/install_python_stack.py
_INSTALL_SCRIPT = Path(__file__).resolve().parents[2] / "install_python_stack.py"

# backend root (studio/backend), for reading/exec-ing backend sources.
_BACKEND = Path(__file__).resolve().parents[1]


def _load_module(monkeypatch):
    """(Re-)import install_python_stack and return it (mirrors test_pytorch_mirror)."""
    sys.modules.pop("install_python_stack", None)
    monkeypatch.syspath_prepend(str(_INSTALL_SCRIPT.parent))
    import install_python_stack

    return install_python_stack


@pytest.mark.parametrize(
    "torch_version, expected",
    [
        # torch 2.10 on CUDA <= 12 -> 0.16.0 (its cpp is built for torch 2.10.0 and
        # loads against the CUDA-12 PyPI wheel). Independent of patch level.
        ("2.10.0+cu128", "torchao==0.16.0"),
        ("2.10.0+cu126", "torchao==0.16.0"),
        ("2.10.0+rocm6.4", "torchao==0.16.0"),
        ("2.10.0+cpu", "torchao==0.16.0"),
        ("2.10.1", "torchao==0.16.0"),
        ("2.10.0", "torchao==0.16.0"),
        # torch 2.10 on CUDA >= 13 (Blackwell / cu130): 0.16.0's CUDA-12 cpp can't
        # load against a CUDA-13 torch (libcudart.so.12 error), so use 0.17.0.
        ("2.10.0+cu130", "torchao==0.17.0"),
        ("2.10.0+cu140", "torchao==0.17.0"),
        # Pre-release / dev / rc builds: the minor is cleaned of non-digits; the
        # CUDA tag still decides 0.16.0 vs 0.17.0.
        ("2.10.0rc1", "torchao==0.16.0"),
        ("2.10.0.dev20250804+cu130", "torchao==0.17.0"),
        ("2.10.0.dev20250804+cu128", "torchao==0.16.0"),
        ("2.10rc1", "torchao==0.16.0"),
        # torch 2.11 (reachable via ROCm rocm7.2) and forward -> 0.17.0.
        ("2.11.0+cu130", "torchao==0.17.0"),
        ("2.11.0", "torchao==0.17.0"),
        ("2.12.0", "torchao==0.17.0"),
        # torch <=2.9 keeps today's pin (already a correct match for 2.9.0).
        ("2.9.0+cu128", "torchao==0.14.0"),
        ("2.9.1", "torchao==0.14.0"),
        ("2.8.0", "torchao==0.14.0"),
        ("2.4.0", "torchao==0.14.0"),
        # Unparseable / missing / non-2.x major -> conservative default.
        (None, "torchao==0.14.0"),
        ("", "torchao==0.14.0"),
        ("garbage", "torchao==0.14.0"),
        ("2", "torchao==0.14.0"),
        ("3.0.0", "torchao==0.14.0"),
    ],
)
def test_select_torchao_spec(monkeypatch, torch_version, expected):
    mod = _load_module(monkeypatch)
    assert mod._select_torchao_spec(torch_version) == expected


def test_default_spec_matches_table(monkeypatch):
    """The default/floor stays the historical pin so older torch is unchanged."""
    mod = _load_module(monkeypatch)
    assert mod._TORCHAO_DEFAULT_SPEC == "torchao==0.14.0"
    assert mod._select_torchao_spec("2.9.0") == mod._TORCHAO_DEFAULT_SPEC


@pytest.mark.parametrize(
    ("rocm_windows_torch_installed", "installed_torch_is_windows_rocm"),
    [
        (True, False),
        (False, True),
    ],
)
def test_skips_torchao_on_windows_rocm(
    monkeypatch, tmp_path, rocm_windows_torch_installed, installed_torch_is_windows_rocm
):
    """The overrides step must skip torchao on Windows ROCm: no working build exists
    there (it imports an absent c10d backend and crashes transformers.quantizers),
    so the installer skips it and relies on the runtime stub instead."""
    mod = _load_module(monkeypatch)
    installed_specs: list[str] = []
    progress_labels: list[str] = []

    def _record_pip_install(*args, **kwargs):
        installed_specs.extend(str(arg) for arg in args)
        return 0

    unstructured_plugin = tmp_path / "unstructured"
    github_plugin = tmp_path / "github"
    unstructured_plugin.mkdir()
    github_plugin.mkdir()

    subprocess_result = MagicMock()
    subprocess_result.returncode = 0
    subprocess_result.stdout = ""

    monkeypatch.setenv("SKIP_STUDIO_BASE", "1")
    monkeypatch.setattr(mod, "IS_WINDOWS", True)
    monkeypatch.setattr(mod, "IS_MACOS", False)
    monkeypatch.setattr(mod, "IS_MAC_ARM", False)
    monkeypatch.setattr(mod, "NO_TORCH", False)
    monkeypatch.setattr(mod, "_rocm_windows_torch_installed", rocm_windows_torch_installed)
    monkeypatch.setattr(
        mod, "_installed_torch_is_windows_rocm", lambda: installed_torch_is_windows_rocm
    )
    monkeypatch.setattr(mod, "_bootstrap_uv", lambda: False)
    monkeypatch.setattr(mod, "_repair_bad_anyio", lambda: None)
    monkeypatch.setattr(mod, "_ensure_rocm_torch", lambda: None)
    monkeypatch.setattr(mod, "_ensure_cuda_torch", lambda: None)
    monkeypatch.setattr(mod, "_has_usable_nvidia_gpu", lambda: True)
    monkeypatch.setattr(mod, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "pip_install", _record_pip_install)
    monkeypatch.setattr(mod, "_progress", lambda label: progress_labels.append(label))
    monkeypatch.setattr(mod, "LOCAL_DD_UNSTRUCTURED_PLUGIN", unstructured_plugin)
    monkeypatch.setattr(mod, "LOCAL_DD_GITHUB_PLUGIN", github_plugin)
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: subprocess_result)

    assert mod.install_python_stack() == 0

    assert not any(spec.startswith("torchao") for spec in installed_specs)
    assert "dependency overrides (skipped, Windows ROCm)" in progress_labels


# -- Windows-ROCm torchao export gate -----------------------------------------------------------
# torchao is import-stubbed on Windows ROCm (no torch.distributed) and its config classes return
# None, which made TorchAoConfig(quant_type=None) crash. These prove the shared is_win32_rocm()
# gate hides the torchao formats and the defensive path raises a clear error instead.

import core._torchao_stub as _stub


def _func_src(rel, name):
    src = (_BACKEND / rel).read_text(encoding = "utf-8")
    node = next(
        n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.get_source_segment(src, node)


def _exec_func(rel, name):
    """Exec one backend function in isolation, avoiding export.py's heavy import chain."""
    ns: dict = {}
    exec(_func_src(rel, name), ns)
    return ns[name]


@pytest.mark.parametrize(
    ("platform", "hip", "version", "expected"),
    [
        ("win32", "6.4.0", "2.10.0+rocm6.4", True),  # ROCm via torch.version.hip
        ("win32", None, "2.10.0+rocm6.4", True),  # ROCm via __version__ tag only
        ("win32", None, "2.10.0+cu128", False),  # Windows CUDA -> real torchao
        ("linux", "6.4.0", "2.10.0+rocm6.4", False),  # Linux ROCm -> real torchao
        ("darwin", None, "2.10.0", False),  # macOS
    ],
)
def test_is_win32_rocm(monkeypatch, platform, hip, version, expected):
    fake_torch = types.SimpleNamespace(version = types.SimpleNamespace(hip = hip), __version__ = version)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert _stub.is_win32_rocm() is expected


def test_gate_and_stub_share_helper():
    # The stub installer and the export gate must both route through is_win32_rocm() so they can't
    # drift (the gate off while the stub is still active, or the reverse).
    stub_src = (_BACKEND / "core" / "_torchao_stub.py").read_text(encoding = "utf-8")
    assert "def is_win32_rocm(" in stub_src
    assert "is_win32_rocm()" in _func_src(
        "core/_torchao_stub.py", "install_torchao_windows_rocm_stub"
    )
    assert "is_win32_rocm()" in _func_src("core/export/export.py", "_torchao_export_supported")


def test_installer_noop_off_windows_rocm(monkeypatch):
    # is_win32_rocm() False -> installer must not register the finder or seed torchao stubs.
    monkeypatch.setattr(_stub, "is_win32_rocm", lambda: False)
    before = list(sys.meta_path)
    _stub.install_torchao_windows_rocm_stub()
    assert list(sys.meta_path) == before


# (a) gate off on Windows ROCm; (b) unchanged elsewhere


def test_torchao_gate_false_on_windows_rocm(monkeypatch):
    # (a) On Windows ROCm the portable torchao formats are not offered, without importing unsloth.
    monkeypatch.setattr(_stub, "is_win32_rocm", lambda: True)
    assert _exec_func("core/export/export.py", "_torchao_export_supported")() is False


_TORCHAO_ALIASES = {"torchao_fp8", "torchao_int8", "portable_fp8", "portable_int8"}


def _fake_normalize_torchao(save_method):
    # Mirrors unsloth.save._normalize_torchao_method (lower/strip, - and space -> _).
    if not isinstance(save_method, str):
        return None
    key = save_method.lower().strip().replace("-", "_").replace(" ", "_")
    return ("fp8", "torchao-fp8") if key in _TORCHAO_ALIASES else None


def _install_fake_unsloth_save(monkeypatch, *, has_method):
    unsloth = types.ModuleType("unsloth")
    save = types.ModuleType("unsloth.save")
    if has_method:
        save._normalize_torchao_method = _fake_normalize_torchao
    unsloth.save = save
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)
    monkeypatch.setitem(sys.modules, "unsloth.save", save)


def test_torchao_gate_supported_off_windows_rocm(monkeypatch):
    # (b) Off Windows ROCm the gate is unchanged: True when the unsloth build has the method.
    monkeypatch.setattr(_stub, "is_win32_rocm", lambda: False)
    _install_fake_unsloth_save(monkeypatch, has_method = True)
    assert _exec_func("core/export/export.py", "_torchao_export_supported")() is True


def test_torchao_gate_false_when_build_lacks_method(monkeypatch):
    # (b) Off Windows ROCm, an older unsloth without the method is still unsupported.
    monkeypatch.setattr(_stub, "is_win32_rocm", lambda: False)
    _install_fake_unsloth_save(monkeypatch, has_method = False)
    assert _exec_func("core/export/export.py", "_torchao_export_supported")() is False


# (c) defensive early error when torchao is stubbed / unavailable


def _load_export_module_no_torch(monkeypatch):
    """Import core.export.export with torch/unsloth blocked (mirrors test_export_capability), so
    the defensive path runs on CPU with no GPU and no torchao."""
    import builtins
    import importlib

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        # Block real torch/unsloth, but honor injected fakes already in sys.modules.
        top = name.split(".")[0]
        if top in {"torch", "unsloth"} and top not in sys.modules:
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    for m in [k for k in list(sys.modules) if k.split(".")[0] in {"torch", "unsloth"}]:
        monkeypatch.delitem(sys.modules, m, raising = False)
    monkeypatch.delitem(sys.modules, "core.export.export", raising = False)
    monkeypatch.setattr(builtins, "__import__", blocking_import)
    return importlib.import_module("core.export.export")


def _bare_backend(mod):
    be = mod.ExportBackend.__new__(mod.ExportBackend)
    be.current_model = object()
    be.current_tokenizer = object()
    be._audio_type = None
    be.is_peft = True
    return be


def test_torchao_defensive_error_on_windows_rocm(monkeypatch):
    # (c) A forced torchao request reaches the merged path -> clear error, not the NoneType crash.
    mod = _load_export_module_no_torch(monkeypatch)
    monkeypatch.setattr(mod, "_export_runtime_available", lambda: True)
    monkeypatch.setattr(_stub, "is_win32_rocm", lambda: True)

    ok, message, out = _bare_backend(mod).export_merged_model(
        "/tmp/x", compressed_method = "torchao_fp8"
    )
    assert ok is False and out is None
    assert "Windows ROCm" in message and "torchao" in message.lower()


def test_torchao_defensive_error_alias_form_on_windows_rocm(monkeypatch):
    # An equivalent alias unsloth accepts (portable_fp8) must hit the same rejection, not fall
    # through to the misleading NVIDIA compressed-tensors error.
    mod = _load_export_module_no_torch(monkeypatch)
    monkeypatch.setattr(mod, "_export_runtime_available", lambda: True)
    monkeypatch.setattr(_stub, "is_win32_rocm", lambda: True)
    _install_fake_unsloth_save(monkeypatch, has_method = True)

    ok, message, out = _bare_backend(mod).export_merged_model(
        "/tmp/x", compressed_method = "portable_fp8"
    )
    assert ok is False and out is None
    assert "Windows ROCm" in message and "torchao" in message.lower()


def test_is_torchao_alias_recognizes_all_forms(monkeypatch):
    _install_fake_unsloth_save(monkeypatch, has_method = True)
    fn = _exec_func("core/export/export.py", "_is_torchao_alias")
    for alias in ("torchao_fp8", "portable_int8", "portable-fp8", "Portable FP8"):
        assert fn(alias) is True
    for alias in ("fp8", "nvfp4", "w8a8", "", None):
        assert fn(alias) is False


def test_torchao_defensive_error_wired_early():
    # Guard is in export_merged_model before the merge/quant work; alias is normalized (not just the
    # torchao_ prefix) so every torchao form is caught.
    m = _func_src("core/export/export.py", "export_merged_model")
    assert "_is_torchao_alias(compressed_alias)" in m
    assert "_torchao_runtime_unavailable()" in m
    alias_fn = _func_src("core/export/export.py", "_is_torchao_alias")
    assert "_normalize_torchao_method(alias)" in alias_fn
    assert 'startswith("torchao")' in alias_fn


# (issue 2/4) backend win32_rocm flag + single finder registration


def test_export_capability_exposes_win32_rocm(monkeypatch):
    import utils.hardware.hardware as hw

    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(hw, "IS_ROCM", True)
    assert hw.export_capability()["win32_rocm"] is True
    monkeypatch.setattr(hw, "IS_ROCM", False)
    assert hw.export_capability()["win32_rocm"] is False
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(hw, "IS_ROCM", True)
    assert hw.export_capability()["win32_rocm"] is False


def test_installer_registers_finder_once(monkeypatch):
    # Repeated install must not stack duplicate finders. Restore global state after.
    monkeypatch.setattr(_stub, "is_win32_rocm", lambda: True)
    meta_before = list(sys.meta_path)
    tao_before = {k for k in sys.modules if k == "torchao" or k.startswith("torchao.")}
    try:
        _stub.install_torchao_windows_rocm_stub()
        _stub.install_torchao_windows_rocm_stub()
        finders = [f for f in sys.meta_path if isinstance(f, _stub._StubSubpackageFinder)]
        assert len(finders) == 1
    finally:
        sys.meta_path[:] = meta_before
        for k in [
            k
            for k in sys.modules
            if (k == "torchao" or k.startswith("torchao.")) and k not in tao_before
        ]:
            del sys.modules[k]
