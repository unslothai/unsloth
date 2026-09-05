# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_GPU_INIT = Path(__file__).resolve().parents[1] / "unsloth" / "_gpu_init.py"
_HELPER = "_reraise_device_type_error_with_gpu_hint"

# Messages copied from unsloth_zoo/device_type.py.
_ROCM_HINT = "Unsloth detected signs of an AMD ROCm GPU, but your current PyTorch build has no usable HIP accelerator."
_NO_ACCELERATOR = "Unsloth cannot find any torch accelerator? You need a GPU."
_VENDOR_LIST = "Unsloth currently only works on NVIDIA, AMD and Intel GPUs."


@pytest.fixture(scope = "module")
def helper():
    """The shipped function, without importing unsloth (it `del`s the name at import)."""
    tree = ast.parse(_GPU_INIT.read_text(encoding = "utf-8"))
    names = (_HELPER, "_nvidia_smi_gpu_name", "_cuda_visible_devices_hides_nvidia")
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert _HELPER in [n.name for n in wanted]
    namespace = {"subprocess": subprocess, "os": os, "sys": sys}
    exec(compile(ast.Module(body = wanted, type_ignores = []), str(_GPU_INIT), "exec"), namespace)
    return namespace[_HELPER]


def _smi(
    monkeypatch,
    *,
    returncode = 0,
    stdout = "NVIDIA GB10\n",
    raises = None,
):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(argv, returncode, stdout = stdout, stderr = "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    return calls


def _raise_through_handler(helper, original):
    """Call the helper the way _gpu_init does: from inside `except NotImplementedError`."""
    try:
        raise original
    except NotImplementedError as caught:
        helper(caught)


def test_every_raise_site_is_wrapped():
    tree = ast.parse(_GPU_INIT.read_text(encoding = "utf-8"))
    handlers = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ExceptHandler)
        and isinstance(n.type, ast.Name)
        and n.type.id == "NotImplementedError"
        and any(
            isinstance(c, ast.Call) and getattr(c.func, "id", None) == _HELPER for c in ast.walk(n)
        )
    ]
    assert len(handlers) == 3, [h.lineno for h in handlers]

    deletes = [n for n in ast.walk(tree) if isinstance(n, ast.Delete)]
    freed = [
        n.lineno
        for n in deletes
        if any(isinstance(t, ast.Name) and t.id == _HELPER for t in ast.walk(n))
    ]
    assert len(freed) == 1
    assert freed[0] > max(h.lineno for h in handlers)


@pytest.mark.parametrize("message", [_NO_ACCELERATOR, _VENDOR_LIST])
def test_generic_failure_gets_the_reinstall_hint(helper, monkeypatch, message):
    original = NotImplementedError(message)
    _smi(monkeypatch)

    with pytest.raises(NotImplementedError) as excinfo:
        helper(original)

    assert "NVIDIA GB10" in str(excinfo.value)
    assert "torch.cuda.is_available() is False" in str(excinfo.value)
    assert excinfo.value.__cause__ is original


def test_hint_diagnoses_and_links_rather_than_printing_a_command(helper, monkeypatch):
    _smi(monkeypatch)

    with pytest.raises(NotImplementedError) as excinfo:
        helper(NotImplementedError(_NO_ACCELERATOR))

    message = str(excinfo.value)
    assert "https://github.com/unslothai/unsloth#-install" in message
    assert sys.executable in message
    assert not re.search(r"pip install|cu\d{3}|--torch-backend|torch[<>=]", message)


def test_rocm_advice_passes_through_without_probing(helper, monkeypatch):
    original = NotImplementedError(_ROCM_HINT)
    calls = _smi(monkeypatch)

    with pytest.raises(NotImplementedError) as excinfo:
        helper(original)

    assert excinfo.value is original
    assert calls == []


@pytest.mark.parametrize("mask", ["", "-1", " ", " -1 "])
def test_deliberately_hidden_gpu_is_not_a_broken_build(helper, monkeypatch, mask):
    original = NotImplementedError(_NO_ACCELERATOR)
    calls = _smi(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)

    with pytest.raises(NotImplementedError) as excinfo:
        helper(original)

    assert excinfo.value is original
    assert calls == []


@pytest.mark.parametrize("mask", ["0", "0,1", "1", "GPU-fake-uuid"])
def test_a_partial_mask_keeps_the_hint_but_names_itself(helper, monkeypatch, mask):
    _smi(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)

    with pytest.raises(NotImplementedError) as excinfo:
        helper(NotImplementedError(_NO_ACCELERATOR))

    assert "NVIDIA GB10" in str(excinfo.value)
    assert f"CUDA_VISIBLE_DEVICES is set to {mask!r}" in str(excinfo.value)


def test_no_mask_note_when_the_variable_is_unset(helper, monkeypatch):
    _smi(monkeypatch)

    with pytest.raises(NotImplementedError) as excinfo:
        helper(NotImplementedError(_NO_ACCELERATOR))

    assert "CUDA_VISIBLE_DEVICES" not in str(excinfo.value)


def test_undecodable_probe_output_does_not_escape_the_handler(helper, monkeypatch):
    """The probe decodes with errors="replace", so bad bytes cannot raise UnicodeDecodeError."""
    seen = {}

    def fake_run(argv, **kwargs):
        seen.update(kwargs)
        stdout = b"NVIDIA GB\xc5\x31\n".decode("utf-8", kwargs.get("errors", "strict"))
        return subprocess.CompletedProcess(argv, 0, stdout = stdout, stderr = "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)

    with pytest.raises(NotImplementedError) as excinfo:
        helper(NotImplementedError(_NO_ACCELERATOR))

    assert seen["errors"] == "replace"
    assert "NVIDIA GB" in str(excinfo.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"raises": FileNotFoundError(2, "No such file or directory", "nvidia-smi")},
        {"raises": subprocess.TimeoutExpired("nvidia-smi", 5)},
        {"returncode": 9},
        {"stdout": "  \n"},
    ],
    ids = ["missing", "timeout", "nonzero_exit", "empty_stdout"],
)
def test_probe_failures_return_the_original_error(helper, monkeypatch, kwargs):
    original = NotImplementedError(_NO_ACCELERATOR)
    _smi(monkeypatch, **kwargs)

    with pytest.raises(NotImplementedError) as excinfo:
        _raise_through_handler(helper, original)

    assert excinfo.value is original
    assert excinfo.value.__cause__ is None
    # Do not show the probe failure before the original error.
    assert excinfo.value.__context__ is None
