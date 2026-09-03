# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The "torch cannot see your GPU" hint, and the four ways it must stay quiet.

`_reraise_device_type_error_with_gpu_hint` replaces unsloth_zoo's bare "you need a
GPU" with the reinstall it actually takes. Everything here pins a case where that
replacement would be WRONG, because each one costs a user a pointless torch
reinstall or buries the real error:

- Zoo already gave ROCm repair advice. It is told apart by "ROCm", not "AMD": zoo's
  generic message names AMD as a supported vendor ("Unsloth currently only works on
  NVIDIA, AMD and Intel GPUs."), and that is the message raised on any torch without
  `torch.accelerator` (2.6+), i.e. exactly the mismatched build this hint is for.
- CUDA_VISIBLE_DEVICES=""/-1 hides the GPU on purpose. nvidia-smi ignores the
  variable (see install.sh `_cvd_hides_nvidia`), so the mask is indistinguishable
  from a broken build unless we look.
- nvidia-smi cannot answer. The probe returns None rather than raising, so the
  original error is re-raised outside the handler; raising inside it chains the
  probe's FileNotFoundError on as __context__ and every CPU-only host reads a
  spurious nvidia-smi traceback first.

The command it prints is asserted too: torch alone leaves a stale torchvision that
fails the next import, and a hardcoded cuXXX is wrong on pre-Turing hosts.
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest

_GPU_INIT = Path(__file__).resolve().parents[1] / "unsloth" / "_gpu_init.py"
_HELPER = "_reraise_device_type_error_with_gpu_hint"

# Zoo messages, verbatim from unsloth_zoo/device_type.py.
_ROCM_HINT = "Unsloth detected signs of an AMD ROCm GPU, but your current PyTorch build has no usable HIP accelerator."
_NO_ACCELERATOR = "Unsloth cannot find any torch accelerator? You need a GPU."
_VENDOR_LIST = "Unsloth currently only works on NVIDIA, AMD and Intel GPUs."


@pytest.fixture(scope = "module")
def helper():
    """The shipped function, without importing unsloth (it `del`s the name at import)."""
    tree = ast.parse(_GPU_INIT.read_text())
    names = (_HELPER, "_nvidia_smi_gpu_name", "_cuda_visible_devices_hides_nvidia")
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert _HELPER in [n.name for n in wanted]
    namespace = {"subprocess": subprocess, "os": os}
    exec(compile(ast.Module(body = wanted, type_ignores = []), str(_GPU_INIT), "exec"), namespace)
    return namespace[_HELPER]


def _smi(
    monkeypatch,
    *,
    returncode = 0,
    stdout = "NVIDIA GB10\n",
    raises = None,
):
    """Stub nvidia-smi and record whether it was reached at all."""
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


@pytest.mark.parametrize("message", [_NO_ACCELERATOR, _VENDOR_LIST])
def test_generic_failure_gets_the_reinstall_hint(helper, monkeypatch, message):
    original = NotImplementedError(message)
    _smi(monkeypatch)

    with pytest.raises(NotImplementedError) as excinfo:
        helper(original)

    assert "NVIDIA GB10" in str(excinfo.value)
    assert "torch.cuda.is_available() is False" in str(excinfo.value)
    assert excinfo.value.__cause__ is original


def test_hint_repairs_the_whole_torch_trio(helper, monkeypatch):
    _smi(monkeypatch)

    with pytest.raises(NotImplementedError) as excinfo:
        helper(NotImplementedError(_NO_ACCELERATOR))

    command = next(line for line in str(excinfo.value).splitlines() if "pip install" in line)
    assert "torchvision" in command and "torchaudio" in command
    # No concrete family: cu128/cu130 have no kernels for pre-Turing cards.
    assert "/cuXXX" in command
    assert "cu126" not in command and "cu128" not in command and "cu130" not in command


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


@pytest.mark.parametrize("mask", ["0", "0,1", "GPU-fake-uuid"])
def test_a_mask_that_still_shows_a_device_keeps_the_hint(helper, monkeypatch, mask):
    _smi(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)

    with pytest.raises(NotImplementedError) as excinfo:
        helper(NotImplementedError(_NO_ACCELERATOR))

    assert "NVIDIA GB10" in str(excinfo.value)


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
    # The probe's own failure must not surface: a CPU-only host would otherwise read
    # "FileNotFoundError: nvidia-smi ... During handling of the above exception".
    assert excinfo.value.__context__ is None
