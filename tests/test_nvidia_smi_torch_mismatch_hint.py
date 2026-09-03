# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The "torch cannot see your GPU" hint must survive zoo's other wording.

`_reraise_device_type_error_with_gpu_hint` steps aside when unsloth_zoo already
gave vendor-specific advice, and it decides that from the message text. Zoo has
two shapes to tell apart:

    _amd_installation_hint():  "Unsloth detected signs of an AMD ROCm GPU, ..."
    generic, no accelerator:   "Unsloth cannot find any torch accelerator? You need a GPU."
                               "Unsloth currently only works on NVIDIA, AMD and Intel GPUs."

Both generic messages mean "probe nvidia-smi"; only the first is zoo's own repair
advice. Matching "AMD" cannot separate them, because the second generic message
names AMD as a supported vendor -- and that is the one raised on any torch without
`torch.accelerator` (added in 2.6), which is exactly the mismatched-build host the
hint exists for. So the guard keys on "ROCm".

The rest is the fallback table from the PR: every way the probe can come back
empty returns the original error, unchanged.
"""

from __future__ import annotations

import ast
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
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == _HELPER)
    namespace = {"subprocess": subprocess}
    exec(compile(ast.Module(body = [node], type_ignores = []), str(_GPU_INIT), "exec"), namespace)
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
    return calls


@pytest.mark.parametrize("message", [_NO_ACCELERATOR, _VENDOR_LIST])
def test_generic_failure_gets_the_reinstall_hint(helper, monkeypatch, message):
    original = NotImplementedError(message)
    _smi(monkeypatch)

    with pytest.raises(NotImplementedError) as excinfo:
        helper(original)

    assert "NVIDIA GB10" in str(excinfo.value)
    assert "torch.cuda.is_available() is False" in str(excinfo.value)
    assert excinfo.value.__cause__ is original


def test_rocm_advice_passes_through_without_probing(helper, monkeypatch):
    original = NotImplementedError(_ROCM_HINT)
    calls = _smi(monkeypatch)

    with pytest.raises(NotImplementedError) as excinfo:
        helper(original)

    assert excinfo.value is original
    assert calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"raises": FileNotFoundError("nvidia-smi")},
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
        helper(original)

    assert excinfo.value is original
