# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""install.sh's post-install check that the CUDA torch has kernels for the GPUs, run against a
stub torch with the driver libraries hidden."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

INSTALL_SH = Path(__file__).resolve().parents[3] / "install.sh"

CU126 = "sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90"
CU130 = "sm_75 sm_80 sm_86 sm_90 sm_100 sm_120"

_SITECUSTOMIZE = """
import ctypes
_real = ctypes.CDLL
class _NoDriver(ctypes.CDLL):
    def __init__(self, name, *a, **k):
        if name and ("nvidia-ml" in name or "libcuda" in name):
            raise OSError(name)
        super().__init__(name, *a, **k)
ctypes.CDLL = _NoDriver
"""

_TORCH = """
class version:
    cuda = {cuda!r}
    hip = None
__version__ = {ver!r}
class _C:
    @staticmethod
    def _cuda_getArchFlags():
        return {archs!r}
class cuda:
    @staticmethod
    def is_available():
        return True
    @staticmethod
    def device_count():
        return {n}
    @staticmethod
    def get_device_capability(i):
        return {caps!r}[i]
"""


def _check_source() -> str:
    text = INSTALL_SH.read_text(encoding = "utf-8")
    m = re.search(r"_run_bounded --secs 120 \"\$_VENV_PY\" -c '\n(.*?)\n' 2>/dev/null", text, re.S)
    assert m, "the post-install arch check was not found"
    return m.group(1)


def _run(tmp_path, cuda, archs, caps):
    stub = tmp_path / "stub"
    (stub / "torch").mkdir(parents = True)
    (stub / "sitecustomize.py").write_text(_SITECUSTOMIZE)
    (stub / "torch" / "__init__.py").write_text(
        _TORCH.format(
            cuda = cuda, ver = f"2.11.0+cu{cuda.replace('.', '')}", archs = archs, n = len(caps), caps = caps
        )
    )
    out = subprocess.run(
        [sys.executable, "-c", _check_source()],
        env = {**os.environ, "PYTHONPATH": str(stub)},
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert out.returncode == 0, out.stderr
    line = [l for l in out.stdout.splitlines() if l.startswith("UNSLOTH_ARCH_CHECK=")]
    return line[-1].split("=", 1)[1].split("|") if line else None


@pytest.mark.parametrize(
    "cuda, archs, caps, expected",
    [
        ("12.6", CU126, [(10, 0)], ("none", "10.0", "cu128")),
        ("12.6", CU126, [(6, 1), (10, 0)], ("some", "10.0", "cu128")),
        ("13.0", CU130, [(6, 1)], ("none", "6.1", "cu126")),
        # No wheel family serves Kepler, and re-installing the same family fixes nothing.
        ("11.8", CU126, [(3, 7)], ("nofix", "3.7", "cu126")),
        ("12.8", CU130, [(13, 0)], ("nofix", "13.0", "cu128")),
    ],
)
def test_uncovered_gpus(tmp_path, cuda, archs, caps, expected):
    status, missing, _torch, _archs, family = _run(tmp_path, cuda, archs, caps)
    assert (status, missing, family) == expected


@pytest.mark.parametrize(
    "archs, caps",
    [
        (CU130, [(10, 3)]),  # sm_100 cubin runs on 10.3
        (CU130, [(12, 1)]),
        ("sm_80 sm_90a compute_90", [(9, 0)]),
        ("sm_80 compute_80", [(8, 9)]),  # PTX JIT forward
    ],
)
def test_covered_gpus_pass_silently(tmp_path, archs, caps):
    assert _run(tmp_path, "13.0", archs, caps) is None


def test_nofix_status_only_warns():
    text = INSTALL_SH.read_text(encoding = "utf-8")
    assert '[ "$_ac_status" = "none" ] && [ "$_torch_index_pinned" = false ]' in text
