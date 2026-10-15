# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A CUDA spoof must supply torch._C._cuda_getCurrentRawStream; kernels/utils.py reads it at import."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

import pytest

_TESTS = pathlib.Path(__file__).resolve().parent
_SPOOFS = ("_zoo_aggressive_cuda_spoof.py", "_zoo_rocm_spoof.py")


def _probe(spoof_module: str) -> str:
    """Delete the symbol, apply the spoof, then evaluate what unsloth evaluates."""
    script = textwrap.dedent(
        f"""
        import ctypes, sys
        sys.path.insert(0, {str(_TESTS)!r})
        import torch

        # Stand in for a CPU-only wheel, which simply does not carry it.
        if hasattr(torch._C, "_cuda_getCurrentRawStream"):
            delattr(torch._C, "_cuda_getCurrentRawStream")
        try:
            torch._C._cuda_getCurrentRawStream(0)
            print("PRE unexpectedly-present")
        except AttributeError:
            print("PRE absent")

        import {spoof_module} as spoof
        spoof.apply()

        # unsloth/kernels/utils.py, verbatim in shape.
        try:
            handle = ctypes.c_void_p(torch._C._cuda_getCurrentRawStream(0))
        except AttributeError as e:
            print("POST AttributeError", e)
        else:
            print("POST ok", handle.value)
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output = True, text = True)
    assert (
        "PRE absent" in out.stdout
    ), f"the probe never staged the absence: {out.stdout}\n{out.stderr}"
    return out.stdout


@pytest.mark.parametrize("filename", _SPOOFS)
def test_the_spoof_supplies_the_raw_stream_handle(filename):
    spoof = filename[: -len(".py")]
    if not (_TESTS / filename).is_file():
        pytest.skip(f"{filename} is gone")
    printed = _probe(spoof)
    assert "POST AttributeError" not in printed, (
        f"{filename} leaves torch._C._cuda_getCurrentRawStream absent, so importing unsloth "
        f"under it dies the way the GRPO smoke leg did: {printed}"
    )
    # Callers wrap the null stream in ctypes.c_void_p, where 0 reads back as None.
    assert "POST ok None" in printed, printed


def test_the_spoof_leaves_a_real_raw_stream_alone():
    """Only supplied when absent: a machine with CUDA keeps the handle its driver gives."""
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(_TESTS)!r})
        import torch
        sentinel = object()
        torch._C._cuda_getCurrentRawStream = lambda index = 0: 4242
        import _zoo_aggressive_cuda_spoof as spoof
        spoof.apply()
        print("VALUE", torch._C._cuda_getCurrentRawStream(0))
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output = True, text = True)
    assert (
        "VALUE 4242" in out.stdout
    ), f"the spoof overwrote a raw-stream handle that was already there: {out.stdout}\n{out.stderr}"
