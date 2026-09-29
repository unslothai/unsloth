# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A CUDA spoof has to supply `torch._C._cuda_getCurrentRawStream`.

`unsloth/kernels/utils.py` snapshots each device's raw stream AT IMPORT, inside
`if DEVICE_COUNT > 0` -- a branch the spoof makes true by answering
`torch.cuda.device_count()` with 1:

    torch._C._cuda_getCurrentRawStream(index)

A CPU-only torch wheel does not export that symbol. The spoof already knew as much -- it
imports bitsandbytes before flipping `is_available()` precisely because bitsandbytes reads
it -- but it only worked around the one importer and never supplied the symbol, so anything
reading it afterwards still died.

The notebooks smoke matrix showed it on exactly one leg: seven passed and
`nb/Llama3.1_(8B)-GRPO.ipynb` failed with

    AttributeError: module 'torch._C' has no attribute '_cuda_getCurrentRawStream'

That leg's install cell is the one pulling vLLM and the CUDA userspace packages
(cuda-python, cuda-bindings, flashinfer), which is what carried unsloth into the
`DEVICE_COUNT > 0` branch on a CPU-only torch.

Run in a subprocess: applying a spoof mutates the interpreter's torch for good, and this
host's torch exports the symbol for real, so the absence has to be staged somewhere
disposable.
"""

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
    # The null stream. Callers wrap it in ctypes.c_void_p, where 0 reads back as None.
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
