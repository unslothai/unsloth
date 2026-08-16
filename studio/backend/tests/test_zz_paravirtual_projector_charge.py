# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The projector's planner charge on the pre-existing paravirtual Metal pin.

That pin is not new. What is new is that a CPU-pinned projector is charged 0
instead of its file size plus the 1.4x _MMPROJ_VRAM_SAFETY surcharge, and
_mmproj_cpu_pinned is seeded True from the paravirtual reason before the fit
runs. So a virtualised Mac loading a vision GGUF gets a different context out of
the planner than origin/main did, with no new request field involved.

The budget that context is sized against is _apple_metal_memory_budget_bytes,
which is UNIFIED memory -- MLX's Metal working set, or RAM. The projector still
occupies that memory when it runs on the CPU, and the branch's own auto-pin
comment says as much ("Metal enumerates no GPU and its memory is unified, so
pinning frees nothing"). The numbers below are the ones origin/main and the
branch actually emitted for the same request.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_llama_cpp_placement import _backend, _launch  # noqa: E402,F401

import core.inference.llama_cpp as llama_cpp  # noqa: E402

MIB = 1024 * 1024
GIB = 1024**3

MODEL_BYTES = 6 * GIB
PROJECTOR_BYTES = 900 * MIB
UNIFIED_BUDGET_BYTES = 10 * GIB
KV_PER_TOKEN = 256 * 1024  # big and round, so the cap inverts by hand
NATIVE_CTX = 32768

# Captured by executing this same builder against origin/main (17363f8a2).
MAIN_MODEL_SIZE_FIT = 7_868_514_303
MAIN_CTX = 10752


def _metal_backend(
    tmp_path: Path,
    *,
    mmproj_bytes = PROJECTOR_BYTES,
    record: dict,
):
    """A Mac: no GPU enumerated, a unified-memory budget, a KV estimate."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [])
    mmproj = tmp_path / "model-mmproj.gguf"
    mmproj.write_bytes(b"\x00" * 16)
    backend._resolve_launch_mmproj_path = lambda **kwargs: str(mmproj)
    backend._mmproj_vram_bytes = lambda _path: mmproj_bytes
    backend._mmproj_matches_model_family = lambda *a, **k: True
    backend._get_gguf_size_bytes = lambda _path: MODEL_BYTES
    backend._estimate_compute_buffer_bytes = lambda *a, **k: 100 * MIB
    backend._compute_buffer_ctx_bytes = lambda *a, **k: 0
    backend._context_length = NATIVE_CTX
    backend._can_estimate_kv = lambda: True
    backend._estimate_kv_cache_bytes = lambda ctx, *a, **k: int(ctx) * KV_PER_TOKEN
    backend._apple_metal_memory_budget_bytes = lambda: UNIFIED_BUDGET_BYTES

    def _fit(requested_ctx, available_mib, model_size_bytes, *a, **kw):
        # Same shape as the real fit, deterministic, and it records the one
        # number this file is about: what the planner thinks the load weighs.
        record["model_size_fit"] = model_size_bytes
        record["available_mib"] = available_mib
        budget = available_mib * MIB
        ctx = requested_ctx
        while ctx > 256 and model_size_bytes + ctx * KV_PER_TOKEN > budget:
            ctx -= 256
        return max(256, ctx)

    backend._fit_context_to_vram = _fit
    return backend, gguf


def _launch_metal(
    tmp_path,
    *,
    paravirtual,
    monkeypatch,
    mmproj_bytes = PROJECTOR_BYTES,
    extra_args = None,
):
    tmp_path.mkdir(parents = True, exist_ok = True)
    monkeypatch.setattr(llama_cpp, "_metal_device_is_paravirtual", lambda: paravirtual)
    record: dict = {}
    backend, gguf = _metal_backend(tmp_path, mmproj_bytes = mmproj_bytes, record = record)
    cmd = [
        str(a)
        for a in _launch(backend, gguf, is_vision = True, n_ctx = 0, extra_args = extra_args)["cmd"]
    ]
    record["backend"] = backend
    record["ctx"] = int(cmd[cmd.index("-c") + 1])
    record["cmd"] = cmd
    return record


def test_a_physical_mac_still_charges_the_projector(tmp_path, monkeypatch):
    """The control. Without the paravirtual pin nothing is charged 0, so this is
    the number origin/main produced on both kinds of Mac."""
    got = _launch_metal(tmp_path, paravirtual = False, monkeypatch = monkeypatch)

    assert got["model_size_fit"] == MAIN_MODEL_SIZE_FIT
    assert got["ctx"] == MAIN_CTX


def test_a_paravirtual_mac_charges_the_same_as_before(tmp_path, monkeypatch):
    """The regression this file exists for.

    The branch's 0 charge reached the paravirtual Metal pin, whose budget is
    UNIFIED memory. Pinning the projector to the CPU frees nothing there, so the
    planner handed out a context sized against 1260 MiB the machine does not
    have. Physical and virtualised Macs must plan identically.
    """
    got = _launch_metal(tmp_path, paravirtual = True, monkeypatch = monkeypatch)

    assert got["model_size_fit"] == MAIN_MODEL_SIZE_FIT
    assert got["ctx"] == MAIN_CTX


def test_the_pin_frees_no_unified_memory(tmp_path, monkeypatch):
    """The size of what the regression gave away: 900 MiB of projector plus the
    360 MiB _MMPROJ_VRAM_SAFETY surcharge, out of a 10 GiB unified budget, which
    the fit then spent on KV. Both Macs must weigh the load the same."""
    physical = _launch_metal(tmp_path / "a", paravirtual = False, monkeypatch = monkeypatch)
    virtual = _launch_metal(tmp_path / "b", paravirtual = True, monkeypatch = monkeypatch)

    assert virtual["model_size_fit"] == physical["model_size_fit"]
    assert virtual["ctx"] == physical["ctx"]
    # And the projector is really in that number, so the equality above is not
    # two zeroes agreeing.
    assert physical["model_size_fit"] > MODEL_BYTES + PROJECTOR_BYTES


def test_a_bigger_projector_does_not_diverge(tmp_path, monkeypatch):
    """Not a rounding error: unfixed, a 3 GiB projector moved the emitted context
    by 3.9x, all of it unified memory the projector is still holding."""
    physical = _launch_metal(
        tmp_path / "a", paravirtual = False, monkeypatch = monkeypatch, mmproj_bytes = 3 * GIB
    )
    virtual = _launch_metal(
        tmp_path / "b", paravirtual = True, monkeypatch = monkeypatch, mmproj_bytes = 3 * GIB
    )

    assert physical["ctx"] == 4096
    assert virtual["ctx"] == 4096


def test_the_argv_itself_is_unchanged_on_a_paravirtual_mac(tmp_path, monkeypatch):
    """Only the planner output moved. The placement flags are origin/main's, so
    an argv-only comparison would report this cell clean -- which is why the
    context above has to be asserted separately."""
    got = _launch_metal(tmp_path, paravirtual = True, monkeypatch = monkeypatch)

    assert "--no-mmproj-offload" in got["cmd"]
    assert got["cmd"].count("--no-mmproj-offload") == 1
    assert got["cmd"][got["cmd"].index("--device") + 1] == "none"
    assert "--gpu-layers" in got["cmd"]
    assert got["cmd"][got["cmd"].index("--gpu-layers") + 1] == "0"
