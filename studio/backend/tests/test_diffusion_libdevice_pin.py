# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``diffusion_aot_cache.pin_triton_libdevice``: resolve Triton's libdevice before the first compiled forward, so a
restart's FX graph cache lookup computes the same ``extern_libs_key`` the compile that wrote the entry saved."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from core.inference import diffusion_aot_cache as aot


@pytest.fixture
def fake_stack(monkeypatch):
    """A torch / Triton pair where the lazy resolver switches the knob, like torch 2.14 with a CUDA toolkit."""
    knobs = types.SimpleNamespace(nvidia = types.SimpleNamespace(libdevice_path = None))
    calls = []

    def resolve():
        calls.append(1)
        if knobs.nvidia.libdevice_path is None:
            knobs.nvidia.libdevice_path = "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc"

    triton = types.ModuleType("triton")
    triton.knobs = knobs
    monkeypatch.setitem(sys.modules, "triton", triton)
    tasks = types.ModuleType("torch._inductor.runtime.compile_tasks")
    tasks._set_triton_libdevice_path_impl = resolve
    monkeypatch.setitem(sys.modules, "torch._inductor.runtime.compile_tasks", tasks)
    import torch._inductor.runtime as runtime

    monkeypatch.setattr(runtime, "compile_tasks", tasks, raising = False)
    monkeypatch.setattr(torch.version, "cuda", "13.0", raising = False)
    utils_triton = getattr(torch.utils, "_triton", None)
    if utils_triton is None or not hasattr(utils_triton, "_extern_libs_key"):
        fake = types.SimpleNamespace(_extern_libs_key = lambda backend: "")
        monkeypatch.setattr(torch.utils, "_triton", fake, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_PIN_LIBDEVICE", raising = False)
    return knobs, calls


def test_pins_the_toolkit_libdevice_up_front(fake_stack):
    knobs, calls = fake_stack
    path = aot.pin_triton_libdevice()
    assert calls == [1]
    assert path == knobs.nvidia.libdevice_path == "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc"


def test_kill_switch(fake_stack, monkeypatch):
    knobs, calls = fake_stack
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PIN_LIBDEVICE", "0")
    assert aot.pin_triton_libdevice() is None
    assert calls == [] and knobs.nvidia.libdevice_path is None


def test_a_build_without_the_extern_libs_check_is_left_alone(fake_stack, monkeypatch):
    """Older torch never keyed cache entries on libdevice, so there is no miss to prevent."""
    knobs, calls = fake_stack
    monkeypatch.setattr(torch.utils, "_triton", types.SimpleNamespace(), raising = False)
    assert aot.pin_triton_libdevice() is None
    assert calls == []


def test_cpu_only_torch_is_left_alone(fake_stack, monkeypatch):
    knobs, calls = fake_stack
    monkeypatch.setattr(torch.version, "cuda", None, raising = False)
    assert aot.pin_triton_libdevice() is None
    assert calls == []


def test_never_raises(fake_stack, monkeypatch):
    knobs, calls = fake_stack

    def boom():
        raise RuntimeError("resolver moved")

    sys.modules["torch._inductor.runtime.compile_tasks"]._set_triton_libdevice_path_impl = boom
    assert aot.pin_triton_libdevice() is None


def test_regional_compile_pins_when_it_turns_on_emulate_precision_casts(monkeypatch):
    """The pin rides on the knob that makes torch switch libdevice, in the one place Studio sets it."""
    import inspect

    from core.inference import diffusion_speed

    src = inspect.getsource(diffusion_speed._compile_repeated_blocks)
    knob = src.index('"emulate_precision_casts", True')
    pin = src.index("pin_triton_libdevice")
    assert knob < pin < knob + 400
