# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``diffusion_aot_cache.pin_triton_libdevice``: resolve Triton's libdevice before the first compiled forward, so a
restart's FX graph cache lookup computes the same ``extern_libs_key`` the compile that wrote the entry saved."""

from __future__ import annotations

import sys
import types

import pytest
import torch

# The real runtime package first: importing it after ``triton`` is stubbed pulls dynamo, which reads triton.language.
import torch._inductor.runtime  # noqa: E402,F401

from core.inference import diffusion_aot_cache as aot

TOOLKIT = "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc"


def _stack(monkeypatch, *, policy: str):
    """A torch / Triton pair. ``policy`` is the gated resolver's rule: "2.13" switches under emulate_precision_casts
    (torch 2.13+), "2.12" only under eager_numerics.use_pytorch_libdevice (torch 2.12)."""
    knobs = types.SimpleNamespace(nvidia = types.SimpleNamespace(libdevice_path = None))
    config = types.SimpleNamespace(
        emulate_precision_casts = True,
        eager_numerics = types.SimpleNamespace(use_pytorch_libdevice = False),
    )
    calls = []

    def impl():
        calls.append(1)
        if knobs.nvidia.libdevice_path is None:
            knobs.nvidia.libdevice_path = TOOLKIT

    def gated():
        wanted = config.eager_numerics.use_pytorch_libdevice
        if policy == "2.13":
            wanted = wanted or config.emulate_precision_casts
        if wanted:
            impl()

    triton = types.ModuleType("triton")
    triton.knobs = knobs
    monkeypatch.setitem(sys.modules, "triton", triton)
    tasks = types.ModuleType("torch._inductor.runtime.compile_tasks")
    tasks._set_triton_libdevice_path = gated
    tasks._set_triton_libdevice_path_impl = impl
    monkeypatch.setitem(sys.modules, "torch._inductor.runtime.compile_tasks", tasks)
    import torch._inductor.runtime as runtime

    monkeypatch.setattr(runtime, "compile_tasks", tasks, raising = False)
    monkeypatch.setattr(torch.version, "cuda", "13.0", raising = False)
    utils_triton = types.ModuleType("torch.utils._triton")
    utils_triton._extern_libs_key = lambda backend: ""
    monkeypatch.setitem(sys.modules, "torch.utils._triton", utils_triton)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_PIN_LIBDEVICE", raising = False)
    return knobs, calls, config


@pytest.fixture
def fake_stack(monkeypatch):
    """torch 2.14 with a CUDA toolkit: the gated resolver switches the knob under emulate_precision_casts."""
    knobs, calls, _ = _stack(monkeypatch, policy = "2.13")
    return knobs, calls


def test_pins_the_toolkit_libdevice_up_front(fake_stack):
    knobs, calls = fake_stack
    path = aot.pin_triton_libdevice()
    assert calls == [1]
    assert path == knobs.nvidia.libdevice_path == TOOLKIT


def test_torch_2_12_policy_is_kept(monkeypatch):
    """torch 2.12 keys the FX cache on libdevice but only switches under use_pytorch_libdevice: emulate_precision_casts
    alone must leave the bundled file in place, as a cold compile there would."""
    knobs, calls, config = _stack(monkeypatch, policy = "2.12")
    assert aot.pin_triton_libdevice() is None
    assert calls == [] and knobs.nvidia.libdevice_path is None
    config.eager_numerics.use_pytorch_libdevice = True
    assert aot.pin_triton_libdevice() == TOOLKIT and calls == [1]


def test_a_build_without_the_gated_resolver_is_left_alone(fake_stack, monkeypatch):
    knobs, calls = fake_stack
    monkeypatch.delattr(
        sys.modules["torch._inductor.runtime.compile_tasks"], "_set_triton_libdevice_path"
    )
    assert aot.pin_triton_libdevice() is None
    assert calls == [] and knobs.nvidia.libdevice_path is None


def test_kill_switch(fake_stack, monkeypatch):
    knobs, calls = fake_stack
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PIN_LIBDEVICE", "0")
    assert aot.pin_triton_libdevice() is None
    assert calls == [] and knobs.nvidia.libdevice_path is None


def test_a_build_without_the_extern_libs_check_is_left_alone(fake_stack, monkeypatch):
    """Older torch never keyed cache entries on libdevice, so there is no miss to prevent."""
    knobs, calls = fake_stack
    monkeypatch.setitem(sys.modules, "torch.utils._triton", types.ModuleType("torch.utils._triton"))
    assert aot.pin_triton_libdevice() is None
    assert calls == []


def test_cpu_only_torch_is_left_alone(fake_stack, monkeypatch):
    knobs, calls = fake_stack
    monkeypatch.setattr(torch.version, "cuda", None, raising = False)
    assert aot.pin_triton_libdevice() is None
    assert calls == []


def test_works_before_anything_imported_the_triton_utils(fake_stack, monkeypatch):
    """torch.utils._triton is a lazily imported submodule: probing it as an attribute found nothing in a real run."""
    monkeypatch.delattr(torch.utils, "_triton", raising = False)
    knobs, calls = fake_stack
    assert aot.pin_triton_libdevice() is not None and calls == [1]


def test_never_raises(fake_stack, monkeypatch):
    knobs, calls = fake_stack

    def boom():
        raise RuntimeError("resolver moved")

    sys.modules["torch._inductor.runtime.compile_tasks"]._set_triton_libdevice_path = boom
    assert aot.pin_triton_libdevice() is None


def test_regional_compile_pins_when_it_turns_on_emulate_precision_casts(monkeypatch):
    """The pin rides on the knob that makes torch switch libdevice, in the one place Studio sets it."""
    import inspect

    from core.inference import diffusion_speed

    src = inspect.getsource(diffusion_speed._compile_repeated_blocks)
    knob = src.index('"emulate_precision_casts", True')
    pin = src.index("pin_triton_libdevice")
    assert knob < pin < knob + 400
