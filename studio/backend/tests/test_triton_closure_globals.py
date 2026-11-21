# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Triton 3.2 / 3.3 (torch 2.6 / 2.7) look a JIT kernel's names up in its module globals only: every kernel a Studio
factory builds must find ``tl``, its helper JIT functions and its constants there, not only in the factory closure
(CPU, nothing compiles)."""

import importlib

import pytest

pytest.importorskip("triton")

FACTORIES = [
    ("core.inference.diffusion_convrot_quant", "_kernels"),
    ("core.inference.diffusion_flux2_rope", "_kernel"),
    ("core.inference.diffusion_flux3_qk_rope", "_kernel"),
    ("core.inference.diffusion_int8_fused", "_kernels"),
    ("core.inference.diffusion_int8_gemm", "_kernels"),
    ("core.inference.diffusion_int8_gemv", "_kernel"),
    ("core.inference.diffusion_moss_fused", "_qk_rope_kernel"),
    ("core.inference.diffusion_native_w8a8_rdna", "_kernels"),
    ("core.inference.diffusion_qwenimage21_eager", "_kernel"),
    ("core.inference.diffusion_rocm_fused", "_adaln_kernel"),
    ("core.inference.diffusion_vae_fused", "_kernels"),
    ("core.inference.diffusion_vae_strips", "_residual_kernel"),
    ("core.inference.video_minimax_h3_qknorm", "_kernel"),
    ("core.inference.video_minimax_h3_rowmod", "_kernels"),
    ("core.inference.video_minimax_h3_vae", "_kernels"),
    ("core.inference.video_wan_fused", "_kernels"),
    ("core.training.diffusion_frozen_fp8", "_transpose_kernel"),
]


def _kernels(obj, seen):
    # every JITFunction reachable from a factory's result (namespaces, containers, launchers, called JIT helpers)
    if obj is None or id(obj) in seen:
        return []
    seen.add(id(obj))
    if hasattr(obj, "run") and hasattr(getattr(obj, "fn", None), "__code__"):
        return [obj] + [
            k for c in obj.fn.__closure__ or () for k in _kernels(c.cell_contents, seen)
        ]
    if isinstance(obj, (tuple, list)):
        return [k for o in obj for k in _kernels(o, seen)]
    if isinstance(obj, dict):
        return [k for o in obj.values() for k in _kernels(o, seen)]
    if hasattr(obj, "__code__"):
        return [k for c in obj.__closure__ or () for k in _kernels(c.cell_contents, seen)]
    if type(obj).__module__ == "types":
        return [k for o in vars(obj).values() for k in _kernels(o, seen)]
    return []


def _closure_only(kernel):
    fn = kernel.fn
    return [
        f"{fn.__name__}:{name}"
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ())
        if fn.__globals__.get(name) is not cell.cell_contents
    ]


@pytest.mark.parametrize(
    "module, factory", FACTORIES, ids = [m.rsplit(".", 1)[1] for m, _ in FACTORIES]
)
def test_every_kernel_resolves_its_names_from_module_globals(module, factory):
    built = getattr(importlib.import_module(module), factory)()
    kernels = _kernels(built, set())
    assert kernels, f"{module}.{factory} built no kernel here"
    assert [n for k in kernels for n in _closure_only(k)] == []


_FACTORY_SRC = """
def factory():
    import triton
    import triton.language as tl

    @triton.jit
    def helper(x):
        return x + 1

    @triton.jit
    def kernel(p):
        tl.store(p, helper(tl.load(p)))

    def launch(t):
        return kernel[(1,)](t)

    return launch
"""


def test_publish_reaches_helpers_and_launchers_and_keeps_existing_globals(tmp_path, monkeypatch):
    from core.inference import triton_closure_globals as G

    (tmp_path / "fake_triton_kernels.py").write_text(_FACTORY_SRC, encoding = "utf-8")
    (tmp_path / "fake_triton_kernels_tl.py").write_text(
        'tl = "a module-level tl"\n' + _FACTORY_SRC, encoding = "utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = importlib.import_module("fake_triton_kernels")
    assert not hasattr(mod, "tl") and not hasattr(mod, "helper")
    G.publishes_closure_globals(mod.factory)()
    # found through the plain launcher's closure: the kernel's names, including the JIT helper it calls
    assert mod.tl.__name__ == "triton.language" and mod.helper.fn.__name__ == "helper"
    other = importlib.import_module("fake_triton_kernels_tl")
    G.publishes_closure_globals(other.factory)()
    assert other.tl == "a module-level tl"  # an existing global is never overwritten
