# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NVIDIA Triton 3.3 (torch 2.7) cannot compile the INT8 attention rotation's ``tl.join``: that version builds the
rotation without it, every other Triton keeps the join, persisted verdicts key on the Triton version, and a fallback
to native attention names its reason."""

import ast
import importlib
import inspect
import sys
import types

import pytest

pytest.importorskip("torch")

from core.inference import diffusion_int8_attention as I  # noqa: E402
from core.inference import diffusion_int8_attention_triton as T  # noqa: E402
from core.inference import diffusion_probe_cache as P  # noqa: E402


@pytest.mark.parametrize(
    "version, rocm, join_free",
    [
        ("3.3.1", False, True),
        ("3.3.0", False, True),
        ("3.3.1", True, False),
        ("3.2.0", False, False),
        ("3.4.0", False, False),
        ("3.7.1", False, False),
        ("3.10.0", False, False),
        ("dev", False, False),
    ],
)
def test_join_free_rotation_only_on_nvidia_triton_33(monkeypatch, version, rocm, join_free):
    monkeypatch.setattr(T, "triton", types.SimpleNamespace(__version__ = version))
    monkeypatch.setattr(T, "_rocm_build", lambda: rocm)
    assert T.join_free_rotation() is join_free


def test_no_triton_keeps_the_join(monkeypatch):
    monkeypatch.setattr(T, "triton", None)
    assert T.join_free_rotation() is False
    assert T.triton_version() is None


@pytest.mark.parametrize("version", ["3.3.1", "3.2.0", "3.7.1"])
def test_triton_available_on_every_version(monkeypatch, version):
    monkeypatch.setattr(T, "triton", types.SimpleNamespace(__version__ = version))
    I.triton_available.cache_clear()
    try:
        assert I.triton_available() is True
    finally:
        I.triton_available.cache_clear()


def test_no_triton_is_unavailable(monkeypatch):
    monkeypatch.setattr(T, "triton", None)
    I.triton_available.cache_clear()
    try:
        assert I.triton_available() is False
    finally:
        I.triton_available.cache_clear()


def _rotate_bodies() -> dict:
    """{'join': source, 'join_free': source} of the two ``_rotate`` definitions, from the module's AST."""
    tree = ast.parse(inspect.getsource(T))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "not join_free_rotation()":
            for branch, body in (("join", node.body), ("join_free", node.orelse)):
                for item in body:
                    if isinstance(item, ast.FunctionDef) and item.name == "_rotate":
                        found[branch] = ast.unparse(item)
    return found


def test_rotation_variants_are_chosen_in_python():
    # A constexpr global would not do: Triton 3.2 / 3.3 leave its value out of the cache key, so the two bodies
    # would share compiled kernels. Two definitions with their own source keep them apart.
    bodies = _rotate_bodies()
    assert set(bodies) == {"join", "join_free"}
    assert "tl.join" in bodies["join"]
    assert "tl.join" not in bodies["join_free"] and "tl.where" in bodies["join_free"]
    assert bodies["join"] != bodies["join_free"]


@pytest.mark.skipif(T.triton is None, reason = "needs Triton to define the kernels")
@pytest.mark.parametrize(
    "version, hip, join_in_rotate",
    [("3.3.1", None, False), ("3.7.1", None, True), ("3.3.1", "6.4", True), ("3.7.1", "6.4", True)],
)
def test_module_defines_the_variant_for_its_triton(monkeypatch, version, hip, join_in_rotate):
    import torch
    import triton

    monkeypatch.setattr(triton, "__version__", version)
    # the build identity is pinned: the host's own (a ROCm torch reads hip) must not pick the body
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch, "__version__", "2.7.1+rocm6.4" if hip else "2.7.1+cu126")
    name = "core.inference.diffusion_int8_attention_triton"
    fresh = importlib.reload(sys.modules[name])
    try:
        assert ("tl.join" in inspect.getsource(fresh._rotate.fn)) is join_in_rotate
    finally:
        monkeypatch.undo()
        importlib.reload(sys.modules[name])


@pytest.mark.parametrize(
    "dists, expected",
    [
        ({"triton": "3.3.1"}, "3.3.1"),
        ({"triton-windows": "3.3.1.post19"}, "triton-windows==3.3.1.post19"),
        ({"pytorch-triton-rocm": "3.3.1"}, "pytorch-triton-rocm==3.3.1"),
        ({"triton-rocm": "3.7.0"}, "triton-rocm==3.7.0"),
        ({"triton-xpu": "3.6.0"}, "triton-xpu==3.6.0"),
        ({"triton": "3.4.0", "pytorch-triton-rocm": "3.3.1"}, "3.4.0"),
        ({}, None),
    ],
)
def test_verdict_key_carries_the_triton_version(monkeypatch, dists, expected):
    monkeypatch.setattr(P, "_version", lambda name: dists.get(name))
    assert P._triton_version() == expected


def test_verdict_keys_differ_across_triton_versions(monkeypatch):
    keys = []
    for version in ("3.3.1", "3.4.0"):
        monkeypatch.setattr(
            P, "_version", lambda name, v = version: v if name == "triton-windows" else None
        )
        keys.append(P._key({"triton": P._triton_version()}))
    assert keys[0] != keys[1]


def test_fingerprint_uses_the_triton_lookup():
    assert '"triton": _triton_version()' in inspect.getsource(P)


@pytest.mark.parametrize(
    "setting, available, expected",
    [
        ("cuda", True, f"{I.IMPL_ENV}=cuda"),
        ("auto", False, "Triton is not installed"),
        ("auto", True, "no Triton config for sm_90 head_dim=128"),
    ],
)
def test_triton_decline_reason(monkeypatch, setting, available, expected):
    import torch

    monkeypatch.setattr(I, "impl_setting", lambda: setting)
    monkeypatch.setattr(I, "triton_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (9, 0))
    assert I.triton_decline(0, 128) == expected


def test_no_kernel_verdict_names_both_reasons(monkeypatch):
    import torch

    monkeypatch.setattr(I, "_is_rocm", lambda: False)
    monkeypatch.setattr(I, "impl_setting", lambda: "auto")
    monkeypatch.setattr(I, "triton_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (8, 9))
    monkeypatch.setitem(I._CUDA_FAILED, (0, 128), "NVRTC not found")
    assert I._no_kernel_reason(0, 128) == "NVRTC not found; Triton: Triton is not installed"
    assert I._no_kernel_reason(0, 64) == "no kernel for this card (Triton: Triton is not installed)"


def _split_flux3(
    monkeypatch,
    cuda_fails_on: int,
    generic: bool = False,
):
    """A FLUX.3 (CUDA-only Auto family) split over cards 0 and 1 on sm89: the CUDA kernel passes its race on card 0
    and fails to load on ``cuda_fails_on``. Returns (dit, probes, served-on-card-1 thunk)."""
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch

    from core.inference import diffusion_flux3_attn as flux3

    class Processor:
        _unsloth_int8 = False
        _parallel_config = None
        _attention_backend = "native"

    class Flux3Transformer2DModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            module = self.attention = torch.nn.Module()
            module.head_dim = 128
            module.heads = 2
            module.processor = Processor()

    probes = []

    def check(device, dtype, heads, head_dim, incumbent, timed, impl):
        probes.append((device.index, impl, timed))
        if device.index == cuda_fails_on and impl == "cuda":
            return {
                "ok": False,
                "impl": impl,
                "failed": True,
                "error": "simulated CUDA module load failure",
            }
        I._VERIFIED.add((device.index, head_dim, str(dtype), impl))
        return {"ok": True, "impl": impl, "min_tokens": 0 if timed else None}

    for name in ("_VERIFIED", "_VERDICTS", "_CUDA_FAILED", "_BROKEN"):
        monkeypatch.setattr(I, name, type(getattr(I, name))())
    monkeypatch.setenv(I.IMPL_ENV, "auto")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PROBE_CACHE", "0")
    monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_ATTENTION_DISPATCH", raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (8, 9))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(flux3, "ensure_strided_processors", lambda *a, **k: None)
    monkeypatch.setattr(I, "_is_rocm", lambda: False)
    monkeypatch.setattr(I, "triton_available", lambda: True)
    monkeypatch.setattr(I, "_secondary_cards", lambda primary, transformer = None: (1,))
    monkeypatch.setattr(I, "_check", check)
    monkeypatch.setattr(I, "_run", lambda q, k, v, impl, config: impl)
    monkeypatch.setattr(I, "_cuda_attention", lambda *a, **k: "cuda")
    monkeypatch.setattr(I, "native_attention", lambda q, k, v: "native")
    dit = Flux3Transformer2DModel()
    target = SimpleNamespace(device = "cuda:0", dtype = torch.bfloat16)
    q = SimpleNamespace(
        is_cuda = True,
        device = torch.device("cuda:1"),
        dtype = torch.bfloat16,
        shape = (1, 4096, 2, 128),
        dim = lambda: 4,
        requires_grad = False,
    )
    with torch.no_grad():
        if generic:
            # the generic Auto path (no family floor): the primary's own timed race, then the secondaries
            installed = I.install_int8_attention(dit, device = "cuda:0", dtype = torch.bfloat16)
        else:
            installed = I.install_auto(SimpleNamespace(transformer = dit), target)

    def served():
        with torch.no_grad():
            return I.int8_attention_impl(q, q, q)

    return installed, dit, probes, served


def test_split_cuda_only_family_keeps_native_on_a_secondary_whose_cuda_kernel_fails(monkeypatch):
    installed, dit, probes, served = _split_flux3(monkeypatch, cuda_fails_on = 1)
    assert installed == 1 and dit.attention.processor._unsloth_int8
    # card 1 never runs an (unraced) Triton check, and its calls stay on native attention as before Triton 3.3
    assert not [p for p in probes if p[0] == 1 and p[1] == "triton"]
    assert served() == "native"


def test_split_secondary_still_runs_the_primary_pick(monkeypatch):
    installed, dit, probes, served = _split_flux3(monkeypatch, cuda_fails_on = -1)
    assert installed == 1
    assert (1, "cuda", False) in probes
    assert served() == "cuda"


def test_verdict_impl_only_declines_another_implementation_unchecked(monkeypatch):
    import torch

    monkeypatch.setattr(I, "_VERDICTS", {})
    monkeypatch.setattr(I, "implementation", lambda index, head_dim: "triton")
    monkeypatch.setattr(I, "_check", lambda *a, **k: pytest.fail("must not check"))
    found = I.verdict(
        torch.device("cuda", 1), torch.bfloat16, 2, 128, timed = False, impl_only = "cuda"
    )
    assert (
        found["ok"] is False and found["impl"] == "triton" and "cuda kernel only" in found["error"]
    )


def test_generic_auto_secondary_whose_cuda_kernel_fails_keeps_native(monkeypatch):
    installed, dit, probes, served = _split_flux3(monkeypatch, cuda_fails_on = 1, generic = True)
    assert installed == 1 and dit.attention.processor._unsloth_int8
    assert (0, "cuda", True) in probes
    # the primary raced the CUDA kernel: card 1 may not adopt an unraced Triton stand-in
    assert not [p for p in probes if p[0] == 1 and p[1] == "triton"]
    assert served() == "native"


def test_generic_auto_secondary_still_runs_the_primary_pick(monkeypatch):
    installed, dit, probes, served = _split_flux3(monkeypatch, cuda_fails_on = -1, generic = True)
    assert installed == 1
    assert (1, "cuda", False) in probes
    assert served() == "cuda"
