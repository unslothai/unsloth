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
@pytest.mark.parametrize("version, join_in_rotate", [("3.3.1", False), ("3.7.1", True)])
def test_module_defines_the_variant_for_its_triton(monkeypatch, version, join_in_rotate):
    import triton

    monkeypatch.setattr(triton, "__version__", version)
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
