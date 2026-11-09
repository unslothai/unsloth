# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every Triton launch in the VAE kernels and the MOSS QK-norm RoPE runs under the input's own card: Triton launches on
the CURRENT device, so a tensor on another card fails the launch and the fused path drops to stock. CPU only."""

from __future__ import annotations

import ast
import contextlib
import inspect
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_moss_fused as M  # noqa: E402
from core.inference import diffusion_vae_fused as V  # noqa: E402


def _launches(module) -> list:
    """(line, inside a ``with _on_card(...)``) for every ``kernel[grid](...)`` call."""
    tree = ast.parse(inspect.getsource(module))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    out = []
    for node in ast.walk(tree):
        # kernel[grid](...): a grid subscript, not an index like ``to_out[0](o)``
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Subscript)
            and not isinstance(node.func.slice, ast.Constant)
        ):
            continue
        scoped, p = False, parents.get(node)
        while p is not None and not isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if isinstance(p, ast.With) and any(
                isinstance(i.context_expr, ast.Call)
                and getattr(i.context_expr.func, "id", None) == "_on_card"
                for i in p.items
            ):
                scoped = True
            p = parents.get(p)
        out.append((node.lineno, scoped))
    return out


@pytest.mark.parametrize("module, count", [(V, 9), (M, 1)])
def test_every_launch_is_scoped_to_its_card(module, count):
    launches = _launches(module)
    assert len(launches) == count
    assert all(scoped for _line, scoped in launches), launches


def _cuda_like(index: int):
    return types.SimpleNamespace(is_cuda = True, device = torch.device("cuda", index))


@pytest.fixture(autouse = True)
def _thread_on_card_0(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


def test_scope_is_the_tensor_card(monkeypatch):
    monkeypatch.delenv(V.DEVICE_SCOPE_ENV, raising = False)
    scope = V._on_card(_cuda_like(1))
    assert isinstance(scope, torch.cuda.device) and scope.idx == 1
    scope = M._on_card(_cuda_like(3))
    assert isinstance(scope, torch.cuda.device) and scope.idx == 3


def test_the_current_card_needs_no_switch(monkeypatch):
    monkeypatch.delenv(V.DEVICE_SCOPE_ENV, raising = False)
    assert isinstance(V._on_card(_cuda_like(0)), contextlib.nullcontext)


@pytest.mark.parametrize("off", ["0", "off", "false"])
def test_kill_switch_and_cpu_launch_unscoped(monkeypatch, off):
    monkeypatch.setenv(V.DEVICE_SCOPE_ENV, off)
    assert isinstance(V._on_card(_cuda_like(1)), contextlib.nullcontext)
    monkeypatch.delenv(V.DEVICE_SCOPE_ENV)
    assert isinstance(V._on_card(torch.zeros(1)), contextlib.nullcontext)


def test_traced_launch_left_to_inductor(monkeypatch):
    """Under torch.compile Inductor guards the device itself; a torch.cuda.device here would break the graph."""
    monkeypatch.delenv(V.DEVICE_SCOPE_ENV, raising = False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert isinstance(V._on_card(_cuda_like(1)), contextlib.nullcontext)
