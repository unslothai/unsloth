# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Block CUDA graphs: parameter gradients leave the graph pool even when requires_grad changes.

TRL's DPOTrainer scores the reference with PEFT's set_adapter("ref") (or disable_adapter), which
freezes the trained adapter. When the first graphed block call was that no-grad reference forward,
only the "ref" weights got the copy-out hook, so the trained adapter's .grad stayed in the graph
pool and clip_grad_norm read memory the next replay had overwritten ("accessing gradient tensor
output of CUDAGraphs that has been overwritten by a subsequent run")."""

import importlib.util
import os
import sys

import pytest
import torch

_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unsloth", "models"
)


def _load():
    # By file path: importing the unsloth package needs an accelerator.
    spec = importlib.util.spec_from_file_location(
        "_ublock_compile_cg_adapter", os.path.join(_DIR, "_block_compile.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bc = _load()

# Replayed-backward stand-in: one reused weight grad buffer per shape, like the cudagraph pool.
_POOL = {}


class _PoolMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return x @ weight.t()

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        g_w = (grad.reshape(-1, grad.shape[-1]).t() @ x.reshape(-1, x.shape[-1])).contiguous()
        buf = _POOL.setdefault(tuple(g_w.shape), torch.empty_like(g_w))
        buf.copy_(g_w)
        return grad @ weight, buf.detach()


@pytest.fixture(autouse = True)
def _block_env(monkeypatch):
    for key in ("UNSLOTH_BLOCK_COMPILE", "UNSLOTH_COMPILE_DISABLE", "UNSLOTH_FORCE_FLOAT32"):
        monkeypatch.delenv(key, raising = False)
    monkeypatch.setattr(bc, "_CUDAGRAPH_STATE", bc._new_cudagraph_state())
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    _POOL.clear()


class AdapterDecoderLayer(torch.nn.Module):
    """Two adapters like PEFT LoRA: only the active one runs; switching flips requires_grad."""

    def __init__(self):
        super().__init__()
        self.adapters = torch.nn.ParameterDict(
            {"default": torch.nn.Parameter(torch.randn(8, 8) / 8)}
        )
        self.active = "default"

    def set_adapter(self, name):
        self.active = name
        for key, weight in self.adapters.items():
            weight.requires_grad_(key == name)

    def forward(self, hidden_states, **kwargs):
        return torch.sin(_PoolMatmul.apply(hidden_states, self.adapters[self.active]))


bc._wrap(AdapterDecoderLayer, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})


def _layers(n = 3):
    torch.manual_seed(0)
    return [AdapterDecoderLayer().train() for _ in range(n)]


def _forward(layers, x):
    y = x
    for layer in layers:
        y = layer(y)
    return y


def _reference(weights, x):
    refs = [torch.nn.Parameter(w.detach().clone()) for w in weights]
    y = x
    for w in refs:
        y = torch.sin(y @ w.t())
    y.square().mean().backward()
    return [w.grad for w in refs]


def _assert_out_of_pool(weights, want):
    ptrs = {buf.data_ptr() for buf in _POOL.values()}
    for weight, grad in zip(weights, want):
        assert weight.grad is not None
        assert weight.grad.data_ptr() not in ptrs
        torch.testing.assert_close(weight.grad, grad)


def _add_ref_adapter(layers):
    for layer in layers:
        layer.adapters["ref"] = torch.nn.Parameter(layer.adapters["default"].detach().clone())
        layer.set_adapter("default")


def test_reference_forward_first_still_copies_trained_adapter_grads():
    """DPO order: the first graphed call is the reference forward under set_adapter("ref")."""
    layers = _layers()
    _add_ref_adapter(layers)
    x = torch.randn(2, 4, 8)
    for layer in layers:
        layer.set_adapter("ref")
    with torch.no_grad():
        _forward(layers, x)
    for layer in layers:
        layer.set_adapter("default")
    _forward(layers, x).square().mean().backward()
    weights = [layer.adapters["default"] for layer in layers]
    _assert_out_of_pool(weights, _reference(weights, x))


def test_disabled_adapter_forward_first_still_copies_grads():
    """GRPO / KTO / DPO without a "ref" adapter: disable_adapter freezes the adapter instead."""
    layers = _layers()
    x = torch.randn(2, 4, 8)
    for layer in layers:
        layer.adapters["default"].requires_grad_(False)
    with torch.no_grad():
        _forward(layers, x)
    for layer in layers:
        layer.adapters["default"].requires_grad_(True)
    _forward(layers, x).square().mean().backward()
    weights = [layer.adapters["default"] for layer in layers]
    _assert_out_of_pool(weights, _reference(weights, x))


def test_adapter_added_after_graphed_training_is_copied():
    """A parameter registered after the first graphed step (a new adapter) gets the hook too."""
    layers = _layers()
    x = torch.randn(2, 4, 8)
    _forward(layers, x).square().mean().backward()
    for layer in layers:
        layer.adapters["second"] = torch.nn.Parameter(torch.randn(8, 8) / 8)
        layer.set_adapter("second")
    _POOL.clear()
    _forward(layers, x).square().mean().backward()
    weights = [layer.adapters["second"] for layer in layers]
    _assert_out_of_pool(weights, _reference(weights, x))


def test_each_parameter_is_hooked_once():
    layers = _layers()
    x = torch.randn(2, 4, 8)
    for _ in range(3):
        _forward(layers, x).square().mean().backward()
    for layer in layers:
        assert len(layer.adapters["default"]._backward_hooks) == 1
