# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Block CUDA graphs (opt-in) must copy block outputs and input gradients out of the graph pool.

Under Unsloth checkpointing every block call starts a new cudagraph tree generation, so a
tensor the checkpoint keeps (the saved block input, the gradient handed to the previous
block) would be overwritten by the next replay.
"""

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
        "_ublock_compile_cg", os.path.join(_DIR, "_block_compile.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bc = _load()


@pytest.fixture(autouse = True)
def _block_env(monkeypatch):
    # The CPU rest shard runs with UNSLOTH_COMPILE_DISABLE=1, which makes every wrapped block take
    # the eager path, so the graph-cap test would see no compiled call at all.
    for key in ("UNSLOTH_BLOCK_COMPILE", "UNSLOTH_COMPILE_DISABLE", "UNSLOTH_FORCE_FLOAT32"):
        monkeypatch.delenv(key, raising = False)


def test_clone_grad_is_identity_with_copied_gradient():
    x = torch.randn(3, 4, requires_grad = True)
    seen = {}

    def keep(grad):
        seen["ptr"] = grad.data_ptr()
        return grad

    y = bc._CloneGrad.apply(x)
    y.register_hook(keep)
    assert torch.equal(y, x)
    (y * 2).sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 2.0))
    assert x.grad.data_ptr() != seen["ptr"]


def test_clone_outputs_copies_every_tensor():
    a, b = torch.randn(2), torch.randn(3)
    out = bc._clone_outputs((a, None, b))
    assert out[1] is None
    assert torch.equal(out[0], a) and out[0].data_ptr() != a.data_ptr()
    assert torch.equal(out[2], b) and out[2].data_ptr() != b.data_ptr()
    single = bc._clone_outputs(a)
    assert torch.equal(single, a) and single.data_ptr() != a.data_ptr()


def test_cudagraphs_opt_in_flag(monkeypatch):
    try:
        monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS", "1")
        assert _load()._COMPILE_OPTIONS["triton.cudagraphs"] is True
        monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS", "0")
        assert _load()._COMPILE_OPTIONS["triton.cudagraphs"] is False
    finally:
        monkeypatch.delenv("UNSLOTH_BLOCK_CUDAGRAPHS", raising = False)
        _load()


def _cap_layer_step(monkeypatch):
    calls = []

    def fake_compile(fn, **kwargs):
        tag = "graphs" if kwargs.get("options", {}).get("triton.cudagraphs") else "plain"

        def run(*args, **kw):
            calls.append(tag)
            return fn(*args, **kw)

        return run

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(bc, "_CUDAGRAPH_MAX_SHAPES", 2)

    class CapDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states)

    bc._wrap(CapDecoderLayer, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    layer = CapDecoderLayer().train()

    def step(L):
        calls.clear()
        x = torch.randn(2, L, 8, requires_grad = True)
        layer(x).sum().backward()
        assert x.grad is not None and x.grad.shape == x.shape
        return set(calls)

    return step


def test_new_shapes_past_the_cap_skip_graphs(monkeypatch):
    """Variable lengths recorded a graph set per new length: 3.5x slower steps and a pool that
    grew every step on an L4. One shape past the cap runs every later call without graphs."""
    step = _cap_layer_step(monkeypatch)
    assert step(16) == {"graphs"}
    assert step(24) == {"graphs"}
    assert step(16) == {"graphs"}
    assert step(32) == {"plain"}  # third shape: over the cap
    assert step(16) == {"plain"}  # graphs stay off, also for admitted shapes
    assert step(40) == {"plain"}


def test_cap_reads_the_environment(monkeypatch):
    try:
        monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS_MAX_SHAPES", "3")
        assert _load()._CUDAGRAPH_MAX_SHAPES == 3
        monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS_MAX_SHAPES", "junk")
        assert _load()._CUDAGRAPH_MAX_SHAPES == 8
    finally:
        monkeypatch.delenv("UNSLOTH_BLOCK_CUDAGRAPHS_MAX_SHAPES", raising = False)
        _load()


def test_compile_disable_still_bypasses_the_block(monkeypatch):
    # The kill switch is read per call, so it also turns off a block wrapped before it was set.
    step = _cap_layer_step(monkeypatch)
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "1")
    assert step(16) == set()


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
        return (
            grad @ weight,
            buf.detach(),
        )


def _pool_layers(monkeypatch, graphs):
    _POOL.clear()
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)

    class PoolDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 8) / 8)

        def forward(self, hidden_states, **kwargs):
            return torch.sin(_PoolMatmul.apply(hidden_states, self.weight))

    bc._wrap(PoolDecoderLayer, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": graphs})
    torch.manual_seed(0)
    layers = [PoolDecoderLayer().train() for _ in range(3)]
    refs = [torch.nn.Parameter(l.weight.detach().clone()) for l in layers]
    return layers, refs


def _reference_grads(refs, xs):
    for x in xs:
        y = x
        for w in refs:
            y = torch.sin(y @ w.t())
        y.square().mean().backward()
    return [w.grad for w in refs]


@pytest.mark.parametrize("micro_steps", [1, 2])
def test_param_grads_are_copied_out_of_the_graph_pool(monkeypatch, micro_steps):
    """Reentrant checkpointing: a .grad left in the replay buffer is overwritten by the next replay."""
    from torch.utils.checkpoint import checkpoint

    layers, refs = _pool_layers(monkeypatch, graphs = True)
    xs = [torch.randn(2, 4, 8) for _ in range(micro_steps)]
    for x in xs:
        y = x.clone().requires_grad_()
        for layer in layers:
            y = checkpoint(layer, y, use_reentrant = True)
        y.square().mean().backward()
    want = _reference_grads(refs, xs)
    ptrs = {buf.data_ptr() for buf in _POOL.values()}
    for layer, grad in zip(layers, want):
        assert layer.weight.grad.data_ptr() not in ptrs
        torch.testing.assert_close(layer.weight.grad, grad)


def test_preallocated_grads_are_accumulated_in_place(monkeypatch):
    # An existing .grad (zero_grad(set_to_none=False), DDP bucket views) is not replaced.
    layers, refs = _pool_layers(monkeypatch, graphs = True)
    for layer in layers:
        layer.weight.grad = torch.zeros_like(layer.weight)
    owned = [layer.weight.grad.data_ptr() for layer in layers]
    x = torch.randn(2, 4, 8)
    y = x
    for layer in layers:
        y = layer(y)
    y.square().mean().backward()
    want = _reference_grads(refs, [x])
    for layer, ptr, grad in zip(layers, owned, want):
        assert layer.weight.grad.data_ptr() == ptr
        torch.testing.assert_close(layer.weight.grad, grad)


def test_param_grads_hooked_only_with_graphs(monkeypatch):
    layers, _ = _pool_layers(monkeypatch, graphs = False)
    x = torch.randn(2, 4, 8)
    y = x
    for layer in layers:
        y = layer(y)
    y.sum().backward()
    assert not any(l.__dict__.get("_unsloth_block_grad_hooks") for l in layers)


def test_nonreentrant_checkpoint_runs_without_graphs(monkeypatch):
    """Non-reentrant checkpoint forward and recompute skip graphs; reentrant keeps them."""
    if not hasattr(torch._C._autograd, "_top_saved_tensors_default_hooks"):
        pytest.skip("torch < 2.8 cannot read the saved tensor hooks stack")
    from torch.utils.checkpoint import checkpoint

    seen = []

    def fake_compile(fn, **kwargs):
        tag = "graphs" if kwargs.get("options", {}).get("triton.cudagraphs") else "plain"

        def inner(*args, **kw):
            seen.append(tag)
            return fn(*args, **kw)

        return inner

    monkeypatch.setattr(torch, "compile", fake_compile)

    class CkptDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states)

    bc._wrap(CkptDecoderLayer, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    layer = CkptDecoderLayer().train()
    for reentrant, want in ((False, {"plain"}), (True, {"graphs"})):
        seen.clear()
        x = torch.randn(2, 16, 8, requires_grad = True)
        checkpoint(layer, x, use_reentrant = reentrant).sum().backward()
        assert len(seen) == 2 and set(seen) == want
        assert x.grad is not None
    assert not bc._in_nonreentrant_checkpoint()
