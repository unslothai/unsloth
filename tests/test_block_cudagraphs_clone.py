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
