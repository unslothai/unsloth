# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Frozen-prefix skip inside the caller's torch.compile: its hooks register and remove module hooks per
forward, which Dynamo cannot trace (`setattr() on unsupported type` under fullgraph=True). While compiling
the skip must stand aside, so a fullgraph training step compiles and matches the skip turned off. CPU."""

import importlib.util
import pathlib
from functools import partial

import pytest
import torch
import torch.nn as nn
import torch.utils.checkpoint

pytest.importorskip("transformers")
modeling_layers = pytest.importorskip("transformers.modeling_layers")

_PATH = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_frozen_prefix.py"
_spec = importlib.util.spec_from_file_location("_unsloth_frozen_prefix_compile_under_test", _PATH)
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)


class _Layer(modeling_layers.GradientCheckpointingLayer):
    def __init__(self, trainable):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.7), requires_grad = trainable)
        self.gradient_checkpointing = True
        self._gradient_checkpointing_func = partial(
            torch.utils.checkpoint.checkpoint, use_reentrant = False
        )

    def forward(self, x):
        return torch.sin(self.weight * x)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(False), _Layer(False), _Layer(True)])

    def forward(self, x):
        h = x + 1
        for layer in self.layers:
            h = layer(h)
        return h


def _grad(install, warm):
    torch._dynamo.reset()
    model = _Model().train()
    if install:
        fp.install_frozen_prefix_skip(model)
    x = torch.arange(1.0, 10.0)
    if warm:
        model(x).sum().backward()
        model.zero_grad(set_to_none = True)
    compiled = torch.compile(model, backend = "eager", fullgraph = True, dynamic = True)
    compiled(x).sum().backward()
    return model.layers[2].weight.grad.clone(), model


def test_frozen_prefix_module_is_this_tree():
    assert pathlib.Path(fp.__file__).resolve() == _PATH


def test_eager_forward_engages():
    model = _Model().train()
    fp.install_frozen_prefix_skip(model)
    before = fp.FROZEN_PREFIX_STATS["forwards"]
    model(torch.arange(1.0, 10.0)).sum().backward()
    assert fp.FROZEN_PREFIX_STATS["forwards"] == before + 1


@pytest.mark.parametrize("warm", [False, True])
def test_fullgraph_outer_compile(monkeypatch, warm):
    monkeypatch.setenv("UNSLOTH_FROZEN_PREFIX_SKIP", "1")
    ref, _ = _grad(install = False, warm = warm)
    grad, model = _grad(install = True, warm = warm)
    torch.testing.assert_close(grad, ref, rtol = 0, atol = 0)
    assert all(layer.gradient_checkpointing for layer in model.layers)
    assert not any(layer._forward_pre_hooks or layer._forward_hooks for layer in model.layers)
