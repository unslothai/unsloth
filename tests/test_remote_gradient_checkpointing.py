# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Remote decoder loops that never call a checkpoint function (Kimi-K2.7's DeepSeek-V3 port)
must still checkpoint once gradient checkpointing is enabled.

The backbone below mirrors that file: it declares `gradient_checkpointing` and support for it,
but its loop calls every layer directly, so enabling the flag used to change nothing.
"""

import sys

import pytest
import torch
import torch.nn as nn

transformers = pytest.importorskip("transformers")
from transformers import PretrainedConfig, PreTrainedModel

from unsloth.models.loader_utils import install_remote_gradient_checkpointing

REMOTE = "transformers_modules.some_repo.modeling_x"


class RemoteLayer(nn.Module):
    calls = 0

    def __init__(self, hidden):
        super().__init__()
        self.up = nn.Linear(hidden, 4 * hidden)
        self.down = nn.Linear(4 * hidden, hidden)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        past_key_value = None,
        use_cache = False,
    ):
        type(self).calls += 1
        return (hidden_states + self.down(torch.relu(self.up(hidden_states))),)


class RemoteBackbone(PreTrainedModel):
    config_class = PretrainedConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config,
        layer_cls = RemoteLayer,
    ):
        super().__init__(config)
        self.layers = nn.ModuleList(layer_cls(16) for _ in range(3))
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        past_key_values = None,
    ):
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask = None, past_key_value = past_key_values
            )[0]
        return hidden_states


for _cls in (RemoteLayer, RemoteBackbone):
    _cls.__module__ = REMOTE
# transformers reads the defining module's source for some class checks.
sys.modules.setdefault(REMOTE, sys.modules[__name__])


def _build(layer_cls = RemoteLayer):
    torch.manual_seed(0)
    return RemoteBackbone(PretrainedConfig(), layer_cls = layer_cls)


def _step(model, **kwargs):
    model.train()
    x = torch.randn(2, 8, 16, requires_grad = True)
    before = type(model.layers[0]).calls
    model(x, **kwargs).square().sum().backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    return type(model.layers[0]).calls - before, grads


def test_the_flag_really_checkpoints_a_remote_loop():
    reference = _build()
    install_remote_gradient_checkpointing(reference, verbose = False)
    calls_off, grads_off = _step(reference)
    assert calls_off == 3

    model = _build()
    install_remote_gradient_checkpointing(model, verbose = False)
    assert RemoteLayer.__dict__["forward"]._unsloth_manual_checkpoint
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    calls_on, grads_on = _step(model)
    assert calls_on == 6  # every layer recomputed in the backward
    assert all(torch.allclose(a, b) for a, b in zip(grads_off, grads_on))


def test_the_installed_checkpoint_function_is_used():
    model = _build()
    install_remote_gradient_checkpointing(model, verbose = False)
    model.gradient_checkpointing_enable()
    seen = []
    original = model._gradient_checkpointing_func

    def spy(function, *args):
        seen.append(len(args))
        return original(function, *args)

    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module._gradient_checkpointing_func = spy
    _step(model)
    assert seen == [1, 1, 1]


def test_a_cache_is_never_recomputed():
    model = _build()
    install_remote_gradient_checkpointing(model, verbose = False)
    model.gradient_checkpointing_enable()
    calls, _ = _step(model, past_key_values = object())
    assert calls == 3


def test_a_loop_that_already_checkpoints_is_left_alone():
    class CheckpointingBackbone(RemoteBackbone):
        def forward(
            self,
            hidden_states,
            past_key_values = None,
        ):
            for layer in self.layers:
                hidden_states = self._gradient_checkpointing_func(layer.__call__, hidden_states)[0]
            return hidden_states

    CheckpointingBackbone.__module__ = REMOTE
    torch.manual_seed(0)
    model = CheckpointingBackbone(PretrainedConfig())
    assert install_remote_gradient_checkpointing(model, verbose = False) == []


def test_native_models_are_never_wrapped():
    class NativeLayer(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.up = nn.Linear(hidden, hidden)

        def forward(self, hidden_states, **kwargs):
            return (self.up(hidden_states),)

    class NativeBackbone(RemoteBackbone):
        pass

    native = "transformers.models.x.modeling_x"
    NativeLayer.__module__ = NativeBackbone.__module__ = native
    sys.modules.setdefault(native, sys.modules[__name__])
    model = NativeBackbone(PretrainedConfig(), layer_cls = NativeLayer)
    assert install_remote_gradient_checkpointing(model, verbose = False) == []
    assert not hasattr(NativeLayer.forward, "_unsloth_manual_checkpoint")
