# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Frozen-prefix skip when the model writes into the hidden states between layers: Qwen3-VL's DeepStack
(transformers 4.57) adds visual features in place after the first decoder layers. The cut on layer k-1's
output must stay writable in place (not a leaf that requires grad), and loss and gradients must match the
skip turned off. CPU, a toy stack on every transformers plus a tiny random Qwen3-VL where it exists."""

import importlib.util
import inspect
import pathlib
from functools import partial

import pytest
import torch
import torch.nn as nn
import torch.utils.checkpoint

transformers = pytest.importorskip("transformers")
modeling_layers = pytest.importorskip("transformers.modeling_layers")

_PATH = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_frozen_prefix.py"
_spec = importlib.util.spec_from_file_location("_unsloth_frozen_prefix_inplace_under_test", _PATH)
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)


class _Layer(modeling_layers.GradientCheckpointingLayer):
    def __init__(self, trainable, reentrant):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.linear.requires_grad_(trainable)
        self.gradient_checkpointing = True
        self._gradient_checkpointing_func = partial(
            torch.utils.checkpoint.checkpoint, use_reentrant = reentrant
        )

    def forward(self, x):
        return x + torch.tanh(self.linear(x))


class _Model(nn.Module):
    def __init__(self, reentrant):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(16, 8)
        self.embed.requires_grad_(False)
        self.layers = nn.ModuleList(
            [_Layer(False, reentrant), _Layer(False, reentrant), _Layer(True, reentrant)]
        )

    def forward(self, ids):
        h = self.embed(ids)
        if torch.is_grad_enabled():
            h.requires_grad_(True)  # enable_input_require_grads
        for index, layer in enumerate(self.layers):
            h = layer(h)
            if index < 2:
                h[:, 0] = h[:, 0] + 1.0  # DeepStack-style in-place write between layers
        return h.square().mean()


def _toy_step(install, reentrant):
    model = _Model(reentrant).train()
    if install:
        fp.install_frozen_prefix_skip(model)
    loss = model(torch.arange(12).view(2, 6))
    loss.backward()
    return loss.detach(), [p.grad.clone() for p in model.parameters() if p.requires_grad]


def test_frozen_prefix_module_is_this_tree():
    assert pathlib.Path(fp.__file__).resolve() == _PATH


def test_cut_is_exact_and_not_a_leaf():
    hidden = torch.tensor([[-0.0, 0.0, 1.5, float("nan"), float("inf")]])
    expect = hidden.clone()
    cut = fp._leaf(hidden)
    assert cut.requires_grad and not cut.is_leaf
    assert torch.equal(cut.isnan(), expect.isnan())
    finite = ~expect.isnan()
    assert torch.equal(cut.detach()[finite], expect[finite])
    assert torch.equal(torch.signbit(cut.detach()), torch.signbit(expect))


@pytest.mark.parametrize("reentrant", [True, False])
def test_inplace_write_after_the_cut(monkeypatch, reentrant):
    monkeypatch.setenv("UNSLOTH_FROZEN_PREFIX_SKIP", "1")
    ref_loss, ref_grads = _toy_step(install = False, reentrant = reentrant)
    before = fp.FROZEN_PREFIX_STATS["forwards"]
    loss, grads = _toy_step(install = True, reentrant = reentrant)
    assert fp.FROZEN_PREFIX_STATS["forwards"] == before + 1
    assert torch.equal(loss, ref_loss)
    for grad, ref in zip(grads, ref_grads):
        torch.testing.assert_close(grad, ref, rtol = 0, atol = 0)


def _qwen3_vl():
    config_cls = getattr(transformers, "Qwen3VLConfig", None)
    model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if config_cls is None or model_cls is None:
        pytest.skip("this transformers has no Qwen3-VL")
    config = config_cls(
        text_config = dict(
            vocab_size = 256,
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 4,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            head_dim = 8,
            max_position_embeddings = 64,
            rope_scaling = {"rope_type": "default", "mrope_section": [1, 1, 2]},
        ),
        vision_config = dict(
            depth = 4,
            hidden_size = 32,
            intermediate_size = 64,
            num_heads = 4,
            out_hidden_size = 32,
            patch_size = 2,
            temporal_patch_size = 2,
            spatial_merge_size = 2,
            deepstack_visual_indexes = [0, 1, 2],
            num_position_embeddings = 16,
        ),
        image_token_id = 10,
        video_token_id = 11,
        vision_start_token_id = 12,
        vision_end_token_id = 13,
    )
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    return model_cls, config


@pytest.mark.parametrize("reentrant", [True, False])
def test_qwen3_vl_deepstack_image_batch(monkeypatch, reentrant):
    model_cls, config = _qwen3_vl()
    monkeypatch.setenv("UNSLOTH_FROZEN_PREFIX_SKIP", "1")

    def step(install):
        torch.manual_seed(0)
        model = model_cls(config)
        model.requires_grad_(False)
        for index in (2, 3):
            model.model.language_model.layers[index].requires_grad_(True)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs = {"use_reentrant": reentrant}
        )
        model.enable_input_require_grads()
        model.train()
        if install:
            fp.install_frozen_prefix_skip(model)
        ids = torch.tensor([[1, 12, 10, 10, 10, 10, 13, 2, 3, 4]])
        torch.manual_seed(1)
        pixels = torch.randn(16, 24)
        inputs = dict(
            input_ids = ids, pixel_values = pixels, image_grid_thw = torch.tensor([[1, 4, 4]]), labels = ids
        )
        if "mm_token_type_ids" in inspect.signature(model.model.forward).parameters:
            inputs["mm_token_type_ids"] = (ids == 10).long()  # transformers 5.x M-RoPE
        loss = model(**inputs).loss
        loss.backward()
        return loss.detach(), [p.grad.clone() for p in model.parameters() if p.requires_grad]

    ref_loss, ref_grads = step(install = False)
    before = fp.FROZEN_PREFIX_STATS["forwards"]
    loss, grads = step(install = True)
    assert fp.FROZEN_PREFIX_STATS["forwards"] == before + 1
    assert torch.equal(loss, ref_loss)
    for grad, ref in zip(grads, ref_grads):
        torch.testing.assert_close(grad, ref, rtol = 0, atol = 0)
