# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Frozen-prefix skip on a looped stack (recurrent-depth models such as Ouro run `self.layers` several
times per forward). The second pass through the frozen prefix is fed by the trainable upper layers, so
it must keep its graph: the upper adapters' gradients must match the skip turned off. CPU, tiny Llama."""

import importlib.util
import pathlib
import types

import pytest
import torch

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
pytest.importorskip("transformers.modeling_layers")

_PATH = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_frozen_prefix.py"
_spec = importlib.util.spec_from_file_location("_unsloth_frozen_prefix_looped_under_test", _PATH)
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)

LAYERS = 4


def _looped_forward(loops):
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        cache_position = None,
        **kwargs,
    ):
        hidden = self.embed_tokens(input_ids)
        positions = torch.arange(hidden.shape[1], device = hidden.device)[None]
        position_embeddings = self.rotary_emb(hidden, positions)
        for _ in range(loops):
            for layer in self.layers:
                out = layer(
                    hidden,
                    attention_mask = None,
                    position_ids = positions,
                    position_embeddings = position_embeddings,
                )
                hidden = out[0] if isinstance(out, tuple) else out
        return transformers.modeling_outputs.BaseModelOutputWithPast(
            last_hidden_state = self.norm(hidden)
        )

    return forward


def _model(loops, install):
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = LAYERS,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        max_position_embeddings = 64,
    )
    config._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(config)
    model.model.forward = types.MethodType(_looped_forward(loops), model.model)
    lora = peft.LoraConfig(
        r = 4,
        lora_alpha = 8,
        target_modules = ["q_proj", "v_proj", "down_proj"],
        layers_to_transform = [2, 3],
        init_lora_weights = False,
    )
    model = peft.get_peft_model(model, lora)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    model.enable_input_require_grads()
    model.train()
    if install:
        fp.install_frozen_prefix_skip(model)
    return model


def _step(model):
    ids = torch.randint(0, 128, (2, 16), generator = torch.Generator().manual_seed(1))
    model.zero_grad(set_to_none = True)
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    return loss.detach(), grads


def test_frozen_prefix_module_is_this_tree():
    assert pathlib.Path(fp.__file__).resolve() == _PATH


@pytest.mark.parametrize("loops", [1, 2, 3])
def test_looped_stack_keeps_upper_adapter_gradients(monkeypatch, loops):
    monkeypatch.setenv("UNSLOTH_FROZEN_PREFIX_SKIP", "1")
    ref_loss, ref_grads = _step(_model(loops, install = False))
    before = fp.FROZEN_PREFIX_STATS["forwards"]
    model = _model(loops, install = True)
    loss, grads = _step(model)
    assert fp.FROZEN_PREFIX_STATS["forwards"] == before + 1  # the skip engaged
    assert torch.equal(loss, ref_loss)
    assert grads.keys() == ref_grads.keys() and grads
    for name, grad in grads.items():
        torch.testing.assert_close(grad, ref_grads[name], rtol = 1e-5, atol = 1e-7, msg = name)
    # The forward left the layers as it found them.
    layers = model.base_model.model.model.layers
    assert all(layer.gradient_checkpointing for layer in layers)
    assert not any(layer._forward_pre_hooks or layer._forward_hooks for layer in layers)
