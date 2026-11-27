# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A frozen vision tower runs no backward, and trainable grads are unchanged."""

import inspect

import pytest

transformers = pytest.importorskip("transformers")

import unsloth  # noqa: E402,F401  applies patch_enable_input_require_grads
import torch  # noqa: E402
from transformers import PreTrainedModel  # noqa: E402

try:
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
except ImportError:
    pytest.skip("transformers has no Qwen2.5-VL", allow_module_level = True)

if "for module in self.modules()" not in inspect.getsource(
    PreTrainedModel.enable_input_require_grads
):
    pytest.skip("this transformers hooks only the text embeddings", allow_module_level = True)

IMAGE_TOKEN = 7
GRID = (1, 4, 4)  # 4 image tokens after the 2 x 2 merge


def _model(train_tower = False):
    torch.manual_seed(0)
    text = dict(
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = 64,
        rope_scaling = {"type": "default", "rope_type": "default", "mrope_section": [2, 1, 1]},
    )
    config = Qwen2_5_VLConfig(
        text_config = text,
        vision_config = dict(
            depth = 2,
            hidden_size = 16,
            intermediate_size = 32,
            num_heads = 2,
            out_hidden_size = 32,
            patch_size = 2,
            spatial_merge_size = 2,
            temporal_patch_size = 2,
            window_size = 4,
            fullatt_block_indexes = [1],
        ),
        image_token_id = IMAGE_TOKEN,
        vision_start_token_id = 5,
        vision_end_token_id = 6,
        video_token_id = 8,
        **text,
    )
    model = Qwen2_5_VLForConditionalGeneration(config).float()
    model.requires_grad_(False)
    trainable = [model.model.language_model.layers[0].self_attn.q_proj.weight]
    if train_tower:
        trainable.append(model.model.visual.blocks[0].mlp.up_proj.weight)
    for p in trainable:
        p.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    model.enable_input_require_grads()
    model.train()
    return model, trainable


def _inputs():
    g = torch.Generator().manual_seed(1)
    ids = torch.tensor([[1, 5] + [IMAGE_TOKEN] * 4 + [6, 2, 3, 4]])
    patches = GRID[0] * GRID[1] * GRID[2]
    pixel_values = torch.randn(patches, 3 * 2 * 2 * 2, generator = g)
    return dict(
        input_ids = ids,
        attention_mask = torch.ones_like(ids),
        pixel_values = pixel_values,
        image_grid_thw = torch.tensor([GRID]),
        labels = ids,
    )


def _step(
    model,
    trainable,
    force_tower_grad = False,
):
    visual = model.model.visual
    state = {"backward": 0, "out_requires_grad": None}

    def count(module, args, kwargs):
        x = args[0] if args else kwargs["hidden_states"]
        if x.requires_grad:
            x.register_hook(lambda g: state.__setitem__("backward", state["backward"] + 1))

    def force(module, args, output):
        output.requires_grad_(True)  # what every sub-model embedding hook did before

    def out(module, args, output):
        tensor = output if isinstance(output, torch.Tensor) else output.pooler_output
        state["out_requires_grad"] = tensor.requires_grad

    handles = [
        visual.blocks[0].register_forward_pre_hook(count, with_kwargs = True),
        visual.register_forward_hook(out),
    ]
    if force_tower_grad:
        handles.append(visual.patch_embed.register_forward_hook(force))
    for p in trainable:
        p.grad = None
    loss = model(**_inputs()).loss
    loss.backward()
    for h in handles:
        h.remove()
    return loss.detach(), [p.grad.clone() for p in trainable], state


def test_frozen_tower_runs_no_backward_and_grads_match():
    model, trainable = _model()
    loss_ref, grads_ref, ref = _step(model, trainable, force_tower_grad = True)
    loss, grads, state = _step(model, trainable)
    assert ref["backward"] >= 1 and ref["out_requires_grad"]
    assert state["backward"] == 0 and state["out_requires_grad"] is False
    assert torch.equal(loss, loss_ref)
    assert grads[0].abs().sum() > 0
    for g, g_ref in zip(grads, grads_ref):
        assert torch.equal(g, g_ref)


def test_tower_with_a_trainable_weight_keeps_its_backward():
    model, trainable = _model(train_tower = True)
    _, grads, state = _step(model, trainable)
    assert state["backward"] >= 1 and state["out_requires_grad"]
    assert all(g.abs().sum() > 0 for g in grads)


def test_trainability_is_read_at_forward_time():
    model, trainable = _model()
    tower_weight = model.model.visual.blocks[0].mlp.up_proj.weight
    tower_weight.requires_grad_(True)  # unfrozen after enable_input_require_grads
    _, grads, state = _step(model, trainable + [tower_weight])
    assert state["backward"] >= 1 and grads[1].abs().sum() > 0
    tower_weight.requires_grad_(False)
    tower_weight.grad = None
    _, _, state = _step(model, trainable)
    assert state["backward"] == 0


def test_text_embedding_hook_is_unchanged():
    model, _ = _model()
    names = [
        getattr(h, "__qualname__", "") for h in model.get_input_embeddings()._forward_hooks.values()
    ]
    assert names and all("_sub_model_hook" not in n for n in names)
    tower = [
        getattr(h, "__qualname__", "")
        for h in model.model.visual.patch_embed._forward_hooks.values()
    ]
    assert tower and all("_sub_model_hook" in n for n in tower)
    assert all(
        getattr(h, "__name__", None) == "make_inputs_require_grads"
        for h in model.model.visual.patch_embed._forward_hooks.values()
    )


def test_compiled_hook_keeps_the_flag_for_a_tower_unfrozen_later():
    model, trainable = _model()
    _, _, state = _step(model, trainable)
    assert state["backward"] == 0
    compiled = torch.compile(model.model.visual.patch_embed, backend = "eager", fullgraph = True)
    x = _inputs()["pixel_values"]
    assert compiled(x).requires_grad  # traced while frozen: kept, so a later unfreeze still trains
    model.model.visual.blocks[0].mlp.up_proj.weight.requires_grad_(True)
    assert compiled(x).requires_grad


def test_keep_frozen_towers_env_is_read_per_call(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FROZEN_TOWER_INPUT_GRAD", "1")
    model, _ = _model()
    tower = [
        getattr(h, "__qualname__", "")
        for h in model.model.visual.patch_embed._forward_hooks.values()
    ]
    assert tower and all("_sub_model_hook" not in n for n in tower)
