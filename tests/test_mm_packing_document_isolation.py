# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Packed mRoPE rows carry no mask: transformers must derive per-document isolation from the reset
position_ids. Changing only document A must leave document B's states and A's gradients from a B-only
objective untouched; where transformers cannot isolate (torch < 2.6 on 4.x), packing is refused. CPU only."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")

from unsloth.utils import mm_packing  # noqa: E402


def _tiny_qwen2_5_vl():
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

    config = Qwen2_5_VLConfig(
        text_config = dict(
            hidden_size = 32,
            intermediate_size = 64,
            vocab_size = 100,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            num_hidden_layers = 2,
            rope_scaling = dict(rope_type = "default", mrope_section = [1, 1, 2]),
        ),
        vision_config = dict(
            hidden_size = 32,
            intermediate_size = 64,
            depth = 1,
            num_heads = 4,
            out_hidden_size = 32,
            fullatt_block_indexes = [0],
        ),
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(4)
    return Qwen2_5_VLModel(config).train()


def _flip_document_a(model, rope_index):
    """B's hidden-state change and the largest A-embedding gradient of a B-only objective."""
    states, grads = [], []
    for first in ([1, 2, 3], [9, 10, 11]):
        batch = mm_packing.pack_multimodal_documents(
            [{"input_ids": torch.tensor([first])}, {"input_ids": torch.tensor([[6, 7, 8]])}],
            rope_index = rope_index,
        )
        # As the training forward runs it: Unsloth's compute_loss passes use_cache=False for packed rows.
        out = model(**batch, use_cache = False).last_hidden_state[:, 3:]
        out[..., 0].sum().backward()
        grads.append(model.language_model.embed_tokens.weight.grad[first].abs().max().item())
        model.zero_grad()
        states.append(out.detach())
    return (states[0] - states[1]).abs().max().item(), max(grads)


def test_the_module_under_test_is_this_tree():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    assert (
        pathlib.Path(mm_packing.__file__).resolve() == root / "unsloth" / "utils" / "mm_packing.py"
    )


def test_changing_document_a_leaves_document_b_alone():
    model = _tiny_qwen2_5_vl()
    rope_index = mm_packing.prepare_packed_multimodal_model(model)
    assert rope_index is not None
    b_change, a_grad = _flip_document_a(model, rope_index)
    assert b_change == 0.0
    assert a_grad == 0.0


def test_packing_is_refused_where_transformers_cannot_isolate_documents(monkeypatch):
    masking_utils = pytest.importorskip("transformers.masking_utils")
    if not hasattr(masking_utils, "find_packed_sequence_indices"):
        pytest.skip("transformers without packed-sequence masks")
    model = _tiny_qwen2_5_vl()
    rope_index = mm_packing._find_rope_index(model)
    # transformers 4.x on torch < 2.6: the packed-sequence mask is skipped.
    monkeypatch.setattr(masking_utils, "_is_torch_greater_or_equal_than_2_6", False, raising = False)
    if "_is_torch_greater_or_equal_than_2_6" in masking_utils.create_causal_mask.__code__.co_names:
        # The leak is real there: B changes with A and A gets gradient from B.
        b_change, a_grad = _flip_document_a(model, rope_index)
        assert b_change > 1e-3 and a_grad > 1e-3
    with pytest.raises(ValueError, match = "per-document attention mask"):
        mm_packing.prepare_packed_multimodal_model(model)


def test_packing_is_refused_before_transformers_had_packed_masks(monkeypatch):
    masking_utils = pytest.importorskip("transformers.masking_utils")
    monkeypatch.delattr(masking_utils, "find_packed_sequence_indices", raising = False)
    with pytest.raises(ValueError, match = "per-document attention mask"):
        mm_packing.prepare_packed_multimodal_model(_tiny_qwen2_5_vl())


def test_the_training_forward_of_a_packed_vlm_batch_runs_without_a_cache():
    """A KV cache drops transformers' packed mask: Unsloth's Trainer.compute_loss must pass use_cache=False."""
    pytest.importorskip("unsloth.models._utils")
    from types import SimpleNamespace

    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
    )
    from unsloth.models._utils import _unsloth_pre_compute_loss

    model = Qwen2_5_VLForConditionalGeneration(_tiny_qwen2_5_vl().config)
    batch = mm_packing.pack_multimodal_documents(
        [{"input_ids": torch.tensor([[1, 2, 3]])}, {"input_ids": torch.tensor([[6, 7, 8]])}],
        rope_index = mm_packing._find_rope_index(model),
    )
    seen = []

    class _Trainer:
        args = SimpleNamespace(gradient_accumulation_steps = 1)
        model_accepts_loss_kwargs = True

        def _old_compute_loss(self, model, inputs, *args, **kwargs):
            seen.append(inputs)
            return torch.tensor(0.0)

    _unsloth_pre_compute_loss(_Trainer(), model, batch, num_items_in_batch = torch.tensor(4))
    assert seen[0]["use_cache"] is False
