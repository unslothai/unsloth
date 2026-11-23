# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Real-torch trim tests, kept apart so test_diffusion_attention.py stays collectable without torch."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_attention as att  # noqa: E402


def test_trim_stream_drops_trailing_padding():
    states = torch.arange(6.0).reshape(1, 6, 1)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
    out_s, out_m, all_valid = att._trim_stream(states, mask)
    assert out_s.shape == (1, 3, 1)
    assert torch.equal(out_s[0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert out_m.shape == (1, 3) and all_valid is True


def test_trim_stream_layout_agnostic_drops_only_global_padding():
    # any(dim=0) keeps columns valid for any element, regardless of padding side.
    states = torch.arange(4.0).reshape(1, 4, 1)
    mask = torch.tensor([[0, 0, 1, 1]])
    out_s, out_m, all_valid = att._trim_stream(states, mask)
    assert torch.equal(out_s[0, :, 0], torch.tensor([2.0, 3.0])) and all_valid is True


def test_trim_stream_full_mask_is_noop():
    states = torch.ones(1, 4, 2)
    mask = torch.ones(1, 4, dtype = torch.long)
    out_s, out_m, all_valid = att._trim_stream(states, mask)
    assert out_s.shape == (1, 4, 2) and all_valid is True


def test_trim_stream_none_mask_passthrough():
    states = torch.ones(1, 4, 2)
    out_s, out_m, all_valid = att._trim_stream(states, None)
    assert out_s is states and out_m is None and all_valid is True


def test_trim_stream_mixed_batch_not_all_valid():
    # A column valid for only one element stays padded -> all_valid False -> keep the dense mask.
    states = torch.ones(2, 4, 1)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    out_s, out_m, all_valid = att._trim_stream(states, mask)
    assert out_s.shape == (2, 3, 1)
    assert all_valid is False


def _fake_dit(n_blocks = 2):
    blocks = [types.SimpleNamespace(attn = types.SimpleNamespace()) for _ in range(n_blocks)]
    return types.SimpleNamespace(transformer_blocks = blocks)


def test_trim_pre_hook_empties_t2v_image_and_trims_and_flags():
    dit = _fake_dit()
    kwargs = {
        "image_embeds": torch.zeros(1, 5, 3),
        "encoder_hidden_states": torch.arange(4.0).reshape(1, 4, 1),
        "encoder_attention_mask": torch.tensor([[1, 1, 0, 0]]),
        "encoder_hidden_states_2": torch.arange(3.0).reshape(1, 3, 1),
        "encoder_attention_mask_2": torch.tensor([[1, 0, 0]]),
    }
    args, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert out["image_embeds"].shape == (1, 0, 3)
    assert out["encoder_hidden_states"].shape == (1, 2, 1)
    assert out["encoder_hidden_states_2"].shape == (1, 1, 1)
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is True for b in dit.transformer_blocks)


def test_trim_stream_all_invalid_yields_empty_but_valid():
    # A fully padded secondary stream trims to 0 length with all_valid True (vacuous).
    states = torch.ones(1, 5, 2)
    mask = torch.zeros(1, 5, dtype = torch.long)
    out_s, out_m, all_valid = att._trim_stream(states, mask)
    assert out_s.shape == (1, 0, 2) and all_valid is True


def test_trim_pre_hook_byt5_all_invalid_keeps_fast_path():
    dit = _fake_dit()
    kwargs = {
        "image_embeds": torch.zeros(1, 5, 3),
        "encoder_hidden_states": torch.arange(4.0).reshape(1, 4, 1),
        "encoder_attention_mask": torch.tensor([[1, 1, 1, 0]]),
        "encoder_hidden_states_2": torch.ones(1, 6, 1),
        "encoder_attention_mask_2": torch.zeros(1, 6, dtype = torch.long),
    }
    _, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert out["encoder_hidden_states"].shape == (1, 3, 1)
    assert out["encoder_hidden_states_2"].shape == (1, 0, 1)
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is True for b in dit.transformer_blocks)


def test_trim_pre_hook_empty_primary_reverts_and_disables():
    # 0 valid mllm tokens: the TokenRefiner cannot take a 0-length sequence, so revert to stock.
    dit = _fake_dit()
    mllm = torch.ones(1, 4, 1)
    kwargs = {
        "image_embeds": torch.zeros(1, 5, 3),
        "encoder_hidden_states": mllm,
        "encoder_attention_mask": torch.zeros(1, 4, dtype = torch.long),
    }
    _, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert out["encoder_hidden_states"] is mllm
    assert out["image_embeds"].shape == (1, 5, 3)
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is False for b in dit.transformer_blocks)


def test_trim_pre_hook_keeps_i2v_image():
    dit = _fake_dit()
    img = torch.ones(1, 5, 3)
    kwargs = {
        "image_embeds": img,
        "encoder_hidden_states": torch.arange(4.0).reshape(1, 4, 1),
        "encoder_attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }
    _, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert out["image_embeds"] is img
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is True for b in dit.transformer_blocks)


def test_trim_pre_hook_mixed_batch_flags_false():
    dit = _fake_dit()
    kwargs = {
        "image_embeds": torch.zeros(2, 2, 3),
        "encoder_hidden_states": torch.ones(2, 4, 1),
        "encoder_attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
    }
    _, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is False for b in dit.transformer_blocks)


def test_trim_pre_hook_never_raises_sets_flag_false():
    dit = _fake_dit()
    kwargs = {"encoder_hidden_states": torch.ones(1, 2, 1), "encoder_attention_mask": "oops"}
    args, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is False for b in dit.transformer_blocks)


def test_trim_pre_hook_restores_inputs_on_midtrim_failure():
    # The fallback must restore the ORIGINAL kwargs, never a half-trimmed mix.
    dit = _fake_dit()
    img = torch.zeros(1, 5, 3)
    mllm = torch.arange(4.0).reshape(1, 4, 1)
    mllm_mask = torch.tensor([[1, 1, 0, 0]])
    byt5 = torch.ones(1, 3, 1)
    kwargs = {
        "image_embeds": img,
        "encoder_hidden_states": mllm,
        "encoder_attention_mask": mllm_mask,
        "encoder_hidden_states_2": byt5,
        "encoder_attention_mask_2": "oops",
    }
    _, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert out["image_embeds"] is img
    assert out["encoder_hidden_states"] is mllm
    assert out["encoder_attention_mask"] is mllm_mask
    assert out["encoder_hidden_states_2"] is byt5
    assert out["encoder_attention_mask_2"] == "oops"
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is False for b in dit.transformer_blocks)


def test_trim_pre_hook_absent_stream_not_written_back():
    # Positional encoder_hidden_states: do not write it back (would collide), drop the fast path.
    dit = _fake_dit()
    kwargs = {"image_embeds": torch.zeros(1, 4, 3)}
    _, out = att._hunyuan_trim_pre_hook(dit, (torch.ones(1, 5, 1),), kwargs)
    assert "encoder_hidden_states" not in out
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is False for b in dit.transformer_blocks)


def test_install_trim_noop_for_non_hunyuan_family():
    fam = types.SimpleNamespace(transformer_class = "WanTransformer3DModel")
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())
    assert att.install_hunyuan_attention_trim(pipe, fam) is False


def test_install_trim_noop_when_transformer_class_mismatch():
    fam = types.SimpleNamespace(transformer_class = "HunyuanVideo15Transformer3DModel")
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())
    assert att.install_hunyuan_attention_trim(pipe, fam) is False


def test_set_and_post_hook_clear_null_mask_flag():
    dit = _fake_dit()
    att._set_hunyuan_null_mask(dit, True)
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is True for b in dit.transformer_blocks)
    sentinel = object()
    returned = att._hunyuan_trim_post_hook(dit, (), sentinel)
    assert returned is sentinel
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is False for b in dit.transformer_blocks)


def test_post_hook_always_clears_flag_after_forward_and_on_exception():
    # The flag is True only during the hooked forward; always_call clears it even when forward raises.
    class _DiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = [
                types.SimpleNamespace(attn = types.SimpleNamespace()) for _ in range(2)
            ]
            self.boom = False

        def forward(self):
            assert all(getattr(b.attn, att._NULL_ATTN_FLAG) for b in self.transformer_blocks)
            if self.boom:
                raise RuntimeError("mid-forward boom")
            return "ok"

    dit = _DiT()
    dit.register_forward_pre_hook(lambda m, _a: att._set_hunyuan_null_mask(m, True))
    dit.register_forward_hook(att._hunyuan_trim_post_hook, always_call = True)

    assert dit() == "ok"
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is False for b in dit.transformer_blocks)

    dit.boom = True
    with pytest.raises(RuntimeError):
        dit()
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is False for b in dit.transformer_blocks)


def _t2v_kwargs(device = "cpu"):
    return {
        "image_embeds": torch.zeros(1, 5, 3, device = device),
        "encoder_hidden_states": torch.arange(8.0, device = device).reshape(1, 4, 2),
        "encoder_attention_mask": torch.tensor([[1, 1, 0, 0]], device = device),
        "encoder_hidden_states_2": torch.arange(3.0, device = device).reshape(1, 3, 1),
        "encoder_attention_mask_2": torch.tensor([[1, 0, 0]], device = device),
    }


def test_trim_pre_hook_plans_once_per_input_tensors_and_matches_the_stock_trim():
    dit = _fake_dit()
    base = _t2v_kwargs()
    first = att._hunyuan_trim_pre_hook(dit, (), dict(base))[1]
    second = att._hunyuan_trim_pre_hook(dit, (), dict(base))[1]
    for key in base:
        assert torch.equal(first[key], second[key])
    want_s, want_m, _ = att._trim_stream(
        base["encoder_hidden_states"], base["encoder_attention_mask"]
    )
    assert torch.equal(second["encoder_hidden_states"], want_s) and torch.equal(
        second["encoder_attention_mask"], want_m
    )
    assert len(dit.__dict__[att._TRIM_MEMO_ATTR]) == 1
    base["encoder_attention_mask"][0, 2] = 1
    third = att._hunyuan_trim_pre_hook(dit, (), dict(base))[1]
    assert third["encoder_hidden_states"].shape == (1, 3, 2)
    assert len(dit.__dict__[att._TRIM_MEMO_ATTR]) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_trim_pre_hook_makes_no_host_wait_after_the_first_step():
    import warnings

    dit = _fake_dit()
    base = _t2v_kwargs("cuda")
    att._hunyuan_trim_pre_hook(dit, (), dict(base))
    torch.cuda.synchronize()
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record = True) as caught:
            warnings.simplefilter("always")
            out = att._hunyuan_trim_pre_hook(dit, (), dict(base))[1]
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    assert not [w for w in caught if "synchroniz" in str(w.message).lower()]
    assert out["encoder_hidden_states"].shape == (1, 2, 2)
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is True for b in dit.transformer_blocks)


def test_the_null_mask_flag_keys_the_cuda_graph():
    # The flag picks the attention branch inside the forward: the graph layer keys on it (GRAPH_KEY_EXTRA_ATTR).
    import core.inference.diffusion_cuda_graph as cg

    dit = _fake_dit()
    dit.transformer_blocks[0].attn.processor = None
    assert att._hunyuan_null_mask_state(dit) is False
    att._set_hunyuan_null_mask(dit, True)
    assert att._hunyuan_null_mask_state(dit) is True
    assert cg.GRAPH_KEY_EXTRA_ATTR == "_unsloth_graph_key_extra"


def test_trim_pre_hook_trims_and_plans_once_for_inference_tensors():
    # Inference tensors have no version counter (reading one raises); the hook must still trim.
    dit = _fake_dit()
    with torch.inference_mode():
        base = _t2v_kwargs()
        out = att._hunyuan_trim_pre_hook(dit, (), dict(base))[1]
        again = att._hunyuan_trim_pre_hook(dit, (), dict(base))[1]
    assert out["image_embeds"].shape == (1, 0, 3)
    assert out["encoder_hidden_states"].shape == (1, 2, 2) and out[
        "encoder_hidden_states_2"
    ].shape == (1, 1, 1)
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is True for b in dit.transformer_blocks)
    assert torch.equal(out["encoder_hidden_states"], again["encoder_hidden_states"])
    assert len(dit.__dict__[att._TRIM_MEMO_ATTR]) == 1
