# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the HunyuanVideo-1.5 padded-text attention trim.

``_trim_stream`` / ``_hunyuan_trim_pre_hook`` / ``install_hunyuan_attention_trim`` use real torch
tensor ops, so unlike the attention-backend policy tests in ``test_diffusion_attention.py`` these
require torch. Kept in a separate module so that file stays collectable without torch installed.
"""

from __future__ import annotations

import types

import pytest

# Skip at collection (not abort) when torch is absent so the rest of the backend suite
# stays collectable, matching how the policy tests next door gate their heavy imports.
torch = pytest.importorskip("torch")

import core.inference.diffusion_attention as att  # noqa: E402


def test_trim_stream_drops_trailing_padding():
    # right-padded (valid prefix): drop the globally-invalid tail, keep valid, flag all_valid.
    states = torch.arange(6.0).reshape(1, 6, 1)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
    out_s, out_m, all_valid = att._trim_stream(states, mask)
    assert out_s.shape == (1, 3, 1)
    assert torch.equal(out_s[0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert out_m.shape == (1, 3) and all_valid is True


def test_trim_stream_layout_agnostic_drops_only_global_padding():
    # left-padded (valid suffix): any(dim=0) keeps positions valid for at least one element,
    # so the leading globally-invalid columns are dropped regardless of padding side.
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
    # batch>1 with different valid sets: the union is kept, but a column valid for only one
    # element remains partially padded -> all_valid False -> caller keeps the dense mask.
    states = torch.ones(2, 4, 1)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])  # elem1 has 2 valid, elem2 has 3
    out_s, out_m, all_valid = att._trim_stream(states, mask)
    assert out_s.shape == (2, 3, 1)  # dropped the last col (invalid for both)
    assert all_valid is False


def _fake_dit(n_blocks = 2):
    blocks = [types.SimpleNamespace(attn = types.SimpleNamespace()) for _ in range(n_blocks)]
    return types.SimpleNamespace(transformer_blocks = blocks)


def test_trim_pre_hook_empties_t2v_image_and_trims_and_flags():
    dit = _fake_dit()
    kwargs = {
        "image_embeds": torch.zeros(1, 5, 3),  # all-zero -> t2v -> emptied
        "encoder_hidden_states": torch.arange(4.0).reshape(1, 4, 1),
        "encoder_attention_mask": torch.tensor([[1, 1, 0, 0]]),
        "encoder_hidden_states_2": torch.arange(3.0).reshape(1, 3, 1),
        "encoder_attention_mask_2": torch.tensor([[1, 0, 0]]),
    }
    args, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert out["image_embeds"].shape == (1, 0, 3)  # image tokens dropped
    assert out["encoder_hidden_states"].shape == (1, 2, 1)  # mllm trimmed to 2 valid
    assert out["encoder_hidden_states_2"].shape == (1, 1, 1)  # byt5 trimmed to 1 valid
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is True for b in dit.transformer_blocks)


def test_trim_stream_all_invalid_yields_empty_but_valid():
    # A fully-padded secondary stream (e.g. unused byt5 in t2v) trims to 0 length and reports
    # all_valid True (vacuous) so it does NOT drop the fast path -- it just contributes no tokens.
    states = torch.ones(1, 5, 2)
    mask = torch.zeros(1, 5, dtype = torch.long)
    out_s, out_m, all_valid = att._trim_stream(states, mask)
    assert out_s.shape == (1, 0, 2) and all_valid is True


def test_trim_pre_hook_byt5_all_invalid_keeps_fast_path():
    # The real t2v case: byt5 is entirely padding (valid=0). It must be emptied WITHOUT dropping
    # the null-mask fast path, since mllm still carries the prompt.
    dit = _fake_dit()
    kwargs = {
        "image_embeds": torch.zeros(1, 5, 3),
        "encoder_hidden_states": torch.arange(4.0).reshape(1, 4, 1),
        "encoder_attention_mask": torch.tensor([[1, 1, 1, 0]]),
        "encoder_hidden_states_2": torch.ones(1, 6, 1),
        "encoder_attention_mask_2": torch.zeros(1, 6, dtype = torch.long),  # all padding
    }
    _, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert out["encoder_hidden_states"].shape == (1, 3, 1)
    assert out["encoder_hidden_states_2"].shape == (1, 0, 1)  # byt5 emptied
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is True for b in dit.transformer_blocks)


def test_trim_pre_hook_empty_primary_reverts_and_disables():
    # Pathological empty prompt: mllm has 0 valid tokens. The TokenRefiner must not get a
    # 0-length sequence -> revert all inputs to original and take the stock dense-mask path.
    dit = _fake_dit()
    mllm = torch.ones(1, 4, 1)
    kwargs = {
        "image_embeds": torch.zeros(1, 5, 3),
        "encoder_hidden_states": mllm,
        "encoder_attention_mask": torch.zeros(1, 4, dtype = torch.long),  # 0 valid
    }
    _, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert out["encoder_hidden_states"] is mllm  # reverted (not emptied)
    assert out["image_embeds"].shape == (1, 5, 3)  # image revert too
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is False for b in dit.transformer_blocks)


def test_trim_pre_hook_keeps_i2v_image():
    dit = _fake_dit()
    img = torch.ones(1, 5, 3)  # nonzero -> i2v -> kept
    kwargs = {
        "image_embeds": img,
        "encoder_hidden_states": torch.arange(4.0).reshape(1, 4, 1),
        "encoder_attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }
    _, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert out["image_embeds"] is img  # not emptied
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
    # A malformed mask (not a tensor) must not break the forward: flag False, no exception.
    dit = _fake_dit()
    kwargs = {"encoder_hidden_states": torch.ones(1, 2, 1), "encoder_attention_mask": "oops"}
    args, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is False for b in dit.transformer_blocks)


def test_trim_pre_hook_restores_inputs_on_midtrim_failure():
    # A later stream trips the trim AFTER earlier inputs were already mutated (image emptied, mllm
    # trimmed). The fallback must restore the caller's ORIGINAL kwargs so the stock dense-mask path
    # (flag False) runs on exactly what it expects -- never a half-trimmed mix.
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
        "encoder_attention_mask_2": "oops",  # malformed -> _trim_stream raises after mllm is trimmed
    }
    _, out = att._hunyuan_trim_pre_hook(dit, (), kwargs)
    assert out["image_embeds"] is img  # emptied then restored
    assert out["encoder_hidden_states"] is mllm  # trimmed then restored
    assert out["encoder_attention_mask"] is mllm_mask
    assert out["encoder_hidden_states_2"] is byt5
    assert out["encoder_attention_mask_2"] == "oops"
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is False for b in dit.transformer_blocks)


def test_trim_pre_hook_absent_stream_not_written_back():
    # If encoder_hidden_states is absent from kwargs (a caller passing it positionally), the hook
    # must NOT write it back as None (that would collide: "got multiple values for argument") and
    # must drop the fast path (flag False) rather than null a mask it never verified.
    dit = _fake_dit()
    kwargs = {"image_embeds": torch.zeros(1, 4, 3)}  # no encoder_hidden_states key
    _, out = att._hunyuan_trim_pre_hook(dit, (torch.ones(1, 5, 1),), kwargs)
    assert "encoder_hidden_states" not in out
    assert all(getattr(b.attn, att._NULL_ATTN_FLAG) is False for b in dit.transformer_blocks)


def test_install_trim_noop_for_non_hunyuan_family():
    fam = types.SimpleNamespace(transformer_class = "WanTransformer3DModel")
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())
    assert att.install_hunyuan_attention_trim(pipe, fam) is False


def test_install_trim_noop_when_transformer_class_mismatch():
    # Family claims Hunyuan but the loaded module isn't -> no processors touched, no diffusers
    # import; returns False rather than swapping an unknown attention processor.
    fam = types.SimpleNamespace(transformer_class = "HunyuanVideo15Transformer3DModel")
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())  # class name mismatch
    assert att.install_hunyuan_attention_trim(pipe, fam) is False
