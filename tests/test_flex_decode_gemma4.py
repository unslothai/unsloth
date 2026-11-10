# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Gemma 4 greedy generate() with one-token steps on unsloth_flex_decode equals SDPA decode: real head dims
(sliding 256, full 512), GQA, sliding windows (512 E2B-like, 1024 26B-like), KV-shared consumer layers (E2B) and
k_eq_v full layers (26B), batch 1 and a left-padded ragged batch, decoding past the window. zoo's flex decode is
stood in by eager flex_attention over the same mask semantics (CPU has no flex_decoding kernel). float64: SDPA and
flex then agree bit for bit, so any difference is a routing / mask error, not rounding (Gemma 4's attention scale
is 1.0, which turns fp32 summation-order noise into visible logit differences)."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")

import unsloth.models._flex_decode as FD
import unsloth.models.vision as vision
from unsloth.models.vision import _NonFlexGenerate

if not (
    "q_length"
    in __import__("inspect")
    .signature(__import__("transformers.masking_utils", fromlist = ["x"]).sdpa_mask)
    .parameters
):
    pytest.skip("transformers 5.x mask interface needed", allow_module_level = True)


def _ref_block_mask(
    batch_size,
    kv_length,
    q_offset,
    kv_offset = 0,
    *,
    mask_function = None,
    attention_mask = None,
    window = None,
    device = None,
    graph_contract = True,
):
    # zoo's decode_block_mask contract: slot j holds position kv_offset + j, the query sits at q_offset,
    # mask_function is transformers' (causal / causal + sliding overlay), the 2D mask adds per-slot padding.
    from torch.nn.attention.flex_attention import create_block_mask

    # Copies: a hybrid StaticLayer's cumulative_length is bumped in place before attention reads the mask.
    q_o = q_offset.clone() if isinstance(q_offset, torch.Tensor) else torch.tensor(q_offset)
    kv_o = kv_offset.clone() if isinstance(kv_offset, torch.Tensor) else torch.tensor(kv_offset)
    pad = None
    if attention_mask is not None:
        full = attention_mask.to(torch.bool)
        pos = torch.arange(kv_length) + kv_o
        pad = full[:, pos.clamp(max = full.shape[-1] - 1)] & (pos < full.shape[-1])

    def mask_mod(b, h, q, kv):
        m = mask_function(b, h, q + q_o, kv + kv_o)
        if pad is not None:
            m = m & pad[b, kv]
        return m

    return create_block_mask(mask_mod, batch_size, None, 1, kv_length, device = device)


def _ref_attention(
    q,
    k,
    v,
    block_mask,
    scale,
    *,
    sinks = None,
    softcap = None,
):
    from torch.nn.attention.flex_attention import flex_attention
    return flex_attention(
        q, k, v, block_mask = block_mask, scale = scale, enable_gqa = q.shape[1] != k.shape[1]
    )


@pytest.fixture
def flex(monkeypatch):
    # The routing is under test, not zoo's kernels: always the reference mask / attention (CPU float64).
    monkeypatch.setattr(FD, "decode_block_mask", _ref_block_mask)
    monkeypatch.setattr(FD, "flex_decode_attention", _ref_attention)
    monkeypatch.setattr(
        FD, "flex_decode_supported", lambda head_dim, n_q, n_kv, dtype, device: True
    )
    FD._REGISTERED = False
    FD._SUPPORT_CACHE.clear()
    assert FD.register()
    monkeypatch.setattr(FD, "wanted", lambda device: True)
    # No CUDA graphs on CPU: force flex decode for eager calls too.
    monkeypatch.setenv(FD.ENV, "1")
    monkeypatch.setattr(vision, "_model_device", lambda model: torch.device("cpu"))
    return FD


def _gemma4(kind, window = None):
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    torch.manual_seed(0)
    if kind == "e2b":
        kw = dict(
            num_hidden_layers = 10,
            layer_types = (["sliding_attention"] * 4 + ["full_attention"]) * 2,
            num_attention_heads = 8,
            num_key_value_heads = 1,
            head_dim = 256,
            global_head_dim = 512,
            sliding_window = 512,
            num_kv_shared_layers = 5,
            hidden_size_per_layer_input = 8,
        )
    else:
        kw = dict(
            num_hidden_layers = 6,
            layer_types = ["sliding_attention"] * 5 + ["full_attention"],
            num_attention_heads = 16,
            num_key_value_heads = 8,
            head_dim = 256,
            global_head_dim = 512,
            num_global_key_value_heads = 2,
            attention_k_eq_v = True,
            sliding_window = 1024,
            hidden_size_per_layer_input = 0,
        )
    if window is not None:
        kw["sliding_window"] = window
    config = Gemma4TextConfig(
        vocab_size = 128,
        vocab_size_per_layer_input = 128,
        hidden_size = 64,
        intermediate_size = 64,
        max_position_embeddings = 4096,
        final_logit_softcapping = 30.0,
        pad_token_id = 0,
        eos_token_id = 1,
        bos_token_id = 2,
        attn_implementation = "sdpa",
        **kw,
    )
    model = Gemma4ForCausalLM(config).eval().double()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(4.0)
    return model


def _inputs(prompt, batch):
    torch.manual_seed(1)
    lengths = [prompt] if batch == 1 else [prompt, prompt - 37, prompt // 2, 7]
    ids = torch.zeros(batch, prompt, dtype = torch.long)
    mask = torch.zeros(batch, prompt, dtype = torch.long)
    for i, n in enumerate(lengths):
        ids[i, prompt - n :] = torch.randint(3, 128, (n,))
        mask[i, prompt - n :] = 1
    return ids, mask


def _generate(model, ids, mask, new):
    out = model.generate(
        input_ids = ids,
        attention_mask = mask,
        max_new_tokens = new,
        min_new_tokens = new,
        do_sample = False,
        cache_implementation = "static",
        output_logits = True,
        return_dict_in_generate = True,
        pad_token_id = 0,
    )
    return out.sequences, torch.stack(out.logits, 1)


# The checkpoints' windows on CPU take minutes: over the default per-test timeout on a CI runner.
_SLOW = [pytest.mark.slow, pytest.mark.timeout(900)]


@pytest.mark.parametrize(
    "kind,window,prompt,batch,new",
    [
        ("e2b", 32, 40, 4, 40),
        ("26b", 32, 40, 4, 40),
        # The checkpoints' windows (512 / 1024); a batch past the 1024 window is covered by the window-32 rows.
        pytest.param("e2b", None, 400, 1, 160, marks = _SLOW),
        pytest.param("e2b", None, 400, 4, 160, marks = _SLOW),
        pytest.param("26b", None, 900, 1, 140, marks = _SLOW),
    ],
)
def test_gemma4_flex_decode_matches_sdpa_past_the_window(flex, kind, window, prompt, batch, new):
    model = _gemma4(kind, window)
    ids, mask = _inputs(prompt, batch)
    with torch.no_grad():
        ref_seq, ref_logits = _generate(model, ids, mask, new)
        FD.reset_stats(model)
        with _NonFlexGenerate(model):
            seq, logits = _generate(model, ids, mask, new)
    assert model.config._attn_implementation == "sdpa"
    stats = FD.stats(model)
    window = model.config.sliding_window
    assert prompt + new > window, "the test must decode past the sliding window"
    # Every attention layer ran every decode step on flex (KV-shared consumers included), none fell back.
    assert len(stats) == model.config.num_hidden_layers
    assert all(s["flex"] == new - 1 and not s["fallback"] for s in stats.values()), stats
    assert torch.equal(seq, ref_seq)
    assert (logits - ref_logits).abs().max().item() < 1e-9


def test_gemma4_unsupported_full_layers_stay_on_sdpa(flex, monkeypatch):
    # head_dim 512 without flex decode support (sm75): full layers keep SDPA, sliding layers decode on flex.
    monkeypatch.setattr(
        FD, "flex_decode_supported", lambda head_dim, n_q, n_kv, dtype, device: head_dim != 512
    )
    FD._SUPPORT_CACHE.clear()
    model = _gemma4("e2b", 32)
    ids, mask = _inputs(40, 4)
    with torch.no_grad():
        ref_seq, ref_logits = _generate(model, ids, mask, 24)
        FD.reset_stats(model)
        with _NonFlexGenerate(model):
            seq, logits = _generate(model, ids, mask, 24)
    stats = FD.stats(model)
    full = [s for s in stats.values() if s["kind"] == "full"]
    sliding = [s for s in stats.values() if s["kind"] == "sliding"]
    assert full and sliding
    assert all(s["flex"] == 0 and sum(s["fallback"].values()) == 23 for s in full), full
    assert all("head_dim 512" in next(iter(s["fallback"])) for s in full)
    assert all(s["flex"] == 23 and not s["fallback"] for s in sliding), sliding
    assert torch.equal(seq, ref_seq)
    assert (logits - ref_logits).abs().max().item() < 1e-9


def test_gemma4_with_zoo_decode_block_mask(flex, monkeypatch):
    zoo = pytest.importorskip("unsloth_zoo.flex_attention.decode")
    monkeypatch.setattr(FD, "decode_block_mask", zoo.decode_block_mask)
    model = _gemma4("e2b", 32)
    ids, mask = _inputs(40, 4)
    with torch.no_grad():
        ref_seq, ref_logits = _generate(model, ids, mask, 40)
        FD.reset_stats(model)
        with _NonFlexGenerate(model):
            seq, logits = _generate(model, ids, mask, 40)
    assert all(s["flex"] == 39 and not s["fallback"] for s in FD.stats(model).values())
    assert torch.equal(seq, ref_seq)
    assert (logits - ref_logits).abs().max().item() < 1e-9
