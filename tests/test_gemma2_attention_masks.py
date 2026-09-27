# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Gemma2 decode and prefill masks: padding, single-row decode and unwindowed global layers."""

import types

import pytest

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.models.gemma2 as g2
from real_accelerator import has_real_cuda

torch = pytest.importorskip("torch")


class _Captured(Exception):
    pass


def _masks_reaching_attention(
    monkeypatch,
    flash,
    bsz = 2,
    second_device = "cpu",
):
    seen = []

    def attention(self, hidden_states, past_key_value, position_ids, attention_mask, **kw):
        seen.append((kw["use_sliding_window"], attention_mask))
        if len(seen) == 2:
            raise _Captured
        return hidden_states, past_key_value

    monkeypatch.setattr(g2, "HAS_FLASH_ATTENTION_SOFTCAPPING", flash)
    monkeypatch.setattr(g2, "DEVICE_COUNT", 0)
    monkeypatch.setattr(g2, "Gemma2Attention_fast_forward_inference", attention)
    monkeypatch.setattr(g2, "fast_rms_layernorm_inference_gemma", lambda ln, x, w: x)
    monkeypatch.setattr(g2, "fast_geglu_inference", lambda mlp, x: x)
    # No per-device fp32 buffers on CPU; the stubbed norms never read them.
    monkeypatch.setattr(
        g2, "per_layer_device", lambda layer: (torch.device(layer.device), slice(None))
    )

    hidden, cached = 8, 5
    layer = types.SimpleNamespace(
        device = "cpu",
        input_layernorm = types.SimpleNamespace(weight = torch.ones(hidden)),
        post_attention_layernorm = None,
        pre_feedforward_layernorm = None,
        post_feedforward_layernorm = None,
        self_attn = None,
        mlp = None,
    )
    config = types.SimpleNamespace(
        hidden_size = hidden, sliding_window = 4096, torch_dtype = torch.float32
    )
    model = types.SimpleNamespace(
        layers = [layer, types.SimpleNamespace(**{**vars(layer), "device": second_device})],
        embed_tokens = lambda ids: torch.zeros(*ids.shape, hidden),
    )
    self = types.SimpleNamespace(model = model, config = config, max_seq_length = 64)
    past = [(torch.zeros(bsz, 1, cached, 4), torch.zeros(bsz, 1, cached, 4))] * 2
    attention_mask = torch.tensor([[1] * (cached + 1), [0, 0] + [1] * (cached - 1)])[:bsz]
    with pytest.raises(_Captured):
        g2.Gemma2Model_fast_forward_inference(
            self,
            torch.zeros(bsz, 1, dtype = torch.long),
            past,
            torch.full((bsz, 1), cached),
            attention_mask = attention_mask,
        )
    return seen


@pytest.mark.parametrize("flash", [False, True])
def test_batched_decode_masks_left_padding(monkeypatch, flash):
    seen = _masks_reaching_attention(monkeypatch, flash)
    assert [sliding for sliding, _ in seen] == [True, False]
    for _, mask in seen:
        assert isinstance(mask, torch.Tensor)
        assert mask.shape[-1] == 6
        assert torch.all(mask[0] == 0)
        assert torch.all(mask[1, ..., :2] < -1e30)
        assert torch.all(mask[1, ..., 2:] == 0)


@pytest.mark.parametrize("flash", [False, True])
def test_single_row_decode_passes_no_mask(monkeypatch, flash):
    # A 2D mask reaching the sliding layer raised IndexError past the window.
    seen = _masks_reaching_attention(monkeypatch, flash, bsz = 1)
    assert [mask for _, mask in seen] == [None, None]


@pytest.mark.skipif(
    not has_real_cuda(), reason = "loads tiny Gemma checkpoints through FastLanguageModel on CUDA"
)
def test_prefill_global_layers_see_past_the_window():
    from unsloth import FastLanguageModel

    model, _ = FastLanguageModel.from_pretrained(
        "trl-internal-testing/tiny-Gemma2ForCausalLM",
        max_seq_length = 64,
        load_in_4bit = False,
        dtype = torch.float32,
    )
    FastLanguageModel.for_inference(model)
    window, n = 4, 12
    model.config.sliding_window = window
    masks = {}
    for idx, layer in enumerate(model.model.layers[:2]):
        layer.register_forward_pre_hook(
            lambda mod, args, kwargs, idx = idx: masks.__setitem__(idx, kwargs["causal_mask"]),
            with_kwargs = True,
        )
    ids = torch.arange(10, 10 + n, device = "cuda")[None]
    attention_mask = torch.ones_like(ids)
    attention_mask[:, 0] = 0
    with torch.no_grad():
        model(input_ids = ids, attention_mask = attention_mask)
    q = torch.arange(n)[:, None]
    k = torch.arange(n)[None]
    causal = (k <= q) & (k >= 1)  # key 0 is padding
    kept = {idx: (m.reshape(-1, n, n)[0] == 0).cpu() for idx, m in masks.items()}
    assert torch.equal(kept[1][1:], causal[1:])
    assert not kept[0][-1, 1]
    assert torch.equal(kept[0][1:], (causal & (q - k <= window))[1:])


def test_decode_masks_follow_each_layer_device(monkeypatch):
    # Pipeline-parallel device maps: a mask left on the first device fails on the next GPU's layer.
    seen = _masks_reaching_attention(monkeypatch, flash = False, second_device = "meta")
    assert [mask.device.type for _, mask in seen] == ["cpu", "meta"]


@pytest.mark.skipif(
    not has_real_cuda(), reason = "loads tiny Gemma checkpoints through FastLanguageModel on CUDA"
)
def test_prefill_masks_short_batches_and_4d(monkeypatch):
    import unsloth.models.llama as llama_module
    from unsloth import FastLanguageModel

    # The masks under test feed the softcapping kernels, not flash-attn.
    monkeypatch.setattr(llama_module, "HAS_FLASH_ATTENTION_SOFTCAPPING", False)
    monkeypatch.setattr(g2, "HAS_FLASH_ATTENTION_SOFTCAPPING", False)

    model, _ = FastLanguageModel.from_pretrained(
        "trl-internal-testing/tiny-Gemma2ForCausalLM",
        max_seq_length = 64,
        load_in_4bit = False,
        dtype = torch.float32,
    )
    FastLanguageModel.for_inference(model)
    torch.manual_seed(0)
    with torch.no_grad():
        # More rows than tokens: a 4D mask sliced as mask[:q_len, :q_len] used to cut the batch axis.
        ids = torch.randint(5, 100, (8, 2), device = "cuda")
        batched = model(input_ids = ids, attention_mask = torch.ones_like(ids)).logits
        single = model(input_ids = ids[3:4], attention_mask = torch.ones_like(ids[3:4])).logits
        torch.testing.assert_close(batched[3:4], single, rtol = 1e-4, atol = 1e-4)
        # A caller-built 4D causal mask matches the equivalent 2D mask.
        ids = torch.randint(5, 100, (2, 5), device = "cuda")
        keep = torch.ones(5, 5, dtype = torch.bool, device = "cuda").tril()
        mask4d = torch.zeros(2, 1, 5, 5, device = "cuda").masked_fill(
            ~keep, torch.finfo(torch.float32).min
        )
        from_4d = model(input_ids = ids, attention_mask = mask4d).logits
        from_2d = model(input_ids = ids, attention_mask = torch.ones_like(ids)).logits
        torch.testing.assert_close(from_4d, from_2d, rtol = 1e-4, atol = 1e-4)


@pytest.mark.skipif(
    not has_real_cuda(), reason = "loads tiny Gemma checkpoints through FastLanguageModel on CUDA"
)
def test_unpadded_prefill_uses_static_masks(monkeypatch):
    import unsloth.models.llama as llama_module
    from unsloth import FastLanguageModel

    monkeypatch.setattr(llama_module, "HAS_FLASH_ATTENTION_SOFTCAPPING", False)
    monkeypatch.setattr(g2, "HAS_FLASH_ATTENTION_SOFTCAPPING", False)
    model, _ = FastLanguageModel.from_pretrained(
        "trl-internal-testing/tiny-Gemma2ForCausalLM",
        max_seq_length = 64,
        load_in_4bit = False,
        dtype = torch.float32,
    )
    FastLanguageModel.for_inference(model)
    window, n = 4, 12
    model.config.sliding_window = window
    masks = {}
    for idx, layer in enumerate(model.model.layers[:2]):
        layer.register_forward_pre_hook(
            lambda mod, args, kwargs, idx = idx: masks.__setitem__(idx, kwargs["causal_mask"]),
            with_kwargs = True,
        )
    ids = torch.arange(10, 10 + n, device = "cuda").repeat(8, 1)
    with torch.no_grad():
        model(input_ids = ids, attention_mask = torch.ones_like(ids))
    # Shared [max_seq_length, max_seq_length] masks, not one [bsz, 1, n, n] copy per row.
    assert all(m.dim() == 2 for m in masks.values())
    kept = {idx: (m[:n, :n] == 0).cpu() for idx, m in masks.items()}
    assert kept[1][-1].sum() == n
    assert kept[0][-1].sum() < n
