# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Packed rows on Unsloth's own models keep their documents apart on the flash-attn path, in training and
in eval (where use_cache stays on), and Mistral's window matches transformers' (k > q - sliding_window)."""

import pytest
import torch
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

pytestmark = pytest.mark.skipif(
    not has_real_cuda(), reason = "Unsloth's model forwards need a CUDA device"
)

WINDOW = 4
LENGTHS = [7, 9, 5]


def _reference(q, k, v, causal, window_size, softmax_scale):
    # q [T, H, D], k / v [T, Hk, D]: flash-attn's semantics, computed in float32.
    T, H, D = q.shape
    k = k.repeat_interleave(H // k.shape[1], dim = 1)
    v = v.repeat_interleave(H // v.shape[1], dim = 1)
    scale = D**-0.5 if softmax_scale is None else softmax_scale
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    qi = torch.arange(T, device = q.device)[:, None]
    ki = torch.arange(T, device = q.device)[None, :]
    keep = torch.ones(T, T, dtype = torch.bool, device = q.device)
    if causal:
        keep &= ki <= qi
    if window_size[0] >= 0:
        keep &= qi - ki <= window_size[0]
    scores = scores.masked_fill(~keep, float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float()).to(q.dtype)


def _flash_attn_func(
    q,
    k,
    v,
    dropout_p = 0.0,
    softmax_scale = None,
    causal = False,
    window_size = (-1, -1),
    **kwargs,
):
    return torch.stack(
        [
            _reference(q[b], k[b], v[b], causal, window_size, softmax_scale)
            for b in range(q.shape[0])
        ]
    )


def _flash_attn_varlen_func(
    q,
    k,
    v,
    cu_q,
    cu_k,
    max_q,
    max_k,
    dropout_p = 0.0,
    softmax_scale = None,
    causal = False,
    window_size = (-1, -1),
    **kwargs,
):
    out = torch.empty_like(q)
    cu = cu_q.tolist()
    for start, end in zip(cu[:-1], cu[1:]):
        out[start:end] = _reference(
            q[start:end], k[start:end], v[start:end], causal, window_size, softmax_scale
        )
    return out


@pytest.fixture(scope = "module")
def model():
    from unsloth import FastLanguageModel

    try:
        model, _ = FastLanguageModel.from_pretrained(
            model_name = "hf-internal-testing/tiny-random-MistralForCausalLM",
            max_seq_length = 64,
            load_in_4bit = False,
            dtype = torch.bfloat16,
        )
    except OSError as exc:  # pragma: no cover - offline CI
        pytest.skip(f"Requires access to the tiny Mistral checkpoint: {exc}")
    model.config.sliding_window = WINDOW
    for layer in model.model.layers:
        layer.self_attn.config.sliding_window = WINDOW
    # Training mode: Unsloth builds no 4D mask there, so the flash / SDPA backends run.
    model.train()
    return model


def _logits(model, ids, **kwargs):
    with torch.no_grad():
        return model(input_ids = ids, **kwargs).logits.float()


def _use_fake_flash(monkeypatch):
    from unsloth.utils import attention_dispatch as ad

    monkeypatch.setattr(ad, "HAS_FLASH_ATTENTION", True)
    monkeypatch.setattr(ad, "flash_attn_func", _flash_attn_func)
    monkeypatch.setattr(ad, "flash_attn_varlen_func", _flash_attn_varlen_func)


@pytest.fixture
def fake_flash(monkeypatch):
    _use_fake_flash(monkeypatch)


def test_packed_documents_do_not_attend_across_each_other(model, fake_flash):
    torch.manual_seed(0)
    docs = [torch.randint(5, 100, (n,), device = "cuda") for n in LENGTHS]
    packed = torch.cat(docs)[None]
    position_ids = torch.cat([torch.arange(n, device = "cuda") for n in LENGTHS])[None]
    out = _logits(
        model,
        packed,
        position_ids = position_ids,
        packed_seq_lengths = torch.tensor(LENGTHS, dtype = torch.int32),
    )
    alone = torch.cat([_logits(model, d[None]) for d in docs], dim = 1)
    torch.testing.assert_close(out, alone, atol = 2e-2, rtol = 2e-2)


def test_flash_window_matches_sdpa(model, monkeypatch):
    # One document longer than the window: dense flash and SDPA must drop the same keys.
    from unsloth.utils import attention_dispatch as ad

    torch.manual_seed(1)
    ids = torch.randint(5, 100, (1, 12), device = "cuda")
    sdpa = _logits(model, ids)
    monkeypatch.setattr(ad, "HAS_FLASH_ATTENTION", True)
    monkeypatch.setattr(ad, "flash_attn_func", _flash_attn_func)
    monkeypatch.setattr(ad, "flash_attn_varlen_func", _flash_attn_varlen_func)
    torch.testing.assert_close(_logits(model, ids), sdpa, atol = 2e-2, rtol = 2e-2)


@pytest.mark.parametrize("window", [WINDOW, None])
@pytest.mark.parametrize("flash", [False, True])
def test_packed_eval_keeps_documents_apart(model, monkeypatch, window, flash):
    # Eval builds a causal mask over the whole row unless the row is packed (padding-free eval batches are).
    torch.manual_seed(2)
    docs = [torch.randint(5, 100, (n,), device = "cuda") for n in LENGTHS]
    packed = torch.cat(docs)[None]
    position_ids = torch.cat([torch.arange(n, device = "cuda") for n in LENGTHS])[None]
    if flash:
        _use_fake_flash(monkeypatch)
    model.config.sliding_window = window
    model.eval()
    try:
        out = _logits(
            model,
            packed,
            position_ids = position_ids,
            use_cache = True,
            packed_seq_lengths = torch.tensor(LENGTHS, dtype = torch.int32),
        )
        alone = torch.cat([_logits(model, d[None]) for d in docs], dim = 1)
    finally:
        model.config.sliding_window = WINDOW
        model.train()
    torch.testing.assert_close(out, alone, atol = 2e-2, rtol = 2e-2)


@pytest.mark.parametrize(
    "name",
    [
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "trl-internal-testing/tiny-Qwen3ForCausalLM",
    ],
)
def test_packed_eval_with_cache_keeps_documents_apart_on_flash(name, monkeypatch):
    # Eval returns a cache, which must not turn a packed row into one dense flash call.
    from unsloth import FastLanguageModel

    try:
        other, _ = FastLanguageModel.from_pretrained(
            model_name = name, max_seq_length = 64, load_in_4bit = False, dtype = torch.bfloat16
        )
    except OSError as exc:  # pragma: no cover - offline CI
        pytest.skip(f"Requires access to {name}: {exc}")
    _use_fake_flash(monkeypatch)
    other.eval()
    torch.manual_seed(3)
    vocab = other.config.vocab_size
    docs = [torch.randint(5, min(vocab, 100), (n,), device = "cuda") for n in LENGTHS]
    out = _logits(
        other,
        torch.cat(docs)[None],
        position_ids = torch.cat([torch.arange(n, device = "cuda") for n in LENGTHS])[None],
        use_cache = True,
        packed_seq_lengths = torch.tensor(LENGTHS, dtype = torch.int32),
    )
    alone = torch.cat([_logits(other, d[None], use_cache = True) for d in docs], dim = 1)
    torch.testing.assert_close(out, alone, atol = 2e-2, rtol = 2e-2)
