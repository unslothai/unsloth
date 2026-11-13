# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPU checks for the native decode path: grouped attention reads the KV heads in place instead of
repeating them per query head, and per-step scratch is only made on devices that hold layers."""

import types

import pytest

import unsloth  # noqa: F401  (must precede transformers)
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from unsloth.models import llama
import unsloth.models.gemma2 as g2

N_HEADS, N_KV, HEAD_DIM, HIDDEN = 8, 2, 16, 32


def _attention(bsz, dtype):
    torch.manual_seed(0)

    def linear(i, o):
        layer = torch.nn.Linear(i, o, bias = False).to(dtype)
        layer.weight.requires_grad_(False)
        return layer

    config = types.SimpleNamespace(
        num_attention_heads = N_HEADS,
        num_key_value_heads = N_KV,
        hidden_size = HIDDEN,
        sliding_window = None,
    )
    # cos = 1 and sin = 0 make RoPE the identity, so the reference below needs no rotation.
    rotary = types.SimpleNamespace(
        extend_rope_embedding = lambda x, n: None,
        get_cached = lambda n, i: (torch.ones(4096, HEAD_DIM), torch.zeros(4096, HEAD_DIM)),
    )
    return types.SimpleNamespace(
        config = config,
        num_key_value_groups = N_HEADS // N_KV,
        head_dim = HEAD_DIM,
        q_proj = linear(HIDDEN, N_HEADS * HEAD_DIM),
        k_proj = linear(HIDDEN, N_KV * HEAD_DIM),
        v_proj = linear(HIDDEN, N_KV * HEAD_DIM),
        o_proj = linear(N_HEADS * HEAD_DIM, HIDDEN),
        rotary_emb = rotary,
    )


def _step(attn, X, past, seq_len, do_prefill):
    bsz = X.shape[0]
    return llama.LlamaAttention_fast_forward_inference(
        attn,
        X,
        past,
        torch.full((bsz, 1), seq_len),
        attention_mask = None,
        do_prefill = do_prefill,
    )


def _reference(attn, X, K, V):
    # Repeat K/V per query head, fp32 softmax, as the expand path did.
    bsz = X.shape[0]
    Q = (X @ attn.q_proj.weight.t()).view(bsz, N_HEADS, 1, HEAD_DIM).float()
    K = K.float().repeat_interleave(N_HEADS // N_KV, dim = 1)
    V = V.float().repeat_interleave(N_HEADS // N_KV, dim = 1)
    A = torch.softmax((Q / HEAD_DIM**0.5) @ K.transpose(2, 3), dim = -1) @ V
    return A.transpose(1, 2).reshape(bsz, 1, -1) @ attn.o_proj.weight.float().t()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bsz1_decode_matches_repeated_kv(dtype):
    attn = _attention(1, dtype)
    cached = 7
    past = (
        torch.randn(1, N_KV, cached, HEAD_DIM, dtype = dtype),
        torch.randn(1, N_KV, cached, HEAD_DIM, dtype = dtype),
    )
    X = torch.randn(1, 1, HIDDEN, dtype = dtype)
    out, (K, V) = _step(attn, X, past, cached, do_prefill = True)
    assert K.shape == (1, N_KV, cached + 1, HEAD_DIM)
    tol = 1e-5 if dtype == torch.float32 else 3e-2
    torch.testing.assert_close(out.float(), _reference(attn, X, K, V), atol = tol, rtol = tol)

    # Second step reuses the paged cache and the preallocated score buffer.
    X = torch.randn(1, 1, HIDDEN, dtype = dtype)
    out, (K, V) = _step(attn, X, (K, V), cached + 1, do_prefill = False)
    torch.testing.assert_close(out.float(), _reference(attn, X, K, V), atol = tol, rtol = tol)


class _ShapeRecorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_dispatch__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        out = func(*args, **(kwargs or {}))
        for t in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(t, torch.Tensor):
                self.shapes.append((str(func), tuple(t.shape)))
        return out


def test_bsz1_decode_does_not_materialise_kv_per_query_head():
    attn = _attention(1, torch.float32)
    cached = 9
    past = (torch.randn(1, N_KV, cached, HEAD_DIM), torch.randn(1, N_KV, cached, HEAD_DIM))
    _, past = _step(attn, torch.randn(1, 1, HIDDEN), past, cached, do_prefill = True)
    with _ShapeRecorder() as rec:
        _step(attn, torch.randn(1, 1, HIDDEN), past, cached + 1, do_prefill = False)
    kv_len = cached + 2
    repeated = [
        (op, shape)
        for op, shape in rec.shapes
        if shape in ((1, N_HEADS, kv_len, HEAD_DIM), (1, N_KV, N_HEADS // N_KV, kv_len, HEAD_DIM))
    ]
    assert repeated == [], repeated


@pytest.mark.parametrize("L", [5, 16, 21])
def test_gqa_helpers_write_into_head_layout_buffers(L):
    # 5: shorter than one aligned block; 16: aligned; 21: aligned prefix plus a remainder.
    torch.manual_seed(0)
    bsz, M = 2, 32
    Q = torch.randn(bsz, 1, N_HEADS, HEAD_DIM, dtype = torch.float64).transpose(1, 2)
    paged = torch.randn(M, 2, bsz, N_KV, HEAD_DIM, dtype = torch.float64)
    K = paged[:L, 0].permute(1, 2, 0, 3)
    V = paged[:L, 1].permute(1, 2, 0, 3)
    scores = torch.empty(bsz, N_HEADS, 1, M, dtype = torch.float64)[:, :, :, :L]
    A = llama._gqa_decode_scores(Q, K, scores)
    assert A.data_ptr() == scores.data_ptr()
    Kr = K.repeat_interleave(N_HEADS // N_KV, dim = 1)
    Vr = V.repeat_interleave(N_HEADS // N_KV, dim = 1)
    torch.testing.assert_close(A, Q @ Kr.transpose(2, 3))
    expected = A @ Vr
    out = llama._gqa_decode_values(A, V, Q)
    assert out.data_ptr() == Q.data_ptr()
    torch.testing.assert_close(out, expected)


def _record_to(monkeypatch):
    seen = []
    original = torch.Tensor.to

    def to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if isinstance(target, torch.device):
            seen.append(target)
            # No real accelerator here: stand in for the copy.
            return self if target.type == "cpu" else self.clone()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    return seen


def test_llama_decode_scratch_only_on_layer_devices(monkeypatch):
    monkeypatch.setattr(llama, "DEVICE_COUNT", 4)
    monkeypatch.setattr(llama, "DEVICE_TYPE_TORCH", "cpu")
    monkeypatch.setattr(llama, "per_layer_device", lambda layer: (torch.device("cpu"), 0))
    monkeypatch.setattr(llama, "fast_rms_layernorm_inference", lambda ln, X, **kw: X)

    def add_norm(
        ln,
        X,
        residual,
        add = True,
        **kw,
    ):
        if add:
            X += residual
        residual.copy_(X)
        return X

    monkeypatch.setattr(llama, "add_rms_layernorm_inference", add_norm)
    gates = []

    def attention(self, hidden_states, past_key_value, **kw):
        return hidden_states, past_key_value

    def mlp(
        self,
        X,
        temp_gate = None,
        temp_up = None,
    ):
        gates.append(temp_gate)
        return X

    forward = llama._LlamaModel_fast_forward_inference(attention, mlp)
    hidden, layers = 8, 3
    config = types.SimpleNamespace(
        hidden_size = hidden, intermediate_size = 16, torch_dtype = torch.float32
    )
    model = types.SimpleNamespace(
        layers = [
            types.SimpleNamespace(
                self_attn = None, mlp = None, input_layernorm = None, post_attention_layernorm = None
            )
        ]
        * layers,
        embed_tokens = lambda ids: torch.zeros(*ids.shape, hidden),
        norm = None,
    )
    self = types.SimpleNamespace(model = model, config = config, max_seq_length = 64)
    past = [(torch.zeros(1, 1, 5, 4), torch.zeros(1, 1, 5, 4))] * layers
    seen = _record_to(monkeypatch)
    forward(self, torch.zeros(1, 1, dtype = torch.long), past, torch.full((1, 1), 5))
    # Every layer is on device 0, so devices 1-3 must never receive a scratch copy.
    assert {d.index for d in seen if d.index is not None} == {0}, seen
    assert len(gates) == layers and all(g is gates[0] for g in gates)


def test_gemma2_decode_norm_scratch_only_on_layer_devices(monkeypatch):
    made = []
    empty_like = torch.empty_like

    def recording_empty_like(
        t,
        *args,
        device = None,
        **kwargs,
    ):
        made.append(device)
        return empty_like(t, *args, **kwargs)

    class _Stop(Exception):
        pass

    def attention(self, hidden_states, past_key_value, position_ids, attention_mask, **kw):
        raise _Stop

    monkeypatch.setattr(torch, "empty_like", recording_empty_like)
    monkeypatch.setattr(g2, "DEVICE_COUNT", 3)
    monkeypatch.setattr(llama, "DEVICE_TYPE_TORCH", "cpu")
    monkeypatch.setattr(g2, "HAS_FLASH_ATTENTION_SOFTCAPPING", False)
    monkeypatch.setattr(g2, "Gemma2Attention_fast_forward_inference", attention)
    monkeypatch.setattr(g2, "fast_rms_layernorm_inference_gemma", lambda ln, x, w: x)
    monkeypatch.setattr(g2, "per_layer_device", lambda layer: (torch.device("cpu"), 0))
    hidden = 8
    layer = types.SimpleNamespace(
        input_layernorm = types.SimpleNamespace(weight = torch.ones(hidden)), self_attn = None
    )
    config = types.SimpleNamespace(
        hidden_size = hidden, sliding_window = 4096, torch_dtype = torch.float32
    )
    model = types.SimpleNamespace(
        layers = [layer, layer], embed_tokens = lambda ids: torch.zeros(*ids.shape, hidden)
    )
    self = types.SimpleNamespace(model = model, config = config, max_seq_length = 64)
    past = [(torch.zeros(1, 1, 5, 4), torch.zeros(1, 1, 5, 4))] * 2
    with pytest.raises(_Stop):
        g2.Gemma2Model_fast_forward_inference(
            self, torch.zeros(1, 1, dtype = torch.long), past, torch.full((1, 1), 5)
        )
    assert [getattr(d, "index", None) for d in made if d is not None] == [0], made
