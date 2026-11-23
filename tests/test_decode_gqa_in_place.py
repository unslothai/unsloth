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
    empty_like, empty = torch.empty_like, torch.empty

    # The scratch is sized from the norm weight: empty_like before #18038, empty(weight.shape) since (a DTensor
    # weight under FSDP2 block streaming must not be copied). Record the device of either allocator.
    def recording_empty_like(
        t,
        *args,
        device = None,
        **kwargs,
    ):
        made.append(device)
        return empty_like(t, *args, **kwargs)

    def recording_empty(
        *args,
        device = None,
        **kwargs,
    ):
        made.append(device)
        return empty(*args, **kwargs)

    class _Stop(Exception):
        pass

    def attention(self, hidden_states, past_key_value, position_ids, attention_mask, **kw):
        raise _Stop

    monkeypatch.setattr(torch, "empty_like", recording_empty_like)
    monkeypatch.setattr(torch, "empty", recording_empty)
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


def _keep_mask(bsz, kv_len):
    # Left padding: row i drops its first 2 * i positions (row 0 none), as a batched rollout does.
    pos = torch.arange(kv_len)[None, :]
    return (pos >= 2 * torch.arange(bsz)[:, None])[:, None, None, :]


def _masked_reference(attn, X, K, V, keep):
    bsz = X.shape[0]
    Q = (X.double() @ attn.q_proj.weight.double().t()).view(bsz, N_HEADS, 1, HEAD_DIM)
    K = K.double().repeat_interleave(N_HEADS // N_KV, dim = 1)
    V = V.double().repeat_interleave(N_HEADS // N_KV, dim = 1)
    S = (Q / HEAD_DIM**0.5) @ K.transpose(2, 3)
    A = torch.softmax(S.masked_fill(~keep, float("-inf")), dim = -1) @ V
    return A.transpose(1, 2).reshape(bsz, 1, -1) @ attn.o_proj.weight.double().t()


def _batched_steps(
    monkeypatch,
    mode,
    dtype,
    bsz = 3,
    cached = 11,
):
    monkeypatch.setattr(llama, "GROUPED_DECODE", mode)
    attn = _attention(bsz, dtype)
    torch.manual_seed(1)
    past = (
        torch.randn(bsz, N_KV, cached, HEAD_DIM, dtype = dtype),
        torch.randn(bsz, N_KV, cached, HEAD_DIM, dtype = dtype),
    )
    outs = []
    calls = llama.GROUPED_DECODE_CALLS[0]
    for step in range(2):
        X = torch.randn(bsz, 1, HIDDEN, dtype = dtype)
        keep = _keep_mask(bsz, cached + 1 + step)
        out, past = llama.LlamaAttention_fast_forward_inference(
            attn,
            X,
            past,
            torch.full((bsz, 1), cached + step),
            attention_mask = keep,
            do_prefill = step == 0,
        )
        # out is the layer's reused output scratch: keep a copy before the next step.
        outs.append((out.clone(), X, keep) + tuple(past))
    return attn, outs, llama.GROUPED_DECODE_CALLS[0] - calls


@pytest.mark.parametrize("sdpa_gqa", [True, False], ids = ["sdpa_gqa", "expand"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batched_decode_grouped_matches_sdpa_with_left_padding(monkeypatch, dtype, sdpa_gqa):
    assert "s7_decode" in llama.__file__ or hasattr(llama, "_grouped_decode_batched")
    monkeypatch.setattr(llama, "SDPA_HAS_GQA", sdpa_gqa and llama.SDPA_HAS_GQA)
    attn0, base, calls0 = _batched_steps(monkeypatch, "0", dtype)
    attn1, grouped, calls1 = _batched_steps(monkeypatch, "1", dtype)
    # "0" keeps today's batched route; "1" takes the grouped route in every layer call.
    assert calls0 == 0 and calls1 == 2
    tol = 1e-5 if dtype == torch.float32 else 3e-2
    for (o0, X, keep, K, V), (o1, *_rest) in zip(base, grouped):
        ref = _masked_reference(attn1, X, K, V, keep)
        torch.testing.assert_close(o1.double(), ref, atol = tol, rtol = tol)
        torch.testing.assert_close(o1.float(), o0.float(), atol = tol, rtol = tol)


def test_batched_grouped_decode_does_not_materialise_kv_per_query_head(monkeypatch):
    monkeypatch.setattr(llama, "GROUPED_DECODE", "1")
    monkeypatch.setattr(llama, "SDPA_HAS_GQA", False)  # the route it replaces would expand K / V
    bsz, cached = 3, 9
    attn = _attention(bsz, torch.float32)
    past = (torch.randn(bsz, N_KV, cached, HEAD_DIM), torch.randn(bsz, N_KV, cached, HEAD_DIM))
    _, past = llama.LlamaAttention_fast_forward_inference(
        attn, torch.randn(bsz, 1, HIDDEN), past, torch.full((bsz, 1), cached),
        attention_mask = _keep_mask(bsz, cached + 1), do_prefill = True,
    )  # fmt: skip
    with _ShapeRecorder() as rec:
        llama.LlamaAttention_fast_forward_inference(
            attn, torch.randn(bsz, 1, HIDDEN), past, torch.full((bsz, 1), cached + 1),
            attention_mask = _keep_mask(bsz, cached + 2), do_prefill = False,
        )  # fmt: skip
    kv_len = cached + 2
    repeated = [
        (op, shape)
        for op, shape in rec.shapes
        if shape
        in ((bsz, N_HEADS, kv_len, HEAD_DIM), (bsz, N_KV, N_HEADS // N_KV, kv_len, HEAD_DIM))
    ]
    assert repeated == [], repeated


@pytest.mark.parametrize("sdpa_gqa_with_batch_mask", [True, False], ids = ["llama", "qwen3"])
def test_batched_core_and_span_routes_agree(monkeypatch, sdpa_gqa_with_batch_mask):
    # The captured-step attention (_decode_attention_core) and the whole-step span (_decode_attention_span).
    torch.manual_seed(2)
    bsz, kv_len, rows = 3, 13, 32
    attn = _attention(bsz, torch.float64)
    attn.paged_attention = torch.randn(rows, 2, bsz, N_KV, HEAD_DIM, dtype = torch.float64)
    attn.paged_attention_K = attn.paged_attention[:, 0]
    attn.paged_attention_V = attn.paged_attention[:, 1]
    attn.attention = torch.empty(bsz, N_HEADS, 1, rows, dtype = torch.float64)
    attn.scalar = 1.0 / HEAD_DIM**0.5
    keep = _keep_mask(bsz, kv_len)
    Q = torch.randn(bsz, 1, N_HEADS, HEAD_DIM, dtype = torch.float64).transpose(1, 2)
    span = 16
    span_keep = torch.zeros(bsz, 1, 1, rows, dtype = torch.bool)
    span_keep[..., :kv_len] = keep
    static = types.SimpleNamespace(span = span, keep = span_keep, drop = ~span_keep)
    out = {}
    for mode in ("0", "1"):
        monkeypatch.setattr(llama, "GROUPED_DECODE", mode)
        calls = llama.GROUPED_DECODE_CALLS[0]
        core = llama._decode_attention_core(
            attn, Q.clone(), kv_len, keep, bsz, sdpa_gqa_with_batch_mask
        )[0]
        whole = llama._decode_attention_span(attn, Q.clone(), static, bsz, sdpa_gqa_with_batch_mask)
        out[mode] = (core, whole, llama.GROUPED_DECODE_CALLS[0] - calls)
    assert out["0"][2] == 0 and out["1"][2] == 2
    # The grouped route's softmax runs in float32 (as at bsz == 1), so float64 inputs agree to ~1e-7.
    for i in range(2):
        torch.testing.assert_close(out["1"][i], out["0"][i], atol = 1e-6, rtol = 1e-5)
    torch.testing.assert_close(out["1"][0], out["1"][1], atol = 1e-6, rtol = 1e-5)


def test_grouped_decode_auto_follows_the_card_rule(monkeypatch):
    monkeypatch.setattr(llama, "GROUPED_DECODE", "auto")
    cpu = torch.device("cpu")
    gate = llama._grouped_decode_batched
    # No measured rule off CUDA: batched decode keeps SDPA.
    assert llama._grouped_decode_rule(cpu) == (None, None, None)
    assert not gate(8, 4096, True, True, cpu)
    # (SDPA GQA unmasked, SDPA GQA masked, expanded K / V)
    monkeypatch.setattr(llama, "_grouped_decode_rule", lambda device: (None, 8192, 1024))
    assert not gate(16, 8192, False, False, cpu)  # unmasked SDPA GQA: never on this card
    assert not gate(8, 1023, False, True, cpu)  # 8184 < 8192
    assert gate(8, 1024, False, True, cpu)
    assert not gate(2, 511, True, True, cpu)  # 1022 < 1024 vs the expand route
    assert gate(2, 512, True, False, cpu)  # expand ignores the mask
    monkeypatch.setattr(llama, "GROUPED_DECODE", "0")
    assert not gate(16, 8192, True, True, cpu)
    monkeypatch.setattr(llama, "GROUPED_DECODE", "1")
    assert gate(2, 1, False, False, cpu)


@pytest.mark.parametrize(
    "name, rule",
    [
        ("Tesla T4", (8192, 1024, 256)),
        ("NVIDIA L4", (None, 2048, 1024)),
        ("NVIDIA A100-SXM4-40GB", (None, 16384, 8192)),
        ("NVIDIA A100 80GB PCIe", (None, 16384, 8192)),
        ("NVIDIA L40S", (None, None, None)),  # not an L4
        ("NVIDIA RTX A1000", (None, None, None)),
        ("NVIDIA H100 80GB HBM3", (None, None, None)),  # unmeasured: keeps SDPA
    ],
)
def test_grouped_decode_rule_matches_whole_card_names(monkeypatch, name, rule):
    monkeypatch.setattr(llama, "_grouped_decode_rules", {})
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device = None: name)
    assert llama._grouped_decode_rule(torch.device("cuda", 0)) == rule
