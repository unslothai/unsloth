# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""FastLanguageModel packed rows keep the compiled decoder block: lengths become device cu_seqlens
eagerly, attention reads nothing from the host, one graph serves every layout (CPU)."""

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")
ad = pytest.importorskip("unsloth.utils.attention_dispatch")
from real_accelerator import has_real_cuda  # noqa: E402  tests/_shared, on sys.path via tests/conftest.py
from unsloth.utils import packing  # noqa: E402

_DIR = os.path.join(os.path.dirname(__file__), "..", "unsloth", "models")


def _load_block_compile():
    spec = importlib.util.spec_from_file_location(
        "_u_bc_packed", os.path.join(_DIR, "_block_compile.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _prepared(lengths, total):
    cu = torch.zeros(len(lengths) + 1, dtype = torch.int32)
    cu[1:] = torch.tensor(lengths).cumsum(0)
    if sum(lengths) < total:
        cu = torch.cat([cu, torch.tensor([total], dtype = torch.int32)])
    longest = max(max(lengths), total - sum(lengths))
    return cu, torch.empty(max(longest, 2), dtype = torch.uint8)


@pytest.fixture(autouse = True)
def _block_env(monkeypatch):
    for key in ("UNSLOTH_BLOCK_COMPILE", "UNSLOTH_COMPILE_DISABLE", "UNSLOTH_FORCE_FLOAT32"):
        monkeypatch.delenv(key, raising = False)


def test_flm_packed_rows_run_the_compiled_block(monkeypatch, capsys):
    M = _load_block_compile()
    calls, seen = [], []

    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            calls.append(sorted(kw))
            return fn(*args, **kw)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    verdict = {"ok": True}

    def prepare(layer, hidden_states, kwargs):
        seen.append(kwargs["packed_seq_lengths"].tolist())
        if not verdict["ok"]:
            return None
        out = {k: v for k, v in kwargs.items() if k != "packed_seq_lengths"}
        out[packing.PACKED_BLOCK_KWARG] = _prepared(kwargs["packed_seq_lengths"].tolist(), 4)
        return out

    dispatch = type(sys)("unsloth.utils.attention_dispatch")
    dispatch.prepare_packed_block = prepare
    monkeypatch.setitem(sys.modules, M._DISPATCH_MODULE, dispatch)

    class UnslothDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(self, hidden_states, **kwargs):
            assert "packed_seq_lengths" not in kwargs or not verdict["ok"]
            return self.proj(hidden_states)

    UnslothDecoderLayer.forward.__module__ = "unsloth.models.llama"
    M._wrap(UnslothDecoderLayer, {})
    layer = UnslothDecoderLayer().train()
    x = torch.randn(1, 4, 8)
    with torch.no_grad():
        layer(x, packed_seq_lengths = torch.tensor([1, 3]))
    assert calls and all(packing.PACKED_BLOCK_KWARG in kw for kw in calls)
    assert seen == [[1, 3]]
    assert "uncompiled" not in capsys.readouterr().out
    calls.clear()
    verdict["ok"] = False
    with torch.no_grad():
        layer(x, packed_seq_lengths = torch.tensor([2, 2]))
    assert calls == []
    assert "uncompiled" in capsys.readouterr().out


def test_prepared_seq_info_reads_nothing_from_the_host():
    cu, carrier = _prepared([3, 1, 5], 12)
    info = packing.get_packed_info_from_kwargs({packing.PACKED_BLOCK_KWARG: (cu, carrier)}, "cpu")
    assert isinstance(info, packing.PackedVarlen)
    assert info.lengths.tolist() == [3, 1, 5, 3] and info.max_seqlen == 5
    assert packing.cover_padded_cu_seqlens(info, 12) == (cu, 5)


def _attend(
    lengths,
    total,
    window = None,
):
    def attend(Q, K, V, cu, carrier):
        config = ad.AttentionConfig(
            backend = ad.SDPA, n_kv_heads = K.shape[1], n_groups = Q.shape[1] // K.shape[1]
        )
        context = ad.AttentionContext(
            bsz = 1,
            q_len = Q.shape[2],
            kv_seq_len = K.shape[2],
            n_heads = Q.shape[1],
            head_dim = Q.shape[3],
            requires_grad = True,
            seq_info = packing.get_packed_info_from_kwargs(
                {packing.PACKED_BLOCK_KWARG: (cu, carrier)}, Q.device
            ),
            attention_mask = None,
            causal_mask = None,
            sliding_window = window,
        )
        return ad.run_attention(config = config, context = context, Q = Q, K = K, V = V)

    return attend


def _per_document(
    Q,
    K,
    V,
    lengths,
    total,
    window = None,
):
    K, V = K.repeat_interleave(2, dim = 1), V.repeat_interleave(2, dim = 1)
    outs, start = [], 0
    for length in list(lengths) + ([total - sum(lengths)] if sum(lengths) < total else []):
        q, k, v = (x[:, :, start : start + length] for x in (Q, K, V))
        mask = torch.ones(length, length, dtype = torch.bool, device = Q.device).tril()
        if window is not None:
            idx = torch.arange(length, device = Q.device)
            mask &= idx[None, :] > idx[:, None] - window
        outs.append(torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = mask))
        start += length
    return torch.cat(outs, dim = 2).transpose(1, 2)


@pytest.mark.parametrize("window", [None, 3])
def test_prepared_attention_is_block_diagonal_and_compiles_once(window):
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    compiled = torch.compile(_attend(None, None, window), backend = backend, fullgraph = True)
    for lengths, total in (([3, 1, 5, 7], 18), ([1, 9, 2, 6, 4, 2], 24)):
        g = torch.Generator().manual_seed(total)
        Q = torch.randn(1, 4, total, 16, generator = g)
        K, V = (torch.randn(1, 2, total, 16, generator = g) for _ in range(2))
        cu, carrier = _prepared(lengths, total)
        for t in (Q, K, V):
            torch._dynamo.mark_dynamic(t, 2)
        torch._dynamo.mark_dynamic(cu, 0)
        torch._dynamo.mark_dynamic(carrier, 0)
        out = compiled(Q, K, V, cu, carrier)
        torch.testing.assert_close(out, _per_document(Q, K, V, lengths, total, window))
        Q2 = Q.clone()
        Q2[:, :, : lengths[0]] += 1.0
        K2 = K.clone()
        K2[:, :, : lengths[0]] -= 1.0
        out2 = compiled(Q2, K2, V, cu, carrier)
        assert torch.equal(out2[:, lengths[0] :], out[:, lengths[0] :])
    assert len(graphs) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
@pytest.mark.parametrize("window", [None, 5])
def test_prepared_attention_runs_flash_varlen_in_one_graph_on_cuda(window):
    """sm80+: aten's flash varlen inside the graph (not the dense-mask fallback), no recompile."""
    if torch.version.hip is not None or torch.cuda.get_device_capability()[0] < 8:
        # ROCm reports gfx capabilities >= 8 but has no traceable varlen kernel: packed rows stay uncompiled there.
        pytest.skip("flash varlen needs CUDA sm80+")
    graphs, calls = [], []

    def backend(gm, example_inputs):
        graphs.append(gm)
        calls.extend(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        return gm.forward

    compiled = torch.compile(_attend(None, None, window), backend = backend, fullgraph = True)
    for lengths, total in (([3, 1, 50, 7], 64), ([1, 90, 2, 26, 4, 2], 128)):
        g = torch.Generator(device = "cuda").manual_seed(total)
        Q = torch.randn(1, 4, total, 64, generator = g, device = "cuda", dtype = torch.bfloat16)
        K, V = (
            torch.randn(1, 2, total, 64, generator = g, device = "cuda", dtype = torch.bfloat16)
            for _ in range(2)
        )
        cu, carrier = (t.cuda() for t in _prepared(lengths, total))
        for t in (Q, K, V):
            torch._dynamo.mark_dynamic(t, 2)
        torch._dynamo.mark_dynamic(cu, 0)
        torch._dynamo.mark_dynamic(carrier, 0)
        out = compiled(Q, K, V, cu, carrier)
        ref = _per_document(Q.float(), K.float(), V.float(), lengths, total, window)
        torch.testing.assert_close(out.float(), ref, atol = 2e-2, rtol = 2e-2)
    assert len(graphs) == 1
    assert any("_flash_attention_forward" in c for c in calls), calls


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
def test_prepared_attention_downcasts_fp32_projections_to_flash_varlen():
    """DoRA's fp32 q/k/v keep flash varlen, not a dense mask over the whole packed row."""
    if torch.version.hip is not None or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("flash varlen needs CUDA sm80+")
    calls = []

    def backend(gm, example_inputs):
        calls.extend(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        return gm.forward

    lengths, total = [3, 1, 50, 10], 64
    g = torch.Generator(device = "cuda").manual_seed(0)
    Q = torch.randn(1, 4, total, 64, generator = g, device = "cuda")
    K, V = (torch.randn(1, 2, total, 64, generator = g, device = "cuda") for _ in range(2))
    cu, carrier = (t.cuda() for t in _prepared(lengths, total))
    out = torch.compile(_attend(None, None), backend = backend, fullgraph = True)(Q, K, V, cu, carrier)
    assert any("_flash_attention_forward" in c for c in calls), calls
    torch.testing.assert_close(
        out.float(), _per_document(Q, K, V, lengths, total), atol = 2e-2, rtol = 2e-2
    )


def test_packed_block_kernel_declines_dropout_and_the_varlen_kill_switch(monkeypatch):
    import types

    from unsloth.utils import attention_dispatch as AD

    monkeypatch.setattr(AD, "_packed_varlen_ok_eager", lambda *a, **k: True)
    monkeypatch.delenv("UNSLOTH_TORCH_VARLEN", raising = False)
    config = types.SimpleNamespace(num_attention_heads = 4, num_key_value_heads = 2)
    layer = torch.nn.Module()
    layer.self_attn = types.SimpleNamespace(config = config, head_dim = 16, attention_dropout = 0.0)
    hidden = torch.zeros(1, 8, 64, dtype = torch.bfloat16)
    assert AD._packed_block_kernel(layer.train(), hidden)
    layer.self_attn.attention_dropout = 0.1
    assert not AD._packed_block_kernel(layer.train(), hidden)
    assert AD._packed_block_kernel(layer.eval(), hidden)
    layer.self_attn.attention_dropout = 0.0
    monkeypatch.setenv("UNSLOTH_TORCH_VARLEN", "0")
    assert not AD._packed_block_kernel(layer.train(), hidden)
    monkeypatch.delenv("UNSLOTH_TORCH_VARLEN")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel([SDPBackend.MATH]):
        assert not AD._packed_block_kernel(layer.train(), hidden)
    assert AD._packed_block_kernel(layer.train(), hidden)


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
def test_prepare_rechecks_the_flash_policy_and_declines_inference_lengths(monkeypatch):
    import types

    from unsloth.utils import attention_dispatch as AD

    if torch.version.hip is not None or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("flash varlen needs sm80+ CUDA")
    monkeypatch.setattr(AD, "_packed_varlen_ok_eager", lambda *a, **k: True)
    config = types.SimpleNamespace(num_attention_heads = 4, num_key_value_heads = 2)
    layer = torch.nn.Module()
    layer.self_attn = types.SimpleNamespace(config = config, head_dim = 16, attention_dropout = 0.0)
    layer.train()
    hidden = torch.zeros(1, 8, 64, dtype = torch.bfloat16, device = "cuda")
    lengths = torch.tensor([3, 5], device = "cuda")
    assert AD.prepare_packed_block(layer, hidden, {"packed_seq_lengths": lengths}) is not None
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(
        [SDPBackend.MATH]
    ):  # same lengths tensor: a cache hit must not skip the policy
        assert AD.prepare_packed_block(layer, hidden, {"packed_seq_lengths": lengths}) is None
    with torch.inference_mode():
        frozen = torch.tensor([3, 5], device = "cuda")
    assert AD.prepare_packed_block(layer, hidden, {"packed_seq_lengths": frozen}) is None
