# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""UNSLOTH_GC_SAVE_ATTENTION=1 through run_attention: inside an Unsloth gradient checkpoint the replay
reuses the first pass's attention kernel output (CPU: SDPA's CPU flash kernel, which returns the
logsumexp its backward needs)."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ad = pytest.importorskip("unsloth.utils.attention_dispatch")
G = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
SA = pytest.importorskip("unsloth_zoo.saved_attention")

H, KV, D, S = 4, 2, 16, 32


class _Kernels(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.n = 0

    def __torch_dispatch__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        if func is torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default:
            self.n += 1
        return func(*args, **(kwargs or {}))


def _attend(backend = None):
    backend = backend or ad.SDPA

    def attend(Q, K, V):
        config = ad.AttentionConfig(backend = backend, n_kv_heads = KV, n_groups = H // KV)
        context = ad.AttentionContext(
            bsz = Q.shape[0],
            q_len = Q.shape[2],
            kv_seq_len = K.shape[2],
            n_heads = H,
            head_dim = D,
            requires_grad = True,
            seq_info = None,
            attention_mask = None,
            causal_mask = None,
            training = True,
        )
        return ad.run_attention(config = config, context = context, Q = Q, K = K, V = V)

    return attend


def _weights():
    g = torch.Generator().manual_seed(0)
    return [(torch.randn(H * D, n * D, generator = g) * 0.1).requires_grad_() for n in (H, KV, KV, H)]


def _layers(
    weights,
    attend,
    n_layers = 2,
):
    wq, wk, wv, wo = weights

    def layer(h):
        q = (h @ wq).view(1, S, H, D).transpose(1, 2)
        k = (h @ wk).view(1, S, KV, D).transpose(1, 2)
        v = (h @ wv).view(1, S, KV, D).transpose(1, 2)
        return h + attend(q, k, v).reshape(1, S, H * D) @ wo

    x = torch.randn(1, S, H * D, generator = torch.Generator().manual_seed(1)).requires_grad_()
    h = x
    for _ in range(n_layers):
        h = G.UnslothCheckpointFunction.apply(layer, True, h)
    return h.pow(2).mean()


@pytest.fixture
def checkpointing(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DISABLE_PINNED_MEMORY", "1")
    monkeypatch.setenv("UNSLOTH_DISABLE_DOUBLE_BUFFER", "1")
    G._double_buffer_disabled.cache_clear()
    G.initialize_unsloth_gradient_checkpointing(torch.float32)
    monkeypatch.setattr(G, "MINIMUM_SIZE", 1 << 62)
    SA.SAVED_ATTENTION_STATS.clear()
    yield
    SA.SAVED_ATTENTION_STATS.clear()


def _grads(weights, attend):
    for w in weights:
        w.grad = None
    counter = _Kernels()
    with counter:
        loss = _layers(weights, attend)
        loss.backward()
    return loss.detach(), [w.grad.clone() for w in weights], counter.n


def test_replay_reuses_the_attention_output(monkeypatch, checkpointing):
    weights = _weights()
    full = _grads(weights, _attend())
    monkeypatch.setattr(ad, "_SAVE_ATTENTION", True)
    monkeypatch.setattr(ad, "_saved_attention_call", SA.saved_attention_call)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    saved = _grads(weights, _attend())
    torch.testing.assert_close(saved[0], full[0])
    for a, b in zip(saved[1], full[1]):
        torch.testing.assert_close(a, b)
    # 2 layers: today 2 forwards + 2 replays; saved: the replays reuse the first pass.
    assert full[2] == 4 and saved[2] == 2
    assert SA.SAVED_ATTENTION_STATS["reused"] == 2


def test_inference_and_unset_flag_take_the_plain_call(monkeypatch, checkpointing):
    calls = []
    monkeypatch.setattr(ad, "_saved_attention_call", lambda fn, *t: calls.append(1) or fn(*t))
    q = torch.randn(1, H, S, D)
    k = torch.randn(1, KV, S, D)
    monkeypatch.setattr(ad, "_SAVE_ATTENTION", False)
    _attend()(q, k, k)
    assert calls == []
    monkeypatch.setattr(ad, "_SAVE_ATTENTION", True)
    _attend()(q, k, k)
    assert calls == [1]


def test_a_traced_caller_keeps_its_traced_backend_choice(monkeypatch, checkpointing):
    """Under compile, plain causal xformers calls take SDPA; the saving call graph breaks into eager code,
    which must still take SDPA (xformers' grad layout has no kernel on some GPUs, none on CPU)."""
    monkeypatch.setattr(ad, "_SAVE_ATTENTION", True)
    monkeypatch.setattr(ad, "_saved_attention_call", SA.saved_attention_call)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    seen = []
    original = ad._run_attention

    def spy(**kwargs):
        seen.append(ad._compiling())
        return original(**kwargs)

    monkeypatch.setattr(ad, "_run_attention", spy)
    torch._dynamo.reset()
    attend = torch.compile(_attend(ad.XFORMERS), backend = "eager", fullgraph = False)
    weights = _weights()
    for w in weights:
        w.grad = None
    loss = _layers(weights, attend)
    loss.backward()
    assert seen and all(seen)
    assert SA.SAVED_ATTENTION_STATS["reused"] == 2


def test_eager_saving_call_keeps_the_operand_device_guard(monkeypatch):
    """The saving call runs eagerly even under a traced caller: xformers' grouped forward still needs the
    operand's device made current (device_map), which only Inductor's own device guard provided."""

    class _Device:
        type, index = "cuda", 1

    class _X:
        is_cuda = True
        device = _Device()

    entered = []
    monkeypatch.setattr(ad, "DEVICE_TYPE", "cuda")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "device", lambda device: entered.append(device.index) or "guard"
    )
    monkeypatch.setattr(ad, "_AS_TRACED", [True])
    assert ad._compiling()
    assert ad._on_tensor_device(_X()) == "guard" and entered == [1]
