# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""UNSLOTH_GC_SAVE_ATTENTION=1 through run_attention: inside an Unsloth gradient checkpoint the replay
reuses the first pass's attention kernel output (CPU: SDPA's CPU flash kernel, which returns the
logsumexp its backward needs)."""

import pytest
import torch
import torch.utils.checkpoint
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


def _force_saved_attention(monkeypatch):
    monkeypatch.setattr(ad, "_SAVE_ATTENTION", True)
    monkeypatch.setattr(ad, "_saved_attention_call", SA.saved_attention_call)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    # A ROCm build keeps the explicit setting off (its GPU SDPA kernels are not intercepted); these tests
    # run the CPU kernel, which is intercepted on every build.
    monkeypatch.setattr(torch.version, "hip", None, raising = False)


def test_replay_reuses_the_attention_output(monkeypatch, checkpointing):
    weights = _weights()
    full = _grads(weights, _attend())
    _force_saved_attention(monkeypatch)
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
    _force_saved_attention(monkeypatch)
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


# Automatic mode (UNSLOTH_GC_SAVE_ATTENTION unset / =auto): long SDPA training calls only.


def _fake_gpu(monkeypatch, capability = (8, 0)):
    monkeypatch.delenv("UNSLOTH_GC_SAVE_ATTENTION", raising = False)
    monkeypatch.setattr(SA, "_AUTO_ACTIVE", [False])
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)


def test_auto_mode_routes_long_calls_on_a_measured_gpu(monkeypatch):
    _fake_gpu(monkeypatch)
    call, min_seq = ad._load_saved_attention()
    assert call is SA.saved_attention_call
    assert min_seq == SA._AUTO_MIN_SEQ[(8, 0)] == 8192
    assert SA._AUTO_ACTIVE[0] is True


@pytest.mark.parametrize("setting", ["0", "false"])
def test_auto_mode_kill_switch(monkeypatch, setting):
    _fake_gpu(monkeypatch)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", setting)
    assert ad._load_saved_attention() == (None, None)
    assert SA._AUTO_ACTIVE[0] is False


def test_auto_mode_off_on_unmeasured_gpu_and_rocm(monkeypatch):
    _fake_gpu(monkeypatch, (10, 0))
    assert ad._load_saved_attention() == (None, None)
    _fake_gpu(monkeypatch, (8, 0))
    monkeypatch.setattr(torch.version, "hip", "6.4", raising = False)
    assert ad._load_saved_attention() == (None, None)


def test_explicit_setting_routes_every_call(monkeypatch):
    _fake_gpu(monkeypatch)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    assert ad._load_saved_attention() == (SA.saved_attention_call, None)


def test_an_unsloth_zoo_without_auto_mode_stays_off(monkeypatch):
    import sys
    import types

    old = types.ModuleType("unsloth_zoo.saved_attention")
    old.save_attention_enabled = lambda: False
    old.saved_attention_call = SA.saved_attention_call
    monkeypatch.setitem(sys.modules, "unsloth_zoo.saved_attention", old)
    _fake_gpu(monkeypatch)
    assert ad._load_saved_attention() == (None, None)


def _routed(
    monkeypatch,
    q_len,
    backend = None,
    packed = False,
):
    calls = []
    # Automatic mode only routes where a frame can exist: under Unsloth's smart checkpoint.
    monkeypatch.setattr(torch.utils.checkpoint, "CheckpointFunction", G.UnslothCheckpointFunction)
    monkeypatch.delenv("UNSLOTH_GC_SAVE_ATTENTION", raising = False)
    monkeypatch.setattr(ad, "_SAVE_ATTENTION", True)
    monkeypatch.setattr(ad, "_SAVE_ATTENTION_MIN_SEQ", 16)
    monkeypatch.setattr(ad, "_saved_attention_call", lambda fn, *t: calls.append(1) or fn(*t))
    q = torch.randn(1, H, q_len, D)
    k = torch.randn(1, KV, q_len, D)
    config = ad.AttentionConfig(backend = backend or ad.SDPA, n_kv_heads = KV, n_groups = H // KV)
    context = ad.AttentionContext(
        bsz = 1,
        q_len = q_len,
        kv_seq_len = q_len,
        n_heads = H,
        head_dim = D,
        requires_grad = True,
        seq_info = (None, None, q_len) if packed else None,
        attention_mask = None,
        causal_mask = None,
        training = True,
    )
    routed = ad._auto_saves_attention(config, context)
    if not packed and config.backend == ad.SDPA:
        ad.run_attention(config = config, context = context, Q = q, K = k, V = k)
        assert bool(calls) == routed
    return routed


def test_auto_mode_routes_only_long_unpacked_sdpa_calls(monkeypatch):
    assert _routed(monkeypatch, 32) is True
    assert _routed(monkeypatch, 16) is True
    # Shorter calls keep the plain (traceable) call: no graph break.
    assert _routed(monkeypatch, 15) is False
    assert _routed(monkeypatch, 32, backend = ad.XFORMERS) is False
    assert _routed(monkeypatch, 32, packed = True) is False


def test_auto_mode_reuses_through_run_attention(monkeypatch, checkpointing):
    """End to end on CPU: first step measured, second step kept and reused, same grads."""
    weights = _weights()
    full = [_grads(weights, _attend()) for _ in range(2)]
    monkeypatch.setattr(torch.utils.checkpoint, "CheckpointFunction", G.UnslothCheckpointFunction)
    monkeypatch.setattr(ad, "_SAVE_ATTENTION", True)
    monkeypatch.setattr(ad, "_SAVE_ATTENTION_MIN_SEQ", S)
    monkeypatch.setattr(ad, "_saved_attention_call", SA.saved_attention_call)
    monkeypatch.setattr(SA, "_AUTO_ACTIVE", [True])
    monkeypatch.setattr(SA, "_BUDGETS", {})
    monkeypatch.setattr(SA, "_device_peak", lambda device: 1 << 20)
    monkeypatch.setattr(SA, "_device_capacity", lambda device: 1 << 30)
    monkeypatch.setattr(SA, "_AUTO_MARGIN_BYTES", 0)
    monkeypatch.delenv("UNSLOTH_GC_SAVE_ATTENTION", raising = False)
    saved = [_grads(weights, _attend()) for _ in range(2)]
    for a, b in zip(saved, full):
        torch.testing.assert_close(a[0], b[0])
        for x, y in zip(a[1], b[1]):
            torch.testing.assert_close(x, y)
    assert [s[2] for s in full] == [4, 4] and [s[2] for s in saved] == [4, 2], dict(
        SA.SAVED_ATTENTION_STATS
    )
    assert SA.SAVED_ATTENTION_STATS["declined_unmeasured"] == 2
    assert SA.SAVED_ATTENTION_STATS["reused"] == 2


# ROCm with UNSLOTH_GC_SAVE_ATTENTION=1: unsloth_zoo decides per checkpoint frame and device.


def test_rocm_explicit_setting_binds_from_configuration(monkeypatch):
    """Routing is bound from the setting at import, never from the current GPU's eligibility (which
    would freeze the first GPU's answer and allocate on it before ranks are placed)."""
    if not hasattr(SA, "save_attention_requested"):
        pytest.skip(
            "unsloth_zoo without per-device ROCm eligibility: the setting is read as before"
        )
    monkeypatch.setattr(
        SA, "_ROCM_SDPA_SELECT", getattr(SA, "_ROCM_SDPA_SELECT", None), raising = False
    )
    monkeypatch.setattr(ad, "_ROCM_SDPA_GUARD", True)
    monkeypatch.setattr(torch.version, "hip", "7.13.0", raising = False)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    touched = []
    monkeypatch.setattr(SA, "save_attention_enabled", lambda *a: touched.append(a) or False)
    monkeypatch.setattr(
        SA, "_rocm_sdpa_is_kept", lambda *a, **k: touched.append(a) or False, raising = False
    )
    assert ad._load_saved_attention() == (SA.saved_attention_call, None)
    assert touched == []
    assert SA._ROCM_SDPA_SELECT is ad._select_rocm_saved_sdpa


def test_rocm_explicit_routing_without_a_frame_compiles_fullgraph(monkeypatch, checkpointing):
    """No frame (no Unsloth checkpoint, or a device whose SDPA keeps nothing): the call stays in the
    caller's graph, so fullgraph=True training compiles; inside a checkpoint the replay still reuses."""
    monkeypatch.setattr(ad, "_ROCM_SDPA_GUARD", True)
    monkeypatch.setattr(torch.version, "hip", "7.13.0", raising = False)
    monkeypatch.setattr(ad, "_SAVE_ATTENTION", True)
    monkeypatch.setattr(ad, "_SAVE_ATTENTION_MIN_SEQ", None)
    monkeypatch.setattr(ad, "_saved_attention_call", SA.saved_attention_call)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    q = torch.randn(1, H, S, D, requires_grad = True)
    k = torch.randn(1, KV, S, D, requires_grad = True)
    torch._dynamo.reset()
    try:
        out = torch.compile(_attend(), backend = "eager", fullgraph = True)(q, k, k)
    finally:
        torch._dynamo.reset()
    torch.testing.assert_close(out, _attend()(q, k, k))
    weights = _weights()
    full = _grads(weights, _attend())
    saved = _grads(weights, _attend())
    torch.testing.assert_close(saved[0], full[0])
    assert SA.SAVED_ATTENTION_STATS["reused"] == 4 and saved[2] == 2


def test_rocm_saved_attention_selector_applies_the_sdpa_check(monkeypatch):
    from torch.nn.attention import SDPBackend

    import unsloth.utils.rocm_sdpa_check as check

    monkeypatch.setattr(ad, "_ROCM_SDPA_GUARD", True)
    q = torch.empty(1, 2, 16, 64, dtype = torch.bfloat16, requires_grad = True)
    verdict = {"flash": True, "efficient": True}
    monkeypatch.setattr(check, "rocm_sdpa_verdict", lambda index = 0: verdict)
    assert ad._select_rocm_saved_sdpa(q) == int(SDPBackend.FLASH_ATTENTION)
    # Flash failed rocm_sdpa_check: never reported as a kept kernel, even when it is the only one enabled.
    verdict = {"flash": False, "efficient": True}
    assert ad._select_rocm_saved_sdpa(q) not in (
        int(SDPBackend.FLASH_ATTENTION),
        int(SDPBackend.EFFICIENT_ATTENTION),
    )
    from torch.nn.attention import sdpa_kernel

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        assert ad._select_rocm_saved_sdpa(q) == int(SDPBackend.ERROR)
