# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An explicit ``sage`` request never raises mid-generation: calls Sage cannot take run native, and the load-time check
verifies the kernel's numbers at the DiT's head dims, not just that it launched."""

from __future__ import annotations

import sys
import types

import pytest

import core.inference.diffusion_attention as att
from core.inference.diffusion_attention import apply_attention_backend

torch = pytest.importorskip("torch")
dispatch = pytest.importorskip("diffusers.models.attention_dispatch")
F = torch.nn.functional


class _Transformer:
    def __init__(self, head_dim = None):
        self.calls: list = []
        self.config = {"attention_head_dim": head_dim} if head_dim is not None else {}

    def set_attention_backend(self, name):
        self.calls.append(name)


class _Logger:
    def __init__(self):
        self.warnings: list = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def info(self, *_a, **_k):
        pass


def _target(dtype = "bf16", device = "cuda"):
    return types.SimpleNamespace(device = device, dtype = dtype, torch_device = device)


@pytest.fixture(autouse = True)
def _isolated(monkeypatch):
    monkeypatch.setattr(att, "_SAGE_PROBE_CACHE", {})
    monkeypatch.setattr(att, "_ensure_attention_backend_installed", lambda *a, **k: None)
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "native")
    monkeypatch.setattr(att, "warn_if_sdpa_math_only", lambda *a, **k: False)
    monkeypatch.setattr(att, "_indexed_cuda_device", lambda device: device)
    backends = dispatch._AttentionBackendRegistry._backends
    saved = backends[dispatch.AttentionBackendName.SAGE]
    yield
    backends[dispatch.AttentionBackendName.SAGE] = saved


def _qkv(head_dim = 64, dtype = None, heads = 2, tokens = 16):
    gen = torch.Generator().manual_seed(0)
    return tuple(
        torch.randn((1, tokens, heads, head_dim), generator = gen, dtype = dtype or torch.float32)
        for _ in range(3)
    )


def _native(q, k, v, mask = None):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask = mask
    ).transpose(1, 2)


def _dispatch_sage(q, k, v, mask = None):
    return dispatch.dispatch_attention_fn(
        q, k, v, attn_mask = mask, backend = dispatch.AttentionBackendName.SAGE
    )


def _engage_sage(monkeypatch, head_dim = 64):
    monkeypatch.setattr(att, "_run_sage_probe", lambda d, dt, hd = 128: "")
    t = _Transformer(head_dim)
    engaged = apply_attention_backend(
        types.SimpleNamespace(transformer = t), "sage", target = _target()
    )
    return engaged, t


# --- calls Sage cannot take run native instead of raising -------------------------------------------------------


def test_masked_call_runs_native_instead_of_raising(monkeypatch):
    engaged, t = _engage_sage(monkeypatch)
    assert engaged == "sage" and t.calls == ["sage"]
    q, k, v = _qkv()
    mask = torch.ones((1, 1, 1, q.shape[1]), dtype = torch.bool)
    mask[..., -3:] = False
    out = _dispatch_sage(q, k, v, mask)
    torch.testing.assert_close(out, _native(q, k, v, mask))


@pytest.mark.parametrize(
    "kind, qkv",
    [
        ("head_dim", lambda: _qkv(head_dim = 256, dtype = torch.bfloat16)),
        ("dtype", lambda: _qkv(head_dim = 64, dtype = torch.float32)),
    ],
)
def test_unservable_shape_or_dtype_runs_native(monkeypatch, kind, qkv):
    _engage_sage(monkeypatch)
    q, k, v = qkv()
    assert att._sage_reroute_reason(q, k, v, None) == kind
    out = _dispatch_sage(q, k, v)
    torch.testing.assert_close(out, _native(q, k, v))


def test_servable_call_reaches_the_sage_kernel(monkeypatch):
    backends = dispatch._AttentionBackendRegistry._backends
    calls: list = []

    def _fake_sage(query, key, value, attn_mask = None, is_causal = False, scale = None, return_lse = False,
                   _parallel_config = None):
        calls.append(tuple(query.shape))
        return "sage-ran"

    backends[dispatch.AttentionBackendName.SAGE] = _fake_sage
    assert att._install_sage_dispatch_guard() is True
    q, k, v = _qkv(head_dim = 128, dtype = torch.bfloat16)
    # Every reason except the device is clear on CPU tensors, so this CPU call runs native ...
    assert att._sage_reroute_reason(q, k, v, None) == "device"
    torch.testing.assert_close(_dispatch_sage(q, k, v), _native(q, k, v))
    assert calls == []

    # ... and the same call on a CUDA tensor is the one Sage takes.
    class _OnCuda(torch.Tensor):
        @property
        def device(self):
            return types.SimpleNamespace(type = "cuda")

    qc, kc, vc = (t.as_subclass(_OnCuda) for t in (q, k, v))
    assert att._sage_reroute_reason(qc, kc, vc, None) is None
    monkeypatch.setattr(att, "_sage_reroute_reason", lambda *a: None)
    assert _dispatch_sage(q, k, v) == "sage-ran" and calls == [tuple(q.shape)]


def test_guard_is_installed_once(monkeypatch):
    _engage_sage(monkeypatch)
    first = dispatch._AttentionBackendRegistry._backends[dispatch.AttentionBackendName.SAGE]
    assert att._install_sage_dispatch_guard() is True
    assert dispatch._AttentionBackendRegistry._backends[dispatch.AttentionBackendName.SAGE] is first


def test_rerouted_calls_are_counted_and_logged_once(monkeypatch, caplog):
    monkeypatch.setattr(att, "_SAGE_ROUTED", {})
    monkeypatch.setattr(att, "_SAGE_ROUTED_LOGGED", set())
    _engage_sage(monkeypatch)
    q, k, v = _qkv()
    mask = torch.ones((1, 1, 1, q.shape[1]), dtype = torch.bool)
    with caplog.at_level("WARNING"):
        _dispatch_sage(q, k, v, mask)
        _dispatch_sage(q, k, v, mask)
    assert att._SAGE_ROUTED == {"attn_mask": 2}
    assert sum("cannot take this attention call (attn_mask)" in r.getMessage() for r in caplog.records) == 1


def test_guarded_masked_call_traces_without_a_graph_break(monkeypatch):
    _engage_sage(monkeypatch)
    q, k, v = _qkv()
    mask = torch.ones((1, 1, 1, q.shape[1]), dtype = torch.bool)
    mask[..., :2] = False
    torch._dynamo.reset()
    compiled = torch.compile(_dispatch_sage, backend = "eager", fullgraph = True)
    torch.testing.assert_close(compiled(q, k, v, mask), _native(q, k, v, mask))
    torch._dynamo.reset()


def test_without_the_guard_sage_is_not_engaged(monkeypatch):
    monkeypatch.setattr(att, "_run_sage_probe", lambda d, dt, hd = 128: "")
    monkeypatch.setattr(att, "_install_sage_dispatch_guard", lambda: False)
    t, log = _Transformer(64), _Logger()
    assert (
        apply_attention_backend(
            types.SimpleNamespace(transformer = t), "sage", logger = log, target = _target()
        )
        is None
    )
    assert "sage" not in t.calls and any("masked attention" in w for w in log.warnings)


# --- load-time refusals ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "torch.float32", "fp32"])
def test_float32_pipeline_never_engages_sage(monkeypatch, dtype):
    seen: list = []
    monkeypatch.setattr(att, "_run_sage_probe", lambda d, dt, hd = 128: seen.append(hd) or "")
    t, log = _Transformer(128), _Logger()
    assert (
        apply_attention_backend(
            types.SimpleNamespace(transformer = t), "sage", logger = log, target = _target(dtype)
        )
        is None
    )
    assert "sage" not in t.calls and seen == []
    assert any("float32" in w for w in log.warnings)


def test_head_dims_all_above_128_never_engage_sage(monkeypatch):
    monkeypatch.setattr(att, "_run_sage_probe", lambda d, dt, hd = 128: "")
    t, log = _Transformer(256), _Logger()
    assert (
        apply_attention_backend(
            types.SimpleNamespace(transformer = t), "sage", logger = log, target = _target()
        )
        is None
    )
    assert "sage" not in t.calls and any("head_dim <= 128" in w for w in log.warnings)


def test_probe_runs_at_the_dit_head_dims(monkeypatch):
    seen: list = []
    monkeypatch.setattr(att, "_run_sage_probe", lambda d, dt, hd = 128: seen.append(hd) or "")
    t = _Transformer(64)
    assert (
        apply_attention_backend(types.SimpleNamespace(transformer = t), "sage", target = _target())
        == "sage"
    )
    assert seen == [64]


def test_missing_package_falls_back_with_a_reason_and_is_not_cached(monkeypatch):
    calls: list = []

    def _probe(d, dt, hd = 128):
        calls.append(hd)
        raise ImportError("No module named 'sageattention'")

    monkeypatch.setattr(att, "_run_sage_probe", _probe)
    t, log = _Transformer(128), _Logger()
    assert (
        apply_attention_backend(
            types.SimpleNamespace(transformer = t), "sage", logger = log, target = _target()
        )
        is None
    )
    assert "sage" not in t.calls
    assert any("could not be imported" in w for w in log.warnings)
    assert att._SAGE_PROBE_CACHE == {}
    assert att._sage_kernel_runs(_target()) is False and len(calls) == 2


# --- the self-check verifies numbers, not just a launch ----------------------------------------------------------


def _fake_sage(monkeypatch, fn):
    monkeypatch.setitem(sys.modules, "sageattention", types.SimpleNamespace(sageattn = fn))


def _exact_sageattn(q, k, v, tensor_layout = "NHD"):
    assert tensor_layout == "NHD"
    return _native(q.float(), k.float(), v.float()).to(q.dtype)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_self_check_passes_an_accurate_kernel(monkeypatch, head_dim):
    _fake_sage(monkeypatch, _exact_sageattn)
    assert att._run_sage_probe("cpu", torch.bfloat16, head_dim) == ""


@pytest.mark.parametrize(
    "wrong",
    [
        lambda q, k, v, tensor_layout = "NHD": torch.zeros_like(q),
        lambda q, k, v, tensor_layout = "NHD": v,
        lambda q, k, v, tensor_layout = "NHD": torch.full_like(q, float("nan")),
        lambda q, k, v, tensor_layout = "NHD": q[:, :1],
    ],
)
def test_self_check_rejects_a_kernel_that_runs_but_is_wrong(monkeypatch, wrong):
    _fake_sage(monkeypatch, wrong)
    error = att._run_sage_probe("cpu", torch.bfloat16, 128)
    assert error.startswith("self-check")


def test_self_check_failure_falls_back_at_load(monkeypatch):
    _fake_sage(monkeypatch, lambda q, k, v, tensor_layout = "NHD": torch.zeros_like(q))
    monkeypatch.setattr(att, "_indexed_cuda_device", lambda device: "cpu")
    t, log = _Transformer(128), _Logger()
    assert (
        apply_attention_backend(
            types.SimpleNamespace(transformer = t), "sage", logger = log, target = _target()
        )
        is None
    )
    assert "sage" not in t.calls and any("self-check" in w for w in log.warnings)


def test_probe_head_dims_ignore_dims_sage_cannot_serve():
    assert att._sage_probe_head_dims({64, 256}) == (64,)
    assert att._sage_probe_head_dims(set()) == (128,)
    assert att._sage_probe_head_dims({True, 96, 128}) == (96, 128)
