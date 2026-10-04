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
    monkeypatch.setattr(att, "_sage_version_too_old", lambda: None, raising = False)
    # The pip SageAttention 2 path; the hub path is in test_diffusion_attention_install.py.
    monkeypatch.setattr(att, "_pip_sage2_installed", lambda: True, raising = False)
    backends = dispatch._AttentionBackendRegistry._backends
    saved = backends[dispatch.AttentionBackendName.SAGE]
    saved_fa4 = backends[dispatch.AttentionBackendName.FLASH_4_HUB]
    monkeypatch.setattr(att, "_FA4_PROBE_CACHE", {}, raising = False)
    yield
    backends[dispatch.AttentionBackendName.SAGE] = saved
    backends[dispatch.AttentionBackendName.FLASH_4_HUB] = saved_fa4


def _qkv(
    head_dim = 64,
    dtype = None,
    heads = 2,
    tokens = 16,
):
    gen = torch.Generator().manual_seed(0)
    return tuple(
        torch.randn((1, tokens, heads, head_dim), generator = gen, dtype = dtype or torch.float32)
        for _ in range(3)
    )


def _native(
    q,
    k,
    v,
    mask = None,
):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask = mask
    ).transpose(1, 2)


def _dispatch_sage(
    q,
    k,
    v,
    mask = None,
):
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

    def _fake_sage(
        query,
        key,
        value,
        attn_mask = None,
        is_causal = False,
        scale = None,
        return_lse = False,
        _parallel_config = None,
    ):
        calls.append(tuple(query.shape))
        return torch.full_like(query, 7.0)

    backends[dispatch.AttentionBackendName.SAGE] = _fake_sage
    assert att._install_sage_dispatch_guard() is True
    q, k, v = _qkv(head_dim = 128, dtype = torch.bfloat16)
    # Only the device reason fires on CPU ...
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
    out = _dispatch_sage(q, k, v)
    assert calls == [tuple(q.shape)] and bool((out == 7.0).all())


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
    assert (
        sum("cannot take this attention call (attn_mask)" in r.getMessage() for r in caplog.records)
        == 1
    )


def test_guarded_masked_call_traces_without_a_graph_break(monkeypatch):
    _engage_sage(monkeypatch)
    q, k, v = _qkv()
    mask = torch.ones((1, 1, 1, q.shape[1]), dtype = torch.bool)
    mask[..., :2] = False
    # Not dispatch_attention_fn: Dynamo cannot trace diffusers' AttentionBackendName(...) lookup on torch 2.6 / 2.11.
    guarded = dispatch._AttentionBackendRegistry._backends[dispatch.AttentionBackendName.SAGE]
    torch._dynamo.reset()
    compiled = torch.compile(
        lambda q, k, v, m: guarded(query = q, key = k, value = v, attn_mask = m),
        backend = "eager",
        fullgraph = True,
    )
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

    def _probe(
        d,
        dt,
        hd = 128,
    ):
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


def _fake_sage(monkeypatch, fn):
    monkeypatch.setitem(sys.modules, "sageattention", types.SimpleNamespace(sageattn = fn))


def _exact_sageattn(
    q,
    k,
    v,
    tensor_layout = "NHD",
):
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


# Sage is CUDA-only and slower than AOTriton SDPA on gfx1151:
# https://rocm.blogs.amd.com/software-tools-optimization/comfyui-fa-backends/README.html


@pytest.mark.parametrize(
    "hip, version", [("7.2.1", "2.9.1+rocm7.2.1"), (None, "2.10.0a0+rocm7.10.0a20251116")]
)
def test_sage_is_never_selected_on_rocm(monkeypatch, hip, version):
    from core.inference.diffusion_attention import select_attention_backend

    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(torch, "__version__", version)
    rocm = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
    for speed in (True, False):
        assert select_attention_backend(rocm, "sage", speed_active = speed) is None
        assert select_attention_backend(rocm, "auto", speed_active = speed) != "sage"


def test_auto_never_selects_sage_on_nvidia(monkeypatch):
    from core.inference.diffusion_attention import select_attention_backend
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    for cap in ((7, 5), (8, 0), (8, 9), (9, 0), (10, 0), (12, 0)):
        monkeypatch.setattr(att, "_cuda_capability", lambda cap = cap: cap)
        for speed in (True, False):
            assert select_attention_backend(_target(), "auto", speed_active = speed) != "sage"


def _arch_reading_sage(
    query,
    key,
    value,
    attn_mask = None,
    is_causal = False,
    scale = None,
    return_lse = False,
    _parallel_config = None,
):
    torch._dynamo.graph_break()  # stands in for sageattn's get_cuda_arch_versions(): Python Dynamo cannot trace
    return _native(query.float(), key.float(), value.float()).to(query.dtype)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_sage_call_compiles_fullgraph_through_the_guard(monkeypatch, head_dim):
    backends = dispatch._AttentionBackendRegistry._backends
    backends[dispatch.AttentionBackendName.SAGE] = _arch_reading_sage
    assert att._install_sage_dispatch_guard() is True
    monkeypatch.setattr(att, "_sage_reroute_reason", lambda *a: None)
    guarded = backends[dispatch.AttentionBackendName.SAGE]
    q, k, v = _qkv(head_dim = head_dim, dtype = torch.bfloat16)

    def block(q, k, v):
        return guarded(query = q * 1.0, key = k, value = v).sum(-1)

    torch._dynamo.reset()
    compiled = torch.compile(block, backend = "aot_eager", fullgraph = True)
    torch.testing.assert_close(
        compiled(q, k, v), _native(q.float(), k.float(), v.float()).to(q.dtype).sum(-1)
    )
    torch._dynamo.reset()


def test_unguarded_sage_call_breaks_a_fullgraph_compile():
    """Negative control: the same function called directly cannot be traced."""
    q, k, v = _qkv(head_dim = 64, dtype = torch.bfloat16)
    torch._dynamo.reset()
    compiled = torch.compile(
        lambda q, k, v: _arch_reading_sage(q, k, v), backend = "aot_eager", fullgraph = True
    )
    with pytest.raises(Exception):
        compiled(q, k, v)
    torch._dynamo.reset()


def test_engaged_backend_is_tagged_on_every_dit(monkeypatch):
    monkeypatch.setattr(att, "_run_sage_probe", lambda d, dt, hd = 128: "")
    t, t2 = _Transformer(128), _Transformer(128)
    pipe = types.SimpleNamespace(transformer = t, transformer_2 = t2)
    assert apply_attention_backend(pipe, "sage", target = _target()) == "sage"
    assert t._unsloth_attention_backend == "sage" and t2._unsloth_attention_backend == "sage"
    # A later load on the same modules that falls back clears the tag.
    monkeypatch.setattr(
        att,
        "_run_sage_probe",
        lambda d, dt, hd = 128: "ValueError: Unsupported CUDA architecture: sm100",
    )
    monkeypatch.setattr(att, "_SAGE_PROBE_CACHE", {})
    assert apply_attention_backend(pipe, "sage", target = _target()) is None
    assert t._unsloth_attention_backend is None and t2._unsloth_attention_backend is None


_REAL_SAGE_VERSION_TOO_OLD = getattr(att, "_sage_version_too_old", None)


@pytest.mark.parametrize(
    "installed, refused", [("1.0.6", True), ("2.0.1", True), ("2.1.1", False), ("2.2.0", False)]
)
def test_sage_version_floor(monkeypatch, installed, refused):
    import importlib.metadata as md

    real = md.version
    monkeypatch.setattr(
        md, "version", lambda name: installed if name == "sageattention" else real(name)
    )
    monkeypatch.setattr(dispatch, "_REQUIRED_SAGE_VERSION", "2.1.1", raising = False)
    assert callable(_REAL_SAGE_VERSION_TOO_OLD)
    reason = _REAL_SAGE_VERSION_TOO_OLD()
    assert bool(reason) is refused
    if refused:
        assert f"sageattention {installed} is older than 2.1.1" in reason


def test_old_sageattention_is_ignored_for_the_hub_build(monkeypatch):
    # sageattention 1.0.6 is never probed; with no hub build the load keeps the default backend and says why.
    seen: list = []
    monkeypatch.setattr(att, "_run_sage_probe", lambda d, dt, hd = 128, **k: seen.append(hd) or "")
    monkeypatch.setattr(att, "_pip_sage2_installed", lambda: False)
    monkeypatch.setattr(
        att,
        "_sage_version_too_old",
        lambda: "sageattention 1.0.6 is older than 2.1.1, the "
        "SageAttention 2 release diffusers needs",
    )
    monkeypatch.setattr(att, "_load_sage_hub_kernel", lambda: (None, "no build for this torch"))
    t, log = _Transformer(128), _Logger()
    assert (
        apply_attention_backend(
            types.SimpleNamespace(transformer = t), "sage", logger = log, target = _target()
        )
        is None
    )
    assert "sage" not in t.calls and "sage_hub" not in t.calls and seen == []
    assert any(
        "no build for this torch" in w and "source build or a community wheel" in w
        for w in log.warnings
    )


def test_sm100_explicit_sage_falls_back_and_auto_never_picks_it(monkeypatch):
    from core.inference.diffusion_attention import select_attention_backend

    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    monkeypatch.setattr(att, "_cuda_capability", lambda: (10, 0))
    for speed in (True, False):
        assert select_attention_backend(_target(), "auto", speed_active = speed) == (
            "_native_cudnn" if speed else None
        )
    # Every released SageAttention 2 raises on sm100; the load keeps the default backend and says why.
    monkeypatch.setattr(
        att,
        "_run_sage_probe",
        lambda d, dt, hd = 128: "ValueError: Unsupported CUDA architecture: sm100",
    )
    t, log = _Transformer(128), _Logger()
    assert (
        apply_attention_backend(
            types.SimpleNamespace(transformer = t), "sage", logger = log, target = _target()
        )
        is None
    )
    assert any("sm100" in w for w in log.warnings)


def _engage_fa4(monkeypatch, probe = ""):
    monkeypatch.setattr(att, "_run_fa4_probe", lambda d, dt, hd = 128: probe)
    t, log = _Transformer(128), _Logger()
    engaged = apply_attention_backend(
        types.SimpleNamespace(transformer = t), "flash_4_hub", logger = log, target = _target()
    )
    return engaged, t, log


def test_fa4_masked_call_runs_native_instead_of_raising(monkeypatch):
    engaged, t, _ = _engage_fa4(monkeypatch)
    assert engaged == "flash_4_hub" and t.calls == ["flash_4_hub"]
    q, k, v = _qkv(dtype = torch.bfloat16)
    mask = torch.ones((1, 1, 1, q.shape[1]), dtype = torch.bool)
    mask[..., -3:] = False
    out = dispatch.dispatch_attention_fn(
        q, k, v, attn_mask = mask, backend = dispatch.AttentionBackendName.FLASH_4_HUB
    )
    torch.testing.assert_close(out, _native(q, k, v, mask))
    assert att._fa4_reroute_reason(q, k, v, mask) == "attn_mask"


@pytest.mark.parametrize(
    "error",
    [
        "ValueError: too many values to unpack (expected 2)",
        "RuntimeError: CUDA error: no kernel image is available for execution on the device",
        "self-check: cosine 0.10000, relative L1 0.9000 vs fp32 at head_dim 128",
    ],
)
def test_fa4_failed_probe_unpins_and_logs(monkeypatch, error):
    engaged, t, log = _engage_fa4(monkeypatch, error)
    assert engaged is None
    assert t.calls == ["flash_4_hub", "native"]
    assert getattr(t, "_unsloth_attention_backend", "unset") is None
    assert any("FlashAttention 4 does not run correctly" in w and error in w for w in log.warnings)


def test_fa4_probe_runs_at_dit_head_dims_and_is_cached(monkeypatch):
    seen: list = []
    monkeypatch.setattr(att, "_run_fa4_probe", lambda d, dt, hd = 128: seen.append(hd) or "")
    for _ in range(2):
        t = _Transformer(64)
        assert (
            apply_attention_backend(
                types.SimpleNamespace(transformer = t), "flash_4_hub", target = _target()
            )
            == "flash_4_hub"
        )
    assert seen == [64]


def test_fa4_unaskable_probe_keeps_the_request(monkeypatch):
    def _oom(
        d,
        dt,
        hd = 128,
    ):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(att, "_run_fa4_probe", _oom)
    t = _Transformer(128)
    assert (
        apply_attention_backend(
            types.SimpleNamespace(transformer = t), "flash_4_hub", target = _target()
        )
        == "flash_4_hub"
    )


def test_fa4_real_probe_reports_an_api_mismatch(monkeypatch):
    """The probe goes through diffusers' dispatch, so a kernel returning a shape the caller does not expect (the
    torch 2.12.1 + flash-attn-4 4.0.0b33 class of break) is an answer, not a crash."""

    def _five_values(
        query,
        key,
        value,
        attn_mask = None,
        scale = None,
        is_causal = False,
        return_lse = False,
        _parallel_config = None,
    ):
        a, b = (query, key, value, None, None)  # noqa: F841 - unpacks 5 into 2, as torch's FA4 hook does
        return a

    dispatch._AttentionBackendRegistry._backends[dispatch.AttentionBackendName.FLASH_4_HUB] = (
        _five_values
    )
    monkeypatch.setattr(
        dispatch, "_check_attention_backend_requirements", lambda *a, **k: None, raising = False
    )
    error = att._run_fa4_probe("cpu", torch.bfloat16, 64)
    assert error.startswith("ValueError") and "unpack" in error


def test_fa4_real_probe_passes_an_exact_kernel(monkeypatch):
    def _exact(
        query,
        key,
        value,
        attn_mask = None,
        scale = None,
        is_causal = False,
        return_lse = False,
        _parallel_config = None,
    ):
        return _native(query.float(), key.float(), value.float()).to(query.dtype)

    dispatch._AttentionBackendRegistry._backends[dispatch.AttentionBackendName.FLASH_4_HUB] = _exact
    monkeypatch.setattr(
        dispatch, "_check_attention_backend_requirements", lambda *a, **k: None, raising = False
    )
    assert att._run_fa4_probe("cpu", torch.bfloat16, 128) == ""
