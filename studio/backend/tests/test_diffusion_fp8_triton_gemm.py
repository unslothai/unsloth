# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fused Triton row-wise fp8 GEMM: routing per arch / mode, fallbacks, operand gate, interpreter numerics."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

pytest.importorskip("torchao")
from torchao.quantization.quantize_.workflows.float8 import float8_tensor as f8t  # noqa: E402

from core.inference import diffusion_compile_config as compile_config  # noqa: E402
from core.inference import diffusion_fp8_triton_gemm as tg  # noqa: E402
from core.inference import diffusion_fp8_unit_scale as us  # noqa: E402
from core.inference import diffusion_transformer_quant as tq  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]


def _sibling(name: str):
    # The fake-card helpers of the tensorwise-route tests, loaded by path (two "tests" packages share the name).
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"_fp8_triton_{name}", Path(__file__).with_name(f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rf = _sibling("test_diffusion_fp8_rowwise_fallback")


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    us.uninstall()
    compile_config._reset_for_tests()
    for name in (us.UNIT_SCALE_ENV, us.MIN_ROWS_ENV, tg.ENV):
        monkeypatch.delenv(name, raising = False)
    for k in us._COUNTS:
        us._COUNTS[k] = 0
    us._ROWWISE.clear()
    us._FP32_OUT.clear()
    for name, stub in (
        ("_persisted_rowwise", lambda index: None),
        ("_persist_rowwise", lambda index, verdict: None),
        ("_has_context", lambda index: False),
        ("_current_device", lambda: 0),
    ):
        monkeypatch.setattr(us, name, stub)
    from core import _torchao_stub

    monkeypatch.setattr(_torchao_stub, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(_torchao_stub, "is_stubbed", lambda name: False)
    original = f8t.addmm_float8_unwrapped_inference
    yield
    us.uninstall()
    compile_config._reset_for_tests()
    us._ROWWISE.clear()
    us._FP32_OUT.clear()
    f8t.addmm_float8_unwrapped_inference = original


class _Kernel:
    """Stands in for the Triton kernel on a CPU host: records calls, returns the dequantised reference."""

    def __init__(
        self,
        monkeypatch,
        *,
        verify = True,
        eager_result = "ref",
        traced = True,
    ):
        self.eager, self.traced, self.verified = [], [], []
        monkeypatch.setattr(tg, "available", lambda: True)
        monkeypatch.setattr(tg, "traced_available", lambda: traced)
        monkeypatch.setattr(tg, "eligible", lambda a, sa, b, sb, bias, dt: True)

        def fake_verify(index, dt):
            self.verified.append((index, dt))
            return verify

        def fake_eager(a, sa, b, sb, bias, dt):
            self.eager.append(dt)
            if eager_result is None:
                return None
            return rf._reference(a.t, sa, b, sb, bias).to(dt)

        def fake_traced(a, sa, b, sb, bias, dt):
            self.traced.append(dt)
            return rf._reference(a.t, sa, b, sb, bias).to(dt)

        monkeypatch.setattr(tg, "verify", fake_verify)
        monkeypatch.setattr(
            tg, "verified", lambda index, dt: verify and (index, dt) in self.verified
        )
        monkeypatch.setattr(tg, "eager_gemm", fake_eager)
        monkeypatch.setattr(tg, "traced_gemm", fake_traced)


def _call(
    m = 32,
    bias = False,
    out = torch.bfloat16,
    fast = True,
):
    a, sa, b, sb = rf._operands(m)
    bv = torch.randn(48).to(torch.bfloat16) if bias else None
    return f8t.addmm_float8_unwrapped_inference(
        rf._Dev(a), sa, b, sb, out, bias = bv, use_fast_accum = fast
    ), (a, sa, b, sb, bv)


@pytest.mark.parametrize("cap", [(12, 0), (12, 1), (8, 9)])
@pytest.mark.parametrize("bias", [False, True])
def test_fallback_card_runs_the_fused_kernel_in_eager(monkeypatch, cap, bias):
    calls = rf._fake_card(monkeypatch, cap, rowwise_ok = False)
    k = _Kernel(monkeypatch)
    tq._make_quant_config(tq.TQ_FP8)
    assert k.verified == [(0, torch.bfloat16)]
    calls.clear()
    got, (a, sa, b, sb, bv) = _call(bias = bias)
    assert k.eager == [torch.bfloat16] and calls == []  # no tensorwise GEMM, no epilogue
    assert torch.equal(got, rf._reference(a, sa, b, sb, bv).to(torch.bfloat16))
    assert us.stats()["triton_gemm"] == 1 and us.stats()["unit_scale"] == 0
    status = us.gemm_path(0)
    assert status["path"] == "unit_scale_fallback"
    assert status["triton_gemm"] == {"eager": True, "compiled": cap[0] == 12}


@pytest.mark.parametrize("cap,compiled", [((12, 0), True), ((12, 1), True), ((8, 9), False)])
def test_traced_calls_take_the_triton_op_on_sm12x_only(monkeypatch, cap, compiled):
    # sm_89: the compiled fallback is as fast or faster at 512 rows (L4), so a trace keeps the tensorwise GEMM there.
    calls = rf._fake_card(monkeypatch, cap, rowwise_ok = False)
    k = _Kernel(monkeypatch)
    tq._make_quant_config(tq.TQ_FP8)
    calls.clear()
    monkeypatch.setattr(us, "_traced", lambda t: True)
    monkeypatch.setattr(us, "_compiling", lambda: True)
    _call()
    assert k.eager == []
    if compiled:
        assert k.traced == [torch.bfloat16] and calls == []
    else:
        assert k.traced == [] and calls == ["tensorwise32"]


def test_kill_switch_keeps_the_tensorwise_route_and_its_tag(monkeypatch):
    import torch.compiler.config as ccfg
    from core.inference import diffusion_compile_cache as cc

    monkeypatch.setattr(ccfg, "cache_key_tag", "")
    monkeypatch.setenv(tg.ENV, "0")
    assert cc.graph_switches().get(tg.ENV) == "off"
    calls = rf._fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    k = _Kernel(monkeypatch)
    tq._make_quant_config(tq.TQ_FP8)
    assert k.verified == []
    assert ccfg.cache_key_tag == "unsloth_fp8_unit_scale=256:rowwise-missing=0:acc32=0"
    calls.clear()
    _call()
    assert calls == ["tensorwise32"] and k.eager == []
    assert us.gemm_path(0)["triton_gemm"] == {"eager": False, "compiled": False}


def test_compiled_kernel_keys_the_compile_cache_on_its_source(monkeypatch):
    import torch.compiler.config as ccfg

    monkeypatch.setattr(ccfg, "cache_key_tag", "")
    rf._fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    _Kernel(monkeypatch)
    tq._make_quant_config(tq.TQ_FP8)
    digest = tg.source_digest()
    assert len(digest) == 12 and digest != "unknown"
    assert ccfg.cache_key_tag == (
        f"unsloth_fp8_unit_scale=256:rowwise-missing=0:acc32=0:gemm=triton-0-{digest}"
    )


def test_eager_only_card_keeps_the_fallback_tag(monkeypatch):
    # sm_89 compiles the same graph as before: its cache key must not move.
    import torch.compiler.config as ccfg

    monkeypatch.setattr(ccfg, "cache_key_tag", "")
    rf._fake_card(monkeypatch, (8, 9), rowwise_ok = False)
    _Kernel(monkeypatch)
    tq._make_quant_config(tq.TQ_FP8)
    assert ccfg.cache_key_tag == "unsloth_fp8_unit_scale=256:rowwise-missing=0:acc32=0"


def test_no_triton_op_keeps_compiled_on_the_fallback(monkeypatch):
    import torch.compiler.config as ccfg

    monkeypatch.setattr(ccfg, "cache_key_tag", "")
    rf._fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    k = _Kernel(monkeypatch, traced = False)
    tq._make_quant_config(tq.TQ_FP8)
    assert ccfg.cache_key_tag == "unsloth_fp8_unit_scale=256:rowwise-missing=0:acc32=0"
    assert us.gemm_path(0)["triton_gemm"] == {"eager": True, "compiled": False}
    _call()
    assert k.eager == [torch.bfloat16]


@pytest.mark.parametrize("cap", [(10, 0), (9, 0), (10, 3)])
def test_unmeasured_arches_keep_the_tensorwise_route(monkeypatch, cap):
    calls = rf._fake_card(monkeypatch, cap, rowwise_ok = False)
    k = _Kernel(monkeypatch)
    tq._make_quant_config(tq.TQ_FP8)
    assert k.verified == []
    calls.clear()
    _call()
    assert calls == ["tensorwise32"] and k.eager == []


def test_failed_self_check_keeps_the_tensorwise_route(monkeypatch):
    calls = rf._fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    k = _Kernel(monkeypatch, verify = False)
    tq._make_quant_config(tq.TQ_FP8)
    calls.clear()
    _call()
    assert calls == ["tensorwise32"] and k.eager == []


def test_a_declined_eager_launch_falls_back_to_the_tensorwise_gemm(monkeypatch):
    # eager_gemm returns None (capture before tuning, a failed launch, an OOM while autotuning): same call, old route.
    calls = rf._fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    k = _Kernel(monkeypatch, eager_result = None)
    tq._make_quant_config(tq.TQ_FP8)
    calls.clear()
    got, (a, sa, b, sb, bv) = _call()
    assert k.eager == [torch.bfloat16] and calls == ["tensorwise32"]
    assert us.stats()["triton_gemm"] == 0 and us.stats()["unit_scale"] == 1
    want = rf._reference(a, sa, b, sb, bv)
    assert (got.float() - want).abs().max() <= 2**-7 * want.abs().max()


def test_rocm_never_loads_the_kernel(monkeypatch):
    from core import _torchao_stub

    rf._fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    monkeypatch.setenv(us.UNIT_SCALE_ENV, "fallback")
    k = _Kernel(monkeypatch)
    monkeypatch.setattr(_torchao_stub, "torch_is_rocm", lambda: True)
    assert us._triton_cards(frozenset({0}), {0: (12, 0)}) == (frozenset(), frozenset(), None)
    assert k.verified == []


def test_rowwise_cards_never_reach_the_kernel(monkeypatch):
    # Linux (row-wise kernel present): these cards run byte for byte what they ran before.
    seen = []
    monkeypatch.setattr(
        us, "_triton_cards", lambda *a, **k: seen.append(a) or (frozenset(), frozenset(), None)
    )
    rf._fake_card(monkeypatch, (12, 0), rowwise_ok = True)
    tq._make_quant_config(tq.TQ_FP8)
    assert us.stats()["installed"] is False
    us.uninstall()
    rf._fake_card(monkeypatch, (8, 9), rowwise_ok = True)
    us.install(fast_accum = False)
    assert us.stats()["installed"] is True and us.fallback_cards() == frozenset()
    assert all(not in_use for in_use, *_ in seen)


# ---------------------------------------------------------------- the real gate and arch table (no stubs)


def _meta_operands(
    m,
    k,
    n,
    *,
    a_dtype = torch.float8_e4m3fn,
    col_major = True,
    bias = None,
):
    a = torch.empty((m, k), device = "meta", dtype = a_dtype)
    w = torch.empty((n, k), device = "meta", dtype = torch.float8_e4m3fn)
    b = w.t() if col_major else torch.empty((k, n), device = "meta", dtype = torch.float8_e4m3fn)
    sa = torch.empty((m, 1), device = "meta")
    sb = torch.empty((n, 1), device = "meta").t()
    bv = torch.empty((n,), device = "meta", dtype = torch.bfloat16) if bias else None
    return a, sa, b, sb, bv


def test_eligible_gate():
    E = tg.eligible
    a, sa, b, sb, bv = _meta_operands(4608, 3072, 18432, bias = True)
    assert E(a, sa, b, sb, bv, torch.bfloat16) and E(a, sa, b, sb, None, torch.float16)
    assert not E(a, sa, b, sb, bv, torch.float32)
    assert not E(
        *_meta_operands(77, 3072, 3072, a_dtype = torch.float8_e5m2)[:4], None, torch.bfloat16
    )
    assert not E(*_meta_operands(77, 3072, 3072, col_major = False)[:4], None, torch.bfloat16)
    a2, sa2, b2, sb2, _ = _meta_operands(77, 3072, 3072)
    assert not E(
        a2.t().contiguous().t(), sa2, b2, sb2, None, torch.bfloat16
    )  # column-major activation
    assert not E(
        a2, sa2, b2, torch.empty((1, 1), device = "meta"), None, torch.bfloat16
    )  # tensorwise weight scale
    assert not E(a2, sa2, b2, sb2, torch.empty((3071,), device = "meta"), torch.bfloat16)
    # 32-bit offsets: an output past 2**31 elements keeps the fallback
    big = _meta_operands(80000, 3072, 27648)
    assert not E(*big[:4], None, torch.bfloat16)
    assert E(*_meta_operands(70000, 3072, 27648)[:4], None, torch.bfloat16)


def test_card_paths_table():
    assert tg.card_paths((8, 9)) == (True, False)
    assert tg.card_paths((12, 0)) == (True, True)
    assert tg.card_paths((12, 1)) == (True, True)
    for cap in ((8, 6), (8, 0), (9, 0), (10, 0), (10, 3), (11, 0), (7, 5), (9, 4), ()):
        assert tg.card_paths(cap) == (False, False)


def test_kill_switch_words(monkeypatch):
    for word in ("0", "off", "false", "no", "none", " OFF "):
        monkeypatch.setenv(tg.ENV, word)
        assert not tg.enabled()
    for word in ("", "1", "on", "auto"):
        monkeypatch.setenv(tg.ENV, word)
        assert tg.enabled()


def test_bucket_bounds_autotune_keys():
    assert [tg._bucket(m) for m in (1, 2, 3, 512, 513, 4608, 8192, 9000, 70000)] == [
        1,
        2,
        4,
        512,
        1024,
        8192,
        8192,
        8192,
        8192,
    ]


def test_eager_gemm_declines_an_untuned_shape_inside_a_capture(monkeypatch):
    if not tg.available():
        pytest.skip("no triton")
    monkeypatch.setattr(tg, "_capturing", lambda device: True)
    monkeypatch.setitem(tg._STATE, "failed", {})
    monkeypatch.setitem(tg._STATE, "verified", {(0, torch.bfloat16): True})
    monkeypatch.setitem(tg._STATE, "tuned", set())
    launched = []
    monkeypatch.setattr(tg, "_launch", lambda *a: launched.append(a))
    a, sa, b, sb, _ = _meta_operands(512, 3072, 3072)

    class _A:
        device = type("D", (), {"index": 0, "type": "cuda"})()
        shape = a.shape
        dtype = a.dtype

        @staticmethod
        def stride(i):
            return a.stride(i)

    assert tg.eager_gemm(_A(), sa, b, sb, None, torch.bfloat16) is None and launched == []
    # unverified dtype inside a capture: no self-check is run there either
    monkeypatch.setitem(tg._STATE, "verified", {})
    monkeypatch.setattr(tg, "verify", lambda *a: launched.append("verify") or True)
    assert tg.eager_gemm(_A(), sa, b, sb, None, torch.float16) is None and launched == []


# ---------------------------------------------------------------- kernel numbers, Triton interpreter (CPU)

_INTERP = textwrap.dedent(
    """
    import sys, torch
    sys.path.insert(0, sys.argv[1])
    from core.inference import diffusion_fp8_triton_gemm as tg
    assert tg._INTERPRET and tg.available(), tg._STATE["error"]
    def bf16_ulps(got, want):
        # bf16 steps between got and want (the interpreter casts fp32 -> bf16 by truncation, not round-to-nearest)
        g = got.view(torch.int16).to(torch.int32)
        w = want.view(torch.int16).to(torch.int32)
        return int((g - w).abs().max())
    bad = []
    # 1. exact operands: every config, bias / none, fp16 out must match the fp32 reference bit for bit
    a, sa, b, sb, bias = tg._check_operands("cpu")
    ref = (a.float() * sa) @ (b.float() * sb)
    for cfg in tg._fp8_rowwise_gemm.configs:
        for bb in (None, bias.to(torch.bfloat16)):
            want = ref + bb.float() if bb is not None else ref
            got = tg._launch(tg._ConfigLaunch(cfg), a, sa, b, sb, bb, torch.float16)
            if not torch.equal(got, want.to(torch.float16)):
                bad.append(("fp16", str(cfg), bb is not None))
            got = tg._launch(tg._ConfigLaunch(cfg), a, sa, b, sb, bb, torch.bfloat16)
            if bf16_ulps(got, want.to(torch.bfloat16)) > 1:
                bad.append(("bf16", str(cfg), bb is not None))
    # 2. random e4m3 operands with real scales, odd M, K and N off every block size
    g = torch.Generator().manual_seed(0)
    for m, k, n in ((1, 48, 64), (37, 112, 80), (130, 240, 272)):
        x = torch.randn(m, k, generator=g) * 3
        w = torch.randn(n, k, generator=g) * 0.02
        sa = x.abs().amax(1, keepdim=True) / 448
        sw = w.abs().amax(1, keepdim=True) / 448
        aq = (x / sa).to(torch.float8_e4m3fn)
        bq = (w / sw).to(torch.float8_e4m3fn).t()
        sb = sw.t().contiguous()
        bias = torch.randn(n, generator=g).to(torch.bfloat16)
        ref = ((aq.double() * sa.double()) @ (bq.double() * sb.double()) + bias.double()).float()
        cfg = tg._fp8_rowwise_gemm.configs[3]
        got = tg._launch(tg._ConfigLaunch(cfg), aq, sa, bq, sb, bias, torch.float16).float()
        # fp32 accumulation vs fp64: within a couple of fp16 roundings of the output
        tol = 2 * 2.0 ** -10 * ref.abs().max().item()
        err = (got - ref).abs().max().item()
        if not err <= tol:
            bad.append(("random", m, k, n, err, tol))
    print("BAD", bad)
    sys.exit(1 if bad else 0)
    """
)


def test_kernel_matches_the_reference_under_the_triton_interpreter(tmp_path):
    pytest.importorskip("triton")
    script = tmp_path / "interp.py"
    script.write_text(_INTERP)
    env = {**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.run(
        [sys.executable, "-I", str(script), str(BACKEND)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 900,
    )
    assert p.returncode == 0, (p.stdout + p.stderr)[-4000:]


def test_triton_op_registers_and_traces_with_fake_tensors():
    # A triton_op with a real schema whose fake run gives the right shape / dtype.
    if not tg.available() or not hasattr(torch.library, "triton_op"):
        pytest.skip("no triton / torch.library.triton_op")
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    op = tg._build_op()
    assert op is not None, tg.describe()
    assert str(op._schema) == (
        "unsloth_studio::fp8_rowwise_gemm(Tensor a, Tensor a_scale, Tensor b, Tensor b_scale, Tensor? bias, "
        "ScalarType output_dtype) -> Tensor"
    )
    with FakeTensorMode():
        a = torch.empty(77, 208, dtype = torch.float8_e4m3fn)
        w = torch.empty(144, 208, dtype = torch.float8_e4m3fn)
        sa, sb = torch.empty(77, 1), torch.empty(144, 1).t()
        out = op(a, sa, w.t(), sb, torch.empty(144, dtype = torch.bfloat16), torch.bfloat16)
        assert out.shape == (77, 144) and out.dtype == torch.bfloat16
        assert op(a, sa, w.t(), sb, None, torch.float16).dtype == torch.float16

    def f(a, sa, b, sb):
        return op(a, sa, b, sb, None, torch.bfloat16)

    a = torch.zeros(77, 208).to(torch.float8_e4m3fn)
    w = torch.zeros(144, 208).to(torch.float8_e4m3fn)
    g = make_fx(f, tracing_mode = "symbolic")(a, torch.ones(77, 1), w.t(), torch.ones(1, 144))
    assert "unsloth_studio.fp8_rowwise_gemm" in g.code


def test_precise_accumulate_keeps_the_tensorwise_gemm(monkeypatch):
    # The kernel was matched against cuBLAS's fast accumulate; a precise-accumulate request keeps cuBLAS's own.
    calls = rf._fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    k = _Kernel(monkeypatch)
    tq._make_quant_config(tq.TQ_FP8)
    calls.clear()
    _call(fast = False)
    assert calls == ["tensorwise32"] and k.eager == []


def test_traced_call_needs_a_self_checked_dtype(monkeypatch):
    # No launch can run inside a trace, so a dtype the install did not check (fp16 here) keeps the fallback there.
    calls = rf._fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    k = _Kernel(monkeypatch)
    tq._make_quant_config(tq.TQ_FP8)
    calls.clear()
    monkeypatch.setattr(us, "_traced", lambda t: True)
    monkeypatch.setattr(us, "_compiling", lambda: True)
    _call(out = torch.float16)
    assert k.traced == [] and calls == ["tensorwise32"]
    _call(out = torch.bfloat16)
    assert k.traced == [torch.bfloat16]


def test_install_survives_a_failing_kernel_setup(monkeypatch):
    calls = rf._fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    _Kernel(monkeypatch)

    def boom(cap):
        raise RuntimeError("boom")

    monkeypatch.setattr(tg, "card_paths", boom)
    tq._make_quant_config(tq.TQ_FP8)
    assert us.stats()["installed"] is True and us.fallback_cards() == frozenset({0})
    calls.clear()
    _call()
    assert calls == ["tensorwise32"]


def test_verify_oom_is_no_verdict(monkeypatch):
    if not tg.available():
        pytest.skip("no triton")
    import contextlib

    monkeypatch.setitem(tg._STATE, "verified", {})
    monkeypatch.setitem(tg._STATE, "failed", {})
    monkeypatch.setitem(tg._STATE, "check", {})

    def oom(device):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(tg, "_check_operands", oom)
    monkeypatch.setattr(torch.cuda, "device", lambda index: contextlib.nullcontext())
    assert tg.verify(0, torch.bfloat16) is False
    assert (0, torch.bfloat16) not in tg._STATE["verified"]  # checked again next time
    assert not tg._STATE["failed"]


def test_autotune_key_tells_apart_what_triton_does():
    # A capture launches only an already-seen autotune / specialization key (dtypes and stride classes included).
    a, sa, b, sb, _ = _meta_operands(512, 3072, 3072)

    class _A:
        device = type("D", (), {"index": 0})()
        shape = a.shape
        dtype = a.dtype

        @staticmethod
        def stride(i):
            return a.stride(i)

    bf = torch.empty((3072,), device = "meta", dtype = torch.bfloat16)
    f32 = torch.empty((3072,), device = "meta", dtype = torch.float32)
    keys = {
        tg._key(_A, sa, b, sb, None, torch.bfloat16),
        tg._key(_A, sa, b, sb, bf, torch.bfloat16),
        tg._key(_A, sa, b, sb, f32, torch.bfloat16),
        tg._key(_A, sa, b, sb, bf, torch.float16),
    }
    assert len(keys) == 4
    assert tg._key(_A, sa, b, sb, bf, torch.bfloat16) == tg._key(
        _A, sa, b, sb, bf.clone(), torch.bfloat16
    )


def test_eligible_bounds_strided_offsets():
    # A row-padded activation view: sizes fit 32 bits, the offsets the kernel forms do not.
    base = torch.empty((40000, 60000), device = "meta", dtype = torch.float8_e4m3fn)
    a = base[:, :3072]
    w = torch.empty((3072, 3072), device = "meta", dtype = torch.float8_e4m3fn)
    sa = torch.empty((40000, 1), device = "meta")
    sb = torch.empty((3072, 1), device = "meta").t()
    assert a.numel() < 2**31 and (a.shape[0] - 1) * a.stride(0) > 2**31
    assert not tg.eligible(a, sa, w.t(), sb, None, torch.bfloat16)
    assert tg.eligible(a.contiguous(), sa, w.t(), sb, None, torch.bfloat16)
