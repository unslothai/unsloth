# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fused Triton row-wise fp8 GEMM: routing per arch / mode, fallbacks, operand gate, interpreter numerics."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import types
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
_REAL_PERSISTED, _REAL_PERSIST = tg._persisted, tg._persist
_CHECK_OPERANDS = tg._check_operands


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
    # No on-disk verdicts unless a test asks for them (``verdict_cache``).
    monkeypatch.setattr(tg, "_persisted", lambda index, slot: None)
    monkeypatch.setattr(tg, "_persist", lambda index, slot, value: None)
    monkeypatch.setitem(tg._STATE, "persisted", set())
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
        fast = True,
        precise = False,
    ):
        self.eager, self.traced, self.verified = [], [], []
        # Accuracy verdict per accumulate mode (fast, precise), any dtype.
        self.kinds = {True: fast, False: precise}
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
        monkeypatch.setattr(tg, "verify_accuracy", lambda index: any(self.kinds.values()))
        monkeypatch.setattr(tg, "accurate", lambda index, dt, f: self.kinds[bool(f)])
        monkeypatch.setattr(tg, "accuracy_pending", lambda index: False)
        monkeypatch.setattr(tg, "card_failed", lambda index: False)


def _status(
    eager,
    compiled,
    *,
    fast = True,
    precise = False,
    kinds = None,
):
    if kinds is None:
        kinds = ([f"0bf16{m}" for m, on in (("f", fast), ("p", precise)) if on]) if compiled else []
    return {
        "eager": eager,
        "compiled": compiled,
        "fast_accumulate": fast and eager,
        "precise_accumulate": precise and eager,
        "compiled_kinds": sorted(kinds),
    }


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
    assert status["triton_gemm"] == _status(True, cap[0] == 12)


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
    assert us.gemm_path(0)["triton_gemm"] == _status(False, False, fast = False)


def test_compiled_kernel_keys_the_compile_cache_on_its_source(monkeypatch):
    import torch.compiler.config as ccfg

    monkeypatch.setattr(ccfg, "cache_key_tag", "")
    rf._fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    _Kernel(monkeypatch)
    tq._make_quant_config(tq.TQ_FP8)
    digest = tg.source_digest()
    assert len(digest) == 12 and digest != "unknown"
    assert ccfg.cache_key_tag == (
        f"unsloth_fp8_unit_scale=256:rowwise-missing=0:acc32=0:gemm=triton-0-{digest}:kinds=0bf16f"
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
    assert us.gemm_path(0)["triton_gemm"] == _status(True, False)
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
    assert us._triton_cards(frozenset({0}), {0: (12, 0)}) == (
        frozenset(),
        frozenset(),
        frozenset(),
        None,
    )
    assert k.verified == []


def test_rowwise_cards_never_reach_the_kernel(monkeypatch):
    # Linux (row-wise kernel present): these cards run byte for byte what they ran before.
    seen = []
    monkeypatch.setattr(
        us,
        "_triton_cards",
        lambda *a, **k: seen.append(a) or (frozenset(), frozenset(), frozenset(), None),
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


@pytest.mark.parametrize("fast_ok,precise_ok", [(True, False), (False, True), (False, False)])
def test_each_accumulate_mode_takes_the_kernel_on_its_own_accuracy_verdict(
    monkeypatch, fast_ok, precise_ok
):
    # Fast and precise accumulate requests are gated separately, eager and traced: a mode whose accuracy check failed
    # keeps the tensorwise cuBLAS GEMM, the other one still takes the kernel.
    calls = rf._fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    k = _Kernel(monkeypatch, fast = fast_ok, precise = precise_ok)
    tq._make_quant_config(tq.TQ_FP8)
    status = us.gemm_path(0)["triton_gemm"]
    assert status == _status(True, fast_ok or precise_ok, fast = fast_ok, precise = precise_ok)
    for traced in (False, True):
        _traced_calls(monkeypatch, traced)
        for fast, ok in ((True, fast_ok), (False, precise_ok)):
            calls.clear()
            k.eager.clear()
            k.traced.clear()
            _call(fast = fast)
            took = k.traced if traced else k.eager
            assert (took == [torch.bfloat16]) is ok and (calls == []) is ok, (traced, fast)


@pytest.mark.parametrize("cap,compiled", [((12, 0), True), ((12, 1), True), ((8, 9), False)])
@pytest.mark.parametrize("meta_fast", [False, None])
def test_prequant_checkpoint_load_routes_precise_gemms_to_the_kernel(
    monkeypatch, cap, compiled, meta_fast
):
    # Studio's own FP8 load seeds the hosted pre-quantized denoiser (unsloth/FLUX.2-klein-4B-FP8): its Float8Tensors
    # are baked with Float8MMConfig(use_fast_accum=False), and the dispatcher is installed from the checkpoint, not
    # from _make_quant_config. On a card whose accuracy check passed precise accumulate, those GEMMs must take the
    # kernel, eager and (sm_12x) traced, exactly like the quantize_ route.
    from core.inference import diffusion_prequant as dp

    calls = rf._fake_card(monkeypatch, cap, rowwise_ok = False)
    k = _Kernel(monkeypatch, precise = True)
    meta = {"scheme": "fp8"}
    if meta_fast is not None:
        meta["fast_accum"] = meta_fast
    dp._install_fp8_unit_scale_for({"metadata": meta, "state_dict": {}})
    assert us.stats()["installed"] is True and us.fallback_cards() == frozenset({0})
    assert us.gemm_path(0)["triton_gemm"] == _status(True, compiled, precise = True)
    calls.clear()
    _call(fast = False)
    assert k.eager == [torch.bfloat16] and calls == []
    monkeypatch.setattr(us, "_traced", lambda t: True)
    monkeypatch.setattr(us, "_compiling", lambda: True)
    _call(fast = False)
    if compiled:
        assert k.traced == [torch.bfloat16] and calls == []
    else:
        assert k.traced == [] and calls == ["tensorwise32"]


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


def test_traced_kinds_are_frozen_at_install_and_named_in_the_tag(monkeypatch):
    # An fp16 self-check that passes after install (from an eager fp16 call) must not change what a trace emits under
    # the same compile cache tag: the next install picks it up and names it.
    import torch.compiler.config as ccfg

    monkeypatch.setattr(ccfg, "cache_key_tag", "")
    calls = rf._fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    k = _Kernel(monkeypatch)
    tq._make_quant_config(tq.TQ_FP8)
    tag = ccfg.cache_key_tag
    assert tag.endswith(":kinds=0bf16f")
    k.verified.append((0, torch.float16))  # the eager fp16 self-check passed since
    _traced_calls(monkeypatch, True)
    calls.clear()
    _call(out = torch.float16)
    assert k.traced == [] and calls == ["tensorwise32"] and ccfg.cache_key_tag == tag
    _traced_calls(monkeypatch, False)
    us.uninstall()
    us.install()
    assert ccfg.cache_key_tag.endswith(":kinds=0bf16f+0fp16f")
    assert us.gemm_path(0)["triton_gemm"]["compiled_kinds"] == ["0bf16f", "0fp16f"]
    _traced_calls(monkeypatch, True)
    _call(out = torch.float16)
    assert k.traced == [torch.float16]


def test_traced_call_keeps_the_fallback_after_a_launch_failure(monkeypatch):
    calls = rf._fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    k = _Kernel(monkeypatch)
    tq._make_quant_config(tq.TQ_FP8)
    monkeypatch.setattr(tg, "card_failed", lambda index: True)
    _traced_calls(monkeypatch, True)
    calls.clear()
    _call()
    assert k.traced == [] and calls == ["tensorwise32"]


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
    assert tg.check_pending(0, torch.bfloat16) is True  # the dispatcher keeps the card a candidate
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


class _Deferred:
    """A self-check (bf16 or accuracy) that hits an OOM until ``passes()``: no verdict meanwhile, then a pass."""

    def __init__(
        self,
        monkeypatch,
        *,
        bf16_oom = True,
        precise_oom = False,
    ):
        self.bf16_ok, self.precise_ok = not bf16_oom, not precise_oom
        self.bf16_calls, self.precise_calls = [], []
        self.capturing = False
        self.precise_known = set()
        monkeypatch.setattr(us, "_TRITON_RETRY_S", 0.0, raising = False)

        def verify(index, dt):
            self.bf16_calls.append((index, dt))
            return self.bf16_ok

        def verify_accuracy(index):
            self.precise_calls.append(index)
            if self.precise_ok:
                self.precise_known.add(index)
            return self.precise_ok

        monkeypatch.setattr(tg, "verify", verify)
        monkeypatch.setattr(tg, "verified", lambda index, dt: self.bf16_ok and dt == torch.bfloat16)
        monkeypatch.setattr(tg, "check_pending", lambda index, dt: not self.bf16_ok, raising = False)
        monkeypatch.setattr(tg, "verify_accuracy", verify_accuracy)
        monkeypatch.setattr(tg, "accurate", lambda index, dt, fast: index in self.precise_known)
        monkeypatch.setattr(
            tg,
            "accuracy_pending",
            lambda index: self.bf16_ok and index not in self.precise_known,
            raising = False,
        )
        monkeypatch.setattr(tg, "_capturing", lambda device: self.capturing)

    def passes(self):
        self.bf16_ok = self.precise_ok = True


def _traced_calls(monkeypatch, on):
    monkeypatch.setattr(us, "_traced", lambda t: on)
    monkeypatch.setattr(us, "_compiling", lambda: on)


def test_oom_deferred_self_check_retries_from_an_eager_call(monkeypatch):
    # Codex P2: an OOM in the install-time self-check left the card out of the dispatcher for the whole process.
    import torch.compiler.config as ccfg

    monkeypatch.setattr(ccfg, "cache_key_tag", "")
    calls = rf._fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    k = _Kernel(monkeypatch)
    d = _Deferred(monkeypatch)
    tq._make_quant_config(tq.TQ_FP8)
    fallback_tag = "unsloth_fp8_unit_scale=256:rowwise-missing=0:acc32=0"
    assert ccfg.cache_key_tag == fallback_tag
    assert us.gemm_path(0)["triton_gemm"]["eager"] is False
    calls.clear()
    _call()  # still OOM: the tensorwise GEMM, a retry was made
    assert calls == ["tensorwise32"] and k.eager == [] and len(d.bf16_calls) == 2
    d.passes()
    calls.clear()
    _call()
    assert k.eager == [torch.bfloat16] and calls == []
    # graphs traced from here on keep the fallback until the next install
    assert us.gemm_path(0)["triton_gemm"] == _status(True, False, precise = True)
    assert ccfg.cache_key_tag == fallback_tag
    _traced_calls(monkeypatch, True)
    calls.clear()
    _call()
    assert k.traced == [] and calls == ["tensorwise32"]
    _traced_calls(monkeypatch, False)
    # The next install (the next load) rebuilds the dispatcher: compiled path and its cache tag.
    us.install()
    assert us.gemm_path(0)["triton_gemm"] == _status(True, True, precise = True)
    assert ccfg.cache_key_tag == (
        f"{fallback_tag}:gemm=triton-0-{tg.source_digest()}:kinds=0bf16f+0bf16p"
    )
    _traced_calls(monkeypatch, True)
    _call()
    assert k.traced == [torch.bfloat16]


def test_deferred_self_check_never_retries_inside_a_capture_or_a_trace(monkeypatch):
    calls = rf._fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    k = _Kernel(monkeypatch)
    d = _Deferred(monkeypatch)
    tq._make_quant_config(tq.TQ_FP8)
    assert len(d.bf16_calls) == 1
    d.passes()  # memory is free now, but no retry may run here:
    _traced_calls(monkeypatch, True)
    _call()
    d.capturing = True
    _traced_calls(monkeypatch, False)
    _call()
    assert len(d.bf16_calls) == 1 and k.eager == [] and k.traced == []
    d.capturing = False
    calls.clear()
    _call()
    assert len(d.bf16_calls) == 2 and k.eager == [torch.bfloat16] and calls == []


def test_deferred_self_check_backs_off_between_retries(monkeypatch):
    rf._fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    _Kernel(monkeypatch)
    d = _Deferred(monkeypatch)
    monkeypatch.setattr(us, "_TRITON_RETRY_S", 3600.0, raising = False)
    tq._make_quant_config(tq.TQ_FP8)
    for _ in range(5):
        _call()
    assert len(d.bf16_calls) == 2  # install + one eager retry, then the back-off holds


def test_oom_deferred_accuracy_check_retries_from_an_eager_call(monkeypatch):
    # The self-check passed but the accuracy check hit an OOM: no request kind has a verdict, so fast and precise
    # requests alike keep the tensorwise GEMM (eager and traced) until a retry measures it.
    import torch.compiler.config as ccfg

    monkeypatch.setattr(ccfg, "cache_key_tag", "")
    calls = rf._fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    k = _Kernel(monkeypatch)
    d = _Deferred(monkeypatch, bf16_oom = False, precise_oom = True)
    tq._make_quant_config(tq.TQ_FP8)
    tag = "unsloth_fp8_unit_scale=256:rowwise-missing=0:acc32=0"
    assert ccfg.cache_key_tag == tag
    assert us.gemm_path(0)["triton_gemm"] == _status(True, False, fast = False)
    for fast in (True, False):
        calls.clear()
        _call(fast = fast)
        assert calls == ["tensorwise32"] and k.eager == []
    d.passes()
    calls.clear()
    _call(fast = False)
    assert k.eager == [torch.bfloat16] and calls == []
    # A GEMM traced now keeps the fallback the cache tag names; the next install moves both.
    _traced_calls(monkeypatch, True)
    for fast in (True, False):
        calls.clear()
        _call(fast = fast)
        assert k.traced == [] and calls == ["tensorwise32"]
    _traced_calls(monkeypatch, False)
    us.install()
    assert ccfg.cache_key_tag == f"{tag}:gemm=triton-0-{tg.source_digest()}:kinds=0bf16f+0bf16p"
    _traced_calls(monkeypatch, True)
    _call(fast = False)
    _call(fast = True)
    assert k.traced == [torch.bfloat16, torch.bfloat16]


def test_a_real_self_check_failure_is_not_retried(monkeypatch):
    rf._fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    _Kernel(monkeypatch)
    d = _Deferred(monkeypatch)
    tq._make_quant_config(tq.TQ_FP8)
    monkeypatch.setattr(
        tg, "check_pending", lambda index, dt: False, raising = False
    )  # a mismatch verdict, cached
    _call()
    _call()
    assert len(d.bf16_calls) == 2  # the one retry learned the verdict, then nothing more


# ---------------------------------------------------------------- accuracy gate vs the tensorwise path (CPU)


def _accuracy_env(
    monkeypatch,
    *,
    shapes = ((48, 3072, 64), (48, 9216, 64)),
    fp32_out = True,
):
    import contextlib

    if not tg.available():
        pytest.skip("no triton")
    monkeypatch.setitem(tg._STATE, "accuracy", {})
    monkeypatch.setattr(tg, "verified", lambda index, dt: True)
    monkeypatch.setattr(tg, "_on_card", lambda index: contextlib.nullcontext())
    monkeypatch.setattr(tg, "_device", lambda index: "cpu")
    monkeypatch.setattr(tg, "_ACC_SHAPES", shapes)
    us._FP32_OUT[0] = fp32_out
    seen = []

    def run(kernel, cublas):
        """kernel(a, b) -> fp32 [M, N] unscaled accumulator; cublas(a, b, fast) likewise."""

        def scaled_mm(a, b, scale_a, scale_b, out_dtype, use_fast_accum):
            seen.append(use_fast_accum)
            return cublas(a, b, use_fast_accum).to(out_dtype)

        def launch(kern, a, sa, b, sb, bias, dt):
            out = kernel(a, b) * sa * sb
            if bias is not None:
                out = out + bias.float()
            return out.to(dt)

        monkeypatch.setattr(torch, "_scaled_mm", scaled_mm)
        monkeypatch.setattr(tg, "_launch", launch)
        tg._STATE["accuracy"].clear()
        tg.verify_accuracy(0)
        return tg._STATE["accuracy"][0]

    return run, seen


def _exact(a, b):
    return (a.double() @ b.double()).float()


def _scaled(err):
    return lambda a, b, *fast: (a.double() @ b.double() * (1 + err)).float()


def test_accuracy_shapes_are_production_gemms():
    assert {(m, k) for m, k, n in tg._ACC_SHAPES} == {
        (512, 3072),
        (512, 9216),
        (4096, 3072),
        (4096, 9216),
    }
    assert {n for _, _, n in tg._ACC_SHAPES} == {3072}


@pytest.mark.parametrize(
    "kernel_err,fast_err,precise_err,want",
    [
        # 2.9x cuBLAS precise (the Spark residue ratio) but far under every floor: all kinds pass.
        (2.9e-7, 1e-7, 1e-7, dict(bf16f = True, bf16p = True, fp16f = True, fp16p = True)),
        # Over the fp16 floor and 1.05x cuBLAS precise, under cuBLAS fast and the bf16 floor.
        (2e-5, 1e-4, 1e-6, dict(bf16f = True, bf16p = True, fp16f = True, fp16p = False)),
        # Above both floors: the 1.05x rule decides, per accumulate mode.
        (1.04e-4, 1e-3, 1e-4, dict(bf16f = True, bf16p = True, fp16f = True, fp16p = True)),
        (1.2e-4, 1e-3, 1e-4, dict(bf16f = True, bf16p = False, fp16f = True, fp16p = False)),
        # A kernel off by a quarter of a bf16 ulp: nothing ships.
        (1e-3, 1e-4, 1e-6, dict(bf16f = False, bf16p = False, fp16f = False, fp16p = False)),
    ],
)
def test_accuracy_gate_compares_with_the_existing_path_per_kind(
    monkeypatch, kernel_err, fast_err, precise_err, want
):
    run, seen = _accuracy_env(monkeypatch)
    info = run(
        _scaled(kernel_err),
        lambda a, b, fast: _scaled(fast_err if fast else precise_err)(a, b),
    )
    got = {
        f"{short}{'f' if fast else 'p'}": tg.accurate(0, dt, fast)
        for dt, short in ((torch.bfloat16, "bf16"), (torch.float16, "fp16"))
        for fast in (True, False)
    }
    assert got == want, info
    assert seen == [True, False, True, False]  # both accumulate modes measured, per shape
    row = info["shapes"][0]
    # the metric reads the injected relative error (fp32 rounding adds ~1e-8)
    assert (
        abs(row["kernel_fp32"] / kernel_err - 1) < 0.3
        and abs(row["precise_fp32"] / precise_err - 1) < 0.3
    )
    assert info["existing_acc"] == "float32" and "accuracy" in tg.describe()


def test_accuracy_gate_passes_fp32_accumulation_and_fails_a_14_bit_accumulator(monkeypatch):
    # Emulated on CPU at production K: a tensor-core style blocked fp32 accumulator passes every kind against cuBLAS
    # fp32; a reduced-precision (14-bit, Hopper-style fast accumulate) accumulator fails every kind.
    run, _ = _accuracy_env(monkeypatch)

    def blocked(bits):
        def kernel(a, b):
            A, B = a.float(), b.float()
            acc = torch.zeros(A.shape[0], B.shape[1])
            for k0 in range(0, A.shape[1], 128):
                acc = acc + A[:, k0 : k0 + 128] @ B[k0 : k0 + 128]
                if bits:
                    m, e = torch.frexp(acc)
                    acc = torch.ldexp(torch.round(m * 2**bits) / 2**bits, e)
            return acc

        return kernel

    cublas = lambda a, b, fast: a.float() @ b.float()  # noqa: E731
    run(blocked(0), cublas)
    assert all(tg._STATE["accuracy"][0]["ok"].values()), tg._STATE["accuracy"][0]
    run(blocked(14), cublas)
    assert not any(tg._STATE["accuracy"][0]["ok"].values()), tg._STATE["accuracy"][0]


def test_accuracy_gate_compares_with_a_bf16_accumulator_where_fp32_out_is_missing(monkeypatch):
    run, _ = _accuracy_env(monkeypatch, fp32_out = False)
    info = run(_exact, lambda a, b, fast: _exact(a, b))
    assert info["existing_acc"] == "bfloat16" and all(info["ok"].values())


def test_accuracy_check_oom_is_no_verdict(monkeypatch):
    run, _ = _accuracy_env(monkeypatch)

    def oom(*a, **k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(tg, "_accuracy_operands", oom)
    assert tg.verify_accuracy(0) is False
    assert 0 not in tg._STATE["accuracy"] and tg.accuracy_pending(0)
    assert not tg.accurate(0, torch.bfloat16, True)


def test_accuracy_check_error_fails_closed(monkeypatch):
    run, _ = _accuracy_env(monkeypatch)
    monkeypatch.setattr(tg, "_accuracy_operands", lambda *a: 1 / 0)
    assert tg.verify_accuracy(0) is False
    assert "ZeroDivisionError" in tg._STATE["accuracy"][0]["error"]
    assert not tg.accuracy_pending(0)


# ---------------------------------------------------------------- verdicts persisted across processes


@pytest.fixture
def verdict_cache(monkeypatch, tmp_path):
    """A real ``diffusion_probe_cache.Namespace`` on a tmp file, a fake card fingerprint the test can change, and
    ``new_process()`` dropping every in-process verdict and slot."""
    from core.inference import diffusion_probe_cache as pc

    fp = {
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "capability": "12.0",
        "driver": "nvrm580.95",
        "torch": "2.9.0",
    }
    monkeypatch.delenv("UNSLOTH_DIFFUSION_PROBE_CACHE", raising = False)
    monkeypatch.setattr(pc, "_cache_file", lambda: tmp_path / "diffusion_quant_probe.json")
    monkeypatch.setattr(pc, "fingerprint", lambda card: dict(fp, card = card))
    monkeypatch.setattr(tg, "_persisted", _REAL_PERSISTED)
    monkeypatch.setattr(tg, "_persist", _REAL_PERSIST)

    def new_process():
        monkeypatch.setattr(tg, "_PERSIST", [])
        tg._STATE["accuracy"].clear()
        tg._STATE["persisted"].clear()

    new_process()
    return types.SimpleNamespace(
        fp = fp, new_process = new_process, path = tmp_path / "diffusion_fp8_triton_gemm.json"
    )


def _measure_fails(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("measured on a cache hit")

    monkeypatch.setattr(tg, "_accuracy_operands", boom)


def test_accuracy_verdict_cache_hit_skips_the_measurement(monkeypatch, verdict_cache):
    run, seen = _accuracy_env(monkeypatch)
    info = run(_scaled(1.2e-4), lambda a, b, fast: _scaled(1e-3 if fast else 1e-4)(a, b))
    assert verdict_cache.path.is_file() and seen  # miss: measured and stored
    want = dict(info["ok"])
    assert (
        want["bfloat16:fast"] and not want["bfloat16:precise"]
    )  # a mixed verdict, persisted as measured
    verdict_cache.new_process()
    seen.clear()
    _measure_fails(monkeypatch)
    assert tg.verify_accuracy(0) is True
    assert tg._STATE["accuracy"][0]["ok"] == want and tg._STATE["accuracy"][0]["persisted"] is True
    assert not seen and (0, "accuracy") in tg._STATE["persisted"]
    assert tg.accurate(0, torch.bfloat16, True) and not tg.accurate(0, torch.bfloat16, False)
    assert "0:accuracy" in tg.describe()["persisted"]


@pytest.mark.parametrize("change", ["driver", "gpu", "torch", "shapes", "source", "existing_acc"])
def test_accuracy_verdict_cache_key_change_is_a_miss(monkeypatch, verdict_cache, change):
    run, seen = _accuracy_env(monkeypatch)
    run(_exact, lambda a, b, fast: _exact(a, b))
    verdict_cache.new_process()
    seen.clear()
    if change == "driver":
        verdict_cache.fp["driver"] = "nvrm590.10"
    elif change == "gpu":
        verdict_cache.fp.update(gpu_name = "NVIDIA GeForce RTX 4090", capability = "8.9")
    elif change == "torch":
        verdict_cache.fp["torch"] = "2.10.0"
    elif change == "shapes":
        monkeypatch.setattr(tg, "_ACC_SHAPES", ((48, 3072, 64),))
    elif change == "source":
        from core.inference import diffusion_probe_cache as pc
        monkeypatch.setattr(pc.Namespace, "_source_digest", lambda self: "edited")
    else:
        us._FP32_OUT[0] = False  # the tensorwise path it was measured against changed
    assert tg.verify_accuracy(0) is True
    assert seen and "persisted" not in tg._STATE["accuracy"][0]  # measured again


def test_accuracy_verdict_cache_never_stores_an_error_or_an_oom(monkeypatch, verdict_cache):
    run, seen = _accuracy_env(monkeypatch)
    monkeypatch.setattr(tg, "_accuracy_operands", lambda *a: 1 / 0)
    assert tg.verify_accuracy(0) is False
    tg._STATE["accuracy"].clear()

    def oom(*a, **k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(tg, "_accuracy_operands", oom)
    assert tg.verify_accuracy(0) is False
    assert not verdict_cache.path.exists()
    assert tg._persisted(0, "accuracy") is None


def test_accuracy_verdict_cache_ignores_a_corrupt_file(monkeypatch, verdict_cache):
    run, seen = _accuracy_env(monkeypatch)
    verdict_cache.path.write_text("{not json", encoding = "utf-8")
    info = run(_exact, lambda a, b, fast: _exact(a, b))
    assert seen and all(info["ok"].values())
    verdict_cache.new_process()
    seen.clear()
    _measure_fails(monkeypatch)
    assert tg.verify_accuracy(0) is True and not seen  # rewritten whole by the store


def test_accuracy_verdict_cache_rejects_a_malformed_entry(monkeypatch, verdict_cache):
    run, seen = _accuracy_env(monkeypatch)
    for bad in ({"ok": {"bfloat16:fast": True}}, {"ok": "yes"}, ["x"], {"error": "x", "ok": {}}):
        tg._persist(0, "accuracy", bad)
        verdict_cache.new_process()
        seen.clear()
        run(_exact, lambda a, b, fast: _exact(a, b))
        assert seen and "persisted" not in tg._STATE["accuracy"][0], bad


def _self_check_env(monkeypatch):
    import contextlib

    if not tg.available():
        pytest.skip("no triton")
    monkeypatch.setitem(tg._STATE, "verified", {})
    monkeypatch.setitem(tg._STATE, "failed", {})
    monkeypatch.setitem(tg._STATE, "check", {})
    monkeypatch.setattr(tg, "_on_card", lambda index: contextlib.nullcontext())
    monkeypatch.setattr(tg, "_device", lambda index: "cpu")
    calls = []

    def operands(device):
        calls.append(device)
        raise RuntimeError("compile failed")

    monkeypatch.setattr(tg, "_check_operands", operands)
    return calls


def test_self_check_failure_is_never_persisted_and_a_pass_is(monkeypatch, verdict_cache):
    calls = _self_check_env(monkeypatch)
    assert tg.verify(0, torch.bfloat16) is False and calls
    assert tg._persisted(0, "verify|bfloat16") is None
    # a pass stored by an earlier process: adopted without a launch, per dtype
    tg._persist(0, "verify|bfloat16", True)
    verdict_cache.new_process()
    tg._STATE["verified"].clear()
    calls.clear()
    assert tg.verify(0, torch.bfloat16) is True and not calls
    assert (0, "verify|bfloat16") in tg._STATE["persisted"]
    assert tg.verify(0, torch.float16) is False and calls  # no stored fp16 pass: measured
    # a launch failure in this process still turns the card off
    tg._STATE["failed"][0] = "boom"
    assert not tg.verified(0, torch.bfloat16)


def test_self_check_pass_is_stored_and_adopted_by_the_next_process(monkeypatch, verdict_cache):
    _self_check_env(monkeypatch)
    launches = []

    def launch(kern, a, sa, b, sb, bias, dt):  # a bit-exact kernel
        launches.append(dt)
        out = (a.float() * sa) @ (b.float() * sb)
        return (out + bias.float() if bias is not None else out).to(dt)

    monkeypatch.setattr(tg, "_check_operands", lambda device: _CHECK_OPERANDS("cpu"))
    monkeypatch.setattr(tg, "_launch", launch)
    assert tg.verify(0, torch.bfloat16) is True and launches
    assert tg._persisted(0, "verify|bfloat16") is True
    verdict_cache.new_process()
    tg._STATE["verified"].clear()
    launches.clear()
    assert tg.verify(0, torch.bfloat16) is True and not launches


# ---------------------------------------------------------------- self-check depth, autotune key, retries, floor


def test_self_check_runs_the_pipelined_k_loop_in_steady_state():
    if not tg.available():
        pytest.skip("no triton")
    a, sa, b, sb, bias = tg._check_operands("cpu")
    k = a.shape[1]
    assert k == tg._CHECK_K and k % 16 == 0 and k % 64 != 0
    assert 36 * k < 2**24  # |integer sum| stays exact in fp32
    for cfg in tg._fp8_rowwise_gemm.configs:
        iters = -(-k // cfg.kwargs["BLOCK_K"])
        assert iters >= cfg.num_stages + 1, cfg


def test_autotune_key_names_the_device_arch(monkeypatch):
    if not tg.available():
        pytest.skip("no triton")
    assert "ARCH" in tg._fp8_rowwise_gemm.keys
    assert "ARCH" in tg._fp8_rowwise_gemm_kernel.do_not_specialize or any(
        getattr(p, "name", None) == "ARCH" and getattr(p, "do_not_specialize", False)
        for p in getattr(tg._fp8_rowwise_gemm_kernel, "params", ())
    )
    monkeypatch.setattr(tg, "_arch", lambda device: 121)
    seen = {}

    class _Rec:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                seen.update(zip(tg._fp8_rowwise_gemm_kernel.arg_names, args))

            return run

    a, sa, b, sb, _ = _meta_operands(512, 3072, 3072)
    tg._launch(_Rec(), a, sa, b, sb, None, torch.bfloat16)
    assert seen["ARCH"] == 121 and seen["M_BUCKET"] == 512


def test_traced_symbolic_rows_get_one_fixed_bucket():
    class _Sym:  # a SymInt stand-in: not an int
        pass

    assert tg._bucket(_Sym()) == tg._TRACED_BUCKET


def test_eager_autotune_oom_retries_the_shape_with_backoff(monkeypatch):
    if not tg.available():
        pytest.skip("no triton")
    now = [1000.0]
    monkeypatch.setattr(tg.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(tg, "_capturing", lambda device: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setitem(tg._STATE, "failed", {})
    monkeypatch.setitem(tg._STATE, "verified", {(0, torch.bfloat16): True})
    monkeypatch.setitem(tg._STATE, "tuned", set())
    monkeypatch.setitem(tg._STATE, "skip", {})
    launches = []
    oom = [True]

    def launch(*a):
        launches.append(1)
        if oom[0]:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return "out"

    monkeypatch.setattr(tg, "_launch", launch)
    a, sa, b, sb, _ = _meta_operands(512, 3072, 3072)

    class _A:
        device = type("D", (), {"index": 0, "type": "cuda"})()
        shape = a.shape
        dtype = a.dtype

        @staticmethod
        def stride(i):
            return a.stride(i)

    call = lambda: tg.eager_gemm(_A(), sa, b, sb, None, torch.bfloat16)  # noqa: E731
    assert call() is None and len(launches) == 1
    assert call() is None and len(launches) == 1  # backing off
    now[0] += tg._SKIP_RETRY_S + 1
    assert call() is None and len(launches) == 2  # retried, OOM again: the wait doubles
    now[0] += tg._SKIP_RETRY_S + 1
    assert call() is None and len(launches) == 2
    now[0] += tg._SKIP_RETRY_S
    oom[0] = False
    assert call() == "out" and len(launches) == 3
    assert not tg._STATE["skip"] and not tg._STATE["failed"]


def test_version_floor_parses_triton_versions():
    v = tg._version_tuple
    assert v("3.2.0") == (3, 2) and v("3.2.0.post12") == (3, 2) and v("3.10.1") == (3, 10)
    assert v("3.1.0") < tg._MIN_TRITON <= v("3.2.0") and v("") == () and v("dev") == ()


_OLD_TRITON = textwrap.dedent(
    """
    import sys, types
    sys.path.insert(0, sys.argv[1])
    triton = types.ModuleType("triton")
    triton.__version__ = "3.1.0"
    language = types.ModuleType("triton.language")
    language.inline_asm_elementwise = object()
    triton.language = language
    sys.modules["triton"], sys.modules["triton.language"] = triton, language
    from core.inference import diffusion_fp8_triton_gemm as tg
    assert not tg.available(), tg.describe()
    assert "3.1.0 < 3.2" in tg.describe()["error"], tg.describe()
    print("OK")
    """
)


def test_triton_below_the_floor_keeps_the_fallback(tmp_path):
    script = tmp_path / "old_triton.py"
    script.write_text(_OLD_TRITON)
    p = subprocess.run(
        [sys.executable, "-I", str(script), str(BACKEND)],
        capture_output = True,
        text = True,
        timeout = 300,
    )
    assert p.returncode == 0 and "OK" in p.stdout, (p.stdout + p.stderr)[-4000:]


_PTX = textwrap.dedent(
    """
    import re, sys
    sys.path.insert(0, sys.argv[1])
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from core.inference import diffusion_fp8_triton_gemm as tg
    k = tg._fp8_rowwise_gemm_kernel
    const = ("HAS_BIAS", "PIN_RN", "BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M")
    ptrs = {"a_ptr": "*fp8e4nv", "b_ptr": "*fp8e4nv", "sa_ptr": "*fp32", "sb_ptr": "*fp32", "bias_ptr": "*bf16",
            "c_ptr": "*bf16"}
    cfg = tg._fp8_rowwise_gemm.configs[-1]
    bad = []
    for cap in (89, 120):
        sig = {n: ptrs.get(n, "constexpr" if n in const else "i32") for n in k.arg_names}
        consts = {"HAS_BIAS": True, "PIN_RN": True, **cfg.kwargs}
        try:
            src = ASTSource(fn=k, signature=sig, constexprs=consts)
        except TypeError:
            src = ASTSource(fn=k, signature=sig, constants=consts)
        ptx = triton.compile(src, target=GPUTarget("cuda", cap, 32),
                             options={"num_warps": cfg.num_warps, "num_stages": cfg.num_stages}).asm["ptx"]
        ops = set(re.findall(r"\\b(?:mul|add|sub|fma)(?:\\.[a-z]+)*\\.f32\\b", ptx))
        if ops != {"mul.rn.f32", "add.rn.f32"} or "mma.sync" not in ptx:
            bad.append((cap, sorted(ops)))
    print("BAD", bad)
    sys.exit(1 if bad else 0)
    """
)


def test_epilogue_ptx_rounds_to_nearest_without_ftz_or_fma(tmp_path):
    # libdevice mul_rn / add_rn lowered to mul.rn.ftz.f32 / add.rn.ftz.f32 (denormals flushed). Offline compile, no GPU.
    pytest.importorskip("triton")
    pytest.importorskip("triton.backends.compiler")
    if not tg.available():
        pytest.skip("no triton")
    script = tmp_path / "ptx.py"
    script.write_text(_PTX)
    env = {**os.environ, "TRITON_CACHE_DIR": str(tmp_path / "cache"), "CUDA_VISIBLE_DEVICES": ""}
    env.pop("TRITON_INTERPRET", None)
    p = subprocess.run(
        [sys.executable, "-I", str(script), str(BACKEND)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 900,
    )
    assert p.returncode == 0, (p.stdout + p.stderr)[-4000:]
