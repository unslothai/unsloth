# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FP8 on a torch build without the row-wise _scaled_mm (Windows wheels): the load path must route every fp8 GEMM to
the tensorwise unit-scale fallback, and leave builds with the row-wise kernel exactly as they were (CPU, emulated)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torchao")
from torchao.quantization.quantize_.workflows.float8 import float8_tensor as f8t  # noqa: E402

from core.inference import diffusion_compile_config as compile_config  # noqa: E402
from core.inference import diffusion_fp8_unit_scale as us  # noqa: E402
from core.inference import diffusion_transformer_quant as tq  # noqa: E402

ROWWISE_ERROR = "Rowwise scaling is not currently supported on your device"
_REAL_ONES = torch.ones


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    us.uninstall()
    compile_config._reset_for_tests()
    monkeypatch.delenv(us.UNIT_SCALE_ENV, raising = False)
    monkeypatch.delenv(us.MIN_ROWS_ENV, raising = False)
    for k in us._COUNTS:
        us._COUNTS[k] = 0
    getattr(us, "_ROWWISE", {}).clear()
    getattr(us, "_FP32_OUT", {}).clear()
    # Never touch the real probe file from a unit test.
    for name, stub in (
        ("_persisted_rowwise", lambda index: None),
        ("_persist_rowwise", lambda index, verdict: None),
        ("_has_context", lambda index: False),
        ("_current_device", lambda: 0),
    ):
        if hasattr(us, name):
            monkeypatch.setattr(us, name, stub)
    # The fake card is a CUDA build whatever torch runs the suite: a ROCm host torch (Studio's Windows AMD install)
    # would otherwise turn the fallback off for every test. The ROCm refusal has its own tests.
    from core import _torchao_stub

    monkeypatch.setattr(_torchao_stub, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(_torchao_stub, "is_stubbed", lambda name: False)
    original = f8t.addmm_float8_unwrapped_inference
    yield
    us.uninstall()
    compile_config._reset_for_tests()
    getattr(us, "_ROWWISE", {}).clear()
    getattr(us, "_FP32_OUT", {}).clear()
    f8t.addmm_float8_unwrapped_inference = original


class _FakeDevice:
    type = "cuda"
    index = 0


class _Dev:
    """A CPU tensor that reports cuda:0, so the dispatcher's device checks see a GPU operand."""

    def __init__(self, t):
        self.t = t
        self.device = _FakeDevice()
        self.shape = t.shape

    def dim(self):
        return self.t.dim()


def _fake_scaled_mm(rowwise_ok: bool, calls: list):
    """torch._scaled_mm as a Windows build has it (``rowwise_ok=False``): tensorwise (scalar scales) runs, row-wise
    (2-D scales) raises the error the Windows wheels raise."""

    def scaled_mm(
        a,
        b,
        scale_a,
        scale_b,
        out_dtype = None,
        use_fast_accum = False,
        bias = None,
        **_,
    ):
        a = a.t if isinstance(a, _Dev) else a
        rowwise = scale_a.numel() > 1 or scale_b.numel() > 1
        calls.append(
            "rowwise" if rowwise else "tensorwise32" if out_dtype == torch.float32 else "tensorwise"
        )
        if rowwise and not rowwise_ok:
            raise RuntimeError(ROWWISE_ERROR)
        acc = (a.float() * scale_a.float()) @ (b.float() * scale_b.float())
        if bias is not None:
            acc = acc + bias.float()
        return acc.to(out_dtype)

    return scaled_mm


def _fake_card(
    monkeypatch,
    cap,
    *,
    rowwise_ok: bool,
    cards = (0,),
):
    """One or more sm_``cap`` CUDA cards on a CPU host, with a torch._scaled_mm that has (or lacks) row-wise scaling.
    The per-card probe runs for real against the stub, so the routing decision is what the load path would make."""
    calls: list = []
    monkeypatch.setattr(us, "_device_caps", lambda: {i: cap for i in cards})
    monkeypatch.setattr(torch, "_scaled_mm", _fake_scaled_mm(rowwise_ok, calls))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: cap)
    # The probe and the dispatcher allocate on "cuda:N"; keep those on the CPU.
    monkeypatch.setattr(torch, "ones", lambda *a, device = None, **k: _REAL_ONES(*a, **k))
    return calls


def _operands(
    m,
    k = 64,
    n = 48,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(m, k, generator = g) * 3).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, generator = g) * 3).to(torch.float8_e4m3fn).t()
    sa = torch.rand(m, 1, generator = g) + 0.1
    sb = torch.rand(1, n, generator = g) + 0.1
    return a, sa, b, sb


def _reference(
    a,
    sa,
    b,
    sb,
    bias = None,
):
    """Dequantise, then matmul in fp32: what the fp8 linear approximates."""
    out = (a.float() * sa.float()) @ (b.float() * sb.float())
    return out + bias.float() if bias is not None else out


@pytest.mark.parametrize("cap", [(12, 0), (12, 1), (10, 0), (8, 9), (9, 0)])
@pytest.mark.parametrize("m", [1, 32, 4096])
@pytest.mark.parametrize("bias", [False, True])
def test_fp8_load_path_runs_without_the_rowwise_kernel(monkeypatch, cap, m, bias):
    # Studio's FP8 load quantizes with _make_quant_config(TQ_FP8) at Auto (fast accumulate): that must select the
    # fallback, and every GEMM after it (the 32-row smoke probe included) must avoid the row-wise kernel.
    calls = _fake_card(monkeypatch, cap, rowwise_ok = False)
    tq._make_quant_config(tq.TQ_FP8)
    assert us.stats()["installed"] is True
    calls.clear()
    a, sa, b, sb = _operands(m)
    bv = torch.randn(48).to(torch.bfloat16) if bias else None
    got = f8t.addmm_float8_unwrapped_inference(
        _Dev(a), sa, b, sb, torch.bfloat16, bias = bv, use_fast_accum = True
    )
    # fp32 accumulator: no bf16 rounding before the scales, only the one bf16 rounding of the output.
    assert calls == ["tensorwise32"]
    want = _reference(a, sa, b, sb, bv)
    err = (got.float() - want).abs().max()
    assert err <= 2**-7 * want.abs().max()
    assert torch.isfinite(got).all()
    assert us.stats()["unit_scale"] == 1 and us.stats()["rowwise"] == 0


def test_status_names_the_fallback_path(monkeypatch):
    _fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    tq._make_quant_config(tq.TQ_FP8)
    status = us.gemm_path(0)
    assert status["path"] == "unit_scale_fallback" and status["rowwise_kernel"] is False
    assert status["accumulator"] == "fp32"

    from types import SimpleNamespace

    from core.inference.diffusion import _fp8_gemm_status

    us.uninstall()
    assert us.gemm_path(0)["path"] == "torchao_rowwise"
    _fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    tq._make_quant_config(tq.TQ_FP8)
    state = SimpleNamespace(transformer_quant = "fp8", text_encoder_quant = None, pipe = None)
    assert _fp8_gemm_status(state)["path"] == "unit_scale_fallback"
    assert (
        _fp8_gemm_status(SimpleNamespace(transformer_quant = "int8", text_encoder_quant = None)) is None
    )


def test_only_cards_in_use_are_probed_and_each_keeps_its_own_verdict(monkeypatch):
    # No card has a CUDA context: only the current device is probed, and its verdict is not copied to its siblings.
    calls = _fake_card(monkeypatch, (12, 0), rowwise_ok = False, cards = (0, 1, 2))
    stored = {}
    monkeypatch.setattr(
        us, "_persist_rowwise", lambda index, verdict: stored.__setitem__(index, verdict)
    )
    tq._make_quant_config(tq.TQ_FP8)
    assert calls == ["tensorwise", "rowwise", "tensorwise32"]
    assert us.fallback_cards() == frozenset({0})
    assert stored == {0: False} and set(us._ROWWISE) == {0}
    import torch.compiler.config as ccfg

    assert ccfg.cache_key_tag.endswith("unsloth_fp8_unit_scale=256:rowwise-missing=0:acc32=0")
    us.uninstall()
    assert "unsloth_fp8_unit_scale" not in ccfg.cache_key_tag
    # A later process reads the verdict back instead of probing.
    us._ROWWISE.clear()
    calls.clear()
    monkeypatch.setattr(us, "_persisted_rowwise", lambda index: stored.get(index))
    tq._make_quant_config(tq.TQ_FP8)
    assert "rowwise" not in calls and us.fallback_cards() == frozenset({0})


def test_cards_with_a_context_are_probed_one_by_one(monkeypatch):
    calls = _fake_card(monkeypatch, (12, 0), rowwise_ok = False, cards = (0, 1, 2))
    stored = {}
    monkeypatch.setattr(
        us, "_persist_rowwise", lambda index, verdict: stored.__setitem__(index, verdict)
    )
    monkeypatch.setattr(us, "_has_context", lambda index: index in (1, 2))
    tq._make_quant_config(tq.TQ_FP8)
    assert calls.count("rowwise") == 2
    assert stored == {1: False, 2: False} and 0 not in us._ROWWISE
    assert us.fallback_cards() == frozenset({1, 2})


def test_a_card_used_after_install_joins_the_fallback(monkeypatch):
    _fake_card(monkeypatch, (12, 0), rowwise_ok = False, cards = (0, 1))
    tq._make_quant_config(tq.TQ_FP8)
    assert us.fallback_cards() == frozenset({0})
    monkeypatch.setattr(us, "_has_context", lambda index: True)
    tq._make_quant_config(tq.TQ_FP8)
    assert us.fallback_cards() == frozenset({0, 1})


@pytest.mark.parametrize(
    "error",
    [
        "oom",
        "CUDA error: an illegal memory access was encountered",
        "CUBLAS_STATUS_EXECUTION_FAILED when calling cublasLtMatmul",
    ],
)
def test_a_transient_rowwise_failure_is_no_verdict(monkeypatch, error):
    # Only torch's "not supported" means the kernel is missing; an OOM or any other error is neither cached nor
    # persisted, and the card keeps torchao's own path until a later probe settles it.
    calls = _fake_card(monkeypatch, (12, 0), rowwise_ok = True)
    stored = {}
    monkeypatch.setattr(
        us, "_persist_rowwise", lambda index, verdict: stored.__setitem__(index, verdict)
    )
    inner = torch._scaled_mm
    exc = (
        torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB")
        if error == "oom"
        else RuntimeError(error)
    )

    def flaky(a, b, scale_a, scale_b, **kw):
        if scale_a.numel() > 1:
            raise exc
        return inner(a, b, scale_a, scale_b, **kw)

    monkeypatch.setattr(torch, "_scaled_mm", flaky)
    before = f8t.addmm_float8_unwrapped_inference
    tq._make_quant_config(tq.TQ_FP8)
    assert stored == {} and 0 not in us._ROWWISE
    assert us.stats()["installed"] is False and f8t.addmm_float8_unwrapped_inference is before
    # Memory freed: the next load probes again and finds the kernel.
    monkeypatch.setattr(torch, "_scaled_mm", inner)
    tq._make_quant_config(tq.TQ_FP8)
    assert stored == {0: True} and us._ROWWISE[0] is True


def test_legacy_tuple_returning_scaled_mm_still_detects_a_missing_rowwise_kernel(monkeypatch):
    # torch < 2.5 returns (out, amax) from _scaled_mm.
    _fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    inner = torch._scaled_mm
    monkeypatch.setattr(torch, "_scaled_mm", lambda *a, **k: (inner(*a, **k), torch.tensor(0.0)))
    assert us._probe_rowwise(0) is False


def test_traced_epilogue_does_not_specialize_on_the_row_count():
    # AOTAutograd traces torchao's subclass at dispatch level, where is_compiling() is False: the eager chunk loop
    # would bake the traced row count into the graph (a recompile per M, a ConstraintViolation under mark_dynamic).
    from torch.fx.experimental.proxy_tensor import make_fx

    def fn(acc, sa, sb):
        return us._scale_epilogue(acc, sa, sb, None, torch.bfloat16)

    gm = make_fx(fn, tracing_mode = "symbolic")(
        torch.randn(64, 48), torch.rand(64, 1), torch.rand(1, 48)
    )
    acc, sa, sb = torch.randn(130, 48), torch.rand(130, 1), torch.rand(1, 48)
    assert torch.equal(gm(acc, sa, sb), fn(acc, sa, sb))
    assert not any(
        n.target is torch.ops.aten.copy_.default for n in gm.graph.nodes if n.op == "call_function"
    )


def test_smoke_probe_verdicts_are_rekeyed_on_the_dispatcher_source():
    # A pre-fix "fp8 unusable" verdict persisted on Windows must not outlive this change.
    from core.inference import diffusion_probe_cache
    assert "diffusion_fp8_unit_scale.py" in diffusion_probe_cache._PROBE_SOURCES


@pytest.mark.parametrize("cap", [(12, 0), (10, 0), (9, 0)])
@pytest.mark.parametrize("fast_accum", [None, True, False])
def test_rowwise_builds_are_untouched(monkeypatch, cap, fast_accum):
    # Linux wheels have the row-wise kernel: nothing installs, torchao's GEMM and the compile cache key stay as-is.
    import torch.compiler.config as ccfg

    monkeypatch.setattr(ccfg, "cache_key_tag", "")
    _fake_card(monkeypatch, cap, rowwise_ok = True)
    before = f8t.addmm_float8_unwrapped_inference
    tq._make_quant_config(tq.TQ_FP8, fast_accum = fast_accum)
    assert us.stats()["installed"] is False
    assert f8t.addmm_float8_unwrapped_inference is before
    assert ccfg.cache_key_tag == ""


def test_rowwise_ada_keeps_its_precise_accumulate_install(monkeypatch):
    import torch.compiler.config as ccfg

    monkeypatch.setattr(ccfg, "cache_key_tag", "")
    calls = _fake_card(monkeypatch, (8, 9), rowwise_ok = True)
    tq._make_quant_config(tq.TQ_FP8)  # Auto = fast accumulate: stays on torchao, as before
    assert us.stats()["installed"] is False
    tq._make_quant_config(tq.TQ_FP8, fast_accum = False)
    assert us.stats()["installed"] is True
    assert ccfg.cache_key_tag == "unsloth_fp8_unit_scale=256"
    calls.clear()
    fn = f8t.addmm_float8_unwrapped_inference
    small = _operands(32)
    fn(_Dev(small[0]), small[1], small[2], small[3], torch.bfloat16)
    fn(_Dev(small[0]), small[1], small[2], small[3], torch.bfloat16, use_fast_accum = True)
    big = _operands(512)
    fn(_Dev(big[0]), big[1], big[2], big[3], torch.bfloat16, use_fast_accum = True)
    fn(_Dev(big[0]), big[1], big[2], big[3], torch.bfloat16)
    assert calls == ["rowwise", "rowwise", "rowwise", "tensorwise"]


def test_fallback_env_forces_the_windows_route_where_rowwise_works(monkeypatch):
    # The correctness A/B on Linux: UNSLOTH_DIFFUSION_FP8_UNIT_SCALE=fallback runs exactly the Windows dispatch.
    from core.inference import diffusion_compile_cache as cc

    calls = _fake_card(monkeypatch, (12, 0), rowwise_ok = True)
    monkeypatch.setenv(us.UNIT_SCALE_ENV, "fallback")
    assert cc.graph_switches().get(us.UNIT_SCALE_ENV) == "fallback"
    tq._make_quant_config(tq.TQ_FP8)
    assert calls == ["tensorwise32"]  # no row-wise probe, only the accumulator check
    assert us.fallback_cards() == frozenset({0})
    assert us.gemm_path(0)["reason"].endswith("=fallback")
    a, sa, b, sb = _operands(32)
    f8t.addmm_float8_unwrapped_inference(_Dev(a), sa, b, sb, torch.bfloat16, use_fast_accum = True)
    assert calls == ["tensorwise32", "tensorwise32"]


def test_no_fp8_gemm_at_all_does_not_install(monkeypatch):
    # Neither kernel runs (a card the probe cannot use): nothing to fall back to, torchao's error stands.
    monkeypatch.setattr(us, "_device_caps", lambda: {0: (12, 0)})
    monkeypatch.setattr(us, "_probe_rowwise", lambda index: None)
    tq._make_quant_config(tq.TQ_FP8)
    assert us.stats()["installed"] is False


def test_fallback_keys_the_compile_cache_apart_from_every_other_install(monkeypatch):
    # The Dynamo graph is the same on every route, so the tag is what keeps a compiled row-wise / bf16-accumulator
    # graph from being reused here (a G4 A/B reused one across processes before the accumulator was in the tag).
    import torch.compiler.config as ccfg

    monkeypatch.setattr(ccfg, "cache_key_tag", "")
    _fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    tq._make_quant_config(tq.TQ_FP8)
    assert ccfg.cache_key_tag == "unsloth_fp8_unit_scale=256:rowwise-missing=0:acc32=0"


def test_fallback_keeps_a_bf16_accumulator_where_fp32_output_fails(monkeypatch):
    calls = _fake_card(monkeypatch, (12, 0), rowwise_ok = False)
    inner = torch._scaled_mm

    def no_fp32(
        *args,
        out_dtype = None,
        **kw,
    ):
        if out_dtype == torch.float32:
            raise RuntimeError("fp32 output unsupported")
        return inner(*args, out_dtype = out_dtype, **kw)

    monkeypatch.setattr(torch, "_scaled_mm", no_fp32)
    tq._make_quant_config(tq.TQ_FP8)
    assert us.gemm_path(0)["accumulator"] == "bf16"
    calls.clear()
    a, sa, b, sb = _operands(64)
    got = f8t.addmm_float8_unwrapped_inference(
        _Dev(a), sa, b, sb, torch.bfloat16, use_fast_accum = True
    )
    assert calls == ["tensorwise"]
    want = _reference(a, sa, b, sb)
    assert (got.float() - want).abs().max() <= 2**-6 * want.abs().max()


@pytest.mark.parametrize("refusal", ["rocm", "stubbed_torchao"])
def test_rocm_and_stubbed_torchao_are_never_probed(monkeypatch, refusal):
    # Studio keeps ROCm on GGUF; a gfx capability (11, 5) must not reach an fp8 GEMM probe that may not fail cleanly.
    from core import _torchao_stub

    calls = _fake_card(monkeypatch, (11, 5), rowwise_ok = False)
    monkeypatch.setattr(_torchao_stub, "torch_is_rocm", lambda: refusal == "rocm")
    monkeypatch.setattr(_torchao_stub, "is_stubbed", lambda name: refusal == "stubbed_torchao")
    before = f8t.addmm_float8_unwrapped_inference
    tq._make_quant_config(tq.TQ_FP8)
    assert calls == [] and us.stats()["installed"] is False
    assert f8t.addmm_float8_unwrapped_inference is before


def test_status_reports_the_card_of_the_fp8_component(monkeypatch):
    # FP8 text encoder on fallback card 1, INT8 denoiser on card 0: the status is the encoder's card, not the
    # denoiser's, since the dispatcher routes by operand device.
    from types import SimpleNamespace

    from core.inference.diffusion import _fp8_gemm_status

    _fake_card(monkeypatch, (12, 0), rowwise_ok = False, cards = (0, 1))
    monkeypatch.setattr(us, "_has_context", lambda index: index == 1)
    tq._make_quant_config(tq.TQ_FP8)
    assert us.fallback_cards() == frozenset({1})

    class _Mod:
        def __init__(self, index):
            self.p = SimpleNamespace(device = SimpleNamespace(type = "cuda", index = index))

        def parameters(self):
            return iter([self.p])

    pipe = SimpleNamespace(transformer = _Mod(0), text_encoder = _Mod(1))
    state = SimpleNamespace(transformer_quant = "int8", text_encoder_quant = "fp8", pipe = pipe)
    status = _fp8_gemm_status(state)
    assert status["device_index"] == 1 and status["path"] == "unit_scale_fallback"
    assert status["component"] == "text_encoder"
    state = SimpleNamespace(transformer_quant = "fp8", text_encoder_quant = "fp8", pipe = pipe)
    assert _fp8_gemm_status(state)["device_index"] == 0


def test_accumulator_check_waits_for_an_idle_fallback_card_to_be_used(monkeypatch):
    # cuda:1 (another arch) is a fallback card only through a persisted verdict and has no CUDA context: no fp32
    # GEMM may run there (it would open a context on a card the load never uses) until it is used.
    _fake_card(monkeypatch, (12, 0), rowwise_ok = False, cards = (0, 1))
    monkeypatch.setattr(us, "_device_caps", lambda: {0: (12, 0), 1: (10, 0)})
    monkeypatch.setattr(us, "_persisted_rowwise", lambda index: False)
    monkeypatch.setattr(us, "_has_context", lambda index: index == 0)
    checked = []
    real = us._fp32_out_runs

    def recording(index):
        if index not in us._FP32_OUT:  # a GEMM actually runs, not a cached answer
            checked.append(index)
        return real(index)

    monkeypatch.setattr(us, "_fp32_out_runs", recording)
    tq._make_quant_config(tq.TQ_FP8)
    assert us.fallback_cards() == frozenset({0, 1})
    assert checked == [0]
    assert us.gemm_path(0)["accumulator"] == "fp32" and us.gemm_path(1)["accumulator"] == "bf16"
    # cuda:1 is used now: the next install rebuilds the dispatcher with its own accumulator check.
    monkeypatch.setattr(us, "_has_context", lambda index: True)
    tq._make_quant_config(tq.TQ_FP8)
    assert checked == [0, 1]
    assert us.gemm_path(1)["accumulator"] == "fp32"


def test_a_rocm_torch_keeps_the_fallback_off(monkeypatch):
    # The real ROCm check, not a stub: torch.version.hip set, as on Studio's Windows AMD torch (2.11.0+rocm7.14.0).
    from core import _torchao_stub

    monkeypatch.undo()
    us._ROWWISE.clear()
    monkeypatch.setattr(us, "_persisted_rowwise", lambda index: None)
    monkeypatch.setattr(us, "_persist_rowwise", lambda index, verdict: None)
    monkeypatch.setattr(us, "_has_context", lambda index: False)
    monkeypatch.setattr(us, "_current_device", lambda: 0)
    calls = _fake_card(monkeypatch, (11, 5), rowwise_ok = False)
    monkeypatch.setattr(torch.version, "hip", "7.14.0", raising = False)
    assert _torchao_stub.torch_is_rocm() is True
    before = f8t.addmm_float8_unwrapped_inference
    tq._make_quant_config(tq.TQ_FP8)
    assert calls == [] and us.stats()["installed"] is False
    assert f8t.addmm_float8_unwrapped_inference is before
