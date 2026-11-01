# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""SDPA must stay off cuDNN for exactly the calls torch later refused itself:
- pytorch#193893 (guard pytorch#194963, in torch 2.14): one-query-row decode on SM 10.x / 11.x with
  cuDNN 9.19 - 9.25.0 except 9.24.1 gives wrong attention.
- pytorch#196678 (guard pytorch#196861, in torch 2.15): non-causal SDPA backward on SM90 with
  cuDNN < 9.26 and a key length of 64 mod 128 gives NaN dQ.
CPU only: capability, cuDNN and torch versions are mocked; the wrapper is checked by recording the
SDPA backend flags its inner call runs under."""

import types

import pytest

torch = pytest.importorskip("torch")
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

import unsloth.import_fixes as fixes  # noqa: E402


@pytest.mark.parametrize(
    "torch_version, cudnn, cap, expected",
    [
        ("2.13.0+cu130", 92000, (10, 0), True),  # the issue's B200 setup on 2.13
        ("2.11.0+cu128", 91900, (10, 0), True),  # the original report
        ("2.12.1+cu130", 92000, (10, 3), True),  # GB300 / B300
        ("2.13.0+cu130", 92000, (11, 0), True),  # Thor
        ("2.13.0+cu130", 92500, (10, 0), True),  # 9.25.0 still broken
        ("2.13.0+cu130", 92400, (10, 0), True),
        ("2.13.0+cu130", 92401, (10, 0), False),  # 9.24.1 fixed
        ("2.13.0+cu130", 92501, (10, 0), False),  # 9.25.1 fixed
        ("2.13.0+cu130", 92600, (10, 0), False),
        ("2.13.0+cu126", 91002, (10, 0), False),  # cu126 wheels ship cuDNN 9.10.2
        ("2.13.0+cu130", 91800, (10, 0), False),
        ("2.13.0+cu130", 92000, (12, 0), False),  # sm_120 is clean
        ("2.13.0+cu130", 92000, (9, 0), False),
        ("2.14.0+cu130", 92400, (10, 0), False),  # torch 2.14 refuses it itself
        ("2.14.1+cu130", 92400, (10, 0), False),
        ("2.15.0+cu132", 92600, (10, 0), False),
        ("2.13.0+cu130", None, (10, 0), False),
        ("not a version", 92000, (10, 0), False),
    ],
)
def test_decode_gate(torch_version, cudnn, cap, expected):
    assert fixes._cudnn_sdpa_decode_corrupts(torch_version, cudnn, cap) is expected


@pytest.mark.parametrize(
    "torch_version, cudnn, cap, expected",
    [
        ("2.14.1+cu130", 92400, (9, 0), True),  # what pip install unsloth gets on H100
        ("2.14.0+cu130", 92400, (9, 0), True),
        ("2.13.0+cu130", 92000, (9, 0), True),  # the issue's version
        ("2.11.0+cu128", 91900, (9, 0), True),
        ("2.14.0+cu130", 92501, (9, 0), True),  # 9.25.1 still broken (issue comment)
        ("2.14.0+cu126", 91002, (9, 0), True),  # torch's own guard is any cuDNN < 9.26
        ("2.14.0+cu130", 92600, (9, 0), False),  # cuDNN 9.26 fixed
        ("2.15.0+cu132", 92600, (9, 0), False),
        ("2.15.0+cu132", 92400, (9, 0), False),  # torch 2.15 refuses it itself
        ("2.14.0+cu130", 92400, (10, 0), False),
        ("2.14.0+cu130", 92400, (8, 0), False),
        ("2.14.0+cu130", 92400, (8, 9), False),
        ("2.14.0+cu130", 92400, (12, 0), False),
        ("2.14.0+cu130", None, (9, 0), False),
    ],
)
def test_dq_nan_gate(torch_version, cudnn, cap, expected):
    assert fixes._cudnn_sdpa_dq_nan(torch_version, cudnn, cap) is expected


class _FakeCuda:
    """Enough of a CUDA tensor for the per-call decision (no GPU here)."""

    def __init__(
        self,
        *shape,
        dtype = None,
        requires_grad = False,
        index = 0,
    ):
        self.shape = torch.Size(shape)
        self.dtype = dtype or torch.bfloat16
        self.requires_grad = requires_grad
        self.is_cuda = True
        self.is_nested = False
        self.device = types.SimpleNamespace(type = "cuda", index = index)

    def dim(self):
        return len(self.shape)


D0 = frozenset({0})
ALL_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
    SDPBackend.CUDNN_ATTENTION,
]
NONE = frozenset()


def _bad(
    decode,
    dq,
    q,
    k,
    is_causal = False,
):
    return fixes._sdpa_cudnn_known_bad(decode, dq, q, k, is_causal)


class TestPerCallDecision:
    def test_decode_one_query_row(self):
        q, k = _FakeCuda(2, 28, 1, 128), _FakeCuda(2, 4, 257, 128)
        assert _bad(D0, NONE, q, k) == "decode"
        assert _bad(D0, NONE, q, k, is_causal = True) == "decode"

    def test_decode_only_one_row(self):
        assert _bad(D0, NONE, _FakeCuda(2, 28, 2, 128), _FakeCuda(2, 4, 257, 128)) is None
        assert _bad(D0, NONE, _FakeCuda(1, 28, 512, 128), _FakeCuda(1, 4, 512, 128)) is None

    def test_decode_other_device_or_cpu(self):
        q, k = _FakeCuda(1, 8, 1, 64, index = 1), _FakeCuda(1, 8, 9, 64, index = 1)
        assert _bad(D0, NONE, q, k) is None
        cpu = torch.zeros(1, 8, 1, 64, dtype = torch.bfloat16)
        assert _bad(frozenset({0, None}), frozenset({0, None}), cpu, cpu) is None

    def test_decode_fp32_only_under_autocast(self):
        q, k = _FakeCuda(1, 8, 1, 64, dtype = torch.float32), _FakeCuda(1, 8, 9, 64)
        assert _bad(D0, NONE, q, k) is None

    @pytest.mark.parametrize("s_kv", [64, 192, 704, 2112])
    def test_dq_nan_lengths(self, s_kv):
        q = _FakeCuda(1, 8, s_kv, 64, requires_grad = True)
        k = _FakeCuda(1, 8, s_kv, 64)
        assert _bad(NONE, D0, q, k) == "dq_nan"

    @pytest.mark.parametrize("s_kv", [63, 65, 128, 640, 768, 1024])
    def test_dq_nan_other_lengths_keep_cudnn(self, s_kv):
        q = _FakeCuda(1, 8, s_kv, 64, requires_grad = True)
        assert _bad(NONE, D0, q, _FakeCuda(1, 8, s_kv, 64)) is None

    def test_dq_nan_needs_noncausal_grad(self):
        k = _FakeCuda(1, 8, 704, 64)
        assert (
            _bad(NONE, D0, _FakeCuda(1, 8, 704, 64, requires_grad = True), k, is_causal = True) is None
        )
        assert _bad(NONE, D0, _FakeCuda(1, 8, 704, 64), k) is None
        with torch.no_grad():
            assert _bad(NONE, D0, _FakeCuda(1, 8, 704, 64, requires_grad = True), k) is None

    def test_dq_nan_reads_key_length(self):
        # Upstream gates on s_k: a 1-row training query against 704 keys is caught, 640 is not.
        q = _FakeCuda(1, 8, 100, 64, requires_grad = True)
        assert _bad(NONE, D0, q, _FakeCuda(1, 8, 704, 64)) == "dq_nan"
        assert _bad(NONE, D0, q, _FakeCuda(1, 8, 640, 64)) is None

    def test_dq_nan_dtype(self):
        k = _FakeCuda(1, 8, 704, 64)
        for dtype in (torch.float16, torch.bfloat16):
            assert _bad(NONE, D0, _FakeCuda(1, 8, 704, 64, dtype = dtype, requires_grad = True), k)
        q32 = _FakeCuda(1, 8, 704, 64, dtype = torch.float32, requires_grad = True)
        assert _bad(NONE, D0, q32, k) is None


class TestDevices:
    def _mock(
        self,
        monkeypatch,
        version,
        cudnn,
        caps,
        hip = None,
    ):
        monkeypatch.setattr(torch, "__version__", version)
        monkeypatch.setattr(torch.version, "hip", hip, raising = False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: len(caps))
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i = 0: caps[i])
        monkeypatch.setattr(torch.backends.cudnn, "version", lambda: cudnn)

    def test_mixed_host(self, monkeypatch):
        self._mock(monkeypatch, "2.13.0+cu130", 92000, [(9, 0), (10, 0), (12, 0), (11, 0)])
        assert fixes._sdpa_cudnn_known_bad_devices() == (frozenset({1, 3}), frozenset({0}))

    def test_torch_214_h100(self, monkeypatch):
        self._mock(monkeypatch, "2.14.0+cu130", 92400, [(9, 0), (10, 0)])
        assert fixes._sdpa_cudnn_known_bad_devices() == (frozenset(), frozenset({0}))

    def test_torch_215_is_clean(self, monkeypatch):
        self._mock(monkeypatch, "2.15.0+cu132", 92600, [(9, 0), (10, 0)])
        assert fixes._sdpa_cudnn_known_bad_devices() == (frozenset(), frozenset())

    def test_rocm_is_clean(self, monkeypatch):
        self._mock(monkeypatch, "2.13.0+rocm7.2", 92000, [(9, 0)], hip = "7.2")
        assert fixes._sdpa_cudnn_known_bad_devices() == (frozenset(), frozenset())


@pytest.fixture
def recorder(monkeypatch):
    """Installs the fix over a recording SDPA (devices mocked), restoring F afterwards."""
    import torch.nn.functional as F

    calls = []

    def fake_sdpa(
        query,
        key,
        value,
        attn_mask = None,
        dropout_p = 0.0,
        is_causal = False,
        **kw,
    ):
        cuda = torch.backends.cuda
        calls.append(
            dict(
                cudnn = cuda.cudnn_sdp_enabled(),
                flash = cuda.flash_sdp_enabled(),
                efficient = cuda.mem_efficient_sdp_enabled(),
                math = cuda.math_sdp_enabled(),
                mask = attn_mask,
                is_causal = is_causal,
                kw = kw,
            )
        )
        return "out"

    monkeypatch.setattr(F, "scaled_dot_product_attention", fake_sdpa)
    monkeypatch.setattr(fixes, "_sdpa_cudnn_known_bad_devices", lambda: (D0, D0))
    monkeypatch.delenv("UNSLOTH_ALLOW_BUGGY_CUDNN_SDPA", raising = False)
    fixes.fix_cudnn_sdpa_known_bad_shapes()
    assert F.scaled_dot_product_attention is not fake_sdpa
    # The CPU test session may start with some backends off; every test starts from all on.
    with sdpa_kernel(ALL_BACKENDS):
        yield F, calls


ALL_ON = dict(cudnn = True, flash = True, efficient = True, math = True)
NO_CUDNN = dict(cudnn = False, flash = True, efficient = True, math = True)


def _flags(call):
    return {k: call[k] for k in ("cudnn", "flash", "efficient", "math")}


class TestWrapper:
    def test_trigger_calls_run_without_cudnn(self, recorder):
        F, calls = recorder
        mask = object()
        assert F.scaled_dot_product_attention(
            _FakeCuda(1, 28, 1, 128), _FakeCuda(1, 4, 300, 128), _FakeCuda(1, 4, 300, 128),
            attn_mask = mask, enable_gqa = True,
        ) == "out"  # fmt: skip
        q = _FakeCuda(2, 8, 704, 64, requires_grad = True)
        F.scaled_dot_product_attention(q, _FakeCuda(2, 8, 704, 64), _FakeCuda(2, 8, 704, 64), mask)
        assert [_flags(c) for c in calls] == [NO_CUDNN, NO_CUDNN]
        # Arguments pass through untouched.
        assert calls[0]["mask"] is mask and calls[0]["kw"] == {"enable_gqa": True}
        # The policy is restored after the call.
        assert torch.backends.cuda.cudnn_sdp_enabled()

    def test_other_calls_keep_cudnn(self, recorder):
        F, calls = recorder
        q = _FakeCuda(2, 8, 768, 64, requires_grad = True)
        F.scaled_dot_product_attention(q, _FakeCuda(2, 8, 768, 64), _FakeCuda(2, 8, 768, 64))
        q = _FakeCuda(2, 8, 704, 64, requires_grad = True)
        F.scaled_dot_product_attention(q, q, q, None, 0.0, True)  # causal
        F.scaled_dot_product_attention(_FakeCuda(1, 8, 4, 64), _FakeCuda(1, 8, 9, 64), q)  # prefill
        assert [_flags(c) for c in calls] == [ALL_ON] * 3
        assert calls[1]["is_causal"] is True

    def test_user_policy_is_honoured(self, recorder):
        F, calls = recorder
        q = _FakeCuda(1, 8, 1, 64)
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION]):
            F.scaled_dot_product_attention(q, q, q)
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):  # cuDNN-only: left alone
            F.scaled_dot_product_attention(q, q, q)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):  # cuDNN already off
            F.scaled_dot_product_attention(q, q, q)
        assert [_flags(c) for c in calls] == [
            dict(cudnn = False, flash = False, efficient = False, math = True),
            dict(cudnn = True, flash = False, efficient = False, math = False),
            dict(cudnn = False, flash = True, efficient = False, math = False),
        ]

    def test_idempotent(self, recorder):
        F, _ = recorder
        wrapped = F.scaled_dot_product_attention
        fixes.fix_cudnn_sdpa_known_bad_shapes()
        assert F.scaled_dot_product_attention is wrapped

    def test_compiled_fullgraph(self, recorder, monkeypatch):
        # CPU tensors stand in for the trigger: the wrapper must trace without a graph break, and
        # the traced graph must carry the backend switch around the real SDPA.
        F, _ = recorder
        monkeypatch.setattr(fixes, "_sdpa_cudnn_known_bad", lambda *a: "dq_nan")
        real = torch._C._nn.scaled_dot_product_attention
        graphs = []

        def backend(gm, example_inputs):
            graphs.append(gm.code)
            return gm.forward

        torch._dynamo.reset()
        # Rebuild the wrapper over the real SDPA so the compiled call runs real attention.
        monkeypatch.setattr(F, "scaled_dot_product_attention", real)
        fixes.fix_cudnn_sdpa_known_bad_shapes()
        wrapped = F.scaled_dot_product_attention
        assert wrapped is not real

        def fn(q):
            return wrapped(q, q, q)

        q = torch.randn(1, 2, 4, 8)
        out = torch.compile(fn, backend = backend, fullgraph = True)(q)
        torch.testing.assert_close(out, real(q, q, q))
        code = "\n".join(graphs)
        # cuDNN switched off around the call by one graph call (not at trace time).
        assert "_sdpa_without_cudnn(" in code, code
        torch._dynamo.reset()


def test_cudnn_off_tests_the_key_length_value():
    with sdpa_kernel(ALL_BACKENDS):
        for length, off in ((192, True), (256, False), (None, True)):
            fixes._sdpa_cudnn_off(length)
            assert torch.backends.cuda.cudnn_sdp_enabled() is not off
            fixes._sdpa_cudnn_restore()
            assert torch.backends.cuda.cudnn_sdp_enabled()
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):  # cuDNN-only: left alone
        fixes._sdpa_cudnn_off(192)
        assert torch.backends.cuda.cudnn_sdp_enabled()
        fixes._sdpa_cudnn_restore()


@pytest.fixture
def cpu_dq_nan(monkeypatch):
    """Real CPU SDPA standing in for an affected H100: the device check is mocked (Dynamo reads
    is_cuda from the fake tensor), the rest of the dQ-NaN rule, key-length test included, is real."""
    import torch.nn.functional as F

    # Before the key-length helper existed the test was inline: s_kv % 128 == 64.
    length = getattr(fixes, "_sdpa_cudnn_dq_nan_length", lambda k: k.shape[-2] % 128 == 64)

    def known_bad(decode, dq, query, key, is_causal):
        hit = not is_causal and query.requires_grad and torch.is_grad_enabled() and length(key)
        return fixes._SDPA_CUDNN_DQ_NAN if hit else None

    real = torch._C._nn.scaled_dot_product_attention
    monkeypatch.setattr(F, "scaled_dot_product_attention", real)
    monkeypatch.setattr(fixes, "_sdpa_cudnn_known_bad_devices", lambda: (D0, D0))
    monkeypatch.setattr(fixes, "_sdpa_cudnn_known_bad", known_bad)
    monkeypatch.delenv("UNSLOTH_ALLOW_BUGGY_CUDNN_SDPA", raising = False)
    fixes.fix_cudnn_sdpa_known_bad_shapes()
    wrapped = F.scaled_dot_product_attention
    assert wrapped is not real
    torch._dynamo.reset()
    yield wrapped
    torch._dynamo.reset()


def _qkv(s_kv):
    q = torch.randn(1, 2, 8, 16, dtype = torch.float16, requires_grad = True)
    k = torch.randn(1, 2, s_kv, 16, dtype = torch.float16, requires_grad = True)
    return q, k, torch.randn_like(k, requires_grad = True)


@pytest.mark.parametrize("strict", [False, True])
def test_compiled_dynamic_key_length_is_not_specialized(cpu_dq_nan, strict):
    # Lengths on and off 64 mod 128 must share one graph (no s_kv % 128 guard), and a strict
    # mark_dynamic key length must still compile.
    wrapped = cpu_dq_nan
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    fn = torch.compile(lambda q, k, v: wrapped(q, k, v).float().sum(), backend = backend,
                       fullgraph = True, dynamic = True)  # fmt: skip
    with sdpa_kernel([SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION]):
        for s_kv in (192, 256, 320, 384, 448):
            q, k, v = _qkv(s_kv)
            if strict:
                torch._dynamo.mark_dynamic(k, 2, min = 128, max = 512)
                torch._dynamo.mark_dynamic(v, 2, min = 128, max = 512)
            fn(q, k, v).backward()
    assert len(graphs) == 1
    # The cuDNN switch is in the graph: the wrapper engaged.
    names = ("_sdpa_without_cudnn(", "_sdpa_cudnn_off(", "_sdpa_kernel(")
    assert any(name in graphs[0].code for name in names)


def test_compiled_call_reads_policy_at_call_time(cpu_dq_nan):
    # Compiled under one policy, called under another: the caller's later policy decides the
    # backend, and exactly that policy is left in place afterwards.
    wrapped = cpu_dq_nan
    fn = torch.compile(lambda q, k, v: wrapped(q, k, v), backend = lambda gm, ex: gm.forward,
                       fullgraph = True)  # fmt: skip
    q, k, v = _qkv(192)
    cuda = torch.backends.cuda
    with sdpa_kernel(ALL_BACKENDS):
        fn(q, k, v).sum().backward()
    with sdpa_kernel([SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION]):
        fn(q, k, v)
        assert (cuda.flash_sdp_enabled(), cuda.math_sdp_enabled(), cuda.cudnn_sdp_enabled()) == (
            False, True, True,
        )  # fmt: skip
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):  # cuDNN-only: left alone, as unwrapped SDPA
        with pytest.raises(RuntimeError, match = "No viable backend"):
            fn(q, k, v)
        assert not cuda.math_sdp_enabled() and cuda.cudnn_sdp_enabled()


_RAISING = []


def _raising_op():
    """An opaque op that raises at run time (as a CUDA OOM would), not while Dynamo traces it."""
    if not _RAISING:

        @torch.library.custom_op("unsloth_cudnn_sdpa_probe::boom", mutates_args = ())
        def boom(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("CUDA out of memory (simulated)")

        @boom.register_fake
        def _(q, k, v):
            return torch.empty_like(q)

        _RAISING.append(boom)
    return _RAISING[0]


@pytest.mark.parametrize("mode", ["plain", "eager_graph"])
def test_raising_sdpa_restores_the_policy(cpu_dq_nan, monkeypatch, mode):
    # SDPA raising mid-call (e.g. CUDA OOM) must still restore the caller's flags and leave no
    # unmatched switch, also from an executable FX graph, where a Python finally around separate
    # switch / restore nodes does not survive.
    import torch.nn.functional as F

    boom = _raising_op()
    monkeypatch.setattr(F, "scaled_dot_product_attention", lambda q, k, v, *a, **kw: boom(q, k, v))
    fixes.fix_cudnn_sdpa_known_bad_shapes()
    wrapped = F.scaled_dot_product_attention
    torch._dynamo.reset()
    fn = lambda q, k, v: wrapped(q, k, v)  # noqa: E731
    if mode == "eager_graph":
        fn = torch.compile(fn, backend = "eager", fullgraph = True, dynamic = False)
    depth = len(fixes._SDPA_CUDNN_SWITCHED)
    try:
        with sdpa_kernel(CUDNN_MATH):
            before = _sdp_flags()
            for _ in range(2):
                with pytest.raises(RuntimeError, match = "simulated"):
                    fn(*_qkv(192))
                # Checked inside the context: its exit would restore the flags itself.
                assert _sdp_flags() == before
                assert len(fixes._SDPA_CUDNN_SWITCHED) == depth
    finally:
        del fixes._SDPA_CUDNN_SWITCHED[depth:]
        torch._dynamo.reset()


_PROBE = []


def _probe_op():
    """A CompositeImplicitAutograd op that, like aten SDPA, picks its kernel from the live cuDNN
    flag when it is decomposed: per call in eager, once while tracing under AOTAutograd."""
    if not _PROBE:
        lib = torch.library.Library("unsloth_cudnn_sdpa_probe", "FRAGMENT")
        lib.define("probe(Tensor q, Tensor k, Tensor v) -> Tensor")
        lib.impl(
            "probe",
            lambda q, k, v: q * (2.0 if torch._C._get_cudnn_sdp_enabled() else 3.0),
            "CompositeImplicitAutograd",
        )
        _PROBE.append((lib, torch.ops.unsloth_cudnn_sdpa_probe.probe))
    return _PROBE[0][1]


@pytest.fixture
def probe_dq_nan(cpu_dq_nan, monkeypatch):
    """cpu_dq_nan over the probe op: an output of 2 * q ran the cuDNN kernel, 3 * q a fallback."""
    import torch.nn.functional as F

    probe = _probe_op()
    monkeypatch.setattr(F, "scaled_dot_product_attention", lambda q, k, v, *a, **kw: probe(q, k, v))
    fixes.fix_cudnn_sdpa_known_bad_shapes()
    torch._dynamo.reset()
    yield F.scaled_dot_product_attention
    torch._dynamo.reset()


CUDNN_ONLY = [SDPBackend.CUDNN_ATTENTION]
CUDNN_MATH = [SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION]


def _sdp_flags():
    cuda = torch.backends.cuda
    return (cuda.cudnn_sdp_enabled(), cuda.flash_sdp_enabled(), cuda.mem_efficient_sdp_enabled(),
            cuda.math_sdp_enabled())  # fmt: skip


@pytest.fixture
def cudnn_sdp_on():
    """Start from torch's default, cuDNN SDPA on: importing vLLM turns it off process-wide."""
    was = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(True)
    yield
    torch.backends.cuda.enable_cudnn_sdp(was)


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize("first", ["cudnn_only", "fallback"])
def test_aot_compiled_call_follows_a_later_policy(cudnn_sdp_on, probe_dq_nan, backend, first):
    # AOTAutograd fixes SDPA's backend while tracing and Dynamo does not guard the backend flags:
    # a call compiled under one policy and invoked under another must not keep the first one's
    # kernel (a cuDNN-only compile must not leave cuDNN in place once a fallback is enabled).
    wrapped = probe_dq_nan
    # Static shapes: a symbolic key length takes the conservative branch (cuDNN off) instead.
    fn = torch.compile(lambda q, k, v: wrapped(q, k, v), backend = backend, fullgraph = True,
                       dynamic = False)  # fmt: skip
    policies = [CUDNN_ONLY, CUDNN_MATH] if first == "cudnn_only" else [CUDNN_MATH, CUDNN_ONLY]
    for s_kv, hit in ((192, True), (256, False)):
        q, k, v = _qkv(s_kv)
        for policy in policies * 2:
            with sdpa_kernel(policy):
                before = _sdp_flags()
                ratio = (fn(q, k, v) / q).detach().flatten()[0].item()
                expected = 3.0 if hit and policy is CUDNN_MATH else 2.0
                assert ratio == pytest.approx(expected, rel = 0.01), (s_kv, policy)  # fp16 2q / 3q
                # Matches an uncompiled call, and leaves the caller's policy exactly in place.
                eager = (wrapped(q, k, v) / q).detach().flatten()[0].item()
                assert eager == pytest.approx(expected, rel = 0.01)
                assert _sdp_flags() == before
    assert torch.backends.cuda.cudnn_sdp_enabled()


def test_opt_out_and_unaffected_host(monkeypatch):
    import torch.nn.functional as F

    sentinel = lambda *a, **k: None  # noqa: E731
    monkeypatch.setattr(F, "scaled_dot_product_attention", sentinel)
    monkeypatch.setattr(fixes, "_sdpa_cudnn_known_bad_devices", lambda: (D0, D0))
    monkeypatch.setenv("UNSLOTH_ALLOW_BUGGY_CUDNN_SDPA", "1")
    fixes.fix_cudnn_sdpa_known_bad_shapes()
    assert F.scaled_dot_product_attention is sentinel
    monkeypatch.delenv("UNSLOTH_ALLOW_BUGGY_CUDNN_SDPA")
    monkeypatch.setattr(fixes, "_sdpa_cudnn_known_bad_devices", lambda: (NONE, NONE))
    fixes.fix_cudnn_sdpa_known_bad_shapes()
    assert F.scaled_dot_product_attention is sentinel
