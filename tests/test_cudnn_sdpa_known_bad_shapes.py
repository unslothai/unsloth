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
        # cuDNN-free set before the call, the caller's full set restored after it.
        first_switch = code.index("_sdpa_kernel(")
        call = code.index("_nn.scaled_dot_product_attention(")
        assert first_switch < call < code.index("'CUDNN_ATTENTION'"), code
        torch._dynamo.reset()


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
