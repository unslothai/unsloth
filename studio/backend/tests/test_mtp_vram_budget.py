# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the byte-accurate MTP VRAM reserve used by load-time auto-fit.

Guards the regression where Studio advertised a context (e.g. 110k for the
Qwen3.6-27B MTP GGUF) that fit on paper but OOMed mid-generation: the MTP draft
path's VRAM (its own attention KV cache, which grows with context, plus a verify
compute buffer that grows with --spec-draft-n-max) was reserved as a flat 5% of
total VRAM instead of the real, context-aware amount.

Coefficients are calibrated against real llama-server measurements
(Qwen3.6-27B-MTP UD-Q2_K_XL, llama.cpp b9625, B200) -- see
scripts/mtp_vram_calib.py and outputs/mtp_vram_calib.csv. The anchors below assert
the model stays conservative (predicted >= measured) at those points.

Pure: no GPU, network, subprocess, or GGUF I/O.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Stub heavy/unavailable deps before importing the module under test.
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

sys.modules.setdefault("structlog", _types.ModuleType("structlog"))

_httpx_stub = _types.ModuleType("httpx")
for _exc_name in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))
_httpx_stub.Timeout = type("Timeout", (), {"__init__": lambda self, *a, **kw: None})
_httpx_stub.Client = type(
    "Client",
    (),
    {
        "__init__": lambda self, **kw: None,
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: None,
    },
)
sys.modules.setdefault("httpx", _httpx_stub)

from core.inference.llama_cpp import (  # noqa: E402
    LlamaCppBackend,
    _extra_args_mtp_draft_path,
    _extra_args_requests_mtp,
    _extra_args_spec_draft_n_max,
)

MIB = 1024 * 1024
GIB = 1024**3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(
    *,
    nextn=1,
    n_kv_heads=4,
    n_heads=24,
    kv_key_length=256,
    kv_value_length=256,
    embedding_length=5120,
    n_layers=65,
    native_ctx=262144,
):
    """Qwen3.6-27B-MTP-class backend with only the dims the MTP math reads."""
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._nextn_predict_layers = nextn
    b._n_kv_heads = n_kv_heads
    b._n_heads = n_heads
    b._kv_key_length = kv_key_length
    b._kv_value_length = kv_value_length
    b._embedding_length = embedding_length
    b._n_layers = n_layers
    b._context_length = native_ctx
    # Hybrid attention/Mamba (qwen35 path) + remaining KV-estimator fields.
    b._shared_kv_layers = 0
    b._kv_lora_rank = None
    b._sliding_window = None
    b._sliding_window_pattern = None
    b._ssm_inner_size = 6144
    b._full_attention_interval = 4
    b._key_length_mla = None
    b._n_kv_heads_by_layer = None
    b._kv_key_length_swa = None
    b._kv_value_length_swa = None
    return b


# ---------------------------------------------------------------------------
# Draft KV term: scales with context, sized from nextn head dims, f16
# ---------------------------------------------------------------------------


class TestDraftKvBytes:
    def test_scales_linearly_with_context(self):
        b = _make_backend()
        kv_8k = b._mtp_draft_kv_bytes(8192)
        kv_16k = b._mtp_draft_kv_bytes(16384)
        kv_64k = b._mtp_draft_kv_bytes(65536)
        assert kv_8k and kv_16k and kv_64k
        assert kv_16k == pytest.approx(2 * kv_8k)
        assert kv_64k == pytest.approx(8 * kv_8k)

    def test_value_matches_dim_formula_f16(self):
        # SAFETY(1.25) * nextn(1) * n_kv(4) * (256+256) * 2(f16) * ctx
        b = _make_backend()
        ctx = 131072
        expected = int(1.25 * 1 * 4 * 512 * 2 * ctx)
        assert b._mtp_draft_kv_bytes(ctx) == expected

    def test_scales_with_nextn_predict_layers(self):
        one = _make_backend(nextn=1)._mtp_draft_kv_bytes(65536)
        two = _make_backend(nextn=2)._mtp_draft_kv_bytes(65536)
        assert two == pytest.approx(2 * one)

    def test_none_when_dims_missing(self):
        assert _make_backend(nextn=0)._mtp_draft_kv_bytes(65536) is None
        assert _make_backend(kv_key_length=None)._mtp_draft_kv_bytes(65536) is None
        assert _make_backend()._mtp_draft_kv_bytes(0) is None

    def test_independent_of_main_cache_type(self):
        # The draft KV is always f16 (llama.cpp default for the MTP draft
        # context), so it does not take a main cache_type_kv argument at all.
        b = _make_backend()
        assert b._mtp_draft_kv_bytes(65536) > 0


# ---------------------------------------------------------------------------
# Verify term: scales with n_max, NOT with context
# ---------------------------------------------------------------------------


class TestVerifyBytes:
    def test_scales_with_n_max(self):
        b = _make_backend()
        v2 = b._mtp_verify_bytes(2)
        v4 = b._mtp_verify_bytes(4)
        assert v4 > v2
        # Per-n_max term is linear after removing the fixed base.
        from core.inference.llama_cpp import _MTP_FIXED_OVERHEAD_BYTES

        assert (v4 - _MTP_FIXED_OVERHEAD_BYTES) == pytest.approx(2 * (v2 - _MTP_FIXED_OVERHEAD_BYTES))

    def test_per_embd_value(self):
        # 32768 bytes * n_embd * n_max + fixed base
        from core.inference.llama_cpp import (
            _MTP_FIXED_OVERHEAD_BYTES,
            _MTP_VERIFY_BYTES_PER_EMBD,
        )

        b = _make_backend(embedding_length=5120)
        assert b._mtp_verify_bytes(3) == _MTP_VERIFY_BYTES_PER_EMBD * 5120 * 3 + _MTP_FIXED_OVERHEAD_BYTES

    def test_fixed_floor_when_embd_unknown(self):
        from core.inference.llama_cpp import _MTP_FIXED_OVERHEAD_BYTES

        b = _make_backend(embedding_length=0)
        assert b._mtp_verify_bytes(6) == _MTP_FIXED_OVERHEAD_BYTES


# ---------------------------------------------------------------------------
# Total overhead + conservativeness vs real measurements
# ---------------------------------------------------------------------------


class TestOverheadTotal:
    def test_none_when_draft_kv_unsizable(self):
        assert _make_backend(nextn=0)._estimate_mtp_overhead_bytes(65536, spec_draft_n_max=2) is None

    def test_includes_separate_drafter_weights(self):
        b = _make_backend()
        base = b._estimate_mtp_overhead_bytes(65536, spec_draft_n_max=2)
        with_w = b._estimate_mtp_overhead_bytes(65536, spec_draft_n_max=2, draft_weights_bytes=GIB)
        assert with_w - base == GIB

    # (ctx, n_max) -> measured overhead in MiB (B200, b9625, UD-Q2_K_XL).
    @pytest.mark.parametrize(
        "ctx,n_max,measured_mib",
        [
            (8192, 2, 466),
            (65536, 2, 692),
            (131072, 2, 1012),
            (262144, 2, 1652),
            (8192, 6, 1100),
            (131072, 6, 1646),
            (262144, 6, 2286),
        ],
    )
    def test_conservative_vs_measured(self, ctx, n_max, measured_mib):
        b = _make_backend()
        pred_mib = b._estimate_mtp_overhead_bytes(ctx, spec_draft_n_max=n_max) / MIB
        # Never under-reserve; never waste more than ~150 MiB over measured.
        assert measured_mib <= pred_mib <= measured_mib + 150


# ---------------------------------------------------------------------------
# _fit_context_to_vram: MTP reserve lowers the chosen context
# ---------------------------------------------------------------------------


class TestFitContextWithMtp:
    def _fit_backend(self, kv_per_token=325_000):
        b = _make_backend()
        b._can_estimate_kv = lambda: True
        b._estimate_kv_cache_bytes = lambda n, _t=None, **_k: (0 if n <= 0 else n * kv_per_token)
        return b

    def test_overhead_fn_lowers_context(self):
        b = self._fit_backend()
        avail_mib = 24_000
        model = 8 * GIB
        without = b._fit_context_to_vram(131072, avail_mib, model)
        with_mtp = b._fit_context_to_vram(
            131072,
            avail_mib,
            model,
            mtp_overhead_fn=lambda c: b._estimate_mtp_overhead_bytes(c, spec_draft_n_max=2) or 0,
        )
        assert 0 < with_mtp < without

    def test_bigger_n_max_lowers_context_further(self):
        b = self._fit_backend()
        avail_mib, model = 24_000, 8 * GIB
        n2 = b._fit_context_to_vram(
            131072, avail_mib, model,
            mtp_overhead_fn=lambda c: b._estimate_mtp_overhead_bytes(c, spec_draft_n_max=2) or 0,
        )
        n6 = b._fit_context_to_vram(
            131072, avail_mib, model,
            mtp_overhead_fn=lambda c: b._estimate_mtp_overhead_bytes(c, spec_draft_n_max=6) or 0,
        )
        assert 0 < n6 <= n2

    def test_no_mtp_unchanged(self):
        # mtp_overhead_fn=None and mtp_engaged=False -> identical to legacy.
        b = self._fit_backend()
        avail_mib, model = 24_000, 8 * GIB
        a = b._fit_context_to_vram(131072, avail_mib, model)
        bb = b._fit_context_to_vram(131072, avail_mib, model, mtp_engaged=False, mtp_overhead_fn=None)
        assert a == bb

    def test_chosen_context_actually_fits_budget(self):
        # The returned ctx must satisfy weights + KV + MTP <= 90% budget.
        b = self._fit_backend()
        avail_mib, model = 24_000, 8 * GIB
        fn = lambda c: b._estimate_mtp_overhead_bytes(c, spec_draft_n_max=2) or 0  # noqa: E731
        ctx = b._fit_context_to_vram(131072, avail_mib, model, mtp_overhead_fn=fn)
        budget = avail_mib * MIB * 0.90
        assert model + b._estimate_kv_cache_bytes(ctx) + fn(ctx) <= budget


# ---------------------------------------------------------------------------
# extra_args parsing: detect user-enabled MTP + draft depth
# ---------------------------------------------------------------------------


class TestExtraArgsMtpDetection:
    @pytest.mark.parametrize(
        "args,expected",
        [
            (["--spec-type", "draft-mtp"], True),
            (["--spec-type", "mtp"], True),
            (["--spec-type", "ngram-mod,draft-mtp"], True),
            (["--spec-type=draft-mtp"], True),
            (["--spec-type", "ngram-mod"], False),
            (["--spec-default"], False),
            (["-c", "131072"], False),
            (None, False),
            ([], False),
        ],
    )
    def test_requests_mtp(self, args, expected):
        assert _extra_args_requests_mtp(args) is expected

    @pytest.mark.parametrize(
        "args,expected",
        [
            (["--spec-draft-n-max", "4"], 4),
            (["--spec-draft-n-max=6"], 6),
            (["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"], 3),
            (["--spec-draft-n-max", "2", "--spec-draft-n-max", "5"], 5),  # last wins
            (["--spec-draft-n-max", "notanint"], None),
            (["-c", "4096"], None),
            (None, None),
            # Legacy --draft-max alias (older llama.cpp builds).
            (["--draft-max", "6"], 6),
            (["--draft-max=4"], 4),
            (["--spec-type", "draft-mtp", "--draft-max", "6"], 6),
            (["--spec-draft-n-max", "2", "--draft-max", "5"], 5),  # last wins across aliases
        ],
    )
    def test_spec_draft_n_max(self, args, expected):
        assert _extra_args_spec_draft_n_max(args) == expected

    @pytest.mark.parametrize(
        "args,expected",
        [
            (["--model-draft", "/m/draft.gguf"], "/m/draft.gguf"),
            (["--spec-draft-model", "/m/draft.gguf"], "/m/draft.gguf"),
            (["-md", "/m/draft.gguf"], "/m/draft.gguf"),
            (["--model-draft=/m/draft.gguf"], "/m/draft.gguf"),
            (["--model-draft", "--spec-type"], None),  # next token is a flag
            (["-c", "4096"], None),
            (None, None),
        ],
    )
    def test_mtp_draft_path(self, args, expected):
        assert _extra_args_mtp_draft_path(args) == expected


# ---------------------------------------------------------------------------
# Regression: the reported Qwen3.6-27B MTP / 24 GB scenario
# ---------------------------------------------------------------------------


def test_qwen36_class_regression_picks_lower_ctx_with_mtp():
    """A 24 GB card that auto-picks a high context without MTP must pick a
    strictly lower one once the MTP draft reserve is accounted for."""
    b = _make_backend()
    b._can_estimate_kv = lambda: True
    # Real-ish hybrid KV slope so KV grows with ctx like the actual model.
    b._estimate_kv_cache_bytes = lambda n, _t=None, **_k: (0 if n <= 0 else int(n * 66_000))
    avail_mib = 24_000
    model = int(17.9 * GIB)  # UD-Q4_K_XL weights
    no_mtp = b._fit_context_to_vram(262144, avail_mib, model)
    with_mtp = b._fit_context_to_vram(
        262144, avail_mib, model,
        mtp_overhead_fn=lambda c: b._estimate_mtp_overhead_bytes(c, spec_draft_n_max=6) or 0,
    )
    assert 0 < with_mtp < no_mtp
