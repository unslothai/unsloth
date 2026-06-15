# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the deterministic MTP VRAM reserve used by load-time auto-fit.

reserve(ctx) = draft_KV(ctx, draft_cache_type) + separate_drafter_weights, sized
from GGUF dims (embedded head from the main model's dims; separate drafter from
its own KV). Anchors checked against real llama-server measurements. Pure: no
GPU, network, subprocess, or GGUF I/O."""

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
    _CTX_FIT_VRAM_FRACTION,
    LlamaCppBackend,
    _extra_args_draft_cache_types,
    _extra_args_mtp_draft_path,
    _extra_args_n_ubatch,
    _extra_args_requests_mtp,
    _extra_args_spec_draft_n_max,
    _kv_bytes_per_elem,
)

MIB = 1024 * 1024
GIB = 1024**3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(
    *,
    nextn = 1,
    n_kv_heads = 4,
    n_heads = 24,
    kv_key_length = 256,
    kv_value_length = 256,
    embedding_length = 5120,
    n_layers = 65,
    native_ctx = 262144,
):
    """Qwen3.6-27B-MTP-class backend (embedded head) with the MTP-math dims."""
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
    b._draft_backend_cache = None
    return b


class _StubDrafter:
    """Stand-in for a separate drafter backend (no GGUF I/O)."""

    def __init__(self, kv_per_token):
        self._kv_per_token = kv_per_token

    def _can_estimate_kv(self):
        return True

    def _estimate_kv_cache_bytes(
        self,
        n_ctx,
        cache_type = None,
        **_k,
    ):
        bpe = _kv_bytes_per_elem(cache_type)
        return 0 if n_ctx <= 0 else int(n_ctx * self._kv_per_token * bpe / 2.0)


# ---------------------------------------------------------------------------
# Embedded draft KV: deterministic from nextn dims, scales with ctx + draft type
# ---------------------------------------------------------------------------


class TestEmbeddedDraftKv:
    def test_scales_linearly_with_context(self):
        b = _make_backend()
        kv_8k = b._mtp_draft_kv_bytes(8192)
        kv_16k = b._mtp_draft_kv_bytes(16384)
        kv_64k = b._mtp_draft_kv_bytes(65536)
        assert kv_8k and kv_16k and kv_64k
        assert kv_16k == pytest.approx(2 * kv_8k)
        assert kv_64k == pytest.approx(8 * kv_8k)

    def test_value_matches_dim_formula_f16(self):
        # nextn(1) * n_kv(4) * (256+256) * 2(f16) * ctx -- no magic safety factor.
        b = _make_backend()
        ctx = 131072
        expected = int(1 * 4 * 512 * 2.0 * ctx)
        assert b._mtp_draft_kv_bytes(ctx) == expected
        # And that is 512 MiB, matching the measured 27B draft-KV slope (~4 MiB/1k).
        assert b._mtp_draft_kv_bytes(ctx) / MIB == pytest.approx(512, abs = 1)

    def test_scales_with_nextn_predict_layers(self):
        one = _make_backend(nextn = 1)._mtp_draft_kv_bytes(65536)
        two = _make_backend(nextn = 2)._mtp_draft_kv_bytes(65536)
        assert two == pytest.approx(2 * one)

    def test_draft_cache_type_changes_bytes(self):
        b = _make_backend()
        f16 = b._mtp_draft_kv_bytes(65536, draft_cache_type_k = "f16", draft_cache_type_v = "f16")
        q8 = b._mtp_draft_kv_bytes(65536, draft_cache_type_k = "q8_0", draft_cache_type_v = "q8_0")
        q4 = b._mtp_draft_kv_bytes(65536, draft_cache_type_k = "q4_0", draft_cache_type_v = "q4_0")
        assert q8 < f16 and q4 < q8
        assert q8 == pytest.approx(f16 * (34 / 32) / 2.0)
        assert q4 == pytest.approx(f16 * 0.5625 / 2.0)

    def test_draft_kv_split_axes_no_under_reserve(self):
        # One-sided q4_0 (V left f16) must NOT be applied to both axes.
        b = _make_backend()
        both_q4 = b._mtp_draft_kv_bytes(131072, draft_cache_type_k = "q4_0", draft_cache_type_v = "q4_0")
        k_only = b._mtp_draft_kv_bytes(131072, draft_cache_type_k = "q4_0")  # V defaults f16
        both_f16 = b._mtp_draft_kv_bytes(131072, draft_cache_type_k = "f16", draft_cache_type_v = "f16")
        assert both_q4 < k_only < both_f16  # split sits between, never collapses to q4_0

    def test_none_when_dims_missing(self):
        assert _make_backend(nextn = 0)._mtp_draft_kv_bytes(65536) is None
        assert _make_backend(kv_key_length = None)._mtp_draft_kv_bytes(65536) is None
        assert _make_backend()._mtp_draft_kv_bytes(0) is None


# ---------------------------------------------------------------------------
# Separate drafter (Gemma): sized from the drafter GGUF's own dims + weights
# ---------------------------------------------------------------------------


class TestSeparateDrafter:
    def test_uses_drafter_kv_and_weights(self, monkeypatch):
        b = _make_backend(nextn = None)  # main has no embedded head
        stub = _StubDrafter(kv_per_token = 2000)
        monkeypatch.setattr(b, "_draft_backend_for", lambda path: stub)
        ctx = 65536
        kv = b._mtp_draft_kv_bytes(ctx, drafter_path = "/m/draft.gguf")
        assert kv == stub._estimate_kv_cache_bytes(ctx)
        total = b._estimate_mtp_overhead_bytes(
            ctx, drafter_path = "/m/draft.gguf", draft_weights_bytes = GIB
        )
        assert total == kv + GIB

    def test_drafter_kv_scales_with_context(self, monkeypatch):
        b = _make_backend(nextn = None)
        monkeypatch.setattr(b, "_draft_backend_for", lambda path: _StubDrafter(2000))
        a = b._mtp_draft_kv_bytes(16384, drafter_path = "/m/d.gguf")
        c = b._mtp_draft_kv_bytes(65536, drafter_path = "/m/d.gguf")
        assert c == pytest.approx(4 * a)

    def test_none_when_drafter_unreadable(self, monkeypatch):
        b = _make_backend(nextn = None)
        monkeypatch.setattr(b, "_draft_backend_for", lambda path: None)
        assert b._mtp_draft_kv_bytes(65536, drafter_path = "/m/d.gguf") is None
        assert b._estimate_mtp_overhead_bytes(65536, drafter_path = "/m/d.gguf") is None


# ---------------------------------------------------------------------------
# Total overhead = draft KV (+ separate drafter weights); no verify constant
# ---------------------------------------------------------------------------


class TestOverheadTotal:
    def test_equals_draft_kv_for_embedded(self):
        b = _make_backend()
        for ctx in (16384, 65536, 131072):
            assert b._estimate_mtp_overhead_bytes(ctx) == b._mtp_draft_kv_bytes(ctx)

    def test_does_not_depend_on_n_max(self):
        # The verify buffer (the only n_max-dependent term) rides in headroom now.
        b = _make_backend()
        assert b._estimate_mtp_overhead_bytes(
            65536, spec_draft_n_max = 2
        ) == b._estimate_mtp_overhead_bytes(65536, spec_draft_n_max = 6)

    def test_none_when_draft_kv_unsizable(self):
        assert _make_backend(nextn = 0)._estimate_mtp_overhead_bytes(65536) is None

    def test_includes_separate_drafter_weights(self):
        b = _make_backend()
        base = b._estimate_mtp_overhead_bytes(65536)
        with_w = b._estimate_mtp_overhead_bytes(65536, draft_weights_bytes = GIB)
        assert with_w - base == GIB

    @pytest.mark.parametrize(
        "ctx,measured_draft_kv_mib",
        # Measured 27B MTP delta minus the (headroom-covered) ~500 MiB verify
        # buffer leaves the draft KV; the deterministic estimate must match it.
        [(16384, 64), (65536, 256), (131072, 512)],
    )
    def test_draft_kv_matches_measured(self, ctx, measured_draft_kv_mib):
        b = _make_backend()
        pred = b._estimate_mtp_overhead_bytes(ctx) / MIB
        assert pred == pytest.approx(measured_draft_kv_mib, abs = 2)


# ---------------------------------------------------------------------------
# _fit_context_to_vram: MTP reserve lowers the chosen context
# ---------------------------------------------------------------------------


class TestFitContextWithMtp:
    def _fit_backend(self, kv_per_token = 325_000):
        b = _make_backend()
        b._can_estimate_kv = lambda: True
        b._estimate_kv_cache_bytes = lambda n, _t = None, **_k: (0 if n <= 0 else n * kv_per_token)
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
            mtp_overhead_fn = lambda c: b._estimate_mtp_overhead_bytes(c) or 0,
        )
        assert 0 < with_mtp < without

    def test_quantized_draft_kv_allows_more_context(self):
        # q4_0 draft KV is smaller than f16 -> fit can keep a larger context.
        b = self._fit_backend()
        avail_mib, model = 24_000, 8 * GIB
        f16 = b._fit_context_to_vram(
            131072,
            avail_mib,
            model,
            mtp_overhead_fn = lambda c: b._estimate_mtp_overhead_bytes(
                c, draft_cache_type_k = "f16", draft_cache_type_v = "f16"
            )
            or 0,
        )
        q4 = b._fit_context_to_vram(
            131072,
            avail_mib,
            model,
            mtp_overhead_fn = lambda c: b._estimate_mtp_overhead_bytes(
                c, draft_cache_type_k = "q4_0", draft_cache_type_v = "q4_0"
            )
            or 0,
        )
        assert 0 < f16 <= q4

    def test_no_mtp_unchanged(self):
        b = self._fit_backend()
        avail_mib, model = 24_000, 8 * GIB
        a = b._fit_context_to_vram(131072, avail_mib, model)
        bb = b._fit_context_to_vram(
            131072, avail_mib, model, mtp_engaged = False, mtp_overhead_fn = None
        )
        assert a == bb

    def test_chosen_context_actually_fits_budget(self):
        b = self._fit_backend()
        avail_mib, model = 24_000, 8 * GIB
        fn = lambda c: b._estimate_mtp_overhead_bytes(c) or 0  # noqa: E731
        ctx = b._fit_context_to_vram(131072, avail_mib, model, mtp_overhead_fn = fn)
        budget = avail_mib * MIB * _CTX_FIT_VRAM_FRACTION
        assert model + b._estimate_kv_cache_bytes(ctx) + fn(ctx) <= budget


# ---------------------------------------------------------------------------
# extra_args parsing: detect user-enabled MTP + draft depth + draft KV type
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
            (["--draft-max", "6"], 6),
            (["--draft-max=4"], 4),
            (["--spec-type", "draft-mtp", "--draft-max", "6"], 6),
            (["--spec-draft-n-max", "2", "--draft-max", "5"], 5),
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
            (["--model-draft", "--spec-type"], None),
            (["-c", "4096"], None),
            (None, None),
        ],
    )
    def test_mtp_draft_path(self, args, expected):
        assert _extra_args_mtp_draft_path(args) == expected

    @pytest.mark.parametrize(
        "args,expected",
        [
            (["--cache-type-k-draft", "q8_0"], ("q8_0", None)),
            (["--spec-draft-type-k", "q4_0"], ("q4_0", None)),
            (["-ctkd", "q8_0"], ("q8_0", None)),
            (["--cache-type-v-draft", "q4_0"], (None, "q4_0")),  # K stays f16, V only
            (["--cache-type-k-draft", "q4_0", "--cache-type-v-draft", "q8_0"], ("q4_0", "q8_0")),
            (["--cache-type-k-draft=q8_0"], ("q8_0", None)),
            (["--cache-type-k", "q8_0"], (None, None)),  # main type, not draft
            (["-c", "4096"], (None, None)),
            (None, (None, None)),
        ],
    )
    def test_draft_cache_types(self, args, expected):
        assert _extra_args_draft_cache_types(args) == expected

    @pytest.mark.parametrize(
        "args,expected",
        [
            (["--ubatch", "2048"], 2048),
            (["--ubatch-size", "1024"], 1024),
            (["-ub", "4096"], 4096),
            (["--ubatch=512"], 512),
            (["-c", "4096"], None),
            (None, None),
        ],
    )
    def test_n_ubatch(self, args, expected):
        assert _extra_args_n_ubatch(args) == expected


# ---------------------------------------------------------------------------
# Regression: the reported Qwen3.6-27B MTP / 24 GB scenario
# ---------------------------------------------------------------------------


def test_qwen36_class_regression_picks_lower_ctx_with_mtp():
    """A 24 GB card that auto-picks a high context without MTP must pick a
    strictly lower one once the MTP draft reserve is accounted for."""
    b = _make_backend()
    b._can_estimate_kv = lambda: True
    b._estimate_kv_cache_bytes = lambda n, _t = None, **_k: (0 if n <= 0 else int(n * 66_000))
    avail_mib = 24_000
    model = int(17.9 * GIB)  # UD-Q4_K_XL weights
    no_mtp = b._fit_context_to_vram(262144, avail_mib, model)
    with_mtp = b._fit_context_to_vram(
        262144,
        avail_mib,
        model,
        mtp_overhead_fn = lambda c: b._estimate_mtp_overhead_bytes(c) or 0,
    )
    assert 0 < with_mtp < no_mtp
