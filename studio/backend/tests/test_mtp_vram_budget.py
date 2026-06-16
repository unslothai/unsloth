# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the deterministic MTP VRAM reserve used by load-time auto-fit.

reserve(ctx) = draft_KV(ctx, draft_cache_type) + separate_drafter_weights, sized
from GGUF dims (embedded head from the main model's dims; separate drafter from
its own KV). Anchors checked against real llama-server measurements. Pure: no
GPU, network, subprocess, or GGUF I/O."""

from __future__ import annotations

import inspect
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

# httpx -- only stub when the real library is missing. Unconditional stubbing
# shadows HTTPError/Response that huggingface_hub.errors imports at load time,
# silently breaking the transformers introspection tier in tests collected after
# this one (the stub leaks via sys.modules for the whole session).
try:
    import httpx as _httpx_real  # noqa: F401
except ImportError:
    _httpx_stub = _types.ModuleType("httpx")
    for _exc_name in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
        "HTTPError",
        "RequestError",
    ):
        setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))
    _httpx_stub.Timeout = type("Timeout", (), {"__init__": lambda self, *a, **kw: None})
    _httpx_stub.Response = type("Response", (), {})
    _httpx_stub.Client = type(
        "Client",
        (),
        {
            "__init__": lambda self, **kw: None,
            "__enter__": lambda self: self,
            "__exit__": lambda self, *a: None,
        },
    )
    sys.modules["httpx"] = _httpx_stub

from core.inference.llama_cpp import (  # noqa: E402
    _CTX_FIT_VRAM_FRACTION,
    LlamaCppBackend,
    _extra_args_draft_cache_types,
    _extra_args_draft_offloaded_to_cpu,
    _extra_args_mtp_draft_path,
    _extra_args_n_ubatch,
    _extra_args_requests_mtp,
    _extra_args_requests_separate_draft,
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

    def test_embedded_draft_kv_floored_at_f16(self):
        # The embedded MTP head is one layer, so llama.cpp's quantized-KV
        # overhead is not amortized: a quantized draft KV fits LESS context than
        # f16, not more (ggml-org/llama.cpp#24102). The embedded reserve floors a
        # quantized draft type at f16 (never under-reserved); f32 still costs more.
        b = _make_backend()
        f16 = b._mtp_draft_kv_bytes(65536, draft_cache_type_k = "f16", draft_cache_type_v = "f16")
        q8 = b._mtp_draft_kv_bytes(65536, draft_cache_type_k = "q8_0", draft_cache_type_v = "q8_0")
        q4 = b._mtp_draft_kv_bytes(65536, draft_cache_type_k = "q4_0", draft_cache_type_v = "q4_0")
        f32 = b._mtp_draft_kv_bytes(65536, draft_cache_type_k = "f32", draft_cache_type_v = "f32")
        assert q8 == f16 and q4 == f16  # quantized draft KV priced as f16, not less
        assert f32 == pytest.approx(f16 * 2.0)  # f32 genuinely larger, not floored

    def test_draft_kv_split_axes_no_under_reserve(self):
        # A quantized draft type on either or both axes never reserves below the
        # all-f16 value for the single-layer embedded head (the f16 floor; #24102).
        b = _make_backend()
        both_q4 = b._mtp_draft_kv_bytes(
            131072, draft_cache_type_k = "q4_0", draft_cache_type_v = "q4_0"
        )
        k_only = b._mtp_draft_kv_bytes(131072, draft_cache_type_k = "q4_0")  # V defaults f16
        both_f16 = b._mtp_draft_kv_bytes(131072, draft_cache_type_k = "f16", draft_cache_type_v = "f16")
        assert both_q4 == k_only == both_f16  # floored at f16, never under-reserved

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

    def test_keeps_weights_when_drafter_kv_unsizable(self, monkeypatch):
        # KV can't be sized (exotic/remote drafter), but the local weights are
        # known: reserve the weights so a drafter larger than the flat fallback
        # cushion can't slip through and OOM (Finding C). Nothing known -> None.
        b = _make_backend(nextn = None)
        monkeypatch.setattr(b, "_draft_backend_for", lambda path: None)
        assert b._mtp_draft_kv_bytes(65536, drafter_path = "/m/d.gguf") is None
        assert (
            b._estimate_mtp_overhead_bytes(
                65536, drafter_path = "/m/d.gguf", draft_weights_bytes = 3 * GIB
            )
            == 3 * GIB
        )
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

    def test_quantized_embedded_draft_kv_does_not_inflate_context(self):
        # For the single-layer embedded head, quantizing the draft KV does NOT
        # buy more context (it fits less in practice; ggml-org/llama.cpp#24102),
        # so the f16-floored reserve advertises the same context as f16 -- never
        # a larger one off a smaller (unsafe) reserve.
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
        assert 0 < q4 == f16

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
        assert _extra_args_requests_mtp(args, env = {}) is expected

    def test_requests_mtp_env(self):
        # The child honors LLAMA_ARG_SPEC_TYPE; env-requested MTP must reserve too.
        assert _extra_args_requests_mtp([], env = {"LLAMA_ARG_SPEC_TYPE": "draft-mtp"}) is True
        assert _extra_args_requests_mtp([], env = {"LLAMA_ARG_SPEC_TYPE": "ngram-mod,mtp"}) is True
        assert _extra_args_requests_mtp([], env = {"LLAMA_ARG_SPEC_TYPE": "draft-simple"}) is False
        assert _extra_args_requests_mtp([], env = {"LLAMA_ARG_SPEC_TYPE": "none"}) is False

    def test_requests_mtp_effective_spec_type(self):
        # llama.cpp uses the LAST CLI --spec-type and ignores the env when any CLI
        # --spec-type is present. The reserve must track that effective value, not
        # any earlier/MTP-ish one, or it over-reserves a drafter the launch won't
        # load (Finding B).
        env_mtp = {"LLAMA_ARG_SPEC_TYPE": "draft-mtp"}
        # Later CLI value overrides an earlier MTP one (last-wins).
        assert (
            _extra_args_requests_mtp(
                ["--spec-type", "draft-mtp", "--spec-type", "ngram-mod"], env = {}
            )
            is False
        )
        # A non-MTP CLI flag overrides a stale MTP env.
        assert _extra_args_requests_mtp(["--spec-type", "ngram-mod"], env = env_mtp) is False
        assert _extra_args_requests_mtp(["--spec-type", "none"], env = env_mtp) is False
        # A later MTP CLI value still engages.
        assert (
            _extra_args_requests_mtp(
                ["--spec-type", "ngram-mod", "--spec-type", "draft-mtp"], env = {}
            )
            is True
        )
        # Same precedence for separate (draft-simple/eagle3) detection.
        assert (
            _extra_args_requests_separate_draft(
                ["--spec-type", "draft-simple", "--spec-type", "ngram-mod"], env = {}
            )
            is False
        )
        assert (
            _extra_args_requests_separate_draft(
                ["--spec-type", "ngram-mod"], env = {"LLAMA_ARG_SPEC_TYPE": "draft-simple"}
            )
            is False
        )

    @pytest.mark.parametrize(
        "args,expected",
        [
            (["--spec-type", "draft-simple"], True),
            (["--spec-type", "draft-eagle3"], True),
            (["--spec-type=draft-eagle3"], True),
            (["--spec-type", "draft-mtp"], False),  # MTP path handles this one
            (["--spec-type", "ngram-mod"], False),  # loads no draft model
            (["-c", "4096"], False),
            (None, False),
        ],
    )
    def test_requests_separate_draft(self, args, expected):
        assert _extra_args_requests_separate_draft(args, env = {}) is expected

    def test_requests_separate_draft_env(self):
        assert (
            _extra_args_requests_separate_draft([], env = {"LLAMA_ARG_SPEC_TYPE": "draft-simple"})
            is True
        )
        assert (
            _extra_args_requests_separate_draft([], env = {"LLAMA_ARG_SPEC_TYPE": "draft-mtp"})
            is False
        )

    def test_load_model_reserves_for_non_mtp_draft_modes(self):
        # load_model engages the draft reserve for a non-MTP model-based draft mode
        # only when extras also name a drafter (else nothing is loaded to reserve).
        # Strip all whitespace so the check survives any line-wrapping the
        # formatter applies to the call (pre-commit black wraps long lines).
        compact = "".join(inspect.getsource(LlamaCppBackend.load_model).split())
        assert "_user_draft_via_extras" in compact
        # called with extra_args (an env kwarg may follow); prefix match stays
        # robust to that and to any formatter line-wrapping.
        assert "_extra_args_requests_separate_draft(extra_args" in compact
        assert "or_user_draft_via_extras" in compact  # OR'd into the reserve gate

    def test_load_model_gates_env_spec_type_on_off_mode(self):
        # LLAMA_ARG_SPEC_TYPE only reaches the child when Studio emits no spec
        # flag (UI mode "off", no user --spec-type); otherwise the emitted
        # --spec-type/--spec-default overrides the env, so the reserve must not
        # consult it or a stale MTP env over-reserves (Finding F3). Whitespace-
        # stripped so the check survives formatter line-wrapping.
        compact = "".join(inspect.getsource(LlamaCppBackend.load_model).split())
        assert '_mtp_canonical=="off"' in compact  # the env-reaches-child gate
        assert "_extra_args_requests_mtp(extra_args,env=_spec_env)" in compact

    def test_spec_default_overrides_env_mtp(self):
        # --spec-default is a CLI spec flag (resolves to the model default,
        # non-MTP) that overrides a stale LLAMA_ARG_SPEC_TYPE env, so the reserve
        # must not treat it as MTP (reviewer.py R4).
        env_mtp = {"LLAMA_ARG_SPEC_TYPE": "draft-mtp"}
        assert _extra_args_requests_mtp(["--spec-default"], env = env_mtp) is False
        assert (
            _extra_args_requests_separate_draft(
                ["--spec-default"], env = {"LLAMA_ARG_SPEC_TYPE": "draft-simple"}
            )
            is False
        )
        # A later --spec-type still wins over an earlier --spec-default.
        assert (
            _extra_args_requests_mtp(["--spec-default", "--spec-type", "draft-mtp"], env = {}) is True
        )

    def test_load_model_drafter_budget_precedence(self):
        # The budget sizes the drafter the launch actually loads: CLI extras win,
        # then Studio's emitted mtp_draft_path (overrides LLAMA_ARG_SPEC_DRAFT_MODEL),
        # then the env drafter -- not the env before Studio's (reviewer.py R3).
        compact = "".join(inspect.getsource(LlamaCppBackend.load_model).split())
        assert "_cli_draft_for_budget=_extra_args_mtp_draft_path(extra_args,env={})" in compact
        assert "_env_draft_for_budget=_extra_args_mtp_draft_path([],env=os.environ)" in compact
        assert "_cli_draft_for_budgetor_studio_draft_for_budgetor_env_draft_for_budget" in compact

    def test_load_model_drops_cpu_offloaded_drafter_from_budget(self):
        # A drafter offloaded to CPU (--spec-draft-ngl 0 / --spec-draft-device
        # none) consumes no GPU, so it must be dropped from the budget and get no
        # flat reserve either (Finding F2).
        compact = "".join(inspect.getsource(LlamaCppBackend.load_model).split())
        assert "_draft_on_cpu=_extra_args_draft_offloaded_to_cpu(extra_args)" in compact
        assert "if_draft_on_cpu:_mtp_draft_for_budget=None" in compact
        assert "andnot_draft_on_cpu" in compact  # excluded from the flat reserve

    @pytest.mark.parametrize(
        "args,expected",
        [
            (["--spec-draft-ngl", "0"], True),
            (["-ngld", "0"], True),
            (["--spec-draft-ngl=0"], True),
            (["--n-gpu-layers-draft", "0"], True),
            (["--spec-draft-ngl", "20"], False),
            (["--spec-draft-device", "none"], True),
            (["--spec-draft-device", "CPU"], True),
            (["-devd", "cpu,none"], True),
            (["--spec-draft-device", "CUDA0"], False),
            (["--spec-draft-device", "CUDA0,CPU"], False),  # any GPU -> on GPU
            (["-c", "4096"], False),
            (None, False),
        ],
    )
    def test_draft_offloaded_to_cpu(self, args, expected):
        assert _extra_args_draft_offloaded_to_cpu(args) is expected

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
        # env={} isolates pure-CLI behavior from a polluted test environment.
        assert _extra_args_mtp_draft_path(args, env = {}) == expected

    @pytest.mark.parametrize(
        "args,expected",
        [
            # HF draft repo flags are real llama-server flags; the budget must see them.
            (["--spec-draft-hf", "big/repo:Q8_0"], "big/repo:Q8_0"),
            (["-hfd", "big/repo"], "big/repo"),
            (["-hfrd", "big/repo"], "big/repo"),
            (["--hf-repo-draft=big/repo"], "big/repo"),
        ],
    )
    def test_mtp_draft_path_hf_flags(self, args, expected):
        assert _extra_args_mtp_draft_path(args, env = {}) == expected

    def test_mtp_draft_path_env_fallback(self):
        # The child honors LLAMA_ARG_SPEC_DRAFT_MODEL / _HF_REPO; CLI wins over env.
        assert (
            _extra_args_mtp_draft_path([], env = {"LLAMA_ARG_SPEC_DRAFT_MODEL": "/m/e.gguf"})
            == "/m/e.gguf"
        )
        assert _extra_args_mtp_draft_path([], env = {"LLAMA_ARG_SPEC_DRAFT_HF_REPO": "x/y"}) == "x/y"
        assert (
            _extra_args_mtp_draft_path(
                ["-md", "/m/cli.gguf"], env = {"LLAMA_ARG_SPEC_DRAFT_HF_REPO": "x/y"}
            )
            == "/m/cli.gguf"
        )

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
        assert _extra_args_draft_cache_types(args, env = {}) == expected

    def test_draft_cache_types_env_fallback(self):
        # The child honors LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K/_V per axis; CLI wins.
        assert _extra_args_draft_cache_types(
            [], env = {"LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K": "q8_0"}
        ) == ("q8_0", None)
        assert _extra_args_draft_cache_types(
            [],
            env = {
                "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K": "q8_0",
                "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V": "q4_0",
            },
        ) == ("q8_0", "q4_0")
        assert _extra_args_draft_cache_types(
            ["-ctkd", "q4_0"], env = {"LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K": "q8_0"}
        ) == ("q4_0", None)

    @pytest.mark.parametrize(
        "args,expected",
        [
            (["--ubatch-size", "1024"], 1024),
            (["-ub", "4096"], 4096),
            (["--ubatch-size=512"], 512),
            (["--ubatch", "2048"], None),  # not a real llama-server flag; ignore it
            (["-c", "4096"], None),
            (None, None),
        ],
    )
    def test_n_ubatch(self, args, expected):
        assert _extra_args_n_ubatch(args, env = {}) == expected

    def test_n_ubatch_env_fallback(self):
        # The child honors LLAMA_ARG_UBATCH; it must reach the compute-buffer reserve.
        assert _extra_args_n_ubatch([], env = {"LLAMA_ARG_UBATCH": "4096"}) == 4096
        assert (
            _extra_args_n_ubatch(["-ub", "1024"], env = {"LLAMA_ARG_UBATCH": "4096"}) == 1024
        )  # CLI wins
        assert _extra_args_n_ubatch([], env = {"LLAMA_ARG_UBATCH": "notint"}) is None


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


def test_mtp_draft_budget_prefers_user_extras_drafter():
    # A user --model-draft in extras is appended last and wins at launch, so the
    # VRAM budget must size it first; then Studio's emitted mtp_draft_path (which
    # overrides LLAMA_ARG_SPEC_DRAFT_MODEL), then the env drafter (load_model is too
    # entangled to drive end-to-end; assert the precedence at the source level).
    # Whitespace-stripped so the check survives any formatter line-wrapping.
    compact = "".join(inspect.getsource(LlamaCppBackend.load_model).split())
    # CLI extras sized first (env={} so the env doesn't pre-empt Studio's drafter).
    assert "_cli_draft_for_budget=_extra_args_mtp_draft_path(extra_args,env={})" in compact
    # Order: CLI extras, then Studio's mtp_draft_path, then the env drafter.
    assert "_cli_draft_for_budgetor_studio_draft_for_budgetor_env_draft_for_budget" in compact
    # The env must not be consulted before Studio's resolved drafter.
    assert "_extra_args_mtp_draft_path(extra_args)ormtp_draft_path" not in compact
