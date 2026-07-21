# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MTP draft reserve for MLA models keeps a duplicated target KV context.

llama.cpp's MTP speculative decoding allocates a second full copy of the target
model's KV context (``ctx_tgt=yes``) for draft verification, at f16. On MLA
models (GLM-5.x, DeepSeek, Kimi-K2) that copy is ~the main KV again and dwarfs
the tiny embedded draft head, so omitting it let auto-fit pick a context that
fit on paper but OOMed ``cublasCreate`` at the first decode (e.g. GLM-5.2
UD-IQ1_S advertised the native 1M context on 2x B200, then crashed on the first
generation). Non-MLA MTP (Qwen/Gemma) keeps no such copy and must stay exactly
as #6312 tuned it.
"""

import sys
import types as _types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Stub heavy/unavailable deps before importing the module under test, so this
# file is order-independent (importing core.inference pulls in orchestrator ->
# structlog, absent in the lightweight test env). Mirrors test_mtp_vram_budget.
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

sys.modules.setdefault("structlog", _types.ModuleType("structlog"))

# httpx -- only stub when the real library is missing. Unconditional stubbing
# shadows HTTPError/Response that huggingface_hub.errors imports at load time.
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
    LlamaCppBackend,
    _kv_bytes_per_elem,
)

GIB = 1024**3


def _make_mla_backend(
    *,
    n_layers = 79,
    n_kv_heads = 1,
    n_heads = 64,
    kv_key_length = 576,
    kv_value_length = 512,
    kv_lora_rank = 512,
    key_length_mla = 256,
    nextn = 1,
    embedding_length = 6144,
    vocab = 154880,
    native_ctx = 1048576,
):
    """GLM-5.2-class backend: MLA attention + an embedded MTP head."""
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._nextn_predict_layers = nextn
    b._n_kv_heads = n_kv_heads
    b._n_heads = n_heads
    b._kv_key_length = kv_key_length
    b._kv_value_length = kv_value_length
    b._embedding_length = embedding_length
    b._n_layers = n_layers
    b._context_length = native_ctx
    b._shared_kv_layers = 0
    b._kv_lora_rank = kv_lora_rank
    b._sliding_window = None
    b._sliding_window_pattern = None
    b._ssm_inner_size = None
    b._full_attention_interval = None
    b._key_length_mla = key_length_mla
    b._n_kv_heads_by_layer = None
    b._kv_key_length_swa = None
    b._kv_value_length_swa = None
    b._draft_backend_cache = None
    b._vocab_size = vocab
    return b


def _make_non_mla_backend(**kw):
    """Qwen3.6-MTP-class embedded head: no MLA (kv_lora_rank is None)."""
    b = _make_mla_backend(
        n_kv_heads = 4,
        n_heads = 24,
        kv_key_length = 256,
        kv_value_length = 256,
        embedding_length = 5120,
        n_layers = 65,
        native_ctx = 262144,
        **kw,
    )
    b._kv_lora_rank = None
    b._key_length_mla = None
    return b


class TestMlaTargetCtxReserve:
    def test_mla_reserve_includes_target_ctx_copy(self):
        b = _make_mla_backend()
        ctx = 1048576
        draft = b._mtp_draft_kv_bytes(ctx)
        overhead = b._estimate_mtp_overhead_bytes(ctx)
        main_kv_f16 = b._estimate_kv_cache_bytes(ctx, "f16")
        # Overhead = embedded draft head + a full f16 copy of the target KV.
        assert overhead == draft + main_kv_f16
        # The copy dominates: GLM-5.2 @1M is a ~2 GiB head next to a ~89 GiB copy.
        assert overhead / GIB > 80
        assert main_kv_f16 > 30 * draft

    def test_target_copy_is_f16_regardless_of_main_cache_type(self):
        # The MTP target context is always f16 in llama.cpp; the reserve must not
        # shrink when the user runs a quantized main KV.
        b = _make_mla_backend()
        ctx = 262144
        f16 = _kv_bytes_per_elem("f16")
        expected_copy = b._estimate_kv_cache_bytes(ctx, "f16")
        assert b._estimate_mtp_overhead_bytes(ctx) == (
            b._mtp_draft_kv_bytes(ctx) + expected_copy
        )
        assert f16 == 2.0  # sanity: f16 is 2 bytes/elem

    def test_target_copy_scales_linearly_with_context(self):
        b = _make_mla_backend()
        o_64k = b._estimate_mtp_overhead_bytes(65536)
        o_128k = b._estimate_mtp_overhead_bytes(131072)
        assert o_128k == pytest.approx(2 * o_64k)

    def test_non_mla_embedded_head_unchanged(self):
        # Qwen-class MTP keeps no target copy: overhead == draft KV exactly.
        b = _make_non_mla_backend()
        for ctx in (16384, 131072):
            assert b._estimate_mtp_overhead_bytes(ctx) == b._mtp_draft_kv_bytes(ctx)

    def test_mla_reserve_strictly_larger_than_non_mla_shape(self):
        # Same embedded-head dims, MLA toggled on/off: only MLA adds the copy.
        mla = _make_mla_backend()
        non = _make_mla_backend()
        non._kv_lora_rank = None  # flip MLA off, keep every other dim identical
        ctx = 131072
        assert mla._estimate_mtp_overhead_bytes(ctx) > non._estimate_mtp_overhead_bytes(
            ctx
        )

    def test_separate_drafter_mode_drops_target_copy(self):
        # The duplicated target context is MTP-only. draft-simple / draft-eagle3
        # load a small separate drafter with its own KV (counted in the draft KV)
        # and keep no target copy, so even on an MLA model the reserve must drop
        # the f16 copy when mtp_keeps_target_ctx=False -- which is what the loader
        # threads for those modes. The default (True) keeps the MTP copy.
        b = _make_mla_backend()
        ctx = 262144
        mtp = b._estimate_mtp_overhead_bytes(ctx)  # default True == MTP draft
        separate = b._estimate_mtp_overhead_bytes(ctx, mtp_keeps_target_ctx = False)
        # Separate-drafter overhead is exactly the draft KV (no target copy)...
        assert separate == b._mtp_draft_kv_bytes(ctx)
        # ...and the MTP reserve is that plus the full f16 target copy.
        assert mtp == separate + b._estimate_kv_cache_bytes(ctx, "f16")
        assert mtp > separate


class TestMlaFitPreventsOom:
    """The corrected reserve must actually lower the auto-fit context so the
    config holds at runtime instead of OOMing on the first decode."""

    # 2x B200, mirroring the GLM-5.2 UD-IQ1_S crash (only 2 GPUs were selected).
    AVAIL_MIB = 2 * 182010
    TOTAL_MIB = 2 * 182633
    MODEL_BYTES = 200 * GIB  # ~UD-IQ1_S weight footprint
    REQ_CTX = 1048576

    def test_target_copy_lowers_chosen_context(self):
        b = _make_mla_backend()
        with_copy = b._fit_context_to_vram(
            self.REQ_CTX,
            self.AVAIL_MIB,
            self.MODEL_BYTES,
            mtp_engaged = True,
            total_mib = self.TOTAL_MIB,
            mtp_overhead_fn = lambda c: b._estimate_mtp_overhead_bytes(c) or 0,
        )
        # The old behaviour (draft head only, no target copy) kept the full ctx.
        draft_only = b._fit_context_to_vram(
            self.REQ_CTX,
            self.AVAIL_MIB,
            self.MODEL_BYTES,
            mtp_engaged = True,
            total_mib = self.TOTAL_MIB,
            mtp_overhead_fn = lambda c: b._mtp_draft_kv_bytes(c) or 0,
        )
        assert draft_only == self.REQ_CTX  # reproduces the over-advertised context
        assert with_copy < self.REQ_CTX  # corrected reserve backs the context off
