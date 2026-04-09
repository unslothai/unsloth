# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for 5-path architecture-aware KV cache VRAM estimation.

Covers the GGUF metadata parser, _can_estimate_kv gate, all 5 estimation
paths (MLA, Hybrid Mamba, Sliding Window, Standard GQA, Legacy), KV cache
quantization, edge cases, and lifecycle (init/unload/reparse).

Requires no GPU, network, or external libraries beyond pytest.
Cross-platform: Linux, macOS, Windows, WSL.
"""

import io
import struct
import sys
import types as _types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Stub heavy / unavailable external dependencies before importing the
# module under test.  Same pattern as test_native_context_length.py.
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# loggers
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

# structlog
_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)

# httpx
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


class _FakeTimeout:
    def __init__(self, *a, **kw):
        pass


_httpx_stub.Timeout = _FakeTimeout
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

from core.inference.llama_cpp import LlamaCppBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gguf_bytes(arch: str, kv_pairs: dict) -> bytes:
    """Build a minimal GGUF v3 binary blob with the given KV metadata.

    Only supports UINT32 (type 4), UINT64 (type 10), and STRING (type 8)
    values, which is all the metadata parser reads.
    """
    buf = io.BytesIO()
    # Header: magic, version, tensor_count, kv_count
    buf.write(struct.pack("<I", 0x46554747))  # GGUF magic
    buf.write(struct.pack("<I", 3))  # version 3
    buf.write(struct.pack("<Q", 0))  # tensor_count
    buf.write(struct.pack("<Q", len(kv_pairs)))

    for key, val in kv_pairs.items():
        key_bytes = key.encode("utf-8")
        buf.write(struct.pack("<Q", len(key_bytes)))
        buf.write(key_bytes)
        if isinstance(val, str):
            buf.write(struct.pack("<I", 8))  # STRING
            val_bytes = val.encode("utf-8")
            buf.write(struct.pack("<Q", len(val_bytes)))
            buf.write(val_bytes)
        elif isinstance(val, int):
            if val <= 0xFFFFFFFF:
                buf.write(struct.pack("<I", 4))  # UINT32
                buf.write(struct.pack("<I", val))
            else:
                buf.write(struct.pack("<I", 10))  # UINT64
                buf.write(struct.pack("<Q", val))
        else:
            raise TypeError(f"Unsupported value type: {type(val)}")
    return buf.getvalue()


def _backend_from_gguf(arch: str, fields: dict) -> LlamaCppBackend:
    """Create a LlamaCppBackend with parsed GGUF metadata from given fields."""
    kv = {"general.architecture": arch}
    for k, v in fields.items():
        kv[f"{arch}.{k}"] = v
    import tempfile, os

    data = _make_gguf_bytes(arch, kv)
    fd, path = tempfile.mkstemp(suffix = ".gguf")
    try:
        os.write(fd, data)
        os.close(fd)
        b = LlamaCppBackend()
        b._read_gguf_metadata(path)
        return b
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# A. GGUF Parser Tests
# ---------------------------------------------------------------------------


class TestGGUFParserNewFields:
    """Verify that the 8 new architecture-aware fields are correctly parsed."""

    @pytest.mark.parametrize(
        "field,gguf_key,value",
        [
            ("_kv_key_length", "attention.key_length", 128),
            ("_kv_value_length", "attention.value_length", 128),
            ("_sliding_window", "attention.sliding_window", 1024),
            ("_full_attention_interval", "full_attention_interval", 4),
            ("_kv_lora_rank", "attention.kv_lora_rank", 512),
            ("_key_length_mla", "attention.key_length_mla", 256),
            ("_ssm_inner_size", "ssm.inner_size", 6144),
            ("_ssm_state_size", "ssm.state_size", 128),
        ],
    )
    def test_field_parsed(self, field, gguf_key, value):
        b = _backend_from_gguf("testarch", {gguf_key: value})
        assert getattr(b, field) == value

    def test_missing_fields_are_none(self):
        b = _backend_from_gguf("testarch", {"block_count": 10})
        for attr in [
            "_kv_key_length",
            "_kv_value_length",
            "_sliding_window",
            "_full_attention_interval",
            "_kv_lora_rank",
            "_key_length_mla",
            "_ssm_inner_size",
            "_ssm_state_size",
        ]:
            assert getattr(b, attr) is None

    def test_all_13_fields_parsed_together(self):
        fields = {
            "context_length": 131072,
            "block_count": 62,
            "attention.head_count_kv": 16,
            "attention.head_count": 32,
            "embedding_length": 5376,
            "attention.key_length": 128,
            "attention.value_length": 128,
            "attention.sliding_window": 1024,
            "full_attention_interval": 6,
            "attention.kv_lora_rank": 512,
            "attention.key_length_mla": 256,
            "ssm.inner_size": 4096,
            "ssm.state_size": 128,
        }
        b = _backend_from_gguf("testarch", fields)
        assert b._context_length == 131072
        assert b._n_layers == 62
        assert b._n_kv_heads == 16
        assert b._n_heads == 32
        assert b._embedding_length == 5376
        assert b._kv_key_length == 128
        assert b._kv_value_length == 128
        assert b._sliding_window == 1024
        assert b._full_attention_interval == 6
        assert b._kv_lora_rank == 512
        assert b._key_length_mla == 256
        assert b._ssm_inner_size == 4096
        assert b._ssm_state_size == 128


class TestGGUFParserReset:
    """Verify that fields are properly reset between parses."""

    def test_reset_between_parses(self):
        # First parse with all fields
        b = _backend_from_gguf(
            "arch1",
            {
                "block_count": 32,
                "attention.key_length": 128,
                "attention.kv_lora_rank": 512,
                "ssm.inner_size": 4096,
            },
        )
        assert b._kv_key_length == 128
        assert b._kv_lora_rank == 512
        assert b._ssm_inner_size == 4096

        # Second parse without those fields -- they should be None
        kv = {"general.architecture": "arch2", "arch2.block_count": 64}
        import tempfile, os

        data = _make_gguf_bytes("arch2", kv)
        fd, path = tempfile.mkstemp(suffix = ".gguf")
        os.write(fd, data)
        os.close(fd)
        try:
            b._read_gguf_metadata(path)
        finally:
            os.unlink(path)
        assert b._kv_key_length is None
        assert b._kv_lora_rank is None
        assert b._ssm_inner_size is None
        assert b._n_layers == 64


# ---------------------------------------------------------------------------
# B. _can_estimate_kv Gate Tests
# ---------------------------------------------------------------------------


class TestCanEstimateKV:
    """Verify gate logic for all field combinations."""

    def test_no_layers_returns_false(self):
        b = LlamaCppBackend()
        b._n_layers = None
        b._kv_key_length = 128
        assert not b._can_estimate_kv()

    def test_explicit_both_dims_sufficient(self):
        b = LlamaCppBackend()
        b._n_layers = 32
        b._kv_key_length = 128
        b._kv_value_length = 128
        assert b._can_estimate_kv()

    def test_key_length_alone_insufficient(self):
        """key_length without value_length should NOT be enough."""
        b = LlamaCppBackend()
        b._n_layers = 32
        b._kv_key_length = 128
        assert not b._can_estimate_kv()

    def test_kv_lora_rank_sufficient(self):
        b = LlamaCppBackend()
        b._n_layers = 61
        b._kv_lora_rank = 512
        assert b._can_estimate_kv()

    def test_legacy_embed_plus_heads(self):
        b = LlamaCppBackend()
        b._n_layers = 28
        b._embedding_length = 1024
        b._n_heads = 16
        assert b._can_estimate_kv()

    def test_legacy_embed_plus_kv_heads(self):
        b = LlamaCppBackend()
        b._n_layers = 28
        b._embedding_length = 1024
        b._n_kv_heads = 8
        assert b._can_estimate_kv()

    def test_legacy_no_embed_returns_false(self):
        b = LlamaCppBackend()
        b._n_layers = 28
        b._n_heads = 16
        # No embedding_length, no new-style fields
        assert not b._can_estimate_kv()

    def test_fresh_backend_returns_false(self):
        b = LlamaCppBackend()
        assert not b._can_estimate_kv()


# ---------------------------------------------------------------------------
# C. Path 1: MLA Estimation
# ---------------------------------------------------------------------------


class TestMLAEstimation:
    """MLA: K-only cache using compressed KV latent + RoPE."""

    def _mla_backend(self, **overrides):
        defaults = {
            "_n_layers": 61,
            "_n_kv_heads": 1,
            "_n_heads": 128,
            "_embedding_length": 7168,
            "_kv_key_length": 576,
            "_kv_value_length": 512,
            "_kv_lora_rank": 512,
            "_key_length_mla": 192,
        }
        defaults.update(overrides)
        b = LlamaCppBackend()
        for k, v in defaults.items():
            setattr(b, k, v)
        return b

    def test_deepseek_v3_f16(self):
        b = self._mla_backend()
        # 61 layers * 163840 ctx * 1 head * 576 key_len * 2 bpe
        expected = 61 * 163840 * 1 * 576 * 2
        assert b._estimate_kv_cache_bytes(163840, "f16") == expected

    def test_mla_ignores_value_length(self):
        """MLA should NOT add value_length -- V is reconstructed from the latent."""
        b = self._mla_backend()
        result = b._estimate_kv_cache_bytes(1000, "f16")
        # Should be n_layers * ctx * 1 * key_len(576) * 2
        expected = 61 * 1000 * 1 * 576 * 2
        assert result == expected

    def test_mla_fallback_when_no_key_length(self):
        """If key_length is missing, fallback to kv_lora_rank + key_length_mla."""
        b = self._mla_backend(_kv_key_length = None)
        # _key_length_mla=192 in default, so rope_dim=192
        result = b._estimate_kv_cache_bytes(1000, "f16")
        expected = 61 * 1000 * 1 * (512 + 192) * 2  # 704
        assert result == expected

    def test_mla_fallback_no_key_length_mla(self):
        """If both key_length and key_length_mla are missing, fallback to +64."""
        b = self._mla_backend(_kv_key_length = None, _key_length_mla = None)
        result = b._estimate_kv_cache_bytes(1000, "f16")
        expected = 61 * 1000 * 1 * (512 + 64) * 2  # 576
        assert result == expected

    def test_mla_defaults_n_kv_to_1_when_heads_absent(self):
        """MLA should use n_kv=1 even if n_kv_heads is None (not n_heads)."""
        b = self._mla_backend(_n_kv_heads = None)  # n_heads=128 still set
        result = b._estimate_kv_cache_bytes(1000, "f16")
        # Should use n_kv_mla=1, NOT n_heads=128
        expected = 61 * 1000 * 1 * 576 * 2
        assert result == expected

    def test_mla_q4_quantization(self):
        b = self._mla_backend()
        result_f16 = b._estimate_kv_cache_bytes(1000, "f16")
        result_q4 = b._estimate_kv_cache_bytes(1000, "q4_0")
        assert result_q4 < result_f16
        # q4_0 bpe = 0.5625, f16 bpe = 2.0
        assert result_q4 == int(61 * 1000 * 1 * 576 * 0.5625)


# ---------------------------------------------------------------------------
# D. Path 2: Hybrid Mamba Estimation
# ---------------------------------------------------------------------------


class TestHybridMambaEstimation:
    """Hybrid Mamba: only attention layers (1 in N) need KV cache."""

    def _hybrid_backend(self, **overrides):
        defaults = {
            "_n_layers": 64,
            "_n_kv_heads": 4,
            "_n_heads": 24,
            "_embedding_length": 5120,
            "_kv_key_length": 256,
            "_kv_value_length": 256,
            "_full_attention_interval": 4,
            "_ssm_inner_size": 6144,
            "_ssm_state_size": 128,
        }
        defaults.update(overrides)
        b = LlamaCppBackend()
        for k, v in defaults.items():
            setattr(b, k, v)
        return b

    def test_qwen35_27b(self):
        b = self._hybrid_backend()
        # n_attn = 64 // 4 = 16
        expected = 16 * 262144 * 4 * (256 + 256) * 2
        assert b._estimate_kv_cache_bytes(262144, "f16") == expected

    def test_qwen35_35b_a3b(self):
        b = self._hybrid_backend(
            _n_layers = 40,
            _n_kv_heads = 2,
            _n_heads = 16,
            _embedding_length = 2048,
            _ssm_inner_size = 4096,
        )
        # n_attn = 40 // 4 = 10
        expected = 10 * 262144 * 2 * (256 + 256) * 2
        assert b._estimate_kv_cache_bytes(262144, "f16") == expected

    def test_hybrid_without_explicit_dims(self):
        """Fallback to head_dim when key_length/value_length are missing."""
        b = self._hybrid_backend(_kv_key_length = None, _kv_value_length = None)
        head_dim = 5120 // 24  # 213
        expected = 16 * 4096 * 4 * 2 * head_dim * 2
        assert b._estimate_kv_cache_bytes(4096, "f16") == expected

    def test_fai_zero_safety(self):
        """full_attention_interval=0 should not cause ZeroDivisionError."""
        b = self._hybrid_backend(_full_attention_interval = 0)
        result = b._estimate_kv_cache_bytes(4096, "f16")
        # fai=0 -> n_attn = n_layers (all layers)
        expected = 64 * 4096 * 4 * (256 + 256) * 2
        assert result == expected


# ---------------------------------------------------------------------------
# E. Path 3: Sliding Window Estimation
# ---------------------------------------------------------------------------


class TestSlidingWindowEstimation:
    """SWA: half global (full ctx) + half sliding window."""

    def _swa_backend(self, **overrides):
        defaults = {
            "_n_layers": 62,
            "_n_kv_heads": 16,
            "_n_heads": 32,
            "_embedding_length": 5376,
            "_kv_key_length": 128,
            "_kv_value_length": 128,
            "_sliding_window": 1024,
        }
        defaults.update(overrides)
        b = LlamaCppBackend()
        for k, v in defaults.items():
            setattr(b, k, v)
        return b

    def test_gemma3(self):
        b = self._swa_backend()
        # 1/4 heuristic: 62 // 4 = 15 global, 47 SWA
        n_global = max(1, 62 // 4)  # 15
        n_swa = 62 - n_global  # 47
        kv_per = 16 * (128 + 128) * 2
        expected = int(n_global * 131072 * kv_per + n_swa * min(131072, 1024) * kv_per)
        assert b._estimate_kv_cache_bytes(131072, "f16") == expected

    def test_gpt_oss(self):
        b = self._swa_backend(
            _n_layers = 24,
            _n_kv_heads = 8,
            _n_heads = 64,
            _embedding_length = 2880,
            _kv_key_length = 64,
            _kv_value_length = 64,
            _sliding_window = 128,
        )
        # 1/4 heuristic: 24 // 4 = 6 global, 18 SWA
        n_global = max(1, 24 // 4)  # 6
        n_swa = 24 - n_global  # 18
        kv_per = 8 * (64 + 64) * 2
        expected = int(n_global * 131072 * kv_per + n_swa * min(131072, 128) * kv_per)
        assert b._estimate_kv_cache_bytes(131072, "f16") == expected

    def test_ctx_smaller_than_window(self):
        """When context < sliding_window, SWA layers use full context anyway."""
        b = self._swa_backend(_sliding_window = 8192)
        n_global = max(1, 62 // 4)  # 15
        n_swa = 62 - n_global  # 47
        kv_per = 16 * (128 + 128) * 2
        ctx = 4096
        expected = int(n_global * ctx * kv_per + n_swa * min(ctx, 8192) * kv_per)
        # min(4096, 8192) = 4096, so both pools use full ctx
        assert b._estimate_kv_cache_bytes(ctx, "f16") == expected

    def test_odd_layer_count(self):
        """Odd layer count: n_global = max(1, n//4), n_swa = n - n_global."""
        b = self._swa_backend(_n_layers = 63)
        n_global = max(1, 63 // 4)  # 15
        n_swa = 63 - n_global  # 48
        kv_per = 16 * (128 + 128) * 2
        expected = int(n_global * 1000 * kv_per + n_swa * min(1000, 1024) * kv_per)
        assert b._estimate_kv_cache_bytes(1000, "f16") == expected


# ---------------------------------------------------------------------------
# F. Path 4: Standard GQA Estimation
# ---------------------------------------------------------------------------


class TestStandardGQAEstimation:
    """Standard GQA with explicit key_length/value_length."""

    def _gqa_backend(self, **overrides):
        defaults = {
            "_n_layers": 28,
            "_n_kv_heads": 8,
            "_n_heads": 16,
            "_embedding_length": 1024,
            "_kv_key_length": 128,
            "_kv_value_length": 128,
        }
        defaults.update(overrides)
        b = LlamaCppBackend()
        for k, v in defaults.items():
            setattr(b, k, v)
        return b

    def test_qwen3_06b(self):
        b = self._gqa_backend()
        expected = 28 * 40960 * 8 * (128 + 128) * 2
        assert b._estimate_kv_cache_bytes(40960, "f16") == expected

    def test_asymmetric_kv_dims(self):
        """key_length != value_length (some architectures have this)."""
        b = self._gqa_backend(_kv_key_length = 192, _kv_value_length = 64)
        expected = 28 * 4096 * 8 * (192 + 64) * 2
        assert b._estimate_kv_cache_bytes(4096, "f16") == expected

    def test_differs_from_legacy(self):
        """GQA path should differ from legacy when key_length != embed//n_heads."""
        b = self._gqa_backend()
        head_dim = 1024 // 16  # 64
        gqa_result = b._estimate_kv_cache_bytes(4096, "f16")
        # Legacy would use: 2 * 8 * 64 * 28 * 4096 * 2
        legacy_result = int(2 * 8 * head_dim * 28 * 4096 * 2)
        # GQA: 28 * 4096 * 8 * (128+128) * 2 -- uses actual key_length=128
        assert gqa_result != legacy_result
        assert gqa_result > legacy_result  # key_length (128) > head_dim (64)


# ---------------------------------------------------------------------------
# G. Path 5: Legacy Fallback Estimation
# ---------------------------------------------------------------------------


class TestLegacyEstimation:
    """Legacy: embed // n_heads, for old GGUFs without new fields."""

    def _legacy_backend(self, **overrides):
        defaults = {
            "_n_layers": 32,
            "_n_kv_heads": 8,
            "_n_heads": 32,
            "_embedding_length": 4096,
        }
        defaults.update(overrides)
        b = LlamaCppBackend()
        for k, v in defaults.items():
            setattr(b, k, v)
        return b

    def test_basic_legacy(self):
        b = self._legacy_backend()
        head_dim = 4096 // 32  # 128
        expected = int(2 * 8 * 128 * 32 * 4096 * 2)
        assert b._estimate_kv_cache_bytes(4096, "f16") == expected

    def test_legacy_with_only_n_heads(self):
        """n_kv_heads is None, falls back to n_heads."""
        b = self._legacy_backend(_n_kv_heads = None)
        head_dim = 4096 // 32
        expected = int(2 * 32 * head_dim * 32 * 4096 * 2)
        assert b._estimate_kv_cache_bytes(4096, "f16") == expected

    def test_legacy_identical_to_old_formula(self):
        """Confirm legacy path produces the same result as the pre-PR formula."""
        b = self._legacy_backend()
        n_layers = 32
        n_kv_heads = 8
        head_dim = 4096 // 32
        n_ctx = 8192
        bpe = 2.0
        old_formula = int(2 * n_kv_heads * head_dim * n_layers * n_ctx * bpe)
        assert b._estimate_kv_cache_bytes(n_ctx, "f16") == old_formula


# ---------------------------------------------------------------------------
# H. Path Priority (selection order)
# ---------------------------------------------------------------------------


class TestPathPriority:
    """Confirm: MLA > Hybrid Mamba > SWA > GQA > Legacy."""

    def test_mla_takes_priority_over_all(self):
        """If kv_lora_rank is set, MLA path is used even if other fields are present."""
        b = LlamaCppBackend()
        b._n_layers = 61
        b._n_kv_heads = 1
        b._n_heads = 128
        b._embedding_length = 7168
        b._kv_key_length = 576
        b._kv_value_length = 512
        b._kv_lora_rank = 512
        b._ssm_inner_size = 4096  # Would trigger Hybrid
        b._full_attention_interval = 4
        b._sliding_window = 1024  # Would trigger SWA

        # MLA: 61 * 1000 * 1 * 576 * 2
        expected_mla = int(61 * 1000 * 1 * 576 * 2)
        assert b._estimate_kv_cache_bytes(1000, "f16") == expected_mla

    def test_hybrid_over_swa(self):
        """Hybrid takes priority over SWA when both fields present."""
        b = LlamaCppBackend()
        b._n_layers = 64
        b._n_kv_heads = 4
        b._n_heads = 24
        b._embedding_length = 5120
        b._kv_key_length = 256
        b._kv_value_length = 256
        b._ssm_inner_size = 6144
        b._full_attention_interval = 4
        b._sliding_window = 1024  # Would trigger SWA

        n_attn = 64 // 4
        expected_hybrid = int(n_attn * 1000 * 4 * (256 + 256) * 2)
        assert b._estimate_kv_cache_bytes(1000, "f16") == expected_hybrid

    def test_all_paths_produce_different_values(self):
        """With carefully chosen params, each path should yield a distinct value."""
        # Use embedding_length=768 so legacy head_dim (768//16=48) differs from
        # key_length (256), and MLA key_len (256) != legacy K+V (2*48=96).
        params = {
            "_n_layers": 40,
            "_n_kv_heads": 4,
            "_n_heads": 16,
            "_embedding_length": 768,
            "_kv_key_length": 256,
            "_kv_value_length": 256,
        }
        ctx = 4096

        # Path 4: Standard GQA
        b_gqa = LlamaCppBackend()
        for k, v in params.items():
            setattr(b_gqa, k, v)
        gqa_val = b_gqa._estimate_kv_cache_bytes(ctx, "f16")

        # Path 1: MLA
        b_mla = LlamaCppBackend()
        for k, v in params.items():
            setattr(b_mla, k, v)
        b_mla._kv_lora_rank = 512
        mla_val = b_mla._estimate_kv_cache_bytes(ctx, "f16")

        # Path 2: Hybrid Mamba
        b_hybrid = LlamaCppBackend()
        for k, v in params.items():
            setattr(b_hybrid, k, v)
        b_hybrid._ssm_inner_size = 4096
        b_hybrid._full_attention_interval = 4
        hybrid_val = b_hybrid._estimate_kv_cache_bytes(ctx, "f16")

        # Path 3: SWA
        b_swa = LlamaCppBackend()
        for k, v in params.items():
            setattr(b_swa, k, v)
        b_swa._sliding_window = 512
        swa_val = b_swa._estimate_kv_cache_bytes(ctx, "f16")

        # Path 5: Legacy (no key_length/value_length)
        b_legacy = LlamaCppBackend()
        b_legacy._n_layers = 40
        b_legacy._n_kv_heads = 4
        b_legacy._n_heads = 16
        b_legacy._embedding_length = 768
        legacy_val = b_legacy._estimate_kv_cache_bytes(ctx, "f16")

        values = [mla_val, hybrid_val, swa_val, gqa_val, legacy_val]
        assert len(set(values)) == 5, f"Expected 5 distinct values, got {values}"


# ---------------------------------------------------------------------------
# I. KV Cache Quantization
# ---------------------------------------------------------------------------


class TestQuantization:
    """Verify all supported cache_type_kv values produce correct scaling."""

    @pytest.mark.parametrize(
        "cache_type,expected_bpe",
        [
            ("f32", 4.0),
            ("f16", 2.0),
            ("bf16", 2.0),
            ("q8_0", 34 / 32),
            ("q5_1", 0.75),
            ("q5_0", 0.6875),
            ("q4_1", 0.625),
            ("q4_0", 0.5625),
            ("iq4_nl", 0.5625),
            (None, 2.0),  # default is f16
            ("unknown", 2.0),  # unknown falls back to f16
        ],
    )
    def test_quantization_scaling(self, cache_type, expected_bpe):
        b = LlamaCppBackend()
        b._n_layers = 10
        b._n_kv_heads = 1
        b._n_heads = 8
        b._embedding_length = 512
        b._kv_key_length = 64
        b._kv_value_length = 64
        result = b._estimate_kv_cache_bytes(1000, cache_type)
        expected = int(10 * 1000 * 1 * (64 + 64) * expected_bpe)
        assert result == expected


# ---------------------------------------------------------------------------
# J. Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions and degenerate inputs."""

    def test_zero_context(self):
        b = LlamaCppBackend()
        b._n_layers = 32
        b._kv_key_length = 128
        assert b._estimate_kv_cache_bytes(0, "f16") == 0

    def test_negative_context(self):
        b = LlamaCppBackend()
        b._n_layers = 32
        b._kv_key_length = 128
        assert b._estimate_kv_cache_bytes(-1, "f16") == 0

    def test_context_of_one(self):
        b = LlamaCppBackend()
        b._n_layers = 10
        b._n_kv_heads = 1
        b._kv_key_length = 64
        b._kv_value_length = 64
        result = b._estimate_kv_cache_bytes(1, "f16")
        assert result == int(10 * 1 * 1 * (64 + 64) * 2)

    def test_very_large_context(self):
        """1M context should not overflow or crash."""
        b = LlamaCppBackend()
        b._n_layers = 10
        b._n_kv_heads = 1
        b._kv_key_length = 128
        b._kv_value_length = 128
        result = b._estimate_kv_cache_bytes(1_000_000, "f16")
        assert result > 0
        assert isinstance(result, int)

    def test_n_kv_heads_none_falls_to_n_heads(self):
        b = LlamaCppBackend()
        b._n_layers = 10
        b._n_kv_heads = None
        b._n_heads = 8
        b._kv_key_length = 64
        b._kv_value_length = 64
        result = b._estimate_kv_cache_bytes(100, "f16")
        expected = int(10 * 100 * 8 * (64 + 64) * 2)
        assert result == expected

    def test_both_heads_none_falls_to_one(self):
        b = LlamaCppBackend()
        b._n_layers = 10
        b._n_kv_heads = None
        b._n_heads = None
        b._kv_key_length = 64
        b._kv_value_length = 64
        result = b._estimate_kv_cache_bytes(100, "f16")
        expected = int(10 * 100 * 1 * (64 + 64) * 2)
        assert result == expected


# ---------------------------------------------------------------------------
# K. Lifecycle Tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Init, unload, and reparse field management."""

    def test_init_fields_none(self):
        b = LlamaCppBackend()
        for attr in [
            "_kv_key_length",
            "_kv_value_length",
            "_sliding_window",
            "_full_attention_interval",
            "_kv_lora_rank",
            "_key_length_mla",
            "_ssm_inner_size",
            "_ssm_state_size",
        ]:
            assert getattr(b, attr) is None

    def test_unload_resets_fields(self):
        b = LlamaCppBackend()
        b._n_layers = 32
        b._kv_key_length = 128
        b._kv_lora_rank = 512
        b._sliding_window = 1024
        b._ssm_inner_size = 4096
        b._full_attention_interval = 4
        b.unload_model()
        for attr in [
            "_kv_key_length",
            "_kv_value_length",
            "_sliding_window",
            "_full_attention_interval",
            "_kv_lora_rank",
            "_key_length_mla",
            "_ssm_inner_size",
            "_ssm_state_size",
        ]:
            assert getattr(b, attr) is None

    def test_end_to_end_synthetic_mla(self):
        """Full round-trip: write GGUF -> parse -> estimate."""
        b = _backend_from_gguf(
            "deepseek2",
            {
                "context_length": 163840,
                "block_count": 61,
                "attention.head_count_kv": 1,
                "attention.head_count": 128,
                "embedding_length": 7168,
                "attention.key_length": 576,
                "attention.value_length": 512,
                "attention.kv_lora_rank": 512,
                "attention.key_length_mla": 192,
            },
        )
        assert b._can_estimate_kv()
        result = b._estimate_kv_cache_bytes(163840, "f16")
        expected = 61 * 163840 * 1 * 576 * 2
        assert result == expected

    def test_end_to_end_synthetic_hybrid(self):
        b = _backend_from_gguf(
            "qwen35",
            {
                "context_length": 262144,
                "block_count": 64,
                "attention.head_count_kv": 4,
                "attention.head_count": 24,
                "embedding_length": 5120,
                "attention.key_length": 256,
                "attention.value_length": 256,
                "full_attention_interval": 4,
                "ssm.inner_size": 6144,
                "ssm.state_size": 128,
            },
        )
        assert b._can_estimate_kv()
        result = b._estimate_kv_cache_bytes(262144, "f16")
        n_attn = 64 // 4
        expected = n_attn * 262144 * 4 * (256 + 256) * 2
        assert result == expected

    def test_end_to_end_synthetic_swa(self):
        b = _backend_from_gguf(
            "gemma3",
            {
                "context_length": 131072,
                "block_count": 62,
                "attention.head_count_kv": 16,
                "attention.head_count": 32,
                "embedding_length": 5376,
                "attention.key_length": 128,
                "attention.value_length": 128,
                "attention.sliding_window": 1024,
            },
        )
        assert b._can_estimate_kv()
        result = b._estimate_kv_cache_bytes(131072, "f16")
        n_global = max(1, 62 // 4)  # 15
        n_swa = 62 - n_global  # 47
        kv_per = 16 * 256 * 2
        expected = int(n_global * 131072 * kv_per + n_swa * 1024 * kv_per)
        assert result == expected

    def test_end_to_end_synthetic_gqa(self):
        b = _backend_from_gguf(
            "qwen3",
            {
                "context_length": 40960,
                "block_count": 28,
                "attention.head_count_kv": 8,
                "attention.head_count": 16,
                "embedding_length": 1024,
                "attention.key_length": 128,
                "attention.value_length": 128,
            },
        )
        assert b._can_estimate_kv()
        result = b._estimate_kv_cache_bytes(40960, "f16")
        expected = 28 * 40960 * 8 * 256 * 2
        assert result == expected

    def test_end_to_end_synthetic_legacy(self):
        b = _backend_from_gguf(
            "llama",
            {
                "context_length": 4096,
                "block_count": 32,
                "attention.head_count_kv": 8,
                "attention.head_count": 32,
                "embedding_length": 4096,
            },
        )
        assert b._can_estimate_kv()
        result = b._estimate_kv_cache_bytes(4096, "f16")
        head_dim = 4096 // 32
        expected = int(2 * 8 * head_dim * 32 * 4096 * 2)
        assert result == expected
