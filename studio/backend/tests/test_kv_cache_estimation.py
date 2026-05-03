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
import json
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

    Supports the scalar and simple array metadata used by the parser.
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
        elif isinstance(val, list):
            buf.write(struct.pack("<I", 9))  # ARRAY
            is_bool_array = all(isinstance(x, bool) for x in val)
            buf.write(struct.pack("<I", 7 if is_bool_array else 5))
            buf.write(struct.pack("<Q", len(val)))
            if is_bool_array:
                for item in val:
                    buf.write(struct.pack("<?", item))
            else:
                for item in val:
                    buf.write(struct.pack("<i", item))
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


def _backend_from_gguf(
    arch: str, fields: dict, general: dict | None = None
) -> LlamaCppBackend:
    """Create a LlamaCppBackend with parsed GGUF metadata from given fields.

    `general` lets a test inject extra `general.*` metadata (used to
    verify the dynamic SWA resolver picks up source-repo hints from
    GGUFs that ship them).
    """
    kv = {"general.architecture": arch}
    for k, v in (general or {}).items():
        kv[k] = v
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
    """Verify that architecture-aware fields are correctly parsed."""

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
            "_sliding_window_pattern",
            "_full_attention_interval",
            "_kv_lora_rank",
            "_key_length_mla",
            "_kv_key_length_swa",
            "_kv_value_length_swa",
            "_ssm_inner_size",
            "_ssm_state_size",
        ]:
            assert getattr(b, attr) is None

    def test_array_fields_parsed(self):
        b = _backend_from_gguf(
            "gemma4",
            {
                "block_count": 6,
                "attention.head_count_kv": [8, 8, 8, 8, 8, 2],
                "attention.sliding_window_pattern": [
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            },
        )
        # Per-layer KV head count is preserved exactly...
        assert b._n_kv_heads_by_layer == [8, 8, 8, 8, 8, 2]
        # ...and mirrored into the scalar field as a conservative max so
        # non-SWA estimator paths and any caller using
        # `n_kv = self._n_kv_heads or ...` get a safe upper bound.
        assert b._n_kv_heads == 8
        assert b._sliding_window_pattern == [True, True, True, True, True, False]


class TestArchSwaPatternDefaults:
    """Bootstrap arch table fires when GGUF reports `sliding_window` but
    no per-layer pattern (true for every Gemma 2/3/3n/gpt-oss GGUF today)."""

    @pytest.mark.parametrize(
        "arch,n_layers,expected_period",
        [
            ("gemma2", 26, 2),
            ("gemma3", 18, 6),
            ("gemma3n", 35, 5),
            ("gpt_oss", 24, 2),
            ("cohere2", 32, 4),
        ],
    )
    def test_arch_default_pattern_applied(self, arch, n_layers, expected_period):
        b = _backend_from_gguf(
            arch,
            {
                "block_count": n_layers,
                "attention.head_count": 4,
                "attention.head_count_kv": 1,
                "attention.key_length": 256,
                "attention.value_length": 256,
                "attention.sliding_window": 512,
            },
        )
        expected_pattern = [(i + 1) % expected_period != 0 for i in range(n_layers)]
        assert (
            b._sliding_window_pattern == expected_pattern
        ), f"{arch} should expand to period={expected_period}"

    def test_unknown_arch_no_default(self):
        b = _backend_from_gguf(
            "totallymadeupv7",
            {
                "block_count": 24,
                "attention.head_count": 4,
                "attention.head_count_kv": 1,
                "attention.key_length": 128,
                "attention.value_length": 128,
                "attention.sliding_window": 1024,
            },
        )
        assert b._sliding_window_pattern is None

    def test_explicit_pattern_overrides_arch_default(self):
        # Period=6 is the gemma3 default; the explicit array must win.
        b = _backend_from_gguf(
            "gemma3",
            {
                "block_count": 6,
                "attention.head_count": 4,
                "attention.head_count_kv": 1,
                "attention.key_length": 256,
                "attention.value_length": 256,
                "attention.sliding_window": 512,
                "attention.sliding_window_pattern": [
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                ],
            },
        )
        assert b._sliding_window_pattern == [True, False, True, False, True, False]

    def test_no_sliding_window_no_pattern(self):
        b = _backend_from_gguf(
            "gemma3",
            {
                "block_count": 18,
                "attention.head_count": 4,
                "attention.head_count_kv": 1,
                "attention.key_length": 256,
                "attention.value_length": 256,
                # no sliding_window key
            },
        )
        assert b._sliding_window_pattern is None

    @pytest.mark.parametrize(
        "arch", ["llama", "qwen2", "qwen3", "mistral", "mistral3", "glm4", "llama4"]
    )
    def test_non_swa_arch_uses_full_attention_path(self, arch):
        # Pure-GQA arches: GGUF has no sliding_window, no synthetic
        # pattern, estimator hits Path 4.
        b = _backend_from_gguf(
            arch,
            {
                "block_count": 32,
                "attention.head_count": 32,
                "attention.head_count_kv": 8,
                "attention.key_length": 128,
                "attention.value_length": 128,
                "embedding_length": 4096,
            },
        )
        assert b._sliding_window_pattern is None
        assert b._sliding_window is None
        kv = b._estimate_kv_cache_bytes(8192, "f16")
        gqa_expected = 32 * 8192 * 8 * (128 + 128) * 2
        assert kv == gqa_expected

    def test_arch_default_reduces_kv_estimate_vs_legacy(self):
        common = {
            "block_count": 62,
            "attention.head_count": 32,
            "attention.head_count_kv": 16,
            "attention.key_length": 128,
            "attention.value_length": 128,
            "attention.sliding_window": 1024,
            "embedding_length": 5376,
        }
        with_default = _backend_from_gguf("gemma3", common)
        # Arch not in the table -> legacy 1/4 path.
        without_default = _backend_from_gguf("totallymadeupv7", common)

        kv_default = with_default._estimate_kv_cache_bytes(131072, "f16")
        kv_legacy = without_default._estimate_kv_cache_bytes(131072, "f16")
        assert kv_default > 0
        assert kv_legacy > 0
        assert kv_default < kv_legacy, (
            f"arch fallback should under-shoot legacy estimate: "
            f"{kv_default} >= {kv_legacy}"
        )

    def test_scalar_sliding_window_pattern_expanded(self):
        block_count = 8
        b = _backend_from_gguf(
            "gemma3",
            {
                "attention.sliding_window_pattern": 4,
                "block_count": block_count,
                "attention.head_count_kv": 4,
                "attention.key_length": 256,
                "attention.value_length": 256,
                "attention.sliding_window": 1024,
            },
        )
        expected = [(i + 1) % 4 != 0 for i in range(block_count)]
        assert isinstance(b._sliding_window_pattern, list)
        assert b._sliding_window_pattern == expected
        assert b._estimate_kv_cache_bytes(4096, "f16") > 0

    def test_all_fields_parsed_together(self):
        fields = {
            "context_length": 131072,
            "block_count": 62,
            "attention.head_count_kv": 16,
            "attention.head_count": 32,
            "embedding_length": 5376,
            "attention.key_length": 128,
            "attention.value_length": 128,
            "attention.sliding_window": 1024,
            "attention.sliding_window_pattern": [True, False],
            "full_attention_interval": 6,
            "attention.kv_lora_rank": 512,
            "attention.key_length_mla": 256,
            "attention.key_length_swa": 64,
            "attention.value_length_swa": 64,
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
        assert b._sliding_window_pattern == [True, False]
        assert b._full_attention_interval == 6
        assert b._kv_lora_rank == 512
        assert b._key_length_mla == 256
        assert b._kv_key_length_swa == 64
        assert b._kv_value_length_swa == 64
        assert b._ssm_inner_size == 4096
        assert b._ssm_state_size == 128


_SWA_FIELDS = {
    "block_count": 12,
    "attention.head_count": 4,
    "attention.head_count_kv": 1,
    "attention.key_length": 256,
    "attention.value_length": 256,
    "attention.sliding_window": 512,
}


class TestDynamicSwaResolver:
    """4-tier resolver: GGUF metadata, on-disk cache, bootstrap, HF fetch."""

    def _isolate_cache(self, monkeypatch, tmp_path):
        from core.inference import llama_cpp as lc

        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
        monkeypatch.setattr(lc, "_SWA_CACHE", None)
        return tmp_path

    def test_period_from_layer_types_finds_smallest_period(self):
        from core.inference.llama_cpp import _period_from_layer_types

        # gemma3 (1 global per 6), gpt-oss (alternating), gemma3n (1 per 5).
        assert (
            _period_from_layer_types(
                (["sliding_attention"] * 5 + ["full_attention"]) * 4
            )
            == 6
        )
        assert (
            _period_from_layer_types(["sliding_attention", "full_attention"] * 12) == 2
        )
        assert (
            _period_from_layer_types(
                (["sliding_attention"] * 4 + ["full_attention"]) * 7
            )
            == 5
        )

    def test_period_from_layer_types_returns_none_for_aperiodic(self):
        from core.inference.llama_cpp import _period_from_layer_types

        lt = [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
        ]
        assert _period_from_layer_types(lt) is None

    def test_hf_repo_from_url(self):
        from core.inference.llama_cpp import _hf_repo_from_url

        assert (
            _hf_repo_from_url("https://huggingface.co/google/gemma-3-1b-it")
            == "google/gemma-3-1b-it"
        )
        assert (
            _hf_repo_from_url(
                "https://huggingface.co/google/gemma-3-1b-it/blob/main/config.json"
            )
            == "google/gemma-3-1b-it"
        )
        for bad in [
            "https://huggingface.co/google",
            "https://example.com/foo/bar",
            None,
            "",
        ]:
            assert _hf_repo_from_url(bad) is None

    def test_bootstrap_tier_used_when_no_cache(self, monkeypatch, tmp_path):
        self._isolate_cache(monkeypatch, tmp_path)
        from core.inference import llama_cpp as lc

        def boom(*a, **kw):
            raise AssertionError("HF fetch must not run when bootstrap covers the arch")

        monkeypatch.setattr(lc, "_fetch_swa_entry_from_hf", boom)
        b = _backend_from_gguf("gemma3", dict(_SWA_FIELDS, block_count = 18))
        assert b._sliding_window_pattern == [(i + 1) % 6 != 0 for i in range(18)]

    def test_disk_cache_takes_precedence_over_bootstrap(self, monkeypatch, tmp_path):
        self._isolate_cache(monkeypatch, tmp_path)
        # Override bootstrap=6 with a cached period=3.
        with open(tmp_path / "swa_cache.json", "w") as f:
            json.dump({"gemma3": 3}, f)
        b = _backend_from_gguf("gemma3", dict(_SWA_FIELDS, block_count = 18))
        assert b._sliding_window_pattern == [(i + 1) % 3 != 0 for i in range(18)]

    def test_disk_cache_supports_array_entries(self, monkeypatch, tmp_path):
        # Aperiodic mask gets tiled across n_layers.
        self._isolate_cache(monkeypatch, tmp_path)
        mask = [True, False, True, True, False, True, False, False]
        with open(tmp_path / "swa_cache.json", "w") as f:
            json.dump({"customarch": mask}, f)
        b = _backend_from_gguf("customarch", dict(_SWA_FIELDS, block_count = 16))
        assert b._sliding_window_pattern == [bool(mask[i % 8]) for i in range(16)]

    def test_hf_fetch_populates_cache(self, monkeypatch, tmp_path):
        self._isolate_cache(monkeypatch, tmp_path)
        from core.inference import llama_cpp as lc

        calls = []

        def fake_fetch(repo_id):
            calls.append(repo_id)
            return 4 if repo_id == "vendor/newmodel-1b-instruct" else None

        monkeypatch.setattr(lc, "_fetch_swa_entry_from_hf", fake_fetch)
        b = _backend_from_gguf(
            "newmodel",
            _SWA_FIELDS,
            general = {
                "general.source.huggingface.repository": "vendor/newmodel-1b-instruct"
            },
        )
        assert b._sliding_window_pattern == [(i + 1) % 4 != 0 for i in range(12)]
        assert calls == ["vendor/newmodel-1b-instruct"]
        with open(tmp_path / "swa_cache.json") as f:
            assert json.load(f) == {"newmodel": 4}

    def test_hf_fetch_falls_back_to_other_candidates(self, monkeypatch, tmp_path):
        self._isolate_cache(monkeypatch, tmp_path)
        from core.inference import llama_cpp as lc

        monkeypatch.setattr(
            lc,
            "_fetch_swa_entry_from_hf",
            lambda r: 6 if r == "vendor/newmodel-base" else None,
        )
        b = _backend_from_gguf(
            "newmodel",
            _SWA_FIELDS,
            general = {
                "general.base_model.0.repo_url": "https://huggingface.co/vendor/newmodel-base"
            },
        )
        assert b._sliding_window_pattern == [(i + 1) % 6 != 0 for i in range(12)]

    def test_offline_env_skips_network(self, monkeypatch, tmp_path):
        self._isolate_cache(monkeypatch, tmp_path)
        monkeypatch.setenv("UNSLOTH_STUDIO_OFFLINE", "1")
        from core.inference import llama_cpp as lc

        def boom(*a, **kw):
            raise AssertionError("HF fetch must not run when offline=1")

        monkeypatch.setattr(lc, "_fetch_swa_entry_from_hf", boom)
        b = _backend_from_gguf(
            "newmodel",
            _SWA_FIELDS,
            general = {"general.source.huggingface.repository": "vendor/newmodel"},
        )
        assert b._sliding_window_pattern is None

    def test_hf_fetch_failure_falls_through_silently(self, monkeypatch, tmp_path):
        self._isolate_cache(monkeypatch, tmp_path)
        from core.inference import llama_cpp as lc

        monkeypatch.setattr(lc, "_fetch_swa_entry_from_hf", lambda repo_id: None)
        # Force the failure into the Tier 3 path; bypass Tier 2.5.
        monkeypatch.setattr(
            lc, "_resolve_swa_entry_from_transformers", lambda arch: None
        )
        b = _backend_from_gguf(
            "newmodel",
            _SWA_FIELDS,
            general = {"general.source.huggingface.repository": "vendor/does-not-exist"},
        )
        assert b._sliding_window_pattern is None
        assert not (tmp_path / "swa_cache.json").exists()


class TestTransformersIntrospection:
    """Tier 2.5: default-init the matching Config; on failure, parse via inspect."""

    def _isolate_cache(self, monkeypatch, tmp_path):
        from core.inference import llama_cpp as lc

        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
        monkeypatch.setattr(lc, "_SWA_CACHE", None)
        return tmp_path

    def test_arch_aliases_normalises_hyphen_underscore(self):
        from core.inference.llama_cpp import _arch_aliases

        aliases = _arch_aliases("falcon-h1")
        assert aliases[0] == "falcon-h1" and "falcon_h1" in aliases
        assert _arch_aliases("gemma3") == ("gemma3",)
        assert _arch_aliases("") == ()

    def test_resolves_real_transformers_arches(self):
        from core.inference.llama_cpp import _resolve_swa_entry_from_transformers

        assert _resolve_swa_entry_from_transformers("gemma3") == 6
        assert _resolve_swa_entry_from_transformers("gemma2") == 2
        assert _resolve_swa_entry_from_transformers("cohere2") == 4

    def test_falls_back_to_inspect_when_default_init_raises(self, monkeypatch):
        from core.inference import llama_cpp as lc

        class _FakeBrokenConfig:
            """Class with sliding_window_pattern: int = 7 in its docstring."""

            def __init__(self, required_arg):
                raise TypeError("requires an argument")

        class _FakeLazyMapping(dict):
            def __getitem__(self, k):
                return (
                    _FakeBrokenConfig if k == "brokenarch" else super().__getitem__(k)
                )

        import sys, types as _types

        fake_auto = _types.ModuleType("transformers.models.auto.configuration_auto")
        fake_auto.CONFIG_MAPPING_NAMES = {"brokenarch": "FakeBroken"}
        fake_auto.CONFIG_MAPPING = _FakeLazyMapping({"brokenarch": "FakeBroken"})
        monkeypatch.setitem(
            sys.modules, "transformers.models.auto.configuration_auto", fake_auto
        )
        assert lc._resolve_swa_entry_from_transformers("brokenarch") == 7

    def test_returns_none_when_transformers_unavailable(self, monkeypatch):
        from core.inference import llama_cpp as lc
        import sys

        orig_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def fake_import(name, *a, **kw):
            if name.startswith("transformers"):
                raise ImportError("transformers not installed")
            return orig_import(name, *a, **kw)

        monkeypatch.setattr("builtins.__import__", fake_import)
        for k in list(sys.modules):
            if k.startswith("transformers"):
                monkeypatch.delitem(sys.modules, k, raising = False)
        assert lc._resolve_swa_entry_from_transformers("gemma3") is None

    def test_returns_none_for_arch_unknown_to_transformers(self):
        from core.inference.llama_cpp import _resolve_swa_entry_from_transformers

        assert _resolve_swa_entry_from_transformers("totally-fake-arch-xyz") is None

    def test_full_resolver_uses_transformers_before_hf_fetch(
        self, monkeypatch, tmp_path
    ):
        # With bootstrap empty, Tier 2.5 must answer before Tier 3 fires.
        self._isolate_cache(monkeypatch, tmp_path)
        from core.inference import llama_cpp as lc

        monkeypatch.setattr(lc, "_BOOTSTRAP_SWA_DEFAULTS", {})

        def boom(repo_id):
            raise AssertionError("Tier 3 must not run when Tier 2.5 has the answer")

        monkeypatch.setattr(lc, "_fetch_swa_entry_from_hf", boom)
        b = _backend_from_gguf(
            "gemma3",
            dict(_SWA_FIELDS, block_count = 18),
            general = {"general.source.huggingface.repository": "google/gemma-3-1b-it"},
        )
        assert b._sliding_window_pattern == [(i + 1) % 6 != 0 for i in range(18)]
        with open(tmp_path / "swa_cache.json") as f:
            assert json.load(f) == {"gemma3": 6}


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
                "attention.head_count_kv": [8, 2],
                "attention.sliding_window_pattern": [True, False],
                "attention.key_length_swa": 64,
                "attention.value_length_swa": 64,
                "ssm.inner_size": 4096,
            },
        )
        assert b._kv_key_length == 128
        assert b._kv_lora_rank == 512
        assert b._n_kv_heads_by_layer == [8, 2]
        assert b._sliding_window_pattern == [True, False]
        assert b._kv_key_length_swa == 64
        assert b._kv_value_length_swa == 64
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
        assert b._n_kv_heads_by_layer is None
        assert b._sliding_window_pattern is None
        assert b._kv_key_length_swa is None
        assert b._kv_value_length_swa is None
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
        # SWA cache is double-buffered: 2 * sliding_window cells, capped at n_ctx.
        swa_cells = min(131072, 2 * 1024)
        expected = int(n_global * 131072 * kv_per + n_swa * swa_cells * kv_per)
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
        swa_cells = min(131072, 2 * 128)
        expected = int(n_global * 131072 * kv_per + n_swa * swa_cells * kv_per)
        assert b._estimate_kv_cache_bytes(131072, "f16") == expected

    def test_gemma4_per_layer_swa_metadata(self):
        b = self._swa_backend(
            _n_layers = 30,
            _n_kv_heads = None,
            _n_kv_heads_by_layer = [8, 8, 8, 8, 8, 2] * 5,
            _n_heads = 16,
            _embedding_length = 2816,
            _kv_key_length = 512,
            _kv_value_length = 512,
            _sliding_window = 1024,
            _sliding_window_pattern = [True, True, True, True, True, False] * 5,
            _kv_key_length_swa = 256,
            _kv_value_length_swa = 256,
        )

        full_layers = 5
        sliding_layers = 25

        def expected(ctx):
            full = full_layers * ctx * 2 * (512 + 512) * 2
            sliding = sliding_layers * min(ctx, 2 * 1024) * 8 * (256 + 256) * 2
            return int(full + sliding)

        for ctx in (4096, 46500, 262144):
            assert b._estimate_kv_cache_bytes(ctx, "f16") == expected(ctx)

    def test_ctx_smaller_than_window(self):
        """When context < 2 * sliding_window, SWA cache caps at ctx."""
        b = self._swa_backend(_sliding_window = 8192)
        n_global = max(1, 62 // 4)  # 15
        n_swa = 62 - n_global  # 47
        kv_per = 16 * (128 + 128) * 2
        ctx = 4096
        expected = int(n_global * ctx * kv_per + n_swa * min(ctx, 2 * 8192) * kv_per)
        assert b._estimate_kv_cache_bytes(ctx, "f16") == expected

    def test_odd_layer_count(self):
        b = self._swa_backend(_n_layers = 63)
        n_global = max(1, 63 // 4)  # 15
        n_swa = 63 - n_global  # 48
        kv_per = 16 * (128 + 128) * 2
        expected = int(n_global * 1000 * kv_per + n_swa * min(1000, 2 * 1024) * kv_per)
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
# J2. Server-flag knobs (--swa-full, --kv-unified/--parallel,
#     --ctx-checkpoints, --kv-offload)
# ---------------------------------------------------------------------------


class TestServerFlags:
    """Estimator should mirror llama-server CLI flags that change KV size."""

    def _swa_backend(self, **overrides):
        defaults = {
            "_n_layers": 26,
            "_n_kv_heads": 4,
            "_n_heads": 8,
            "_embedding_length": 1152,
            "_kv_key_length": 256,
            "_kv_value_length": 256,
            "_sliding_window": 512,
            "_sliding_window_pattern": [True, True, True, True, True, False] * 4
            + [True, True],
        }
        defaults.update(overrides)
        b = LlamaCppBackend()
        for k, v in defaults.items():
            setattr(b, k, v)
        return b

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

    # ── --swa-full ──────────────────────────────────────────────────

    def test_swa_full_collapses_pattern_path_to_full_ctx(self):
        b = self._swa_backend()
        ctx = 32_768
        flagged = b._estimate_kv_cache_bytes(ctx, "f16", swa_full = True)
        # With swa_full, every layer caches n_ctx -- equals path 4 sizing.
        kv_per_token = 4 * (256 + 256) * 2  # n_kv_heads * (k+v) * f16
        expected = 26 * ctx * kv_per_token
        assert flagged == expected
        assert flagged > b._estimate_kv_cache_bytes(ctx, "f16")

    def test_swa_full_collapses_legacy_path_to_full_ctx(self):
        # No per-layer pattern -> 1/4-global heuristic; swa_full overrides.
        b = self._swa_backend(_sliding_window_pattern = None)
        ctx = 16_384
        flagged = b._estimate_kv_cache_bytes(ctx, "f16", swa_full = True)
        n_global = max(1, 26 // 4)
        n_swa = 26 - n_global
        kv_per = 4 * (256 + 256) * 2
        # swa_cells == n_ctx when swa_full=True
        expected = n_global * ctx * kv_per + n_swa * ctx * kv_per
        assert flagged == expected

    def test_swa_full_no_op_for_non_swa_model(self):
        b = self._gqa_backend()
        baseline = b._estimate_kv_cache_bytes(8192, "f16")
        flagged = b._estimate_kv_cache_bytes(8192, "f16", swa_full = True)
        assert flagged == baseline

    def test_swa_full_suppresses_checkpoint_term(self):
        b = self._swa_backend()
        with_cp = b._estimate_kv_cache_bytes(8192, "f16", ctx_checkpoints = 8)
        with_cp_full = b._estimate_kv_cache_bytes(
            8192, "f16", ctx_checkpoints = 8, swa_full = True
        )
        no_cp_full = b._estimate_kv_cache_bytes(8192, "f16", swa_full = True)
        # Checkpoints only matter when SWA layers don't already keep n_ctx.
        assert with_cp_full == no_cp_full
        assert with_cp > b._estimate_kv_cache_bytes(8192, "f16")

    # ── --parallel + --kv-unified ──────────────────────────────────

    def test_unified_kv_ignores_n_parallel(self):
        b = self._gqa_backend()
        baseline = b._estimate_kv_cache_bytes(4096, "f16")
        for slots in (1, 2, 4, 8):
            assert (
                b._estimate_kv_cache_bytes(
                    4096, "f16", n_parallel = slots, kv_unified = True
                )
                == baseline
            )

    def test_non_unified_multiplies_kv_by_n_parallel(self):
        b = self._gqa_backend()
        baseline = b._estimate_kv_cache_bytes(4096, "f16")
        for slots in (1, 2, 4, 8):
            scaled = b._estimate_kv_cache_bytes(
                4096, "f16", n_parallel = slots, kv_unified = False
            )
            assert scaled == baseline * slots

    def test_non_unified_with_zero_parallel_floors_at_one(self):
        b = self._gqa_backend()
        baseline = b._estimate_kv_cache_bytes(4096, "f16")
        scaled = b._estimate_kv_cache_bytes(
            4096, "f16", n_parallel = 0, kv_unified = False
        )
        assert scaled == baseline

    def test_non_unified_scales_swa_path(self):
        b = self._swa_backend()
        baseline = b._estimate_kv_cache_bytes(8192, "f16")
        scaled = b._estimate_kv_cache_bytes(
            8192, "f16", n_parallel = 3, kv_unified = False
        )
        assert scaled == baseline * 3

    def test_non_unified_scales_mla_path(self):
        b = LlamaCppBackend()
        b._n_layers = 60
        b._n_kv_heads = 1
        b._kv_lora_rank = 512
        b._key_length_mla = 64
        b._kv_key_length = 576
        baseline = b._estimate_kv_cache_bytes(8192, "f16")
        scaled = b._estimate_kv_cache_bytes(
            8192, "f16", n_parallel = 4, kv_unified = False
        )
        assert scaled == baseline * 4

    # ── --ctx-checkpoints ──────────────────────────────────────────

    def test_ctx_checkpoints_zero_is_no_op(self):
        b = self._swa_backend()
        baseline = b._estimate_kv_cache_bytes(8192, "f16")
        assert b._estimate_kv_cache_bytes(8192, "f16", ctx_checkpoints = 0) == baseline

    def test_ctx_checkpoints_no_op_for_non_swa(self):
        b = self._gqa_backend()
        baseline = b._estimate_kv_cache_bytes(8192, "f16")
        assert b._estimate_kv_cache_bytes(8192, "f16", ctx_checkpoints = 32) == baseline

    def test_ctx_checkpoints_pattern_path_adds_known_bytes(self):
        b = self._swa_backend()
        ctx = 8192
        baseline = b._estimate_kv_cache_bytes(ctx, "f16")
        flagged = b._estimate_kv_cache_bytes(ctx, "f16", ctx_checkpoints = 4)
        # 22 SWA layers * 4 checkpoints * 512 cells * 4 heads * (256+256) * 2 bytes
        n_swa_layers = sum(
            1 for f in [True, True, True, True, True, False] * 4 + [True, True] if f
        )
        per_layer = 4 * 512 * 4 * (256 + 256) * 2
        assert flagged == baseline + n_swa_layers * per_layer

    def test_ctx_checkpoints_legacy_path_adds_known_bytes(self):
        b = self._swa_backend(_sliding_window_pattern = None)
        ctx = 8192
        baseline = b._estimate_kv_cache_bytes(ctx, "f16")
        flagged = b._estimate_kv_cache_bytes(ctx, "f16", ctx_checkpoints = 4)
        n_global = max(1, 26 // 4)
        n_swa = 26 - n_global
        kv_per = 4 * (256 + 256) * 2
        extra = 4 * n_swa * 512 * kv_per  # ctx_checkpoints * n_swa * sliding * kv_per
        assert flagged == baseline + extra

    def test_ctx_checkpoints_compose_with_n_parallel(self):
        b = self._swa_backend()
        ctx = 8192
        single = b._estimate_kv_cache_bytes(ctx, "f16", ctx_checkpoints = 4)
        triple = b._estimate_kv_cache_bytes(
            ctx, "f16", ctx_checkpoints = 4, n_parallel = 3, kv_unified = False
        )
        assert triple == single * 3

    # ── --kv-offload (kv_on_gpu) ───────────────────────────────────

    def test_fit_returns_requested_when_kv_off_gpu(self):
        b = self._gqa_backend()
        # Tiny VRAM budget -- normally would force a reduction.
        fitted = b._fit_context_to_vram(
            requested_ctx = 32_768,
            available_mib = 1,
            model_size_bytes = 100,
            cache_type_kv = "f16",
            kv_on_gpu = False,
        )
        assert fitted == 32_768

    def test_fit_reduces_when_kv_on_gpu(self):
        b = self._gqa_backend()
        fitted = b._fit_context_to_vram(
            requested_ctx = 32_768,
            available_mib = 64,
            model_size_bytes = 1024 * 1024,  # 1 MiB
            cache_type_kv = "f16",
            kv_on_gpu = True,
        )
        assert fitted < 32_768

    def test_fit_threads_swa_full_through_estimator(self):
        # SWA model, generous budget; both should fit but cache size differs.
        b = self._swa_backend()
        ctx = 8192
        kv_default = b._estimate_kv_cache_bytes(ctx, "f16")
        kv_full = b._estimate_kv_cache_bytes(ctx, "f16", swa_full = True)
        assert kv_full > kv_default
        # Budget = model + kv_default (rounded up) -- swa_full should not fit.
        budget_mib = (1024 * 1024 + kv_default) / (1024 * 1024) / 0.90 + 1
        fitted_default = b._fit_context_to_vram(
            requested_ctx = ctx,
            available_mib = int(budget_mib),
            model_size_bytes = 1024 * 1024,
            cache_type_kv = "f16",
        )
        fitted_full = b._fit_context_to_vram(
            requested_ctx = ctx,
            available_mib = int(budget_mib),
            model_size_bytes = 1024 * 1024,
            cache_type_kv = "f16",
            swa_full = True,
        )
        assert fitted_default == ctx
        assert fitted_full < ctx


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
            "_sliding_window_pattern",
            "_full_attention_interval",
            "_kv_lora_rank",
            "_key_length_mla",
            "_kv_key_length_swa",
            "_kv_value_length_swa",
            "_ssm_inner_size",
            "_ssm_state_size",
        ]:
            assert getattr(b, attr) is None
        assert b._n_kv_heads_by_layer is None

    def test_unload_resets_fields(self):
        b = LlamaCppBackend()
        b._n_layers = 32
        b._kv_key_length = 128
        b._kv_lora_rank = 512
        b._sliding_window = 1024
        b._sliding_window_pattern = [True, False]
        b._n_kv_heads_by_layer = [8, 2]
        b._kv_key_length_swa = 64
        b._kv_value_length_swa = 64
        b._ssm_inner_size = 4096
        b._full_attention_interval = 4
        b.unload_model()
        for attr in [
            "_kv_key_length",
            "_kv_value_length",
            "_sliding_window",
            "_sliding_window_pattern",
            "_full_attention_interval",
            "_kv_lora_rank",
            "_key_length_mla",
            "_kv_key_length_swa",
            "_kv_value_length_swa",
            "_ssm_inner_size",
            "_ssm_state_size",
        ]:
            assert getattr(b, attr) is None
        assert b._n_kv_heads_by_layer is None

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
        # gemma3 -> period 6 from the bootstrap table, SWA cache
        # double-buffered to 2 * sliding_window cells.
        period = 6
        kv_per = 16 * 256 * 2
        expected = 0
        for i in range(62):
            is_swa = (i + 1) % period != 0
            layer_ctx = min(131072, 2 * 1024) if is_swa else 131072
            expected += layer_ctx * kv_per
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
