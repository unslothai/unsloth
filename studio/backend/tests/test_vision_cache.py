# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for is_vision_model() caching behaviour.

``_vision_detection_cache`` mirrors the ``_audio_detection_cache``
pattern used by ``detect_audio_type()``. These tests verify:

* Repeated calls for the same model hit the cache.
* Different models each trigger their own detection.
* Both True and False results are cached.
* The subprocess path (transformers 5.x models) is cached.
* Exceptions that fall back to False are cached.
"""

import sys
import types as _types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# sys.path + logger stub — same pattern as the rest of the test suite
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

from utils.models.model_config import (
    ModelConfig,
    is_vision_model,
    _is_vision_model_uncached,
    _vision_detection_cache,
)


# Helpers


@pytest.fixture(autouse = True)
def _clear_vision_cache(tmp_path, monkeypatch):
    """Ensure every test starts with a fresh cache, from an empty working dir.

    ``is_vision_model`` calls ``is_local_path`` first: any relative model id that
    happens to exist on disk (``Path(name).exists()``) is treated as a local
    model, short-circuiting before the mocked detection internals run. The CI cwd
    (``studio/backend``) and the HF cache can contain dirs whose names collide
    with the synthetic remote ids used here (``org/my-vlm``, ``model-a``,
    ``broken/model`` ...), which made these tests fail with "called 0 times".
    Running each test from a fresh empty ``tmp_path`` removes that collision
    while leaving the real ``is_local_path`` logic intact (the local-GGUF tests
    pass absolute ``tmp_path`` paths, unaffected by cwd).
    """
    monkeypatch.chdir(tmp_path)
    _vision_detection_cache.clear()
    yield
    _vision_detection_cache.clear()


# Cache hit / miss tests


class TestVisionCacheHitMiss:
    """Verify the cache prevents redundant detection calls."""

    @patch("utils.models.model_config._is_vision_model_uncached", return_value = True)
    def test_second_call_uses_cache(self, mock_uncached):
        """Two calls for the same model invoke the uncached fn once."""
        assert is_vision_model("org/my-vlm") is True
        assert is_vision_model("org/my-vlm") is True
        mock_uncached.assert_called_once_with("org/my-vlm", None, local_files_only = False)

    @patch("utils.models.model_config._is_vision_model_uncached", return_value = False)
    def test_different_models_each_detected(self, mock_uncached):
        """Different model names should each trigger detection."""
        is_vision_model("model-a")
        is_vision_model("model-b")
        assert mock_uncached.call_count == 2

    @patch("utils.models.model_config._is_vision_model_uncached", return_value = True)
    def test_cache_returns_correct_value(self, mock_uncached):
        """The cached value must match what _is_vision_model_uncached returned."""
        first = is_vision_model("org/vlm")
        second = is_vision_model("org/vlm")
        assert first is True
        assert second is True


class TestVisionCacheStoresFalse:
    """Non-VLM results (False) must also be cached to avoid re-detection."""

    @patch("utils.models.model_config._is_vision_model_uncached", return_value = False)
    def test_false_result_cached(self, mock_uncached):
        assert is_vision_model("org/text-only") is False
        assert is_vision_model("org/text-only") is False
        mock_uncached.assert_called_once()
        assert _vision_detection_cache[("org/text-only", None, False)] is False


# Subprocess path (transformers 5.x) caching


class TestVisionCacheSubprocessPath:
    """transformers 5.x models go through _is_vision_model_subprocess.
    The cache should spawn the subprocess at most once per model per
    process."""

    @patch("utils.models.model_config._raw_config_has_vision_config", return_value = None)
    @patch("utils.models.model_config._is_vision_model_subprocess", return_value = True)
    @patch("utils.transformers_version.needs_transformers_5", return_value = True)
    def test_subprocess_called_once_with_cache(self, mock_needs_t5, mock_subprocess, mock_raw):
        """When the raw-config reader is inconclusive (None), the transformers
        5.x subprocess fires only on the first call; the second is cached."""
        # First call: raw None -> subprocess
        assert is_vision_model("unsloth/Qwen3.5-2B") is True
        # Second call: cache hit, no subprocess
        assert is_vision_model("unsloth/Qwen3.5-2B") is True

        mock_subprocess.assert_called_once()
        assert _vision_detection_cache[("unsloth/Qwen3.5-2B", None, False)] is True

    @patch("utils.models.model_config._raw_config_has_vision_config", return_value = True)
    @patch("utils.models.model_config._is_vision_model_subprocess", return_value = None)
    @patch("utils.transformers_version.needs_transformers_5", return_value = True)
    def test_raw_config_primary_skips_subprocess(
        self, mock_needs_t5, mock_subprocess, mock_raw_config
    ):
        # The raw config.json read is the primary path; a definitive answer there never
        # reaches the transformers-5.x subprocess or needs_transformers_5 routing.
        assert is_vision_model("unsloth/gemma-4-E4B-it") is True
        assert is_vision_model("unsloth/gemma-4-E4B-it") is True

        mock_raw_config.assert_called_once_with(
            "unsloth/gemma-4-E4B-it", hf_token = None, local_files_only = False
        )
        mock_subprocess.assert_not_called()


# ---------------------------------------------------------------------------
# Local GGUF capability path
# ---------------------------------------------------------------------------


class TestLocalGgufVisionDetection:
    @patch(
        "utils.models.model_config._is_vision_model_subprocess",
        side_effect = AssertionError("GGUF must not use Transformers vision detection"),
    )
    def test_qwen36_gguf_with_mmproj_skips_transformers(self, mock_subprocess, tmp_path):
        model = tmp_path / "Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf"
        model.write_bytes(b"")
        (tmp_path / "mmproj-F32.gguf").write_bytes(b"")

        assert is_vision_model(str(model)) is True
        mock_subprocess.assert_not_called()

    @patch(
        "utils.models.model_config._is_vision_model_subprocess",
        side_effect = AssertionError("GGUF must not use Transformers vision detection"),
    )
    def test_direct_gguf_in_variant_subdir_finds_snapshot_mmproj(self, mock_subprocess, tmp_path):
        variant_dir = tmp_path / "BF16"
        variant_dir.mkdir()
        model = variant_dir / "Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf"
        model.write_bytes(b"")
        (tmp_path / "mmproj-F32.gguf").write_bytes(b"")

        assert is_vision_model(str(model)) is True
        mock_subprocess.assert_not_called()

    @patch(
        "utils.models.model_config._is_vision_model_subprocess",
        side_effect = AssertionError("GGUF must not use Transformers vision detection"),
    )
    def test_qwen36_gguf_without_mmproj_skips_transformers(self, mock_subprocess, tmp_path):
        model = tmp_path / "Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf"
        model.write_bytes(b"")

        assert is_vision_model(str(model)) is False
        mock_subprocess.assert_not_called()

    def test_local_gguf_check_observes_mmproj_added_later(self, tmp_path):
        model = tmp_path / "Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf"
        model.write_bytes(b"")

        assert is_vision_model(str(model)) is False
        (tmp_path / "mmproj-F32.gguf").write_bytes(b"")
        assert is_vision_model(str(model)) is True

    @patch(
        "utils.models.model_config._is_vision_model_subprocess",
        side_effect = AssertionError("GGUF must not use Transformers vision detection"),
    )
    def test_ui_selection_returns_local_gguf_config(self, mock_subprocess, tmp_path):
        model = tmp_path / "Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf"
        model.write_bytes(b"")
        mmproj = tmp_path / "mmproj-F32.gguf"
        mmproj.write_bytes(b"")

        config = ModelConfig.from_ui_selection(str(model), None)

        assert config is not None
        assert config.is_gguf is True
        assert config.is_vision is True
        assert config.gguf_mmproj_file == str(mmproj.resolve())
        mock_subprocess.assert_not_called()

    @patch(
        "utils.models.model_config._is_vision_model_subprocess",
        side_effect = AssertionError("GGUF must not use Transformers vision detection"),
    )
    def test_ui_selection_direct_gguf_in_variant_subdir_keeps_mmproj(
        self, mock_subprocess, tmp_path
    ):
        variant_dir = tmp_path / "BF16"
        variant_dir.mkdir()
        model = variant_dir / "Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf"
        model.write_bytes(b"")
        mmproj = tmp_path / "mmproj-F32.gguf"
        mmproj.write_bytes(b"")

        config = ModelConfig.from_ui_selection(str(model), None)

        assert config is not None
        assert config.is_gguf is True
        assert config.is_vision is True
        assert config.gguf_mmproj_file == str(mmproj.resolve())
        mock_subprocess.assert_not_called()


# ---------------------------------------------------------------------------
# Exception handling — cache the False fallback


class TestVisionCacheOnException:
    """On exception, _is_vision_model_uncached distinguishes permanent
    failures (cached as False) from transient ones (returned as None,
    not cached, so the next call retries). Verify both contracts."""

    @patch(
        "utils.models.model_config.load_model_config",
        side_effect = ValueError("bad config"),
    )
    @patch("utils.transformers_version.needs_transformers_5", return_value = False)
    def test_permanent_exception_result_cached(self, mock_needs_t5, mock_load_config):
        """A permanent failure (ValueError / RepositoryNotFoundError /
        GatedRepoError / JSONDecodeError) is caught, returns False, and
        that False is cached so subsequent calls don't retry. ValueError
        stands in as the simplest cacheable exception type."""
        # First call raises -> False; second is a cache hit.
        assert is_vision_model("broken/model") is False
        assert is_vision_model("broken/model") is False
        mock_load_config.assert_called_once()

    @patch(
        "utils.models.model_config.load_model_config",
        side_effect = OSError("network down"),
    )
    @patch("utils.transformers_version.needs_transformers_5", return_value = False)
    def test_transient_exception_not_cached(self, mock_needs_t5, mock_load_config):
        """A transient failure (OSError, timeouts) returns None from
        _is_vision_model_uncached, surfaces as False, and is NOT cached
        so the next call retries."""
        # First call: OSError -> False, not cached; second call retries.
        assert is_vision_model("broken/model") is False
        assert is_vision_model("broken/model") is False
        assert mock_load_config.call_count == 2


# Direct detection path (non-transformers-5 models) caching


class TestVisionCacheDirectPath:
    """Models that do NOT need transformers 5.x detect via
    load_model_config directly. The cache must work the same way."""

    @patch("utils.models.model_config._raw_config_has_vision_config", return_value = None)
    @patch("utils.transformers_version.needs_transformers_5", return_value = False)
    @patch("utils.models.model_config.load_model_config")
    def test_direct_vlm_detection_cached(self, mock_load_config, mock_needs_t5, mock_raw):
        """A standard VLM detected via architecture suffix should be cached."""
        cfg = MagicMock(spec = [])  # strict: only explicitly set attrs exist
        cfg.model_type = "gemma3"
        cfg.architectures = ["Gemma3ForConditionalGeneration"]
        mock_load_config.return_value = cfg

        assert is_vision_model("google/gemma-3-4b-it") is True
        assert is_vision_model("google/gemma-3-4b-it") is True
        # load_model_config should only be called once
        mock_load_config.assert_called_once()

    @patch("utils.models.model_config._raw_config_has_vision_config", return_value = None)
    @patch("utils.transformers_version.needs_transformers_5", return_value = False)
    @patch("utils.models.model_config.load_model_config")
    def test_direct_non_vlm_detection_cached(self, mock_load_config, mock_needs_t5, mock_raw):
        """A standard text model (no VLM indicators) should cache False."""
        cfg = MagicMock(spec = [])  # spec=[] means no attributes at all
        cfg.model_type = "llama"
        cfg.architectures = ["LlamaForCausalLM"]
        mock_load_config.return_value = cfg

        # No VLM suffix, no vision_config, etc.
        assert is_vision_model("meta-llama/Llama-3-8B") is False
        assert is_vision_model("meta-llama/Llama-3-8B") is False
        mock_load_config.assert_called_once()

    @patch("utils.models.model_config._raw_config_has_vision_config", return_value = None)
    @patch("utils.transformers_version.needs_transformers_5", return_value = False)
    @patch("utils.models.model_config.load_model_config")
    def test_vision_config_attr_detected_and_cached(
        self, mock_load_config, mock_needs_t5, mock_raw
    ):
        """Models with vision_config (LLaVA, Qwen2-VL, etc.) should be cached as True."""
        cfg = MagicMock(spec = [])  # strict: only explicitly set attrs exist
        cfg.model_type = "qwen2_vl"
        cfg.architectures = ["Qwen2VLForCausalLM"]  # Doesn't match VLM suffixes
        cfg.vision_config = {"hidden_size": 1024}
        mock_load_config.return_value = cfg

        assert is_vision_model("Qwen/Qwen2-VL-7B") is True
        assert is_vision_model("Qwen/Qwen2-VL-7B") is True
        mock_load_config.assert_called_once()

    @patch("utils.models.model_config._raw_config_has_vision_config", return_value = None)
    @patch("utils.transformers_version.needs_transformers_5", return_value = False)
    @patch("utils.models.model_config.load_model_config")
    def test_gemma4_model_type_detected_and_cached(self, mock_load_config, mock_needs_t5, mock_raw):
        cfg = MagicMock(spec = [])
        cfg.model_type = "gemma4"
        cfg.architectures = ["Gemma4ForConditionalGeneration"]
        mock_load_config.return_value = cfg

        assert is_vision_model("google/gemma-4-E4B-it") is True
        assert is_vision_model("google/gemma-4-E4B-it") is True
        mock_load_config.assert_called_once()

    @patch("utils.models.model_config._raw_config_has_vision_config", return_value = None)
    @patch("utils.transformers_version.needs_transformers_5", return_value = False)
    @patch("utils.models.model_config.load_model_config")
    def test_gemma4_audio_subconfig_not_detected_as_vision(
        self, mock_load_config, mock_needs_t5, mock_raw
    ):
        cfg = MagicMock(spec = [])
        cfg.model_type = "gemma4_audio"
        cfg.architectures = ["Gemma4AudioModel"]
        mock_load_config.return_value = cfg

        assert is_vision_model("local/gemma4-audio-encoder") is False
        assert is_vision_model("local/gemma4-audio-encoder") is False
        mock_load_config.assert_called_once()

    @patch("utils.models.model_config._raw_config_has_vision_config", return_value = None)
    @patch("utils.transformers_version.needs_transformers_5", return_value = False)
    @patch("utils.models.model_config.load_model_config")
    def test_gemma4_text_subconfig_not_detected_as_vision(
        self, mock_load_config, mock_needs_t5, mock_raw
    ):
        cfg = MagicMock(spec = [])
        cfg.model_type = "gemma4_text"
        cfg.architectures = ["Gemma4ForCausalLM"]
        mock_load_config.return_value = cfg

        assert is_vision_model("local/gemma-4-text") is False
        assert is_vision_model("local/gemma-4-text") is False
        mock_load_config.assert_called_once()

    @patch("utils.models.model_config._raw_config_has_vision_config", return_value = None)
    @patch("utils.transformers_version.needs_transformers_5", return_value = False)
    @patch("utils.models.model_config.load_model_config")
    def test_audio_model_excluded_and_cached(self, mock_load_config, mock_needs_t5, mock_raw):
        """Audio-only models (csm, whisper) with ForConditionalGeneration
        should be excluded from VLM detection and cached as False."""
        cfg = MagicMock(spec = [])  # strict: only explicitly set attrs exist
        cfg.model_type = "whisper"
        cfg.architectures = ["WhisperForConditionalGeneration"]
        mock_load_config.return_value = cfg

        assert is_vision_model("openai/whisper-large-v3") is False
        assert is_vision_model("openai/whisper-large-v3") is False
        mock_load_config.assert_called_once()


# hf_token handling


class TestVisionCacheTokenHandling:
    """The cache is keyed on (model_name, hf_token). Different tokens
    for the same model trigger separate detections for gated models."""

    @patch("utils.models.model_config._is_vision_model_uncached", return_value = True)
    def test_different_tokens_trigger_new_detection(self, mock_uncached):
        """Different tokens trigger separate detections for gated models
        (e.g. unauthenticated probe → False, then authenticated
        re-check)."""
        assert is_vision_model("gated/model", hf_token = "token-a") is True
        assert is_vision_model("gated/model", hf_token = "token-b") is True
        assert mock_uncached.call_count == 2

    @patch("utils.models.model_config._is_vision_model_uncached", return_value = True)
    def test_same_token_uses_cache(self, mock_uncached):
        """Repeated calls with identical model + token should hit cache."""
        assert is_vision_model("gated/model", hf_token = "token-a") is True
        assert is_vision_model("gated/model", hf_token = "token-a") is True
        mock_uncached.assert_called_once()


class TestVisionCacheLocalOnly:
    """local_files_only is in the cache key: an offline negative must not be reused by a
    later online probe (else a VLM is routed through the text loader until restart)."""

    def test_local_only_negative_does_not_poison_online(self, monkeypatch):
        import utils.models.model_config as mc

        mc._vision_detection_cache.clear()
        monkeypatch.setattr(mc, "is_local_path", lambda *_a, **_k: False)
        monkeypatch.setattr(mc, "resolve_cached_repo_id_case", lambda n, *_a, **_k: n)
        # Pin env-offline off so the key tracks the kwarg.
        monkeypatch.setattr(mc, "_env_offline", lambda: False)

        seen = []

        def _probe(
            name,
            hf_token = None,
            local_files_only = False,
        ):
            seen.append(local_files_only)
            # Offline can't fetch -> not a VLM; online reveals the VLM.
            return False if local_files_only else True

        monkeypatch.setattr(mc, "_is_vision_model_uncached", _probe)

        # Offline probe caches False under a local-only key.
        assert mc.is_vision_model("some/vlm", local_files_only = True) is False
        # A later online probe must re-run (different key) and detect the VLM.
        assert mc.is_vision_model("some/vlm", local_files_only = False) is True
        assert seen == [True, False]
        # The online positive is then cached for subsequent online callers.
        assert mc.is_vision_model("some/vlm", local_files_only = False) is True
        assert seen == [True, False]
        mc._vision_detection_cache.clear()


# ---------------------------------------------------------------------------
# Direct unit tests for _raw_config_has_vision_config
# ---------------------------------------------------------------------------


import json as _json

from utils.models.model_config import (
    _AUDIO_ONLY_MODEL_TYPES,
    _VISION_CHECK_INLINE_HELPERS,
    _VISION_CHECK_SCRIPT,
    _is_vlm,
    _raw_config_has_vision_config,
)


def _write_config(tmp_path, config):
    (tmp_path / "config.json").write_text(_json.dumps(config))
    return tmp_path


class TestRawConfigVlmDetection:
    """Direct coverage of _raw_config_has_vision_config across the same
    indicator set used by _is_vlm. The cache integration tests above mock
    this function; these exercise its real implementation."""

    def test_truthy_vision_config(self, tmp_path):
        p = _write_config(tmp_path, {"vision_config": {"hidden_size": 1024}})
        assert _raw_config_has_vision_config(str(p)) is True

    def test_empty_vision_config_key(self, tmp_path):
        p = _write_config(tmp_path, {"vision_config": {}})
        assert _raw_config_has_vision_config(str(p)) is True

    def test_arch_suffix_detection(self, tmp_path):
        p = _write_config(
            tmp_path,
            {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "model_type": "gemma4",
            },
        )
        assert _raw_config_has_vision_config(str(p)) is True

    def test_img_processor_key(self, tmp_path):
        p = _write_config(tmp_path, {"img_processor": {"image_size": 336}})
        assert _raw_config_has_vision_config(str(p)) is True

    def test_image_token_index_key(self, tmp_path):
        p = _write_config(tmp_path, {"image_token_index": 32000})
        assert _raw_config_has_vision_config(str(p)) is True

    def test_known_vlm_model_type(self, tmp_path):
        p = _write_config(tmp_path, {"model_type": "gemma4"})
        assert _raw_config_has_vision_config(str(p)) is True

    def test_plain_text_model_returns_false(self, tmp_path):
        p = _write_config(
            tmp_path,
            {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
        )
        assert _raw_config_has_vision_config(str(p)) is False

    def test_missing_config_returns_none(self, tmp_path):
        assert _raw_config_has_vision_config(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Self-contained subprocess script (no parent backend imports)
# ---------------------------------------------------------------------------


class TestSubprocessScript:
    def test_does_not_import_parent_module(self):
        assert "from utils.models.model_config" not in _VISION_CHECK_SCRIPT

    def test_inline_is_vlm_executes_correctly(self):
        ns: dict = {}
        exec(_VISION_CHECK_INLINE_HELPERS, ns)
        inline_is_vlm = ns["_is_vlm"]

        class _C:
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)

        assert (
            inline_is_vlm(
                _C(
                    model_type = "gemma4",
                    architectures = ["Gemma4ForConditionalGeneration"],
                )
            )
            is True
        )
        assert (
            inline_is_vlm(_C(model_type = "gemma4_text", architectures = ["Gemma4ForCausalLM"]))
            is False
        )
        assert inline_is_vlm(_C(model_type = "llama", architectures = ["LlamaForCausalLM"])) is False


# ---------------------------------------------------------------------------
# Audio-only model exclusion must apply across every detection path
# ---------------------------------------------------------------------------


class TestVlmAudioExclusion:
    """The {csm, whisper} guard previously lived only in the direct caller
    branch. These tests assert it now applies inside _is_vlm, the raw
    fallback, and the inlined subprocess helper too."""

    def test_audio_only_set_canonical(self):
        # Derived from the transformers audio registry, so a superset of {csm, whisper}.
        assert {"csm", "whisper"} <= _AUDIO_ONLY_MODEL_TYPES

    def test_is_vlm_excludes_whisper(self):
        cfg = MagicMock(spec = [])
        cfg.model_type = "whisper"
        cfg.architectures = ["WhisperForConditionalGeneration"]
        assert _is_vlm(cfg) is False

    def test_raw_fallback_excludes_whisper(self, tmp_path):
        p = _write_config(
            tmp_path,
            {
                "architectures": ["WhisperForConditionalGeneration"],
                "model_type": "whisper",
            },
        )
        assert _raw_config_has_vision_config(str(p)) is False

    def test_inline_subprocess_helper_excludes_whisper(self):
        ns: dict = {}
        exec(_VISION_CHECK_INLINE_HELPERS, ns)
        cfg = MagicMock(spec = [])
        cfg.model_type = "whisper"
        cfg.architectures = ["WhisperForConditionalGeneration"]
        assert ns["_is_vlm"](cfg) is False

    @patch("utils.models.model_config._is_vision_model_subprocess", return_value = None)
    @patch("utils.transformers_version.needs_transformers_5", return_value = True)
    def test_t5_subprocess_none_falls_back_through_raw_for_whisper(
        self, mock_needs_t5, mock_subprocess, tmp_path
    ):
        _write_config(
            tmp_path,
            {
                "architectures": ["WhisperForConditionalGeneration"],
                "model_type": "whisper",
            },
        )
        assert is_vision_model(str(tmp_path)) is False


class TestAudioDetectionCacheTokenAware:
    """The audio cache mirrors the vision cache: keyed by (model, token_fingerprint)
    so an unauthenticated miss cannot poison a later authenticated lookup."""

    def test_audio_cache_is_token_aware(self, monkeypatch):
        import utils.models.model_config as mc

        mc._audio_detection_cache.clear()
        calls = []

        def _fake(
            name,
            hf_token = None,
            local_files_only = False,
        ):
            calls.append(hf_token)
            # Gated repo: only an authenticated probe can read the tokenizer.
            return ("bicodec", True) if hf_token else (None, True)

        monkeypatch.setattr(mc, "_detect_audio_from_tokenizer", _fake)
        monkeypatch.setattr(mc, "is_local_path", lambda *_a, **_k: False)
        monkeypatch.setattr(mc, "resolve_cached_repo_id_case", lambda n, *_a, **_k: n)

        # Unauthenticated miss caches None under (name, None)...
        assert mc.detect_audio_type("private/spark") is None
        # ...but the authenticated call uses a different key and is NOT poisoned.
        assert mc.detect_audio_type("private/spark", hf_token = "hf_x") == "bicodec"
        assert calls == [None, "hf_x"]

        # Same (model, token) is served from cache (no third probe).
        assert mc.detect_audio_type("private/spark", hf_token = "hf_x") == "bicodec"
        assert calls == [None, "hf_x"]
        mc._audio_detection_cache.clear()

    def test_transient_none_is_not_cached_but_definitive_none_is(self, monkeypatch):
        """A transient probe failure (definitive=False) must retry; a clean
        'not audio' read (definitive=True) caches so we don't re-probe."""
        import utils.models.model_config as mc

        mc._audio_detection_cache.clear()
        monkeypatch.setattr(mc, "is_local_path", lambda *_a, **_k: False)
        monkeypatch.setattr(mc, "resolve_cached_repo_id_case", lambda n, *_a, **_k: n)

        transient_calls = []

        def _transient(
            name,
            hf_token = None,
            local_files_only = False,
        ):
            transient_calls.append(hf_token)
            return (None, False)  # network/5xx -- not cacheable

        monkeypatch.setattr(mc, "_detect_audio_from_tokenizer", _transient)
        assert mc.detect_audio_type("flaky/model") is None
        assert mc.detect_audio_type("flaky/model") is None
        # Re-probed both times: the transient None was never cached.
        assert transient_calls == [None, None]

        definitive_calls = []

        def _definitive(
            name,
            hf_token = None,
            local_files_only = False,
        ):
            definitive_calls.append(hf_token)
            return (None, True)  # read the config, no audio tokens

        monkeypatch.setattr(mc, "_detect_audio_from_tokenizer", _definitive)
        assert mc.detect_audio_type("plain/text-model") is None
        assert mc.detect_audio_type("plain/text-model") is None
        # Probed once: the definitive None was cached.
        assert definitive_calls == [None]
        mc._audio_detection_cache.clear()

    def test_local_only_negative_does_not_poison_online(self, monkeypatch):
        """An offline negative must not be reused by a later online probe (else an audio
        model is routed through the text loader until restart)."""
        import utils.models.model_config as mc

        mc._audio_detection_cache.clear()
        monkeypatch.setattr(mc, "is_local_path", lambda *_a, **_k: False)
        monkeypatch.setattr(mc, "resolve_cached_repo_id_case", lambda n, *_a, **_k: n)
        # Pin env-offline off so the key tracks the kwarg.
        monkeypatch.setattr(mc, "_env_offline", lambda: False)

        seen = []

        def _probe(
            name,
            hf_token = None,
            local_files_only = False,
        ):
            seen.append(local_files_only)
            # Offline: nothing on disk -> not audio; online reveals the audio model.
            return (None, True) if local_files_only else ("snac", True)

        monkeypatch.setattr(mc, "_detect_audio_from_tokenizer", _probe)

        # Offline probe caches None under a local-only key.
        assert mc.detect_audio_type("some/audio-model", local_files_only = True) is None
        # A later online probe must re-run (different key) and detect the audio model.
        assert mc.detect_audio_type("some/audio-model", local_files_only = False) == "snac"
        assert seen == [True, False]
        # The online positive is then cached for subsequent online callers.
        assert mc.detect_audio_type("some/audio-model", local_files_only = False) == "snac"
        assert seen == [True, False]
        mc._audio_detection_cache.clear()

    def test_env_offline_negative_does_not_poison_online(self, monkeypatch):
        """An env-offline probe (default local_files_only=False) must cache under the
        effective-offline key, so clearing the env var later doesn't leak a stale negative."""
        import utils.models.model_config as mc

        mc._audio_detection_cache.clear()
        monkeypatch.setattr(mc, "is_local_path", lambda *_a, **_k: False)
        monkeypatch.setattr(mc, "resolve_cached_repo_id_case", lambda n, *_a, **_k: n)

        env_offline = {"v": True}
        monkeypatch.setattr(mc, "_env_offline", lambda: env_offline["v"])

        seen = []

        def _probe(
            name,
            hf_token = None,
            local_files_only = False,
        ):
            seen.append(local_files_only)
            return (None, True) if local_files_only else ("snac", True)

        monkeypatch.setattr(mc, "_detect_audio_from_tokenizer", _probe)

        # Env offline + default kwarg -> probe runs offline; None cached under the offline key.
        assert mc.detect_audio_type("some/audio-model") is None
        assert seen == [True]
        # Env var cleared: a fresh online probe must re-run (different key) and detect.
        env_offline["v"] = False
        assert mc.detect_audio_type("some/audio-model") == "snac"
        assert seen == [True, False]
        mc._audio_detection_cache.clear()


class TestEnvOfflineParsing:
    """_env_offline accepts the canonical truthy set (strip+lower, on/true/yes/1); it gates
    the requests.get fallback and the cache keys, so 'on' or ' 1 ' must still count as offline."""

    def test_truthy_values_recognized(self, monkeypatch):
        import utils.models.model_config as mc
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            for val in ("1", "true", "TRUE", "yes", "Yes", "on", "ON", " 1 ", " on ", "\ttrue\n"):
                monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
                monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
                monkeypatch.setenv(var, val)
                assert mc._env_offline() is True, f"{var}={val!r} should be offline"

    def test_falsy_values_not_offline(self, monkeypatch):
        import utils.models.model_config as mc

        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
        assert mc._env_offline() is False
        for val in ("", "0", "false", "no", "off", "2", "onn"):
            monkeypatch.setenv("HF_HUB_OFFLINE", val)
            assert mc._env_offline() is False, f"HF_HUB_OFFLINE={val!r} should not be offline"
