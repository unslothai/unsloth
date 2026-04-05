# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for is_vision_model() caching behaviour.

The vision detection cache (``_vision_detection_cache``) mirrors the existing
``_audio_detection_cache`` pattern used by ``detect_audio_type()``.  These
tests verify that:

* Repeated calls for the same model hit the cache (no redundant work).
* Different models each trigger their own detection.
* Both True and False results are cached.
* The subprocess path (transformers 5.x models) is also cached.
* Exceptions that fall back to False are cached.
"""

import sys
import types as _types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# sys.path + logger stub — same pattern as the rest of the test suite
# ---------------------------------------------------------------------------
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

from utils.models.model_config import (
    is_vision_model,
    _is_vision_model_uncached,
    _vision_detection_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse = True)
def _clear_vision_cache():
    """Ensure every test starts with a fresh cache."""
    _vision_detection_cache.clear()
    yield
    _vision_detection_cache.clear()


def _make_config(**attrs):
    """Return a lightweight mock config with the given attributes."""
    cfg = MagicMock()
    # Remove all default MagicMock attributes so hasattr checks are explicit
    cfg.configure_mock(**attrs)
    # Only expose the attrs we explicitly set
    real_attrs = set(attrs.keys())
    original_hasattr = hasattr

    def _controlled_hasattr(obj, name):
        if obj is cfg:
            return name in real_attrs
        return original_hasattr(obj, name)

    return cfg, _controlled_hasattr


# ---------------------------------------------------------------------------
# Cache hit / miss tests
# ---------------------------------------------------------------------------


class TestVisionCacheHitMiss:
    """Verify the cache prevents redundant detection calls."""

    @patch("utils.models.model_config._is_vision_model_uncached", return_value = True)
    def test_second_call_uses_cache(self, mock_uncached):
        """Calling is_vision_model() twice for the same model should invoke
        the uncached function only once."""
        assert is_vision_model("org/my-vlm") is True
        assert is_vision_model("org/my-vlm") is True
        mock_uncached.assert_called_once_with("org/my-vlm", None)

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
        assert _vision_detection_cache["org/text-only"] is False


# ---------------------------------------------------------------------------
# Subprocess path (transformers 5.x) caching
# ---------------------------------------------------------------------------


class TestVisionCacheSubprocessPath:
    """Models needing transformers 5.x go through _is_vision_model_subprocess.
    The cache should prevent the subprocess from being spawned more than once
    per model per process."""

    @patch("utils.models.model_config._is_vision_model_subprocess", return_value = True)
    @patch("utils.transformers_version.needs_transformers_5", return_value = True)
    def test_subprocess_called_once_with_cache(self, mock_needs_t5, mock_subprocess):
        """Subprocess should only fire on the first call; second is cached."""
        # First call: goes through uncached → subprocess
        assert is_vision_model("unsloth/Qwen3.5-2B") is True
        # Second call: cache hit, no subprocess
        assert is_vision_model("unsloth/Qwen3.5-2B") is True

        mock_subprocess.assert_called_once()
        assert _vision_detection_cache["unsloth/Qwen3.5-2B"] is True


# ---------------------------------------------------------------------------
# Exception handling — cache the False fallback
# ---------------------------------------------------------------------------


class TestVisionCacheOnException:
    """When detection raises an exception, the function returns False.
    That False must be cached so subsequent calls don't retry and fail again."""

    @patch(
        "utils.models.model_config._is_vision_model_uncached",
        side_effect = [False],
    )
    def test_exception_result_cached(self, mock_uncached):
        """After an exception-triggered False, the cache should serve False."""
        # The uncached function returns False (simulating the except branch)
        assert is_vision_model("broken/model") is False
        # Second call should not invoke uncached again
        assert is_vision_model("broken/model") is False
        mock_uncached.assert_called_once()


# ---------------------------------------------------------------------------
# Direct detection path (non-transformers-5 models) caching
# ---------------------------------------------------------------------------


class TestVisionCacheDirectPath:
    """For models that do NOT need transformers 5.x, the detection goes through
    load_model_config directly. The cache must work the same way."""

    @patch("utils.transformers_version.needs_transformers_5", return_value = False)
    @patch("utils.models.model_config.load_model_config")
    def test_direct_vlm_detection_cached(self, mock_load_config, mock_needs_t5):
        """A standard VLM detected via vision_config should be cached."""
        cfg = MagicMock()
        cfg.model_type = "gemma3"
        cfg.architectures = ["Gemma3ForConditionalGeneration"]
        mock_load_config.return_value = cfg

        assert is_vision_model("google/gemma-3-4b-it") is True
        assert is_vision_model("google/gemma-3-4b-it") is True
        # load_model_config should only be called once
        mock_load_config.assert_called_once()

    @patch("utils.transformers_version.needs_transformers_5", return_value = False)
    @patch("utils.models.model_config.load_model_config")
    def test_direct_non_vlm_detection_cached(self, mock_load_config, mock_needs_t5):
        """A standard text model (no VLM indicators) should cache False."""
        cfg = MagicMock(spec = [])  # spec=[] means no attributes at all
        cfg.model_type = "llama"
        cfg.architectures = ["LlamaForCausalLM"]
        mock_load_config.return_value = cfg

        # LlamaForCausalLM doesn't end with VLM suffixes, no vision_config, etc.
        assert is_vision_model("meta-llama/Llama-3-8B") is False
        assert is_vision_model("meta-llama/Llama-3-8B") is False
        mock_load_config.assert_called_once()

    @patch("utils.transformers_version.needs_transformers_5", return_value = False)
    @patch("utils.models.model_config.load_model_config")
    def test_vision_config_attr_detected_and_cached(
        self, mock_load_config, mock_needs_t5
    ):
        """Models with vision_config (LLaVA, Qwen2-VL, etc.) should be cached as True."""
        cfg = MagicMock()
        cfg.model_type = "qwen2_vl"
        cfg.architectures = ["Qwen2VLForCausalLM"]  # Doesn't match VLM suffixes
        cfg.vision_config = {"hidden_size": 1024}
        mock_load_config.return_value = cfg

        assert is_vision_model("Qwen/Qwen2-VL-7B") is True
        assert is_vision_model("Qwen/Qwen2-VL-7B") is True
        mock_load_config.assert_called_once()

    @patch("utils.transformers_version.needs_transformers_5", return_value = False)
    @patch("utils.models.model_config.load_model_config")
    def test_audio_model_excluded_and_cached(self, mock_load_config, mock_needs_t5):
        """Audio-only models (csm, whisper) with ForConditionalGeneration
        should be excluded from VLM detection and cached as False."""
        cfg = MagicMock()
        cfg.model_type = "whisper"
        cfg.architectures = ["WhisperForConditionalGeneration"]
        mock_load_config.return_value = cfg

        assert is_vision_model("openai/whisper-large-v3") is False
        assert is_vision_model("openai/whisper-large-v3") is False
        mock_load_config.assert_called_once()


# ---------------------------------------------------------------------------
# hf_token handling
# ---------------------------------------------------------------------------


class TestVisionCacheTokenHandling:
    """The cache is keyed on model_name only (same as _audio_detection_cache).
    Different tokens for the same model should use the cached result."""

    @patch("utils.models.model_config._is_vision_model_uncached", return_value = True)
    def test_same_model_different_tokens_uses_cache(self, mock_uncached):
        """Second call with a different token should still hit cache."""
        assert is_vision_model("gated/model", hf_token = "token-a") is True
        assert is_vision_model("gated/model", hf_token = "token-b") is True
        mock_uncached.assert_called_once_with("gated/model", "token-a")
