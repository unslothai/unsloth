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
        assert _vision_detection_cache[("org/text-only", None)] is False


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
        assert _vision_detection_cache[("unsloth/Qwen3.5-2B", None)] is True


# ---------------------------------------------------------------------------
# Exception handling — cache the False fallback
# ---------------------------------------------------------------------------


class TestVisionCacheOnException:
    """When detection raises an exception, _is_vision_model_uncached
    distinguishes permanent failures (cached as False) from transient
    failures (returned as None, not cached so the next call can retry).
    Verify both contracts."""

    @patch(
        "utils.models.model_config.load_model_config",
        side_effect = ValueError("bad config"),
    )
    @patch("utils.transformers_version.needs_transformers_5", return_value = False)
    def test_permanent_exception_result_cached(self, mock_needs_t5, mock_load_config):
        """A permanent failure (ValueError / RepositoryNotFoundError /
        GatedRepoError / JSONDecodeError) should be caught, return False,
        and that False should be cached so subsequent calls don't retry.

        ValueError is used here because it's the simplest of the
        code-path's cacheable exception types and does not require an
        import of huggingface_hub errors (whose module path varies
        across versions)."""
        # First call: load_model_config raises -> except branch -> False.
        assert is_vision_model("broken/model") is False
        # Second call: cache hit, load_model_config not called again.
        assert is_vision_model("broken/model") is False
        mock_load_config.assert_called_once()

    @patch(
        "utils.models.model_config.load_model_config",
        side_effect = OSError("network down"),
    )
    @patch("utils.transformers_version.needs_transformers_5", return_value = False)
    def test_transient_exception_not_cached(self, mock_needs_t5, mock_load_config):
        """A transient failure (OSError, timeouts) should return None from
        _is_vision_model_uncached, surface as False to the caller, and
        NOT be cached, so the next call retries detection.  This matches
        the documented behaviour on _vision_detection_cache:
        'transient failures (network errors, timeouts) are NOT cached so
        they can be retried.'"""
        # First call: load_model_config raises OSError -> uncached None
        # -> caller returns False without caching.
        assert is_vision_model("broken/model") is False
        # Second call: cache miss again, load_model_config called a
        # second time.
        assert is_vision_model("broken/model") is False
        assert mock_load_config.call_count == 2


# ---------------------------------------------------------------------------
# Direct detection path (non-transformers-5 models) caching
# ---------------------------------------------------------------------------


class TestVisionCacheDirectPath:
    """For models that do NOT need transformers 5.x, the detection goes through
    load_model_config directly. The cache must work the same way."""

    @patch("utils.transformers_version.needs_transformers_5", return_value = False)
    @patch("utils.models.model_config.load_model_config")
    def test_direct_vlm_detection_cached(self, mock_load_config, mock_needs_t5):
        """A standard VLM detected via architecture suffix should be cached."""
        cfg = MagicMock(spec = [])  # strict: only explicitly set attrs exist
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
        cfg = MagicMock(spec = [])  # strict: only explicitly set attrs exist
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
        cfg = MagicMock(spec = [])  # strict: only explicitly set attrs exist
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
    """The cache is keyed on (model_name, hf_token).
    Different tokens for the same model should trigger separate detections
    to handle gated models correctly."""

    @patch("utils.models.model_config._is_vision_model_uncached", return_value = True)
    def test_different_tokens_trigger_new_detection(self, mock_uncached):
        """Calls with different tokens should trigger separate detections to
        handle gated models correctly (e.g. unauthenticated probe → False,
        then authenticated call should re-check)."""
        assert is_vision_model("gated/model", hf_token = "token-a") is True
        assert is_vision_model("gated/model", hf_token = "token-b") is True
        assert mock_uncached.call_count == 2

    @patch("utils.models.model_config._is_vision_model_uncached", return_value = True)
    def test_same_token_uses_cache(self, mock_uncached):
        """Repeated calls with identical model + token should hit cache."""
        assert is_vision_model("gated/model", hf_token = "token-a") is True
        assert is_vision_model("gated/model", hf_token = "token-a") is True
        mock_uncached.assert_called_once()
