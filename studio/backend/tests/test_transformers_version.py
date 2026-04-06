# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for transformers version detection with local checkpoint fallbacks."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# We need to be able to import the module under test.  The studio backend
# uses relative-style imports (``from utils.…``), so we add the backend
# directory to *sys.path* if it is not already there.
# ---------------------------------------------------------------------------
import sys

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Stub the custom logger before importing the module under test so it
# doesn't fail on the ``from loggers import get_logger`` line.
import types as _types

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

from utils.transformers_version import (
    _resolve_base_model,
    _check_tokenizer_config_needs_v5,
    _tokenizer_class_cache,
    needs_transformers_5,
)
from utils.models import model_config


# ---------------------------------------------------------------------------
# _resolve_base_model — config.json fallback
# ---------------------------------------------------------------------------


class TestResolveBaseModel:
    """Tests for _resolve_base_model() local config fallbacks."""

    def test_adapter_config_takes_priority(self, tmp_path: Path):
        """adapter_config.json should be preferred over config.json."""
        adapter_cfg = {"base_model_name_or_path": "meta-llama/Llama-3-8B"}
        config_cfg = {"_name_or_path": "different/model"}
        (tmp_path / "adapter_config.json").write_text(json.dumps(adapter_cfg))
        (tmp_path / "config.json").write_text(json.dumps(config_cfg))

        result = _resolve_base_model(str(tmp_path))
        assert result == "meta-llama/Llama-3-8B"

    def test_config_json_fallback_model_name(self, tmp_path: Path):
        """config.json model_name should resolve when no adapter_config."""
        config_cfg = {"model_name": "Qwen/Qwen3.5-9B"}
        (tmp_path / "config.json").write_text(json.dumps(config_cfg))

        result = _resolve_base_model(str(tmp_path))
        assert result == "Qwen/Qwen3.5-9B"

    def test_config_json_fallback_name_or_path(self, tmp_path: Path):
        """config.json _name_or_path should resolve as secondary fallback."""
        config_cfg = {"_name_or_path": "Qwen/Qwen3.5-9B"}
        (tmp_path / "config.json").write_text(json.dumps(config_cfg))

        result = _resolve_base_model(str(tmp_path))
        assert result == "Qwen/Qwen3.5-9B"

    def test_model_name_takes_priority_over_name_or_path(self, tmp_path: Path):
        """model_name should be preferred over _name_or_path."""
        config_cfg = {
            "model_name": "Qwen/Qwen3.5-9B",
            "_name_or_path": "some/other-model",
        }
        (tmp_path / "config.json").write_text(json.dumps(config_cfg))

        result = _resolve_base_model(str(tmp_path))
        assert result == "Qwen/Qwen3.5-9B"

    def test_config_json_skips_self_referencing(self, tmp_path: Path):
        """config.json should be ignored if model_name == the checkpoint path."""
        config_cfg = {"model_name": str(tmp_path)}
        (tmp_path / "config.json").write_text(json.dumps(config_cfg))

        result = _resolve_base_model(str(tmp_path))
        # Should fall through, not return the self-referencing path
        assert result == str(tmp_path)

    def test_no_config_files(self, tmp_path: Path):
        """Returns original name when no config files are present."""
        result = _resolve_base_model(str(tmp_path))
        assert result == str(tmp_path)

    def test_plain_hf_id_passthrough(self):
        """Plain HuggingFace model IDs pass through unchanged."""
        result = _resolve_base_model("meta-llama/Llama-3-8B")
        assert result == "meta-llama/Llama-3-8B"


# ---------------------------------------------------------------------------
# _check_tokenizer_config_needs_v5 — local file check
# ---------------------------------------------------------------------------


class TestCheckTokenizerConfigNeedsV5:
    """Tests for local tokenizer_config.json fallback."""

    def setup_method(self):
        _tokenizer_class_cache.clear()

    def test_local_tokenizer_config_v5(self, tmp_path: Path):
        """Local tokenizer_config.json with v5 tokenizer should return True."""
        tc = {"tokenizer_class": "TokenizersBackend"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))

        result = _check_tokenizer_config_needs_v5(str(tmp_path))
        assert result is True

    def test_local_tokenizer_config_v4(self, tmp_path: Path):
        """Local tokenizer_config.json with standard tokenizer should return False."""
        tc = {"tokenizer_class": "LlamaTokenizerFast"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))

        result = _check_tokenizer_config_needs_v5(str(tmp_path))
        assert result is False

    def test_local_file_skips_network(self, tmp_path: Path):
        """When local file exists, no network request should be made."""
        tc = {"tokenizer_class": "LlamaTokenizerFast"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))

        with patch("urllib.request.urlopen") as mock_urlopen:
            result = _check_tokenizer_config_needs_v5(str(tmp_path))
            mock_urlopen.assert_not_called()
        assert result is False

    def test_result_is_cached(self, tmp_path: Path):
        """Subsequent calls should use the cache."""
        tc = {"tokenizer_class": "TokenizersBackend"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))

        key = str(tmp_path)
        _check_tokenizer_config_needs_v5(key)
        assert key in _tokenizer_class_cache
        assert _tokenizer_class_cache[key] is True


# ---------------------------------------------------------------------------
# needs_transformers_5 — integration-level
# ---------------------------------------------------------------------------


class TestNeedsTransformers5:
    """Integration tests for the top-level needs_transformers_5() function."""

    def setup_method(self):
        _tokenizer_class_cache.clear()

    def test_qwen35_substring(self):
        assert needs_transformers_5("Qwen/Qwen3.5-9B") is True

    def test_qwen3_30b_a3b_substring(self):
        assert needs_transformers_5("Qwen/Qwen3-30B-A3B-Instruct-2507") is True

    def test_ministral_substring(self):
        assert needs_transformers_5("mistralai/Ministral-3-8B-Instruct-2512") is True

    def test_llama_does_not_need_v5(self):
        """Standard models should not trigger v5."""
        # Patch network call to avoid real fetch
        with patch(
            "utils.transformers_version._check_tokenizer_config_needs_v5",
            return_value = False,
        ):
            assert needs_transformers_5("meta-llama/Llama-3-8B") is False

    def test_local_checkpoint_resolved_via_config(self, tmp_path: Path):
        """A local checkpoint with config.json pointing to Qwen3.5 should need v5."""
        config_cfg = {"model_name": "Qwen/Qwen3.5-9B"}
        (tmp_path / "config.json").write_text(json.dumps(config_cfg))

        # _resolve_base_model is called by ensure_transformers_version,
        # but needs_transformers_5 just does substring matching.
        # We test the full resolution chain here:
        resolved = _resolve_base_model(str(tmp_path))
        assert needs_transformers_5(resolved) is True


class TestVisionModelDetection:
    def test_is_vlm_config_accepts_transformers_objects(self):
        config = _types.SimpleNamespace(
            model_type = "qwen3_5",
            architectures = ["Qwen3_5ForConditionalGeneration"],
            vision_config = {},
        )
        assert model_config._is_vlm_config(config) is True

    def test_is_vlm_config_accepts_raw_config_dicts(self):
        config = {
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"],
            "vision_config": {},
        }
        assert model_config._is_vlm_config(config) is True

    def test_is_vlm_config_rejects_audio_only_conditional_generation(self):
        config = {
            "model_type": "whisper",
            "architectures": ["WhisperForConditionalGeneration"],
        }
        assert model_config._is_vlm_config(config) is False

    def test_is_vision_model_falls_back_to_raw_metadata_for_t5_models(self):
        with (
            patch("utils.transformers_version.needs_transformers_5", return_value = True),
            patch(
                "utils.models.model_config.load_model_config",
                side_effect = RuntimeError("direct load failed"),
            ),
            patch(
                "utils.models.model_config._is_vision_model_subprocess",
                return_value = None,
            ),
            patch(
                "utils.models.model_config._load_model_config_metadata",
                return_value = (
                    {
                        "model_type": "qwen3_5",
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                        "vision_config": {},
                    },
                    None,
                ),
            ),
        ):
            assert model_config.is_vision_model("unsloth/Qwen3.5-4B") is True
