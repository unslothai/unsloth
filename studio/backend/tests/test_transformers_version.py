# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for transformers version detection with local checkpoint fallbacks."""

import json
import logging
import os
import pytest
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# The studio backend uses relative-style imports (``from utils.…``), so
# add the backend directory to *sys.path* if not already present.
# ---------------------------------------------------------------------------
import sys

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Stub the custom logger before import so ``from loggers import
# get_logger`` doesn't fail.
import types as _types

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

from utils.transformers_version import (
    _resolve_base_model,
    _check_tokenizer_config_needs_v5,
    _check_config_needs_510,
    _check_config_needs_550,
    _config_needs_530,
    _tier_from_name,
    _config_json_cache,
    _tokenizer_class_cache,
    _config_needs_510_cache,
    _config_needs_550_cache,
    needs_transformers_5,
    get_transformers_tier,
    activate_transformers_for_subprocess,
    _venv_dir_is_valid,
    _ensure_venv_dir,
)


@pytest.fixture(autouse = True)
def _capturable_logger(monkeypatch):
    """Make the ``caplog`` assertions independent of test collection order.

    The ``sys.modules.setdefault("loggers", ...)`` stub above only installs the
    stdlib-logger stub when ``loggers`` has not been imported yet. In a full
    backend pytest run another module (e.g. ``test_log_filter_no_truncation``,
    collected earlier) imports the real ``loggers`` first, so the stub is a
    no-op and ``transformers_version.logger`` ends up a structlog/stdout logger
    that ``caplog`` cannot see -- the tier/activation/install log assertions
    would then fail even though the line was emitted. Bind a real stdlib logger
    for the duration of each test so the module logs through ``logging`` and
    ``caplog`` captures them regardless of import order.
    """
    monkeypatch.setattr(
        "utils.transformers_version.logger",
        logging.getLogger("utils.transformers_version"),
    )


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
        # Falls through; does not return the self-referencing path.
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
        # Patch network call to avoid a real fetch.
        with patch(
            "utils.transformers_version._check_tokenizer_config_needs_v5",
            return_value = False,
        ):
            assert needs_transformers_5("meta-llama/Llama-3-8B") is False

    def test_local_checkpoint_resolved_via_config(self, tmp_path: Path):
        """Local checkpoint with config.json pointing to Qwen3.5 needs v5."""
        config_cfg = {"model_name": "Qwen/Qwen3.5-9B"}
        (tmp_path / "config.json").write_text(json.dumps(config_cfg))

        # needs_transformers_5 only does substring matching, so test the
        # full resolution chain via _resolve_base_model here.
        resolved = _resolve_base_model(str(tmp_path))
        assert needs_transformers_5(resolved) is True


# ---------------------------------------------------------------------------
# _check_config_needs_550 — config.json architecture/model_type check
# ---------------------------------------------------------------------------


class TestCheckConfigNeeds550:
    """Tests for _check_config_needs_550() local config.json checks."""

    def setup_method(self):
        _config_json_cache.clear()
        _config_needs_550_cache.clear()

    def test_gemma4_architecture(self, tmp_path: Path):
        """config.json with Gemma4ForConditionalGeneration should return True."""
        cfg = {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        assert _check_config_needs_550(str(tmp_path)) is True

    def test_gemma4_model_type_only(self, tmp_path: Path):
        """config.json with model_type=gemma4 (no architectures) should return True."""
        cfg = {"model_type": "gemma4"}
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        assert _check_config_needs_550(str(tmp_path)) is True

    def test_llama_architecture(self, tmp_path: Path):
        """config.json with LlamaForCausalLM should return False."""
        cfg = {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        assert _check_config_needs_550(str(tmp_path)) is False

    def test_no_config_json(self, tmp_path: Path):
        """Missing config.json should return False (fail-open)."""
        # Patch network call to avoid a real fetch.
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("no network")
            assert _check_config_needs_550(str(tmp_path)) is False

    def test_result_is_cached(self, tmp_path: Path):
        """Subsequent calls should use the cache."""
        cfg = {"architectures": ["Gemma4ForConditionalGeneration"]}
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        key = str(tmp_path)
        _check_config_needs_550(key)
        assert key in _config_needs_550_cache
        assert _config_needs_550_cache[key] is True

    def test_local_file_skips_network(self, tmp_path: Path):
        """When local config.json exists, no network request should be made."""
        cfg = {"architectures": ["LlamaForCausalLM"]}
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        with patch("urllib.request.urlopen") as mock_urlopen:
            _check_config_needs_550(str(tmp_path))
            mock_urlopen.assert_not_called()


# ---------------------------------------------------------------------------
# _check_config_needs_510 — config.json architecture/model_type check
# ---------------------------------------------------------------------------


class TestCheckConfigNeeds510:
    """Tests for _check_config_needs_510() local config.json checks."""

    def setup_method(self):
        _config_json_cache.clear()
        _config_needs_510_cache.clear()

    def test_gemma4_unified_architecture(self, tmp_path: Path):
        """config.json with Gemma4UnifiedForConditionalGeneration should return True."""
        cfg = {
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "model_type": "gemma4_unified",
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        assert _check_config_needs_510(str(tmp_path)) is True

    def test_gemma4_unified_model_type_only(self, tmp_path: Path):
        """config.json with model_type=gemma4_unified should return True."""
        cfg = {"model_type": "gemma4_unified"}
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        assert _check_config_needs_510(str(tmp_path)) is True

    def test_gemma4_unified_assistant_architecture(self, tmp_path: Path):
        """Assistant Gemma 4 Unified configs should return True."""
        cfg = {
            "architectures": ["Gemma4UnifiedAssistantForCausalLM"],
            "model_type": "gemma4_unified_assistant",
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        assert _check_config_needs_510(str(tmp_path)) is True

    def test_gemma4_unified_assistant_model_type_only(self, tmp_path: Path):
        """Assistant Gemma 4 Unified model_type should return True."""
        cfg = {"model_type": "gemma4_unified_assistant"}
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        assert _check_config_needs_510(str(tmp_path)) is True

    def test_gemma4_assistant_architecture(self, tmp_path: Path):
        """Assistant Gemma 4 configs should return True."""
        cfg = {
            "architectures": ["Gemma4AssistantForCausalLM"],
            "model_type": "gemma4_assistant",
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        assert _check_config_needs_510(str(tmp_path)) is True

    def test_gemma4_assistant_model_type_only(self, tmp_path: Path):
        """Assistant Gemma 4 model_type should return True."""
        cfg = {"model_type": "gemma4_assistant"}
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        assert _check_config_needs_510(str(tmp_path)) is True

    def test_gemma4_non_unified_returns_false(self, tmp_path: Path):
        """Older Gemma 4 config should stay on the 550 tier."""
        cfg = {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        assert _check_config_needs_510(str(tmp_path)) is False

    def test_no_config_json(self, tmp_path: Path):
        """Missing config.json should return False (fail-open)."""
        # Patch network call to avoid real fetch
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("no network")
            assert _check_config_needs_510(str(tmp_path)) is False

    def test_result_is_cached(self, tmp_path: Path):
        """Subsequent calls should use the cache."""
        cfg = {"architectures": ["Gemma4UnifiedForConditionalGeneration"]}
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        key = str(tmp_path)
        _check_config_needs_510(key)
        assert key in _config_needs_510_cache
        assert _config_needs_510_cache[key] is True

    def test_local_file_skips_network(self, tmp_path: Path):
        """When local config.json exists, no network request should be made."""
        cfg = {"architectures": ["LlamaForCausalLM"]}
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        with patch("urllib.request.urlopen") as mock_urlopen:
            _check_config_needs_510(str(tmp_path))
            mock_urlopen.assert_not_called()


# ---------------------------------------------------------------------------
# get_transformers_tier — tier detection
# ---------------------------------------------------------------------------


class TestGetTransformersTier:
    """Tests for get_transformers_tier() tiered version detection."""

    def setup_method(self):
        _tokenizer_class_cache.clear()
        _config_json_cache.clear()
        _config_needs_510_cache.clear()
        _config_needs_550_cache.clear()

    def test_gemma4_substring_returns_550(self):
        assert get_transformers_tier("google/gemma-4-E2B-it") == "550"

    def test_gemma4_12b_substring_returns_510(self):
        assert get_transformers_tier("unsloth/gemma-4-12b-it") == "510"

    def test_gemma4_assistant_substring_returns_510(self):
        assert get_transformers_tier("google/gemma-4-E2B-it-assistant") == "510"

    def test_gemma4_alt_substring_returns_550(self):
        assert get_transformers_tier("unsloth/gemma4-E4B-it") == "550"

    def test_gemma4_config_json_returns_550(self, tmp_path: Path):
        """Local checkpoint with Gemma4 architecture → 550."""
        cfg = {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        assert get_transformers_tier(str(tmp_path)) == "550"

    def test_gemma4_unified_config_json_returns_510(self, tmp_path: Path):
        """Local checkpoint with Gemma4 Unified architecture → 510."""
        cfg = {
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "model_type": "gemma4_unified",
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        assert get_transformers_tier(str(tmp_path)) == "510"

    def test_gemma4_assistant_config_json_returns_510(self, tmp_path: Path):
        """Local checkpoint with Gemma4 Assistant architecture → 510."""
        cfg = {
            "architectures": ["Gemma4AssistantForCausalLM"],
            "model_type": "gemma4_assistant",
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        assert get_transformers_tier(str(tmp_path)) == "510"

    def test_local_config_json_short_circuits_path_substrings(self, tmp_path: Path):
        """Local config.json should prevent false matches from parent directory names."""
        model_dir = tmp_path / "gemma-4-12b-experiment" / "llama-checkpoint"
        model_dir.mkdir(parents = True)
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["LlamaForCausalLM"],
                    "model_type": "llama",
                }
            )
        )
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "LlamaTokenizerFast"})
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            assert get_transformers_tier(str(model_dir)) == "default"
            mock_urlopen.assert_not_called()

    def test_remote_config_json_is_fetched_once_for_config_tiers(self):
        """510 and 550 slow-path checks should share one config.json fetch."""

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "architectures": ["Gemma4ForConditionalGeneration"],
                        "model_type": "gemma4",
                    }
                ).encode()

        with patch("urllib.request.urlopen", return_value = _Response()) as mock_urlopen:
            assert get_transformers_tier("org/no-fast-substring-model") == "550"

        assert mock_urlopen.call_count == 1

    def test_qwen35_returns_530(self):
        with (
            patch(
                "utils.transformers_version._check_config_needs_550",
                return_value = False,
            ),
            patch(
                "utils.transformers_version._check_config_needs_510",
                return_value = False,
            ),
        ):
            assert get_transformers_tier("Qwen/Qwen3.5-9B") == "530"

    def test_ministral_returns_530(self):
        with (
            patch(
                "utils.transformers_version._check_config_needs_550",
                return_value = False,
            ),
            patch(
                "utils.transformers_version._check_config_needs_510",
                return_value = False,
            ),
        ):
            assert get_transformers_tier("mistralai/Ministral-3-8B-Instruct-2512") == "530"

    def test_llama_returns_default(self):
        with (
            patch(
                "utils.transformers_version._check_config_needs_550",
                return_value = False,
            ),
            patch(
                "utils.transformers_version._check_config_needs_510",
                return_value = False,
            ),
            patch(
                "utils.transformers_version._check_tokenizer_config_needs_v5",
                return_value = False,
            ),
        ):
            assert get_transformers_tier("meta-llama/Llama-3-8B") == "default"

    def test_550_checked_before_530(self):
        """5.5.0 is checked before 5.3.0 - a model matching both gets 550."""
        assert get_transformers_tier("gemma-4-model") == "550"

    # ---- issue #6103: the tier decision must be traceable in the logs ----

    def test_tier_550_selection_is_logged(self, caplog):
        caplog.set_level(logging.INFO)
        assert get_transformers_tier("google/gemma-4-E2B-it") == "550"
        text = " ".join(r.getMessage() for r in caplog.records).lower()
        assert "550" in text, f"tier selection not logged: {text!r}"
        assert "gemma-4-e2b-it" in text, f"tier log omits the model: {text!r}"

    def test_tier_530_selection_is_logged(self, caplog):
        caplog.set_level(logging.INFO)
        with patch(
            "utils.transformers_version._check_config_needs_550",
            return_value = False,
        ):
            assert get_transformers_tier("Qwen/Qwen3.5-9B") == "530"
        text = " ".join(r.getMessage() for r in caplog.records).lower()
        assert "530" in text, f"tier selection not logged: {text!r}"
        assert "qwen3.5-9b" in text, f"tier log omits the model: {text!r}"

    def test_tier_default_selection_is_logged(self, caplog):
        caplog.set_level(logging.INFO)
        with (
            patch(
                "utils.transformers_version._check_config_needs_510",
                return_value = False,
            ),
            patch(
                "utils.transformers_version._check_config_needs_550",
                return_value = False,
            ),
            patch(
                "utils.transformers_version._check_tokenizer_config_needs_v5",
                return_value = False,
            ),
        ):
            assert get_transformers_tier("meta-llama/Llama-3-8B") == "default"
        text = " ".join(r.getMessage() for r in caplog.records).lower()
        assert "default" in text, f"tier selection not logged: {text!r}"

    def test_local_config_json_selection_is_logged(self, tmp_path: Path, caplog):
        cfg = {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4"}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        caplog.set_level(logging.INFO)
        assert get_transformers_tier(str(tmp_path)) == "550"
        text = " ".join(r.getMessage() for r in caplog.records).lower()
        assert "550" in text and "local config.json" in text, f"local tier not logged: {text!r}"

    def test_needs_transformers_5_compat(self):
        """needs_transformers_5 should return True for 510, 530, and 550 models."""
        assert needs_transformers_5("unsloth/gemma-4-12b-it") is True
        assert needs_transformers_5("google/gemma-4-E2B-it") is True
        with (
            patch(
                "utils.transformers_version._check_config_needs_550",
                return_value = False,
            ),
            patch(
                "utils.transformers_version._check_config_needs_510",
                return_value = False,
            ),
        ):
            assert needs_transformers_5("Qwen/Qwen3.5-9B") is True
        with (
            patch(
                "utils.transformers_version._check_config_needs_550",
                return_value = False,
            ),
            patch(
                "utils.transformers_version._check_config_needs_510",
                return_value = False,
            ),
            patch(
                "utils.transformers_version._check_tokenizer_config_needs_v5",
                return_value = False,
            ),
        ):
            assert needs_transformers_5("meta-llama/Llama-3-8B") is False


# ---------------------------------------------------------------------------
# activate_transformers_for_subprocess — issue #6103
# The early log must make clear it only prepends to sys.path; the real
# confirmation comes later from "Subprocess loaded transformers X.X.X".
# ---------------------------------------------------------------------------


class TestActivateLoggingClarity:
    """issue #6103: 'Activated transformers' was misleading (path-prepend only)."""

    def _snapshot_env(self):
        return list(sys.path), os.environ.get("PYTHONPATH")

    def _restore_env(self, snapshot):
        saved_path, saved_pp = snapshot
        sys.path[:] = saved_path
        if saved_pp is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = saved_pp

    def test_activate_550_log_clarifies_path_prepend_only(self, caplog):
        caplog.set_level(logging.INFO)
        snap = self._snapshot_env()
        try:
            with (
                patch(
                    "utils.transformers_version._resolve_base_model",
                    side_effect = lambda m: m,
                ),
                patch(
                    "utils.transformers_version.get_transformers_tier",
                    return_value = "550",
                ),
                patch(
                    "utils.transformers_version._ensure_venv_t5_550_exists",
                    return_value = True,
                ),
            ):
                activate_transformers_for_subprocess("google/gemma-4-E2B-it")
        finally:
            self._restore_env(snap)

        text = " ".join(r.getMessage() for r in caplog.records).lower()
        assert "5.5.0" in text, f"version not logged: {text!r}"
        # Must signal this is only a sys.path manipulation, not a confirmed import.
        assert (
            "sys.path" in text or "path only" in text
        ), f"early activation log does not clarify it is path-prepend only: {text!r}"

    def test_activate_530_log_clarifies_path_prepend_only(self, caplog):
        caplog.set_level(logging.INFO)
        snap = self._snapshot_env()
        try:
            with (
                patch(
                    "utils.transformers_version._resolve_base_model",
                    side_effect = lambda m: m,
                ),
                patch(
                    "utils.transformers_version.get_transformers_tier",
                    return_value = "530",
                ),
                patch(
                    "utils.transformers_version._ensure_venv_t5_530_exists",
                    return_value = True,
                ),
            ):
                activate_transformers_for_subprocess("Qwen/Qwen3.5-9B")
        finally:
            self._restore_env(snap)

        text = " ".join(r.getMessage() for r in caplog.records).lower()
        assert "5.3.0" in text, f"version not logged: {text!r}"
        assert (
            "sys.path" in text or "path only" in text
        ), f"early activation log does not clarify it is path-prepend only: {text!r}"


# ---------------------------------------------------------------------------
# _venv_dir_is_valid — issue #6103
# A version mismatch triggers a full wipe + reinstall, so it must be logged
# at WARNING (not INFO) so the reinstall is visible.
# ---------------------------------------------------------------------------


class TestVenvDirIsValidLogging:
    def _make_venv(self, venv_dir: Path, pkg: str, version: str):
        """Create a fake target-dir install of *pkg* at *version*."""
        (venv_dir / pkg).mkdir(parents = True)
        di = venv_dir / f"{pkg}-{version}.dist-info"
        di.mkdir()
        (di / "METADATA").write_text(f"Name: {pkg}\nVersion: {version}\n")

    def test_version_mismatch_logged_at_warning(self, tmp_path: Path, caplog):
        venv_dir = tmp_path / "venv"
        self._make_venv(venv_dir, "transformers", "5.0.0")  # wrong version

        caplog.set_level(logging.INFO)
        result = _venv_dir_is_valid(str(venv_dir), ("transformers==5.3.0",))

        assert result is False
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, (
            "version mismatch must be logged at WARNING; got: "
            f"{[(r.levelname, r.getMessage()) for r in caplog.records]!r}"
        )
        joined = " ".join(r.getMessage() for r in warnings)
        assert (
            "5.0.0" in joined and "5.3.0" in joined
        ), f"mismatch log omits the versions: {joined!r}"

    def test_correct_version_does_not_warn(self, tmp_path: Path, caplog):
        venv_dir = tmp_path / "venv"
        self._make_venv(venv_dir, "transformers", "5.3.0")  # correct version

        caplog.set_level(logging.INFO)
        result = _venv_dir_is_valid(str(venv_dir), ("transformers==5.3.0",))

        assert result is True
        assert not [
            r for r in caplog.records if r.levelno >= logging.WARNING
        ], "no warning expected when the installed version matches"


# ---------------------------------------------------------------------------
# _ensure_venv_dir — issue #6103
# A slow runtime install must log each package as it starts, otherwise it
# looks like a hang.
# ---------------------------------------------------------------------------


class TestEnsureVenvDirProgressLogging:
    def test_logs_each_package_with_progress(self, tmp_path: Path, caplog):
        installed = []
        caplog.set_level(logging.INFO)
        with (
            patch(
                "utils.transformers_version._venv_dir_is_valid",
                return_value = False,
            ),
            patch(
                "utils.transformers_version._install_to_dir",
                side_effect = lambda pkg, d: (installed.append(pkg), True)[1],
            ),
        ):
            ok = _ensure_venv_dir(
                str(tmp_path / "venv"),
                ("transformers==5.3.0", "tokenizers==0.21.0"),
                "transformers 5.3.0",
            )

        assert ok is True
        assert installed == ["transformers==5.3.0", "tokenizers==0.21.0"]

        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "transformers==5.3.0" in msgs, f"first package not logged: {msgs!r}"
        assert "tokenizers==0.21.0" in msgs, f"second package not logged: {msgs!r}"
        # progress counter present so a slow install is not mistaken for a hang
        assert "1/2" in msgs and "2/2" in msgs, f"progress count missing: {msgs!r}"

    def test_no_install_logging_when_venv_already_valid(self, tmp_path: Path, caplog):
        caplog.set_level(logging.INFO)
        with (
            patch(
                "utils.transformers_version._venv_dir_is_valid",
                return_value = True,
            ),
            patch(
                "utils.transformers_version._install_to_dir",
            ) as mock_install,
        ):
            ok = _ensure_venv_dir(
                str(tmp_path / "venv"),
                ("transformers==5.3.0",),
                "transformers 5.3.0",
            )

        assert ok is True
        mock_install.assert_not_called()
        assert "Installing" not in " ".join(r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# _tier_from_name — shared name-based detection helper
# ---------------------------------------------------------------------------


class TestTierFromName:
    """Unit tests for _tier_from_name(), which backs both the fast substring
    path and the config _name_or_path fallback."""

    def test_returns_none_for_unknown(self):
        assert _tier_from_name("meta-llama/Llama-3-8B") is None

    def test_gemma4_returns_550(self):
        tier, _ = _tier_from_name("google/gemma-4-E2B-it")
        assert tier == "550"

    def test_gemma4_assistant_returns_510(self):
        tier, match = _tier_from_name("google/gemma-4-E2B-it-assistant")
        assert tier == "510"
        assert "assistant" in match

    def test_gemma4_12b_returns_510(self):
        tier, _ = _tier_from_name("unsloth/gemma-4-12b-it")
        assert tier == "510"

    def test_qwen35_returns_530(self):
        tier, match = _tier_from_name("Qwen/Qwen3.5-7B")
        assert tier == "530"
        assert "qwen3.5" in match

    def test_ministral3_returns_530(self):
        # The existing substring "ministral-3-" matches the 2512 naming style.
        tier, _ = _tier_from_name("mistralai/Ministral-3-8B-Instruct-2512")
        assert tier == "530"

    def test_qwen3_moe_substring_returns_530(self):
        tier, _ = _tier_from_name("Qwen/Qwen3-30B-A3B-Instruct-2507")
        assert tier == "530"

    def test_510_beats_550(self):
        """gemma-4-12b matches 510 (checked first), not 550."""
        tier, _ = _tier_from_name("google/gemma-4-12b-it")
        assert tier == "510"

    def test_550_beats_530(self):
        """gemma-4 matches 550, not 530."""
        tier, _ = _tier_from_name("gemma-4-model")
        assert tier == "550"


# ---------------------------------------------------------------------------
# Local-folder tier detection via config.json
#
# When a local checkpoint's config.json architecture/model_type matches a known
# sidecar set, that's the authoritative answer.  When it doesn't match (unknown
# or future family), the HF model ID from _name_or_path / model_name in the
# config is run through the same name-based rules so renamed folders are still
# routed correctly without introducing path false-positives.
# ---------------------------------------------------------------------------


class TestLocalConfig530Tier:
    def setup_method(self):
        _config_json_cache.clear()
        _tokenizer_class_cache.clear()

    # --- config-set matches -------------------------------------------------

    def test_config_needs_530_qwen3_5_model_type(self):
        assert _config_needs_530({"model_type": "qwen3_5"}) is True

    def test_config_needs_530_qwen3_5_conditional_generation(self):
        assert _config_needs_530({"architectures": ["Qwen3_5ForConditionalGeneration"]}) is True

    def test_config_needs_530_qwen3_moe(self):
        assert _config_needs_530({"model_type": "qwen3_moe"}) is True

    def test_config_needs_530_glm4_moe_lite(self):
        assert _config_needs_530({"model_type": "glm4_moe_lite"}) is True

    def test_config_needs_530_lfm2_vl(self):
        assert _config_needs_530({"model_type": "lfm2_vl"}) is True

    def test_config_needs_530_plain_qwen3_is_false(self):
        """Regular Qwen3 (non-MoE, non-3.5) must not be promoted to 5.3.0."""
        assert _config_needs_530({"model_type": "qwen3"}) is False

    def test_tier_local_qwen35_config_selects_530(self, tmp_path: Path):
        """Reported case: a local Qwen3.5 folder routes to 530 via config.json."""
        d = tmp_path / "Qwen3.5-2B"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
        assert get_transformers_tier(str(d)) == "530"

    def test_tier_local_qwen3_moe_config_selects_530(self, tmp_path: Path):
        """Local Qwen3 MoE checkpoint routes to 530 via config.json."""
        d = tmp_path / "my-qwen3-moe"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"]})
        )
        assert get_transformers_tier(str(d)) == "530"

    def test_tier_local_glm4_moe_lite_config_selects_530(self, tmp_path: Path):
        """Local GLM-4.7-Flash checkpoint routes to 530 via config.json."""
        d = tmp_path / "my-glm-model"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({"model_type": "glm4_moe_lite", "architectures": ["Glm4MoeLiteForCausalLM"]})
        )
        assert get_transformers_tier(str(d)) == "530"

    def test_tier_local_lfm2_vl_config_selects_530(self, tmp_path: Path):
        """Local LFM2.5-VL checkpoint routes to 530 via config.json."""
        d = tmp_path / "my-liquid-model"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {"model_type": "lfm2_vl", "architectures": ["Lfm2VlForConditionalGeneration"]}
            )
        )
        assert get_transformers_tier(str(d)) == "530"

    # --- _name_or_path fallback ---------------------------------------------

    def test_renamed_folder_falls_back_to_hf_id_in_config(self, tmp_path: Path):
        """A renamed local folder with an unrecognised model_type but a known
        HF ID in _name_or_path still routes to the correct tier."""
        d = tmp_path / "my-custom-name"
        d.mkdir()
        # Simulate a future/unknown model_type; the HF ID carries the tier signal.
        (d / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "future_unknown_type",
                    "_name_or_path": "Qwen/Qwen3.5-7B",
                }
            )
        )
        assert get_transformers_tier(str(d)) == "530"

    def test_hf_id_fallback_respects_550_tier(self, tmp_path: Path):
        """_name_or_path pointing to a Gemma-4 HF ID routes to 550."""
        d = tmp_path / "renamed-gemma"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "future_unknown_type",
                    "_name_or_path": "google/gemma-4-E2B-it",
                }
            )
        )
        assert get_transformers_tier(str(d)) == "550"

    def test_hf_id_fallback_skipped_when_same_as_path(self, tmp_path: Path):
        """If _name_or_path equals the model path, skip the name fallback to
        avoid false positives from self-referencing configs."""
        d = tmp_path / "qwen3.5-experiment"
        d.mkdir()
        # _name_or_path is the local path itself (e.g. saved via save_pretrained)
        (d / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "_name_or_path": str(d),
                }
            )
        )
        with patch(
            "utils.transformers_version._check_tokenizer_config_needs_v5", return_value = False
        ):
            # "qwen3.5" is in the path but config says llama and _name_or_path
            # is self-referencing — must not be promoted to 530.
            assert get_transformers_tier(str(d)) == "default"

    # --- false-positive guard -----------------------------------------------

    def test_tier_local_plain_model_still_default(self, tmp_path: Path):
        """A local non-5.x checkpoint returns default; the directory-name
        false-positive guard is preserved."""
        d = tmp_path / "checkpoint-1000"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama"})
        )
        with patch(
            "utils.transformers_version._check_tokenizer_config_needs_v5",
            return_value = False,
        ):
            assert get_transformers_tier(str(d)) == "default"
