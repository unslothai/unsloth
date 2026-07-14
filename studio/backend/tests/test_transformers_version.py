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
    _is_lora_adapter_dir,
    _has_adapter_weights,
    _remote_lora_base,
    _check_tokenizer_config_needs_v5,
    _check_config_needs_510,
    _check_config_needs_530,
    _check_config_needs_550,
    _config_needs_510,
    _config_needs_530,
    _norm_separators,
    _tier_from_name,
    _looks_like_hf_id,
    _nemotron_h_needs_mlp_support,
    _config_json_from_hf_cache,
    _load_config_json,
    _higher_tier,
    _config_json_cache,
    _tokenizer_class_cache,
    _config_needs_510_cache,
    _config_needs_530_cache,
    _config_needs_550_cache,
    _probe_tier_cache,
    _probe_tier,
    _stderr_is_transient,
    needs_transformers_5,
    get_transformers_tier,
    activate_transformers_for_subprocess,
    _venv_dir_is_valid,
    _ensure_venv_dir,
    hf_endpoint_unreachable,
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

    def test_non_string_base_does_not_crash(self, tmp_path: Path):
        """A malformed config (list/dict for model_name) must not raise."""
        config_cfg = {"model_name": ["x"], "_name_or_path": "Qwen/Qwen3.5-9B"}
        (tmp_path / "config.json").write_text(json.dumps(config_cfg))

        # Skips the non-string model_name and falls through to _name_or_path.
        assert _resolve_base_model(str(tmp_path)) == "Qwen/Qwen3.5-9B"

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


class TestRemoteLoraBase:
    """_remote_lora_base reads a remote adapter's base from its Hub adapter_config.json."""

    @staticmethod
    def _resp(cfg: dict):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(cfg).encode()

        return _Resp()

    def test_fetches_base_from_remote_adapter_config(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        cfg = {"base_model_name_or_path": "nvidia/NVIDIA-Nemotron-3-Nano-4B"}
        with patch("urllib.request.urlopen", return_value = self._resp(cfg)):
            assert (
                _remote_lora_base("someuser/my-nemotron-lora") == "nvidia/NVIDIA-Nemotron-3-Nano-4B"
            )

    def test_local_or_noncanonical_returns_none(self):
        assert _remote_lora_base("/local/dir/adapter") is None
        assert _remote_lora_base("plainname") is None

    def test_respects_hf_endpoint(self, monkeypatch):
        # Enterprise mirror: the fetch must target HF_ENDPOINT, not hardcoded huggingface.co.
        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        monkeypatch.setenv("HF_ENDPOINT", "https://hf.mirror.internal")
        seen = {}

        def fake_urlopen(req, timeout = 10):
            seen["url"] = req.full_url
            return self._resp({"base_model_name_or_path": "org/base"})

        with patch("urllib.request.urlopen", side_effect = fake_urlopen):
            assert _remote_lora_base("user/adapter") == "org/base"
        assert seen["url"].startswith("https://hf.mirror.internal/user/adapter/raw/main/")

    @staticmethod
    def _seed_adapter_cache(
        hub: Path,
        repo_id: str,
        base: str,
        commit: str = "deadbeef",
    ):
        repo = hub / ("models--" + repo_id.replace("/", "--"))
        snap = repo / "snapshots" / commit
        snap.mkdir(parents = True)
        (snap / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
        (repo / "refs").mkdir(parents = True)
        (repo / "refs" / "main").write_text(commit)

    def test_offline_reads_base_from_hf_cache(self, tmp_path: Path, monkeypatch):
        self._seed_adapter_cache(tmp_path, "user/cached-lora", "nvidia/Nemotron-H-8B")
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        with patch("urllib.request.urlopen") as mock_url:
            assert _remote_lora_base("user/cached-lora") == "nvidia/Nemotron-H-8B"
            mock_url.assert_not_called()  # offline: cache only, no network

    def test_fetch_failure_falls_back_to_cache(self, tmp_path: Path, monkeypatch):
        self._seed_adapter_cache(tmp_path, "user/cached-lora", "nvidia/Nemotron-H-8B")
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        with patch("urllib.request.urlopen", side_effect = OSError("boom")):
            assert _remote_lora_base("user/cached-lora") == "nvidia/Nemotron-H-8B"

    def test_offline_uncached_makes_no_request(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        with patch("urllib.request.urlopen") as mock_url:
            assert _remote_lora_base("org/adapter") is None
            mock_url.assert_not_called()

    def test_non_adapter_repo_returns_none(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        with patch("urllib.request.urlopen", side_effect = OSError("boom")):
            assert _remote_lora_base("org/not-an-adapter") is None

    def test_existing_relative_path_not_treated_as_repo(self, monkeypatch):
        # An existing one-slash relative path (e.g. outputs/run1) is a local checkpoint, not
        # a Hub repo: no request, no risk of matching an unrelated remote/cached adapter.
        import utils.paths as paths
        monkeypatch.setattr(paths, "is_local_path", lambda p: True)
        with patch("urllib.request.urlopen") as mock_url:
            assert _remote_lora_base("outputs/run1") is None
            mock_url.assert_not_called()

    def test_404_returns_none_not_stale_cache(self, tmp_path: Path, monkeypatch):
        import urllib.error

        # The repo is now a full model (adapter_config.json 404s) but a stale LoRA snapshot is
        # cached: a definitive 404 must return None, not the stale base.
        self._seed_adapter_cache(tmp_path, "user/was-a-lora", "old/base")
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        err = urllib.error.HTTPError("url", 404, "Not Found", {}, None)
        with patch("urllib.request.urlopen", side_effect = err):
            assert _remote_lora_base("user/was-a-lora") is None

    def test_transient_http_error_falls_back_to_cache(self, tmp_path: Path, monkeypatch):
        import urllib.error

        self._seed_adapter_cache(tmp_path, "user/cached-lora", "nvidia/Nemotron-H-8B")
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        err = urllib.error.HTTPError("url", 503, "Service Unavailable", {}, None)
        with patch("urllib.request.urlopen", side_effect = err):
            assert _remote_lora_base("user/cached-lora") == "nvidia/Nemotron-H-8B"


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

        key = (str(tmp_path), None)
        _check_tokenizer_config_needs_v5(str(tmp_path))
        assert key in _tokenizer_class_cache
        assert _tokenizer_class_cache[key] is True

    def test_token_cache_isolation_and_auth_fetch(self, monkeypatch):
        # A gated repo: the unauthenticated miss (cached under (model, None)) must not block a
        # later authed fetch (separate key), and the token rides in the Authorization header.
        import utils.transformers_version as tv

        monkeypatch.setattr(tv, "_env_offline", lambda: False)
        seen_auth = []

        class _Resp:
            def __init__(self, body):
                self._b = body

            def read(self):
                return self._b.encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout = 10):
            auth = req.get_header("Authorization")
            seen_auth.append(auth)
            if auth:
                return _Resp(json.dumps({"tokenizer_class": "TokenizersBackend"}))
            raise OSError("HTTP 401")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        assert _check_tokenizer_config_needs_v5("org/gated") is False  # unauth miss
        assert _check_tokenizer_config_needs_v5("org/gated", "tok") is True  # authed hit
        assert seen_auth == [None, "Bearer tok"]
        assert _tokenizer_class_cache[("org/gated", None)] is False  # miss not poisoning


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

        key = (str(tmp_path), None)
        _check_config_needs_550(str(tmp_path))
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

        key = (str(tmp_path), None)
        _check_config_needs_510(str(tmp_path))
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
# NemotronH dense (MLP) models need the 5.10 tier
# ---------------------------------------------------------------------------


class TestNemotronHNeedsMlpSupport:
    """Dense NemotronH configs (MLP layers) require transformers >= 5.10."""

    def test_hybrid_override_pattern_with_dash(self):
        cfg = {
            "model_type": "nemotron_h",
            "hybrid_override_pattern": "M-M-M*-M-",
        }
        assert _nemotron_h_needs_mlp_support(cfg) is True

    def test_layers_block_type_with_mlp(self):
        cfg = {
            "model_type": "nemotron_h",
            "layers_block_type": ["mamba", "mlp", "attention", "mamba"],
        }
        assert _nemotron_h_needs_mlp_support(cfg) is True

    def test_nemotron_h_moe_only_returns_false(self):
        """A pure MoE NemotronH (no MLP) does not need the 5.10 tier."""
        cfg = {
            "model_type": "nemotron_h",
            "hybrid_override_pattern": "MEME*MEM",
        }
        assert _nemotron_h_needs_mlp_support(cfg) is False

    def test_non_nemotron_with_dash_returns_false(self):
        """The dash heuristic only applies to nemotron_h configs."""
        cfg = {"model_type": "llama", "hybrid_override_pattern": "M-M-"}
        assert _nemotron_h_needs_mlp_support(cfg) is False

    def test_config_needs_510_includes_dense_nemotron_h(self):
        cfg = {
            "model_type": "nemotron_h",
            "hybrid_override_pattern": "M-M-M*-",
        }
        assert _config_needs_510(cfg) is True

    def test_nested_llm_config_with_dash(self):
        # VL wrapper (e.g. NemotronH_Nano_VL_V2): dense LM is under llm_config.
        cfg = {
            "model_type": "NemotronH_Nano_VL_V2",
            "llm_config": {"model_type": "nemotron_h", "hybrid_override_pattern": "M-M*-"},
        }
        assert _nemotron_h_needs_mlp_support(cfg) is True
        assert _config_needs_510(cfg) is True

    def test_nested_text_config_with_mlp(self):
        cfg = {
            "model_type": "wrapper",
            "text_config": {"model_type": "nemotron_h", "layers_block_type": ["mamba", "mlp"]},
        }
        assert _nemotron_h_needs_mlp_support(cfg) is True

    def test_nested_non_nemotron_returns_false(self):
        cfg = {"model_type": "wrapper", "llm_config": {"model_type": "llama"}}
        assert _nemotron_h_needs_mlp_support(cfg) is False

    def test_non_dict_and_missing_nested_do_not_raise(self):
        assert _nemotron_h_needs_mlp_support(None) is False
        assert _nemotron_h_needs_mlp_support({"model_type": "wrapper", "llm_config": None}) is False


def _hf_response(cfg: dict):
    """A urlopen() context-manager stand-in returning *cfg* as JSON bytes."""

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(cfg).encode()

    return _Resp()


class TestConfigJsonHfCacheFallback:
    """HF hub cache is consulted only offline or after a failed fetch (never stale online)."""

    def setup_method(self):
        _config_json_cache.clear()

    @staticmethod
    def _seed_cache(
        hub: Path,
        repo_id: str,
        cfg: dict,
        commit: str = "deadbeef",
    ):
        repo = hub / ("models--" + repo_id.replace("/", "--"))
        snap = repo / "snapshots" / commit
        snap.mkdir(parents = True)
        (snap / "config.json").write_text(json.dumps(cfg))
        (repo / "refs").mkdir(parents = True)
        (repo / "refs" / "main").write_text(commit)

    def test_offline_reads_from_cache(self, tmp_path: Path, monkeypatch):
        cfg = {"model_type": "nemotron_h", "hybrid_override_pattern": "M-M*-"}
        self._seed_cache(tmp_path, "unsloth/NVIDIA-Nemotron-3-Nano-4B", cfg)
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        with patch("urllib.request.urlopen") as mock_url:
            assert _load_config_json("unsloth/NVIDIA-Nemotron-3-Nano-4B") == cfg
            mock_url.assert_not_called()

    def test_online_prefers_network_over_cache(self, tmp_path: Path, monkeypatch):
        stale = {"model_type": "nemotron_h", "hybrid_override_pattern": "MMMM"}
        fresh = {"model_type": "nemotron_h", "hybrid_override_pattern": "M-M*-"}
        self._seed_cache(tmp_path, "org/model", stale)
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
        with patch("urllib.request.urlopen", return_value = _hf_response(fresh)):
            assert _load_config_json("org/model") == fresh  # network wins, not stale cache

    def test_network_failure_falls_back_to_cache(self, tmp_path: Path, monkeypatch):
        cfg = {"model_type": "nemotron_h", "hybrid_override_pattern": "M-M*-"}
        self._seed_cache(tmp_path, "org/model", cfg)
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        with patch("urllib.request.urlopen", side_effect = OSError("boom")):
            assert _load_config_json("org/model") == cfg

    def test_offline_uncached_returns_none(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        with patch("urllib.request.urlopen") as mock_url:
            assert _load_config_json("private/unknown") is None
            mock_url.assert_not_called()

    def test_helper_ignores_local_paths(self, tmp_path: Path):
        # A filesystem path is not a repo id; never treat it as one.
        assert _config_json_from_hf_cache(str(tmp_path)) is None
        assert _config_json_from_hf_cache("plainname") is None

    def test_no_refs_main_picks_newest_snapshot(self, tmp_path: Path, monkeypatch):
        # No refs/main (commit-pinned downloads): lexicographic order would pick the older
        # SHA; selection must follow mtime so the newest snapshot wins.
        repo = tmp_path / "models--org--model"
        old = repo / "snapshots" / "0000old"
        new = repo / "snapshots" / "ffffnew"
        old.mkdir(parents = True)
        new.mkdir(parents = True)
        (old / "config.json").write_text(json.dumps({"model_type": "stale"}))
        (new / "config.json").write_text(json.dumps({"model_type": "fresh"}))
        os.utime(old / "config.json", (1000, 1000))
        os.utime(new / "config.json", (2000, 2000))
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        assert _config_json_from_hf_cache("org/model") == {"model_type": "fresh"}

    def test_transient_failure_does_not_cache_fallback(self, tmp_path: Path, monkeypatch):
        stale = {"model_type": "nemotron_h", "hybrid_override_pattern": "MMMM"}
        fresh = {"model_type": "nemotron_h", "hybrid_override_pattern": "M-M*-"}
        self._seed_cache(tmp_path, "org/model", stale)
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        # Network fails -> serve the cached snapshot, but it must not be memoized.
        with patch("urllib.request.urlopen", side_effect = OSError("boom")):
            assert _load_config_json("org/model") == stale
        # Connectivity returns: the next call must hit the network for the fresh config.
        with patch("urllib.request.urlopen", return_value = _hf_response(fresh)):
            assert _load_config_json("org/model") == fresh

    def test_auth_failure_does_not_serve_cache(self, tmp_path: Path, monkeypatch):
        import urllib.error

        # config.json cached from an earlier authorized session; an unauthenticated 4xx
        # must not be handed that private metadata.
        cfg = {"model_type": "nemotron_h", "hybrid_override_pattern": "M-M*-"}
        self._seed_cache(tmp_path, "private/model", cfg)
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        for code in (401, 403, 404):
            _config_json_cache.clear()
            err = urllib.error.HTTPError("url", code, "denied", {}, None)
            with patch("urllib.request.urlopen", side_effect = err):
                assert _load_config_json("private/model") is None

    def test_server_error_still_falls_back_to_cache(self, tmp_path: Path, monkeypatch):
        import urllib.error

        cfg = {"model_type": "nemotron_h", "hybrid_override_pattern": "M-M*-"}
        self._seed_cache(tmp_path, "org/model", cfg)
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        # A 5xx is transient, not an access decision: keep serving the cache.
        err = urllib.error.HTTPError("url", 503, "busy", {}, None)
        with patch("urllib.request.urlopen", side_effect = err):
            assert _load_config_json("org/model") == cfg


class TestTierCheckTransientRetry:
    """tier-needs checks must not memoize a transient fetch fallback."""

    def setup_method(self):
        _config_json_cache.clear()
        _config_needs_510_cache.clear()
        _config_needs_550_cache.clear()

    @staticmethod
    def _seed_cache(
        hub: Path,
        repo_id: str,
        cfg: dict,
        commit: str = "deadbeef",
    ):
        repo = hub / ("models--" + repo_id.replace("/", "--"))
        snap = repo / "snapshots" / commit
        snap.mkdir(parents = True)
        (snap / "config.json").write_text(json.dumps(cfg))
        (repo / "refs").mkdir(parents = True)
        (repo / "refs" / "main").write_text(commit)

    def test_transient_fallback_not_memoized_then_retries(self, tmp_path: Path, monkeypatch):
        stale = {"model_type": "llama"}  # does not need 510
        fresh = {"architectures": ["Gemma4UnifiedForConditionalGeneration"]}  # needs 510
        self._seed_cache(tmp_path, "org/model", stale)
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        # Network blip -> serve the cache, but do NOT pin the tier result.
        with patch("urllib.request.urlopen", side_effect = OSError("boom")):
            assert _check_config_needs_510("org/model") is False
        assert ("org/model", None) not in _config_needs_510_cache
        # Connectivity returns: the next call re-fetches and sees the higher tier.
        with patch("urllib.request.urlopen", return_value = _hf_response(fresh)):
            assert _check_config_needs_510("org/model") is True
        assert _config_needs_510_cache[("org/model", None)] is True  # definitive read memoized

    def test_definitive_network_read_is_memoized(self, tmp_path: Path, monkeypatch):
        fresh = {"architectures": ["Gemma4ForConditionalGeneration"]}  # needs 550
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        with patch("urllib.request.urlopen", return_value = _hf_response(fresh)) as mock_url:
            assert _check_config_needs_550("org/model") is True
            assert _check_config_needs_550("org/model") is True
            assert mock_url.call_count == 1  # second call served from the tier cache


class TestHigherTier:
    def test_picks_stronger_tier(self):
        assert _higher_tier("default", "510") == "510"
        assert _higher_tier("530", "550") == "550"
        assert _higher_tier("510", "default") == "510"
        assert _higher_tier("default", "default") == "default"


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

    def test_dense_nemotron_h_config_json_returns_510(self, tmp_path: Path):
        """Local dense NemotronH checkpoint → 510 (MLP layers need >= 5.10)."""
        cfg = {
            "model_type": "nemotron_h",
            "hybrid_override_pattern": "M-M-M*-M-",
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        # A v5 tokenizer would otherwise route this to 530; 510 must win.
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "TokenizersBackend"})
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            assert get_transformers_tier(str(tmp_path)) == "510"
            mock_urlopen.assert_not_called()

    def test_dense_nemotron_h_remote_config_returns_510(self):
        """Remote dense NemotronH (HF id) → 510 via config.json fetch, not 530."""

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "model_type": "nemotron_h",
                        "hybrid_override_pattern": "M-M-M*-M-",
                    }
                ).encode()

        with patch("urllib.request.urlopen", return_value = _Response()):
            assert get_transformers_tier("unsloth/NVIDIA-Nemotron-3-Nano-4B") == "510"

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


def _proc(returncode, stderr = ""):
    from types import SimpleNamespace
    return SimpleNamespace(returncode = returncode, stdout = "", stderr = stderr)


class TestProbeTier:
    """_probe_tier resolves the tier by parsing config in each sidecar and escalating."""

    def setup_method(self):
        _probe_tier_cache.clear()

    def _patch_common(self, monkeypatch):
        for fn in (
            "_ensure_venv_t5_530_exists",
            "_ensure_venv_t5_550_exists",
            "_ensure_venv_t5_510_exists",
        ):
            monkeypatch.setattr(f"utils.transformers_version.{fn}", lambda: True)
        monkeypatch.delenv("UNSLOTH_DISABLE_TIER_PROBE", raising = False)

    def _venv_dirs(self):
        import utils.transformers_version as tv
        return [tv._VENV_T5_530_DIR, tv._VENV_T5_550_DIR, tv._VENV_T5_510_DIR]

    def test_escalates_to_first_parsing_tier(self, monkeypatch):
        self._patch_common(monkeypatch)
        seen = []
        results = iter([_proc(1, "KeyError: '-'"), _proc(1, "KeyError: '-'"), _proc(0)])

        def fake_run(cmd, **k):
            seen.append(cmd[3])  # target_dir
            return next(results)

        monkeypatch.setattr("utils.transformers_version.subprocess.run", fake_run)
        assert _probe_tier("org/dense-nemotron", None, "x") == "510"
        assert seen == self._venv_dirs()  # escalated 530 -> 550 -> 510

    def test_first_success_stops_escalation(self, monkeypatch):
        self._patch_common(monkeypatch)
        calls = []
        monkeypatch.setattr(
            "utils.transformers_version.subprocess.run",
            lambda cmd, **k: calls.append(cmd[3]) or _proc(0),
        )
        assert _probe_tier("org/m", None, "x") == "530"
        assert len(calls) == 1

    def test_middle_tier_parses(self, monkeypatch):
        self._patch_common(monkeypatch)
        results = iter([_proc(1, "ValueError: bad"), _proc(0)])
        monkeypatch.setattr(
            "utils.transformers_version.subprocess.run", lambda cmd, **k: next(results)
        )
        assert _probe_tier("org/m", None, "x") == "550"

    def test_nothing_parses_stays_530_and_caches(self, monkeypatch):
        # All tiers probed, none parse -> a remote-code model that loads via its own code;
        # keep 530 (never jump to 510). Conclusive, so cached by model_name.
        self._patch_common(monkeypatch)
        monkeypatch.setattr(
            "utils.transformers_version.subprocess.run",
            lambda cmd, **k: _proc(1, "KeyError: '-'"),
        )
        assert _probe_tier("org/m", None, "x") == "530"
        assert _probe_tier_cache["org/m"] == "530"

    def test_partial_sidecars_no_parse_is_530_uncached(self, monkeypatch):
        # 510 sidecar missing and 530/550 fail to parse -> environment is incomplete, so we
        # cannot conclude; return 530 uncached so it is retried once 510 is available.
        monkeypatch.delenv("UNSLOTH_DISABLE_TIER_PROBE", raising = False)
        for fn in ("_ensure_venv_t5_530_exists", "_ensure_venv_t5_550_exists"):
            monkeypatch.setattr(f"utils.transformers_version.{fn}", lambda: True)
        monkeypatch.setattr("utils.transformers_version._ensure_venv_t5_510_exists", lambda: False)
        monkeypatch.setattr(
            "utils.transformers_version.subprocess.run",
            lambda cmd, **k: _proc(1, "KeyError: '-'"),
        )
        assert _probe_tier("org/m", None, "x") == "530"
        assert "org/m" not in _probe_tier_cache

    def test_success_not_cached_when_lower_tier_skipped(self, monkeypatch):
        # 530 sidecar unavailable but 550 parses: return 550 (best effort now) but do NOT
        # cache it, since once 530 is installed it may be the lowest valid tier.
        monkeypatch.delenv("UNSLOTH_DISABLE_TIER_PROBE", raising = False)
        monkeypatch.setattr("utils.transformers_version._ensure_venv_t5_530_exists", lambda: False)
        for fn in ("_ensure_venv_t5_550_exists", "_ensure_venv_t5_510_exists"):
            monkeypatch.setattr(f"utils.transformers_version.{fn}", lambda: True)
        monkeypatch.setattr("utils.transformers_version.subprocess.run", lambda cmd, **k: _proc(0))
        assert _probe_tier("org/m", None, "x") == "550"
        assert "org/m" not in _probe_tier_cache  # skipped a lower tier -> not pinned

    def test_cache_hit_skips_subprocess(self, monkeypatch):
        self._patch_common(monkeypatch)
        monkeypatch.setattr("utils.transformers_version.subprocess.run", lambda cmd, **k: _proc(0))
        assert _probe_tier("org/m", None, "x") == "530"

        def boom(cmd, **k):
            raise AssertionError("should not re-probe a cached model_name")

        monkeypatch.setattr("utils.transformers_version.subprocess.run", boom)
        assert _probe_tier("org/m", None, "x") == "530"

    def test_transient_failure_is_530_and_uncached(self, monkeypatch):
        self._patch_common(monkeypatch)
        monkeypatch.setattr(
            "utils.transformers_version.subprocess.run",
            lambda cmd, **k: _proc(1, "ConnectionError: Max retries exceeded"),
        )
        assert _probe_tier("org/m", None, "x") == "530"
        assert "org/m" not in _probe_tier_cache  # retried next load

    def test_timeout_is_530_and_uncached(self, monkeypatch):
        import subprocess as _sp

        self._patch_common(monkeypatch)

        def timeout(cmd, **k):
            raise _sp.TimeoutExpired(cmd, 60)

        monkeypatch.setattr("utils.transformers_version.subprocess.run", timeout)
        assert _probe_tier("org/m", None, "x") == "530"
        assert "org/m" not in _probe_tier_cache

    def test_all_venvs_missing_is_530_no_spawn(self, monkeypatch):
        for fn in (
            "_ensure_venv_t5_530_exists",
            "_ensure_venv_t5_550_exists",
            "_ensure_venv_t5_510_exists",
        ):
            monkeypatch.setattr(f"utils.transformers_version.{fn}", lambda: False)

        def boom(cmd, **k):
            raise AssertionError("no sidecar available; must not spawn")

        monkeypatch.setattr("utils.transformers_version.subprocess.run", boom)
        assert _probe_tier("org/m", None, "x") == "530"
        assert "org/m" not in _probe_tier_cache  # nothing probed -> uncached

    def test_probe_does_not_import_hub(self, monkeypatch):
        # The probe must not import huggingface_hub: that would land before the sidecar is on
        # sys.path (activation never purges), pinning the default-env hub. So no in-process sha.
        self._patch_common(monkeypatch)
        monkeypatch.setattr("utils.transformers_version.subprocess.run", lambda cmd, **k: _proc(0))
        sys.modules.pop("huggingface_hub", None)
        _probe_tier("org/m", None, "x")
        assert "huggingface_hub" not in sys.modules

    def test_disable_flag_skips_probe(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_DISABLE_TIER_PROBE", "1")

        def boom(cmd, **k):
            raise AssertionError("probe disabled; must not spawn")

        monkeypatch.setattr("utils.transformers_version.subprocess.run", boom)
        assert _probe_tier("org/m", None, "x") == "530"

    def test_get_tier_uses_probe_for_remote_tokenizer_signal(self, monkeypatch):
        # tokenizer says 5.x but no architecture/substring match -> probe (not a 530 guess).
        monkeypatch.setattr(
            "utils.transformers_version._check_config_needs_510", lambda m, t = None: False
        )
        monkeypatch.setattr(
            "utils.transformers_version._check_config_needs_550", lambda m, t = None: False
        )
        monkeypatch.setattr(
            "utils.transformers_version._check_tokenizer_config_needs_v5", lambda m, t = None: True
        )
        monkeypatch.setattr("utils.transformers_version._probe_tier", lambda m, t, reason: "510")
        assert get_transformers_tier("org/unknown-5x-arch") == "510"

    def test_stderr_is_transient(self):
        assert _stderr_is_transient("ConnectionError: x") is True
        assert _stderr_is_transient("GatedRepoError: need token") is True
        assert _stderr_is_transient("KeyError: '-'") is False
        assert _stderr_is_transient("ValueError: bad pattern") is False

    def test_get_tier_threads_token_to_checks(self, monkeypatch):
        # A gated/private model: the token must reach the config/tokenizer checks (and the
        # probe), otherwise the authed-only signal is missed and it falls to default 4.x.
        seen = {}
        monkeypatch.setattr(
            "utils.transformers_version._check_config_needs_510",
            lambda m, t = None: seen.update({"510": t}) or False,
        )
        monkeypatch.setattr(
            "utils.transformers_version._check_config_needs_550",
            lambda m, t = None: seen.update({"550": t}) or False,
        )
        monkeypatch.setattr(
            "utils.transformers_version._check_tokenizer_config_needs_v5",
            lambda m, t = None: seen.update({"tok": t}) or True,
        )
        monkeypatch.setattr(
            "utils.transformers_version._probe_tier",
            lambda m, t, reason: seen.update({"probe": t}) or "510",
        )
        assert get_transformers_tier("org/gated-5x", "hf_abc") == "510"
        assert seen == {"510": "hf_abc", "550": "hf_abc", "tok": "hf_abc", "probe": "hf_abc"}

    def test_activate_threads_token_to_tier(self, monkeypatch):
        # activate_transformers_for_subprocess must forward hf_token to tier detection, or
        # the gated-model checks above run unauthenticated and the fix is unreachable.
        seen = {}
        monkeypatch.setattr("utils.transformers_version._resolve_base_model", lambda m: m)
        monkeypatch.setattr(
            "utils.transformers_version.get_transformers_tier",
            lambda m, t = None: seen.update({"model": m, "token": t}) or "default",
        )
        activate_transformers_for_subprocess("org/gated", "hf_xyz")
        assert seen == {"model": "org/gated", "token": "hf_xyz"}

    def test_local_checkpoint_reprobes_after_config_change(self, monkeypatch, tmp_path):
        # A local checkpoint overwritten in place must re-probe: the cache key folds in the
        # config.json signature, so a different config does not serve the stale tier.
        self._patch_common(monkeypatch)
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"model_type": "a"}))
        local = str(tmp_path)
        calls = []
        monkeypatch.setattr(
            "utils.transformers_version.subprocess.run",
            lambda cmd, **k: calls.append(1) or _proc(0),
        )
        assert _probe_tier(local, None, "x") == "530"
        assert _probe_tier(local, None, "x") == "530"  # cache hit, no re-spawn
        assert len(calls) == 1
        cfg.write_text(json.dumps({"model_type": "a_longer_value_changing_the_size"}))
        assert _probe_tier(local, None, "x") == "530"
        assert len(calls) == 2  # signature changed -> re-probed

    def test_probe_child_enables_implicit_token(self, monkeypatch):
        # With a token, the probe child must clear an inherited HF_HUB_DISABLE_IMPLICIT_TOKEN=1
        # so HF_TOKEN authenticates the gated config fetch instead of 401ing to 530.
        self._patch_common(monkeypatch)
        monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        captured = {}

        def fake_run(cmd, **k):
            captured.update(k.get("env") or {})
            return _proc(0)

        monkeypatch.setattr("utils.transformers_version.subprocess.run", fake_run)
        assert _probe_tier("org/gated", "secret-token", "x") == "530"
        assert captured.get("HF_TOKEN") == "secret-token"
        assert captured.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") == "0"


class TestProbeGating:
    """probe=False suppresses sidecar probes (the log-only needs_transformers_5 path); a
    config saved by transformers 5.x is probed default-first so a new 5.x-only arch is
    caught without mis-routing a 4.57.x-loadable model onto a sidecar."""

    def setup_method(self):
        _probe_tier_cache.clear()
        _config_json_cache.clear()
        _config_needs_510_cache.clear()
        _config_needs_550_cache.clear()
        _tokenizer_class_cache.clear()

    def _patch_venvs(self, monkeypatch):
        for fn in (
            "_ensure_venv_t5_530_exists",
            "_ensure_venv_t5_550_exists",
            "_ensure_venv_t5_510_exists",
        ):
            monkeypatch.setattr(f"utils.transformers_version.{fn}", lambda: True)
        monkeypatch.delenv("UNSLOTH_DISABLE_TIER_PROBE", raising = False)

    def _patch_checks_to_tokenizer(self, monkeypatch):
        monkeypatch.setattr(
            "utils.transformers_version._check_config_needs_510", lambda m, t = None: False
        )
        monkeypatch.setattr(
            "utils.transformers_version._check_config_needs_550", lambda m, t = None: False
        )
        monkeypatch.setattr(
            "utils.transformers_version._check_tokenizer_config_needs_v5", lambda m, t = None: True
        )

    # ---- needs_transformers_5 / probe=False must not spawn probes --------------

    def test_needs_transformers_5_does_not_spawn_probe(self, monkeypatch):
        self._patch_checks_to_tokenizer(monkeypatch)

        def boom(cmd, **k):
            raise AssertionError("needs_transformers_5 must not spawn a probe")

        monkeypatch.setattr("utils.transformers_version.subprocess.run", boom)
        # Still correctly reports 5.x from the tokenizer signal, just without probing.
        assert needs_transformers_5("org/unknown-5x") is True

    def test_probe_false_returns_530_for_tokenizer_signal(self, monkeypatch):
        self._patch_checks_to_tokenizer(monkeypatch)

        def boom(cmd, **k):
            raise AssertionError("probe=False must not spawn a probe")

        monkeypatch.setattr("utils.transformers_version.subprocess.run", boom)
        assert get_transformers_tier("org/unknown-5x", probe = False) == "530"

    # ---- version-field probe is default-first (no mis-routing of 4.x models) ----

    def test_version_field_probe_stays_default_when_default_parses(self, monkeypatch):
        self._patch_venvs(monkeypatch)
        monkeypatch.setattr(
            "utils.transformers_version._check_tokenizer_config_needs_v5", lambda m, t = None: False
        )
        _config_json_cache[("org/new", None)] = {
            "model_type": "brandnew",
            "transformers_version": "5.0.0",
        }
        seen = []
        monkeypatch.setattr(
            "utils.transformers_version.subprocess.run",
            lambda cmd, **k: seen.append(cmd[3]) or _proc(0),
        )
        assert get_transformers_tier("org/new") == "default"
        assert seen == [""]  # probed the ambient default tier first, it parsed -> stayed default

    def test_version_field_probe_escalates_when_default_fails(self, monkeypatch):
        import utils.transformers_version as tv

        self._patch_venvs(monkeypatch)
        monkeypatch.setattr(
            "utils.transformers_version._check_tokenizer_config_needs_v5", lambda m, t = None: False
        )
        _config_json_cache[("org/new", None)] = {
            "model_type": "brandnew",
            "transformers_version": "5.6.0",
        }
        results = iter([_proc(1, "KeyError: 'x'"), _proc(1, "KeyError: 'x'"), _proc(0)])
        seen = []
        monkeypatch.setattr(
            "utils.transformers_version.subprocess.run",
            lambda cmd, **k: seen.append(cmd[3]) or next(results),
        )
        assert get_transformers_tier("org/new") == "550"
        assert seen == ["", tv._VENV_T5_530_DIR, tv._VENV_T5_550_DIR]

    def test_ordinary_4x_config_does_not_probe(self, monkeypatch):
        self._patch_venvs(monkeypatch)
        monkeypatch.setattr(
            "utils.transformers_version._check_tokenizer_config_needs_v5", lambda m, t = None: False
        )
        _config_json_cache[("org/llama", None)] = {
            "model_type": "llama",
            "transformers_version": "4.57.0",
        }

        def boom(cmd, **k):
            raise AssertionError("a 4.x-saved config must not trigger a probe")

        monkeypatch.setattr("utils.transformers_version.subprocess.run", boom)
        assert get_transformers_tier("org/llama") == "default"

    def test_needs_transformers_5_true_for_version_field_only(self, monkeypatch):
        # A 5.x-saved standard-tokenizer model must report as 5.x (for vision routing)
        # without spawning a probe.
        monkeypatch.setattr(
            "utils.transformers_version._check_config_needs_510", lambda m, t = None: False
        )
        monkeypatch.setattr(
            "utils.transformers_version._check_config_needs_550", lambda m, t = None: False
        )
        monkeypatch.setattr(
            "utils.transformers_version._check_tokenizer_config_needs_v5", lambda m, t = None: False
        )
        _config_json_cache[("org/new", None)] = {
            "model_type": "brandnew",
            "transformers_version": "5.2.0",
        }

        def boom(cmd, **k):
            raise AssertionError("needs_transformers_5 must not spawn a probe")

        monkeypatch.setattr("utils.transformers_version.subprocess.run", boom)
        assert needs_transformers_5("org/new") is True

    def test_default_first_result_not_reused_for_tokenizer_path(self, monkeypatch, tmp_path):
        # A default-first probe can cache "default"; a later tokenizer/known-5.x call
        # (floor=530) must re-probe, not reuse that "default".
        self._patch_venvs(monkeypatch)
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "brandnew", "transformers_version": "5.0.0"})
        )
        local = str(tmp_path)
        monkeypatch.setattr("utils.transformers_version.subprocess.run", lambda cmd, **k: _proc(0))
        assert (
            _probe_tier(local, None, "version", include_default = True, floor = "default") == "default"
        )
        seen = []
        monkeypatch.setattr(
            "utils.transformers_version.subprocess.run",
            lambda cmd, **k: seen.append(cmd[3]) or _proc(0),
        )
        # Tokenizer/known-5.x mode (floor=530): must re-probe and never reuse "default".
        assert _probe_tier(local, None, "tokenizer needs 5.x") == "530"
        assert seen, "tokenizer path reused the cached default result instead of re-probing"


class TestLocalCheckpointFilesAppear:
    """A local checkpoint dir inspected before its files exist must not cache the miss or hit
    the network, so files written later in the same process are still read (in-progress
    checkpoints)."""

    def setup_method(self):
        _tokenizer_class_cache.clear()
        _config_json_cache.clear()

    def test_tokenizer_config_appearing_later_is_read(self, tmp_path: Path, monkeypatch):
        local = str(tmp_path)

        def boom(*a, **k):
            raise AssertionError("a local checkpoint must not be fetched from the Hub")

        monkeypatch.setattr("urllib.request.urlopen", boom)
        # Before the file exists: not 5.x, no network, and the miss must not be pinned.
        assert _check_tokenizer_config_needs_v5(local) is False
        # The file appears with a 5.x-only tokenizer -> the next call must read it.
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "TokenizersBackend"})
        )
        assert _check_tokenizer_config_needs_v5(local) is True

    def test_config_json_appearing_later_is_read(self, tmp_path: Path, monkeypatch):
        local = str(tmp_path)

        def boom(*a, **k):
            raise AssertionError("a local checkpoint must not be fetched from the Hub")

        monkeypatch.setattr("urllib.request.urlopen", boom)
        assert _load_config_json(local) is None
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
        assert _load_config_json(local) == {"model_type": "gemma4"}


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

    def test_activate_prefers_local_checkpoint_tier_over_resolved_base(self, caplog, tmp_path):
        # Base resolves to an offline/private id (default tier); the local config.json wins.
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        local = str(tmp_path)
        caplog.set_level(logging.INFO)
        snap = self._snapshot_env()
        tiers = {local: "510", "private/base": "default"}
        try:
            with (
                patch(
                    "utils.transformers_version._resolve_base_model",
                    return_value = "private/base",
                ),
                patch(
                    "utils.transformers_version.get_transformers_tier",
                    side_effect = lambda m, t = None: tiers[m],
                ),
                patch(
                    "utils.transformers_version._ensure_venv_t5_510_exists",
                    return_value = True,
                ),
            ):
                activate_transformers_for_subprocess(local)
        finally:
            self._restore_env(snap)

        text = " ".join(r.getMessage() for r in caplog.records).lower()
        assert "5.10.2" in text, f"local checkpoint tier did not win: {text!r}"

    def test_activate_adapter_without_config_skips_path_name_recheck(self, caplog, tmp_path):
        # LoRA adapter in a dir named 'gemma-4' (base resolves elsewhere): the resolved
        # base drives the tier; the path name must not re-check or upgrade it.
        adapter = tmp_path / "gemma-4-experiment" / "llama-lora"
        adapter.mkdir(parents = True)
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "meta/llama"})
        )
        local = str(adapter)
        caplog.set_level(logging.INFO)
        snap = self._snapshot_env()
        seen = []

        def fake_tier(m, t = None):
            seen.append(m)
            return "550" if "gemma-4" in m else "default"

        try:
            with (
                patch(
                    "utils.transformers_version._resolve_base_model",
                    return_value = "meta/llama",
                ),
                patch(
                    "utils.transformers_version.get_transformers_tier",
                    side_effect = fake_tier,
                ),
            ):
                activate_transformers_for_subprocess(local)
        finally:
            self._restore_env(snap)

        assert seen == ["meta/llama"], f"adapter path was re-checked via substrings: {seen!r}"
        text = " ".join(r.getMessage() for r in caplog.records).lower()
        assert "default transformers" in text, f"adapter wrongly upgraded: {text!r}"


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
        _config_needs_530_cache.clear()

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

    def test_config_needs_530_qwen3_5_moe(self):
        """Qwen3.5 MoE (Qwen3.5-35B-A3B / 122B-A10B) uses qwen3_5_moe ids."""
        assert (
            _config_needs_530(
                {
                    "model_type": "qwen3_5_moe",
                    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                }
            )
            is True
        )

    def test_config_needs_530_qwen3_next(self):
        assert (
            _config_needs_530(
                {"model_type": "qwen3_next", "architectures": ["Qwen3NextForCausalLM"]}
            )
            is True
        )

    def test_config_needs_530_qwen3_5_text_towers(self):
        """Text-tower configs (architectures may be stripped) still need 5.3.0."""
        assert _config_needs_530({"model_type": "qwen3_5_text"}) is True
        assert _config_needs_530({"model_type": "qwen3_5_moe_text"}) is True

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

    def test_tier_local_qwen35_moe_config_selects_530(self, tmp_path: Path):
        """A renamed Qwen3.5 MoE folder (no name hint) routes to 530 via config."""
        d = tmp_path / "my-custom-moe"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5_moe",
                    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                }
            )
        )
        assert get_transformers_tier(str(d)) == "530"

    # --- Qwen3.6 reuses Qwen3.5 config ids but routes to 550 by name ---------

    def test_local_qwen36_config_keeps_550_name_tier(self, tmp_path: Path):
        """Qwen3.6 config carries qwen3_5 ids; a higher-tier name match wins."""
        d = tmp_path / "Qwen3.6-27B"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}
            )
        )
        assert get_transformers_tier(str(d)) == "550"

    def test_local_qwen36_moe_via_name_or_path_keeps_550(self, tmp_path: Path):
        """Renamed Qwen3.6 MoE folder: _name_or_path carries the 5.5 name signal."""
        d = tmp_path / "renamed-q36-moe"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5_moe",
                    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                    "_name_or_path": "Qwen/Qwen3.6-35B-A3B",
                }
            )
        )
        assert get_transformers_tier(str(d)) == "550"

    def test_stale_absolute_name_or_path_not_promoted(self, tmp_path: Path):
        """A non-5.x checkpoint with a stale absolute _name_or_path isn't name-matched."""
        d = tmp_path / "my-llama-ckpt"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({"model_type": "llama", "_name_or_path": "/old/run/qwen3.5-source"})
        )
        with patch(
            "utils.transformers_version._check_tokenizer_config_needs_v5", return_value = False
        ):
            assert get_transformers_tier(str(d)) == "default"

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

    def test_hf_id_fallback_not_triggered_when_name_or_path_is_absolute_self(self, tmp_path: Path):
        """_name_or_path == absolute path of the same checkpoint while model_name
        is a relative path: the two strings differ, but both point to the same
        directory.  The absolute path must not be scanned for tier substrings."""
        d = tmp_path / "qwen3.5-experiment"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    # absolute path — textually different from a relative model_name
                    "_name_or_path": str(d),
                }
            )
        )
        with patch(
            "utils.transformers_version._check_tokenizer_config_needs_v5", return_value = False
        ):
            # Even though str(d) contains "qwen3.5", the local-dir branch recurses
            # into config checks on the resolved path, which returns default.
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


# ---------------------------------------------------------------------------
# _check_config_needs_530 — slow HF-ID path (network stub)
# ---------------------------------------------------------------------------


class TestCheckConfigNeeds530:
    """_check_config_needs_530 is used in the slow HF-ID fallback path for
    private or renamed repos whose names don't contain a 5.3 substring."""

    def setup_method(self):
        _config_json_cache.clear()
        _config_needs_530_cache.clear()

    def test_returns_true_for_qwen3_5_model_type(self):
        with patch(
            "utils.transformers_version._load_config_json",
            return_value = {"model_type": "qwen3_5"},
        ):
            assert _check_config_needs_530("some-private/qwen3.5-variant") is True

    def test_returns_true_for_qwen3_moe_architecture(self):
        with patch(
            "utils.transformers_version._load_config_json",
            return_value = {"architectures": ["Qwen3MoeForCausalLM"]},
        ):
            assert _check_config_needs_530("org/private-moe-model") is True

    def test_returns_false_for_llama(self):
        with patch(
            "utils.transformers_version._load_config_json",
            return_value = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
        ):
            assert _check_config_needs_530("meta-llama/Llama-3-8B") is False

    def test_returns_false_when_config_unavailable(self):
        with patch("utils.transformers_version._load_config_json", return_value = None):
            assert _check_config_needs_530("org/unreachable-model") is False

    def test_result_is_cached(self, tmp_path: Path):
        """A definitive (local) read is cached by (model, token), mirroring 510/550."""
        cfg = {"model_type": "qwen3_5"}
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        key = (str(tmp_path), None)
        _check_config_needs_530(str(tmp_path))
        assert key in _config_needs_530_cache
        assert _config_needs_530_cache[key] is True


# ---------------------------------------------------------------------------
# _norm_separators
# ---------------------------------------------------------------------------


class TestNormSeparators:
    def test_underscore_to_hyphen(self):
        assert _norm_separators("qwen3_5") == "qwen3-5"

    def test_dot_preserved(self):
        assert _norm_separators("qwen3.5") == "qwen3.5"

    def test_hyphen_unchanged(self):
        assert _norm_separators("gemma-4") == "gemma-4"

    def test_mixed(self):
        assert _norm_separators("Qwen3_5.MoE") == "Qwen3-5.MoE"

    def test_whitespace_to_hyphen(self):
        assert _norm_separators("some model") == "some-model"

    def test_empty(self):
        assert _norm_separators("") == ""


# ---------------------------------------------------------------------------
# _tier_from_name — separator-insensitive matching
# ---------------------------------------------------------------------------


class TestTierFromNameSeparatorNorm:
    """Verify that underscore/dot aliases in model IDs resolve to the same
    tier as their canonical hyphen/dot counterparts."""

    def test_qwen3_underscore_5_returns_530(self):
        tier, _ = _tier_from_name("Qwen/Qwen3_5-7B")
        assert tier == "530"

    def test_qwen3_next_underscore_returns_530(self):
        tier, _ = _tier_from_name("org/Qwen3_Next-14B")
        assert tier == "530"

    def test_gemma_4_underscore_returns_550(self):
        tier, _ = _tier_from_name("google/gemma_4_E2B_it")
        assert tier == "550"

    def test_gemma_4_12b_underscore_returns_510(self):
        tier, _ = _tier_from_name("unsloth/gemma_4_12b_it")
        assert tier == "510"

    def test_canonical_dot_still_works(self):
        tier, _ = _tier_from_name("Qwen/Qwen3.5-7B")
        assert tier == "530"

    def test_unrelated_underscores_not_promoted(self):
        assert _tier_from_name("meta_llama/Llama_3_8B") is None

    def test_qwen3_hyphen_6_size_not_promoted(self):
        """Qwen3-6B is a size name, not the qwen3.6 release line."""
        assert _tier_from_name("Qwen/Qwen3-6B-Instruct") is None

    def test_qwen3_hyphen_5_size_not_promoted(self):
        assert _tier_from_name("Qwen/Qwen3-5B") is None


# ---------------------------------------------------------------------------
# _resolve_base_model — model_name-then-_name_or_path fallback
# ---------------------------------------------------------------------------


class TestResolveBaseModelNameOrPathFallback:
    """When config.json has both 'model_name' (self-referential local path) and
    '_name_or_path' (original HF ID), _resolve_base_model must use _name_or_path."""

    def test_name_or_path_used_when_model_name_is_self_ref(self, tmp_path: Path):
        d = tmp_path / "my-qwen35-finetune"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {
                    "model_name": str(d),
                    "_name_or_path": "Qwen/Qwen3.5-7B",
                }
            )
        )
        assert _resolve_base_model(str(d)) == "Qwen/Qwen3.5-7B"

    def test_model_name_used_when_not_self_ref(self, tmp_path: Path):
        d = tmp_path / "adapter"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {
                    "model_name": "unsloth/Qwen3.5-7B-bnb-4bit",
                    "_name_or_path": "Qwen/Qwen3.5-7B",
                }
            )
        )
        # model_name is not the local path, so it wins
        assert _resolve_base_model(str(d)) == "unsloth/Qwen3.5-7B-bnb-4bit"

    def test_tier_resolved_via_name_or_path_when_model_name_self_refs(self, tmp_path: Path):
        """End-to-end: get_transformers_tier picks up the sidecar tier from
        _name_or_path even when model_name is set to the checkpoint's own path."""
        d = tmp_path / "my-custom-finetune"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "future_unknown_type",
                    "model_name": str(d),
                    "_name_or_path": "Qwen/Qwen3.5-7B",
                }
            )
        )
        assert get_transformers_tier(str(d)) == "530"

    def test_local_config_tier_not_bypassed_by_private_name_or_path(self, tmp_path: Path):
        """Full checkpoint with model_type: qwen3_5 must still route to 530 even
        when _name_or_path is a private HF ID with no recognisable tier substring.

        Regression guard: before the adapter-only pre-resolve fix,
        activate_transformers_for_subprocess would resolve to the private HF ID
        and then fail to probe it offline, returning default instead of 530.
        """
        d = tmp_path / "my-finetuned-model"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "model_name": str(d),
                    "_name_or_path": "my-org/private-custom-id",
                }
            )
        )
        # get_transformers_tier reads config.json directly and returns 530
        # without needing to probe the private HF ID.
        assert get_transformers_tier(str(d)) == "530"


# ---------------------------------------------------------------------------
# adapter_model-only LoRA resolution (no adapter_config.json)
# ---------------------------------------------------------------------------


class TestAdapterModelOnlyLoRA:
    """A LoRA dir with adapter_model*.safetensors but no adapter_config.json must
    still be detected as an adapter and resolved to its base model so the worker
    activates the base model's sidecar instead of tiering off the adapter folder."""

    def test_has_adapter_weights_detects_safetensors_and_bin(self, tmp_path: Path):
        d = tmp_path / "adapter"
        d.mkdir()
        assert _has_adapter_weights(d) is False
        (d / "adapter_model.safetensors").write_text("")
        assert _has_adapter_weights(d) is True
        d2 = tmp_path / "adapter_bin"
        d2.mkdir()
        (d2 / "adapter_model.bin").write_text("")
        assert _has_adapter_weights(d2) is True

    def test_is_lora_adapter_dir_for_config_and_weights_only(self, tmp_path: Path):
        # adapter_config.json present
        a = tmp_path / "cfg"
        a.mkdir()
        (a / "adapter_config.json").write_text("{}")
        assert _is_lora_adapter_dir(a) is True
        # adapter_model weights only, no config
        b = tmp_path / "weights_only"
        b.mkdir()
        (b / "adapter_model.safetensors").write_text("")
        assert _is_lora_adapter_dir(b) is True
        # plain checkpoint dir (neither)
        c = tmp_path / "plain"
        c.mkdir()
        (c / "config.json").write_text("{}")
        assert _is_lora_adapter_dir(c) is False
        # not a directory
        assert _is_lora_adapter_dir(tmp_path / "missing") is False

    def test_resolve_adapter_only_lora_via_unsloth_dir_name(self, tmp_path: Path):
        """adapter_model-only LoRA with the unsloth_<model>_<ts> naming resolves to
        unsloth/<model> through the import-light directory-name parse."""
        d = tmp_path / "unsloth_Qwen3.5-7B_20260620"
        d.mkdir()
        (d / "adapter_model.safetensors").write_text("")
        assert _resolve_base_model(str(d)) == "unsloth/Qwen3.5-7B"

    def test_activation_pre_resolves_adapter_only_lora(self, tmp_path: Path):
        """Regression: activate_transformers_for_subprocess must pre-resolve an
        adapter_model-only LoRA dir (weights present, adapter_config.json absent).
        Before the gate used _is_lora_adapter_dir, the adapter_config-only check
        skipped resolution and the worker tiered off the adapter folder itself."""
        d = tmp_path / "my-custom-lora"
        d.mkdir()
        (d / "adapter_model.safetensors").write_text("")
        snap = (list(sys.path), os.environ.get("PYTHONPATH"))
        try:
            with (
                patch(
                    "utils.transformers_version._resolve_base_model",
                    side_effect = lambda m: m,
                ) as mock_resolve,
                patch(
                    "utils.transformers_version.get_transformers_tier",
                    return_value = "default",
                ),
            ):
                activate_transformers_for_subprocess(str(d))
        finally:
            sys.path[:] = snap[0]
            if snap[1] is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = snap[1]
        mock_resolve.assert_called_once_with(str(d))


# ---------------------------------------------------------------------------
# 530-config override must not be flipped by stale local path hints
# ---------------------------------------------------------------------------


class TestConfig530OverrideGuard:
    """A correct 530 config must not be flipped to 550 by a 5.5-looking substring in
    a stale/renamed local path; only a real Hub id or the folder basename may override."""

    def test_stale_local_path_does_not_flip_530_to_550(self, tmp_path: Path):
        d = tmp_path / "my-qwen35-run"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "_name_or_path": "/old/run/qwen3.6-source",
                }
            )
        )
        # Stale path is not a Hub id, so the 530 config wins over its qwen3.6 substring.
        assert get_transformers_tier(str(d)) == "530"

    def test_current_basename_can_still_override_to_550(self, tmp_path: Path):
        d = tmp_path / "Qwen3.6-27B"
        d.mkdir()
        # Qwen3.6 reuses the qwen3_5 config id but is a 5.5 model by name.
        (d / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5", "_name_or_path": str(d)})
        )
        assert get_transformers_tier(str(d)) == "550"

    def test_real_hub_id_can_still_override_to_550(self, tmp_path: Path):
        d = tmp_path / "my-finetune"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5", "_name_or_path": "Qwen/Qwen3.6-27B"})
        )
        assert get_transformers_tier(str(d)) == "550"

    def test_remote_qwen36_name_or_path_overrides_530(self):
        """Slow path: a fetched qwen3_5 config naming Qwen3.6 in _name_or_path -> 550."""
        _config_needs_530_cache.clear()
        _config_json_cache.clear()
        with patch(
            "utils.transformers_version._load_config_json",
            return_value = {"model_type": "qwen3_5", "_name_or_path": "Qwen/Qwen3.6-27B"},
        ):
            assert get_transformers_tier("private/renamed-q36") == "550"


class TestLooksLikeHfId:
    def test_empty_and_whitespace_are_not_ids(self):
        assert _looks_like_hf_id("") is False
        assert _looks_like_hf_id("   ") is False

    def test_plain_hub_id(self):
        assert _looks_like_hf_id("Qwen/Qwen3.5-7B") is True

    def test_absolute_and_dot_paths_are_not_ids(self):
        assert _looks_like_hf_id("/old/run/qwen3.5-source") is False
        assert _looks_like_hf_id("./qwen3.5-source") is False

    def test_existing_local_path_is_not_an_id(self, tmp_path: Path):
        d = tmp_path / "Qwen3.5-7B"
        d.mkdir()
        import os as _os

        cwd = _os.getcwd()
        try:
            _os.chdir(tmp_path)
            # "Qwen3.5-7B" exists relative to cwd, so it is a path, not a Hub id.
            assert _looks_like_hf_id("Qwen3.5-7B") is False
        finally:
            _os.chdir(cwd)


class TestMalformedInputRobustness:
    """Tier detection fails open to default instead of crashing on bad input."""

    def setup_method(self):
        _config_json_cache.clear()
        _config_needs_530_cache.clear()

    def test_non_string_model_type_does_not_crash(self):
        assert _config_needs_530({"model_type": ["qwen3_5"]}) is False

    def test_non_list_architectures_does_not_crash(self):
        assert _config_needs_530({"architectures": "Qwen3_5ForCausalLM"}) is False

    def test_local_config_non_string_fields_returns_default(self, tmp_path: Path):
        d = tmp_path / "weird"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({"model_type": ["qwen3_5"], "_name_or_path": {"x": 1}})
        )
        with patch(
            "utils.transformers_version._check_tokenizer_config_needs_v5", return_value = False
        ):
            assert get_transformers_tier(str(d)) == "default"

    def test_pathological_long_name_does_not_crash(self):
        # An over-long name makes is_file() raise OSError; must fail open.
        assert get_transformers_tier("x" * 5000) == "default"

    def test_empty_name_returns_default(self):
        assert get_transformers_tier("") == "default"


# ---------------------------------------------------------------------------
# Offline negatives must not poison the version caches (persistent worker)
# ---------------------------------------------------------------------------


class TestOfflineCacheNotPoisoned:
    """An offline first load must not leave a stale negative for a later online read."""

    def setup_method(self):
        _tokenizer_class_cache.clear()
        _config_json_cache.clear()

    def test_offline_tokenizer_assumption_not_cached(self, monkeypatch):
        import utils.transformers_version as tv

        monkeypatch.setattr(tv, "_env_offline", lambda: True)
        # No local file, not a local dir -> offline branch returns False without caching.
        assert _check_tokenizer_config_needs_v5("org/uncached") is False
        assert ("org/uncached", None) not in _tokenizer_class_cache

    def test_offline_then_online_refetches(self, monkeypatch):
        import utils.transformers_version as tv

        # 1) Offline: returns False, nothing cached.
        monkeypatch.setattr(tv, "_env_offline", lambda: True)
        assert _check_tokenizer_config_needs_v5("org/needs5") is False
        assert ("org/needs5", None) not in _tokenizer_class_cache

        # 2) Back online: the real fetch runs (cache was not poisoned) and is honored.
        monkeypatch.setattr(tv, "_env_offline", lambda: False)

        class _Resp:
            def read(self):
                return json.dumps({"tokenizer_class": "TokenizersBackend"}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout = 10: _Resp())
        assert _check_tokenizer_config_needs_v5("org/needs5") is True

    def test_offline_config_miss_not_cached(self, monkeypatch):
        import utils.transformers_version as tv

        monkeypatch.setattr(tv, "_env_offline", lambda: True)
        monkeypatch.setattr(tv, "_config_json_from_hf_cache", lambda name: None)
        assert _load_config_json("org/uncached-config", None) is None
        assert ("org/uncached-config", None) not in _config_json_cache


# ---------------------------------------------------------------------------
# hf_endpoint_unreachable — bounded, proxy/egress-aware reachability probe
# ---------------------------------------------------------------------------


class TestHfEndpointUnreachable:
    def test_reachable_returns_false(self, monkeypatch):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
        assert hf_endpoint_unreachable(timeout = 2) is False

    def test_gateway_error_is_unreachable(self, monkeypatch):
        import urllib.error

        def _gw(*a, **k):
            raise urllib.error.HTTPError("http://x", 504, "Gateway Timeout", {}, None)

        monkeypatch.setattr("urllib.request.urlopen", _gw)
        assert hf_endpoint_unreachable(timeout = 2) is True

    def test_other_http_status_is_reachable(self, monkeypatch):
        import urllib.error

        def _405(*a, **k):
            raise urllib.error.HTTPError("http://x", 405, "Method Not Allowed", {}, None)

        monkeypatch.setattr("urllib.request.urlopen", _405)
        assert hf_endpoint_unreachable(timeout = 2) is False

    def test_tls_failure_is_reachable(self, monkeypatch):
        import ssl
        import urllib.error

        def _tls(*a, **k):
            raise urllib.error.URLError(ssl.SSLCertVerificationError("self-signed"))

        monkeypatch.setattr("urllib.request.urlopen", _tls)
        # TLS reached the server: treat as reachable so the load surfaces the cert error.
        assert hf_endpoint_unreachable(timeout = 2) is False

    def test_dns_failure_is_unreachable(self, monkeypatch):
        import socket
        import urllib.error

        def _dns(*a, **k):
            raise urllib.error.URLError(socket.gaierror(-2, "Name or service not known"))

        monkeypatch.setattr("urllib.request.urlopen", _dns)
        assert hf_endpoint_unreachable(timeout = 2) is True

    def test_hung_probe_is_bounded(self, monkeypatch):
        import time

        def _hang(*a, **k):
            time.sleep(30)

        monkeypatch.setattr("urllib.request.urlopen", _hang)
        t0 = time.time()
        result = hf_endpoint_unreachable(timeout = 2)
        assert result is True and (time.time() - t0) < 6.0


class TestLatestTierActiveFor:
    """latest_tier_active_for: the 16-bit guard for the consented latest sidecar."""

    @staticmethod
    def _pin(
        monkeypatch,
        tv,
        version = "5.13.1",
    ):
        monkeypatch.setattr(tv, "latest_venv_pinned_version", lambda: version)
        monkeypatch.setattr(tv, "_remote_lora_base", lambda name, hf_token = None: None)

    def test_true_when_tier_latest(self, monkeypatch):
        import utils.transformers_version as tv

        self._pin(monkeypatch, tv)
        monkeypatch.setattr(tv, "get_transformers_tier", lambda *a, **k: "latest")
        assert tv.latest_tier_active_for("Zyphra/ZAYA1-8B") is True

    def test_false_for_fixed_tiers(self, monkeypatch):
        import utils.transformers_version as tv
        self._pin(monkeypatch, tv)
        for tier in ("default", "530", "550", "510"):
            monkeypatch.setattr(tv, "get_transformers_tier", lambda *a, _t = tier, **k: _t)
            assert tv.latest_tier_active_for("some/model") is False

    def test_false_without_pin_and_no_resolution(self, monkeypatch):
        """No sidecar pin returns False before any tier or network resolution."""
        import utils.transformers_version as tv

        def _boom(*a, **k):
            raise AssertionError("must not resolve without a pin")

        monkeypatch.setattr(tv, "latest_venv_pinned_version", lambda: None)
        monkeypatch.setattr(tv, "_remote_lora_base", _boom)
        monkeypatch.setattr(tv, "get_transformers_tier", _boom)
        assert tv.latest_tier_active_for("Zyphra/ZAYA1-8B") is False

    def test_never_raises(self, monkeypatch):
        import utils.transformers_version as tv

        def _boom(*a, **k):
            raise RuntimeError("tier resolution exploded")

        self._pin(monkeypatch, tv)
        monkeypatch.setattr(tv, "get_transformers_tier", _boom)
        assert tv.latest_tier_active_for("some/model") is False

    def test_remote_lora_base_is_resolved(self, monkeypatch):
        """A remote adapter is judged by its base model, like worker activation."""
        import utils.transformers_version as tv

        monkeypatch.setattr(tv, "latest_venv_pinned_version", lambda: "5.13.1")
        monkeypatch.setattr(tv, "_remote_lora_base", lambda name, hf_token = None: "Zyphra/ZAYA1-8B")
        tiers = {"Zyphra/ZAYA1-8B": "latest"}
        monkeypatch.setattr(
            tv, "get_transformers_tier", lambda name, *a, **k: tiers.get(name, "default")
        )
        assert tv.latest_tier_active_for("someuser/zaya-lora") is True

    def test_local_checkpoint_config_upgrades(self, monkeypatch, tmp_path):
        """An adapter dir with its own config.json merges tiers like activation does."""
        import utils.transformers_version as tv

        adapter = tmp_path / "ckpt"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_text("x")
        (adapter / "config.json").write_text("{}")
        self._pin(monkeypatch, tv)
        monkeypatch.setattr(tv, "_resolve_base_model", lambda name: "base/model")
        tiers = {"base/model": "default", str(adapter): "latest"}
        monkeypatch.setattr(
            tv, "get_transformers_tier", lambda name, *a, **k: tiers.get(name, "default")
        )
        assert tv.latest_tier_active_for(str(adapter)) is True


class TestLatestTierForces16Bit:
    """The inference worker and load route refuse bnb 4-bit on the latest sidecar."""

    def _read(self, rel):
        backend_dir = Path(__file__).resolve().parent.parent
        return (backend_dir / rel).read_text()

    def test_worker_guard_present(self):
        src = self._read("core/inference/worker.py")
        assert "latest_tier_active_for" in src, (
            "core/inference/worker.py must force load_in_4bit=False when "
            "latest_tier_active_for(model) is true: transformers' grouped-MoE "
            "kernels crash on bnb-quantized expert weights for brand-new "
            "architectures."
        )

    def test_route_guard_present(self):
        src = self._read("routes/inference.py")
        assert "latest_tier_active_for" in src, (
            "routes/inference.py must size the VRAM guard with the same 16-bit "
            "flip the worker applies for latest-sidecar models."
        )

    def test_validate_route_mirrors_16bit_flip(self):
        # Without the same flip, /validate sizes 4-bit and /load then 409s.
        src = self._read("routes/inference.py")
        body = src.split("async def validate_model", 1)[1].split("\nasync def ", 1)[0]
        assert "latest_tier_active_for" in body, (
            "validate_model must apply the latest-sidecar 16-bit flip before "
            "_guard_chat_load_against_training so /validate and /load agree."
        )
        # First-time loads have no pin yet, so an installable upgrade must also size 16-bit.
        assert body.index("check_upgrade_for_model") < body.index(
            "_guard_chat_load_against_training"
        ), "the upgrade check must run before the training guard"
        assert (
            "supported_in_pypi" in body.split("_guard_chat_load_against_training")[0]
        ), "an installable upgrade must force 16-bit sizing for the guard"

    def test_validate_offered_upgrade_preserves_custom_code_4bit(self):
        # A merely-offered (not installed) upgrade must NOT force 16-bit sizing when the
        # model has a custom-code (auto_map) fallback: /load loads it 4-bit without the
        # install, and the install route refuses during active training, so 16-bit sizing
        # here would 409 the only viable 4-bit path.
        src = self._read("routes/inference.py")
        body = src.split("async def validate_model", 1)[1].split("\nasync def ", 1)[0]
        flip = body.split("Mirror /load's latest-sidecar 16-bit flip", 1)[1].split(
            "_guard_chat_load_against_training", 1
        )[0]
        assert "not requires_trust_remote_code" in flip, (
            "the offered-upgrade 16-bit flip must be gated on the absence of a custom-code "
            "fallback so /validate does not 409 a 4-bit load /load would allow"
        )
        # requires_trust_remote_code must be resolved before the flip consumes it.
        assert body.index("requires_trust_remote_code = any(") < body.index(
            "not requires_trust_remote_code"
        )

    def test_install_route_guards_active_latest_workers(self):
        # Stage-and-swap replaces .venv_t5_latest in place, so a live worker on the
        # old sidecar would lazy-import files from the new version.
        src = self._read("routes/inference.py")
        body = src.split("async def install_latest_transformers_route", 1)[1].split(
            "\nasync def ", 1
        )[0]
        assert (
            "is_training_active" in body
            and "is_export_active" in body
            and "inference_lifecycle_gate" in body
        ), (
            "install_latest_transformers_route must refuse while training or export "
            "runs, and hold the lifecycle gate while unloading the chat model and "
            "swapping the sidecar."
        )
        # The unload (via before_swap so failed installs keep the model), the export-worker
        # teardown, and the install must all sit INSIDE the gate so no /load interleaves.
        assert "unload_model(active)" in body
        assert "cleanup_memory()" in body
        # Export teardown precedes the chat unload so its failure aborts with the model still loaded.
        assert body.index("cleanup_memory()") < body.index("unload_model(active)")
        assert "install_latest_transformers(" in body and "_unload_before_swap" in body
        # The gate must be owned by the shielded task, not the request coroutine: a cancelled
        # POST unwinding an async-with would release the only guard /load honors mid-install.
        gated_task = body.split("async def _gated_install", 1)[1]
        assert "inference_lifecycle_gate():" in gated_task
        assert "asyncio.to_thread(_run_install)" in gated_task
        # The reservation must be taken BEFORE the (awaitable) gate wait, or a
        # training/export start could slip in while this request queues on the gate.
        assert body.index("try_begin_sidecar_swap()") < body.index(
            "inference_lifecycle_gate():"
        ), "the swap reservation must be raised before waiting on the lifecycle gate"
        # A failed teardown must abort the swap (raise), not fall through to it.
        assert body.count("raise RuntimeError") >= 3, (
            "export, chat-unload, and idle-worker teardown failures must raise so "
            "the staged install never swaps under a live worker"
        )
        # The installer thread owns (and releases) the reservation, shielded from
        # request cancellation, so a cancelled POST cannot unlock a live swap.
        assert "asyncio.shield" in body and "end_sidecar_swap()" in body
        # In-flight generation streams predate the gate; the route refuses rather than kill them
        # via the before_swap unload. The count is rechecked UNDER the gate, since a wait on a
        # long /load outlasts the pre-gate fast path and streams take this same gate.
        assert "other_inference_request_count" in body
        gated_task = body.split("async def _gated_install", 1)[1]
        assert "other_inference_request_count" in gated_task

    def test_start_routes_refuse_during_install(self):
        # A worker spawned mid-swap could activate a half-replaced sidecar.
        training = self._read("routes/training.py")
        start = training.split("async def start_training", 1)[1].split("\nasync def ", 1)[0]
        assert (
            "is_install_in_progress" in start
        ), "training /start must refuse while a transformers install is in progress"
        export = self._read("routes/export.py")
        helper = export.split("def _ensure_export_supported", 1)[1].split("\ndef ", 1)[0]
        assert (
            "is_install_in_progress" in helper
        ), "mutating export routes must refuse while a transformers install is in progress"

    def test_spawn_sites_recheck_reservation(self):
        # The route-level guards are one-shot; validation between them and the
        # actual spawn can outlast an install's start, so the spawn itself rechecks.
        training = self._read("core/training/training.py")
        assert (
            training.count("sidecar_swap_in_progress()") >= 2
        ), "both training spawn sites must recheck the sidecar swap reservation"
        export = self._read("core/export/orchestrator.py")
        spawn = export.split("def _spawn_subprocess", 1)[1].split("\n    def ", 1)[0]
        assert (
            "sidecar_swap_kind()" in spawn
        ), "the export subprocess spawn must recheck the sidecar swap reservation"
        # Training marks the spawn active BEFORE its recheck, so either side sees the other:
        # is_training_active covers the window between proc.start() and the _proc assignment.
        assert training.index("self._spawn_in_progress = True") < training.index(
            "if sidecar_swap_in_progress():"
        )
        active = training.split("def is_training_active", 1)[1].split("\n    def ", 1)[0]
        assert "_spawn_in_progress" in active
        # Export load-checkpoint refuses BEFORE tearing down the old worker, so a
        # lost race against an install keeps the loaded checkpoint (no bare 500).
        loadck = export.split("def load_checkpoint", 1)[1].split("\n    def ", 1)[0]
        assert loadck.index("sidecar_swap_in_progress()") < loadck.index("_shutdown_subprocess()")
        # The training handshake precedes the VRAM-freeing before_spawn hook, so
        # losing the race never tears down chat/export for a run that won't spawn.
        assert training.index("self._spawn_in_progress = True") < training.index("before_spawn()")
        # The spawn-time export check is op-aware for installs (the install side
        # aborts on is_export_active) but always refuses for repairs, which have
        # no such abort and can be rebuilding the sidecar right now.
        assert (
            '_swap_kind == "repair" or (_swap_kind is not None and not self._export_active)'
            in spawn
        )


class TestSidecarSwapReservation:
    """The lazy repair takes the same reservation the install route and worker starts use."""

    def _repair_setup(self, monkeypatch, tmp_path):
        import utils.transformers_version as tv

        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(tmp_path / "venv_t5_latest"))
        monkeypatch.setattr(
            tv,
            "_latest_pin_data",
            lambda: {
                "version": "5.99.0",
                "packages": ["transformers==5.99.0"],
            },
        )
        monkeypatch.setattr(tv, "_venv_dir_is_valid", lambda d, p: False)
        monkeypatch.setattr(tv, "_env_offline", lambda: False)
        return tv

    def test_repair_holds_reservation_during_swap(self, monkeypatch, tmp_path):
        tv = self._repair_setup(monkeypatch, tmp_path)
        seen = {}

        def _fake_swap(
            version,
            packages,
            before_swap = None,
        ):
            seen["active_during_swap"] = tv.sidecar_swap_in_progress()
            return True

        monkeypatch.setattr(tv, "_stage_and_swap_latest_venv", _fake_swap)
        assert tv._ensure_venv_t5_latest_exists() is True
        assert seen["active_during_swap"] is True
        assert tv.sidecar_swap_in_progress() is False

    def test_foreign_process_lock_file_visible(self, monkeypatch, tmp_path):
        """A repair in a LIVE worker subprocess is seen (via the lock file) by this
        process, and its lock is never broken while the owner is alive."""
        import os
        import time
        import utils.transformers_version as tv

        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(tmp_path / "venv_t5_latest"))
        lock = tv._swap_lock_path()
        lock.parent.mkdir(parents = True, exist_ok = True)
        # A live owner (this process): visible and never reclaimed, even once aged past
        # the cutoff -- a slow but live pip install must keep its lock.
        lock.write_text('{"pid": %d}' % os.getpid())
        assert tv.sidecar_swap_in_progress() is True
        assert tv.try_begin_sidecar_swap() is False
        old_ts = time.time() - 3 * 60 * 60
        os.utime(lock, (old_ts, old_ts))
        assert tv.sidecar_swap_in_progress() is True
        assert tv.try_begin_sidecar_swap() is False

    def test_dead_owner_lock_reclaimed_promptly(self, monkeypatch, tmp_path):
        """A fresh lock whose recorded owner is dead is reclaimed at once, not after the
        long cutoff: a crash mid-install must not wedge loads/training/export for hours."""
        import utils.transformers_version as tv

        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(tmp_path / "venv_t5_latest"))
        lock = tv._swap_lock_path()
        lock.parent.mkdir(parents = True, exist_ok = True)
        # 999999 is not a live PID: a fresh dead-owner lock is immediately stale.
        lock.write_text('{"pid": 999999, "kind": "install"}')
        assert tv._pid_alive(999999) is False
        assert tv.sidecar_swap_in_progress() is False
        assert tv.try_begin_sidecar_swap() is True
        try:
            assert lock.is_file()
        finally:
            tv.end_sidecar_swap()
        assert not lock.exists()

    def test_unreadable_pid_lock_uses_age_cutoff(self, monkeypatch, tmp_path):
        """A lock with no readable owner PID (mid create-before-write, or corrupt) is not
        reclaimed while fresh -- only after the long cutoff -- so a lock a live owner just
        created is not stolen before its PID lands."""
        import os
        import time
        import utils.transformers_version as tv

        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(tmp_path / "venv_t5_latest"))
        lock = tv._swap_lock_path()
        lock.parent.mkdir(parents = True, exist_ok = True)
        lock.write_text("")  # created but metadata not yet written
        assert tv.sidecar_swap_in_progress() is True
        old_ts = time.time() - (tv._SWAP_LOCK_STALE_SECS + 60)
        os.utime(lock, (old_ts, old_ts))
        assert tv.sidecar_swap_in_progress() is False

    def test_repair_refused_while_install_holds_reservation(self, monkeypatch, tmp_path):
        tv = self._repair_setup(monkeypatch, tmp_path)

        def _must_not_run(*a, **k):
            raise AssertionError("repair must not swap while an install is in progress")

        monkeypatch.setattr(tv, "_stage_and_swap_latest_venv", _must_not_run)
        assert tv.try_begin_sidecar_swap() is True
        try:
            assert tv._ensure_venv_t5_latest_exists() is False
        finally:
            tv.end_sidecar_swap()


class TestRecoverStrandedSidecar:
    """A swap whose activation rename AND rollback both fail strands the previous sidecar
    at .old with no live dir (its pin marker went with it). Reading the pin self-heals it,
    but never while a swap legitimately holds the reservation."""

    def _setup(self, monkeypatch, tmp_path):
        import utils.transformers_version as tv

        live = str(tmp_path / "venv_t5_latest")
        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", live)
        # Stranded state: live gone, previous sidecar (with its marker) sits at .old.
        retired = Path(live + ".old")
        retired.mkdir(parents = True)
        (retired / tv._LATEST_PIN_MARKER).write_text(
            '{"version": "5.99.0", "packages": ["transformers==5.99.0"]}'
        )
        return tv, Path(live), retired

    def test_stranded_old_recovered_on_pin_read(self, monkeypatch, tmp_path):
        tv, live, retired = self._setup(monkeypatch, tmp_path)
        data = tv._latest_pin_data()
        assert live.is_dir()
        assert not retired.exists()
        assert data is not None and data["version"] == "5.99.0"

    def test_stranded_recovery_skipped_during_swap(self, monkeypatch, tmp_path):
        tv, live, retired = self._setup(monkeypatch, tmp_path)
        assert tv.try_begin_sidecar_swap() is True
        try:
            # A swap holds the reservation and may be mid-rename; do not race it.
            assert tv._latest_pin_data() is None
            assert not live.exists()
            assert retired.is_dir()
        finally:
            tv.end_sidecar_swap()
        # Once the swap is done, the next pin read recovers the stranded sidecar.
        assert tv._latest_pin_data() is not None
        assert live.is_dir()


class TestCachedLatestMappingRevalidated:
    """A cached 'latest' mapping is dropped and re-resolved when the sidecar since broke
    in-process, so routing self-heals instead of trusting a mapping parsed from a sidecar
    that no longer exists (which would keep routing latest-only models to a broken tier)."""

    def test_broken_sidecar_drops_cached_latest_mapping(self, monkeypatch):
        import utils.transformers_version as tv

        monkeypatch.setattr(tv, "_config_mapping_cache", {"latest": frozenset({"brandnew"})})
        monkeypatch.setattr(tv, "_latest_sidecar_intact", lambda: False)
        seen = {"n": 0}

        def _fake_overlay(tier):
            seen["n"] += 1
            return None  # broken/unavailable -> empty, uncached

        monkeypatch.setattr(tv, "_overlay_transformers_dir", _fake_overlay)
        assert tv._config_model_types("latest") == frozenset()
        assert seen["n"] == 1  # re-resolved, not served from the stale cache
        assert "latest" not in tv._config_mapping_cache

    def test_intact_sidecar_serves_cached_latest_mapping(self, monkeypatch):
        import utils.transformers_version as tv

        monkeypatch.setattr(tv, "_config_mapping_cache", {"latest": frozenset({"brandnew"})})
        monkeypatch.setattr(tv, "_latest_sidecar_intact", lambda: True)
        monkeypatch.setattr(
            tv,
            "_overlay_transformers_dir",
            lambda tier: pytest.fail("intact sidecar must serve the cache without re-resolving"),
        )
        assert tv._config_model_types("latest") == frozenset({"brandnew"})

    def test_non_latest_cache_not_revalidated(self, monkeypatch):
        import utils.transformers_version as tv

        monkeypatch.setattr(tv, "_config_mapping_cache", {"530": frozenset({"gemma3"})})
        monkeypatch.setattr(
            tv,
            "_latest_sidecar_intact",
            lambda: pytest.fail("non-latest tiers must not pay the sidecar-intact check"),
        )
        assert tv._config_model_types("530") == frozenset({"gemma3"})

    def test_deleted_pin_drops_cached_latest_mapping(self, monkeypatch, tmp_path):
        # A pin marker deleted after the mapping was cached makes _latest_pin_data None;
        # the cache must be dropped (not trusted), so routing re-resolves to no latest tier
        # rather than routing to a latest tier that then fails worker activation.
        import utils.transformers_version as tv

        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(tmp_path / "venv_t5_latest"))
        monkeypatch.setattr(tv, "_latest_tier_disabled", lambda: False)
        monkeypatch.setattr(tv, "_config_mapping_cache", {"latest": frozenset({"brandnew"})})
        # No pin marker on disk -> _latest_pin_data() is None -> not intact.
        assert tv._latest_sidecar_intact() is False
        assert tv._config_model_types("latest") == frozenset()
        assert "latest" not in tv._config_mapping_cache


class TestOverlayRepairsIncompleteSidecar:
    """Routing self-heals a pinned latest sidecar that is present but incomplete,
    not only one whose transformers/ dir vanished: workers refuse parent-only
    repairs, so a sidecar missing a pinned package would fail every load."""

    def _setup(self, monkeypatch, tmp_path, valid):
        import utils.transformers_version as tv

        live = tmp_path / "venv_t5_latest"
        (live / "transformers").mkdir(parents = True)
        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(live))
        monkeypatch.setattr(tv, "_latest_tier_disabled", lambda: False)
        monkeypatch.setattr(tv, "latest_venv_pinned_version", lambda: "5.99.0")
        monkeypatch.setattr(
            tv,
            "_latest_pin_data",
            lambda: {"version": "5.99.0", "packages": ["transformers==5.99.0", "tiktoken"]},
        )
        monkeypatch.setattr(tv, "_venv_dir_is_valid", lambda d, p: valid)
        monkeypatch.setattr(tv, "_latest_repair_failed_at", 0.0)
        return tv

    def test_incomplete_sidecar_triggers_repair(self, monkeypatch, tmp_path):
        tv = self._setup(monkeypatch, tmp_path, valid = False)
        called = {"n": 0}

        def _fake_repair():
            called["n"] += 1
            return True

        monkeypatch.setattr(tv, "_ensure_venv_t5_latest_exists", _fake_repair)
        src = tv._overlay_transformers_dir("latest")
        assert called["n"] == 1
        assert src == str(tmp_path / "venv_t5_latest" / "transformers")

    def test_intact_sidecar_skips_repair(self, monkeypatch, tmp_path):
        tv = self._setup(monkeypatch, tmp_path, valid = True)

        def _must_not_run():
            raise AssertionError("intact sidecar must not trigger a repair")

        monkeypatch.setattr(tv, "_ensure_venv_t5_latest_exists", _must_not_run)
        assert tv._overlay_transformers_dir("latest") == str(
            tmp_path / "venv_t5_latest" / "transformers"
        )

    def test_failed_repair_backs_off(self, monkeypatch, tmp_path):
        tv = self._setup(monkeypatch, tmp_path, valid = False)
        called = {"n": 0}

        def _fake_repair():
            called["n"] += 1
            return False

        monkeypatch.setattr(tv, "_ensure_venv_t5_latest_exists", _fake_repair)
        # A failed repair must not route through the broken sidecar, neither on
        # the failing attempt nor while the backoff suppresses the next attempt.
        assert tv._overlay_transformers_dir("latest") is None
        assert tv._overlay_transformers_dir("latest") is None
        assert called["n"] == 1


class TestStageAndSwapBeforeSwap:
    """before_swap fires only when the staged install succeeded and the swap is next."""

    def _setup(self, monkeypatch, tmp_path, build_ok):
        import utils.transformers_version as tv

        live = tmp_path / "venv_latest"
        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(live))

        def _fake_build(target, packages, label):
            if build_ok:
                Path(target).mkdir(parents = True, exist_ok = True)
            return build_ok

        monkeypatch.setattr(tv, "_ensure_venv_dir", _fake_build)
        return tv, live

    def test_called_after_successful_staging(self, monkeypatch, tmp_path):
        tv, live = self._setup(monkeypatch, tmp_path, build_ok = True)
        calls = []
        assert tv._stage_and_swap_latest_venv(
            "5.99.0", ("transformers==5.99.0",), before_swap = lambda: calls.append(1)
        )
        assert calls == [1] and live.is_dir()

    def test_not_called_when_staging_fails(self, monkeypatch, tmp_path):
        tv, live = self._setup(monkeypatch, tmp_path, build_ok = False)
        calls = []
        assert not tv._stage_and_swap_latest_venv(
            "5.99.0", ("transformers==5.99.0",), before_swap = lambda: calls.append(1)
        )
        assert calls == [] and not live.exists()

    def test_failure_in_before_swap_keeps_previous_sidecar(self, monkeypatch, tmp_path):
        tv, live = self._setup(monkeypatch, tmp_path, build_ok = True)
        live.mkdir()
        (live / "sentinel").write_text("old")

        def _boom():
            raise RuntimeError("worker teardown failed")

        assert not tv._stage_and_swap_latest_venv(
            "5.99.0", ("transformers==5.99.0",), before_swap = _boom
        )
        assert (live / "sentinel").read_text() == "old"


class TestKillSwitchBeatsMappingCache:
    def test_cached_latest_probe_ignored_when_disabled(self, monkeypatch):
        import utils.transformers_version as tv

        key = tv._probe_cache_key("some/model")
        monkeypatch.setitem(tv._probe_tier_cache, key, "latest")
        monkeypatch.setenv("UNSLOTH_STUDIO_NO_LATEST_TRANSFORMERS", "1")
        # With the switch set, the cached latest entry must not short-circuit;
        # the probe re-resolves against the non-latest order (stub it to 530).
        monkeypatch.setattr(tv, "_probe_tier_venvs", lambda: {})
        monkeypatch.setattr(tv, "_probe_tier_order", lambda: ())
        assert tv._probe_tier("some/model", None, "test") != "latest"
        # Cached non-latest entries and the unset switch still short-circuit.
        monkeypatch.delenv("UNSLOTH_STUDIO_NO_LATEST_TRANSFORMERS")
        assert tv._probe_tier("some/model", None, "test") == "latest"

    def test_cached_latest_mapping_ignored_when_disabled(self, monkeypatch):
        import utils.transformers_version as tv

        monkeypatch.setitem(tv._config_mapping_cache, "latest", frozenset({"brandnew"}))
        # The cache is trusted only when the sidecar is intact; hold it intact so this
        # test isolates the kill switch, not the sidecar-revalidation path.
        monkeypatch.setattr(tv, "_latest_sidecar_intact", lambda: True)
        monkeypatch.setenv("UNSLOTH_STUDIO_NO_LATEST_TRANSFORMERS", "1")
        assert tv._config_model_types("latest") == frozenset()
        monkeypatch.delenv("UNSLOTH_STUDIO_NO_LATEST_TRANSFORMERS")
        assert tv._config_model_types("latest") == frozenset({"brandnew"})


class TestRaiseTierForNested:
    """_raise_tier_for_nested: a wrapper's nested model_type can raise a fast-path tier."""

    def _patch_types(self, monkeypatch, per_tier):
        import utils.transformers_version as tv
        monkeypatch.setattr(
            tv, "_config_model_types", lambda tier: frozenset(per_tier.get(tier, ()))
        )

    def test_nested_latest_only_type_raises(self, monkeypatch):
        import utils.transformers_version as tv

        self._patch_types(monkeypatch, {"550": {"gemma4"}, "latest": {"gemma4", "brandnew_arch"}})
        cfg = {"model_type": "gemma4", "text_config": {"model_type": "brandnew_arch"}}
        assert tv._raise_tier_for_nested(cfg, "550") == "latest"

    def test_never_lowers_a_fast_path_tier(self, monkeypatch):
        import utils.transformers_version as tv

        # Mapping alone would say 530, but the fast path (e.g. a name override) said 550.
        self._patch_types(monkeypatch, {"530": {"qwen3_5"}, "550": {"qwen3_5"}})
        assert tv._raise_tier_for_nested({"model_type": "qwen3_5"}, "550") == "550"

    def test_no_config_keeps_tier(self):
        import utils.transformers_version as tv
        assert tv._raise_tier_for_nested(None, "550") == "550"

    def test_unknown_nested_type_never_vetoes(self, monkeypatch):
        import utils.transformers_version as tv

        # A nested type unknown everywhere (not even latest) keeps the fast path.
        self._patch_types(monkeypatch, {"550": {"gemma4"}, "latest": {"gemma4"}})
        cfg = {"model_type": "gemma4", "text_config": {"model_type": "unreleased"}}
        assert tv._raise_tier_for_nested(cfg, "550") == "550"

    def test_name_fast_path_folds_when_latest_pinned(self, monkeypatch):
        """A fixed-tier name match with a latest-only model_type routes to latest
        once the sidecar is pinned; without a pin the name tier stands (no I/O)."""
        import utils.transformers_version as tv

        self._patch_types(monkeypatch, {"550": {"gemma4"}, "latest": {"brandnew_arch"}})
        monkeypatch.setattr(tv, "_tier_from_name", lambda name: ("550", "gemma-4"))
        monkeypatch.setattr(
            tv, "_load_config_json", lambda name, tok = None: {"model_type": "brandnew_arch"}
        )
        monkeypatch.setattr(tv, "latest_venv_pinned_version", lambda: "5.99.0")
        assert tv.get_transformers_tier("org/gemma-4-new", probe = False) == "latest"
        monkeypatch.setattr(tv, "latest_venv_pinned_version", lambda: None)
        monkeypatch.setattr(
            tv,
            "_load_config_json",
            lambda name, tok = None: (_ for _ in ()).throw(AssertionError("no I/O without a pin")),
        )
        assert tv.get_transformers_tier("org/gemma-4-new", probe = False) == "550"

    def test_fast_path_folds_nested_tier(self, monkeypatch, tmp_path):
        """End to end: a local wrapper config on a fixed fast path routes to latest
        when its nested type only exists in the installed latest sidecar."""
        import utils.transformers_version as tv

        ckpt = tmp_path / "wrapper"
        ckpt.mkdir()
        (ckpt / "config.json").write_text(
            json.dumps({"model_type": "gemma4", "text_config": {"model_type": "brandnew_arch"}})
        )
        self._patch_types(monkeypatch, {"550": {"gemma4"}, "latest": {"gemma4", "brandnew_arch"}})
        monkeypatch.setattr(tv, "_config_needs_510", lambda cfg: False)
        monkeypatch.setattr(tv, "_config_needs_550", lambda cfg: True)
        assert tv.get_transformers_tier(str(ckpt), probe = False) == "latest"
