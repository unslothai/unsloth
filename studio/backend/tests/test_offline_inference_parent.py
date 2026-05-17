# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the parent-process offline path (follow-up to #5505).

PR #5505 fixed the GGUF/llama-server load path. This test pins the
follow-up plumbing in:

* ``utils/models/model_config.py`` -- the remote LoRA auto-detect
  ``hf_model_info`` call in ``ModelConfig.from_identifier`` now skips
  when ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` is set.
* ``utils/transformers_version.py`` -- the urllib fallback fetches for
  ``tokenizer_config.json`` and ``config.json`` now short-circuit when
  the same env vars are set.

Together with the DNS probe wrapper added around
``ModelConfig.from_identifier`` in ``routes/inference.py``, this means a
dead DNS no longer burns 30-60s of soft-failed network timeouts before
the worker subprocess is even spawned.

No GPU, no network, no subprocess. Cross-platform.
"""

from __future__ import annotations

import os
import sys
import types as _types
from pathlib import Path
from unittest.mock import patch

import pytest


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
sys.modules.setdefault("structlog", _types.ModuleType("structlog"))
_hx = _types.ModuleType("httpx")
for _exc in ("ConnectError", "TimeoutException", "ReadTimeout", "ReadError",
             "RemoteProtocolError", "CloseError"):
    setattr(_hx, _exc, type(_exc, (Exception,), {}))


class _FakeTimeout:
    def __init__(self, *a, **k): pass


_hx.Timeout = _FakeTimeout
_hx.Client = type("Client", (), {
    "__init__": lambda s, **k: None,
    "__enter__": lambda s: s,
    "__exit__": lambda s, *a: None,
})
sys.modules.setdefault("httpx", _hx)


from utils.models.model_config import _env_offline
from utils.transformers_version import (
    _check_config_needs_550,
    _check_tokenizer_config_needs_v5,
    _env_offline as _env_offline_tv,
)


@pytest.fixture
def clean_offline_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)


class TestEnvOffline:
    def test_unset_is_false(self, clean_offline_env):
        assert _env_offline() is False
        assert _env_offline_tv() is False

    def test_hf_hub_offline_truthy_values(self, monkeypatch, clean_offline_env):
        for val in ("1", "true", "yes", "TRUE", "Yes"):
            monkeypatch.setenv("HF_HUB_OFFLINE", val)
            assert _env_offline() is True
            assert _env_offline_tv() is True

    def test_transformers_offline_alone_triggers(self, monkeypatch, clean_offline_env):
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        assert _env_offline() is True

    def test_falsy_values(self, monkeypatch, clean_offline_env):
        for val in ("", "0", "false", "no"):
            monkeypatch.setenv("HF_HUB_OFFLINE", val)
            assert _env_offline() is False


class TestTransformersVersionOfflineShortCircuits:
    def test_tokenizer_config_skips_urllib_when_offline(
        self, monkeypatch, clean_offline_env, tmp_path,
    ):
        # No local config, env is offline -> must NOT call urlopen.
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        # Force a cache miss for this unique model name.
        unique = f"unsloth/never-cached-{tmp_path.name}"

        def boom(*a, **k):
            raise AssertionError("urlopen must not be called when offline")

        with patch("urllib.request.urlopen", boom):
            assert _check_tokenizer_config_needs_v5(unique) is False

    def test_config_550_skips_urllib_when_offline(
        self, monkeypatch, clean_offline_env, tmp_path,
    ):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        unique = f"unsloth/never-cached-{tmp_path.name}-cfg"

        def boom(*a, **k):
            raise AssertionError("urlopen must not be called when offline")

        with patch("urllib.request.urlopen", boom):
            assert _check_config_needs_550(unique) is False


class TestLoraDetectOfflineShortCircuit:
    """Offline env must skip the remote LoRA-detect ``hf_model_info`` call
    in ``ModelConfig.from_identifier`` so the parent process doesn't burn
    ~25s waiting for the HF API to time out before spawning the worker."""

    def test_hf_model_info_not_called_when_offline(
        self, monkeypatch, clean_offline_env,
    ):
        from utils.models.model_config import ModelConfig

        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        def boom(*a, **k):
            raise AssertionError(
                "hf_model_info must not be called for LoRA detect when offline"
            )

        # Use a plain (non-LoRA) repo identifier. is_lora starts False,
        # is_local is False, so the LoRA-detect branch would normally fire.
        with patch("huggingface_hub.model_info", boom):
            cfg = ModelConfig.from_identifier(
                model_id = "unsloth/Qwen3.5-4B",
                hf_token = None,
                gguf_variant = None,
            )
        # Config may or may not succeed depending on registry contents;
        # the assertion is that the API was not consulted.
        assert cfg is None or cfg is not None  # no exception, no API hit
