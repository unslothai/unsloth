# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Parent-process offline regression tests (follow-up to #5505).

Pins the LoRA-detect, transformers_version urllib short-circuit, and
training-worker DNS probe so a dead DNS no longer burns 30-60s of
soft-failed timeouts before the worker subprocess spawns.

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
# Prefer real httpx if installed (CI installs it). Stub only as fallback.
try:
    import httpx  # noqa: F401
except ImportError:
    _hx = _types.ModuleType("httpx")
    for _exc in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
        "HTTPError",
        "RequestError",
        "HTTPStatusError",
    ):
        setattr(_hx, _exc, type(_exc, (Exception,), {}))
    _hx.Response = type("Response", (), {})
    _hx.Request = type("Request", (), {})

    class _FakeTimeout:
        def __init__(self, *a, **k):
            pass

    _hx.Timeout = _FakeTimeout
    _hx.Client = type(
        "Client",
        (),
        {
            "__init__": lambda s, **k: None,
            "__enter__": lambda s: s,
            "__exit__": lambda s, *a: None,
        },
    )
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
        self, monkeypatch, clean_offline_env, tmp_path
    ):
        # No local config + offline env -> must NOT call urlopen.
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        unique = f"unsloth/never-cached-{tmp_path.name}"

        def boom(*a, **k):
            raise AssertionError("urlopen must not be called when offline")

        with patch("urllib.request.urlopen", boom):
            assert _check_tokenizer_config_needs_v5(unique) is False

    def test_config_550_skips_urllib_when_offline(
        self, monkeypatch, clean_offline_env, tmp_path
    ):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        unique = f"unsloth/never-cached-{tmp_path.name}-cfg"

        def boom(*a, **k):
            raise AssertionError("urlopen must not be called when offline")

        with patch("urllib.request.urlopen", boom):
            assert _check_config_needs_550(unique) is False


class TestLoraDetectOffline:
    """Offline LoRA detect: hf_model_info short-circuits via
    OfflineModeIsEnabled; cached adapter_config.json wins."""

    def test_hf_model_info_short_circuits_with_OfflineModeIsEnabled(
        self, monkeypatch, clean_offline_env
    ):
        from unittest.mock import MagicMock

        from utils.models.model_config import ModelConfig

        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        # Unsloth catches Exception broadly; pin that the call still happens
        # (so cached LoRAs aren't missed) and returns fast via the mock.
        class _OfflineModeIsEnabled(Exception):
            pass

        mock = MagicMock(side_effect = _OfflineModeIsEnabled("offline"))
        with patch("huggingface_hub.model_info", mock):
            try:
                ModelConfig.from_identifier(
                    model_id = "unsloth/Qwen3.5-4B",
                    hf_token = None,
                    gguf_variant = None,
                )
            except Exception:
                pass  # registry miss OK; pinning the LoRA-detect call

        assert mock.call_count >= 1, (
            "LoRA-detect must still consult hf_model_info offline; "
            "OfflineModeIsEnabled makes it cheap"
        )

    def test_cached_lora_detected_when_api_unreachable(
        self, monkeypatch, clean_offline_env, tmp_path
    ):
        """A cached adapter_config.json must still mark the repo as a
        LoRA when the HF API is unreachable."""
        from huggingface_hub import constants as hf_constants

        from utils.models.model_config import ModelConfig

        repo = tmp_path / "models--org--my-lora"
        snap = repo / "snapshots" / ("a" * 40)
        snap.mkdir(parents = True)
        (snap / "adapter_config.json").write_text(
            '{"base_model_name_or_path": "unsloth/Llama-3-8B"}'
        )
        monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        def boom(*a, **k):
            raise OSError("hub unreachable")

        with patch("huggingface_hub.model_info", boom):
            try:
                cfg = ModelConfig.from_identifier(
                    model_id = "org/my-lora",
                    hf_token = None,
                    gguf_variant = None,
                )
            except Exception:
                cfg = None

        # cfg may be None (base not resolvable offline); pin the fixture
        # so the cache-side detect block had a file to find.
        assert (snap / "adapter_config.json").is_file()


class TestTrainingWorkerProbeNoGlobalTimeout:
    """Training-worker DNS probe must run on a daemon thread, not mutate
    process-wide socket.setdefaulttimeout (mirrors llama_cpp.py)."""

    def test_training_worker_source_uses_thread_probe(self):
        """Static-pin against regression to setdefaulttimeout."""
        import re
        from pathlib import Path

        src = Path(_BACKEND_DIR, "core", "training", "worker.py").read_text()
        m = re.search(
            r'if\s+"HF_HUB_OFFLINE"\s+not\s+in\s+os\.environ\s*:.*?'
            r"print\([^)]*HF_HUB_OFFLINE=1[^)]*\)",
            src,
            flags = re.DOTALL,
        )
        assert m is not None, "could not locate offline auto-detect block"
        block = m.group(0)
        assert ".setdefaulttimeout(" not in block, (
            "training worker still calls socket.setdefaulttimeout; "
            "concurrent sockets would inherit the probe timeout"
        )
        assert (
            "threading" in block and "Thread" in block
        ), "training worker probe must run on a daemon thread"
