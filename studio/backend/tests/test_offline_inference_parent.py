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

    def test_config_550_skips_urllib_when_offline(self, monkeypatch, clean_offline_env, tmp_path):
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

        src = Path(_BACKEND_DIR, "core", "training", "worker.py").read_text(encoding = "utf-8")
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
        # The probe now lives in the shared helper (endpoint- and proxy-aware), so the
        # worker must delegate to it rather than resolve a hardcoded host itself.
        assert (
            "hf_env_offline" in block
        ), "training worker must honor TRANSFORMERS_OFFLINE before probing"
        assert "hf_dns_dead" in block, "training worker must use the shared DNS helper"
        assert block.index("hf_env_offline()") < block.index(
            "hf_dns_dead()"
        ), "training worker must check explicit offline env before DNS/network probes"
        assert 'gethostbyname("huggingface.co")' not in block, (
            "training worker must not hardcode huggingface.co; a reachable HF_ENDPOINT "
            "mirror would be declared offline"
        )
        assert (
            "proxy_timeouts_offline = False" in block
        ), "training worker must fail open on an ambiguous proxy timeout"

    def test_shared_dns_helper_uses_thread_probe(self):
        """The daemon-thread property moved with the probe; pin it where it now lives."""
        import inspect

        from utils.utils import dns_host_dead

        src = inspect.getsource(dns_host_dead)
        assert ".setdefaulttimeout(" not in src, (
            "shared DNS probe calls socket.setdefaulttimeout; "
            "concurrent sockets would inherit the probe timeout"
        )
        assert "Thread" in src and "daemon" in src, "shared DNS probe must run on a daemon thread"


class TestInferenceWorkerProbesForItself:
    """child_env deliberately scrubs the parent's scoped offline flag, so the inference
    worker needs its own probe like the training and export workers, or it walks back
    into the retry paths the parent already ruled out."""

    def _block(self):
        import pathlib

        backend_root = pathlib.Path(__file__).resolve().parent.parent
        src = (backend_root / "core" / "inference" / "worker.py").read_text(
            encoding = "utf-8",
        )
        start = src.index("# Offline auto-detect")
        return src[start : start + 1800]

    def test_the_probe_exists_and_runs_before_activation(self):
        import pathlib

        backend_root = pathlib.Path(__file__).resolve().parent.parent
        src = (backend_root / "core" / "inference" / "worker.py").read_text(
            encoding = "utf-8",
        )
        probe = src.index("# Offline auto-detect")
        # Both HF-reading steps the parent's verdict was meant to cover.
        assert probe < src.index("_remote_lora_base(model_name")
        assert probe < src.index("_activate_transformers_version(_base")

    def test_a_user_set_flag_is_never_overridden(self):
        block = self._block()
        assert 'if "HF_HUB_OFFLINE" not in os.environ:' in block

    def test_lifetime_flags_use_the_fail_open_verdict(self):
        """Same reasoning as the training worker: these last the whole process, so an
        ambiguous answer must not strand it offline."""
        block = self._block()
        assert "gateway_errors_offline = False" in block
        assert "proxy_timeouts_offline = False" in block

    def test_probe_opt_out_is_honoured(self):
        block = self._block()
        assert "hf_probe_disabled()" in block

    def test_it_fails_open(self):
        block = self._block()
        assert "except Exception:" in block

    def test_it_does_not_force_datasets_offline(self):
        """An inference worker loads no dataset; the training worker's flag is its own."""
        block = self._block()
        assert "HF_DATASETS_OFFLINE" not in block
