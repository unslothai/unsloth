# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend contract for the GGUF reload duplicate-load guard.

``LlamaCppBackend._already_in_target_state`` is the in-process
short-circuit that prevents a serialised duplicate /load from killing
the just-spawned llama-server. These tests pin the local-file
identity, the HF-mode hf_variant fallback, and the ``extra_args``
None-vs-[] inherit semantics so the guard cannot silently regress.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

_httpx_stub = _types.ModuleType("httpx")
for _exc in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc, type(_exc, (Exception,), {}))
_httpx_stub.Timeout = type("T", (), {"__init__": lambda s, *a, **k: None})
_httpx_stub.Client = type(
    "C",
    (),
    {
        "__init__": lambda s, **kw: None,
        "__enter__": lambda s: s,
        "__exit__": lambda s, *a: None,
    },
)
sys.modules.setdefault("httpx", _httpx_stub)

from core.inference.llama_cpp import LlamaCppBackend


class _FakeProcess:
    """Stand-in for subprocess.Popen so atexit cleanup doesn't crash."""

    def terminate(self):
        pass

    def wait(self, timeout = None):
        return 0

    def kill(self):
        pass

    def poll(self):
        return 0


def _loaded_backend(**overrides):
    backend = LlamaCppBackend()
    backend._process = _FakeProcess()  # is_loaded only checks "is not None"
    backend._healthy = True
    backend._model_identifier = "owner/repo"
    backend._hf_variant = "Q4_K_M"
    backend._requested_n_ctx = 8192
    backend._cache_type_kv = None
    backend._speculative_type = None
    backend._chat_template_override = None
    backend._is_vision = False
    backend._extra_args = None
    backend._extra_args_source = None
    backend._gguf_path = None
    for key, value in overrides.items():
        setattr(backend, key, value)
    return backend


# ── Local-file identity via gguf_path ────────────────────────────────


def test_already_in_target_state_uses_gguf_path_when_present(tmp_path):
    gguf_file = tmp_path / "model.Q4_K_M.gguf"
    gguf_file.write_bytes(b"")
    backend = _loaded_backend(
        _hf_variant = "Q4_K_M",
        _gguf_path = str(gguf_file),
    )
    assert (
        backend._already_in_target_state(
            gguf_path = str(gguf_file),
            model_identifier = "owner/repo",
            hf_variant = None,
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = None,
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
        )
        is True
    )


def test_already_in_target_state_rejects_different_gguf_path(tmp_path):
    a = tmp_path / "a.gguf"
    a.write_bytes(b"")
    b = tmp_path / "b.gguf"
    b.write_bytes(b"")
    backend = _loaded_backend(_gguf_path = str(a))
    assert (
        backend._already_in_target_state(
            gguf_path = str(b),
            model_identifier = "owner/repo",
            hf_variant = None,
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = None,
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
        )
        is False
    )


# ── HF mode falls back to hf_variant comparison ──────────────────────


def test_already_in_target_state_falls_back_to_hf_variant_for_hf_loads():
    backend = _loaded_backend(_hf_variant = "Q4_K_M", _gguf_path = None)
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "owner/repo",
            hf_variant = "Q8_0",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = None,
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
        )
        is False
    )


def test_already_in_target_state_hf_same_variant_matches():
    backend = _loaded_backend(_hf_variant = "Q4_K_M", _gguf_path = None)
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "owner/repo",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = None,
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
        )
        is True
    )


# ── extra_args: None inherits, [] forces reload, list enforces ───────


def test_already_in_target_state_none_extras_inherits_stored():
    backend = _loaded_backend(_extra_args = ["--top-k", "20"])
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "owner/repo",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = None,
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
        )
        is True
    )


def test_already_in_target_state_empty_extras_forces_reload_when_stored():
    backend = _loaded_backend(_extra_args = ["--top-k", "20"])
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "owner/repo",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = None,
            chat_template_override = None,
            extra_args = [],
            is_vision = False,
        )
        is False
    )


def test_already_in_target_state_explicit_extras_match():
    backend = _loaded_backend(_extra_args = ["--top-k", "20"])
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "owner/repo",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = None,
            chat_template_override = None,
            extra_args = ["--top-k", "20"],
            is_vision = False,
        )
        is True
    )


def test_extra_args_source_default_is_none():
    backend = LlamaCppBackend()
    assert backend.extra_args_source is None
