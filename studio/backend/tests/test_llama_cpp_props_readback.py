# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the post-launch /props context readback.

llama-server's memory-fit step or --parallel slot split can allocate less
context than the requested -c while Unsloth keeps advertising the requested
value; clients sized to it then die on exceed_context_size_error 400s.
``_reconcile_effective_ctx_with_server`` must adopt the server's real
``default_generation_settings.n_ctx`` whenever it is smaller.

Stubbed httpx; no subprocess, GPU, or network. Cross-platform.
"""

from __future__ import annotations

import json
import sys
import types as _types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Stub heavy/unavailable deps before importing the module under test.
# Mirrors test_llama_cpp_context_fit.py.
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Prefer the real modules so importing this file first cannot poison later
# test modules with stubs; only stub what the environment genuinely lacks.
try:
    import loggers  # noqa: F401
except ImportError:
    _loggers_stub = _types.ModuleType("loggers")
    _loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
    sys.modules.setdefault("loggers", _loggers_stub)

try:
    import structlog  # noqa: F401
except ImportError:
    sys.modules.setdefault("structlog", _types.ModuleType("structlog"))

try:
    import httpx  # noqa: F401
except ImportError:
    _httpx_stub = _types.ModuleType("httpx")
    for _exc_name in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
        "WriteError",
        "HTTPError",
    ):
        setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))

    class _FakeTimeout:
        def __init__(self, *a, **kw):
            pass

    _httpx_stub.Timeout = _FakeTimeout
    _httpx_stub.Client = type(
        "Client",
        (),
        {
            "__init__": lambda self, **kw: None,
            "__enter__": lambda self: self,
            "__exit__": lambda self, *a: None,
        },
    )
    _httpx_stub.get = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("unstubbed httpx.get"))
    sys.modules.setdefault("httpx", _httpx_stub)

from core.inference.llama_cpp import LlamaCppBackend
import core.inference.llama_cpp as llama_cpp_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(
        self,
        status_code = 200,
        body = None,
    ):
        self.status_code = status_code
        self._body = body or {}

    def json(self):
        return self._body


def _make_backend(effective_ctx = 98304, port = 51234):
    inst = LlamaCppBackend.__new__(LlamaCppBackend)
    inst._port = port
    inst._effective_context_length = effective_ctx
    inst._context_length = 262144
    return inst


def _stub_props(
    monkeypatch,
    status_code = 200,
    body = None,
    exc = None,
):
    def fake_get(
        url,
        timeout = None,
        trust_env = None,
    ):
        assert url.endswith("/props")

        assert trust_env is False
        if exc is not None:
            raise exc
        return _FakeResponse(status_code, body)

    monkeypatch.setattr(llama_cpp_mod.httpx, "get", fake_get, raising = False)


# ---------------------------------------------------------------------------
# _query_server_n_ctx parsing
# ---------------------------------------------------------------------------


def test_query_n_ctx_reads_default_generation_settings(monkeypatch):
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 67584}},
    )
    assert _make_backend()._query_server_n_ctx() == 67584


def test_query_n_ctx_non_200_returns_none(monkeypatch):
    _stub_props(monkeypatch, status_code = 503)
    assert _make_backend()._query_server_n_ctx() is None


def test_query_n_ctx_missing_key_returns_none(monkeypatch):
    _stub_props(monkeypatch, body = {"default_generation_settings": {}})
    assert _make_backend()._query_server_n_ctx() is None


def test_query_n_ctx_swallows_transport_errors(monkeypatch):
    _stub_props(monkeypatch, exc = RuntimeError("connection refused"))
    assert _make_backend()._query_server_n_ctx() is None


# ---------------------------------------------------------------------------
# _reconcile_effective_ctx_with_server decisions
# ---------------------------------------------------------------------------


def test_fit_shrunk_ctx_overwrites_advertised_value(monkeypatch):
    """The Nick repro: requested/advertised 98304, server really at 67584."""
    inst = _make_backend(effective_ctx = 98304)
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 67584}},
    )
    inst._reconcile_effective_ctx_with_server()
    assert inst._effective_context_length == 67584
    assert inst.context_length == 67584


def test_matching_ctx_is_left_alone(monkeypatch):
    inst = _make_backend(effective_ctx = 98304)
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 98304}},
    )
    inst._reconcile_effective_ctx_with_server()
    assert inst._effective_context_length == 98304


def test_larger_server_ctx_does_not_inflate_advertised_value(monkeypatch):
    """Never advertise more than the user asked for, even if the server could."""
    inst = _make_backend(effective_ctx = 32768)
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 65536}},
    )
    inst._reconcile_effective_ctx_with_server()
    assert inst._effective_context_length == 32768


def test_unset_effective_ctx_adopts_server_value(monkeypatch):
    inst = _make_backend(effective_ctx = None)
    inst._context_length = None
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 40960}},
    )
    inst._reconcile_effective_ctx_with_server()
    assert inst._effective_context_length == 40960


def test_props_failure_keeps_studio_value(monkeypatch):
    """A flaky /props must never wipe the computed context."""
    inst = _make_backend(effective_ctx = 98304)
    _stub_props(monkeypatch, exc = RuntimeError("boom"))
    inst._reconcile_effective_ctx_with_server()
    assert inst._effective_context_length == 98304


# ---------------------------------------------------------------------------
# _ctx_integrity_flags: keep the per-request window equal to the advertised ctx
# ---------------------------------------------------------------------------

_CAPS_ALL = {"supports_kv_unified": True, "supports_fit_ctx": True}
_CAPS_NONE = {"supports_kv_unified": False, "supports_fit_ctx": False}


def test_kv_unified_added_for_multi_slot():
    """Explicit --parallel N disables llama-server's auto-slots kv-unified
    default, splitting -c into per-slot windows of -c/N; Unsloth must restore
    the shared pool so one request can use the full advertised context."""
    flags = LlamaCppBackend._ctx_integrity_flags(4, False, 98304, 98304, _CAPS_ALL)
    assert "--kv-unified" in flags


def test_kv_unified_skipped_for_single_slot_or_old_build():
    assert "--kv-unified" not in LlamaCppBackend._ctx_integrity_flags(
        1, False, 98304, 98304, _CAPS_ALL
    )
    assert "--kv-unified" not in LlamaCppBackend._ctx_integrity_flags(
        4, False, 98304, 98304, _CAPS_NONE
    )


def test_fit_ctx_floors_explicit_request_under_fit():
    flags = LlamaCppBackend._ctx_integrity_flags(1, True, 98304, 98304, _CAPS_ALL)
    assert flags[flags.index("--fit-ctx") + 1] == "98304"


def test_fit_ctx_skipped_without_fit_or_explicit_ctx_or_support():
    assert "--fit-ctx" not in LlamaCppBackend._ctx_integrity_flags(
        1, False, 98304, 98304, _CAPS_ALL
    )
    assert "--fit-ctx" not in LlamaCppBackend._ctx_integrity_flags(1, True, 0, 262144, _CAPS_ALL)
    assert "--fit-ctx" not in LlamaCppBackend._ctx_integrity_flags(
        1, True, 98304, 98304, _CAPS_NONE
    )


def test_probe_missing_binary_reports_new_capabilities_false():
    info = LlamaCppBackend.probe_server_capabilities(binary = "/nonexistent/llama-server")
    assert info["found"] is False
    assert info["supports_kv_unified"] is False
    assert info["supports_fit_ctx"] is False
