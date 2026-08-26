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


def _make_backend(
    effective_ctx = 98304,
    port = 51234,
    api_key = None,
):
    inst = LlamaCppBackend.__new__(LlamaCppBackend)
    inst._port = port
    # __init__ always sets this; __new__ skips it, and the readback reads it via
    # _auth_headers to authenticate against a --api-key child server.
    inst._api_key = api_key
    inst._effective_context_length = effective_ctx
    inst._context_length = 262144
    inst._effective_parallel_slots = 1
    inst._kv_cache_unified = False
    inst._kv_cache_context_total = None
    return inst


def _stub_props(
    monkeypatch,
    status_code = 200,
    body = None,
    exc = None,
):
    def fake_get(
        url,
        headers = None,
        timeout = None,
        trust_env = None,
    ):
        assert url.endswith("/props")

        assert trust_env is False
        # /props sits behind llama-server's api-key middleware, so a direct-stream
        # child must be addressed with the bearer token; without one the header
        # stays absent rather than becoming a bogus "Bearer None".
        assert headers is None or headers == {"Authorization": "Bearer test-key"}
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


def test_props_keeps_total_cache_context_for_slot_preflight(monkeypatch):
    inst = _make_backend(effective_ctx = 32768)
    inst._effective_parallel_slots = 4
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 8192}},
    )
    inst._reconcile_effective_ctx_with_server()
    assert inst._effective_context_length == 8192
    assert inst._kv_cache_context_total == 32768


def test_props_does_not_multiply_unified_cache_context(monkeypatch):
    inst = _make_backend(effective_ctx = 32768)
    inst._effective_parallel_slots = 4
    inst._kv_cache_unified = True
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 32768}},
    )
    inst._reconcile_effective_ctx_with_server()
    assert inst._effective_context_length == 32768
    assert inst._kv_cache_context_total == 32768


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
    inst._reconcile_effective_ctx_with_server(requested_n_ctx = 32768)
    assert inst._effective_context_length == 32768


def test_explicit_extra_arg_ctx_adopts_larger_confirmed_server_value(monkeypatch):
    """A trailing --ctx-size can override Studio's earlier VRAM-fit ``-c``.

    The resolved explicit request is 100352, Studio's pre-launch estimate is
    65983, and /props confirms that llama-server actually allocated 100352.
    Publish the real window while retaining the VRAM warning threshold.
    """
    inst = _make_backend(effective_ctx = 65983)
    inst._max_context_length = 65983
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 100352}},
    )

    inst._reconcile_effective_ctx_with_server(requested_n_ctx = 100352)

    assert inst._effective_context_length == 100352
    assert inst.context_length == 100352
    assert inst.max_context_length == 65983


def test_no_explicit_flag_never_adopts_a_larger_server_value(monkeypatch):
    """The ceiling is the pass-through flag, not the first-class field.

    Only a --ctx-size emitted after Studio's own -c can make the child allocate
    past the fit, so a load with no flag passes 0 and llama.cpp's own context
    padding cannot be reported as an override the user never wrote.
    """
    inst = _make_backend(effective_ctx = 65983)
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 66048}},
    )

    inst._reconcile_effective_ctx_with_server(requested_n_ctx = 0)

    assert inst._effective_context_length == 65983


def test_unset_effective_ctx_adopts_server_value(monkeypatch):
    inst = _make_backend(effective_ctx = None)
    inst._context_length = None
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 40960}},
    )
    inst._reconcile_effective_ctx_with_server()
    assert inst._effective_context_length == 40960


def test_unset_effective_ctx_still_honours_the_explicit_ceiling(monkeypatch):
    """The unset arm publishes too, so the same ceiling has to bind there."""
    inst = _make_backend(effective_ctx = None)
    inst._context_length = None
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 8192}},
    )

    inst._reconcile_effective_ctx_with_server(requested_n_ctx = 4096)

    assert inst._effective_context_length == 4096


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
    flags = LlamaCppBackend._ctx_integrity_flags(4, False, False, 98304, 98304, _CAPS_ALL)
    assert "--kv-unified" in flags


def test_kv_unified_skipped_for_single_slot_or_old_build():
    assert "--kv-unified" not in LlamaCppBackend._ctx_integrity_flags(
        1, False, False, 98304, 98304, _CAPS_ALL
    )
    assert "--kv-unified" not in LlamaCppBackend._ctx_integrity_flags(
        4, False, False, 98304, 98304, _CAPS_NONE
    )


def test_fit_ctx_floors_explicit_request_under_fit():
    # An explicit requested ctx floors --fit-ctx at that value on any --fit
    # path, including legacy auto (auto_fit False).
    flags = LlamaCppBackend._ctx_integrity_flags(1, True, False, 98304, 98304, _CAPS_ALL)
    assert flags[flags.index("--fit-ctx") + 1] == "98304"


def test_fit_ctx_skipped_without_fit_or_support():
    # No --fit on -> no --fit-ctx.
    assert "--fit-ctx" not in LlamaCppBackend._ctx_integrity_flags(
        1, False, False, 98304, 98304, _CAPS_ALL
    )
    # --fit on but the binary doesn't support --fit-ctx.
    assert "--fit-ctx" not in LlamaCppBackend._ctx_integrity_flags(
        1, True, True, 98304, 98304, _CAPS_NONE
    )


def test_fit_ctx_floors_auto_request_at_8192_only_under_auto_fit():
    # Manual + Auto (auto_fit) floors the auto window at 8192 so --fit can't
    # shrink it to a tiny size.
    flags = LlamaCppBackend._ctx_integrity_flags(1, True, True, 0, 262144, _CAPS_ALL)
    assert flags[flags.index("--fit-ctx") + 1] == "8192"
    # Legacy auto (fit on but not auto_fit) emits -c 0 to pin native, so the
    # 8192 floor must NOT ride along and override that pin.
    assert "--fit-ctx" not in LlamaCppBackend._ctx_integrity_flags(
        1, True, False, 0, 262144, _CAPS_ALL
    )


def test_probe_missing_binary_reports_new_capabilities_false():
    info = LlamaCppBackend.probe_server_capabilities(binary = "/nonexistent/llama-server")
    assert info["found"] is False
    assert info["supports_kv_unified"] is False
    assert info["supports_fit_ctx"] is False
