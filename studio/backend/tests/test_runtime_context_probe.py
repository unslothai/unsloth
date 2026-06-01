# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for post-load runtime context probing on LlamaCppBackend."""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)

_httpx_stub = _types.ModuleType("httpx")
for _exc_name in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))


class _FakeTimeout:
    def __init__(self, *a, **kw):
        pass


_httpx_stub.Timeout = _FakeTimeout
_httpx_stub.get = lambda *a, **kw: None
_httpx_stub.Client = type(
    "Client",
    (),
    {
        "__init__": lambda self, **kw: None,
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: None,
    },
)
sys.modules.setdefault("httpx", _httpx_stub)

from core.inference.llama_cpp import LlamaCppBackend


def _backend(**kwargs):
    inst = LlamaCppBackend.__new__(LlamaCppBackend)
    inst._port = kwargs.get("port", 48507)
    inst._stdout_lines = list(kwargs.get("stdout_lines", []))
    inst._launch_context_length = kwargs.get("launch_context_length")
    inst._effective_context_length = kwargs.get("effective_context_length")
    return inst


class TestParseRuntimeNCtxFromStdout:
    def test_parses_first_slot_line(self):
        inst = _backend(
            stdout_lines = [
                "INFO starting",
                "new slot, n_ctx = 2048",
                "new slot, n_ctx = 4096",
            ]
        )
        assert inst._parse_runtime_n_ctx_from_stdout() == 2048

    def test_returns_none_when_missing(self):
        inst = _backend(stdout_lines = ["INFO starting"])
        assert inst._parse_runtime_n_ctx_from_stdout() is None


class TestRequestedContextLengthProperty:
    def test_exposes_launch_when_fit_reduced(self):
        inst = _backend(launch_context_length = 8192, effective_context_length = 2048)
        assert inst.requested_context_length == 8192

    def test_none_when_launch_matches_runtime(self):
        inst = _backend(launch_context_length = 8192, effective_context_length = 8192)
        assert inst.requested_context_length is None

    def test_none_when_launch_unset(self):
        inst = _backend(effective_context_length = 2048)
        assert inst.requested_context_length is None


class TestProbeRuntimeContextLength:
    def test_prefers_slots_endpoint(self, monkeypatch):
        inst = _backend(
            stdout_lines = ["new slot, n_ctx = 9999"],
        )

        class _Resp:
            status_code = 200

            def json(self):
                return [{"n_ctx": 2048}]

        monkeypatch.setattr(
            "core.inference.llama_cpp.httpx.get",
            lambda url, timeout: _Resp(),
        )
        assert inst._probe_runtime_context_length() == 2048

    def test_falls_back_to_props(self, monkeypatch):
        inst = _backend()

        def fake_get(url, timeout):
            if url.endswith("/slots"):
                raise RuntimeError("slots unavailable")

            class _Resp:
                status_code = 200

                def json(self):
                    return {"default_generation_settings": {"n_ctx": 3072}}

            return _Resp()

        monkeypatch.setattr("core.inference.llama_cpp.httpx.get", fake_get)
        assert inst._probe_runtime_context_length() == 3072

    def test_falls_back_to_stdout(self, monkeypatch):
        inst = _backend(stdout_lines = ["new slot, n_ctx = 1024"])

        def fake_get(url, timeout):
            raise RuntimeError("offline")

        monkeypatch.setattr("core.inference.llama_cpp.httpx.get", fake_get)
        assert inst._probe_runtime_context_length() == 1024

    def test_returns_none_without_port(self):
        inst = _backend(port = None)
        assert inst._probe_runtime_context_length() is None
