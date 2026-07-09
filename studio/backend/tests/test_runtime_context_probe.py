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

_structlog_stub = _types.ModuleType("structlog")

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

# Stub heavy deps only while importing the module under test, then restore
# sys.modules so the fakes can't shadow the real packages for later tests
# (llama_cpp keeps its own references to whatever it imported here).
_inserted_stubs: list[str] = []
for _name, _stub in (
    ("loggers", _loggers_stub),
    ("structlog", _structlog_stub),
    ("httpx", _httpx_stub),
):
    if _name not in sys.modules:
        sys.modules[_name] = _stub
        _inserted_stubs.append(_name)

try:
    from core.inference.llama_cpp import LlamaCppBackend
finally:
    for _name in _inserted_stubs:
        sys.modules.pop(_name, None)


@pytest.fixture(scope = "module", autouse = True)
def _cleanup_llama_cpp_import_cache():
    """Drop cached llama_cpp so later tests re-import real httpx/loggers."""
    yield
    sys.modules.pop("core.inference.llama_cpp", None)
    import core.inference as _pkg

    if hasattr(_pkg, "llama_cpp"):
        delattr(_pkg, "llama_cpp")


def _backend(**kwargs):
    inst = LlamaCppBackend.__new__(LlamaCppBackend)
    inst._port = kwargs.get("port", 48507)
    inst._stdout_lines = list(kwargs.get("stdout_lines", []))
    inst._launch_context_length = kwargs.get("launch_context_length")
    inst._effective_context_length = kwargs.get("effective_context_length")
    inst._requested_context_length = kwargs.get("requested_context_length")
    inst._launch_use_fit = kwargs.get("launch_use_fit")
    inst._launch_n_parallel = kwargs.get("launch_n_parallel")
    inst._launch_kv_unified = kwargs.get("launch_kv_unified", False)
    inst._requested_n_ctx = kwargs.get("requested_n_ctx", 0)
    inst._api_key = kwargs.get("api_key")
    return inst


class TestExpectedPerSlotContext:
    def test_splits_total_across_parallel_slots(self):
        assert LlamaCppBackend._expected_per_slot_context(8192, 4) == 2048

    def test_single_slot_uses_total(self):
        assert LlamaCppBackend._expected_per_slot_context(8192, 1) == 8192


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


class TestApplyRuntimeContextProbe:
    def test_parallel_split_without_fit_reduction(self):
        inst = _backend()
        inst._apply_runtime_context_probe(
            2048,
            launch_ctx = 8192,
            use_fit = True,
            n_parallel = 4,
        )
        assert inst._effective_context_length == 2048
        assert inst.requested_context_length is None

    def test_fit_reduced_single_slot(self):
        inst = _backend()
        inst._apply_runtime_context_probe(
            2048,
            launch_ctx = 8192,
            use_fit = True,
            n_parallel = 1,
        )
        assert inst._effective_context_length == 2048
        assert inst.requested_context_length == 8192

    def test_fit_reduced_below_parallel_expectation(self):
        inst = _backend()
        inst._apply_runtime_context_probe(
            1024,
            launch_ctx = 8192,
            use_fit = True,
            n_parallel = 4,
        )
        assert inst._effective_context_length == 1024
        assert inst.requested_context_length == 2048

    def test_kv_unified_parallel_uses_full_launch_ctx(self):
        inst = _backend()
        inst._apply_runtime_context_probe(
            4096,
            launch_ctx = 8192,
            use_fit = True,
            n_parallel = 4,
            kv_unified = True,
        )
        assert inst._effective_context_length == 4096
        assert inst.requested_context_length == 8192

    def test_no_warning_without_fit(self):
        inst = _backend()
        inst._apply_runtime_context_probe(
            2048,
            launch_ctx = 8192,
            use_fit = False,
            n_parallel = 4,
        )
        assert inst.requested_context_length is None

    def test_caps_runtime_above_launch_expectation(self):
        """Server may round up n_ctx; Studio must not inflate above the launch cap."""
        inst = _backend()
        inst._apply_runtime_context_probe(
            4096,
            launch_ctx = 8192,
            use_fit = False,
            n_parallel = 4,
        )
        assert inst._effective_context_length == 2048
        assert inst.requested_context_length is None

    def test_caps_kv_unified_above_launch_ctx(self):
        """With --kv-unified, cap to the full launch -c, not per-slot."""
        inst = _backend()
        inst._apply_runtime_context_probe(
            16384,
            launch_ctx = 8192,
            use_fit = False,
            n_parallel = 4,
            kv_unified = True,
        )
        assert inst._effective_context_length == 8192
        assert inst.requested_context_length is None

    def test_failed_probe_clears_stale_requested_context(self):
        # A reload can replace the server without unload_model; a probe
        # that then fails must not report the previous load's reduction.
        inst = _backend(requested_context_length = 8192)
        inst._apply_runtime_context_probe(
            None,
            launch_ctx = 4096,
            use_fit = True,
            n_parallel = 1,
        )
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
            lambda url, timeout, **kwargs: _Resp(),
        )
        assert inst._probe_runtime_context_length() == 2048

    def test_falls_back_to_props(self, monkeypatch):
        inst = _backend()

        def fake_get(url, timeout, **kwargs):
            if url.endswith("/slots"):
                raise RuntimeError("slots unavailable")

            class _Resp:
                status_code = 200

                def json(self):
                    return {"default_generation_settings": {"n_ctx": 3072}}

            return _Resp()

        monkeypatch.setattr("core.inference.llama_cpp.httpx.get", fake_get)
        assert inst._probe_runtime_context_length() == 3072

    def test_ignores_malformed_slots_payload(self, monkeypatch):
        inst = _backend()

        class _Resp:
            status_code = 200

            def json(self):
                return ["not-a-dict"]

        monkeypatch.setattr(
            "core.inference.llama_cpp.httpx.get",
            lambda url, timeout, **kwargs: _Resp(),
        )
        assert inst._probe_runtime_context_length() is None

    def test_falls_back_to_stdout(self, monkeypatch):
        inst = _backend(stdout_lines = ["new slot, n_ctx = 1024"])

        def fake_get(url, timeout, **kwargs):
            raise RuntimeError("offline")

        monkeypatch.setattr("core.inference.llama_cpp.httpx.get", fake_get)
        assert inst._probe_runtime_context_length() == 1024

    def test_returns_none_without_port(self):
        inst = _backend(port = None)
        assert inst._probe_runtime_context_length() is None

    def test_probe_authenticates_and_bypasses_proxy(self, monkeypatch):
        # Direct-stream launches start llama-server with --api-key, and /slots
        # and /props are not public llama.cpp endpoints, so the probe must send
        # the Bearer header or it 401s. It must also set trust_env=False so an
        # ambient HTTP(S)_PROXY can't hijack the 127.0.0.1 probe (like the health
        # check and _query_server_n_ctx do).
        inst = _backend(api_key = "secret-key")
        captured = []

        class _Resp:
            status_code = 200

            def json(self):
                return [{"n_ctx": 2048}]

        def fake_get(url, timeout, **kwargs):
            captured.append(kwargs)
            return _Resp()

        monkeypatch.setattr("core.inference.llama_cpp.httpx.get", fake_get)
        assert inst._probe_runtime_context_length() == 2048
        assert captured, "probe did not call httpx.get"
        assert captured[0].get("trust_env") is False
        assert captured[0].get("headers") == {"Authorization": "Bearer secret-key"}

    def test_probe_sends_no_auth_header_when_unauthenticated(self, monkeypatch):
        # Proxied (non direct-stream) launches have no api key; headers stays None
        # so httpx sends the request with no Authorization header, and trust_env
        # is still disabled for the loopback probe.
        inst = _backend()
        captured = []

        def fake_get(url, timeout, **kwargs):
            captured.append(kwargs)
            raise RuntimeError("slots unavailable")

        monkeypatch.setattr("core.inference.llama_cpp.httpx.get", fake_get)
        assert inst._probe_runtime_context_length() is None
        assert captured[0].get("headers") is None
        assert captured[0].get("trust_env") is False


class TestLaunchContextLength:
    def test_exposes_total_c_from_last_load(self):
        inst = _backend(requested_n_ctx = 8192)
        assert inst.launch_context_length == 8192

    def test_auto_load_returns_none_without_recorded_launch(self):
        inst = _backend(requested_n_ctx = 0)
        assert inst.launch_context_length is None

    def test_auto_load_exposes_effective_total_c(self):
        inst = _backend(requested_n_ctx = 0, launch_context_length = 8192)
        assert inst.launch_context_length == 8192

    def test_auto_launch_does_not_flag_fit_reduction(self):
        """When launch_ctx is None (Auto -c), do not compare against pre-fit ctx."""
        inst = _backend()
        inst._apply_runtime_context_probe(
            2048,
            launch_ctx = None,
            use_fit = True,
            n_parallel = 4,
        )
        assert inst._effective_context_length == 2048
        assert inst.requested_context_length is None


class TestReloadMaxSeqLengthContract:
    """Document reload semantics: per-slot context_length must not shrink total -c."""

    def test_parallel_launch_keeps_total_c_for_reload(self):
        inst = _backend(requested_n_ctx = 8192)
        inst._apply_runtime_context_probe(
            2048,
            launch_ctx = 8192,
            use_fit = False,
            n_parallel = 4,
        )
        assert inst.context_length == 2048
        assert inst.launch_context_length == 8192
        assert inst.requested_context_length is None


class TestLaunchedArgvReconciliation:
    """The recorded launch context/fit must reflect the command that actually
    started, so a last-wins pass-through override or a text-only retry does not
    report a stale value that round-trips on the next reload."""

    def test_passthrough_ctx_size_zero_reports_auto_not_stale_explicit(self):
        from core.inference.llama_server_args import parse_ctx_override

        # Studio launched -c 8192 but the user appended a last-wins --ctx-size 0
        # (Auto). The load records the final launched context, so the property
        # reports Auto (None) instead of the stale 8192 request that would
        # silently convert the Auto load into an explicit context on reload.
        launched = ["llama-server", "-c", "8192", "--ctx-size", "0"]
        final_ctx_override = parse_ctx_override(launched)
        assert final_ctx_override == 0
        recorded = final_ctx_override if final_ctx_override is not None else 8192
        assert _backend(requested_n_ctx = recorded).launch_context_length is None

    def test_passthrough_ctx_size_positive_is_recorded(self):
        from core.inference.llama_server_args import parse_ctx_override

        # A positive last-wins pass-through is the real launch context.
        launched = ["llama-server", "-c", "8192", "--ctx-size", "4096"]
        final_ctx_override = parse_ctx_override(launched)
        assert final_ctx_override == 4096
        recorded = final_ctx_override if final_ctx_override is not None else 8192
        assert _backend(requested_n_ctx = recorded).launch_context_length == 4096

    def test_text_only_retry_command_carries_fit_and_context(self):
        from core.inference.llama_cpp import LlamaCppBackend
        from core.inference.llama_server_args import (
            parse_ctx_override,
            parse_fit_override,
        )

        # The text-only retry starts the stripped command, and post-load parsing
        # must read that command: --mmproj is gone but --fit off and -c survive,
        # so the launched fit/context come from the server that is actually live.
        failed_mmproj = [
            "llama-server",
            "-c",
            "8192",
            "--mmproj",
            "m.gguf",
            "--fit",
            "off",
        ]
        text_only = LlamaCppBackend._strip_mmproj_args(failed_mmproj)
        assert "--mmproj" not in text_only
        assert parse_fit_override(text_only) is False
        assert parse_ctx_override(text_only) == 8192
