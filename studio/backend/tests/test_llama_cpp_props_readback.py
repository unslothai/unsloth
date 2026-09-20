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
    inst._stdout_lines = []
    inst._has_video_input = False
    return inst


def _stub_props(
    monkeypatch,
    status_code = 200,
    body = None,
    exc = None,
):
    """Stub ``/props``; ``/slots`` answers 404 so the chain falls through to it.

    Mirrors a ``--no-slots`` child, which is the case these /props tests describe.
    """

    def fake_get(
        url,
        headers = None,
        timeout = None,
        trust_env = None,
    ):
        assert trust_env is False
        # These endpoints sit behind llama-server's api-key middleware, so a
        # direct-stream child must be addressed with the bearer token; without one
        # the header stays absent rather than becoming a bogus "Bearer None".
        assert headers is None or headers == {"Authorization": "Bearer test-key"}
        if url.endswith("/slots"):
            return _FakeResponse(404, {})
        assert url.endswith("/props")
        if exc is not None:
            raise exc
        return _FakeResponse(status_code, body)

    monkeypatch.setattr(llama_cpp_mod.httpx, "get", fake_get, raising = False)


def _stub_endpoints(
    monkeypatch,
    slots = None,
    props = None,
    slots_exc = None,
    props_exc = None,
):
    """Stub both probe endpoints independently.

    ``slots``/``props`` take a ``_FakeResponse``; ``None`` means the endpoint is
    absent (404), which is how a ``--no-slots`` build answers.
    """
    seen = []

    def fake_get(
        url,
        headers = None,
        timeout = None,
        trust_env = None,
    ):
        assert trust_env is False
        seen.append(url)
        if url.endswith("/slots"):
            if slots_exc is not None:
                raise slots_exc
            return slots if slots is not None else _FakeResponse(404, {})
        assert url.endswith("/props")
        if props_exc is not None:
            raise props_exc
        return props if props is not None else _FakeResponse(404, {})

    monkeypatch.setattr(llama_cpp_mod.httpx, "get", fake_get, raising = False)
    return seen


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
# Launch total vs per-slot reporting
#
# After reconciliation ``context_length`` is the PER-SLOT window, which is the
# only number a request may be sized against. The total ``-c`` the child was
# launched with is a different quantity, and the UI needs both: the total is
# what the context control may offer, and the gap between the per-slot
# EXPECTATION and what llama-server really allocated is the --fit reduction.
# ---------------------------------------------------------------------------


def test_launch_context_length_is_read_off_the_spawned_argv(monkeypatch):
    """The launch total comes from the argv that really spawned, not the intent."""
    inst = _make_backend(effective_ctx = 32768)
    inst._effective_parallel_slots = 4
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 8192}},
    )

    inst._reconcile_effective_ctx_with_server(
        launch_cmd = ["llama-server", "-c", "32768", "--parallel", "4"],
    )

    assert inst.launch_context_length == 32768
    assert inst.context_length == 8192


def test_a_pass_through_ctx_size_is_the_launch_total(monkeypatch):
    """--ctx-size last-wins over Studio's own -c, so the argv is the only honest source."""
    inst = _make_backend(effective_ctx = 32768)
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 65536}},
    )

    inst._reconcile_effective_ctx_with_server(
        requested_n_ctx = 65536,
        launch_cmd = ["llama-server", "-c", "32768", "--ctx-size", "65536"],
    )

    assert inst.launch_context_length == 65536


def test_a_clean_parallel_split_is_not_reported_as_a_fit_reduction(monkeypatch):
    """8192 of a 32768 total across 4 slots is the division, not a shortfall."""
    inst = _make_backend(effective_ctx = 32768)
    inst._effective_parallel_slots = 4
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 8192}},
    )

    inst._reconcile_effective_ctx_with_server(
        launch_cmd = ["llama-server", "-c", "32768", "--parallel", "4"],
    )

    assert inst.pre_fit_context_length is None


def test_fit_reduction_reports_the_expected_per_slot_context(monkeypatch):
    """The Nick repro again: -c 98304 on one slot, server really at 67584."""
    inst = _make_backend(effective_ctx = 98304)
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 67584}},
    )

    inst._reconcile_effective_ctx_with_server(
        launch_cmd = ["llama-server", "-c", "98304"],
    )

    assert inst.pre_fit_context_length == 98304
    assert inst.context_length == 67584


def test_fit_reduction_under_a_split_is_measured_against_the_slot_share(monkeypatch):
    """A slot should have had 8192 of the 32768 total; it got 4096."""
    inst = _make_backend(effective_ctx = 32768)
    inst._effective_parallel_slots = 4
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 4096}},
    )

    inst._reconcile_effective_ctx_with_server(
        launch_cmd = ["llama-server", "-c", "32768", "--parallel", "4"],
    )

    assert inst.pre_fit_context_length == 8192


def test_unified_kv_does_not_divide_the_expectation_by_slots(monkeypatch):
    """--kv-unified shares one cache, so every slot expects the whole total.

    Divide here and the expectation (8192) would sit BELOW the real window,
    hiding a genuine halving behind arithmetic that does not apply.
    """
    inst = _make_backend(effective_ctx = 32768)
    inst._effective_parallel_slots = 4
    inst._kv_cache_unified = True
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 16384}},
    )

    inst._reconcile_effective_ctx_with_server(
        launch_cmd = ["llama-server", "-c", "32768", "--parallel", "4", "--kv-unified"],
    )

    assert inst.pre_fit_context_length == 32768


def test_cell_padding_alone_is_not_a_fit_reduction(monkeypatch):
    """-c 100000 pads to 100096 cells; the 96-token gap is rounding, not the fitter."""
    inst = _make_backend(effective_ctx = 100000)
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 100000}},
    )

    inst._reconcile_effective_ctx_with_server(
        launch_cmd = ["llama-server", "-c", "100000"],
    )

    assert inst.pre_fit_context_length is None


def test_auto_context_names_no_launch_total(monkeypatch):
    """``-c 0`` asks llama.cpp to choose, so there is no requested total to report."""
    inst = _make_backend(effective_ctx = 8192)
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 4096}},
    )

    inst._reconcile_effective_ctx_with_server(launch_cmd = ["llama-server", "-c", "0"])

    assert inst.launch_context_length is None
    assert inst.pre_fit_context_length is None


def test_a_trailing_ctx_size_zero_reports_no_launch_total(monkeypatch):
    """A pass-through ``--ctx-size 0`` after Studio's own -c last-wins back to Auto.

    Reporting the earlier nonzero -c here is what turns an auto-context load into a
    fixed one: the frontend would prefer it as the reload max_seq_length, and the
    next Apply would silently pin a context the user never asked for.
    """
    inst = _make_backend(effective_ctx = 32768)
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 4096}},
    )

    inst._reconcile_effective_ctx_with_server(
        launch_cmd = ["llama-server", "-c", "32768", "--ctx-size", "0"],
    )

    assert inst.launch_context_length is None
    assert inst.pre_fit_context_length is None


def test_a_trailing_explicit_ctx_size_is_still_reported(monkeypatch):
    """The mirror image: last-wins the other way names a real total, so report it.

    Without this the rule above could be satisfied by reporting nothing at all.
    """
    inst = _make_backend(effective_ctx = 32768)
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 32768}},
    )

    inst._reconcile_effective_ctx_with_server(
        launch_cmd = ["llama-server", "-c", "0", "--ctx-size", "32768"],
    )

    assert inst.launch_context_length == 32768


def test_a_malformed_ctx_flag_reports_nothing_rather_than_raising(monkeypatch):
    """A bad flag must not take the whole post-launch reconciliation down with it."""
    inst = _make_backend(effective_ctx = 98304)
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 67584}},
    )

    inst._reconcile_effective_ctx_with_server(launch_cmd = ["llama-server", "-c", "wat"])

    assert inst.launch_context_length is None
    assert inst.pre_fit_context_length is None
    # The reconciliation itself still did its job.
    assert inst._effective_context_length == 67584


def test_a_reload_that_needed_no_fit_clears_the_previous_reduction(monkeypatch):
    """Both fields are rewritten every time, or a swap inherits a stale warning."""
    inst = _make_backend(effective_ctx = 32768)
    inst._launch_context_length = 98304
    inst._pre_fit_context_length = 98304
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 32768}},
    )

    inst._reconcile_effective_ctx_with_server(launch_cmd = ["llama-server", "-c", "32768"])

    assert inst.launch_context_length == 32768
    assert inst.pre_fit_context_length is None


def test_recording_the_launch_total_never_rewrites_the_requested_ctx(monkeypatch):
    """The launched context is reported beside the request, never as the request.

    ``_requested_n_ctx`` is what the duplicate-load comparators read, and they
    compare raw request against raw request. Publishing the launched total there
    instead is what made an identical repeat /load relaunch the server.
    """
    inst = _make_backend(effective_ctx = 65983)
    # Auto: the caller sent no context field and let a pass-through flag decide.
    inst._requested_n_ctx = 0
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 100352}},
    )

    inst._reconcile_effective_ctx_with_server(
        requested_n_ctx = 100352,
        launch_cmd = ["llama-server", "-c", "65983", "--ctx-size", "100352"],
    )

    assert inst.launch_context_length == 100352
    assert inst._effective_context_length == 100352
    assert inst._requested_n_ctx == 0, "Auto must stay Auto, or every repeat load reloads"


def test_an_identical_repeat_load_still_dedupes_after_a_pass_through_ctx_size(monkeypatch):
    """The consequence of the invariant above, at the comparator that suffers it."""
    monkeypatch.setattr(
        llama_cpp_mod.LlamaCppBackend,
        "_kill_orphaned_servers",
        staticmethod(lambda: 0),
    )
    inst = llama_cpp_mod.LlamaCppBackend()
    inst._port = 51234
    inst._effective_context_length = 65983
    intent = llama_cpp_mod.GgufLoadIntent(model_identifier = "org/A-GGUF", n_ctx = 0)
    assert inst._runtime_matches_intent(intent, None), "precondition: a fresh Auto load matches"
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 100352}},
    )

    inst._reconcile_effective_ctx_with_server(
        requested_n_ctx = 100352,
        launch_cmd = ["llama-server", "-c", "65983", "--ctx-size", "100352"],
    )

    assert inst.launch_context_length == 100352
    assert inst._runtime_matches_intent(intent, None)


def test_a_failed_probe_still_reports_the_launch_total(monkeypatch):
    """What was launched is known from the argv; only the fit needs the server.

    Skipping the recording when the probe fails would leave the PREVIOUS load's
    total on the backend, which is worse than reporting nothing.
    """
    inst = _make_backend(effective_ctx = 98304)
    inst._launch_context_length = 4096
    inst._pre_fit_context_length = 4096
    _stub_props(monkeypatch, exc = RuntimeError("boom"))

    inst._reconcile_effective_ctx_with_server(launch_cmd = ["llama-server", "-c", "98304"])

    assert inst.launch_context_length == 98304
    assert inst.pre_fit_context_length is None
    # The pre-existing guarantee: a flaky probe never wipes the computed context.
    assert inst._effective_context_length == 98304


def test_no_ctx_flag_on_the_argv_reports_no_launch_total(monkeypatch):
    inst = _make_backend(effective_ctx = 8192)
    _stub_props(
        monkeypatch,
        body = {"default_generation_settings": {"n_ctx": 8192}},
    )

    inst._reconcile_effective_ctx_with_server(launch_cmd = ["llama-server", "-m", "model.gguf"])

    assert inst.launch_context_length is None
    assert inst.pre_fit_context_length is None


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


# ---------------------------------------------------------------------------
# /props input validation
#
# A real defect found reviewing #5911: the parse sits outside the only try that
# guards the request, so another process' malformed JSON reached the load path.
# ---------------------------------------------------------------------------


def test_a_malformed_props_payload_reports_unknown_rather_than_raising(monkeypatch):
    """``default_generation_settings`` is whatever llama-server sent. A non-dict
    raises AttributeError off ``.get`` and a non-numeric n_ctx raises ValueError
    off ``int()`` -- both outside the try that guards the request itself, so both
    propagated out of the readback into the post-health load path and failed the
    load. An unreadable /props means "unknown", not "abort"."""
    for bad in (
        {"default_generation_settings": [{"n_ctx": 8192}]},
        {"default_generation_settings": {"n_ctx": "not-a-number"}},
        {"default_generation_settings": {"n_ctx": [4096]}},
        {"default_generation_settings": {"n_ctx": True}},
    ):
        _stub_props(monkeypatch, body = bad)
        assert _make_backend()._query_server_n_ctx() is None, bad


def test_a_stringified_n_ctx_is_still_read(monkeypatch):
    """The readback this replaced did ``int(n_ctx)`` and so coerced "8192".
    Rejecting it would make a build or proxy that stringifies the field stop
    reconciling silently and go back to advertising the pre-launch estimate --
    the bug this reports. Digits only; "8k" is still a refusal, not a guess."""
    _stub_props(monkeypatch, body = {"default_generation_settings": {"n_ctx": "8192"}})
    assert _make_backend()._query_server_n_ctx() == 8192
    for bad in ("8k", "8192.5", "", "-1"):
        _stub_props(monkeypatch, body = {"default_generation_settings": {"n_ctx": bad}})
        assert _make_backend()._query_server_n_ctx() is None, bad


def test_an_inherited_ctx_size_is_the_launch_total_when_argv_names_none(monkeypatch):
    """Manual + Auto omits -c on purpose and sets LLAMA_ARG_CTX_SIZE instead, which
    llama.cpp reads before argv (test_a_positive_inherited_context_is_kept).
    Reading argv alone reported no launch total, so no reduction was reported and a
    same-model reload fell back to one slot's share and shrank the server again.

    Verified on llama-server b11057: no -c, LLAMA_ARG_CTX_SIZE=8192,
    --parallel 4 --no-kv-unified -> n_ctx_slot 2048, so 8192 really is the total."""
    _stub_endpoints(
        monkeypatch,
        props = _FakeResponse(200, {"default_generation_settings": {"n_ctx": 2048}}),
    )
    inst = _make_backend(effective_ctx = 2048)
    inst._effective_parallel_slots = 4
    inst._reconcile_effective_ctx_with_server(
        0,
        launch_cmd = ["llama-server", "-m", "x.gguf", "--parallel", "4"],
        launch_env = {"LLAMA_ARG_CTX_SIZE": "8192"},
    )
    assert inst.launch_context_length == 8192
    # 8192 over 4 slots is 2048 exactly, so the split explains it and nothing was fitted.
    assert inst.pre_fit_context_length is None


def test_an_explicit_ctx_flag_outranks_the_inherited_one(monkeypatch):
    """llama.cpp parses argv after the environment, so the flag wins."""
    _stub_endpoints(
        monkeypatch,
        props = _FakeResponse(200, {"default_generation_settings": {"n_ctx": 4096}}),
    )
    inst = _make_backend(effective_ctx = 4096)
    inst._reconcile_effective_ctx_with_server(
        0,
        launch_cmd = ["llama-server", "-c", "4096"],
        launch_env = {"LLAMA_ARG_CTX_SIZE": "8192"},
    )
    assert inst.launch_context_length == 4096


def test_an_inherited_zero_or_junk_names_no_launch_total(monkeypatch):
    """LLAMA_ARG_CTX_SIZE=0 is llama.cpp's "pick one for me", the same as -c 0, and
    must not be reported as a total any more than a malformed value is."""
    for bad in ("0", "", "auto", "-1", "8k"):
        _stub_endpoints(
            monkeypatch,
            props = _FakeResponse(200, {"default_generation_settings": {"n_ctx": 2048}}),
        )
        inst = _make_backend(effective_ctx = 2048)
        inst._reconcile_effective_ctx_with_server(
            0, launch_cmd = ["llama-server"], launch_env = {"LLAMA_ARG_CTX_SIZE": bad}
        )
        assert inst.launch_context_length is None, bad
