# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend contract for the GGUF reload duplicate-load guard.

``LlamaCppBackend.adopt_load_intent_if_matched`` short-circuits a duplicate /load so
it cannot kill the just-spawned llama-server. Pins local-file identity, the
HF-mode hf_variant fallback, and ``extra_args`` None-vs-[] inherit semantics.
"""

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
# Only when the real library is absent. sys.modules holds what has been IMPORTED, not
# what is installed, so setdefault does not defer to a real httpx that nothing in this
# process has touched yet: the stub wins and shadows it for the whole session. This stub
# has no Response, and starlette.testclient reads httpx.Response at import, so every
# module collected afterwards that reaches fastapi.testclient or routes.inference dies.
try:
    import httpx  # noqa: F401
except ImportError:
    sys.modules.setdefault("httpx", _httpx_stub)

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend


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
    backend._requested_spec_mode = "auto"
    backend._chat_template_override = None
    backend._is_vision = False
    backend._extra_args = None
    backend._extra_args_source = None
    backend._gguf_path = None
    for key, value in overrides.items():
        setattr(backend, key, value)
    return backend


def _matches(backend: LlamaCppBackend, **kwargs) -> bool:
    return backend.adopt_load_intent_if_matched(GgufLoadIntent(**kwargs))


# ── Local-file identity via gguf_path ────────────────────────────────


def test_already_in_target_state_uses_gguf_path_when_present(tmp_path):
    gguf_file = tmp_path / "model.Q4_K_M.gguf"
    gguf_file.write_bytes(b"")
    backend = _loaded_backend(
        _hf_variant = "Q4_K_M",
        _gguf_path = str(gguf_file),
    )
    assert (
        _matches(
            backend,
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


def test_already_loaded_model_reloads_when_selected_binary_changes():
    backend = _loaded_backend()
    backend._binary_changed_since_launch = lambda: True

    assert (
        _matches(
            backend,
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
        is False
    )


def test_already_in_target_state_rejects_different_gguf_path(tmp_path):
    a = tmp_path / "a.gguf"
    a.write_bytes(b"")
    b = tmp_path / "b.gguf"
    b.write_bytes(b"")
    backend = _loaded_backend(_gguf_path = str(a))
    assert (
        _matches(
            backend,
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
        _matches(
            backend,
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
        _matches(
            backend,
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
        _matches(
            backend,
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
        _matches(
            backend,
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
        _matches(
            backend,
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


class TestRepeatLoadMatchesTheEffectiveCache:
    """A repeat /load of an identical request must reuse the healthy server.

    self._cache_type_kv records only what Unsloth emitted as a MANAGED flag, so a
    cache set through extras or the environment leaves it None on one side and a
    type on the other; the old scalar-against-scalar comparison then read an
    identical repeat as a mismatch and tore the server down to relaunch the same
    thing. Before ggml-org/llama.cpp#23792 the tensor gate hid this by rewriting
    the cache away; a layer load has always had it.
    """

    @staticmethod
    def _backend_running(effective):
        """A backend carrying only the field the comparison reads: the per-axis
        pair the live child was launched with."""
        from core.inference.llama_cpp import LlamaCppBackend

        b = LlamaCppBackend.__new__(LlamaCppBackend)
        b._effective_cache_types = effective
        return b

    @pytest.mark.parametrize(
        "extras,managed",
        [
            (["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"], None),  # extras only
            (["--cache-type-k", "q4_0", "--cache-type-v", "f16"], None),  # asymmetric
            ([], "q8_0"),  # managed only
            ([], None),  # nothing set
        ],
    )
    def test_the_same_request_resolves_to_the_running_pair(self, extras, managed):
        from core.inference.llama_cpp import _planned_main_cache_types

        planned = _planned_main_cache_types(managed, extras)
        running = self._backend_running(planned)

        # The comparison the matcher makes, isolated: same request in, same pair out.
        assert running._effective_cache_types == _planned_main_cache_types(managed, extras)

    def test_a_changed_cache_still_reloads(self):
        from core.inference.llama_cpp import _planned_main_cache_types
        running = self._backend_running(("q8_0", "q8_0"))

        assert running._effective_cache_types != _planned_main_cache_types(
            None, ["--cache-type-k", "f16", "--cache-type-v", "f16"]
        )

    def test_the_matcher_compares_the_pair_not_the_managed_scalar(self):
        """Source-pinned: the scalar cannot describe an extras-only or env cache,
        so reintroducing it here would bring the spurious reload back."""
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        src = "".join(inspect.getsource(LlamaCppBackend._runtime_matches_intent).split())
        assert "self._requested_cache_types!=_planned_main_cache_types(" in src
        assert "_norm(self._cache_type_kv)!=_norm(intent.cache_type_kv)" not in src

    def test_a_launch_time_rewrite_does_not_force_a_reload(self):
        """The comparison is requested-against-requested, so a rewrite the launch
        performed does not make the next identical request look different.

        A build with no --flash-attn resets a quantized V cache to f16 before the
        spawn (and the flash-attn crash recovery does the same), so the pair that
        LAUNCHED is not the pair that was ASKED for. Comparing the running pair
        would then reject every repeat and redo that normalization each time.
        """
        from core.inference.llama_cpp import (
            LlamaCppBackend,
            _effective_main_cache_types,
            _planned_main_cache_types,
        )

        extras = ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
        asked = _planned_main_cache_types(None, extras)
        cmd = ["llama-server", "-m", "/x.gguf", *extras]
        b = LlamaCppBackend.__new__(LlamaCppBackend)
        b._architecture = None
        launched = _effective_main_cache_types(
            LlamaCppBackend._reset_quantized_v_cache(
                cmd, "this build has no --flash-attn", mla = False, draft_mla = None
            ),
            {},
        )

        assert asked == ("q8_0", "q8_0")
        assert launched == ("q8_0", "f16"), launched
        # The matcher reads the first, not the second.
        b._requested_cache_types = asked
        assert b._requested_cache_types == _planned_main_cache_types(None, extras)

    def test_the_requested_pair_is_recorded_next_to_the_effective_one(self):
        """Both are recorded on the same success path, so one cannot drift."""
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        load = "".join(inspect.getsource(LlamaCppBackend.load_model).split())
        assert "self._effective_cache_types=_effective_main_cache_types(" in load
        assert "self._requested_cache_types=_planned_cache_pair" in load
