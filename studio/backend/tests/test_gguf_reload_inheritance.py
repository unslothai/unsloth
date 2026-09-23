# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend contract for the GGUF reload duplicate-load guard.

``LlamaCppBackend.adopt_load_intent_if_matched`` short-circuits a duplicate /load so
it cannot kill the just-spawned llama-server. Pins local-file identity, the
HF-mode hf_variant fallback, and ``extra_args`` None-vs-[] inherit semantics.
"""

from __future__ import annotations

import inspect
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
from models.inference import (
    InferenceStatusResponse,
    LoadRequest,
    LoadResponse,
    ValidateModelRequest,
)


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
    backend._reasoning_budget = -1
    backend._reasoning_budget_message = ""
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


def test_reasoning_budget_schema_contract():
    request = LoadRequest(model_path = "owner/repo")
    assert request.reasoning_budget == -1
    assert request.reasoning_budget_message == ""
    assert LoadRequest(model_path = "owner/repo", reasoning_budget = 0).reasoning_budget == 0
    with pytest.raises(ValueError):
        LoadRequest(model_path = "owner/repo", reasoning_budget = -2)
    with pytest.raises(ValueError, match = "8192-byte"):
        LoadRequest(model_path = "owner/repo", reasoning_budget_message = "😀" * 2_049)
    with pytest.raises(ValueError, match = "NUL"):
        LoadRequest(model_path = "owner/repo", reasoning_budget_message = "bad\0message")
    padded = LoadRequest(model_path = "owner/repo", reasoning_budget_message = "  PAD  ")
    assert padded.reasoning_budget_message == "  PAD  "

    load = LoadResponse(status = "loaded", model = "m", display_name = "m", inference = {})
    status = InferenceStatusResponse()
    assert (load.reasoning_budget, load.reasoning_budget_message) == (-1, "")
    assert (status.reasoning_budget, status.reasoning_budget_message) == (-1, "")


def test_reasoning_budget_is_part_of_backend_dedupe():
    backend = _loaded_backend(
        _reasoning_budget = 64,
        _reasoning_budget_message = "limit",
        _requested_reasoning_budget = 64,
        _requested_reasoning_budget_message = "limit",
    )
    common = dict(
        model_identifier = "owner/repo",
        hf_variant = "Q4_K_M",
        n_ctx = 8192,
        cache_type_kv = None,
        speculative_type = None,
        chat_template_override = None,
        extra_args = None,
        is_vision = False,
        reasoning_budget = 64,
        reasoning_budget_message = "limit",
    )
    assert _matches(backend, **common) is True
    assert _matches(backend, **{**common, "reasoning_budget": 32}) is False
    flags = ["--reasoning-budget", "64", "--reasoning-budget-message", "limit"]
    backend = _loaded_backend(
        _reasoning_budget = 64,
        _reasoning_budget_message = "limit",
        _requested_reasoning_budget = 64,
        _requested_reasoning_budget_message = "limit",
        _extra_args = flags,
    )
    assert (
        _matches(
            backend,
            **{
                **common,
                "reasoning_budget": -1,
                "reasoning_budget_message": "",
                "extra_args": flags,
            },
        )
        is True
    )


def test_reasoning_budget_state_resets_on_unload():
    backend = _loaded_backend(_reasoning_budget = 64, _reasoning_budget_message = "limit")
    backend.unload_model()
    assert backend.reasoning_budget == -1
    assert backend.reasoning_budget_message == ""


def test_load_wires_reasoning_args_and_respawn_snapshot():
    source = inspect.getsource(LlamaCppBackend.load_model)
    assert "_build_reasoning_budget_flags(" in source
    # The respawn snapshot is the intent itself, so the fields have to come off it.
    assert "reasoning_budget = intent.reasoning_budget" in source
    assert "reasoning_budget_message = intent.reasoning_budget_message" in source
    assert source.index("validate_reasoning_budget_capabilities") < source.index(
        "self._kill_process()"
    )


def test_route_checks_reasoning_budget_capabilities_before_teardown():
    route_source = (Path(__file__).resolve().parent.parent / "routes" / "inference.py").read_text(
        encoding = "utf-8"
    )
    preflight = route_source.index("backend.validate_reasoning_budget_capabilities")
    diffusion_rejection = route_source.index(
        "Reasoning Budget settings are not supported for DiffusionGemma models."
    )
    unknown_rejection = route_source.index(
        "Reasoning Budget settings cannot be applied until this GGUF is"
    )
    teardown = route_source.index("# Point of no return for the GGUF path")
    assert preflight < teardown
    assert diffusion_rejection < teardown
    assert unknown_rejection < teardown


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


@pytest.mark.parametrize("model", [LoadRequest, ValidateModelRequest])
def test_reasoning_budget_rejects_booleans(model):
    # bool subclasses int and pydantic parses lax, so `true` would launch a one-token budget.
    with pytest.raises(ValueError, match = "Expected a number, got a boolean"):
        model(model_path = "unsloth/x", reasoning_budget = True)
    assert model(model_path = "unsloth/x", reasoning_budget = 1).reasoning_budget == 1


def test_the_reuse_check_compares_the_request_not_the_environment():
    """An inherited LLAMA_ARG_THINK_BUDGET* cannot be sent or cleared by any request, so comparing
    the live EFFECTIVE value against a resolved request tore down a healthy server on every load."""
    source = inspect.getsource(LlamaCppBackend._runtime_matches_intent)
    assert "self._requested_reasoning_budget" in source
    assert (
        "resolve_reasoning_budget_with_env" not in source
    ), "the reuse check must not fold the environment into the request"

    launch = inspect.getsource(LlamaCppBackend.load_model)
    assert "self._requested_reasoning_budget = reasoning_budget" in launch
    # A probe that could not be read says nothing about the flag, and the child still applies
    # the environment, so only a conclusive "unsupported" may drop it.
    assert "reasoning_budget_probe_inconclusive" in launch


def test_requested_reasoning_budget_is_reported_separately():
    from models.inference import InferenceStatusResponse

    fields = InferenceStatusResponse.model_fields
    for name in ("requested_reasoning_budget", "requested_reasoning_budget_message"):
        assert name in fields, name
    assert fields["requested_reasoning_budget"].default == -1
    assert fields["requested_reasoning_budget_message"].default == ""


def test_an_inherited_env_budget_does_not_force_a_reload():
    """The live EFFECTIVE value carries LLAMA_ARG_THINK_BUDGET*, which no request can send or
    clear. Comparing against it tore down a healthy server on every load and never converged."""
    backend = _loaded_backend(
        # What the environment gave the child...
        _reasoning_budget = 512,
        _reasoning_budget_message = "from env",
        # ...against a load that asked for nothing.
        _requested_reasoning_budget = -1,
        _requested_reasoning_budget_message = "",
    )
    assert (
        _matches(
            backend,
            model_identifier = "owner/repo",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = None,
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
            reasoning_budget = -1,
            reasoning_budget_message = "",
        )
        is True
    )
