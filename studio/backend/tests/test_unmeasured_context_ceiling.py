# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A context nobody could measure is not a ceiling, and must not overrule a request.

Two arms fall back to a short context when the GGUF carries no attention dimensions:
``_plan_tensor_parallel`` (``_TP_UNMEASURED_CTX``) and the Metal unified-memory arm
(``_metal_floor_ctx``). Both used to apply that fallback to an explicit request too, and to
publish it as ``max_context_length``, so 256k selected came back as ``"context_length": 4096,
"max_context_length": 4096, "native_context_length": 262144`` with the slider reporting that
nothing longer fits (#9653).

The fallback is a guess, which cuts both ways, as ``_metal_context_overcommit_message``
already says: it will not refuse against that number, and equally it may not overrule a typed
context or be published as "the largest that fits" once the load has run above it. Auto keeps
the conservative fallback. The reason is still owed, so both arms record
``_unmeasured_context_notice`` on ``last_load_warning``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = str(_TESTS_DIR.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))


def _load(module_name: str, file_name: str):
    """Import a sibling test module by path; tests/ is not a package."""
    spec = importlib.util.spec_from_file_location(module_name, _TESTS_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The Metal harness owns the stubbed Popen, the tiny GGUF and the no-GPU probe that IS the
# Metal condition, plus a `can_estimate_kv` switch this file needs on both settings.
_metal = _load("_metal_guard_for_unmeasured_ctx", "test_metal_explicit_context_guard.py")
_metal_launch = _metal._launch
_ctx_values = _metal._ctx_values

from core.inference.llama_cpp import (  # noqa: E402
    _FIT_MIN_CTX,
    _TP_UNMEASURED_CTX,
    LlamaCppBackend,
)

GB = 1024**3
NATIVE = 262144
TWO_80 = [(0, 80 * 1024), (1, 80 * 1024)]


def _measurable() -> LlamaCppBackend:
    """A backend whose KV cache can be priced (the control for every case below)."""
    backend = LlamaCppBackend()
    backend._n_layers = 28
    backend._embedding_length = 1024
    backend._n_heads = 16
    backend._n_kv_heads = 8
    backend._kv_key_length = 128
    backend._kv_value_length = 128
    backend._context_length = NATIVE
    return backend


def _unmeasurable() -> LlamaCppBackend:
    """The reporter's condition: a header with no attention dimensions."""
    backend = LlamaCppBackend()
    backend._context_length = NATIVE
    assert not backend._can_estimate_kv()
    return backend


# ── the tensor-parallel arm ───────────────────────────────────────────────────


class TestTheTensorParallelFallback:
    def test_an_explicit_request_is_honoured(self):
        """The headline. 262,144 asked for, 262,144 planned, not 4,096."""
        effective, ceiling, *_ = _unmeasurable()._plan_tensor_parallel(
            TWO_80, 20 * GB, NATIVE, max_target_ctx = NATIVE, explicit_ctx = True
        )
        assert effective == NATIVE
        # And the number the UI reads as "the largest that fits" does not contradict the
        # load that just ran: that pair is what the issue quotes from /v1/models.
        assert ceiling >= effective

    def test_auto_still_takes_the_conservative_fallback(self):
        """Auto named no context, so nothing is being overruled and the guess stands."""
        effective, ceiling, *_ = _unmeasurable()._plan_tensor_parallel(
            TWO_80, 20 * GB, NATIVE, max_target_ctx = NATIVE
        )
        assert effective == _TP_UNMEASURED_CTX
        assert ceiling == _TP_UNMEASURED_CTX

    def test_a_measured_cap_still_binds_an_explicit_request(self):
        """The flag must not become a way around the real cap: with a KV estimate the budget
        is measured, ``--fit`` is a no-op in tensor mode, and the cap is all that stands
        between the request and a startup OOM."""
        tight = [(0, 6 * 1024), (1, 6 * 1024)]
        capped, _ceiling, *_ = _measurable()._plan_tensor_parallel(
            tight, 8 * GB, NATIVE, max_target_ctx = NATIVE, explicit_ctx = True
        )
        assert 0 < capped < NATIVE

    def test_a_measurable_plan_that_fits_is_unchanged_by_the_flag(self):
        with_flag = _measurable()._plan_tensor_parallel(
            TWO_80, 20 * GB, NATIVE, max_target_ctx = NATIVE, explicit_ctx = True
        )
        without = _measurable()._plan_tensor_parallel(
            TWO_80, 20 * GB, NATIVE, max_target_ctx = NATIVE
        )
        assert with_flag == without

    def test_the_flag_defaults_off_for_every_existing_caller(self):
        """Auto is the default, so a caller that does not pass the argument keeps the
        behaviour it had (the planner has direct callers, including the tests)."""
        assert _unmeasurable()._plan_tensor_parallel(TWO_80, 20 * GB, NATIVE)[0] == (
            _TP_UNMEASURED_CTX
        )

    @pytest.mark.parametrize("target", [0, -1])
    def test_no_request_is_not_a_request(self, target):
        effective, _ceiling, *_ = _unmeasurable()._plan_tensor_parallel(
            TWO_80, 20 * GB, target, explicit_ctx = True
        )
        assert effective == _TP_UNMEASURED_CTX


# ── the notice ────────────────────────────────────────────────────────────────


class TestTheNotice:
    def test_it_names_the_context_that_was_launched(self):
        message = LlamaCppBackend._unmeasured_context_notice(NATIVE)
        assert message is not None
        assert "262,144" in message

    def test_it_says_the_maximum_is_not_a_measurement(self):
        message = LlamaCppBackend._unmeasured_context_notice(NATIVE)
        assert "not a measured ceiling" in message

    def test_it_does_not_read_as_a_refusal(self):
        """It describes a load that went out as asked; wording that implies otherwise is
        what sends a working load to the issue tracker."""
        message = LlamaCppBackend._unmeasured_context_notice(NATIVE)
        assert "launched as requested" in message
        for blame in ("cannot", "refus", "not supported"):
            assert blame not in message.lower()

    @pytest.mark.parametrize("cache_type", [None, "", "f16", "fp16"])
    def test_the_kv_hint_is_offered_on_an_unquantized_cache(self, cache_type):
        assert "q8_0" in LlamaCppBackend._unmeasured_context_notice(NATIVE, cache_type)

    @pytest.mark.parametrize("cache_type", ["q8_0", "q4_0"])
    def test_it_is_not_offered_once_the_cache_is_quantized(self, cache_type):
        assert "q8_0" not in LlamaCppBackend._unmeasured_context_notice(NATIVE, cache_type)

    @pytest.mark.parametrize("requested", [0, -1])
    def test_nothing_to_report_without_a_request(self, requested):
        assert LlamaCppBackend._unmeasured_context_notice(requested) is None


# ── the Metal arm, through the real load_model ────────────────────────────────


class TestTheMetalArm:
    """Simulation notice: Metal here is a non-zero ``_apple_metal_memory_budget_bytes`` with
    an empty GPU probe, the seam the Metal suites already use. No Metal device is exercised."""

    def _launch_unmeasurable(self, tmp_path, monkeypatch, **kwargs):
        return _metal_launch(
            tmp_path,
            monkeypatch,
            metal = True,
            can_estimate_kv = False,
            real_fit = True,
            budget_bytes = 24 * GB,
            native = NATIVE,
            **kwargs,
        )

    def test_an_explicit_request_reaches_llama_server(self, tmp_path, monkeypatch):
        """Unchanged, and the control for the next case: this arm never refused here, so
        the launch itself was already right."""
        captured = self._launch_unmeasurable(tmp_path, monkeypatch, n_ctx = NATIVE)
        assert _ctx_values(captured["cmd"])[-1] == str(NATIVE)

    def test_the_published_ceiling_is_not_below_what_launched(self, tmp_path, monkeypatch):
        """The issue's /v1/models answer. The floor used to be published while the child
        ran at the request, so the UI reported a maximum the load had already exceeded."""
        captured = self._launch_unmeasurable(tmp_path, monkeypatch, n_ctx = NATIVE)
        backend = captured["backend"]
        assert backend.max_context_length >= NATIVE
        assert backend.native_context_length == NATIVE

    def test_the_load_says_the_ceiling_was_never_measured(self, tmp_path, monkeypatch):
        captured = self._launch_unmeasurable(tmp_path, monkeypatch, n_ctx = NATIVE)
        assert "not a measured ceiling" in (captured["backend"].last_load_warning or "")

    def test_auto_keeps_the_floor(self, tmp_path, monkeypatch):
        """Auto asked for nothing, so it still gets the arm's conservative floor rather
        than the native length it cannot vouch for."""
        captured = self._launch_unmeasurable(tmp_path, monkeypatch, n_ctx = 0)
        assert _ctx_values(captured["cmd"])[-1] == str(_FIT_MIN_CTX)
        assert captured["backend"].max_context_length == _FIT_MIN_CTX

    def test_auto_is_not_told_anything(self, tmp_path, monkeypatch):
        """The notice explains a request that was not checked. Auto made no request, and
        an advisory on every unreadable header is noise."""
        captured = self._launch_unmeasurable(tmp_path, monkeypatch, n_ctx = 0)
        assert captured["backend"].last_load_warning is None

    def test_a_measured_ceiling_still_refuses(self, tmp_path, monkeypatch):
        """The Metal guard is untouched where it has a measurement to refuse against."""
        with pytest.raises(RuntimeError, match = "unified"):
            _metal_launch(
                tmp_path,
                monkeypatch,
                n_ctx = 32768,
                metal = True,
                can_estimate_kv = True,
            )
