# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A load still fitting or starting has resolved its exact setting before any child runs.

Skipped as inactive, a save made in that window answered `reload_required = false`, and the load
then published under the old setting with nothing left to say so.
"""

import inspect

import core.inference.llama_cpp as llama_mod
from core.inference.llama_cpp import LlamaCppBackend
import routes.inference as inference
import routes.settings as settings


class _Starting:
    is_active = False
    is_diffusion = False
    requested_exact_concurrency = "off"

    def __init__(self, pending):
        self._exact_setting_pending = pending


def _primary(monkeypatch, backend):
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(llama_mod, "_serving_backends_snapshot", lambda: [])


def test_a_load_resolved_to_the_old_setting_needs_a_reload(monkeypatch):
    _primary(monkeypatch, _Starting("off"))
    assert settings._exact_concurrency_reload_required("on") is True


def test_a_load_resolved_to_the_saved_setting_does_not(monkeypatch):
    _primary(monkeypatch, _Starting("on"))
    assert settings._exact_concurrency_reload_required("on") is False


def test_no_load_in_flight_answers_as_before(monkeypatch):
    _primary(monkeypatch, _Starting(None))
    assert settings._exact_concurrency_reload_required("on") is False


def test_the_load_marks_what_it_resolved_and_the_scope_clears_it():
    load = inspect.getsource(LlamaCppBackend.load_model)
    resolved = load.index("_exact_setting = capture_exact_concurrency(")
    window = load[resolved : resolved + 400]
    assert "_exact.resolve_exact_setting(intent.exact_concurrency)" in window
    assert 'setattr(self, "_exact_setting_pending", setting)' in window
    scope = inspect.getsource(LlamaCppBackend._serial_load_scope)
    assert "self._exact_setting_pending = None" in scope.split("finally:")[1]
