# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``GET /api/inference/status`` must answer with a GGUF loaded.

The loaded-GGUF branch sits inside a ``try`` whose ``except`` reports 500, so an undefined
name there is invisible until it runs -- which is how ``_native_grant_backed`` once lost its
binding to a refactor while ``is_local_model`` kept reading it. The backend is a stub.
"""

import asyncio
import os

import pytest

import routes.inference as inference_route


class _StatusBackend:
    """A loaded GGUF with the shape ``get_status`` reads, not a full backend.

    Unknown attributes answer None so a later Optional field needs no edit here; typed ones
    are set explicitly so the response model validates for real, not against a mock.
    """

    def __init__(
        self,
        model_identifier,
        *,
        native_grant_backed = None,
        display_label = None,
    ):
        self.model_identifier = model_identifier
        self.is_loaded = True
        if native_grant_backed is not None:
            self._native_grant_backed = native_grant_backed
        if display_label is not None:
            self._native_display_label = display_label
        self.is_vision = False
        self.is_diffusion = False
        self.supports_reasoning = False
        self.reasoning_always_on = False
        self.supports_preserve_thinking = False
        self.supports_tools = False
        self.reasoning_style = "enable_thinking"
        self.reasoning_effort_levels = []
        self.requested_parallel_slots = 1
        self.effective_parallel_slots = 1
        self._is_audio = False
        self._has_audio_input = False
        self.tensor_parallel = False
        self.gpu_memory_mode = "auto"
        self.gpu_layers = 0
        self.n_cpu_moe = 0
        self.n_moe_layers = 0

    def __getattr__(self, name):
        # Not a MagicMock on purpose: a bool or int field must fail validation, not pass.
        if name.startswith("__"):
            raise AttributeError(name)
        return None


@pytest.fixture
def status_route(monkeypatch):
    """The handler with its filesystem and config lookups stubbed out."""
    monkeypatch.setattr(inference_route, "load_inference_config", lambda *a, **k: None)
    monkeypatch.setattr(
        inference_route, "resolve_effective_chat_template_override", lambda **k: None
    )

    def _run(backend):
        monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
        return asyncio.run(inference_route.get_status("tester"))

    return _run


def test_a_loaded_repo_gguf_reports_its_public_id(status_route):
    status = status_route(_StatusBackend("org/A-GGUF", native_grant_backed = False))
    assert status.is_gguf is True
    assert status.active_model == "org/A-GGUF"
    assert status.model_identifier == "org/A-GGUF"
    # Not a path, so provenance falls through to the filesystem check and stays False.
    assert status.is_local_model is False


def test_a_backend_without_the_flag_still_reports(status_route):
    # A server started before the flag existed: read through a default, do not assume it.
    backend = _StatusBackend("org/A-GGUF")
    assert not hasattr(backend, "__dict__") or "_native_grant_backed" not in backend.__dict__
    status = status_route(backend)
    assert status.is_gguf is True
    assert status.is_local_model is False


def test_a_native_lease_load_reports_the_label_not_the_leased_path(status_route):
    # The leased on-disk path is exactly what /status must not hand back.
    leased = os.path.join(os.sep, "models", "private", "A-Q4_K_M.gguf")
    status = status_route(_StatusBackend(leased, native_grant_backed = True))
    assert status.model_identifier is None, "the leased path must not be published"
    assert status.active_model == "A-Q4_K_M.gguf"
    assert status.is_local_model is True


def test_a_local_path_load_without_a_lease_is_still_local(status_route):
    # is_local_model is provenance, not lease bookkeeping: a plain local path counts.
    local = os.path.join(os.sep, "models", "local", "A-Q4_K_M.gguf")
    status = status_route(_StatusBackend(local, native_grant_backed = False))
    assert status.is_gguf is True
    assert status.is_local_model is True
