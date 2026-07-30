# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``GET /api/inference/status`` must answer with a GGUF loaded.

Every field the loaded-GGUF branch publishes is read off the llama backend inside a
``try`` whose ``except`` reports 500, so a name that goes undefined in that branch is
invisible until something loads a model and then asks for status. Nothing in either
suite called the handler that way, which is how the branch shipped broken once:
``_native_grant_backed`` lost its binding when the identity logic moved into
``_llama_status_model_ids``, while ``is_local_model`` kept reading it.

No GPU, no llama-server, no GGUF on disk: the backend is a stub.
"""

import asyncio
import os

import pytest

import routes.inference as inference_route


class _StatusBackend:
    """A loaded GGUF, with the shape ``get_status`` reads rather than a full backend.

    Unknown attributes answer None so an Optional field the handler grows later does not
    have to be added here; the typed ones are set explicitly so the response model
    validates for real instead of being handed a mock.
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
        # Only reached for attributes __init__ did not set. Deliberately not a MagicMock:
        # a bool or int field must fail validation here rather than silently pass.
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
    # A server started before _native_grant_backed existed: the handler must read it
    # through a default rather than assume the attribute is there.
    backend = _StatusBackend("org/A-GGUF")
    assert not hasattr(backend, "__dict__") or "_native_grant_backed" not in backend.__dict__
    status = status_route(backend)
    assert status.is_gguf is True
    assert status.is_local_model is False


def test_a_native_lease_load_reports_the_label_not_the_leased_path(status_route):
    # A native-grant load must publish only the display label: the on-disk path it was
    # leased is exactly what /status is not allowed to hand back.
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
