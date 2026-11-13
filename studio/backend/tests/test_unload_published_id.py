# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /unload finds a model under the ids Studio publishes for it, and logs only real unloads (#13766)."""

import asyncio

import pytest

import core.inference.model_slots as model_slots
import core.inference.orchestrator as orchestrator
import routes.inference as inf
from models.inference import UnloadRequest

SCAN_DIR = "/srv/models/org/X-GGUF"


class FakeLlama:
    holds_no_vram = False

    def __init__(
        self,
        identifier = None,
        variant = None,
        advertised = None,
    ):
        self.model_identifier = identifier
        self.hf_variant = variant
        self._openai_advertised_id = advertised
        self.is_loaded = self.is_active = identifier is not None
        self.is_vision = False
        self.context_length = 4096

    def unload_model(self):
        self.is_loaded = self.is_active = False


class FakeOrchestrator:
    def __init__(
        self,
        active = None,
        advertised = None,
    ):
        self.active_model_name = active
        self.models = {active: {}} if active else {}
        self._openai_advertised_id = advertised
        self.unloaded = []

    def unload_model(self, name):
        if name == self.active_model_name:
            self.unloaded.append(name)
            self.active_model_name = None
            self.models = {}
        return True


@pytest.fixture
def lifecycle(monkeypatch):
    rows = []
    monkeypatch.setattr(model_slots, "slots", [])
    monkeypatch.setattr(model_slots, "stuck", [])
    monkeypatch.setattr(inf, "_raise_if_sidecar_swap_in_progress", lambda: None)
    monkeypatch.setattr(inf.api_monitor, "record_lifecycle", lambda **kwargs: rows.append(kwargs))
    return rows


def _serve(
    monkeypatch,
    llama,
    orch = None,
):
    monkeypatch.setattr(inf, "_llama_cpp_backend", llama)
    monkeypatch.setattr(orchestrator, "_inference_backend", orch or FakeOrchestrator())


def _unload(model_path):
    return asyncio.run(inf._unload_model_impl(UnloadRequest(model_path = model_path), "s"))


@pytest.mark.parametrize(
    "named",
    [
        "org/X-GGUF",  # /v1/models and status.active_model
        "ORG/x-gguf",
        "org/X-GGUF:X-custom-NVFP4",  # the API monitor's loaded-model label
        SCAN_DIR,  # status.model_identifier, what the Unload button sends
    ],
)
def test_scan_folder_gguf_unloads_under_every_published_id(monkeypatch, lifecycle, named):
    llama = FakeLlama(SCAN_DIR, "X-custom-NVFP4", advertised = "org/X-GGUF")
    _serve(monkeypatch, llama)
    assert _unload(named).status == "unloaded"
    assert not llama.is_loaded
    assert [row["model"] for row in lifecycle] == ["org/X-GGUF:X-custom-NVFP4"]


@pytest.mark.parametrize("named", ["org/X-GGUF:Q8_0", "org/Other-GGUF", "X-custom-NVFP4"])
def test_a_name_for_another_model_leaves_it_loaded_and_logs_nothing(monkeypatch, lifecycle, named):
    llama = FakeLlama(SCAN_DIR, "X-custom-NVFP4", advertised = "org/X-GGUF")
    _serve(monkeypatch, llama)
    # Still the documented idempotent no-op for a stale tab.
    assert _unload(named).status == "unloaded"
    assert llama.is_loaded
    assert lifecycle == []


def test_unload_with_nothing_loaded_logs_nothing(monkeypatch, lifecycle):
    _serve(monkeypatch, FakeLlama())
    assert _unload("org/X-GGUF").status == "unloaded"
    assert lifecycle == []


def test_a_load_in_flight_is_not_matched_by_a_stale_advertised_id(monkeypatch):
    llama = FakeLlama("/srv/models/org/New-GGUF", "Q4_K_M", advertised = "org/Old-GGUF")
    llama.is_loaded = False
    assert not inf._names_the_resident_gguf(llama, "org/Old-GGUF")
    assert inf._names_the_resident_gguf(llama, "/srv/models/org/New-GGUF")


def test_auto_switched_safetensors_unloads_under_its_advertised_id(monkeypatch, lifecycle):
    orch = FakeOrchestrator("/srv/models/org/Y", advertised = "org/Y")
    _serve(monkeypatch, FakeLlama(), orch)
    assert _unload("org/Y").status == "unloaded"
    assert orch.unloaded == ["/srv/models/org/Y"]
    assert [row["model"] for row in lifecycle] == ["org/Y"]
