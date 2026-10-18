# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A chat load that took the GPU but fell back to the CPU drops CHAT once nothing holds VRAM.

Otherwise the next image or video load evicts a model that never used the GPU.
"""

import inspect

import pytest

import core.inference.model_slots as model_slots
import routes.inference as inf
from core.inference import gpu_arbiter


class FakeLlama:
    holds_no_vram = False

    def __init__(self, active = False):
        self.is_loaded = self.is_active = active
        self.model_identifier = "org/chat-GGUF" if active else None


class FakeOrchestrator:
    def __init__(
        self,
        active = None,
        entry = None,
    ):
        self.active_model_name = active
        self.models = {active: dict(entry or {})} if active else {}
        self.loading_models = ()


PRIMARY = "org/tts-audio-cpp"


@pytest.fixture
def evicted(monkeypatch):
    calls = []
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: calls.append("chat"))
    monkeypatch.setitem(
        gpu_arbiter._EVICTORS, gpu_arbiter.DIFFUSION, lambda: calls.append("diffusion")
    )
    monkeypatch.setattr(gpu_arbiter, "_release_idle_video_servers", lambda reason: None)
    monkeypatch.setattr(inf, "_llama_cpp_backend", FakeLlama())
    monkeypatch.setattr(model_slots, "slots", [])
    monkeypatch.setattr(model_slots, "stuck", [])
    monkeypatch.setattr(model_slots, "loading", None)
    return calls


def _primary(entry):
    return FakeOrchestrator(PRIMARY, entry)


def test_a_cpu_fallback_primary_drops_chat_so_media_evicts_nothing(evicted):
    assert inf.release_chat_after_cpu_fallback(_primary({"audio_device_fallback": "cpu"}), PRIMARY)
    assert gpu_arbiter.current_owner() is None
    gpu_arbiter.acquire_for(gpu_arbiter.DIFFUSION, lambda: None)
    assert evicted == [] and gpu_arbiter.current_owner() == gpu_arbiter.DIFFUSION


@pytest.mark.parametrize(
    "entry", [{}, {"audio_device_fallback": None}, {"audio_device_fallback": "gpu"}]
)
def test_no_recorded_fallback_keeps_chat(evicted, entry):
    assert not inf.release_chat_after_cpu_fallback(_primary(entry), PRIMARY)
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT


def test_a_replaced_primary_keeps_chat(evicted):
    backend = _primary({"audio_device_fallback": "cpu"})
    backend.active_model_name = "org/other"
    assert not inf.release_chat_after_cpu_fallback(backend, PRIMARY)
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT


def test_a_kept_model_loading_keeps_chat(evicted, monkeypatch):
    slot = model_slots.ExtraSlot(FakeLlama(), FakeOrchestrator(), "owner")
    monkeypatch.setattr(model_slots, "loading", (slot, "org/kept-GGUF"))
    assert not inf.release_chat_after_cpu_fallback(
        _primary({"audio_device_fallback": "cpu"}), PRIMARY
    )
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT


def test_a_kept_model_on_the_gpu_keeps_chat(evicted, monkeypatch):
    kept = model_slots.ExtraSlot(FakeLlama(active = True), FakeOrchestrator(), "owner")
    monkeypatch.setattr(model_slots, "slots", [kept])
    assert model_slots.holds_vram()
    assert not inf.release_chat_after_cpu_fallback(
        _primary({"audio_device_fallback": "cpu"}), PRIMARY
    )
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT
    # A CPU-only kept model holds no VRAM, so the claim goes.
    kept.llama._gpu_offload_active = False
    assert inf.release_chat_after_cpu_fallback(_primary({"audio_device_fallback": "cpu"}), PRIMARY)


def test_a_primary_llama_server_on_the_gpu_keeps_chat(evicted, monkeypatch):
    monkeypatch.setattr(inf, "_llama_cpp_backend", FakeLlama(active = True))
    assert not inf.release_chat_after_cpu_fallback(
        _primary({"audio_device_fallback": "cpu"}), PRIMARY
    )
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT


def test_a_kept_audio_cpp_fallback_beside_a_gpu_primary_keeps_chat(evicted, monkeypatch):
    # The kept model's own orchestrator is not the primary: the call site is gated on `replacing`,
    # and even if it ran, the primary llama-server's VRAM keeps the claim.
    monkeypatch.setattr(inf, "_llama_cpp_backend", FakeLlama(active = True))
    kept = FakeOrchestrator("org/kept-audio-cpp", {"audio_device_fallback": "cpu"})
    monkeypatch.setattr(model_slots, "slots", [model_slots.ExtraSlot(FakeLlama(), kept, "owner")])
    assert not inf.release_chat_after_cpu_fallback(kept, "org/kept-audio-cpp")
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT
    gpu_arbiter.acquire_for(gpu_arbiter.DIFFUSION, lambda: None)
    assert evicted == ["chat"]


def test_the_release_runs_for_a_primary_gpu_load_before_the_response():
    source = inspect.getsource(inf._load_model_impl)
    call = "await asyncio.to_thread(release_chat_after_cpu_fallback, backend, config.identifier)"
    assert source.count(call) == 1
    gate = "if replacing and chat_load_needs_gpu:\n"
    assert source.count(gate) == 1
    at = source.index(call)
    # The gate opens the block, then only the comment line sits between it and the call.
    assert source[source.index(gate) + len(gate) : at].strip().startswith("#")
    assert source[source.index(gate) + len(gate) : at].count("\n") == 1
    # The standard path's response statement follows directly, so the call is never inside its arguments.
    after = source[at + len(call) :]
    assert after.lstrip().startswith("return LoadResponse(")
