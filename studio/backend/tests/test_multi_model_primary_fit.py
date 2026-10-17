# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A primary chat load beside a kept image or video model: make room under When they fit, refuse under Always."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import core.inference.model_slots as model_slots
import core.inference.orchestrator as orchestrator
import routes.inference as inf
from core.inference import gpu_arbiter, gpu_residency
from core.inference.llama_cpp import GpuMemoryShortError
from models.inference import LoadRequest, LoadResponse
from utils import multi_model_settings
from utils.account_context import AccountContext, arun_as

BOB = AccountContext("b" * 32, "bob")
IMG = {"kind": "image", "model": "org/Img"}


class FakeLlama:
    holds_no_vram, is_vision, context_length = False, False, 4096

    def __init__(self, identifier = None, variant = None):
        self.model_identifier, self.hf_variant = identifier, variant
        self.is_loaded = self.is_active = identifier is not None

class FakeOrchestrator:
    active_model_name, models = None, {}


@pytest.fixture
def env(monkeypatch):
    """Primary-only Unsloth Studio with spies on everything that could unload a model."""
    monkeypatch.setattr(inf, "_llama_cpp_backend", FakeLlama("org/A-GGUF", "Q4_K_M"))
    monkeypatch.setattr(orchestrator, "_inference_backend", FakeOrchestrator())
    monkeypatch.setattr(model_slots, "slots", [])
    monkeypatch.setattr(model_slots, "stuck", [])
    monkeypatch.setattr(inf, "LlamaCppBackend", FakeLlama)
    monkeypatch.setattr(inf, "InferenceOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(inf, "_raise_if_sidecar_swap_in_progress", lambda: None)
    monkeypatch.setattr(inf, "release_chat_gpu_claim", lambda: None)
    monkeypatch.setattr(multi_model_settings, "get_multi_model_enabled", lambda: True)
    state = SimpleNamespace(policy = "fit", requests = [], drops = [], evicts = [], unloads = [], free_calls = [],
                            frees = [IMG], errors = [], retry_flags = [])
    monkeypatch.setattr(gpu_residency, "current_policy", lambda: state.policy)
    monkeypatch.setattr(model_slots, "drop", lambda slot: state.drops.append(slot))
    monkeypatch.setattr(model_slots, "evict", lambda *a: state.evicts.append(a) or [])
    monkeypatch.setattr(inf, "_unload_primary", lambda: state.unloads.append(True))

    def free_for(owner, gpu_ids, reason, evicted = None, *, need_mib = None, short_mib = 0):
        state.free_calls.append({"owner": owner, "gpu_ids": tuple(gpu_ids or ()), "short_mib": short_mib})
        evicted.extend(dict(s) for s in state.frees)
        return [s["model"] for s in state.frees]

    monkeypatch.setattr(gpu_residency, "free_for", free_for)
    rows = {0: {"free_mib": 4096, "total_mib": 81920}}
    monkeypatch.setattr(gpu_residency, "_free_mib_by_gpu", lambda: (rows, None))
    monkeypatch.setattr(gpu_arbiter, "resident_slots", lambda o: [dict(IMG)] if o == gpu_arbiter.DIFFUSION else [])

    async def load(request, *args, **kwargs):
        state.requests.append(request)
        state.retry_flags.append(inf._primary_can_retry_fit.get())
        if state.errors and not request.force_alongside:
            raise state.errors.pop(0)
        return LoadResponse(status = "loaded", model = request.model_path, display_name = "A", inference = {})

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", load)
    return state


def _short(capped = False):
    return GpuMemoryShortError("needs 13 GB, 8 GB free", capped = capped, short_mib = 5000, gpu_indices = (0,))


def _gated(account = None):
    async def run():
        coro = inf.load_model_gated(LoadRequest(model_path = "org/C-GGUF"), None, "s")
        response = await (arun_as(account, coro) if account else coro)
        return response, inf._primary_can_retry_fit.get()

    return asyncio.run(run())


def _nothing_unloaded(state):
    assert state.drops == [] and state.evicts == [] and state.unloads == []
    assert model_slots.stuck == []


def test_fit_frees_the_image_model_and_retries(env):
    assert inf._primary_can_retry_fit.get() is False
    env.errors = [_short()]
    response, flag_after = _gated()
    assert len(env.requests) == 2 and not env.requests[1].force_alongside
    assert response.evicted == ["org/Img"]
    assert env.free_calls == [{"owner": gpu_arbiter.CHAT, "gpu_ids": (0,), "short_mib": 5000}]
    assert env.drops == [] and model_slots.stuck == []  # model_slots.drop(None) never runs
    assert env.retry_flags == [True, True] and flag_after is False


def test_fit_with_nothing_freed_lets_llama_server_fit_what_it_can(env):
    env.errors, env.frees = [_short()], []
    response, _ = _gated()
    assert [r.force_alongside for r in env.requests] == [False, True]
    assert response.status == "loaded" and not response.evicted
    _nothing_unloaded(env)


@pytest.mark.parametrize("policy", ["fit", "always"])
def test_capped_fit_retries_at_a_smaller_context(env, policy):
    env.policy, env.errors = policy, [_short(capped = True)]
    response, _ = _gated()
    assert [r.force_alongside for r in env.requests] == [False, True]
    assert response.status == "loaded"
    assert env.free_calls == []
    _nothing_unloaded(env)


def test_always_refuses_with_the_numbers_for_the_owner(env):
    env.policy, env.errors = "always", [_short()]
    with pytest.raises(gpu_arbiter.MultiModelAlwaysRefusal) as info:
        _gated()
    message = str(info.value)
    assert "is set to Always" in message
    # 5000 MiB short beside 4096 MiB free: needs 8.9 GB, 4.0 GB is free.
    assert "needs about 8.9 GB of GPU memory, and 4.0 GB is free" in message
    assert info.value.kind == "image" and info.value.model == "org/Img"
    assert env.free_calls == [] and len(env.requests) == 1
    _nothing_unloaded(env)


def test_always_refuses_a_managed_account_with_the_plain_copy(env):
    env.policy, env.errors = "always", [_short()]
    with pytest.raises(gpu_arbiter.MultiModelAlwaysRefusal) as info:
        _gated(BOB)
    message = str(info.value)
    assert "GB" not in message
    assert "the image model org/Img, which Unsloth Studio is set to keep loaded" in message
    assert env.free_calls == []
    _nothing_unloaded(env)


def _image_resident(monkeypatch):
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.DIFFUSION)
    monkeypatch.setattr(gpu_arbiter, "_residents", {gpu_arbiter.DIFFUSION: frozenset({0})})
    assert gpu_arbiter.holds(gpu_arbiter.DIFFUSION)


def _oom():
    return HTTPException(status_code = 500, detail = "CUDA out of memory")


def test_non_gguf_oom_under_always_beside_an_image_model_is_refused(env, monkeypatch):
    _image_resident(monkeypatch)
    env.policy, env.errors = "always", [_oom()]
    with pytest.raises(gpu_arbiter.MultiModelAlwaysRefusal) as info:
        _gated()
    assert "is set to Always" in str(info.value) and info.value.model == "org/Img"
    assert env.free_calls == []
    _nothing_unloaded(env)


def test_non_gguf_oom_under_fit_keeps_the_original_error(env, monkeypatch):
    _image_resident(monkeypatch)
    env.errors = [_oom()]
    with pytest.raises(HTTPException) as info:
        _gated()
    assert info.value.status_code == 500 and info.value.detail == "CUDA out of memory"
    assert env.free_calls == [] and len(env.requests) == 1
    _nothing_unloaded(env)


def test_non_gguf_oom_under_always_without_media_keeps_the_original_error(env, monkeypatch):
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    env.policy, env.errors = "always", [_oom()]
    with pytest.raises(HTTPException) as info:
        _gated()
    assert info.value.status_code == 500
    _nothing_unloaded(env)


def _placement(resolved, vulkan = False, native_gb = None):
    return SimpleNamespace(
        resolved_gpu_ids = resolved, gpu_ids_are_vulkan_ordinals = vulkan, native_required_gb = native_gb
    )


@pytest.mark.parametrize(
    "held, placement, expected",
    [
        ({}, _placement([0]), False),
        ({1: 4096}, _placement([0]), False),
        ({1: 4096}, _placement([1]), True),
        ({1: 4096}, _placement(None), True),
        ({1: 4096}, _placement([0], vulkan = True), True),
        ({}, _placement([0], vulkan = True), False),
    ],
)
def test_media_on_target_gpus(monkeypatch, held, placement, expected):
    monkeypatch.setattr(gpu_residency, "media_held_mib", lambda: held)
    assert inf._media_on_target_gpus(placement) is expected


def test_chat_fit_kwargs(monkeypatch):
    policy = SimpleNamespace(value = "off")
    monkeypatch.setattr(gpu_residency, "current_policy", lambda: policy.value)
    gguf, native = SimpleNamespace(is_gguf = True), SimpleNamespace(is_gguf = False)
    assert inf._chat_fit_kwargs(gguf, _placement([0])) == {}
    policy.value = "fit"
    token = inf._primary_can_retry_fit.set(True)
    try:
        assert inf._chat_fit_kwargs(gguf, _placement([0])) == {"policy": "fit"}
        # Anything but GGUF has no fit estimate before launch, even inside the retry.
        assert inf._chat_fit_kwargs(native, _placement([0])) == {"policy": "fit", "evict_first": True}
    finally:
        inf._primary_can_retry_fit.reset(token)
    # Auto-switch calls the tracked impl without load_model_gated, so nothing would make room after.
    assert inf._chat_fit_kwargs(gguf, _placement([0])) == {"policy": "fit", "evict_first": True}
    native_need = inf._chat_fit_kwargs(native, _placement([0, 1], native_gb = 8.0))
    assert native_need == {"policy": "fit", "need_mib": {0: 4096, 1: 4096}}
    policy.value = "always"
    assert inf._chat_fit_kwargs(gguf, _placement([0])) == {"policy": "always"}
