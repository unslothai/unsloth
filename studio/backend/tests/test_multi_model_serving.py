# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Models loaded alongside the primary one are routed to by request model name."""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import inspect

import core.inference.orchestrator as orchestrator
import routes.inference as inf
from auth import policy
from auth.authentication import get_current_subject
from core.inference import gpu_arbiter
from models.inference import InferenceStatusResponse, LoadRequest, UnloadRequest
from utils.account_context import AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


class FakeLlama:
    def __init__(
        self,
        identifier = None,
        variant = None,
    ):
        self.model_identifier = identifier
        self.hf_variant = variant
        self.is_loaded = self.is_active = identifier is not None
        self.context_length = 4096

    def unload_model(self):
        self.is_loaded = self.is_active = False

    def _cleanup(self):
        pass


class FakeOrchestrator:
    def __init__(self, active = None):
        self.active_model_name = active
        self.models = {active: {}} if active else {}

    def unload_model(self, name):
        self.active_model_name = None

    def _cleanup(self):
        pass


@pytest.fixture
def backends(monkeypatch):
    primary = FakeLlama("org/A-GGUF", "Q4_K_M")
    extra = inf._ExtraSlot(FakeLlama("org/B-GGUF", "Q8_0"), FakeOrchestrator(), "owner")
    monkeypatch.setattr(inf, "_llama_cpp_backend", primary)
    monkeypatch.setattr(orchestrator, "_inference_backend", FakeOrchestrator())
    monkeypatch.setattr(inf, "_extra_slots", [extra])
    monkeypatch.setattr(inf, "_evicted", {})
    monkeypatch.setattr(inf, "_primary_request", None)
    monkeypatch.setattr(inf, "_evicted_owner", {})
    monkeypatch.setattr(inf, "_primary_account", None)
    return primary, extra


def _routed(requested):
    async def run():
        slot = await inf._route_to_extra_slot(requested)
        return slot, inf.get_llama_cpp_backend()

    return asyncio.run(run())


def test_request_model_name_picks_the_backend(backends):
    primary, extra = backends
    assert _routed("org/B-GGUF") == (extra, extra.llama)
    assert _routed("org/B-GGUF:Q8_0") == (extra, extra.llama)
    assert _routed("org/B-GGUF:Q4_K_M") == (None, primary)
    assert _routed("org/A-GGUF") == (None, primary)
    assert _routed(None) == (None, primary)


def test_a_safetensors_slot_is_served_by_its_own_orchestrator(backends):
    safetensors = inf._ExtraSlot(FakeLlama(), FakeOrchestrator("org/C"), "owner")
    inf._extra_slots.append(safetensors)

    async def run():
        slot = await inf._route_to_extra_slot("org/C")
        return slot, inf.get_llama_cpp_backend().is_loaded, inf.get_inference_backend()

    assert asyncio.run(run()) == (safetensors, False, safetensors.orchestrator)


def test_loaded_models_lists_every_backend(backends):
    app = FastAPI()
    app.include_router(inf.studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    with TestClient(app) as client:
        data = client.get("/api/inference/loaded-models").json()["data"]
    assert {entry["id"] for entry in data} == {"org/A-GGUF", "org/B-GGUF"}
    inf._extra_slots.append(inf._ExtraSlot(FakeLlama(), FakeOrchestrator("org/C"), "owner"))
    with TestClient(app) as client:
        data = client.get("/api/inference/loaded-models").json()["data"]
    assert [entry["id"] for entry in data] == ["org/A-GGUF", "org/B-GGUF", "org/C"]


def _selected(monkeypatch, request):
    monkeypatch.setattr(inf, "LlamaCppBackend", FakeLlama)
    monkeypatch.setattr(inf, "InferenceOrchestrator", FakeOrchestrator)

    async def run():
        slot = await inf._select_load_slot(request)
        assert inf.routed_slot.get() is slot
        return slot

    return asyncio.run(run())


def test_alongside_load_gets_a_new_slot(backends, monkeypatch):
    _, extra = backends
    fresh = _selected(monkeypatch, LoadRequest(model_path = "org/C-GGUF", alongside = True))
    assert fresh not in (None, extra)


def test_plain_load_replaces_the_primary(backends, monkeypatch):
    assert _selected(monkeypatch, LoadRequest(model_path = "org/C-GGUF")) is None


def test_alongside_load_uses_an_empty_primary(backends, monkeypatch):
    primary, _ = backends
    primary.unload_model()
    assert _selected(monkeypatch, LoadRequest(model_path = "org/C-GGUF", alongside = True)) is None


def test_loading_a_model_an_extra_slot_serves_reuses_it(backends, monkeypatch):
    _, extra = backends
    request = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0")
    assert _selected(monkeypatch, request) is extra


def test_unload_drops_only_the_named_extra_slot(backends, monkeypatch):
    primary, extra = backends
    released = []
    monkeypatch.setattr(inf, "release_chat_gpu_claim", lambda: released.append(True))
    response = asyncio.run(inf._unload_model_impl(UnloadRequest(model_path = "org/B-GGUF"), "s"))
    assert response.status == "unloaded"
    assert primary.is_loaded and not extra.llama.is_loaded
    assert inf._extra_slots == [] and released == [True]


def test_a_failed_llama_unload_still_cleans_the_orchestrator(backends):
    _, extra = backends
    cleaned = []
    extra.llama.unload_model = lambda: 1 / 0
    extra.orchestrator._cleanup = lambda: cleaned.append(True)
    with pytest.raises(ZeroDivisionError):
        inf._drop_extra_slot(extra)
    assert inf._extra_slots == [] and cleaned == [True]


def test_a_managed_account_sees_only_its_own_slots(backends, monkeypatch):
    primary, extra = backends
    monkeypatch.setattr(inf.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(inf, "current_account_id", lambda: "someone-else")
    assert _routed("org/B-GGUF") == (None, primary)
    monkeypatch.setattr(inf, "current_account_id", lambda: "owner")
    assert _routed("org/B-GGUF") == (extra, extra.llama)


def test_idle_unload_spares_pinned_slots(backends):
    _, extra = backends
    inf.unload_extra_models(keep = lambda llama: True)
    assert inf._extra_slots == [extra]
    inf.unload_extra_models(keep = lambda llama: False)
    assert inf._extra_slots == []


def test_status_describes_the_named_slot_and_lists_the_rest(backends, monkeypatch):
    async def slot_status(subject):
        return InferenceStatusResponse(active_model = inf.get_llama_cpp_backend().model_identifier)

    monkeypatch.setattr(inf, "_slot_status", slot_status)
    named = asyncio.run(inf.get_status("s", model = "org/B-GGUF"))
    assert (named.active_model, named.loaded) == ("org/B-GGUF", ["org/A-GGUF"])
    primary = asyncio.run(inf.get_status("s"))
    assert (primary.active_model, primary.loaded) == ("org/A-GGUF", ["org/B-GGUF"])


def test_stop_loading_reaches_the_slot_being_filled(backends, monkeypatch):
    primary, _ = backends
    filling = inf._ExtraSlot(FakeLlama(), FakeOrchestrator(), "owner")
    filling.llama.model_identifier = "org/D-GGUF"
    filling.llama.is_active = True
    inf._extra_slots.append(filling)
    monkeypatch.setattr(inf, "_loading_slot", (filling, "org/D-GGUF"))
    response = asyncio.run(inf._unload_model_impl(UnloadRequest(model_path = "org/D-GGUF"), "s"))
    assert response.status == "unloaded"
    assert primary.is_loaded and not filling.llama.is_active


def test_the_primary_wins_a_bare_name_both_serve(backends):
    primary, _ = backends
    inf._extra_slots.append(
        inf._ExtraSlot(FakeLlama("org/A-GGUF", "Q8_0"), FakeOrchestrator(), "owner")
    )
    assert _routed("org/A-GGUF") == (None, primary)
    assert _routed("org/A-GGUF:Q8_0")[0] is inf._extra_slots[-1]


def test_another_account_cannot_stop_a_slot_load(backends, monkeypatch):
    primary, _ = backends
    filling = inf._ExtraSlot(FakeLlama(), FakeOrchestrator(), "owner")
    inf._extra_slots.append(filling)
    monkeypatch.setattr(inf, "_loading_slot", (filling, "org/D-GGUF"))
    monkeypatch.setattr(inf.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(inf, "current_account_id", lambda: "someone-else")
    assert inf._visible_loading_slot() is None


def test_a_slot_evicted_while_it_loads_is_torn_down(backends, monkeypatch):
    monkeypatch.setattr(inf, "LlamaCppBackend", FakeLlama)
    monkeypatch.setattr(inf, "InferenceOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(inf, "_raise_if_sidecar_swap_in_progress", lambda: None)
    spawned = []

    async def load_after_eviction(request, *args, **kwargs):
        slot = inf._extra_slots[-1]
        inf.unload_extra_models()
        slot.llama.is_loaded = slot.llama.is_active = True
        spawned.append(slot)
        return "loaded"

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", load_after_eviction)
    monkeypatch.setattr(inf, "release_chat_gpu_claim", lambda: None)
    request = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    assert asyncio.run(inf.load_model_gated(request, None, "s")) == "loaded"
    assert inf._extra_slots == [] and not spawned[0].llama.is_active


def test_only_a_new_slot_skips_the_running_chat_check(backends, monkeypatch):
    monkeypatch.setattr(inf, "LlamaCppBackend", FakeLlama)
    monkeypatch.setattr(inf, "InferenceOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(inf, "_raise_if_sidecar_swap_in_progress", lambda: None)
    seen = []

    async def load(request, *args, on_reload_confirmed, **kwargs):
        seen.append(on_reload_confirmed is None)
        llama = inf.get_llama_cpp_backend()
        llama.model_identifier = request.model_path
        llama.is_loaded = llama.is_active = True
        return "loaded"

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", load)
    for request in (
        LoadRequest(model_path = "org/C-GGUF", alongside = True),
        LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0", force_reload = True),
        LoadRequest(model_path = "org/D-GGUF"),
    ):
        asyncio.run(inf.load_model_gated(request, None, "s"))
    assert seen == [True, False, False]


def test_loading_alongside_leaves_the_primary_with_the_account_that_loaded_it(monkeypatch):
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setattr(gpu_arbiter, "_owner_account", None)
    gpu_arbiter.acquire_for(gpu_arbiter.CHAT, lambda: None, account_id = ALICE.account_id)
    gpu_arbiter.acquire_for(
        gpu_arbiter.CHAT, lambda: None, account_id = BOB.account_id, alongside = True
    )
    assert gpu_arbiter.owner_account() == ALICE.account_id
    gpu_arbiter.acquire_for(gpu_arbiter.CHAT, lambda: None, account_id = BOB.account_id)
    assert gpu_arbiter.owner_account() == BOB.account_id


def test_an_account_lists_its_own_slot_but_not_a_foreign_primary(backends, monkeypatch):
    primary, extra = backends
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    monkeypatch.setattr(gpu_arbiter, "_owner_account", ALICE.account_id)
    extra.account = BOB.account_id

    def listed():
        return [entry["id"] for entry in inf._openai_model_objects()]

    assert run_as(BOB, listed) == ["org/B-GGUF"]
    assert run_as(ALICE, listed) == ["org/A-GGUF"]


def test_a_slot_load_never_drops_the_chat_claim_the_primary_holds():
    source = inspect.getsource(inf._load_model_impl)
    assert source.count("if replacing and not chat_load_needs_gpu:") == 2
    assert "if not chat_load_needs_gpu:" not in source


def test_a_failed_slot_load_drops_the_slot_and_releases_the_claim(backends, monkeypatch):
    monkeypatch.setattr(inf, "LlamaCppBackend", FakeLlama)
    monkeypatch.setattr(inf, "InferenceOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(inf, "_raise_if_sidecar_swap_in_progress", lambda: None)
    released = []
    monkeypatch.setattr(inf, "release_chat_gpu_claim", lambda: released.append(True))

    async def failing_load(*args, **kwargs):
        raise RuntimeError("no such repo")

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", failing_load)
    request = LoadRequest(model_path = "org/missing-GGUF", alongside = True)
    with pytest.raises(RuntimeError):
        asyncio.run(inf.load_model_gated(request, None, "s"))
    assert len(inf._extra_slots) == 1 and inf._loading_slot is None and released == [True]


def _slot(
    model,
    variant = None,
    last_used = 0.0,
):
    return inf._ExtraSlot(
        FakeLlama(model, variant),
        FakeOrchestrator(),
        "owner",
        LoadRequest(model_path = model, gguf_variant = variant, alongside = True),
        last_used,
    )


def _gated_load_fakes(monkeypatch, short_fits):
    from core.inference.llama_cpp import GpuMemoryShortError

    monkeypatch.setattr(inf, "LlamaCppBackend", FakeLlama)
    monkeypatch.setattr(inf, "InferenceOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(inf, "_raise_if_sidecar_swap_in_progress", lambda: None)
    monkeypatch.setattr(inf, "release_chat_gpu_claim", lambda: None)

    async def load(request, *args, **kwargs):
        if short_fits and not request.force_alongside:
            kind = short_fits.pop()
            raise GpuMemoryShortError(
                "needs 13 GB, 8 GB free", capped = kind == "capped", short_mib = 5000
            )
        llama = inf.get_llama_cpp_backend()
        llama.model_identifier, llama.hf_variant = request.model_path, request.gguf_variant
        llama.is_loaded = llama.is_active = True
        return "loaded"

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", load)


def test_a_short_fit_evicts_the_least_recently_used_slot(backends, monkeypatch):
    _, extra = backends
    extra.request, extra.last_used = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0"), 5.0
    older = _slot("org/D-GGUF", last_used = 1.0)
    inf._extra_slots.append(older)
    _gated_load_fakes(monkeypatch, short_fits = [1])
    request = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    assert asyncio.run(inf.load_model_gated(request, None, "s")) == "loaded"
    assert [s.llama.model_identifier for s in inf._extra_slots] == ["org/B-GGUF", "org/C-GGUF"]
    assert not older.llama.is_active
    assert list(inf._evicted) == ["org/D-GGUF"] and inf._evicted["org/D-GGUF"].alongside
    assert inf._extra_slots[-1].request.model_path == "org/C-GGUF"


def test_eviction_takes_as_many_lru_slots_as_the_shortfall_needs(backends, monkeypatch):
    _, extra = backends
    extra.request, extra.last_used = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0"), 3.0
    extra.llama._planned_vram_mib = {0: 9000}
    small_old = _slot("org/D-GGUF", last_used = 1.0)
    small_old.llama._planned_vram_mib = {0: 500}
    mid = _slot("org/E-GGUF", last_used = 2.0)
    mid.llama._planned_vram_mib = {0: 6000}
    inf._extra_slots += [small_old, mid]
    # 500 + 6000 covers the 5000 MiB shortfall, so the most recent slot is spared.
    assert inf._eviction_victims(None, 5000) == [small_old, mid]
    _gated_load_fakes(monkeypatch, short_fits = [1])
    request = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    assert asyncio.run(inf.load_model_gated(request, None, "s")) == "loaded"
    assert [s.llama.model_identifier for s in inf._extra_slots] == ["org/B-GGUF", "org/C-GGUF"]
    assert set(inf._evicted) == {"org/D-GGUF", "org/E-GGUF"}


def test_a_short_fit_with_nothing_left_to_evict_is_a_409(backends, monkeypatch):
    from fastapi import HTTPException

    _, extra = backends
    _gated_load_fakes(monkeypatch, short_fits = [1, 1])
    request = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inf.load_model_gated(request, None, "s"))
    assert excinfo.value.status_code == 409 and "8 GB free" in excinfo.value.detail
    assert inf._extra_slots == [] and not extra.llama.is_active


def test_a_capped_context_is_taken_only_once_nothing_is_left_to_evict(backends, monkeypatch):
    _, extra = backends
    extra.request = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0")
    _gated_load_fakes(monkeypatch, short_fits = ["capped", "capped"])
    request = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    assert asyncio.run(inf.load_model_gated(request, None, "s")) == "loaded"
    assert not extra.llama.is_active and list(inf._evicted) == ["org/B-GGUF:Q8_0"]
    assert [s.llama.model_identifier for s in inf._extra_slots] == ["org/C-GGUF"]


def test_images_taking_the_gpu_remembers_every_chat_model(backends, monkeypatch):
    _, extra = backends
    extra.request = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0")
    monkeypatch.setattr(inf, "_primary_request", LoadRequest(model_path = "org/A-GGUF"))
    inf.note_chat_evicted()
    inf.unload_extra_models()
    assert set(inf._evicted) == {"org/A-GGUF", "org/B-GGUF:Q8_0"} and inf._extra_slots == []
    assert inf._evicted_request("org/b-gguf") is not None
    assert inf._evicted_request("org/B-GGUF:Q4_K_M") is None
    inf._forget_evicted("org/B-GGUF")
    assert set(inf._evicted) == {"org/A-GGUF"}


def test_an_evicted_model_is_restored_when_a_request_names_it(backends, monkeypatch):
    import auth.authentication as authentication

    monkeypatch.setattr(inf, "_extra_slots", [])
    inf._evicted["org/B-GGUF:Q8_0"] = LoadRequest(
        model_path = "org/B-GGUF", gguf_variant = "Q8_0", alongside = True
    )
    monkeypatch.setattr(authentication, "request_admitted_without_credential", lambda r: False)
    restored = []

    async def gated(request, *args, **kwargs):
        restored.append(
            (request.model_path, request.alongside, kwargs.get("current_request_counted"))
        )
        inf._extra_slots.append(_slot(request.model_path, request.gguf_variant))

    monkeypatch.setattr(inf, "load_model_gated", gated)

    async def run():
        await inf._maybe_auto_switch_model("org/B-GGUF", object(), "s")
        return inf.routed_slot.get()

    assert asyncio.run(run()) is inf._extra_slots[0]
    assert restored == [("org/B-GGUF", True, True)]


def test_a_keyless_caller_restores_nothing(backends, monkeypatch):
    import auth.authentication as authentication
    from contextlib import suppress

    monkeypatch.setattr(inf, "_extra_slots", [])
    inf._evicted["org/B-GGUF"] = LoadRequest(model_path = "org/B-GGUF", alongside = True)
    monkeypatch.setattr(authentication, "request_admitted_without_credential", lambda r: True)
    restored = []

    async def gated(request, *args, **kwargs):
        restored.append(request.model_path)

    monkeypatch.setattr(inf, "load_model_gated", gated)
    with suppress(Exception):
        asyncio.run(inf._maybe_auto_switch_model("org/B-GGUF", object(), "s"))
    assert restored == []


def test_a_manual_unload_forgets_the_evicted_model(backends, monkeypatch):
    monkeypatch.setattr(inf, "release_chat_gpu_claim", lambda: None)
    inf._evicted["org/C-GGUF"] = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    asyncio.run(inf._unload_model_impl(UnloadRequest(model_path = "org/C-GGUF"), "s"))
    assert inf._evicted == {}


def test_a_load_that_tears_nothing_down_stops_no_chat():
    source = inspect.getsource(inf._load_model_impl)
    assert source.count("if serving and on_reload_confirmed is not None:") == 3
    assert source.count("if replacing and serving:") == 2
    assert "if on_reload_confirmed is not None:" not in source


def test_routing_marks_a_slot_as_used(backends):
    _, extra = backends
    assert extra.last_used == 0.0
    _routed("org/B-GGUF")
    assert extra.last_used > 0.0


def test_an_evicted_slot_keeps_its_conversation_kv_until_it_is_back(backends, monkeypatch):
    import core.inference.llama_keepwarm as keepwarm
    import utils.openai_auto_switch_settings as settings

    _, extra = backends
    extra.request = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0")
    manifest = {"dir": "/tmp/x", "slots": [{"id": 0, "filename": "s0.bin"}]}
    extra.llama.save_slots_for_resume = lambda: manifest
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: True)
    monkeypatch.setattr(inf, "_evicted_kv", {})
    inf._drop_extra_slot(extra, stash = True)
    assert inf._evicted_kv == {"org/B-GGUF:Q8_0": manifest}

    restored, deleted = [], []

    def restore(backend, kv):
        # Under the load gate, so a load queued behind this one cannot publish first.
        assert keepwarm._load_lock.locked()
        restored.append((backend, kv))

    monkeypatch.setattr(keepwarm, "restore_kv_resume", restore)
    monkeypatch.setattr(keepwarm, "_delete_resume_files", lambda kv: deleted.append(kv))
    _gated_load_fakes(monkeypatch, short_fits = [])
    request = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0", alongside = True)
    asyncio.run(inf.load_model_gated(request, None, "s"))
    assert (
        len(restored) == 1
        and restored[0][1] is manifest
        and restored[0][0] is inf._extra_slots[-1].llama
    )
    assert inf._evicted_kv == {} and deleted == []

    inf._evicted_kv["org/C-GGUF"] = manifest
    inf._evicted["org/C-GGUF"] = LoadRequest(model_path = "org/C-GGUF")
    inf._forget_evicted("org/C-GGUF")
    assert deleted == [manifest] and inf._evicted_kv == {}


def test_a_load_prices_the_vram_the_other_servers_hold():
    from core.inference import llama_cpp
    from core.inference.llama_cpp import LlamaCppBackend

    a, b, c, stray = (LlamaCppBackend(manages_processes = False) for _ in range(4))
    for backend in (a, b, c):
        llama_cpp.register_serving_backend(backend)
    try:
        a._process = b._process = stray._process = object()
        a._planned_vram_mib = {0: 12000}
        b._planned_vram_mib = {0: 1000, 1: 500}
        stray._planned_vram_mib = {0: 7000}
        assert c._other_planned_vram_mib() == {0: 13000, 1: 500}
        assert a._other_planned_vram_mib() == {0: 1000, 1: 500}
        assert stray._other_planned_vram_mib() == {}
        a._kill_process()
        assert a._planned_vram_mib == {} and c._other_planned_vram_mib() == {0: 1000, 1: 500}
        b._process = None
        assert c._other_planned_vram_mib() == {}
    finally:
        for backend in (a, b, c):
            llama_cpp.unregister_serving_backend(backend)


def _hold_chat_claim(monkeypatch):
    import core.inference.llama_cpp as llama_cpp
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    monkeypatch.setattr(llama_cpp, "chat_load_active", lambda: False)


def test_unloading_the_last_slot_keeps_the_claim_the_primary_holds(backends, monkeypatch):
    primary, _ = backends
    _hold_chat_claim(monkeypatch)
    asyncio.run(inf._unload_model_impl(UnloadRequest(model_path = "org/B-GGUF"), "s"))
    assert primary.is_active and inf._extra_slots == []
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT


def test_a_failed_slot_load_keeps_the_claim_the_primary_holds(backends, monkeypatch):
    primary, _ = backends
    monkeypatch.setattr(inf, "_extra_slots", [])
    _hold_chat_claim(monkeypatch)
    monkeypatch.setattr(inf, "LlamaCppBackend", FakeLlama)
    monkeypatch.setattr(inf, "InferenceOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(inf, "_raise_if_sidecar_swap_in_progress", lambda: None)

    async def failing_load(*args, **kwargs):
        raise RuntimeError("no such repo")

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", failing_load)
    with pytest.raises(RuntimeError):
        asyncio.run(
            inf.load_model_gated(
                LoadRequest(model_path = "org/missing-GGUF", alongside = True), None, "s"
            )
        )
    assert primary.is_active and gpu_arbiter.current_owner() == gpu_arbiter.CHAT


def test_a_generation_is_tracked_on_the_slot_serving_it(backends):
    import threading

    _, extra = backends
    event = threading.Event()

    async def run():
        await inf._route_to_extra_slot("org/B-GGUF")
        with inf._TrackedCancel(event, "k"):
            inside = set(extra.generations)
        return inside

    assert asyncio.run(run()) == {event} and extra.generations == set()


def test_a_slot_still_generating_is_never_evicted(backends):
    import threading

    _, extra = backends
    older = _slot("org/D-GGUF", last_used = 1.0)
    inf._extra_slots.append(older)
    older.generations.add(threading.Event())
    assert inf._eviction_victims(None, 5000) == [extra]
    extra.generations.add(threading.Event())
    assert inf._eviction_victims(None, 5000) == []


def test_unloading_a_generating_slot_is_refused_unless_forced(backends, monkeypatch):
    import threading

    from fastapi import HTTPException

    _, extra = backends
    monkeypatch.setattr(inf, "release_chat_gpu_claim", lambda: True)
    event = threading.Event()
    extra.generations.add(event)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inf._unload_model_impl(UnloadRequest(model_path = "org/B-GGUF"), "s"))
    assert excinfo.value.status_code == 409 and extra.llama.is_loaded and not event.is_set()

    def finish_on_cancel():
        event.wait(5)
        extra.generations.discard(event)

    threading.Thread(target = finish_on_cancel, daemon = True).start()
    forced = UnloadRequest(model_path = "org/B-GGUF", force_cancel_active = True)
    asyncio.run(inf._unload_model_impl(forced, "s"))
    assert event.is_set() and not extra.llama.is_loaded and inf._extra_slots == []


def test_a_swap_that_bypasses_load_model_gated_still_records_the_primary(backends, monkeypatch):
    monkeypatch.setattr(inf, "current_account_id", lambda: "alice")
    inf._evicted["org/C-GGUF"] = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    inf._note_primary_load(LoadRequest(model_path = "org/C-GGUF", hf_token = "secret"))
    assert inf._evicted == {} and inf._primary_account == "alice"
    assert inf._primary_request.model_path == "org/C-GGUF" and inf._primary_request.hf_token is None
    assert "_note_primary_load(load_request)" in inspect.getsource(inf._maybe_auto_switch_model)
    assert "_note_primary_load(request)" in inspect.getsource(inf.load_model_for_preview)


def test_a_managed_account_neither_restores_nor_forgets_another_accounts_model(
    backends, monkeypatch
):
    _, extra = backends
    extra.request = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0", alongside = True)
    inf.note_chat_evicted()
    monkeypatch.setattr(inf.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(inf, "current_account_id", lambda: "someone-else")
    assert inf._evicted_request("org/B-GGUF") is None
    inf._forget_evicted("org/B-GGUF")
    assert list(inf._evicted) == ["org/B-GGUF:Q8_0"]
    monkeypatch.setattr(inf, "current_account_id", lambda: "owner")
    assert inf._evicted_request("org/B-GGUF") is not None


def test_the_load_response_names_the_models_evicted_for_it(backends, monkeypatch):
    from models.inference import LoadResponse

    _, extra = backends
    extra.request = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0")
    _gated_load_fakes(monkeypatch, short_fits = [1])
    loaded = LoadResponse.model_construct(status = "loaded", model = "org/C-GGUF", display_name = "C")

    async def load(request, *args, **kwargs):
        if not request.force_alongside and not getattr(load, "raised", False):
            from core.inference.llama_cpp import GpuMemoryShortError
            load.raised = True
            raise GpuMemoryShortError("short", short_mib = 5000)
        llama = inf.get_llama_cpp_backend()
        llama.model_identifier, llama.is_loaded, llama.is_active = request.model_path, True, True
        return loaded

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", load)
    response = asyncio.run(
        inf.load_model_gated(LoadRequest(model_path = "org/C-GGUF", alongside = True), None, "s")
    )
    assert response.evicted == ["org/B-GGUF:Q8_0"]


def test_a_slot_server_leaves_the_primary_pidfile_alone(monkeypatch):
    import utils.process_lifetime as process_lifetime
    from core.inference.llama_cpp import LlamaCppBackend

    written, cleared, adopted = [], [], []
    monkeypatch.setattr(
        LlamaCppBackend, "_record_server_pid", classmethod(lambda cls, pid: written.append(pid))
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_clear_server_pid", classmethod(lambda cls: cleared.append(True))
    )
    monkeypatch.setattr(process_lifetime, "adopt_pid", adopted.append)
    slot = LlamaCppBackend(manages_processes = False)
    slot._owns_pidfile = False
    slot._note_server_pid(4242)
    slot._process = object()
    slot._kill_process()
    assert written == [] and cleared == [] and adopted == [4242]
    primary = LlamaCppBackend(manages_processes = False)
    primary._note_server_pid(4343)
    assert written == [4343]


def test_a_primary_swap_neither_refuses_on_nor_stops_another_models_chats(backends):
    import threading

    from fastapi import HTTPException
    from state import active_generations

    _, extra = backends
    on_slot, on_primary = threading.Event(), threading.Event()
    extra.generations.add(on_slot)
    with active_generations.ActiveGeneration(on_slot, thread_id = "slot-chat"):
        assert inf._raise_or_cancel_active_generations(force = False, action = "Loading a model") == 0
        with active_generations.ActiveGeneration(on_primary, thread_id = "primary-chat"):
            with pytest.raises(HTTPException) as excinfo:
                inf._raise_or_cancel_active_generations(force = False, action = "Loading a model")
            assert excinfo.value.detail["running"] == 1
            assert excinfo.value.detail["thread_ids"] == ["primary-chat"]
            assert (
                inf._raise_or_cancel_active_generations(force = True, action = "Loading a model") == 1
            )
    assert on_primary.is_set() and not on_slot.is_set()


def test_a_slot_refusal_names_its_own_chats(backends):
    import threading

    from fastapi import HTTPException
    from state import active_generations

    _, extra = backends
    event = threading.Event()
    extra.generations.add(event)
    with active_generations.ActiveGeneration(event, thread_id = "t1"):
        with pytest.raises(HTTPException) as excinfo:
            inf._raise_or_cancel_slot_generations(extra, force = False)
    assert excinfo.value.detail["thread_ids"] == ["t1"]


def test_reloading_a_kept_model_stops_only_its_own_chats(backends, monkeypatch):
    import threading

    from state import active_generations

    _, extra = backends
    monkeypatch.setattr(inf, "_raise_if_sidecar_swap_in_progress", lambda: None)
    monkeypatch.setattr(inf, "release_chat_gpu_claim", lambda: None)

    async def load(request, *args, on_reload_confirmed, **kwargs):
        assert inf.get_llama_cpp_backend() is extra.llama
        on_reload_confirmed(cancel = False)
        on_reload_confirmed(cancel = True)
        return "loaded"

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", load)
    reload = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0", force_reload = True)
    on_primary = threading.Event()
    with active_generations.ActiveGeneration(on_primary, thread_id = "chat-on-A"):
        assert asyncio.run(inf.load_model_gated(reload, None, "s")) == "loaded"
        on_slot = threading.Event()
        extra.generations.add(on_slot)
        forced = reload.model_copy(update = {"force_cancel_active": True})
        assert asyncio.run(inf.load_model_gated(forced, None, "s")) == "loaded"
    assert on_slot.is_set() and not on_primary.is_set()


def test_a_plain_switch_forgets_the_model_it_replaced(backends, monkeypatch):
    monkeypatch.setattr(inf, "_extra_slots", [])
    inf._note_primary_load(LoadRequest(model_path = "org/P-GGUF"))
    inf.note_chat_evicted()
    assert "org/P-GGUF" in inf._evicted
    # Kept loaded, P comes back when named; a plain switch replaced it on purpose.
    inf._note_primary_load(LoadRequest(model_path = "org/Q-GGUF"))
    assert "org/P-GGUF" not in inf._evicted


def test_training_frees_the_models_kept_alongside(backends, monkeypatch):
    from routes import training_vram

    primary, extra = backends
    extra.request = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0", alongside = True)
    primary.unload_model()
    assert training_vram.summarize_resident_chat()["any"] is True
    assert training_vram.free_chat_models_for_training("test") == ["kept:org/B-GGUF"]
    assert inf._extra_slots == [] and not extra.llama.is_active
    assert list(inf._evicted) == ["org/B-GGUF:Q8_0"]


def test_another_quant_of_a_loaded_model_replaces_it_in_place(backends, monkeypatch):
    _, extra = backends
    primary_quant = LoadRequest(model_path = "org/A-GGUF", gguf_variant = "Q8_0", alongside = True)
    assert _selected(monkeypatch, primary_quant) is None
    slot_quant = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q4_K_M", alongside = True)
    assert _selected(monkeypatch, slot_quant) is extra
    assert _selected(monkeypatch, slot_quant.model_copy(update = {"alongside": False})) is extra


def test_a_slot_built_during_a_llama_update_refuses_its_load(backends, monkeypatch):
    primary, _ = backends
    primary._llama_update_in_progress = True
    fresh = _selected(monkeypatch, LoadRequest(model_path = "org/C-GGUF", alongside = True))
    assert fresh.llama._llama_update_in_progress is True


def test_an_alongside_load_counts_as_activity(backends, monkeypatch):
    import core.inference.llama_keepwarm as keepwarm

    stamped = []
    monkeypatch.setattr(keepwarm, "_note_activity", lambda: stamped.append(True))
    _gated_load_fakes(monkeypatch, short_fits = [])
    request = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    assert asyncio.run(inf.load_model_gated(request, None, "s")) == "loaded"
    assert stamped == [True]


def test_a_chat_starting_on_a_victim_during_eviction_spares_it(backends, monkeypatch):
    import threading

    _, extra = backends
    extra.last_used = 9.0
    older = _slot("org/D-GGUF", last_used = 1.0)
    older.llama._planned_vram_mib = {0: 3000}
    mid = _slot("org/E-GGUF", last_used = 2.0)
    mid.llama._planned_vram_mib = {0: 3000}
    inf._extra_slots += [older, mid]
    unload = older.llama.unload_model

    def chat_starts_on_mid():
        mid.generations.add(threading.Event())
        unload()

    older.llama.unload_model = chat_starts_on_mid
    _gated_load_fakes(monkeypatch, short_fits = [1])
    request = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    assert asyncio.run(inf.load_model_gated(request, None, "s")) == "loaded"
    assert mid in inf._extra_slots and mid.llama.is_active
    assert list(inf._evicted) == ["org/D-GGUF"]


def test_victims_that_cannot_make_room_are_left_loaded(backends):
    _, extra = backends
    extra.llama._planned_vram_mib = {0: 3000}
    assert inf._eviction_victims(None, 10000) == []
    assert inf._eviction_victims(None, 2000) == [extra]


def test_two_accounts_stashing_one_model_keep_their_own_entries(backends, monkeypatch):
    _, extra = backends
    extra.account = "alice"
    extra.request = LoadRequest(model_path = "org/B-GGUF", alongside = True)
    bobs = inf._ExtraSlot(
        FakeLlama("org/B-GGUF"),
        FakeOrchestrator(),
        "bob",
        LoadRequest(model_path = "org/B-GGUF", alongside = True, max_seq_length = 8192),
    )
    inf._extra_slots.append(bobs)
    inf.note_chat_evicted()
    monkeypatch.setattr(inf.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(inf, "current_account_id", lambda: "alice")
    assert inf._evicted_request("org/B-GGUF") is extra.request
    monkeypatch.setattr(inf, "current_account_id", lambda: "bob")
    assert inf._evicted_request("org/B-GGUF") is bobs.request


def test_anthropic_messages_answers_with_the_model_it_restored(backends, monkeypatch):
    import auth.authentication as authentication

    primary, _ = backends
    monkeypatch.setattr(inf, "_extra_slots", [])
    inf._evicted["org/B-GGUF"] = LoadRequest(model_path = "org/B-GGUF", alongside = True)
    monkeypatch.setattr(authentication, "request_admitted_without_credential", lambda r: False)

    async def restore(request, *args, **kwargs):
        inf._extra_slots.append(
            inf._ExtraSlot(FakeLlama("org/B-GGUF"), FakeOrchestrator(), "owner")
        )

    monkeypatch.setattr(inf, "load_model_gated", restore)
    used = []

    class Named(Exception):
        pass

    real = inf._llama_public_model_id

    def spy(backend, fallback = None):
        if fallback is None:
            return real(backend)
        used.append(backend)
        raise Named()

    monkeypatch.setattr(inf, "_llama_public_model_id", spy)
    app = FastAPI()
    app.include_router(inf.router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "s"
    with TestClient(app, raise_server_exceptions = False) as client:
        client.post(
            "/v1/messages",
            json = {
                "model": "org/B-GGUF",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert used == [inf._extra_slots[0].llama] and used[0] is not primary


def test_a_routed_request_holds_its_slot_until_it_ends(backends):
    import core.inference.llama_keepwarm as keepwarm

    _, extra = backends
    scope = {}

    async def request():
        keepwarm.set_current_response_scope(scope)
        await inf._route_to_extra_slot("org/B-GGUF")

    asyncio.run(request())
    assert extra.refs == 1
    assert inf._eviction_victims(None, 5000) == [] and not inf._claim_victim(extra)
    keepwarm._run_end_callbacks(scope)
    assert extra.refs == 0 and inf._eviction_victims(None, 5000) == [extra]
    asyncio.run(request())
    assert extra.refs == 0


def test_a_slot_evicted_while_a_request_routes_is_not_served(backends, monkeypatch):
    _, extra = backends
    probe = inf._slot_serving

    def evicted_mid_probe(requested, slots):
        found = probe(requested, slots)
        assert inf._claim_victim(extra)
        return found

    monkeypatch.setattr(inf, "_slot_serving", evicted_mid_probe)
    assert _routed("org/B-GGUF") == (None, backends[0])


def test_a_slot_on_another_gpu_is_no_victim(backends):
    _, extra = backends
    extra.llama._planned_vram_mib = {1: 9000}
    other = _slot("org/D-GGUF", last_used = 5.0)
    other.llama._planned_vram_mib = {0: 6000}
    inf._extra_slots.append(other)
    assert inf._eviction_victims(None, 5000, (0,)) == [other]
    assert inf._eviction_victims(None, 5000) == [extra]
