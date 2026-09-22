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
    def __init__(self, identifier = None, variant = None):
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
    inf._extra_slots.append(inf._ExtraSlot(FakeLlama("org/A-GGUF", "Q8_0"), FakeOrchestrator(), "owner"))
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


def _slot(model, variant = None, last_used = 0.0):
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
    inf._evicted["org/B-GGUF:Q8_0"] = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0", alongside = True)
    monkeypatch.setattr(authentication, "request_admitted_without_credential", lambda r: False)
    restored = []

    async def gated(request, *args, **kwargs):
        restored.append((request.model_path, request.alongside, kwargs.get("current_request_counted")))
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
    monkeypatch.setattr(keepwarm, "restore_kv_resume", lambda backend, kv: restored.append((backend, kv)))
    monkeypatch.setattr(keepwarm, "_delete_resume_files", lambda kv: deleted.append(kv))
    _gated_load_fakes(monkeypatch, short_fits = [])
    request = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0", alongside = True)
    asyncio.run(inf.load_model_gated(request, None, "s"))
    assert len(restored) == 1 and restored[0][1] is manifest and restored[0][0] is inf._extra_slots[-1].llama
    assert inf._evicted_kv == {} and deleted == []

    inf._evicted_kv["org/C-GGUF"] = manifest
    inf._evicted["org/C-GGUF"] = LoadRequest(model_path = "org/C-GGUF")
    inf._forget_evicted("org/C-GGUF")
    assert deleted == [manifest] and inf._evicted_kv == {}


def test_a_load_prices_the_vram_the_other_servers_hold():
    from core.inference.llama_cpp import LlamaCppBackend

    a, b, c = (LlamaCppBackend(manages_processes = False) for _ in range(3))
    from core.inference import llama_cpp
    llama_cpp._live_backends.update((a, b, c))
    a._planned_vram_mib = {0: 12000}
    b._planned_vram_mib = {0: 1000, 1: 500}
    assert c._other_planned_vram_mib() == {0: 13000, 1: 500}
    assert a._other_planned_vram_mib() == {0: 1000, 1: 500}
    a._kill_process()
    assert a._planned_vram_mib == {} and c._other_planned_vram_mib() == {0: 1000, 1: 500}
