# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Models loaded alongside the primary one are routed to by request model name."""

import asyncio
import inspect
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import core.inference.llama_cpp as llama_cpp
import core.inference.llama_keepwarm as keepwarm
import core.inference.model_slots as model_slots
import core.inference.npu_backend as npu_backend
import core.inference.orchestrator as orchestrator
import routes.inference as inf
from auth import policy
from auth.authentication import get_current_subject
from core.inference import gpu_arbiter
from core.inference.llama_cpp import GpuMemoryShortError
from hub.services.models import deletion
from models.inference import (
    ChatCountTokensRequest,
    InferenceStatusResponse,
    LoadRequest,
    LoadResponse,
    UnloadRequest,
)
from routes import training_vram
from state import active_generations
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
        self.is_vision = False
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
    extra = model_slots.ExtraSlot(FakeLlama("org/B-GGUF", "Q8_0"), FakeOrchestrator(), "owner")
    monkeypatch.setattr(inf, "_llama_cpp_backend", primary)
    monkeypatch.setattr(orchestrator, "_inference_backend", FakeOrchestrator())
    monkeypatch.setattr(model_slots, "slots", [extra])
    monkeypatch.setattr(inf, "LlamaCppBackend", FakeLlama)
    monkeypatch.setattr(inf, "InferenceOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(inf, "_raise_if_sidecar_swap_in_progress", lambda: None)
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
    assert extra.last_used > 0.0
    # The primary wins a bare name both serve.
    model_slots.slots.append(
        model_slots.ExtraSlot(FakeLlama("org/A-GGUF", "Q8_0"), FakeOrchestrator(), "owner")
    )
    assert _routed("org/A-GGUF") == (None, primary)
    assert _routed("org/A-GGUF:Q8_0")[0] is model_slots.slots[-1]


def test_a_safetensors_slot_is_served_by_its_own_orchestrator(backends):
    safetensors = model_slots.ExtraSlot(FakeLlama(), FakeOrchestrator("org/C"), "owner")
    model_slots.slots.append(safetensors)

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
    model_slots.slots.append(model_slots.ExtraSlot(FakeLlama(), FakeOrchestrator("org/C"), "owner"))
    with TestClient(app) as client:
        data = client.get("/api/inference/loaded-models").json()["data"]
    assert [entry["id"] for entry in data] == ["org/A-GGUF", "org/B-GGUF", "org/C"]


def _selected(request):
    async def run():
        slot = await inf._select_load_slot(request)
        assert inf.routed_slot.get() is slot
        return slot

    return asyncio.run(run())


def test_a_load_picks_its_slot(backends):
    primary, extra = backends
    alongside = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    assert _selected(alongside) not in (None, extra)
    assert _selected(LoadRequest(model_path = "org/C-GGUF")) is None
    served = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0")
    assert _selected(served) is extra
    # The NPU backend is one per process: it takes the primary's seat even alongside.
    npu = LoadRequest(model_path = "lemonade:qwen3-0.6b-FLM", alongside = True)
    assert _selected(npu) is None
    primary.unload_model()
    assert _selected(alongside) is None


def test_unload_drops_only_the_named_extra_slot(backends, monkeypatch):
    primary, extra = backends
    _hold_chat_claim(monkeypatch)
    released, release = [], inf.release_chat_gpu_claim
    monkeypatch.setattr(inf, "release_chat_gpu_claim", lambda: released.append(release()))
    response = asyncio.run(inf._unload_model_impl(UnloadRequest(model_path = "org/B-GGUF"), "s"))
    assert response.status == "unloaded"
    assert primary.is_loaded and not extra.llama.is_loaded and model_slots.slots == []
    # Tried, but the claim the primary holds stays.
    assert released == [False] and gpu_arbiter.current_owner() == gpu_arbiter.CHAT


def test_a_failed_llama_unload_still_cleans_the_orchestrator(backends):
    _, extra = backends
    cleaned = []
    extra.llama.unload_model = lambda: 1 / 0
    extra.orchestrator._cleanup = lambda: cleaned.append(True)
    with pytest.raises(ZeroDivisionError):
        model_slots.drop(extra)
    assert model_slots.slots == [] and cleaned == [True]


def test_a_managed_account_sees_only_its_own_slots(backends, monkeypatch):
    primary, extra = backends
    monkeypatch.setattr(inf.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(model_slots, "current_account_id", lambda: "someone-else")
    assert _routed("org/B-GGUF") == (None, primary)
    filling = model_slots.ExtraSlot(FakeLlama(), FakeOrchestrator(), "owner")
    monkeypatch.setattr(model_slots, "loading", (filling, "org/D-GGUF"))
    model_slots.slots.append(filling)
    assert model_slots.visible_loading() is None
    monkeypatch.setattr(model_slots, "current_account_id", lambda: "owner")
    assert _routed("org/B-GGUF") == (extra, extra.llama)


def test_idle_unload_spares_pinned_and_filling_slots(backends, monkeypatch):
    _, extra = backends
    assert model_slots.unload_extra_models(keep = lambda llama: True) == 0
    monkeypatch.setattr(model_slots, "loading", (extra, "org/B-GGUF"))
    assert model_slots.unload_extra_models(keep = lambda llama: False, spare_filling = True) == 0
    assert model_slots.slots == [extra] and extra.llama.is_active
    assert model_slots.unload_extra_models(keep = lambda llama: False) == 1
    assert model_slots.slots == []


def test_status_describes_the_named_slot_and_lists_the_rest(backends, monkeypatch):
    async def slot_status(subject):
        return InferenceStatusResponse(active_model = inf.get_llama_cpp_backend().model_identifier)

    monkeypatch.setattr(inf, "_slot_status", slot_status)
    named = asyncio.run(inf.get_status("s", model = "org/B-GGUF"))
    assert (named.active_model, named.loaded) == ("org/B-GGUF", ["org/A-GGUF"])
    primary = asyncio.run(inf.get_status("s"))
    assert (primary.active_model, primary.loaded) == ("org/A-GGUF", ["org/B-GGUF"])

    async def held_behind(subject):
        return InferenceStatusResponse(
            active_model = "org/A-GGUF", loaded = ["org/A-GGUF", "org/held-hf"]
        )

    # serving leaves out a model only held in memory behind the active one.
    monkeypatch.setattr(inf, "_slot_status", held_behind)
    status = asyncio.run(inf.get_status("s"))
    assert status.serving == ["org/A-GGUF", "org/B-GGUF"] and "org/held-hf" in status.loaded
    monkeypatch.setattr(model_slots, "slots", [])
    assert asyncio.run(inf.get_status("s")).serving == ["org/A-GGUF"]


def test_stop_loading_reaches_the_slot_being_filled(backends, monkeypatch):
    primary, _ = backends
    filling = model_slots.ExtraSlot(FakeLlama(), FakeOrchestrator(), "owner")
    filling.llama.model_identifier = "org/D-GGUF"
    filling.llama.is_active = True
    model_slots.slots.append(filling)
    monkeypatch.setattr(model_slots, "loading", (filling, "org/D-GGUF"))
    response = asyncio.run(inf._unload_model_impl(UnloadRequest(model_path = "org/D-GGUF"), "s"))
    assert response.status == "unloaded"
    assert primary.is_loaded and not filling.llama.is_active


def test_a_slot_evicted_while_it_loads_is_torn_down(backends, monkeypatch):
    spawned = []

    async def load_after_eviction(request, *args, **kwargs):
        slot = model_slots.slots[-1]
        model_slots.unload_extra_models()
        slot.llama.is_loaded = slot.llama.is_active = True
        spawned.append(slot)
        return "loaded"

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", load_after_eviction)
    monkeypatch.setattr(inf, "release_chat_gpu_claim", lambda: None)
    request = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    assert asyncio.run(inf.load_model_gated(request, None, "s")) == "loaded"
    assert model_slots.slots == [] and not spawned[0].llama.is_active


def test_only_a_new_slot_skips_the_running_chat_check(backends, monkeypatch):
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


def _slot(
    model,
    variant = None,
    last_used = 0.0,
):
    return model_slots.ExtraSlot(
        FakeLlama(model, variant),
        FakeOrchestrator(),
        "owner",
        LoadRequest(model_path = model, gguf_variant = variant, alongside = True),
        last_used,
    )


def _gated_load_fakes(
    monkeypatch,
    short_fits,
    response = "loaded",
):
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
        return response

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", load)


def test_a_short_fit_evicts_the_least_recently_used_slot_and_names_it(backends, monkeypatch):
    _, extra = backends
    extra.request, extra.last_used = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0"), 5.0
    older = _slot("org/D-GGUF", "Q4_K_M", last_used = 1.0)
    model_slots.slots.append(older)
    loaded = LoadResponse.model_construct(status = "loaded", model = "org/C-GGUF", display_name = "C")
    _gated_load_fakes(monkeypatch, short_fits = [1], response = loaded)
    request = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    assert asyncio.run(inf.load_model_gated(request, None, "s")).evicted == ["org/D-GGUF:Q4_K_M"]
    assert [s.llama.model_identifier for s in model_slots.slots] == ["org/B-GGUF", "org/C-GGUF"]
    assert not older.llama.is_active
    assert model_slots.slots[-1].request.model_path == "org/C-GGUF"


def test_eviction_takes_as_many_lru_slots_as_the_shortfall_needs(backends, monkeypatch):
    _, extra = backends
    extra.request, extra.last_used = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0"), 3.0
    extra.llama._planned_vram_mib = {0: 9000}
    small_old = _slot("org/D-GGUF", last_used = 1.0)
    small_old.llama._planned_vram_mib = {0: 500}
    mid = _slot("org/E-GGUF", last_used = 2.0)
    mid.llama._planned_vram_mib = {0: 6000}
    model_slots.slots += [small_old, mid]
    assert model_slots.eviction_victims(None, 5000) == [small_old, mid]
    _gated_load_fakes(monkeypatch, short_fits = [1])
    request = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    assert asyncio.run(inf.load_model_gated(request, None, "s")) == "loaded"
    assert [s.llama.model_identifier for s in model_slots.slots] == ["org/B-GGUF", "org/C-GGUF"]


def test_a_short_fit_with_nothing_left_to_evict_is_a_409(backends, monkeypatch):
    _, extra = backends
    _gated_load_fakes(monkeypatch, short_fits = [1, 1])
    request = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inf.load_model_gated(request, None, "s"))
    assert excinfo.value.status_code == 409 and "8 GB free" in excinfo.value.detail
    assert model_slots.slots == [] and not extra.llama.is_active


def test_a_capped_context_is_taken_only_once_nothing_is_left_to_evict(backends, monkeypatch):
    _, extra = backends
    extra.request = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0")
    _gated_load_fakes(monkeypatch, short_fits = ["capped", "capped"])
    request = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    assert asyncio.run(inf.load_model_gated(request, None, "s")) == "loaded"
    assert not extra.llama.is_active
    assert [s.llama.model_identifier for s in model_slots.slots] == ["org/C-GGUF"]


def test_a_slot_load_drops_no_claim_and_a_load_that_tears_nothing_down_stops_no_chat():
    source = inspect.getsource(inf._load_model_impl)
    assert source.count("if replacing and not chat_load_needs_gpu:") == 2
    assert "if not chat_load_needs_gpu:" not in source
    assert source.count("if serving and on_reload_confirmed is not None:") == 3
    assert source.count("if replacing and serving:") == 2
    assert "if on_reload_confirmed is not None:" not in source


def test_a_load_prices_the_vram_the_other_servers_hold():
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
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    monkeypatch.setattr(llama_cpp, "chat_load_active", lambda: False)


def test_a_failed_slot_load_drops_the_slot_but_keeps_the_primarys_claim(backends, monkeypatch):
    primary, extra = backends
    _hold_chat_claim(monkeypatch)
    released = []
    release = inf.release_chat_gpu_claim
    monkeypatch.setattr(inf, "release_chat_gpu_claim", lambda: released.append(release()))

    async def failing_load(*args, **kwargs):
        raise RuntimeError("no such repo")

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", failing_load)
    request = LoadRequest(model_path = "org/missing-GGUF", alongside = True)
    with pytest.raises(RuntimeError):
        asyncio.run(inf.load_model_gated(request, None, "s"))
    assert model_slots.slots == [extra] and model_slots.loading is None and released == [False]
    assert primary.is_active and gpu_arbiter.current_owner() == gpu_arbiter.CHAT


def test_a_generation_is_tracked_on_the_slot_serving_it(backends):
    _, extra = backends
    event = threading.Event()

    async def run():
        await inf._route_to_extra_slot("org/B-GGUF")
        with inf._TrackedCancel(event, "k"):
            inside = set(extra.generations)
        return inside

    assert asyncio.run(run()) == {event} and extra.generations == set()


def test_a_slot_still_generating_is_never_evicted(backends):
    _, extra = backends
    older = _slot("org/D-GGUF", last_used = 1.0)
    model_slots.slots.append(older)
    older.generations.add(threading.Event())
    assert model_slots.eviction_victims(None, 5000) == [extra]
    extra.generations.add(threading.Event())
    assert model_slots.eviction_victims(None, 5000) == []


def test_unloading_a_generating_slot_is_refused_unless_forced(backends, monkeypatch):
    _, extra = backends
    monkeypatch.setattr(inf, "release_chat_gpu_claim", lambda: True)
    event = threading.Event()
    extra.generations.add(event)
    with (
        active_generations.ActiveGeneration(event, thread_id = "t1"),
        pytest.raises(HTTPException) as excinfo,
    ):
        asyncio.run(inf._unload_model_impl(UnloadRequest(model_path = "org/B-GGUF"), "s"))
    assert excinfo.value.status_code == 409 and excinfo.value.detail["thread_ids"] == ["t1"]
    assert extra.llama.is_loaded and not event.is_set()

    def finish_on_cancel():
        event.wait(5)
        extra.generations.discard(event)

    threading.Thread(target = finish_on_cancel, daemon = True).start()
    forced = UnloadRequest(model_path = "org/B-GGUF", force_cancel_active = True)
    asyncio.run(inf._unload_model_impl(forced, "s"))
    assert event.is_set() and not extra.llama.is_loaded and model_slots.slots == []


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


def test_reloading_a_kept_model_stops_only_its_own_chats(backends, monkeypatch):
    _, extra = backends
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


def test_training_sizes_and_frees_the_models_kept_alongside(backends, monkeypatch):
    primary, extra = backends
    primary.unload_model()
    # A loaded one is in the free VRAM training reads; only a still-loading one is unsizable.
    summary = training_vram.summarize_resident_chat()
    assert summary["any"] and not summary["loading"]
    monkeypatch.setattr(model_slots, "loading", (extra, "org/D-GGUF"))
    assert training_vram.summarize_resident_chat()["loading"]
    monkeypatch.setattr(model_slots, "loading", None)
    extra.orchestrator.loading_models = {"org/C"}
    assert training_vram.summarize_resident_chat()["loading"]
    extra.orchestrator.loading_models = set()
    assert training_vram.free_chat_models_for_training("test") == ["kept:org/B-GGUF"]
    assert model_slots.slots == [] and not extra.llama.is_active


def test_another_quant_of_a_loaded_model_replaces_it_in_place(backends):
    _, extra = backends
    primary_quant = LoadRequest(model_path = "org/A-GGUF", gguf_variant = "Q8_0", alongside = True)
    assert _selected(primary_quant) is None
    slot_quant = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q4_K_M", alongside = True)
    assert _selected(slot_quant) is extra
    assert _selected(slot_quant.model_copy(update = {"alongside": False})) is extra


def test_an_alongside_load_counts_as_activity(backends, monkeypatch):
    stamped = []
    monkeypatch.setattr(keepwarm, "_note_activity", lambda: stamped.append(True))
    _gated_load_fakes(monkeypatch, short_fits = [])
    request = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    assert asyncio.run(inf.load_model_gated(request, None, "s")) == "loaded"
    assert stamped == [True]


def test_a_chat_starting_on_a_victim_during_eviction_spares_it(backends, monkeypatch):
    _, extra = backends
    extra.last_used = 9.0
    older = _slot("org/D-GGUF", last_used = 1.0)
    older.llama._planned_vram_mib = {0: 3000}
    mid = _slot("org/E-GGUF", last_used = 2.0)
    mid.llama._planned_vram_mib = {0: 3000}
    model_slots.slots += [older, mid]
    unload = older.llama.unload_model

    def chat_starts_on_mid():
        mid.generations.add(threading.Event())
        unload()

    older.llama.unload_model = chat_starts_on_mid
    _gated_load_fakes(monkeypatch, short_fits = [1])
    request = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    assert asyncio.run(inf.load_model_gated(request, None, "s")) == "loaded"
    assert mid in model_slots.slots and mid.llama.is_active


def test_a_routed_request_holds_its_slot_until_it_ends(backends):
    _, extra = backends
    ended = asyncio.Event()
    seen = {}

    async def request():
        await inf._route_to_extra_slot("org/B-GGUF")
        await ended.wait()

    async def main():
        task = asyncio.create_task(request())
        await asyncio.sleep(0)
        seen["refs"] = extra.refs
        seen["victims"] = model_slots.eviction_victims(None, 5000)
        seen["claimed"] = model_slots.claim_victim(extra)
        ended.set()
        await task

    asyncio.run(main())
    assert seen == {"refs": 1, "victims": [], "claimed": False}
    assert extra.refs == 0 and model_slots.eviction_victims(None, 5000) == [extra]


def test_a_chat_run_holds_its_slot_after_its_post_has_answered(backends):
    # A durable chat run's task starts inside POST /chat-runs, which answers 202 before the run routes
    # and preprocesses; the slot must stay held through that window, not end with the POST.

    _, extra = backends
    seen = {}

    async def run():
        await asyncio.sleep(0)
        await inf._route_to_extra_slot("org/B-GGUF")
        await asyncio.sleep(0)
        seen["refs"] = extra.refs
        seen["victims"] = model_slots.eviction_victims(None, 5000)

    async def post_chat_run():
        keepwarm.set_current_response_scope({})
        task = asyncio.create_task(run())
        keepwarm.set_current_response_scope(None)
        await task

    asyncio.run(post_chat_run())
    assert seen == {"refs": 1, "victims": []}
    assert extra.refs == 0


def test_a_slot_evicted_while_a_request_routes_is_not_served(backends, monkeypatch):
    _, extra = backends
    probe = model_slots.serving_slot

    def evicted_mid_probe(requested, slots, satisfies):
        found = probe(requested, slots, satisfies)
        assert model_slots.claim_victim(extra)
        return found

    monkeypatch.setattr(model_slots, "serving_slot", evicted_mid_probe)
    assert _routed("org/B-GGUF") == (None, backends[0])


def test_eviction_skips_other_gpus_and_victims_that_cannot_make_room(backends):
    _, extra = backends
    extra.llama._planned_vram_mib = {1: 9000}
    other = _slot("org/D-GGUF", last_used = 5.0)
    other.llama._planned_vram_mib = {0: 6000}
    model_slots.slots.append(other)
    assert model_slots.eviction_victims(None, 5000, (0,)) == [other]
    assert model_slots.eviction_victims(None, 5000) == [extra]
    # Victims that together cannot make room are left loaded.
    assert model_slots.eviction_victims(None, 20000) == []


def test_a_token_count_waits_only_on_its_own_models_chats(backends):
    _, extra = backends

    async def count_on(model):
        await inf._route_to_extra_slot(model)
        return model_slots.routed_generation_count()

    on_primary = threading.Event()
    with active_generations.ActiveGeneration(on_primary, thread_id = "chat-on-A"):
        assert asyncio.run(count_on("org/B-GGUF")) == 0
        assert asyncio.run(count_on("org/A-GGUF")) == 1
    on_slot = threading.Event()
    with active_generations.ActiveGeneration(on_slot, thread_id = "chat-on-B"):
        extra.generations.add(on_slot)
        assert asyncio.run(count_on("org/B-GGUF")) == 1
        assert asyncio.run(count_on("org/A-GGUF")) == 0


def test_counting_for_an_idle_kept_model_ignores_the_primarys_chat(backends):
    payload = ChatCountTokensRequest(
        model = "org/B-GGUF",
        messages = [{"role": "user", "content": "hi"}],
    )
    with active_generations.ActiveGeneration(threading.Event(), thread_id = "chat-on-A"):
        try:
            asyncio.run(inf.chat_count_tokens(payload, "s", None))
        except HTTPException as exc:
            assert exc.detail != "Cannot count tokens while a generation is in progress."
        except Exception:
            pass


def test_a_model_still_loading_alongside_keeps_the_chat_claim(backends, monkeypatch):
    primary, extra = backends
    _hold_chat_claim(monkeypatch)
    primary.unload_model()
    extra.llama.unload_model()
    extra.orchestrator.loading_models = {"org/C"}
    inf.release_chat_gpu_claim()
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT
    extra.orchestrator.loading_models = set()
    monkeypatch.setattr(model_slots, "loading", (extra, "org/C"))
    inf.release_chat_gpu_claim()
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT
    monkeypatch.setattr(model_slots, "loading", None)
    inf.release_chat_gpu_claim()
    assert gpu_arbiter.current_owner() != gpu_arbiter.CHAT


def test_a_llama_update_stops_every_llama_slot_and_only_those(backends, monkeypatch):
    _, extra = backends
    safetensors = model_slots.ExtraSlot(FakeLlama(), FakeOrchestrator("org/C"), "owner")
    starting = model_slots.ExtraSlot(FakeLlama(), FakeOrchestrator(), "owner")
    model_slots.slots += [safetensors, starting]
    monkeypatch.setattr(model_slots, "loading", (starting, "org/D-GGUF"))
    assert model_slots.unload_llama_slots() == 2
    assert model_slots.slots == [safetensors]
    loading = model_slots.ExtraSlot(FakeLlama(), FakeOrchestrator(), "owner")
    loading.orchestrator.loading_models = {"org/E"}
    model_slots.slots.append(loading)
    monkeypatch.setattr(model_slots, "loading", (loading, "org/E"))
    assert model_slots.unload_llama_slots() == 0


def test_deleting_a_model_another_slot_serves_or_fills_is_refused(backends, monkeypatch):
    _, extra = backends
    assert deletion._llama_cpp_blocks_delete("org/B-GGUF", None)
    assert deletion._llama_cpp_blocks_delete("org/B-GGUF", "Q8_0")
    assert not deletion._llama_cpp_blocks_delete("org/B-GGUF", "Q4_K_M")
    assert not deletion._llama_cpp_blocks_delete("org/Z-GGUF", None)
    model_slots.slots.append(model_slots.ExtraSlot(FakeLlama(), FakeOrchestrator("org/C"), "owner"))
    assert deletion._inference_backend_blocks_delete("org/C")
    assert not deletion._inference_backend_blocks_delete("org/Z")
    starting = model_slots.ExtraSlot(FakeLlama(), FakeOrchestrator(), "owner")
    model_slots.slots.append(starting)
    monkeypatch.setattr(model_slots, "loading", (starting, "org/D-GGUF"))
    assert deletion._llama_cpp_blocks_delete("org/D-GGUF", "Q4_K_M")
    monkeypatch.setattr(model_slots, "loading", None)
    assert not deletion._llama_cpp_blocks_delete("org/D-GGUF", "Q4_K_M")
    starting.orchestrator.loading_models = {"org/E"}
    assert deletion._inference_backend_blocks_delete("org/E")


def test_clearing_the_cache_waits_for_models_kept_alongside(backends, monkeypatch):
    primary, extra = backends
    monkeypatch.setattr(llama_cpp, "chat_load_active", lambda: False)
    primary.unload_model()
    assert deletion.any_model_load_blocks_cache_clear() == (
        "Unload the model before clearing the model cache"
    )
    extra.llama.unload_model()
    monkeypatch.setattr(model_slots, "loading", (extra, "org/B-GGUF"))
    assert "load" in deletion.any_model_load_blocks_cache_clear()


def test_the_model_list_marks_kept_models_loaded(backends, monkeypatch):
    import routes.models as models_routes

    _, extra = backends
    fake = FakeOrchestrator()
    fake.default_models = []
    monkeypatch.setattr(models_routes, "get_inference_backend", lambda: fake)
    model_slots.slots.append(model_slots.ExtraSlot(FakeLlama(), FakeOrchestrator("org/C"), "owner"))
    app = FastAPI()
    app.include_router(models_routes.router, prefix = "/api/models")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    with TestClient(app) as client:
        ids = {m["id"] for m in client.get("/api/models/list").json()["models"]}
    assert {"org/A-GGUF", "org/B-GGUF", "org/C"} <= ids


def test_a_model_that_does_not_fit_says_so_without_api_flags():
    from core.inference.llama_cpp import _gpu_short_message

    spill = _gpu_short_message(13.2, 8.4, 32768, 32768, False)
    assert "13.2 GB" in spill and "8.4 GB" in spill and "Keep other models loaded" in spill
    capped = _gpu_short_message(13.2, 8.4, 32768, 8192, True)
    assert "8192 context" in capped
    assert "force_alongside" not in spill + capped


def test_a_slot_load_during_a_llama_update_is_refused(backends, monkeypatch):
    primary, extra = backends
    _gated_load_fakes(monkeypatch, short_fits = [])
    seen = []
    load = inf._run_tracked_load_model_impl

    async def noting_update(request, *args, **kwargs):
        seen.append(inf.get_llama_cpp_backend()._llama_update_in_progress)
        return await load(request, *args, **kwargs)

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", noting_update)
    reload = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0", force_reload = True)
    primary._llama_update_in_progress = True
    for request in (LoadRequest(model_path = "org/C-GGUF", alongside = True), reload):
        asyncio.run(inf.load_model_gated(request, None, "s"))
    primary._llama_update_in_progress = False
    asyncio.run(inf.load_model_gated(reload, None, "s"))
    assert seen == [True, True, False]


def test_a_resident_npu_model_keeps_to_the_primarys_seat(backends, monkeypatch):
    primary, extra = backends
    primary.unload_model()
    npu_model = SimpleNamespace(model_path = "lemonade:qwen3-0.6b-FLM", id = "qwen3-0.6b-FLM")
    npu = SimpleNamespace(is_loaded = True, loaded_model = npu_model, resident = lambda: None)
    monkeypatch.setattr(npu_backend, "peek_npu_backend", lambda: npu)

    async def route(model):
        slot = await inf._route_to_extra_slot(model)
        return slot, inf._resident_npu_model(), inf._loaded_slot_ident()

    assert asyncio.run(route("org/B-GGUF")) == (extra, None, "org/B-GGUF")
    assert asyncio.run(route("qwen3-0.6b-FLM")) == (None, npu_model, npu_model.model_path)


def test_the_loaded_models_list_names_the_npu_model_once(backends, monkeypatch):
    primary, _ = backends
    primary.unload_model()
    model_slots.slots.append(
        model_slots.ExtraSlot(FakeLlama("org/D-GGUF"), FakeOrchestrator(), "owner")
    )
    npu_model = SimpleNamespace(
        model_path = "lemonade:qwen3-0.6b-FLM",
        id = "qwen3-0.6b-FLM",
        vision = False,
        max_context_length = 4096,
        reasoning = False,
        tools = False,
    )
    resident = SimpleNamespace(model = npu_model, context_length = 4096)
    npu = SimpleNamespace(is_loaded = True, loaded_model = npu_model, resident = lambda: resident)
    monkeypatch.setattr(npu_backend, "peek_npu_backend", lambda: npu)
    app = FastAPI()
    app.include_router(inf.studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    with TestClient(app) as client:
        ids = [entry["id"] for entry in client.get("/api/inference/loaded-models").json()["data"]]
    assert ids.count("lemonade:qwen3-0.6b-FLM") == 1 and "org/B-GGUF" in ids and "org/D-GGUF" in ids


def test_a_load_over_a_streaming_npu_model_asks_before_stopping_it(monkeypatch, tmp_path):
    import struct

    npu = SimpleNamespace(
        is_loaded = True,
        loaded_model = SimpleNamespace(model_path = "lemonade:qwen3-0.6b-FLM", id = "qwen3-0.6b-FLM"),
        resident = lambda: None,
        unload = lambda: None,
    )
    monkeypatch.setattr(orchestrator, "_inference_backend", FakeOrchestrator())
    monkeypatch.setattr(npu_backend, "peek_npu_backend", lambda: npu)
    monkeypatch.setattr(inf, "_raise_if_sidecar_swap_in_progress", lambda: None)
    torn_down = []

    async def _teardown():
        torn_down.append(True)
        raise RuntimeError("NPU unloaded while its chat is still streaming")

    monkeypatch.setattr(inf, "_unload_npu_before_local_load", _teardown)
    asked = []

    def _gate(*, cancel):
        asked.append(cancel)
        if active_generations.count() and not cancel:
            raise LookupError("refused")
        return 0

    def _s(x):
        return struct.pack("<Q", len(x)) + x.encode()

    gguf = tmp_path / "tiny.gguf"
    gguf.write_bytes(
        b"GGUF"
        + struct.pack("<IQQ", 3, 0, 1)
        + _s("general.architecture")
        + struct.pack("<I", 8)
        + _s("llama")
    )

    async def _load():
        with active_generations.ActiveGeneration(
            threading.Event(), thread_id = "t1", run_id = "r1", model = "npu", kind = "chat"
        ):
            with pytest.raises(Exception) as exc:
                await inf._load_model_impl(
                    LoadRequest(model_path = str(gguf)), None, "s", on_reload_confirmed = _gate
                )
        return exc

    exc = asyncio.run(_load())
    assert asked == [False] and not torn_down, exc.value


def test_active_generations_for_a_model_lists_only_its_chats(backends):
    _, extra = backends
    app = FastAPI()
    app.include_router(inf.studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    on_a, on_b = threading.Event(), threading.Event()
    extra.generations.add(on_b)
    with (
        active_generations.ActiveGeneration(on_a, thread_id = "chat-on-A"),
        active_generations.ActiveGeneration(on_b, thread_id = "chat-on-B"),
        TestClient(app) as client,
    ):
        get = lambda q: client.get("/api/inference/active-generations" + q).json()["thread_ids"]
        assert get("?model=org/B-GGUF") == ["chat-on-B"]
        assert get("?model=org/A-GGUF") == ["chat-on-A"]
        assert sorted(get("")) == ["chat-on-A", "chat-on-B"]


def test_an_integrated_gpu_keeps_its_free_memory_next_to_a_loaded_model():
    from core.inference.llama_cpp import _net_of_held_vram

    held = {0: 900}
    # Discrete: capped at total less what the other model planned.
    assert _net_of_held_vram([(0, 20_000, 24_000)], held) == [(0, 20_000, 24_000)]
    assert _net_of_held_vram([(0, 23_500, 24_000)], held) == [(0, 23_100, 24_000)]
    # Integrated (total 0, shared RAM): the free reading is left alone, not zeroed.
    assert _net_of_held_vram([(0, 60_000, 0)], held) == [(0, 60_000, 0)]
    assert _net_of_held_vram([(0, 60_000, 0)], {}) == [(0, 60_000, 0)]


def test_a_public_preview_never_reaches_a_model_kept_alongside(backends):
    primary, extra = backends
    preview = SimpleNamespace(scope = {}, headers = {}, state = SimpleNamespace())
    inf.disable_openai_auto_switch_for_request(preview.scope)

    async def chat(model):
        await inf._maybe_auto_switch_model(model, preview, "s")
        return inf.get_llama_cpp_backend()

    assert asyncio.run(chat("org/B-GGUF")) is primary


def test_a_transformers_install_stops_the_workers_of_models_kept_alongside(backends, monkeypatch):
    import core.export as export
    import core.training as training
    import utils.transformers_latest as transformers_latest
    from models.inference import InstallLatestTransformersRequest

    kept = model_slots.ExtraSlot(FakeLlama(), FakeOrchestrator("org/C"), "owner")
    kept.orchestrator.is_worker_alive = lambda: kept.orchestrator.active_model_name is not None
    kept.orchestrator._cleanup = lambda: setattr(kept.orchestrator, "active_model_name", None)
    model_slots.slots.append(kept)
    idle = SimpleNamespace(
        is_training_active = lambda: False,
        is_export_active = lambda: False,
        current_checkpoint = None,
        cleanup_memory = lambda: None,
        is_worker_alive = lambda: False,
    )
    monkeypatch.setattr(training, "get_training_backend", lambda: idle)
    monkeypatch.setattr(export, "get_export_backend", lambda: idle)
    monkeypatch.setattr(keepwarm, "other_inference_request_count", lambda **kwargs: 0)
    alive_at_swap = []

    def install(version, before_swap, *args):
        before_swap()
        alive_at_swap.append(kept.orchestrator.is_worker_alive())
        return {"success": False, "message": "stop here"}

    monkeypatch.setattr(transformers_latest, "install_latest_transformers", install)
    with pytest.raises(HTTPException):
        asyncio.run(
            inf.install_latest_transformers_route(
                InstallLatestTransformersRequest(version = "9.9.9"), "s"
            )
        )
    assert alive_at_swap == [False] and kept not in model_slots.slots


def test_a_zero_vram_primary_keeps_the_chat_claim_a_kept_model_holds(backends, monkeypatch):
    _, extra = backends
    _hold_chat_claim(monkeypatch)
    source = inspect.getsource(inf._load_model_impl)
    assert "release, CHAT)" not in source and "release(CHAT)" not in source
    assert source.count("_release_chat_for_zero_vram_primary") == 3
    inf._release_chat_for_zero_vram_primary()
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT
    model_slots.drop(extra)
    inf._release_chat_for_zero_vram_primary()
    assert gpu_arbiter.current_owner() is None
