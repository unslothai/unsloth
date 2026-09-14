# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Multiple resident GGUF models: registry, request routing, load/unload seams.

No GPU or llama-server: backends are doubles carrying the attribute surface the
identity checks read, mirroring tests/test_openai_auto_switch.py. The module
registry is swapped per test, so no slot ever leaks into another test's
``get_llama_cpp_backend()``.
"""

import asyncio
import inspect
import threading
from types import SimpleNamespace


import pytest
from fastapi import HTTPException

import routes.inference as inference_route
from core.inference.resident_models import (
    ResidentCapacityError,
    ResidentLlamaRegistry,
    max_resident_slots,
    residency_enabled,
)
from models.inference import LoadRequest


class _ResidentDouble:
    """The attributes resident routing reads off a loaded GGUF backend."""

    def __init__(
        self,
        identifier,
        *,
        hf_variant = None,
        advertised_id = None,
        context_length = None,
        is_embedding_gguf = False,
    ):
        self.model_identifier = identifier
        self.is_loaded = True
        self.is_active = True
        self.hf_variant = hf_variant
        self._openai_advertised_id = advertised_id
        self._openai_gguf_companion_roots = ()
        self._openai_gguf_companion_state = ()
        self.context_length = context_length
        self.is_embedding_gguf = is_embedding_gguf
        self.unloads = 0
        self.loads = 0
        self._audio_probed = True
        self._audio_type = None
        self._is_audio = False
        self.holds_no_vram = False
        self.adopt_matches = True

    def adopt_load_intent_if_matched(self, _intent):
        return self.adopt_matches

    def matches_load_source(self, intent):
        requested = getattr(intent, "model_identifier", None)
        return isinstance(requested, str) and requested.casefold() == self.model_identifier.casefold()


    def load_cancelled(self):
        return False

    def load_model(self, *, intent, load_cancel_event = None):
        self.loads += 1
        self.is_active = True
        self.is_loaded = True
        return True

    def non_chat_gguf_refusal_for_intent(self, _intent):
        return None

    def host_offload_warning_for_intent(self, _intent):
        return None
    def unload_model(self):
        self.unloads += 1
        self.is_loaded = False
        self.is_active = False
        self.model_identifier = None
        return True


def test_loading_slot_is_busy_before_its_backend_spawns(residents, monkeypatch):
    registry, load = residents
    monkeypatch.setenv("UNSLOTH_RESIDENT_MODEL_SLOTS", "1")
    slot, loading = load("org/B-GGUF")
    loading.is_active = False
    loading.is_loaded = False

    registry.start_loading(slot.id)

    assert registry.loading_slots() == [slot]
    assert registry.any_slot_busy()
    assert registry.at_capacity()


@pytest.fixture()
def residents(monkeypatch):
    """A fresh registry under a raised slot cap, swapped into the route module.

    Returns a helper that loads doubles into slots: the first call also makes
    its slot active, standing in for the session's first load.
    """
    monkeypatch.setenv("UNSLOTH_RESIDENT_MODEL_SLOTS", "4")
    pending: list[_ResidentDouble] = []
    registry = ResidentLlamaRegistry(
        default_backend = inference_route._llama_cpp_backend,
        backend_factory = lambda: pending.pop(0) if pending else _ResidentDouble("unused"),
    )
    monkeypatch.setattr(inference_route, "_resident_registry", registry)

    def _load(
        identifier,
        *,
        hf_variant = None,
        advertised_id = None,
        make_active = False,
    ):
        double = _ResidentDouble(identifier, hf_variant = hf_variant, advertised_id = advertised_id)
        pending.append(double)
        slot = registry.open_slot()
        if make_active:
            registry.set_active(slot.id)
        return slot, double

    return registry, _load


# ── Registry invariants ────────────────────────────────────────────


def test_two_models_stay_resident_and_independently_addressable(residents):
    registry, load = residents
    slot_a, a = load("org/A-GGUF", hf_variant = "Q4_K_M", make_active = True)
    slot_b, b = load("org/B-GGUF", hf_variant = "Q8_0")

    backends = inference_route.iter_resident_llama_backends()
    assert set(map(id, backends)) == {id(a), id(b)}
    # Active first, and unique slot identities: no port/process confusion.
    assert backends[0] is a
    assert slot_a.id != slot_b.id
    assert registry.active_backend() is a


def test_slots_only_count_while_their_backend_holds_a_process(residents):
    # The idle auto-unload tears the active backend down off the registry's
    # books; the leftover slot must not consume capacity.
    registry, load = residents
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("UNSLOTH_RESIDENT_MODEL_SLOTS", "2")
    try:
        _, a = load("org/A-GGUF", make_active = True)
        load("org/B-GGUF")
        assert registry.at_capacity()
        a.unload_model()
        assert not registry.at_capacity()
    finally:
        monkeypatch.undo()


def test_unload_one_resident_leaves_the_other_intact(residents):
    registry, load = residents
    _, a = load("org/A-GGUF", make_active = True)
    slot_b, b = load("org/B-GGUF")

    dropped = registry.drop_slot(slot_b.id)

    assert dropped is b
    assert b.unloads == 1
    assert a.unloads == 0 and a.is_loaded
    # The active slot was never B's to clear.
    assert registry.active_backend() is a
    assert inference_route.iter_resident_llama_backends() == [a]


def test_dropping_the_active_slot_promotes_nothing(residents):
    registry, load = residents
    slot_a, a = load("org/A-GGUF", make_active = True)
    load("org/B-GGUF")

    registry.drop_slot(slot_a.id)

    # residents = [B], active = none: the default answers unloaded, so status
    # reports no model while B stays addressable by name.
    assert registry.active_slot() is None
    assert registry.active_backend() is inference_route._llama_cpp_backend
    assert b_ids(registry) == 1


def b_ids(registry) -> int:
    return len(inference_route.iter_resident_llama_backends())


def test_global_shutdown_tears_down_every_slot(residents):
    registry, load = residents
    _, a = load("org/A-GGUF", make_active = True)
    _, b = load("org/B-GGUF")

    swept = registry.teardown_all()

    assert swept == 2
    assert a.unloads == 1 and b.unloads == 1
    assert registry.active_slot() is None
    assert inference_route.iter_resident_llama_backends() == []


def test_capacity_refusal_leaves_existing_residents_untouched(residents):
    registry, load = residents
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("UNSLOTH_RESIDENT_MODEL_SLOTS", "2")
    try:
        _, a = load("org/A-GGUF", make_active = True)
        load("org/B-GGUF")
        with pytest.raises(ResidentCapacityError):
            registry.open_slot()
        assert a.is_loaded and a.unloads == 0
        assert b_ids(registry) == 2
    finally:
        monkeypatch.undo()


@pytest.mark.parametrize("slot_cap", [2, 4])
def test_runtime_changed_secondary_reuses_its_occupied_slot(residents, monkeypatch, slot_cap):
    """A source match reserves B for replacement even when capacity is available."""
    registry, load = residents
    monkeypatch.setenv("UNSLOTH_RESIDENT_MODEL_SLOTS", str(slot_cap))
    _, a = load("org/A-GGUF", make_active = True)
    slot_b, b = load("org/B-GGUF")

    replacement = inference_route._resident_slot_for_matching_gguf_source(
        SimpleNamespace(model_identifier = "org/B-GGUF"), a
    )

    assert replacement is slot_b
    assert registry.slot_for_backend(replacement.backend) is slot_b
    assert len(registry.slots_for_sweep()) == 2
    assert b.is_loaded and a.is_loaded


def test_runtime_changed_secondary_bypasses_full_capacity_refusal(residents, monkeypatch):
    registry, load = residents
    monkeypatch.setenv("UNSLOTH_RESIDENT_MODEL_SLOTS", "2")
    _, a = load("org/A-GGUF", make_active = True)
    slot_b, _ = load("org/B-GGUF")

    assert registry.at_capacity()
    assert (
        inference_route._resident_slot_for_matching_gguf_source(
            SimpleNamespace(model_identifier = "org/B-GGUF"), a
        )
        is slot_b
    )


def test_identical_secondary_promotes_without_reloading(residents, monkeypatch):
    """Same effective runtime reuses B; only a mismatch reaches replacement."""
    registry, load = residents
    _, a = load("org/A-GGUF", make_active = True)
    _, b = load("org/B-GGUF")
    response = object()

    monkeypatch.setattr(
        inference_route,
        "_resolve_model_identifier_for_request",
        lambda *_args, **_kwargs: ("org/B-GGUF", "org/B-GGUF", False),
    )
    monkeypatch.setattr(
        inference_route, "resolve_effective_chat_template_override", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        inference_route, "get_inference_backend", lambda: SimpleNamespace(active_model_name = None)
    )
    monkeypatch.setattr(inference_route, "_active_gguf_intent", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(inference_route, "_gguf_load_response", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(inference_route, "_loaded_is_local_model", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(inference_route, "_request_used_api_key", lambda _request: False)
    monkeypatch.setattr(inference_route.api_monitor, "record_lifecycle", lambda **_kwargs: object())
    monkeypatch.setattr(inference_route.api_monitor, "discard", lambda _event: None)
    monkeypatch.setattr("core.inference.gpu_arbiter.acquire_for_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("hub.services.models.account_access.join_resident", lambda *_args: None)
    result = asyncio.run(
        inference_route._load_model_impl(
            LoadRequest(
                model_path = "org/B-GGUF",
                gguf_variant = "Q4_K_M",
                keep_existing_loaded = True,
            ),
            SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))),
            "tester",
        )
    )

    assert result is response
    assert registry.active_backend() is b
    assert b.unloads == 0 and a.unloads == 0


@pytest.mark.parametrize("slot_cap", [2, 4])
def test_runtime_changed_secondary_reloads_in_place_without_duplicate(
    residents, monkeypatch, slot_cap
):
    """Reloading B uses B's slot; A survives and the resident count stays fixed."""
    import contextlib

    from core.inference.llama_cpp import GgufLoadIntent

    registry, load = residents
    monkeypatch.setenv("UNSLOTH_RESIDENT_MODEL_SLOTS", str(slot_cap))
    _, a = load("org/A-GGUF", make_active = True)
    _, b = load("org/B-GGUF")
    b.adopt_matches = False  # Same source, changed runtime settings.
    response = object()
    config = SimpleNamespace(
        identifier = "org/B-GGUF",
        display_name = "B",
        is_gguf = True,
        is_lora = False,
        is_vision = False,
        is_audio = False,
        is_local = False,
        gguf_hf_repo = None,
        gguf_path = None,
    )
    intent = GgufLoadIntent(model_identifier = "org/B-GGUF", hf_variant = "Q4_K_M")

    async def _placement(*_args, **_kwargs):
        return inference_route._LoadPlacement(None, None, False, False)

    async def _idle(**_kwargs):
        return None

    monkeypatch.setattr(
        inference_route,
        "_resolve_model_identifier_for_request",
        lambda *_args, **_kwargs: ("org/B-GGUF", "org/B-GGUF", False),
    )
    monkeypatch.setattr(
        inference_route, "resolve_effective_chat_template_override", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        inference_route, "get_inference_backend", lambda: SimpleNamespace(active_model_name = None)
    )
    monkeypatch.setattr(inference_route, "_active_gguf_intent", lambda *_args, **_kwargs: intent)
    monkeypatch.setattr(inference_route.ModelConfig, "from_identifier", lambda **_kwargs: config)
    monkeypatch.setattr(
        inference_route, "_hf_offline_if_unreachable_for", lambda *_args: contextlib.nullcontext()
    )
    monkeypatch.setattr(inference_route, "_resolve_inherited_extra_args", lambda *_args: None)
    monkeypatch.setattr(inference_route, "_prepare_load_placement", _placement)
    monkeypatch.setattr(inference_route, "_resolve_gguf_load_intent", lambda *_args, **_kwargs: intent)
    monkeypatch.setattr(
        inference_route, "_guard_chat_load_against_training", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(inference_route, "_raise_if_sidecar_swap_in_progress", lambda: None)
    monkeypatch.setattr(inference_route, "_wait_for_model_switch_idle", _idle)
    monkeypatch.setattr(inference_route, "_close_load_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(inference_route, "_gguf_load_response", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(inference_route, "_request_used_api_key", lambda _request: False)
    monkeypatch.setattr(inference_route, "release_chat_gpu_claim", lambda: True)
    monkeypatch.setattr(inference_route.api_monitor, "record_lifecycle", lambda **_kwargs: object())
    monkeypatch.setattr(inference_route.api_monitor, "fail_open", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "core.inference.llama_cpp.zero_vram_chat_load", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        "core.inference.llama_cpp.chat_load_in_flight", lambda: contextlib.nullcontext()
    )
    monkeypatch.setattr("core.inference.llama_keepwarm.note_model_loaded", lambda _backend: None)
    monkeypatch.setattr("hub.services.models.account_access.publish_resident", lambda *_args: None)

    result = asyncio.run(
        inference_route._load_model_impl(
            LoadRequest(
                model_path = "org/B-GGUF",
                gguf_variant = "Q4_K_M",
                keep_existing_loaded = True,
            ),
            SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))),
            "tester",
        )
    )

    assert result is response
    assert b.loads == 1 and registry.active_backend() is b
    assert a.is_loaded and a.unloads == 0
    assert len(registry.slots_for_sweep()) == 2


def test_concurrent_open_and_drop_keeps_slot_ids_unique(residents, monkeypatch):
    registry, _ = residents
    monkeypatch.setenv("UNSLOTH_RESIDENT_MODEL_SLOTS", "16")
    made: list[int] = []
    guard = threading.Lock()

    def _open():
        slot = registry.open_slot()
        with guard:
            made.append(slot.id)

    threads = [threading.Thread(target = _open) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(made)) == 8

    # Drop from several threads at once: exactly one drop per slot lands.
    dropped = [0]

    def _drop(slot_id):
        if registry.drop_slot(slot_id) is not None:
            with guard:
                dropped[0] += 1

    threads = [threading.Thread(target = _drop, args = (sid,)) for sid in made]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert dropped[0] == 8
    assert b_ids(registry) == 0


# ── Flag off: today's single-model behaviour ──────────────────────


def test_flag_off_keeps_the_module_backend_and_ignores_residents(monkeypatch):
    monkeypatch.delenv("UNSLOTH_RESIDENT_MODEL_SLOTS", raising = False)
    assert not residency_enabled()
    assert max_resident_slots() == 1

    active = inference_route.get_llama_cpp_backend()
    assert active is inference_route._llama_cpp_backend
    # Nothing is loaded on a fresh backend, so the iterated list is empty --
    # exactly the shape the pre-registry is_loaded guards produced.
    assert inference_route.iter_resident_llama_backends() == []


def test_keep_existing_is_rejected_on_a_single_model_server(monkeypatch):
    monkeypatch.delenv("UNSLOTH_RESIDENT_MODEL_SLOTS", raising = False)
    request = LoadRequest(model_path = "org/A-GGUF", gguf_variant = "Q4_K_M")
    request.keep_existing_loaded = True

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route._load_model_impl(request, None, "tester"))

    assert excinfo.value.status_code == 400
    assert "keep_existing_loaded" in excinfo.value.detail


# ── Request routing ────────────────────────────────────────────────


def test_a_named_resident_resolves_to_its_own_backend(residents):
    _, load = residents
    _, a = load("org/A-GGUF", hf_variant = "Q4_K_M", make_active = True)
    _, b = load("org/B-GGUF", hf_variant = "Q8_0")

    resolved = inference_route.resolve_resident_llama("org/B-GGUF")

    assert resolved is not None and resolved.backend is b
    assert resolved.is_active is False
    assert resolved.model_identifier == "org/B-GGUF"
    # And the serving seam hands generation the same backend.
    assert inference_route._serving_llama_backend("org/B-GGUF") is b
    assert inference_route._serving_llama_backend("org/A-GGUF") is a


def test_an_omitted_or_unknown_model_resolves_to_no_resident_never_a_secondary(residents):
    _, load = residents
    _, a = load("org/A-GGUF", make_active = True)
    load("org/B-GGUF")

    # Omitted / non-string / reload-only: no resident is silently chosen.
    assert inference_route.resolve_resident_llama(None) is None
    assert inference_route.resolve_resident_llama("") is None
    assert inference_route.resolve_resident_llama(123) is None
    # An unknown id must NOT resolve to any resident...
    assert inference_route.resolve_resident_llama("org/C-GGUF") is None
    # ...so the serving seam answers the ACTIVE backend, the same object a
    # single-model server would hand out. The refusal itself is upstream
    # (_reject_unservable_model); the contract here is that no secondary and
    # no wrong backend ever answers for the name.
    assert inference_route._serving_llama_backend("org/C-GGUF") is a
    assert inference_route._serving_llama_backend(None) is a


def test_identity_predicates_see_every_resident(residents):
    _, load = residents
    load("org/A-GGUF", make_active = True)
    load("org/B-GGUF", hf_variant = "Q8_0")

    assert inference_route._loaded_satisfies("org/B-GGUF")
    assert inference_route._loaded_identity_satisfies("org/B-GGUF")
    assert not inference_route._loaded_satisfies("org/C-GGUF")


def test_a_quant_tag_must_match_the_resident_variant(residents):
    # Same repo, different quant: never confused with the other resident.
    _, load = residents
    load("org/A-GGUF", hf_variant = "Q4_K_M", make_active = True)
    _, b = load("org/B-GGUF", hf_variant = "Q8_0")

    assert inference_route.resolve_resident_llama("org/B-GGUF:Q8_0").backend is b
    assert inference_route.resolve_resident_llama("org/B-GGUF:Q4_K_M") is None
    assert not inference_route._loaded_satisfies("org/B-GGUF:Q4_K_M")


def test_local_paths_compare_without_substring_matching(residents):
    # Sibling files sharing a basename prefix must not answer for each other.
    # The alias is what a post-recording request matches (the raw load path is
    # deliberately held back from identity matches), so advertise it.
    _, load = residents
    load("/srv/models/model-a.gguf", advertised_id = "/srv/models/model-a.gguf", make_active = True)

    resolved = inference_route.resolve_resident_llama("/srv/models/model-a.gguf")
    assert resolved is not None and resolved.model_identifier == "/srv/models/model-a.gguf"
    assert inference_route.resolve_resident_llama("/srv/models/model-ab.gguf") is None


def test_a_path_named_secondary_resolves_and_records_its_alias(residents):
    # A manual local-path load advertises nothing; the switch helper's alias
    # recording only ever reaches the ACTIVE backend, so the resolver must
    # accept an exact path request for a SECONDARY and record the clean public
    # id itself -- parity with what _record_serving_alias does for the active.
    _, load = residents
    load("org/A-GGUF", make_active = True)
    _, b = load("/srv/models/model-b.gguf")

    resolved = inference_route.resolve_resident_llama("/srv/models/model-b.gguf")
    assert resolved is not None and resolved.backend is b
    assert b._openai_advertised_id == "model-b"
    # Later requests naming the recorded id route the same way, and a sibling
    # path still never matches.
    assert inference_route._serving_llama_backend("model-b") is b
    assert inference_route.resolve_resident_llama("/srv/models/model-bb.gguf") is None


def test_auto_switch_keeps_a_local_path_secondary_resident(monkeypatch, residents):
    """A resolver hit for a manually loaded secondary must not reload it over the active slot."""
    _, load = residents
    _, active = load("org/A-GGUF", make_active = True)
    _, secondary = load("/srv/models/model-b.gguf")
    loads = []

    async def _load(*_args, **_kwargs):
        loads.append(True)

    from core.inference import local_model_resolver as resolver
    from utils import openai_auto_switch_settings as settings

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: False)
    monkeypatch.setattr(resolver, "resolve_trusted_cached_local_gguf", lambda *_a, **_k: None)
    monkeypatch.setattr(
        resolver,
        "resolve_local_gguf",
        lambda *_a, **_k: ("/srv/models/model-b.gguf", None, "model-b", False),
    )
    monkeypatch.setattr(resolver, "local_target_is_gguf", lambda *_a, **_k: True)
    monkeypatch.setattr(resolver, "local_gguf_companion_roots", lambda *_a, **_k: ())
    monkeypatch.setattr(resolver, "local_gguf_companion_state", lambda *_a, **_k: ())
    monkeypatch.setattr(inference_route, "_loaded_identity_satisfies", lambda _model: False)
    monkeypatch.setattr(inference_route, "_auto_download_hf_token", lambda _request: None)
    monkeypatch.setattr(inference_route, "_load_model_impl", _load)

    request = SimpleNamespace(
        state = SimpleNamespace(generation_cancel_event = None),
        scope = {},
        headers = {},
        url = None,
    )
    asyncio.run(
        inference_route._maybe_auto_switch_model("/srv/models/model-b.gguf", request, "tester")
    )

    assert loads == []
    assert active.is_loaded
    assert secondary.is_loaded
    assert secondary._openai_advertised_id == "model-b"


def test_auto_switch_short_circuits_for_a_resident_secondary(monkeypatch, residents):
    # Naming a resident must neither swap nor reload: the switch helper returns
    # without loading, and the handler's serving seam (tested above) picks the
    # named slot. Auto-switch off keeps the refusal path honest instead.
    _, load = residents
    load("org/A-GGUF", make_active = True)
    load("org/B-GGUF")

    loads: list = []

    async def _no_load(*_a, **_k):
        loads.append(1)
        return None

    monkeypatch.setattr(inference_route, "_load_model_impl", _no_load)
    monkeypatch.setattr(
        "utils.openai_auto_switch_settings.get_openai_auto_switch_enabled", lambda: False
    )
    monkeypatch.setattr(
        "utils.openai_auto_switch_settings.idle_unload_is_configured", lambda: False
    )

    class _FakeOrchestrator:
        active_model_name = None
        models = {}
        _openai_advertised_id = None

    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _FakeOrchestrator())
    monkeypatch.setattr(inference_route, "_peek_inference_backend", lambda: _FakeOrchestrator())

    asyncio.run(inference_route._maybe_auto_switch_model("org/B-GGUF", object(), "tester"))
    assert loads == []


# ── Unload routing ────────────────────────────────────────────────


def _fake_orchestrator():
    class _Fake:
        active_model_name = None
        models = {}

        def get_loading_model(self):
            return None

    return _Fake()


def test_unload_may_evict_covers_a_named_secondary(residents, monkeypatch):
    # The refusal before teardown must fire for a secondary too: unloading a
    # model with admitted generations is refused exactly like the active one.
    _, load = residents
    load("org/A-GGUF", make_active = True)
    load("org/B-GGUF")

    monkeypatch.setattr(inference_route, "get_inference_backend", _fake_orchestrator)

    assert inference_route._unload_may_evict("org/B-GGUF")
    assert inference_route._unload_may_evict("org/A-GGUF")
    assert not inference_route._unload_may_evict("org/C-GGUF")


def test_named_secondary_slot_lookup_skips_the_active(residents):
    _, load = residents
    load("org/A-GGUF", make_active = True)
    slot_b, b = load("org/B-GGUF")

    found = inference_route._named_secondary_slot("org/B-GGUF")
    assert found is not None and found.id == slot_b.id
    assert inference_route._named_secondary_slot("org/A-GGUF") is None


# ── GPU-owner eviction sees secondary slots ───────────────────────


def test_gpu_owner_eviction_tears_down_secondary_residents(monkeypatch):
    import core.inference.gpu_arbiter as arb

    unloaded: list = []

    class _FakeLlama:
        is_active = False
        is_loaded = False

        def unload_model(self):
            unloaded.append("active")

        def _wait_for_vram_settle(self, *, since_kill):
            pass

    torn_down: list = []

    class _FakeRegistry:
        def teardown_all(self):
            torn_down.append(True)
            return 1

    class _FakeOrchestrator:
        active_model_name = None
        models = {}
        loading_models = ()

        def _shutdown_subprocess(self, timeout = 5.0):
            pass

    import core.inference as core_inference
    from core.inference import resident_models as resident_models_mod

    monkeypatch.setattr(
        "routes.inference.get_llama_cpp_backend", lambda: _FakeLlama(), raising = False
    )
    monkeypatch.setattr(resident_models_mod, "get_registry", lambda: _FakeRegistry())
    monkeypatch.setattr(core_inference, "get_inference_backend", lambda: _FakeOrchestrator())
    monkeypatch.setattr(arb, "clear_resident", lambda *a, **k: None, raising = False)

    arb._evict_chat()

    # The active backend was already down; the resident sweep still ran, which
    # is what frees the secondary slots' VRAM for the incoming owner.
    assert torn_down == [True]
    assert "active" not in unloaded


# ── Source contracts for the load/seam wiring ─────────────────────
# The load impl's slot targeting and failure cleanup cannot be driven end to
# end without disproportionate machinery (real downloads, GPU placement,
# gates); these lock the wiring behavioral tests cannot reach.


def test_the_serving_seam_resolves_once_in_chat_completions():
    src = inspect.getsource(inference_route.produce_openai_chat_completions)
    seam = src.index("_serving_llama_backend(")
    assert seam < src.index("using_gguf = llama_backend.is_loaded")
    # Exactly one fetch of the serving backend: no later re-read of the active
    # backend (including the MTP crash-recovery paths) can swap the model
    # mid-request.
    assert src.count("get_llama_cpp_backend()") == 0


def test_the_load_path_targets_fresh_slots_and_cleans_them_up():
    src = inspect.getsource(inference_route._load_model_impl)
    gguf_branch = src[src.index("if config.is_gguf:") :]
    # Capacity is checked before the drain and the point of no return.
    assert gguf_branch.index("at_capacity()") < gguf_branch.index("_wait_for_model_switch_idle")
    # The slot is opened after the intent resolves, and set active on success.
    assert "open_slot()" in gguf_branch
    assert "set_active(_resident_target_slot.id)" in gguf_branch
    # Three failed-load cleanup exits plus one prior-active eviction after a
    # successful in-place secondary replacement.
    assert gguf_branch.count("_resident_registry.drop_slot") == 4


def test_promote_and_replace_passes_the_point_of_no_return_before_moving_anything():
    src = inspect.getsource(inference_route._load_model_impl)
    block = src[src.index("Multi-residency: the requested model may already be resident") :]
    block = block[: block.index("if not (request.gguf_variant or is_direct_gguf_request)")]
    # Refuse-on/cancel active generations, then drain, THEN move the active
    # pointer and unload the prior model: a 409 must leave the prior model
    # serving and active.
    guards = block.index("_raise_if_scoped_load_cancelled()")
    assert guards < block.index("on_reload_confirmed(cancel = True)")
    assert block.index("on_reload_confirmed(cancel = True)") < block.index(
        "_wait_for_model_switch_idle"
    )
    assert block.index("_wait_for_model_switch_idle") < block.index("set_active")
    assert block.index("set_active") < block.index("_prior_active_slot.backend.unload_model")


# ── Behavioral: unload dispatch through the real impl ─────────────


async def _null_gate():
    import contextlib

    @contextlib.asynccontextmanager
    async def _gate():
        yield

    return _gate()


async def _noop_drain(*_a, **_k):
    return None


def _noop_raise(*_a, **_k):
    return None


@pytest.fixture()
def _unloadable_world(monkeypatch, residents):
    """The unloaded-down pieces of _unload_model_impl's path.

    The gate, the active-generation refusal and the post-cancel drain are the
    single-model server's machinery and have their own tests; stubbing them
    lets this file exercise the resident DISPATCH behaviorally: which slot
    unloads, what stays, and who is refused.
    """
    import contextlib

    import core.inference.llama_keepwarm as keepwarm

    @contextlib.asynccontextmanager
    async def _gate():
        yield

    monkeypatch.setattr(keepwarm, "inference_lifecycle_gate", _gate)
    monkeypatch.setattr(inference_route, "_raise_or_cancel_active_generations", _noop_raise)
    monkeypatch.setattr(inference_route, "_drain_and_recancel_before_teardown", _noop_drain)
    monkeypatch.setattr(inference_route, "get_inference_backend", _fake_orchestrator)
    return residents


def test_unloading_a_named_secondary_leaves_the_active_model_serving(_unloadable_world):
    from models.inference import UnloadRequest

    registry, load = _unloadable_world
    _, a = load("org/A-GGUF", make_active = True)
    slot_b, b = load("org/B-GGUF")

    response = asyncio.run(
        inference_route._unload_model_impl(UnloadRequest(model_path = "org/B-GGUF"), "tester")
    )

    assert response.status == "unloaded"
    assert b.unloads == 1 and not b.is_loaded
    assert a.unloads == 0 and a.is_loaded
    assert registry.active_backend() is a
    assert inference_route.iter_resident_llama_backends() == [a]
    assert registry.slot_for_backend(b) is None and slot_b.id is not None


def test_stop_loading_cancels_fresh_secondary_without_unloading_active(_unloadable_world):
    from models.inference import UnloadRequest

    registry, load = _unloadable_world
    _, a = load("org/A-GGUF", make_active = True)
    slot_b, b = load("org/B-GGUF")
    b.is_loaded = False
    registry.start_loading(slot_b.id)
    assert inference_route._unload_may_evict("org/B-GGUF")

    response = asyncio.run(
        inference_route._unload_model_impl(UnloadRequest(model_path = "org/B-GGUF"), "tester")
    )

    assert response.status == "unloaded"
    assert b.unloads == 1 and not b.is_active
    assert a.unloads == 0 and a.is_loaded
    assert registry.active_backend() is a
    assert registry.slot_for_backend(b) is None
    assert registry.loading_slots() == []


def test_stop_loading_cancels_the_only_first_resident_slot(_unloadable_world):
    from models.inference import UnloadRequest

    registry, load = _unloadable_world
    slot_b, b = load("org/B-GGUF")
    b.is_loaded = False
    registry.start_loading(slot_b.id)

    response = asyncio.run(
        inference_route._unload_model_impl(UnloadRequest(model_path = "stale-ui-model"), "tester")
    )

    assert response.status == "unloaded"
    assert b.unloads == 1 and not b.is_active
    assert registry.slot_for_backend(b) is None
    assert registry.active_slot() is None


def test_a_foreign_caller_cannot_unload_a_named_secondary(_unloadable_world, monkeypatch):
    # Account isolation: a managed caller that cannot see a resident must not
    # be able to eject it by name. The slot, the active model and the teardown
    # bookkeeping all stay untouched when the 404 raises.
    from models.inference import UnloadRequest
    from utils.account_context import AccountContext, bind_account

    from hub.services.models import account_access

    registry, load = _unloadable_world
    _, a = load("org/A-GGUF", make_active = True)
    slot_b, b = load("org/B-GGUF")

    bob = AccountContext("b" * 32, "bob")
    monkeypatch.setattr(
        account_access,
        "resident_hidden",
        lambda modality, reference = None: reference == "org/B-GGUF",
    )

    token = bind_account(bob)
    try:
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(
                inference_route._unload_model_impl(UnloadRequest(model_path = "org/B-GGUF"), "tester")
            )
    finally:
        from utils.account_context import reset_account
        reset_account(token)

    assert excinfo.value.status_code == 404
    assert b.unloads == 0 and b.is_loaded
    assert a.unloads == 0 and a.is_loaded
    assert registry.active_backend() is a


# ── Behavioral: account visibility of resident listings ───────────


def test_foreign_residents_are_not_listed_for_a_managed_caller(residents, monkeypatch):
    from hub.services.models import account_access

    registry, load = residents
    load("org/A-GGUF", make_active = True)
    load("org/B-GGUF")

    # Managed caller: A (the active, its own publish shape) is visible, B is
    # foreign under every spelling and must not leak into any listing.
    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    monkeypatch.setattr(
        account_access,
        "resident_hidden",
        lambda modality, reference = None: reference is not None and "A" not in str(reference),
    )

    rows = inference_route._resident_models_status_rows()
    assert [row["model"] for row in rows] == ["org/A-GGUF"]
    listed = inference_route._openai_model_objects()
    assert [entry["id"] for entry in listed] == ["org/A-GGUF"]

    # Unmanaged installs and the owner see everything.
    monkeypatch.setattr(account_access, "managed_account", lambda: False)
    assert len(inference_route._resident_models_status_rows()) == 2
    assert len(inference_route._openai_model_objects()) == 2


def test_multi_residency_is_rejected_on_managed_account_installs(monkeypatch):
    monkeypatch.setenv("UNSLOTH_RESIDENT_MODEL_SLOTS", "4")
    monkeypatch.setattr("auth.policy.installation_has_managed_accounts", lambda: True)
    request = LoadRequest(model_path = "org/A-GGUF", gguf_variant = "Q4_K_M")
    request.keep_existing_loaded = True

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route._load_model_impl(request, None, "tester"))

    assert excinfo.value.status_code == 400
    assert "managed accounts" in excinfo.value.detail


# ── Behavioral: keep-existing memory accounting ───────────────────


def test_a_keep_existing_load_is_not_credited_the_resident_llamas_vram(monkeypatch):
    """The audio handoff budget may only count memory the load really frees.

    A normal load evicts the active llama-server, so its startup model buffers
    are credited as reclaimable. A keep-existing load swaps nothing out: the
    same buffers must stay committed, and with nothing else creditable the
    snapshot fails closed (None) rather than advertise free memory that is not.
    """
    import utils.hardware as hardware

    class _ResidentLlama:
        is_loaded = True

        def reclaimable_gpu_memory_gb(self):
            return {0: 4.0}

    class _IdleWorker:
        active_model_name = None

    monkeypatch.setattr(
        hardware,
        "get_visible_gpu_utilization",
        lambda: {"devices": [{"index": 0, "vram_total_gb": 24.0, "vram_used_gb": 20.0}]},
    )
    monkeypatch.setattr("core.inference.gpu_arbiter.owner_snapshot", lambda: ("chat", 7))
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _ResidentLlama())
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _IdleWorker())

    normal = inference_route._native_audio_post_handoff_free_gb()
    assert normal is not None
    assert normal.effective_free_gb[0] == pytest.approx(8.0)  # 4 free + 4 credited

    kept = inference_route._native_audio_post_handoff_free_gb(keep_resident = True)
    # Nothing provably evicted -> no availability published at all.
    assert kept is None


# ── Behavioral: explicit unknown-model routing ────────────────────


def test_an_attributable_unknown_name_is_refused_not_served_by_the_active(monkeypatch, residents):
    """A name this server can attribute (explicit quant) must 404 rather than
    fall back to the active backend; an unattributable vendor-style id keeps
    the documented drop-in rule and passes through."""
    from core.inference import local_model_resolver as resolver

    _, load = residents
    load("org/A-GGUF", make_active = True)

    monkeypatch.setattr(resolver, "index_is_built", lambda: True)
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda *_a, **_k: None)
    monkeypatch.setattr(resolver, "recently_downloaded", lambda *_a, **_k: False)
    monkeypatch.setattr(inference_route, "_advertised_local_path", lambda _base: None)
    monkeypatch.setattr(
        "utils.openai_auto_switch_settings.get_openai_auto_switch_enabled", lambda: False
    )

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route._reject_unservable_model("org/C-GGUF:Q4_K_M", None))
    assert excinfo.value.status_code == 404

    # Unattributable, non-quantified, not a hub repo id: drop-in compatibility
    # passes it through, and the handler's backend choice stays the active one.
    asyncio.run(inference_route._reject_unservable_model("some-vendor-id", None))
    active = inference_route.get_llama_cpp_backend()
    assert inference_route._serving_llama_backend("some-vendor-id") is active


def test_status_resident_rows_flag_the_active_slot(residents):
    registry, load = residents
    load("org/A-GGUF", hf_variant = "Q4_K_M", make_active = True)
    load("org/B-GGUF", hf_variant = "Q8_0")

    rows = inference_route._resident_models_status_rows()

    assert [row["model"] for row in rows] == ["org/A-GGUF", "org/B-GGUF"]
    assert rows[0]["is_active"] is True and rows[1]["is_active"] is False
    assert rows[0]["gguf_variant"] == "Q4_K_M"
    from models.inference import InferenceStatusResponse

    assert "resident_models" in InferenceStatusResponse.model_fields


# ── Review-fix regressions ────────────────────────────────────────


def test_named_studio_embedder_uses_a_secondary_resident_before_fallback(monkeypatch, residents):
    _, load = residents
    load("org/A-GGUF", make_active = True)
    _, embedding = load("org/B-GGUF")
    embedding.is_embedding_gguf = True
    body = {"model": "org/B-GGUF", "input": "hello"}
    checked = []

    async def _studio_request(_request):
        return body, "configured-embedder"

    async def _answers(backend, requested):
        checked.append((backend, requested))
        return backend is embedding

    async def _switch(*_args, **_kwargs):
        raise RuntimeError("reached resident switch")

    async def _fallback(*_args, **_kwargs):
        raise AssertionError("selected resident must not fall back to Studio embeddings")

    monkeypatch.setattr(inference_route, "_should_validate_before_switch", lambda: False)
    monkeypatch.setattr(inference_route, "_studio_embedder_request_body", _studio_request)
    monkeypatch.setattr(inference_route, "_resident_answers_embeddings", _answers)
    monkeypatch.setattr(inference_route, "_auto_switch_from_request_body", _switch)
    monkeypatch.setattr(inference_route, "_studio_embeddings", _fallback)

    with pytest.raises(RuntimeError, match = "reached resident switch"):
        asyncio.run(inference_route.openai_embeddings(object(), "tester"))
    assert checked == [(embedding, "org/B-GGUF")]


def test_tts_budget_uses_the_named_secondary_context(residents):
    _, load = residents
    _, active = load("org/A-GGUF", make_active = True)
    _, secondary = load("org/B-GGUF")
    active.context_length = 4096
    secondary.context_length = 128
    text = "x" * 64
    payload = SimpleNamespace(max_completion_tokens = 512, max_tokens = None)

    assert inference_route._monitor_context_length(secondary) == 128
    assert inference_route._tts_max_new_tokens(payload, text, llama_backend = secondary) == 32
    with pytest.raises(HTTPException) as excinfo:
        inference_route._raise_if_prompt_leaves_no_speech_budget(text, llama_backend = secondary)
    assert excinfo.value.status_code == 400


def test_anthropic_named_secondary_survives_an_unloaded_active_slot(monkeypatch, residents):
    _, load = residents
    _, active = load("org/A-GGUF", make_active = True)
    _, secondary = load("org/B-GGUF")
    active.unload_model()
    switched = []

    async def _switch(model, *_args, **_kwargs):
        switched.append(model)
        raise RuntimeError("reached resident switch")

    from models.inference import AnthropicMessagesRequest

    monkeypatch.setattr(inference_route, "_admit_tool_access", lambda _payload: None)
    monkeypatch.setattr(inference_route, "_automatic_model_load_may_run", lambda: False)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _switch)
    payload = AnthropicMessagesRequest(
        model = "org/B-GGUF",
        max_tokens = 1,
        messages = [{"role": "user", "content": "hello"}],
    )

    with pytest.raises(RuntimeError, match = "reached resident switch"):
        asyncio.run(inference_route.anthropic_messages(payload, object(), "tester"))
    assert switched == ["org/B-GGUF"]
    assert secondary.is_loaded


def test_a_standard_load_rejects_the_keep_flag_and_clears_all_residents():
    src = inspect.getsource(inference_route._load_model_impl)
    standard = src[src.index("# ── Standard path: load via Unsloth/transformers") :]
    # The flag has no meaning on the standard path: refused before the drain...
    assert standard.index("keep_existing_loaded applies to GGUF") < standard.index(
        "_wait_for_model_switch_idle"
    )
    # ...and every resident slot is swept with the active GGUF, so none holds
    # VRAM under the new model.
    assert "_resident_registry.teardown_all" in standard


def test_the_chat_claim_survives_a_zero_vram_load_while_a_resident_holds_vram():
    src = inspect.getsource(inference_route._load_model_impl)
    zero_vram = src[src.index("Drop the stale CHAT claim") :]
    # The conditional claim release, never a bare release(CHAT): a surviving
    # resident must keep chat ownership so a media acquire still evicts it.
    assert "release_chat_gpu_claim" in zero_vram[: zero_vram.index("\n\n")]


def test_release_chat_gpu_claim_keeps_the_claim_while_a_slot_is_busy(monkeypatch, residents):
    registry, load = residents
    _, a = load("org/A-GGUF", make_active = True)
    _, b = load("org/B-GGUF")
    a.is_active = False  # the zero-VRAM load just finished; the active holds nothing

    from core.inference import gpu_arbiter as arb

    arb.acquire_for(arb.CHAT)
    try:

        class _IdleOrchestrator:
            active_model_name = None
            loading_models = ()

        monkeypatch.setattr(inference_route, "_peek_inference_backend", lambda: _IdleOrchestrator())
        # The resident slot still holds a process, so the claim must survive.
        assert not inference_route.release_chat_gpu_claim()
        assert arb.current_owner() == arb.CHAT

        # Once no resident holds a process either, the claim drops as before.
        b.is_active = False
        assert inference_route.release_chat_gpu_claim()
        assert arb.current_owner() is None
    finally:
        arb.release(arb.CHAT)


def test_training_cleanup_frees_gpu_holding_residents_only(monkeypatch, residents):
    registry, load = residents
    _, a = load("org/A-GGUF", make_active = True)
    _, b = load("org/B-GGUF")
    _, c = load("org/C-GGUF")
    c._gpu_offload_active = False  # CPU-only: exempt, keeps running

    import routes.training_vram as training_vram

    class _IdleOrchestrator:
        active_model_name = None
        loading_models = ()

    monkeypatch.setattr("core.inference.get_inference_backend", lambda: _IdleOrchestrator())
    # The active fake holds no VRAM marker, so only the resident sweep matters.
    freed = training_vram.free_chat_models_for_training("test")

    assert "gguf:org/B-GGUF" in freed
    assert c.is_loaded and registry.slot_for_backend(c) is not None
    assert registry.slot_for_backend(b) is None


def test_cache_deletion_is_blocked_by_a_named_secondary_resident(monkeypatch, residents):
    from hub.services.models import deletion

    registry, load = residents
    load("org/A-GGUF", make_active = True)
    load("org/B-GGUF", hf_variant = "Q8_0")

    assert deletion._llama_cpp_blocks_delete("org/B-GGUF", None)
    assert deletion._llama_cpp_blocks_delete("org/B-GGUF", "Q8_0")
    assert not deletion._llama_cpp_blocks_delete("org/UNRELATED-GGUF", None)


def test_status_loaded_lists_residents_when_nothing_is_active():
    src = inspect.getsource(inference_route.get_status)
    # Both non-active-GGUF branches merge the resident rows into loaded, so the
    # field never contradicts resident_models.
    assert 'loaded = [row["model"] for row in _resident_rows' in src
    assert 'row["model"] not in backend.models' in src
