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
from core.inference import gpu_arbiter, gpu_residency, mlx_inference
from core.inference.llama_cpp import GpuMemoryShortError
from models.inference import LoadRequest, LoadResponse
from utils import multi_model_settings
from utils.account_context import AccountContext, arun_as

BOB = AccountContext("b" * 32, "bob")
IMG = {"kind": "image", "model": "org/Img"}


class FakeLlama:
    holds_no_vram, is_vision, context_length = False, False, 4096

    def __init__(
        self,
        identifier = None,
        variant = None,
    ):
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
    state = SimpleNamespace(
        policy = "fit",
        requests = [],
        drops = [],
        evicts = [],
        unloads = [],
        free_calls = [],
        frees = [IMG],
        errors = [],
        retry_flags = [],
    )
    monkeypatch.setattr(gpu_residency, "current_policy", lambda: state.policy)
    monkeypatch.setattr(model_slots, "drop", lambda slot: state.drops.append(slot))
    monkeypatch.setattr(model_slots, "evict", lambda *a: state.evicts.append(a) or [])
    monkeypatch.setattr(inf, "_unload_primary", lambda: state.unloads.append(True))

    def free_for(
        owner,
        gpu_ids,
        reason,
        evicted = None,
        *,
        need_mib = None,
        short_mib = 0,
        media_only = False,
    ):
        call = {"owner": owner, "gpu_ids": tuple(gpu_ids or ()), "short_mib": short_mib}
        state.free_calls.append({**call, "media_only": True} if media_only else call)
        evicted.extend(dict(s) for s in state.frees)
        return [s["model"] for s in state.frees]

    monkeypatch.setattr(gpu_residency, "free_for", free_for)
    rows = {0: {"free_mib": 4096, "total_mib": 81920}}
    monkeypatch.setattr(gpu_residency, "_free_mib_by_gpu", lambda: (rows, None))
    monkeypatch.setattr(
        gpu_arbiter, "resident_slots", lambda o: [dict(IMG)] if o == gpu_arbiter.DIFFUSION else []
    )

    async def load(request, *args, **kwargs):
        state.requests.append(request)
        state.retry_flags.append(inf._primary_can_retry_fit.get())
        if state.errors and not request.force_alongside:
            raise state.errors.pop(0)
        return LoadResponse(
            status = "loaded", model = request.model_path, display_name = "A", inference = {}
        )

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", load)
    return state


def _short(capped = False):
    return GpuMemoryShortError(
        "needs 13 GB, 8 GB free", capped = capped, short_mib = 5000, gpu_indices = (0,)
    )


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


def test_freed_media_is_reported_once(env):
    env.errors = [_short()]
    response, _ = _gated()
    # The route's eviction sink carries it into both fields; the retry must not add it a second time.
    assert response.evicted == ["org/Img"] and response.evicted_slots == [IMG]


def test_an_idle_video_server_freed_alone_still_earns_a_full_size_retry(env, monkeypatch):
    env.errors = [_short()]
    monkeypatch.setattr(
        gpu_residency, "free_for", lambda *a, **k: env.free_calls.append(a) or ["video-server"]
    )
    response, _ = _gated()
    assert [r.force_alongside for r in env.requests] == [False, False] and not response.evicted


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


def _refused(account = None) -> str:
    """The Always refusal reaches the caller as the K2 409, header included."""
    with pytest.raises(HTTPException) as info:
        _gated(account)
    assert info.value.status_code == 409
    assert info.value.headers.get("X-Unsloth-Refusal") == "multi-model-always"
    return str(info.value.detail)


def test_always_refuses_with_the_numbers_for_the_owner(env):
    env.policy, env.errors = "always", [_short()]
    message = _refused()
    assert "is set to Always" in message
    # 5000 MiB short beside 4096 MiB free: needs 8.9 GB, 4.0 GB is free.
    assert "needs about 8.9 GB of GPU memory, and 4.0 GB is free" in message
    assert env.free_calls == [] and len(env.requests) == 1
    _nothing_unloaded(env)


def test_always_refuses_a_managed_account_with_the_plain_copy(env):
    env.policy, env.errors = "always", [_short()]
    message = _refused(BOB)
    assert "GB" not in message
    assert "the image model org/Img, which Unsloth Studio is set to keep loaded" in message
    assert env.free_calls == []
    _nothing_unloaded(env)


def test_always_names_only_media_on_the_short_gpu(env, monkeypatch):
    # The image on GPU 1 isn't in the way of a chat load short on GPU 0; the video model there is.
    vid = {"kind": "video", "model": "org/Vid"}
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.VIDEO)
    monkeypatch.setattr(
        gpu_arbiter,
        "_residents",
        {gpu_arbiter.DIFFUSION: frozenset({1}), gpu_arbiter.VIDEO: frozenset({0})},
    )
    monkeypatch.setattr(
        gpu_arbiter,
        "resident_slots",
        lambda o: {gpu_arbiter.DIFFUSION: [dict(IMG)], gpu_arbiter.VIDEO: [dict(vid)]}.get(o, []),
    )
    env.policy, env.errors = "always", [_short()]
    message = _refused(BOB)
    assert "org/Vid" in message and "org/Img" not in message
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
    message = _refused()
    assert "is set to Always" in message and "org/Img" in message
    assert env.free_calls == []
    _nothing_unloaded(env)


def test_non_gguf_oom_under_always_beside_media_on_another_gpu_keeps_the_original_error(
    env, monkeypatch
):
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    monkeypatch.setattr(
        gpu_arbiter,
        "_residents",
        {gpu_arbiter.DIFFUSION: frozenset({0}), gpu_arbiter.CHAT: frozenset({1})},
    )
    env.policy, env.errors = "always", [_oom()]
    with pytest.raises(HTTPException) as info:
        _gated()
    assert info.value.status_code == 500 and info.value.detail == "CUDA out of memory"
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


def _placement(
    resolved,
    vulkan = False,
    native_gb = None,
):
    return SimpleNamespace(
        requested_gpu_ids = None,
        resolved_gpu_ids = resolved,
        gpu_ids_are_vulkan_ordinals = vulkan,
        native_required_gb = native_gb,
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


def test_mlx_chat_need_prices_one_row_the_prompt_cache_and_a_batch_row_per_extra_slot(monkeypatch):
    gib = 1024**3
    breakdown = SimpleNamespace(
        total_bytes = 3 * gib, weights_bytes = gib, kv_bytes = gib // 2, compute_bytes = gib // 4
    )
    estimate = inf._MlxLoadEstimate(breakdown, None, (), True)
    monkeypatch.setattr(inf, "_mlx_load_estimate", lambda *a: (None, estimate))
    monkeypatch.setattr(
        mlx_inference, "mlx_memory_budget", lambda **k: (10 - k["retains_history"]) << 30
    )
    mlx = SimpleNamespace(is_gguf = False)
    assert inf._mlx_chat_need(None, mlx, "m") is None  # not on Metal
    monkeypatch.setattr(inf, "_unified_metal_host", lambda: True)
    need, growth = inf._mlx_chat_need(None, mlx, "m", 4)
    assert (need, growth) == (4096 + 3 * 768, {"mib": 3072, "row_mib": 768, "slots": 4})
    offloaded = estimate._replace(offloaded = True)
    monkeypatch.setattr(inf, "_mlx_load_estimate", lambda *a: (None, offloaded))
    assert inf._mlx_chat_need(None, mlx, "m", 4) is None


def test_chat_fit_kwargs(monkeypatch):
    policy = SimpleNamespace(value = "off")
    monkeypatch.setattr(gpu_residency, "current_policy", lambda: policy.value)
    monkeypatch.setattr("utils.hardware.hardware.get_parent_visible_gpu_ids", lambda: [])
    gguf, native = SimpleNamespace(is_gguf = True), SimpleNamespace(is_gguf = False)
    assert inf._chat_fit_kwargs(gguf, _placement([0])) == {}
    policy.value = "fit"
    token = inf._primary_can_retry_fit.set(True)
    try:
        assert inf._chat_fit_kwargs(gguf, _placement([0])) == {"policy": "fit", "gpu_ids": [0]}
        # Anything but GGUF has no fit estimate before launch, even inside the retry.
        assert inf._chat_fit_kwargs(native, _placement([0])) == {
            "policy": "fit",
            "gpu_ids": [0],
            "evict_first": True,
        }
    finally:
        inf._primary_can_retry_fit.reset(token)
    # Auto-switch calls the tracked impl without load_model_gated, so nothing would make room after.
    assert inf._chat_fit_kwargs(gguf, _placement([0])) == {
        "policy": "fit",
        "gpu_ids": [0],
        "evict_first": True,
    }
    native_need = inf._chat_fit_kwargs(native, _placement([0, 1], native_gb = 8.0))
    assert native_need == {"policy": "fit", "gpu_ids": [0, 1], "need_mib": {0: 4096, 1: 4096}}
    policy.value = "always"
    assert inf._chat_fit_kwargs(gguf, _placement([0])) == {"policy": "always", "gpu_ids": [0]}
    # No target GPUs (or Vulkan ordinals) leaves every GPU in play.
    assert inf._chat_fit_kwargs(gguf, _placement(None)) == {"policy": "always"}
    assert inf._chat_fit_kwargs(gguf, _placement([0], vulkan = True)) == {"policy": "always"}
    # Unified memory pages past the budget, so Always has no out-of-memory to refuse an unpriced load on.
    monkeypatch.setattr(inf, "_unified_metal_host", lambda: True)
    assert inf._chat_fit_kwargs(native, _placement(None)).get("evict_first") is True
    priced = inf._chat_fit_kwargs(native, _placement(None), mlx_need_mib = 700, mlx_credit_mib = 50)
    assert priced.get("need_mib") == {0: 700} and "evict_first" not in priced
    assert priced["need_mib"].credit_mib == 50


@pytest.mark.parametrize(
    "parent, physical, expected",
    [
        ([], [3], [3]),  # no mask: physical ids are the ordinals
        ([2, 3], [3], [1]),  # CUDA_VISIBLE_DEVICES=2,3
        ([2, 3], [3, 2], [1, 0]),
        ([2, 3], [5], None),  # not visible: every GPU counts
    ],
)
def test_visible_ordinals(monkeypatch, parent, physical, expected):
    monkeypatch.setattr("utils.hardware.hardware.get_parent_visible_gpu_ids", lambda: parent)
    assert inf._visible_ordinals(physical) == expected


def test_the_replace_fallback_never_refuses_a_partial_fit():
    # Nothing retries the fallback load, so a refusal beside media there would escape as a raw GpuMemoryShortError.
    import inspect

    source = inspect.getsource(inf.load_model_gated)
    unload = source.index("await asyncio.to_thread(_unload_primary)")
    assert (
        unload
        < source.index("_primary_can_retry_fit.set(False)", unload)
        < source.index("_run_tracked_load_model_impl", unload)
    )


def test_only_a_bounded_mlx_cache_is_priced_and_its_admitted_width_never_shrinks():
    assert inf._mlx_cache_bounded({"context_length_enforced": True})
    assert inf._mlx_cache_bounded({"context_length_enforced": False, "mlx_context_budget": 4096})
    assert not inf._mlx_cache_bounded({"context_length_enforced": None})
    batching = {"context_length_enforced": True, "context_unbounded_when_batched": True}
    # One slot never batches, so its window holds; more slots batch vision rows past it.
    assert inf._mlx_cache_bounded(batching) and not inf._mlx_cache_bounded(batching, 2)
    assert not inf._mlx_cache_bounded(
        {"context_length_enforced": False, "mlx_context_budget": None}
    )
    entry = {"context_length_enforced": True, "mlx_growth": {"mib": 1, "row_mib": 1, "slots": 1}}
    backend = SimpleNamespace(models = {"m": entry}, active_model_name = "m")
    for slots in (4, 1):
        inf._note_mlx_slots(backend, slots)
    assert entry["mlx_growth"]["slots"] == 4
    vision = {**batching, "mlx_growth": {"mib": 1, "row_mib": 1, "slots": 1}}
    inf._note_mlx_slots(SimpleNamespace(models = {"v": vision}, active_model_name = "v"), 2)
    assert "mlx_growth" not in vision
    inf._note_mlx_slots(SimpleNamespace(models = {}, active_model_name = None), 2)
    import inspect

    source = inspect.getsource(inf._load_model_impl)
    assert "_note_mlx_slots(backend, backend.set_parallel_slots(_n_parallel))" in source
    assert (
        '_mlx_cache_bounded(\n                _resident_entry, mlx_chat_need[1]["slots"]' in source
    )


def test_the_mlx_diffusers_fallback_forgets_the_mlx_price():
    import inspect
    source = inspect.getsource(inf._mlx_fallback_load_gate)
    assert (
        source.index("refuse_unless_fits(")
        < source.index("forget_media_price(DIFFUSION, cancel)")
        < source.index("return start()")
    )


def test_an_mlx_load_is_credited_what_it_replaces_and_more_rows_reprice_it():
    import inspect

    source = inspect.getsource(inf._load_model_impl)
    shortcut = source.index('logger.info(f"Model already loaded (Unsloth)')
    assert (
        "not _mlx_slots_outgrow_price(backend, _n_parallel)" in source[shortcut - 1200 : shortcut]
    )
    inherit = source.index("# A reload for more rows keeps the context the resident was asked for.")
    assert inherit < shortcut and '"max_seq_length_requested"' in source[inherit : inherit + 400]
    # A kept slot's reload frees its own worker too; an outgoing llama-server's footprint is not wired-only.
    credit = source.index(
        "mlx_credit_mib = await asyncio.to_thread(mlx_replaced_credit_mib, backend)"
    )
    assert "gguf_replaced_credit_mib" not in source
    assert credit < source.index("mlx_credit_mib = mlx_credit_mib", credit)
    # An unbounded cache admitted at a price settles after the load, undoing it when refused.
    settle = source.index("settle_unpriced_chat_beside_media, multi_model_mode")
    assert settle < source.index("backend.unload_model, config.identifier", settle)


def test_a_pinned_non_gguf_chat_load_keeps_its_requested_gpus(monkeypatch):
    # Only GGUF and native audio resolve their GPUs: a Transformers load pinned to GPU 1 still targets GPU 1.
    pinned = inf._LoadPlacement([1], None, False, None)
    monkeypatch.setattr(gpu_residency, "current_policy", lambda: "always")
    monkeypatch.setattr("utils.hardware.hardware.get_parent_visible_gpu_ids", lambda: [])
    monkeypatch.setattr(gpu_residency, "media_held_mib", lambda: {0: 4096})
    assert inf._chat_fit_kwargs(SimpleNamespace(is_gguf = False), pinned) == {
        "policy": "always",
        "gpu_ids": [1],
    }
    assert inf._media_on_target_gpus(pinned) is False


@pytest.mark.parametrize(
    ("serving", "policy", "expected"),
    [(False, "always", True), (True, "always", False), (True, "fit", True), (False, "fit", True)],
)
def test_a_serving_primary_reloaded_under_always_is_not_torn_down_for_a_refusal(
    monkeypatch, serving, policy, expected
):
    # Always only refuses; the partial-fit check runs after the old llama-server is killed.
    monkeypatch.setattr(gpu_residency, "media_held_mib", lambda: {0: 4096})
    monkeypatch.setattr("utils.hardware.hardware.get_parent_visible_gpu_ids", lambda: [])
    placement = inf._LoadPlacement([0], [0], False, None)
    token = inf._primary_can_retry_fit.set(True)
    try:
        assert inf._primary_refuses_partial_fit(serving, policy, placement) is expected
    finally:
        inf._primary_can_retry_fit.reset(token)


def test_always_loads_a_primary_that_spills_to_host_ram_without_refusing(env):
    env.policy = "always"
    env.errors = [GpuMemoryShortError("spills", host = True)]
    response, _ = _gated()
    assert response.status == "loaded" and env.requests[-1].force_alongside
    assert env.free_calls == []
    _nothing_unloaded(env)


@pytest.mark.parametrize(
    ("policy", "kept_on", "placement", "expected"),
    [
        ("fit", {0: 6000}, _placement([0]), True),
        ("fit", {0: 6000}, _placement(None), True),
        ("fit", {1: 6000}, _placement([0]), False),
        ("fit", None, _placement([0]), False),
        ("always", {0: 6000}, _placement([0]), False),
    ],
)
def test_a_primary_reload_under_fit_refuses_a_partial_fit_beside_an_idle_kept_chat_model(
    monkeypatch, policy, kept_on, placement, expected
):
    monkeypatch.setattr(gpu_residency, "media_held_mib", lambda: {})
    kept = model_slots.ExtraSlot(FakeLlama("org/B-GGUF"), FakeOrchestrator(), "owner")
    kept.llama._planned_vram_mib = kept_on or {}
    monkeypatch.setattr(model_slots, "slots", [kept] if kept_on else [])
    token = inf._primary_can_retry_fit.set(True)
    try:
        assert inf._primary_refuses_partial_fit(True, policy, placement) is expected
    finally:
        inf._primary_can_retry_fit.reset(token)


def test_an_extra_chat_load_frees_an_idle_image_before_replacing_the_primary(env, monkeypatch):
    # Chat A + an idle image under When they fit: chat B loaded alongside comes up short. The image goes and B
    # loads beside A, instead of A being unloaded (and the image after it, by the replacement primary load).
    extra = model_slots.ExtraSlot(FakeLlama(), FakeOrchestrator(), "owner")

    async def select(_request):
        return extra

    monkeypatch.setattr(inf, "_select_load_slot", select)
    monkeypatch.setattr(multi_model_settings, "get_multi_model_mode", lambda: "fit")
    monkeypatch.setattr(model_slots, "eviction_victims", lambda *a, **k: [])
    monkeypatch.setattr(inf, "_raise_or_cancel_active_generations", lambda **k: None)
    env.errors = [_short()]

    async def run():
        request = LoadRequest(model_path = "org/C-GGUF", alongside = True)
        return await inf.load_model_gated(request, None, "s")

    response = asyncio.run(run())
    assert env.unloads == []
    assert env.free_calls == [
        {"owner": gpu_arbiter.CHAT, "gpu_ids": (0,), "short_mib": 5000, "media_only": True}
    ]
    assert len(env.requests) == 2 and env.requests[1].alongside
    assert response.evicted == ["org/Img"]
