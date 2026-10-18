# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep multiple models loaded set to Always refuses instead of unloading, and every load
reports what it did unload."""

import asyncio
import json
import types

import pytest
from fastapi import HTTPException

import core.inference.diffusion_device as devmod
import core.inference.media_auto_switch as mas
import core.inference.model_slots as model_slots
import routes.inference as inf
from core.inference import gpu_arbiter
from core.inference.gpu_arbiter import MultiModelAlwaysRefusal
from core.inference.llama_cpp import GpuMemoryShortError, _gpu_short_message
from models.inference import InferenceStatusResponse, LoadRequest, LoadResponse
from utils import multi_model_settings

from .test_diffusion_routes import _post_load, client as image_client  # noqa: F401
from .test_media_auto_switch import _switch, backend, catalog, enabled, flux  # noqa: F401
from .test_multi_model_serving import FakeLlama, _gated_load_fakes, _slot, backends  # noqa: F401
from .test_openai_auto_switch import _FakeBackend, _wired
from .test_preview_routes import fake_slot, slot_state  # noqa: F401
from .test_tunnel_safe_long_post import _collect, route  # noqa: F401
from .test_video_routes import _shared_setup_3, client as video_client  # noqa: F401

REFUSAL = "multi-model-always"
_REAL_ENABLED = multi_model_settings.get_multi_model_enabled


def _mode(monkeypatch, mode):
    """Pin the setting; get_multi_model_enabled is the real one, derived from the mode."""
    multi_model_settings.invalidate_multi_model_cache()
    monkeypatch.setattr(multi_model_settings, "get_multi_model_mode", lambda: mode)
    monkeypatch.setattr(multi_model_settings, "get_multi_model_enabled", _REAL_ENABLED)


@pytest.fixture(autouse = True)
def arbiter(monkeypatch):
    for name, value in (("_owner", None), ("_owner_account", None), ("_prior_account", None)):
        monkeypatch.setattr(gpu_arbiter, name, value)
    monkeypatch.setattr(gpu_arbiter, "_residents", {})
    monkeypatch.setattr(gpu_arbiter, "_accounts", {})
    yield
    multi_model_settings.invalidate_multi_model_cache()


def _reporting(monkeypatch, owner, slots):
    """A resident ``owner`` whose eviction is recorded and reports ``slots``."""
    evicted = []
    monkeypatch.setitem(gpu_arbiter._SLOT_REPORTERS, owner, lambda: [dict(s) for s in slots])
    monkeypatch.setitem(gpu_arbiter._EVICTORS, owner, lambda: evicted.append(owner))
    return evicted


def _no_slot_eviction(monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("a kept model was evicted")

    monkeypatch.setattr(model_slots, "evict_these", never)
    monkeypatch.setattr(model_slots, "eviction_victims", never)
    monkeypatch.setattr(model_slots, "evict", never, raising = False)


def _is_refusal(exc):
    assert exc.status_code == 409
    assert exc.headers["X-Unsloth-Refusal"] == REFUSAL
    assert "Retry-After" not in exc.headers


# 1. A short fit under Always refuses; a capped one still retries beside the others.


def test_a_short_fit_under_always_refuses_and_keeps_every_slot(backends, monkeypatch):
    primary, extra = backends
    _mode(monkeypatch, "always")
    _no_slot_eviction(monkeypatch)
    monkeypatch.setattr(inf, "release_chat_gpu_claim", lambda: None)
    copy = _gpu_short_message(13.2, 8.4, 32768, 32768, False)
    assert "is set to Always" in copy

    async def load(request, *args, **kwargs):
        raise GpuMemoryShortError(copy, capped = False, short_mib = 5000)

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", load)
    request = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inf.load_model_gated(request, None, "s"))
    _is_refusal(excinfo.value)
    assert isinstance(excinfo.value.detail, str) and "is set to Always" in excinfo.value.detail
    assert model_slots.slots == [extra] and extra.llama.is_active and primary.is_active


def test_a_capped_fit_under_always_retries_beside_the_others(backends, monkeypatch):
    _, extra = backends
    _mode(monkeypatch, "always")
    _no_slot_eviction(monkeypatch)
    loaded = LoadResponse.model_construct(status = "loaded", model = "org/C-GGUF", display_name = "C")
    _gated_load_fakes(monkeypatch, short_fits = ["capped"], response = loaded)
    request = LoadRequest(model_path = "org/C-GGUF", alongside = True)
    response = asyncio.run(inf.load_model_gated(request, None, "s"))
    assert response.evicted == [] and response.evicted_slots == []
    assert model_slots.slots[0] is extra and extra.llama.is_active
    assert model_slots.slots[-1].request.force_alongside is True


# 2. Another quant of the serving repo is not a replacement.


def test_always_allows_another_quant_of_the_serving_repo_only(backends, monkeypatch):
    _mode(monkeypatch, "always")
    monkeypatch.setattr(inf, "_llama_cpp_backend", FakeLlama("org/X-GGUF", "Q4_K_M"))
    same = LoadRequest(model_path = "org/X-GGUF", gguf_variant = "Q8_0")
    asyncio.run(inf._guard_primary_replacement(same, ("org/X-GGUF:Q8_0", "org/X-GGUF")))
    other = LoadRequest(model_path = "org/Y-GGUF", gguf_variant = "Q8_0")
    with pytest.raises(MultiModelAlwaysRefusal) as excinfo:
        asyncio.run(inf._guard_primary_replacement(other, ("org/Y-GGUF:Q8_0", "org/Y-GGUF")))
    assert excinfo.value.kind == "chat" and "org/X-GGUF" in str(excinfo.value)
    assert "is set to Always" in str(excinfo.value)
    _mode(monkeypatch, "fit")
    asyncio.run(inf._guard_primary_replacement(other, ("org/Y-GGUF",)))


# 3. A CPU TTS load that would replace a chat model of another repo.


def test_a_cpu_tts_replacing_a_chat_model_is_refused_before_teardown(backends, monkeypatch):
    primary, extra = backends
    _mode(monkeypatch, "always")
    _no_slot_eviction(monkeypatch)
    monkeypatch.setattr(inf, "release_chat_gpu_claim", lambda: None)
    unloaded = []
    monkeypatch.setattr(primary, "unload_model", lambda: unloaded.append("primary"))
    gates = []
    monkeypatch.setattr(inf, "_raise_or_cancel_active_generations", lambda **k: gates.append(k))
    request = LoadRequest(model_path = "org/Higgs-TTS", audio_device = "cpu")
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inf.load_model_gated(request, None, "s"))
    _is_refusal(excinfo.value)
    assert "org/A-GGUF" in excinfo.value.detail
    assert unloaded == [] and gates == [] and primary.is_active and model_slots.slots == [extra]


def test_a_cpu_tts_alongside_under_always_takes_a_kept_slot(backends, monkeypatch):
    primary, extra = backends
    _mode(monkeypatch, "always")
    _no_slot_eviction(monkeypatch)
    loaded = LoadResponse.model_construct(status = "loaded", model = "org/Higgs-TTS", display_name = "T")
    _gated_load_fakes(monkeypatch, short_fits = [], response = loaded)
    request = LoadRequest(model_path = "org/Higgs-TTS", audio_device = "cpu", alongside = True)
    response = asyncio.run(inf.load_model_gated(request, None, "s"))
    assert response.evicted == [] and response.evicted_slots == []
    assert (
        model_slots.slots[0] is extra
        and model_slots.slots[-1].request.model_path == "org/Higgs-TTS"
    )
    assert primary.model_identifier == "org/A-GGUF" and primary.is_active


# 4. audio.cpp reports out of memory as a structured detail.


def test_a_structured_out_of_memory_still_makes_room_under_fit(backends, monkeypatch):
    primary, extra = backends
    extra.request = LoadRequest(model_path = "org/B-GGUF", gguf_variant = "Q8_0")
    _mode(monkeypatch, "fit")
    monkeypatch.setattr(inf, "release_chat_gpu_claim", lambda: None)
    oom = {"code": "out_of_memory", "message": "audio.cpp ran out of GPU memory"}
    failures = [HTTPException(503, detail = oom)]
    seen = []

    async def load(request, *args, **kwargs):
        seen.append(request.alongside)
        if failures:
            raise failures.pop()
        llama = inf.get_llama_cpp_backend()
        llama.model_identifier, llama.is_loaded, llama.is_active = request.model_path, True, True
        return LoadResponse.model_construct(status = "loaded", model = request.model_path)

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", load)
    request = LoadRequest(model_path = "org/Drama-TTS", alongside = True)
    response = asyncio.run(inf.load_model_gated(request, None, "s"))
    assert seen == [True, True] and not extra.llama.is_active
    assert response.evicted == ["org/B-GGUF:Q8_0"]
    assert response.evicted_slots == [{"kind": "chat", "model": "org/B-GGUF"}]
    assert [s.request.model_path for s in model_slots.slots] == ["org/Drama-TTS"]


@pytest.mark.parametrize("status", [500, 503, 507])
def test_a_structured_out_of_memory_counts_at_any_status(status):
    assert inf._ran_out_of_memory(
        HTTPException(status, detail = {"code": "out_of_memory", "message": "m"})
    )
    assert not inf._ran_out_of_memory(HTTPException(status, detail = {"code": "bad_request"}))
    assert inf._ran_out_of_memory(HTTPException(status, detail = "CUDA out of memory")) is (
        status == 500
    )


# 5. The image and video load routes.

_CHAT_SLOTS = [
    {"kind": "chat", "model": "org/A-GGUF", "account": None},
    {"kind": "chat", "model": "org/B-GGUF", "account": None},
]
_IMAGE = {"model_path": "unsloth/Z-Image-Turbo-GGUF", "gguf_filename": "q.gguf"}
_VIDEO = {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "ltx-2.3-distilled-Q4_K_M.gguf"}


def _image_on_gpu(monkeypatch):
    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cuda")
    )


def _expect_chat_evicted(body, evicted):
    assert body["evicted_slots"] == [
        {"kind": "chat", "model": "org/A-GGUF"},
        {"kind": "chat", "model": "org/B-GGUF"},
    ]
    assert body["evicted"] == ["org/A-GGUF", "org/B-GGUF"] and evicted == [gpu_arbiter.CHAT]


def test_the_image_load_under_always_refuses_with_the_header(image_client, monkeypatch):
    _image_on_gpu(monkeypatch)
    _mode(monkeypatch, "always")
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    evicted = _reporting(monkeypatch, gpu_arbiter.CHAT, _CHAT_SLOTS)
    resp = _post_load(image_client, **_IMAGE)
    assert resp.status_code == 409 and resp.headers["x-unsloth-refusal"] == REFUSAL
    assert "org/A-GGUF" in resp.json()["detail"] and "is set to Always" in resp.json()["detail"]
    assert evicted == [] and gpu_arbiter.holds(gpu_arbiter.CHAT)


def test_the_image_load_under_off_reports_every_chat_model_it_unloaded(image_client, monkeypatch):
    _image_on_gpu(monkeypatch)
    _mode(monkeypatch, "off")
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    evicted = _reporting(monkeypatch, gpu_arbiter.CHAT, _CHAT_SLOTS)
    resp = _post_load(image_client, **_IMAGE)
    assert resp.status_code == 200
    _expect_chat_evicted(resp.json(), evicted)
    status = image_client.get("/api/inference/images/status").json()
    assert status["evicted"] == [] and status["evicted_slots"] == []


def test_the_video_load_under_always_refuses_with_the_header(video_client, monkeypatch):
    _shared_setup_3(monkeypatch)
    _mode(monkeypatch, "always")
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    evicted = _reporting(monkeypatch, gpu_arbiter.CHAT, _CHAT_SLOTS)
    resp = video_client.post("/api/inference/video/load", json = _VIDEO)
    assert resp.status_code == 409 and resp.headers["x-unsloth-refusal"] == REFUSAL
    assert "is set to Always" in resp.json()["detail"]
    assert evicted == [] and gpu_arbiter.holds(gpu_arbiter.CHAT)


def test_the_video_load_under_off_reports_every_chat_model_it_unloaded(video_client, monkeypatch):
    _shared_setup_3(monkeypatch)
    _mode(monkeypatch, "off")
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    evicted = _reporting(monkeypatch, gpu_arbiter.CHAT, _CHAT_SLOTS)
    resp = video_client.post("/api/inference/video/load", json = _VIDEO)
    assert resp.status_code == 200
    _expect_chat_evicted(resp.json(), evicted)
    status = video_client.get("/api/inference/video/status").json()
    assert status["evicted"] == [] and status["evicted_slots"] == []


# 6. A chat load that frees an image model names it.


def test_a_chat_load_that_evicts_an_image_model_lists_it(backends, monkeypatch):
    _mode(monkeypatch, "off")
    monkeypatch.setattr(model_slots, "slots", [])
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.DIFFUSION)
    evicted = _reporting(
        monkeypatch,
        gpu_arbiter.DIFFUSION,
        [{"kind": "image", "model": "org/Flux", "account": None}],
    )
    monkeypatch.setattr(inf, "release_chat_gpu_claim", lambda: None)

    async def load(request, *args, **kwargs):
        # What _load_model_impl does once the load is known viable.
        await asyncio.to_thread(gpu_arbiter.acquire_for_request, gpu_arbiter.CHAT)
        return LoadResponse.model_construct(status = "loaded", model = request.model_path)

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", load)
    response = asyncio.run(inf.load_model_gated(LoadRequest(model_path = "org/C-GGUF"), None, "s"))
    assert response.evicted_slots == [{"kind": "image", "model": "org/Flux"}]
    assert response.evicted == ["org/Flux"] and evicted == [gpu_arbiter.DIFFUSION]
    assert gpu_arbiter.residents() == {gpu_arbiter.CHAT: None}


def test_idle_kept_models_unloaded_by_a_load_under_off_are_listed_in_both_fields(
    backends, monkeypatch
):
    _mode(monkeypatch, "off")

    async def load(request, *args, **kwargs):
        return LoadResponse.model_construct(status = "loaded", model = request.model_path)

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", load)
    response = asyncio.run(inf.load_model_gated(LoadRequest(model_path = "org/C-GGUF"), None, "s"))
    assert model_slots.slots == []
    assert response.evicted_slots == [{"kind": "chat", "model": "org/B-GGUF"}]
    assert response.evicted == ["org/B-GGUF"]


def test_kept_models_a_vllm_load_retires_are_listed_in_both_fields(backends, monkeypatch):
    _mode(monkeypatch, "fit")
    monkeypatch.setattr(inf, "_gate_kept_models", lambda *args, **kwargs: None)

    async def load(request, *args, **kwargs):
        # What _load_model_impl does at the point of no return for a vLLM or SGLang load.
        await inf._retire_kept_models(request)
        return LoadResponse.model_construct(status = "loaded", model = request.model_path)

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", load)
    request = LoadRequest(model_path = "org/C", engine = "vllm")
    response = asyncio.run(inf.load_model_gated(request, None, "s"))
    assert model_slots.slots == []
    assert response.evicted_slots == [{"kind": "chat", "model": "org/B-GGUF"}]
    assert response.evicted == ["org/B-GGUF"]


def test_an_ordinary_switch_is_listed_in_evicted_slots_but_not_in_the_toast_list(
    backends, monkeypatch
):
    _mode(monkeypatch, "fit")
    monkeypatch.setattr(model_slots, "slots", [])
    monkeypatch.setattr(
        inf, "_primary_chat_reports", lambda: [{"kind": "chat", "model": "org/A", "account": None}]
    )
    monkeypatch.setattr(inf, "_loaded_satisfies", lambda ref: ref == "org/A")

    async def load(request, *args, **kwargs):
        # Twice, as a retried attempt does; the switch is listed once.
        for _ in range(2):
            await inf._guard_primary_replacement(request, (request.model_path,))
        return LoadResponse.model_construct(status = "loaded", model = request.model_path)

    monkeypatch.setattr(inf, "_run_tracked_load_model_impl", load)
    response = asyncio.run(inf.load_model_gated(LoadRequest(model_path = "org/C"), None, "s"))
    assert response.evicted_slots == [{"kind": "chat", "model": "org/A"}]
    assert not response.evicted
    same = asyncio.run(inf.load_model_gated(LoadRequest(model_path = "org/A"), None, "s"))
    assert same.evicted_slots == []


def test_a_vllm_load_under_always_refuses_while_a_kept_slot_is_stuck(backends, monkeypatch):
    _mode(monkeypatch, "always")
    stuck = object()
    monkeypatch.setattr(model_slots, "slots", [])
    monkeypatch.setattr(model_slots, "stuck", [stuck])
    monkeypatch.setattr(
        inf,
        "_kept_slot_reports",
        lambda slot: [{"kind": "chat", "model": "org/Stuck", "account": None}],
    )
    with pytest.raises(MultiModelAlwaysRefusal) as excinfo:
        asyncio.run(
            inf._guard_primary_replacement(LoadRequest(model_path = "org/C", engine = "vllm"), ())
        )
    assert excinfo.value.model == "org/Stuck"


def test_an_engine_switch_under_always_rechecks_before_it_unloads(monkeypatch):
    import core.inference.diffusion_engine_router as router

    unloaded = []
    old = types.SimpleNamespace(unload = lambda: unloaded.append("old"))
    monkeypatch.setattr(router, "_active_engine_name", router.ENGINE_DIFFUSERS)
    monkeypatch.setattr(router, "get_active_diffusion_engine", lambda: old)

    def landed_meanwhile():
        raise gpu_arbiter.refusal_for([{"kind": "image", "model": "org/Flux"}], "image")

    with pytest.raises(MultiModelAlwaysRefusal):
        inf._activate_checked(
            landed_meanwhile, lambda: router._activate(router.ENGINE_SD_CPP, None)
        )
    assert unloaded == [] and router._active_engine_name == router.ENGINE_DIFFUSERS
    assert router.before_engine_switch.get() is None


def _loaded_media(monkeypatch, owner, kind, repo):
    monkeypatch.setattr(
        gpu_arbiter,
        "_media_status",
        lambda o: {"loaded": True, "repo_id": repo} if o == owner else None,
    )
    return _reporting(monkeypatch, owner, [{"kind": kind, "model": repo, "account": None}])


def test_the_image_load_under_always_refuses_to_replace_another_image_model(
    image_client, monkeypatch
):
    _image_on_gpu(monkeypatch)
    _mode(monkeypatch, "always")
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.DIFFUSION)
    evicted = _loaded_media(monkeypatch, gpu_arbiter.DIFFUSION, "image", "org/Flux")
    resp = _post_load(image_client, **_IMAGE)
    assert resp.status_code == 409 and resp.headers["x-unsloth-refusal"] == REFUSAL
    assert "the image model org/Flux" in resp.json()["detail"] and evicted == []


def test_the_video_load_under_always_refuses_to_replace_another_video_model(
    video_client, monkeypatch
):
    _shared_setup_3(monkeypatch)
    _mode(monkeypatch, "always")
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.VIDEO)
    evicted = _loaded_media(monkeypatch, gpu_arbiter.VIDEO, "video", "org/Wan")
    resp = video_client.post("/api/inference/video/load", json = _VIDEO)
    assert resp.status_code == 409 and resp.headers["x-unsloth-refusal"] == REFUSAL
    assert "the video model org/Wan" in resp.json()["detail"] and evicted == []


def test_an_image_load_under_fit_reports_the_image_model_it_replaces(image_client, monkeypatch):
    _image_on_gpu(monkeypatch)
    _mode(monkeypatch, "fit")
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.DIFFUSION)
    _loaded_media(monkeypatch, gpu_arbiter.DIFFUSION, "image", "org/Flux")
    resp = _post_load(image_client, **_IMAGE)
    assert resp.status_code == 200
    assert resp.json()["evicted_slots"] == [{"kind": "image", "model": "org/Flux"}]
    assert resp.json()["evicted"] == ["org/Flux"]


@pytest.mark.parametrize("policy", ["off", "always"])
def test_an_off_torch_image_load_reports_or_refuses_the_native_video_model(monkeypatch, policy):
    import core.inference.video as video_mod

    unloaded = []
    video = types.SimpleNamespace(
        _loading = None, status = lambda: {"engine": "sd_cpp"}, unload = lambda: unloaded.append(1)
    )
    monkeypatch.setattr(video_mod, "get_video_backend", lambda: video)
    _reporting(
        monkeypatch, gpu_arbiter.VIDEO, [{"kind": "video", "model": "org/H3", "account": None}]
    )
    evicted = []
    if policy == "always":
        with pytest.raises(MultiModelAlwaysRefusal):
            inf._unload_native_video_sharing_the_sd_cpp_tree(policy, evicted)
        assert unloaded == [] and evicted == []
    else:
        inf._unload_native_video_sharing_the_sd_cpp_tree(policy, evicted)
        assert unloaded == [1] and evicted == [{"kind": "video", "model": "org/H3"}]


# 7. Auto-switch and the public preview.


def test_the_openai_auto_switch_under_always_answers_an_openai_409(monkeypatch):
    backend, _ = _wired(
        monkeypatch, _FakeBackend("unsloth/A-GGUF"), ("unsloth/B-GGUF", None, "unsloth/B-GGUF")
    )
    _mode(monkeypatch, "always")
    calls = []

    async def impl(
        request,
        fastapi_request,
        current_subject = None,
        **kwargs,
    ):
        # The real impl's first check on a replacing load, before any teardown.
        calls.append(request.model_path)
        await inf._guard_primary_replacement(request, (request.model_path,))
        raise AssertionError("the switch went past the refusal")

    monkeypatch.setattr(inf, "_load_model_impl", impl)
    request = types.SimpleNamespace(
        state = types.SimpleNamespace(),
        url = types.SimpleNamespace(path = "/v1/chat/completions"),
        method = "POST",
        headers = {},
    )
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inf._maybe_auto_switch_model("unsloth/B-GGUF", request, "tester"))
    _is_refusal(excinfo.value)
    error = excinfo.value.detail["error"]
    assert error["code"] == "multi_model_always" and "unsloth/A-GGUF" in error["message"]
    assert calls == ["unsloth/B-GGUF"]
    assert backend.model_identifier == "unsloth/A-GGUF" and backend.is_loaded


def test_the_media_auto_switch_under_always_answers_an_openai_409(
    flux, enabled, backend, monkeypatch
):
    backend.repo_id = "Qwen/Qwen-Image"
    refusal = gpu_arbiter.refusal_for(_CHAT_SLOTS)

    async def start(
        owner,
        pick,
        current_subject,
        hf_token = None,
    ):
        raise refusal.as_http_exception()

    monkeypatch.setattr(mas, "_start_load", start)
    with pytest.raises(HTTPException) as excinfo:
        _switch("black-forest-labs/FLUX.1-dev")
    _is_refusal(excinfo.value)
    error = excinfo.value.detail["error"]
    assert error["code"] == "multi_model_always" and error["message"] == str(refusal)
    assert backend.repo_id == "Qwen/Qwen-Image"


def test_a_preview_refused_under_always_gets_the_preview_503(fake_slot, monkeypatch):
    async def refused(*args, **kwargs):
        raise gpu_arbiter.refusal_for(_CHAT_SLOTS)

    monkeypatch.setattr(inf, "_load_model_impl", refused)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            inf.load_model_for_preview(
                LoadRequest(model_path = "/outputs/run/ckpt-a"),
                types.SimpleNamespace(app = None, scope = {"path": "/p/a/v1/chat/completions"}),
                "admin",
            )
        )
    assert excinfo.value.status_code == 503 and excinfo.value.headers.get("Retry-After")
    assert excinfo.value.detail.startswith("Unsloth is using the GPU")
    assert inf._get_preview_resident() is None


# 8. A refusal found after the padded reply committed keeps its header in the body.


def test_a_padded_refusal_keeps_its_header(route):
    async def slow_refusal():
        await asyncio.sleep(0.2)
        raise gpu_arbiter.refusal_for(_CHAT_SLOTS).as_http_exception()

    async def run():
        return await _collect(await route._tunnel_safe_json(slow_refusal(), label = "t"))

    deferred = json.loads(b"".join(asyncio.run(run())))["_deferred_error"]
    assert deferred["status_code"] == 409 and "is set to Always" in deferred["detail"]
    assert deferred["headers"] == {"x-unsloth-refusal": REFUSAL}


def test_a_deferred_error_keeps_only_the_headers_a_client_reads():
    headers = {"X-Unsloth-Refusal": REFUSAL, "Retry-After": "5", "X-Other": "1"}
    body = json.loads(inf._deferred_error_body(409, "no", headers))["_deferred_error"]
    assert body["headers"] == {"x-unsloth-refusal": REFUSAL, "retry-after": "5"}
    plain = json.loads(inf._deferred_error_body(500, "no", {"X-Other": "1"}))["_deferred_error"]
    assert plain == {"status_code": 500, "detail": "no"}


# 9. Status reports the mode.


@pytest.mark.parametrize("mode", ["off", "fit", "always"])
def test_status_reports_the_multi_model_mode(backends, monkeypatch, mode):
    _mode(monkeypatch, mode)

    async def slot_status(subject):
        return InferenceStatusResponse(active_model = "org/A-GGUF")

    monkeypatch.setattr(inf, "_slot_status", slot_status)
    assert asyncio.run(inf.get_status("s")).multi_model_mode == mode


def test_an_unreadable_primary_under_always_refuses_instead_of_replacing(backends, monkeypatch):
    from core.inference.gpu_arbiter import MultiModelAlwaysRefusal

    _mode(monkeypatch, "always")
    monkeypatch.setattr(model_slots, "slots", [])

    def broken():
        raise RuntimeError("status unreadable")

    monkeypatch.setattr(inf, "_primary_chat_reports", broken)
    with pytest.raises(MultiModelAlwaysRefusal):
        asyncio.run(inf._guard_primary_replacement(LoadRequest(model_path = "org/C"), ("org/C",)))
