# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The chat KV kept across a media swap: saved before the acquire, promoted to the reload stash
only when _evict_chat unloads that model, and bounded so a slow save never stalls the load POST.

The llama backend, the evictors and the settings are fakes: no llama-server, no GPU."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

import pytest

import core.inference.chat_swap_kv as cs
import core.inference.gpu_arbiter as arb
import routes.inference as inference_routes
from core.inference import llama_keepwarm as kw
from utils import openai_auto_switch_settings as settings
from utils.account_context import current_account_id

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_diffusion_routes import client  # noqa: E402,F401


class _FakeLlama:
    def __init__(self, tmp_path):
        self.dir = tmp_path
        self.gguf = tmp_path / "model.gguf"
        self.gguf.write_bytes(b"gguf")
        self._slot_save_binary = ("/bin/llama-server", 111)
        self.saves: list[bool] = []
        self.restores: list[dict] = []
        self.files: list = []
        self.gate = None
        self.entered = threading.Event()
        self.reload()

    def reload(self, gguf = None):
        self.is_loaded = self.is_active = True
        self.model_identifier, self.hf_variant = "unsloth/A-GGUF", "Q4_K_M"
        self._gguf_path = str(gguf or self.gguf)

    def _gguf_file_identity(self, path):
        st = os.stat(path)
        return ((st.st_size, st.st_mtime_ns),)

    def _slot_launch_fingerprint(self):
        return ((), None, None, 1)

    def save_slots_for_resume(self, should_abort):
        self.saves.append(arb._lock.locked())
        gguf = self._gguf_path  # the eviction may unload it while a slow save runs
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(10)
        slot = self.dir / f"resume-{len(self.saves)}-slot0.bin"
        slot.write_bytes(b"kv")
        self.files.append(slot)
        return {
            "dir": str(self.dir),
            "binary": self._slot_save_binary,
            "gguf": gguf,
            "gguf_stat": self._gguf_file_identity(gguf),
            "launch": self._slot_launch_fingerprint(),
            "slots": [{"id": 0, "filename": slot.name, "n_saved": 1}],
        }

    def restore_slots_for_resume(self, manifest):
        self.restores.append(manifest)

    def unload_model(self):
        self.is_loaded = self.is_active = False
        self.model_identifier = self._gguf_path = None


IDENTITY = ("unsloth/A-GGUF", "Q4_K_M", "unsloth/A-GGUF")


@pytest.fixture
def env(monkeypatch, tmp_path):
    llama = _FakeLlama(tmp_path)
    llama.calls = []
    llama.rows = []

    def _evict_chat():
        llama.calls.append(("evict-chat", list(llama.saves)))
        unloaded = arb._chat_identity(llama)
        llama.unload_model()
        arb._commit_swap_kv(unloaded)

    monkeypatch.setattr(arb, "_owner", arb.CHAT)
    monkeypatch.setattr(arb, "_owner_account", current_account_id())
    for owner in (arb.DIFFUSION, arb.VIDEO):
        monkeypatch.setitem(arb._EVICTORS, owner, lambda: None)
    monkeypatch.setitem(arb._EVICTORS, arb.CHAT, _evict_chat)
    monkeypatch.setattr(kw, "_kv_resume", None)
    monkeypatch.setattr(cs, "_pending", None)
    monkeypatch.setattr(cs, "_committed", None)
    monkeypatch.setattr(cs, "_saving", {})
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: True)
    monkeypatch.setattr(inference_routes, "get_llama_cpp_backend", lambda: llama)
    from core.inference.api_monitor import api_monitor

    monkeypatch.setattr(api_monitor, "record_lifecycle", lambda **row: llama.rows.append(row))
    return llama


def _load(
    owner = arb.DIFFUSION,
    media = "image",
    acquire = None,
    **fit,
):
    async def _go():
        swap = await cs.save_chat_kv_before_swap(owner, media, **fit)
        await cs.acquire_then_close(swap, acquire or (lambda: arb.acquire_for(owner)))

    asyncio.run(_go())


def _no_files(llama) -> bool:
    return not any(f.exists() for f in llama.files)


@pytest.mark.parametrize(
    "policy, fits", [("off", None), ("fit", True), ("fit", False), ("always", None)]
)
@pytest.mark.parametrize("current", [None, arb.CHAT, arb.DIFFUSION, arb.VIDEO])
@pytest.mark.parametrize("target", [arb.DIFFUSION, arb.VIDEO, arb.CHAT])
def test_would_evict_matches_acquire_for(monkeypatch, policy, fits, current, target):
    calls: list[str] = []
    monkeypatch.setattr(arb, "_owner", current)
    for owner in (arb.CHAT, arb.DIFFUSION, arb.VIDEO):
        monkeypatch.setitem(arb._EVICTORS, owner, lambda o = owner: calls.append(f"evict-{o}"))
    if fits is not None:
        try:
            from core.inference import gpu_residency
        except ImportError:
            gpu_residency = None
        if hasattr(gpu_residency, "can_load_alongside"):
            monkeypatch.setattr(gpu_residency, "can_load_alongside", lambda *a, **k: (fits, {}))
    need = {0: 1024} if fits is not None else None
    predicted = arb.would_evict(target, policy = policy, need_mib = need)
    kwargs = {"policy": policy}
    if need:
        kwargs["need_mib"] = need
    try:
        arb.acquire_for(target, **kwargs)
    except RuntimeError:
        pass  # a refusal evicts nothing
    assert predicted == ("evict-chat" in calls)


@pytest.mark.parametrize("policy", ["fit", "always"])
@pytest.mark.parametrize("gpu, evicts", [(1, False), (0, True)])
def test_would_evict_honours_the_target_gpu_of_an_unpriced_load(monkeypatch, policy, gpu, evicts):
    # media_fit_kwargs passes gpu_ids even when the load can't be priced; chat on another GPU stays.
    from core.inference import gpu_residency as gr

    calls: list[str] = []
    monkeypatch.setattr(arb, "_owner", arb.CHAT)
    monkeypatch.setattr(arb, "_residents", {arb.CHAT: frozenset({0})})
    monkeypatch.setattr(
        arb, "physical_gpu_ids", lambda ids: None if ids is None else frozenset(ids)
    )
    monkeypatch.setattr(gr, "_claim_holds_anything", lambda name: False)
    for owner in (arb.CHAT, arb.DIFFUSION, arb.VIDEO):
        monkeypatch.setitem(arb._EVICTORS, owner, lambda o = owner: calls.append(f"evict-{o}"))
    predicted = arb.would_evict(arb.DIFFUSION, policy = policy, gpu_ids = [gpu])
    try:
        arb.acquire_for(arb.DIFFUSION, policy = policy, gpu_ids = [gpu])
    except RuntimeError:
        pass
    assert predicted is evicts
    assert predicted == ("evict-chat" in calls)


_ELSEWHERE = {arb.CHAT: frozenset({1})}
_SHARED = {arb.CHAT: frozenset({0}), arb.VIDEO: frozenset({0})}


@pytest.mark.parametrize(
    "residents, held, busy, loading, evicts",
    [
        (_ELSEWHERE, 0, False, True, False),  # chat on other GPUs is never touched
        (_SHARED, 1600, False, False, False),  # the idle video model frees room, with margin
        (_SHARED, 600, False, False, True),  # freed exactly, no margin: save anyway
        (_SHARED, 100, False, False, True),
        (_SHARED, 1600, True, False, True),  # a generating video model stays
        (_SHARED, 1600, False, True, True),  # coreside_locked evicts a chat still loading
    ],
)
def test_would_evict_under_fit_follows_the_free_order(
    monkeypatch, residents, held, busy, loading, evicts
):
    from core.inference import gpu_residency as gr

    monkeypatch.setattr(arb, "_owner", arb.CHAT)
    monkeypatch.setattr(arb, "_residents", residents)
    short = {"reason": "short", "short_mib": {0: 500}}
    monkeypatch.setattr(gr, "can_load_alongside", lambda *a, **k: (False, dict(short)))
    monkeypatch.setattr(gr, "_media_records", lambda: {arb.VIDEO: {0: held}} if held else {})
    monkeypatch.setattr(gr, "_media_busy", lambda name: (False, busy))
    monkeypatch.setattr(gr, "foreign_on", lambda *a: 0)
    monkeypatch.setattr(gr, "chat_load_in_flight", lambda: loading)
    assert arb.would_evict(arb.DIFFUSION, policy = "fit", need_mib = {0: 4000}) is evicts


@pytest.mark.parametrize("other_chat_requests, saved", [(0, True), (1, False)])
def test_a_v1_media_switch_does_not_count_itself_as_chat_work(
    env, monkeypatch, other_chat_requests, saved
):
    monkeypatch.setattr(kw, "_inflight", 1 + other_chat_requests)  # the /v1 call itself is counted
    monkeypatch.setattr(kw, "_pending", 0)
    token = kw._current_response_scope.set({kw._TRACKED_SCOPE_KEY: True})
    try:
        _load()
    finally:
        kw._current_response_scope.reset(token)
    assert (env.saves == [False]) is saved and (kw._kv_resume is not None) is saved


def test_keep_kv_turned_off_during_the_save_drops_it(env, monkeypatch):
    env.gate = threading.Event()

    def _turn_off_then_finish():
        env.entered.wait(10)
        monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: False)
        env.gate.set()

    threading.Thread(target = _turn_off_then_finish).start()
    _load()
    assert env.saves == [False] and kw._kv_resume is None and _no_files(env)


def test_an_off_route_saves_even_if_the_mode_changes_before_the_peek(env, monkeypatch):
    from utils import multi_model_settings

    monkeypatch.setattr(multi_model_settings, "get_multi_model_mode", lambda: "always")
    _load()  # media_fit_kwargs gave {} under Off, so the acquire runs as Off
    assert env.saves == [False] and kw._kv_resume["origin"] == "swap"


def test_a_cancelled_load_keeps_the_save_until_its_acquire_thread_ends(env):
    release = threading.Event()

    def acquire():
        release.wait(10)
        arb.acquire_for(arb.DIFFUSION)

    async def _go():
        swap = await cs.save_chat_kv_before_swap(arb.DIFFUSION, "image")
        request = asyncio.ensure_future(cs.acquire_then_close(swap, acquire))
        await asyncio.sleep(0.05)
        request.cancel()  # the client went away; the acquire thread runs on
        with pytest.raises(asyncio.CancelledError):
            await request
        release.set()
        for _ in range(500):
            if env.rows:
                break
            await asyncio.sleep(0.01)

    asyncio.run(_go())
    assert kw._kv_resume["origin"] == "swap" and env.files[0].exists()
    assert [(r["event"], r["reason"]) for r in env.rows] == [("unload", "swap")]


def test_a_fit_swap_leaves_the_eviction_row_to_the_residency_code(env, monkeypatch):
    monkeypatch.setattr(arb, "would_evict", lambda *a, **k: True)
    _load(policy = "fit")  # When they fit logs the eviction itself, as "evicted"
    assert kw._kv_resume["origin"] == "swap" and env.rows == []


@pytest.mark.parametrize("owner, media", [(arb.DIFFUSION, "image"), (arb.VIDEO, "video")])
def test_the_save_runs_before_the_acquire_and_commits_on_the_eviction(env, owner, media):
    _load(owner, media)
    assert env.saves == [False]  # saved with the arbiter lock free
    assert env.calls == [("evict-chat", [False])]  # the eviction ran after the save returned
    stash = kw._kv_resume
    assert stash["origin"] == "swap" and stash["identity"] == IDENTITY
    assert env.files[0].exists() and cs._pending is None
    assert [(r["event"], r["reason"]) for r in env.rows] == [("unload", "swap")]


def test_the_real_evict_chat_promotes_the_identity_read_before_unload(env, monkeypatch, tmp_path):
    import core.inference as inference_pkg
    import core.inference.llama_cpp as llama_cpp_mod
    import core.inference.model_slots as model_slots
    from hub.services.models import account_access

    class _Orchestrator:
        active_model_name = None
        loading_models = ()
        _managed_engine = None

        def _shutdown_subprocess(self, timeout):
            return True

    env._wait_for_vram_settle = lambda **k: None
    monkeypatch.setattr(model_slots, "unload_extra_models", lambda **k: None)
    monkeypatch.setattr(inference_pkg, "get_inference_backend", lambda: _Orchestrator())
    monkeypatch.setattr(llama_cpp_mod, "chat_load_active", lambda: False)
    monkeypatch.setattr(account_access, "clear_resident", lambda owner: None)
    monkeypatch.setitem(arb._EVICTORS, arb.CHAT, arb._evict_chat)
    _load()
    assert env.is_loaded is False
    assert kw._kv_resume["identity"] == IDENTITY and env.files[0].exists()


class _FakeMlxOrchestrator:
    loading_models = ()
    _managed_engine = None
    _openai_advertised_id = None

    def __init__(self, tmp_path):
        self.root, self.active_model_name, self.restores = tmp_path, "unsloth/A-MLX", []

    def _serves_mlx(self):
        return self.active_model_name is not None

    def save_prompt_cache_for_resume(self, should_abort):
        directory = self.root / "resume-mlx"
        directory.mkdir()
        (directory / "text-0.safetensors").write_bytes(b"kv")
        slot = {"filename": "text-0.safetensors", "store": "text"}
        return {"engine": "mlx", "dir": str(directory), "fingerprint": (), "slots": [slot]}

    def restore_prompt_cache_for_resume(self, manifest):
        self.restores.append(manifest)
        return 1

    def unload_model(self, name):
        self.active_model_name = None

    def _shutdown_subprocess(self, timeout):
        return True


@pytest.mark.parametrize("reload", ["mlx", "gguf"])
def test_an_mlx_chat_keeps_its_prompt_cache_across_the_swap(env, monkeypatch, tmp_path, reload):
    import core.inference as inference_pkg
    import core.inference.llama_cpp as llama_cpp_mod
    import core.inference.model_slots as model_slots
    import core.inference.orchestrator as orchestrator_mod
    from hub.services.models import account_access

    orch = _FakeMlxOrchestrator(tmp_path)
    env.unload_model()
    env._wait_for_vram_settle = lambda **k: None
    monkeypatch.setattr(model_slots, "unload_extra_models", lambda **k: None)
    monkeypatch.setattr(inference_pkg, "get_inference_backend", lambda: orch)
    monkeypatch.setattr(orchestrator_mod, "peek_inference_backend", lambda: orch)
    monkeypatch.setattr(llama_cpp_mod, "chat_load_active", lambda: False)
    monkeypatch.setattr(account_access, "clear_resident", lambda owner: None)
    monkeypatch.setitem(arb._EVICTORS, arb.CHAT, arb._evict_chat)
    _load()
    stash = kw._kv_resume
    assert stash["identity"] == ("unsloth/A-MLX", None, "unsloth/A-MLX", "mlx")
    assert orch.active_model_name is None and Path(stash["dir"]).is_dir()
    assert [(r["event"], r["reason"]) for r in env.rows] == [("unload", "swap")]

    if reload == "mlx":
        orch.active_model_name = "unsloth/A-MLX"
        kw.note_model_loaded(orch)
    else:
        env.reload()
        kw.note_model_loaded(env)
    assert orch.restores == ([stash] if reload == "mlx" else []) and env.restores == []
    assert kw._kv_resume is None and not Path(stash["dir"]).exists()


@pytest.mark.parametrize(
    "case",
    ["generating", "keep_kv_off", "chat_not_resident", "not_gguf", "other_account", "chat_target"],
)
def test_no_save_without_a_chat_eviction_to_survive(env, monkeypatch, case):
    owner = arb.DIFFUSION
    if case == "generating":
        from state import active_generations
        monkeypatch.setattr(active_generations, "count", lambda *a, **k: 1)
    elif case == "keep_kv_off":
        monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: False)
    elif case == "chat_not_resident":
        monkeypatch.setattr(arb, "_owner", None)
    elif case == "not_gguf":
        env._gguf_path = None
    elif case == "other_account":
        monkeypatch.setattr(arb, "_owner_account", "someone-else")
    else:
        owner = arb.CHAT  # would_evict is False
    _load(owner)
    assert env.saves == [] and kw._kv_resume is None and cs._pending is None
    assert cs.saving_progress("image") is None


@pytest.mark.parametrize("same_model", [True, False])
def test_a_chat_reload_restores_the_swap_stash_only_through_the_guard(env, tmp_path, same_model):
    _load()
    other = tmp_path / "other.gguf"
    other.write_bytes(b"other")
    env.reload(None if same_model else other)
    kw.note_model_loaded(env)
    assert len(env.restores) == (1 if same_model else 0)
    assert kw._kv_resume is None and _no_files(env)


@pytest.mark.parametrize("case", ["no_commit", "acquire_raises", "other_model"])
def test_the_pending_save_is_dropped_when_chat_was_not_evicted(env, monkeypatch, case):
    acquire = None
    if case == "no_commit":
        monkeypatch.setitem(arb._EVICTORS, arb.CHAT, lambda: None)
    elif case == "other_model":
        monkeypatch.setitem(arb._EVICTORS, arb.CHAT, lambda: arb._commit_swap_kv(("x", None, "x")))
    else:

        def acquire():
            raise arb.GpuBusyForAnotherAccountError(arb.CHAT, 1)

    try:
        _load(acquire = acquire)
    except arb.GpuBusyForAnotherAccountError:
        assert case == "acquire_raises"
    assert env.saves == [False]
    assert cs._pending is None and kw._kv_resume is None and _no_files(env)
    assert env.rows == []


def test_the_swap_stash_survives_a_media_ttl_save_but_not_keep_kv_off(env, monkeypatch):
    import routes.settings as settings_route

    _load()
    stash = kw._kv_resume
    monkeypatch.setattr(settings_route, "idle_unload_is_configured", lambda: False)
    for keep_kv, kept in ((True, True), (False, False)):
        monkeypatch.setattr(
            settings_route,
            "set_openai_auto_switch",
            lambda *a, keep_kv = keep_kv: (True, 0, keep_kv, False, False, 300, False),
        )
        payload = settings_route.OpenAIAutoSwitchPayload(media_auto_unload_idle_seconds = 300)
        settings_route.update_openai_auto_switch(payload, "tester")
        assert (kw._kv_resume is stash) is kept and env.files[0].exists() is kept


def test_saving_progress_is_the_callers_own():
    with cs._saving_marker("image", "someone-else"):
        assert cs.saving_progress("image") is None
    with cs._saving_marker("image", current_account_id()):
        assert cs.saving_progress("image") == {"phase": cs.SAVING_CHAT_CONTEXT}
        assert cs.saving_progress("video") is None


def _video_client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import core.inference.video as video_module
    from auth.authentication import authenticated_via_api_key, get_current_subject
    from routes.video import router as video_router
    from test_video_routes import _FakeBackend

    backend = _FakeBackend()
    monkeypatch.setattr(video_module, "get_video_backend", lambda: backend)
    app = FastAPI()
    app.include_router(video_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return TestClient(app)


def _cuda(monkeypatch, device = "cuda"):
    import types

    import core.inference.diffusion_device as devmod
    monkeypatch.setattr(
        devmod, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = device)
    )


_LOADS = {
    "image": ("/api/inference/images", {"model_path": "x/z-image", "gguf_filename": "q.gguf"}),
    "video": (
        "/api/inference/video",
        {"model_path": "unsloth/LTX-2.3-GGUF", "gguf_filename": "ltx-2.3-distilled-Q4_K_M.gguf"},
    ),
}


@pytest.mark.parametrize("media", ["image", "video"])
def test_a_slow_save_is_bounded_and_its_late_files_are_deleted(
    client,
    env,
    monkeypatch,
    healthy_diffusers,
    media,  # noqa: F811
):
    _cuda(monkeypatch)
    if media == "video":
        client = _video_client(monkeypatch)
    base, body = _LOADS[media]
    monkeypatch.setattr(cs, "SWAP_KV_SAVE_BUDGET_S", 0.2)
    env.gate = threading.Event()
    answer: dict = {}

    def _post():
        answer["resp"] = client.post(f"{base}/load", json = body)

    poster = threading.Thread(target = _post)
    poster.start()
    try:
        assert env.entered.wait(5)
        during = client.get(f"{base}/load-progress").json()
        poster.join(10)
        # The POST answered while the save was still blocked: the budget, not the save, ended it.
        returned_while_blocked = not poster.is_alive()
    finally:
        env.gate.set()
    poster.join(10)
    assert during["phase"] == "saving_chat_context"
    assert returned_while_blocked and answer["resp"].status_code == 200
    assert [c[0] for c in env.calls] == ["evict-chat"]  # the load went on as before
    assert client.get(f"{base}/load-progress").json()["phase"] != "saving_chat_context"
    deadline = time.monotonic() + 5
    while not (env.files and _no_files(env)) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert env.files and _no_files(env)
    assert kw._kv_resume is None and cs._pending is None


@pytest.mark.parametrize("case", ["keep_kv_off", "cpu"])
def test_load_progress_is_unchanged_without_a_save(client, env, monkeypatch, case):  # noqa: F811
    _cuda(monkeypatch, "cpu" if case == "cpu" else "cuda")
    if case == "keep_kv_off":
        monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: False)
    base, body = _LOADS["image"]
    assert client.post(f"{base}/load", json = body).status_code == 200
    assert env.saves == [] and kw._kv_resume is None
    assert client.get(f"{base}/load-progress").json() == {
        "phase": "ready",
        "bytes_downloaded": 0,
        "bytes_total": 0,
        "fraction": 1.0,
        "error": None,
        "error_code": None,
    }


@pytest.mark.parametrize("saved", [True, False])
@pytest.mark.parametrize("raises", [False, True])
def test_the_saving_phase_lasts_until_the_acquire_registers_the_load(
    env, monkeypatch, saved, raises
):
    if not saved:
        monkeypatch.setattr(env, "save_slots_for_resume", lambda should_abort: None)
    seen = []

    def acquire():
        seen.append(cs.saving_progress("image"))
        arb.acquire_for(arb.DIFFUSION)
        seen.append(cs.saving_progress("image"))
        if raises:
            raise arb.GpuBusyForAnotherAccountError(arb.CHAT, 1)

    try:
        _load(acquire = acquire)
    except arb.GpuBusyForAnotherAccountError:
        assert raises
    assert seen == [{"phase": cs.SAVING_CHAT_CONTEXT}] * 2
    assert cs.saving_progress("image") is None and cs._saving == {}


def test_load_progress_reads_in_flight_through_the_load_preflight(
    client,
    env,
    monkeypatch,
    healthy_diffusers,  # noqa: F811
):
    import core.inference.diffusion_hub_evidence as evidence

    _cuda(monkeypatch)
    base, body = _LOADS["image"]
    assert client.post(f"{base}/load", json = body).status_code == 200
    assert client.get(f"{base}/load-progress").json()["phase"] == "ready"
    entered, gate = threading.Event(), threading.Event()

    def _slow_prefetch(*args, **kwargs):
        entered.set()
        gate.wait(10)

    monkeypatch.setattr(evidence, "prefetch_for_pick", _slow_prefetch)
    monkeypatch.setattr(cs, "_preparing", {}, raising = False)
    answer: dict = {}
    poster = threading.Thread(
        target = lambda: answer.setdefault("resp", client.post(f"{base}/load", json = body))
    )
    poster.start()
    try:
        assert entered.wait(5)
        during = client.get(f"{base}/load-progress").json()
    finally:
        gate.set()
    poster.join(10)
    assert during["phase"] in ("downloading", "finalizing", "saving_chat_context")
    assert answer["resp"].status_code == 200
    assert getattr(cs, "_preparing", {}) == {}
    assert client.get(f"{base}/load-progress").json()["phase"] == "ready"


@pytest.mark.parametrize("phase", ["evidence", "chat_save"])
def test_an_unload_during_the_preflight_cancels_the_load(
    client,
    env,
    monkeypatch,
    healthy_diffusers,
    phase,  # noqa: F811
):
    import core.inference.diffusion_hub_evidence as evidence

    _cuda(monkeypatch)
    base, body = _LOADS["image"]
    monkeypatch.setattr(cs, "_preparing", {})
    gate = threading.Event()
    if phase == "evidence":
        entered = threading.Event()

        def _slow_prefetch(*args, **kwargs):
            entered.set()
            gate.wait(10)

        monkeypatch.setattr(evidence, "prefetch_for_pick", _slow_prefetch)
    else:
        env.gate, entered = gate, env.entered
    answer: dict = {}
    poster = threading.Thread(
        target = lambda: answer.setdefault("resp", client.post(f"{base}/load", json = body))
    )
    poster.start()
    try:
        assert entered.wait(5)
        during = client.get(f"{base}/load-progress").json()["phase"]
        stopped = client.post(f"{base}/unload")
        after = client.get(f"{base}/load-progress").json()["phase"]
    finally:
        gate.set()
    poster.join(10)
    assert not poster.is_alive()
    assert during == ("downloading" if phase == "evidence" else cs.SAVING_CHAT_CONTEXT)
    assert stopped.status_code == 200
    assert after not in ("downloading", "finalizing")
    resp = answer["resp"]
    assert resp.status_code == 409 and resp.json()["detail"] == "Diffusion load was cancelled."
    assert client.get(f"{base}/status").json()["loaded"] is False
    assert env.calls == [] and env.is_loaded and arb._owner == arb.CHAT
    assert cs._pending is None and kw._kv_resume is None and _no_files(env)
    assert cs._preparing == {} and cs._saving == {}


def test_a_cancelled_preflight_stops_before_the_native_video_unload():
    import inspect

    import routes.inference as routes

    src = inspect.getsource(routes.load_diffusion_model_gated)
    assert src.index("raise_if_load_cancelled(") < src.index(
        "_unload_native_video_sharing_the_sd_cpp_tree"
    )


def test_an_unload_while_the_acquire_waits_never_evicts_chat(
    client, env, monkeypatch, healthy_diffusers
):
    _cuda(monkeypatch)
    base, body = _LOADS["image"]
    monkeypatch.setattr(cs, "_preparing", {})
    entered, gate = threading.Event(), threading.Event()
    real_acquire_for = arb.acquire_for

    def _waiting_acquire_for(*args, **kwargs):
        if args and args[0] == arb.DIFFUSION:
            entered.set()
            gate.wait(10)
        return real_acquire_for(*args, **kwargs)

    monkeypatch.setattr(arb, "acquire_for", _waiting_acquire_for)
    answer: dict = {}
    poster = threading.Thread(
        target = lambda: answer.setdefault("resp", client.post(f"{base}/load", json = body))
    )
    poster.start()
    try:
        assert entered.wait(5)
        stopped = client.post(f"{base}/unload")
    finally:
        gate.set()
    poster.join(10)
    assert not poster.is_alive()
    assert stopped.status_code == 200
    resp = answer["resp"]
    assert resp.status_code == 409 and resp.json()["detail"] == "Diffusion load was cancelled."
    assert client.get(f"{base}/status").json()["loaded"] is False
    assert (
        not [c for c in env.calls if c[0] == "evict-chat"]
        and env.is_loaded
        and arb._owner == arb.CHAT
    )
    assert cs._preparing == {}


def _post_in_thread(client, url, body):
    answer: dict = {}
    poster = threading.Thread(target = lambda: answer.setdefault("resp", client.post(url, json = body)))
    poster.start()
    return poster, answer


@pytest.mark.parametrize("media", ["image", "video"])
def test_a_token_unload_while_the_acquire_waits_stops_that_load_before_the_eviction(
    client,
    env,
    monkeypatch,
    healthy_diffusers,
    media,  # noqa: F811
):
    from core.inference import media_load_ids

    _cuda(monkeypatch)
    if media == "video":
        client = _video_client(monkeypatch)
    base, body = _LOADS[media]
    owner = arb.DIFFUSION if media == "image" else arb.VIDEO
    media_load_ids._reset_for_tests()
    monkeypatch.setattr(cs, "_preparing", {})
    entered, gate = threading.Event(), threading.Event()
    real_acquire_for = arb.acquire_for

    def _waiting_acquire_for(*args, **kwargs):
        if args and args[0] == owner:
            entered.set()
            gate.wait(10)
        return real_acquire_for(*args, **kwargs)

    monkeypatch.setattr(arb, "acquire_for", _waiting_acquire_for)
    poster, answer = _post_in_thread(client, f"{base}/load", {**body, "load_token": "t1"})
    try:
        assert entered.wait(5)
        stopped = client.post(f"{base}/unload", json = {"expected_load_id": "t1"})
    finally:
        gate.set()
    poster.join(10)
    assert not poster.is_alive()
    assert stopped.json() == {"unloaded": False, "outcome": "cancelled"}
    resp = answer["resp"]
    assert resp.status_code == 409 and "cancelled" in resp.json()["detail"]
    # The resident chat model was never evicted for a load its unload had already cancelled.
    assert not [c for c in env.calls if c[0] == "evict-chat"]
    assert env.is_loaded and arb._owner == arb.CHAT
    assert client.get(f"{base}/status").json()["loaded"] is False
    assert cs._preparing == {}
    media_load_ids._reset_for_tests()


@pytest.mark.parametrize("media", ["image", "video"])
def test_a_token_unload_leaves_a_preparing_load_with_another_token_alone(
    client,
    env,
    monkeypatch,
    healthy_diffusers,
    media,  # noqa: F811
):
    from core.inference import media_load_ids

    _cuda(monkeypatch)
    if media == "video":
        client = _video_client(monkeypatch)
    base, body = _LOADS[media]
    owner = arb.DIFFUSION if media == "image" else arb.VIDEO
    media_load_ids._reset_for_tests()
    monkeypatch.setattr(cs, "_preparing", {})
    entered, gate = threading.Event(), threading.Event()
    real_acquire_for = arb.acquire_for

    def _waiting_acquire_for(*args, **kwargs):
        if args and args[0] == owner:
            entered.set()
            gate.wait(10)
        return real_acquire_for(*args, **kwargs)

    monkeypatch.setattr(arb, "acquire_for", _waiting_acquire_for)
    poster, answer = _post_in_thread(client, f"{base}/load", {**body, "load_token": "t1"})
    try:
        assert entered.wait(5)
        other = client.post(f"{base}/unload", json = {"expected_load_id": "t2"})
    finally:
        gate.set()
    poster.join(10)
    assert not poster.is_alive()
    assert other.json() == {"unloaded": False, "outcome": "cancelled"}
    assert answer["resp"].status_code == 200
    assert [c[0] for c in env.calls] == ["evict-chat"]
    assert media_load_ids._current["diffusion" if media == "image" else "video"].token == "t1"
    media_load_ids._reset_for_tests()


def test_a_token_unload_during_the_image_preflight_stops_it_before_the_engine_switch(
    client,
    env,
    monkeypatch,
    healthy_diffusers,  # noqa: F811
):
    import core.inference.diffusion_engine_router as router
    import core.inference.diffusion_hub_evidence as evidence
    from core.inference import media_load_ids

    _cuda(monkeypatch)
    base, body = _LOADS["image"]
    media_load_ids._reset_for_tests()
    monkeypatch.setattr(cs, "_preparing", {})
    entered, gate = threading.Event(), threading.Event()

    def _slow_prefetch(*args, **kwargs):
        entered.set()
        gate.wait(10)

    switched = []
    real_select = router.select_and_activate_engine

    def _select(*args, **kwargs):
        switched.append(True)
        return real_select(*args, **kwargs)

    monkeypatch.setattr(evidence, "prefetch_for_pick", _slow_prefetch)
    monkeypatch.setattr(router, "select_and_activate_engine", _select)
    poster, answer = _post_in_thread(client, f"{base}/load", {**body, "load_token": "t1"})
    try:
        assert entered.wait(5)
        stopped = client.post(f"{base}/unload", json = {"expected_load_id": "t1"})
    finally:
        gate.set()
    poster.join(10)
    assert not poster.is_alive()
    assert stopped.json() == {"unloaded": False, "outcome": "cancelled"}
    assert answer["resp"].status_code == 409
    assert switched == [] and not [c for c in env.calls if c[0] == "evict-chat"]
    assert cs._preparing == {}
    media_load_ids._reset_for_tests()


def test_an_engine_switch_waits_for_a_cancel_or_refuses_after_it(monkeypatch):
    import routes.inference as routes

    calls = []
    with cs.preparing_load("image", "t1") as preflight:
        assert (
            routes._activate_unless_cancelled(preflight, None, lambda: calls.append(1) or "e")
            == "e"
        )
        cs.cancel_preparing_loads("image", current_account_id(), load_token = "t2")
        assert not preflight.cancelled
        cs.cancel_preparing_loads("image", current_account_id(), load_token = "t1")
        assert preflight.cancelled

        def _switching():
            from core.inference.diffusion_engine_router import before_engine_switch
            before_engine_switch.get()()
            calls.append(2)

        with pytest.raises(RuntimeError, match = "cancelled"):
            routes._activate_unless_cancelled(preflight, None, _switching)
    assert calls == [1]


def test_a_cancelled_video_preflight_never_unloads_the_image_model():
    with cs.preparing_load("video", "t1"):
        cs.cancel_preparing_loads("video", current_account_id(), load_token = "t1")
        unloaded = []
        with pytest.raises(RuntimeError, match = "Video load was cancelled"):
            cs.run_unless_load_cancelled("Video load was cancelled.", unloaded.append, 1)
    assert unloaded == []


def test_a_token_unload_names_the_engine_the_preflight_settled_on(client, env, monkeypatch):
    # The named POST finishes switching engines while its preflight is being cancelled: the unload
    # must check the load on the engine it registered on, not the one active before the cancel.
    import types

    import core.inference.diffusion_engine_router as router
    from core.inference import media_load_ids

    old = types.SimpleNamespace(name = "old", unload = None)
    new = types.SimpleNamespace(name = "new", unload = None)
    active = {"engine": old}
    monkeypatch.setattr(router, "get_active_diffusion_engine", lambda: active["engine"])

    def _cancel(*args, **kwargs):
        active["engine"] = new  # the switch completed and the load registered before the cancel ran
        return 0

    monkeypatch.setattr(cs, "cancel_preparing_loads", _cancel)
    seen = []

    def _unload_if_current(modality, engine, token, unload):
        seen.append(engine.name)
        return "superseded", None

    monkeypatch.setattr(media_load_ids, "unload_if_current", _unload_if_current)
    resp = client.post(f"{_LOADS['image'][0]}/unload", json = {"expected_load_id": "t1"})
    assert resp.status_code == 200, resp.text
    assert seen == ["new"]
