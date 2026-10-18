# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The chat KV kept across a media swap: saved before the acquire, promoted to the reload stash
only when _evict_chat unloads that model, and bounded so a slow save never stalls the load POST.

The llama backend, the evictors and the settings are fakes: no llama-server, no GPU."""

from __future__ import annotations

import asyncio
import inspect
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
        try:
            await asyncio.to_thread(acquire or (lambda: arb.acquire_for(owner)))
        finally:
            swap.close()

    asyncio.run(_go())


def _no_files(llama) -> bool:
    return not any(f.exists() for f in llama.files)


_TAKES_POLICY = "policy" in inspect.signature(arb.acquire_for).parameters
_TAKES_NEED = "need_mib" in inspect.signature(arb.acquire_for).parameters


@pytest.mark.parametrize(
    "policy, fits", [("off", None), ("fit", True), ("fit", False), ("always", None)]
)
@pytest.mark.parametrize("current", [None, arb.CHAT, arb.DIFFUSION, arb.VIDEO])
@pytest.mark.parametrize("target", [arb.DIFFUSION, arb.VIDEO, arb.CHAT])
def test_would_evict_matches_acquire_for(monkeypatch, policy, fits, current, target):
    if policy != "off" and not _TAKES_POLICY:
        pytest.skip("acquire_for has no policy before R1")
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
    need = {0: 1024} if fits is not None and _TAKES_NEED else None
    predicted = arb.would_evict(target, policy = policy, need_mib = need)
    kwargs = {"policy": policy} if _TAKES_POLICY else {}
    if need:
        kwargs["need_mib"] = need
    try:
        arb.acquire_for(target, **kwargs)
    except RuntimeError:
        pass  # a refusal evicts nothing
    assert predicted == ("evict-chat" in calls)


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
    }
