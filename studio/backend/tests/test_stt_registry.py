# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dictation lifecycle goes through one place, shared with the orchestrator."""

import pytest
import threading

from core.inference import stt_registry


class _Sidecar:
    def __init__(
        self,
        name,
        model = None,
        loading = False,
        fail = False,
    ):
        self.name = name
        self.loaded_model = model
        self.device = f"{name}-device" if model else None
        self._loading = loading
        self._fail = fail
        self.unloaded = False
        self.unload_waits = []
        self.unload_expected = []
        self.loaded_with = None
        self.load_cancel_event = None
        self.loaded_device = None

    def is_loading(self):
        return self._loading

    def load(
        self,
        model,
        request_cancel_event = None,
        device = None,
    ):
        self.loaded_with = model
        self.load_cancel_event = request_cancel_event
        self.loaded_device = device

    def unload(
        self,
        wait = True,
        expected_model = None,
    ):
        if self._fail:
            raise RuntimeError("boom")
        self.unload_waits.append(wait)
        self.unload_expected.append(expected_model)
        self.unloaded = True


def test_unload_attempts_every_engine_even_after_one_raises(monkeypatch):
    made = {}

    def make(name):
        made.setdefault(name, _Sidecar(name, fail = name == "transformers"))
        return made[name]

    monkeypatch.setattr(stt_registry, "sidecar_for", make)
    failed = stt_registry.unload()

    assert list(made) == list(stt_registry.STT_ENGINES)
    assert failed == ["transformers"]
    # The engines after the failure still released.
    assert made["gguf"].unloaded and made["mtmd"].unloaded


def test_load_delegates_to_the_engines_sidecar(monkeypatch):
    sidecar = _Sidecar("mtmd")
    cancel_event = threading.Event()
    monkeypatch.setattr(stt_registry, "sidecar_for", lambda name: sidecar)

    stt_registry.load("qwen3-asr-0.6b", "mtmd", cancel_event)
    assert sidecar.loaded_with == "qwen3-asr-0.6b"
    assert sidecar.load_cancel_event is cancel_event


def test_load_releases_the_other_engines_after_the_target_loads(monkeypatch):
    order = []
    sidecars = {name: _Sidecar(name) for name in stt_registry.STT_ENGINES}
    for name, sidecar in sidecars.items():
        sidecar.unload = lambda wait = True, expected_model = None, name = name: order.append(
            f"unload:{name}"
        )
    target = sidecars["mtmd"]
    target.load = lambda model, request_cancel_event = None, device = None: order.append("load:mtmd")
    monkeypatch.setattr(stt_registry, "sidecar_for", lambda name: sidecars[name])

    stt_registry.load("qwen3-asr-0.6b", "mtmd")

    # Two engines resident at once doubles VRAM for the whole keep-alive window, but the
    # release follows the load: a 409 must not cost the user the engine they were using.
    assert order == ["load:mtmd", "unload:transformers", "unload:gguf"]


def test_load_still_succeeds_when_another_engine_refuses_to_release(monkeypatch):
    sidecars = {name: _Sidecar(name, fail = name == "gguf") for name in stt_registry.STT_ENGINES}
    monkeypatch.setattr(stt_registry, "sidecar_for", lambda name: sidecars[name])

    stt_registry.load("small", "transformers")
    assert sidecars["transformers"].loaded_with == "small"
    assert sidecars["mtmd"].unloaded


def test_resident_reports_whichever_engine_holds_a_model(monkeypatch):
    sidecars = {
        "transformers": _Sidecar("transformers"),
        "gguf": _Sidecar("gguf"),
        "mtmd": _Sidecar("mtmd", model = "qwen3-asr-0.6b"),
    }
    monkeypatch.setattr(stt_registry, "sidecar_for", lambda name: sidecars[name])

    assert stt_registry.resident() == {
        "model": "qwen3-asr-0.6b",
        "engine": "mtmd",
        "device": "mtmd-device",
        "loading": False,
    }


def test_resident_reports_an_in_flight_load(monkeypatch):
    sidecars = {
        "transformers": _Sidecar("transformers"),
        "gguf": _Sidecar("gguf", loading = True),
        "mtmd": _Sidecar("mtmd"),
    }
    monkeypatch.setattr(stt_registry, "sidecar_for", lambda name: sidecars[name])

    resident = stt_registry.resident()
    assert resident["loading"] is True and resident["engine"] == "gguf"


def test_resident_reports_nothing_when_every_engine_is_idle(monkeypatch):
    monkeypatch.setattr(stt_registry, "sidecar_for", lambda name: _Sidecar(name))
    assert stt_registry.resident()["model"] is None


def test_an_unimportable_engine_never_takes_the_status_down(monkeypatch):
    def make(name):
        if name == "gguf":
            raise ImportError("whisper.cpp bindings missing")
        return _Sidecar(name, model = "small" if name == "mtmd" else None)

    monkeypatch.setattr(stt_registry, "sidecar_for", make)
    # gguf raising must not hide the model mtmd is holding.
    assert stt_registry.resident()["model"] == "small"


def test_the_orchestrator_exposes_the_same_lifecycle(monkeypatch):
    from core.inference.orchestrator import InferenceOrchestrator
    for name in ("load_stt_model", "unload_stt_model", "resident_stt_model"):
        assert callable(getattr(InferenceOrchestrator, name, None)), name


def test_the_route_loads_through_the_orchestrator_when_one_exists(monkeypatch):
    import routes.inference as ri
    from core.inference import orchestrator as orch

    class _Backend:
        load_stt_model = "orchestrator-load"
        unload_stt_model = "orchestrator-unload"

    monkeypatch.setattr(orch, "peek_inference_backend", lambda: _Backend())
    assert ri._stt_lifecycle() == ("orchestrator-load", "orchestrator-unload")


def test_a_cold_process_loads_without_building_an_orchestrator(monkeypatch):
    import routes.inference as ri
    from core.inference import orchestrator as orch

    def _never():
        raise AssertionError("dictation must not construct the chat orchestrator")

    monkeypatch.setattr(orch, "peek_inference_backend", lambda: None)
    monkeypatch.setattr(orch, "get_inference_backend", _never)
    # Same functions the orchestrator's methods forward to, so neither path
    # can drift from the other.
    assert ri._stt_lifecycle() == (stt_registry.load, stt_registry.unload)


def test_load_never_blocks_on_an_engine_that_is_serving_a_request(monkeypatch):
    """A transcription holds the sidecar lock for minutes; the new load must not wait."""
    sidecars = {name: _Sidecar(name) for name in stt_registry.STT_ENGINES}
    monkeypatch.setattr(stt_registry, "sidecar_for", lambda name: sidecars[name])

    stt_registry.load("qwen3-asr-0.6b", "mtmd")

    assert sidecars["transformers"].unload_waits == [False]
    assert sidecars["gguf"].unload_waits == [False]
    # A caller releasing every engine on purpose still waits for each one.
    stt_registry.unload()
    assert sidecars["mtmd"].unload_waits == [True]


def test_a_busy_sidecar_keeps_its_model_and_an_idle_one_releases_it():
    """Against the real sidecars: `wait=False` must decline a live request, not block.

    RLock.locked() is 3.14-only and the mtmd sidecar drops _lock before its HTTP call, so
    the busy probe cannot be either.
    """
    from core.inference.stt_ggml_sidecar import get_ggml_stt_sidecar
    from core.inference.stt_mtmd_sidecar import get_mtmd_stt_sidecar

    ggml = get_ggml_stt_sidecar()
    released = []
    ggml._release_locked = lambda: released.append("ggml")
    with ggml._lock:
        blocker = threading.Thread(target = ggml.unload, kwargs = {"wait": False})
        blocker.start()
        blocker.join(timeout = 2)
        assert not blocker.is_alive(), "unload(wait=False) blocked on a held lock"
    assert released == []

    ggml.unload(wait = False)
    assert released == ["ggml"]

    mtmd = get_mtmd_stt_sidecar()
    mtmd_released = []
    mtmd._release_locked = lambda: mtmd_released.append("mtmd")
    mtmd._active_requests = 1
    mtmd.unload(wait = False)
    assert mtmd_released == [], "a transcription outside _lock still counts as busy"
    mtmd._active_requests = 0
    mtmd.unload(wait = False)
    assert mtmd_released == ["mtmd"]


def test_a_failed_load_leaves_the_engine_the_user_was_using(monkeypatch):
    """The sidecars order preflight before release for this reason; so does the registry."""
    sidecars = {name: _Sidecar(name) for name in stt_registry.STT_ENGINES}

    def refuse(
        model,
        request_cancel_event = None,
        device = None,
    ):
        raise RuntimeError("STT model 'x' is not downloaded.")

    sidecars["mtmd"].load = refuse
    monkeypatch.setattr(stt_registry, "sidecar_for", lambda name: sidecars[name])

    with pytest.raises(RuntimeError, match = "not downloaded"):
        stt_registry.load("x", "mtmd")

    assert not sidecars["transformers"].unloaded
    assert not sidecars["gguf"].unloaded


def test_a_non_curated_whisper_cache_row_cannot_chat():
    """`_hidden_stt` comes from the config sniff, so it covers distil-whisper and a user's own
    fine-tune, not just the seven curated ids. can_chat is what auto-load and the chat picker
    filter on, and neither looks at the task."""
    from hub.services.models.cache_inventory import _cache_inventory_fields

    fields = _cache_inventory_fields(
        "distil-whisper/distil-large-v3",
        "safetensors",
        stt_only = True,
    )
    assert fields["capabilities"]["can_chat"] is False
    assert fields["capabilities"]["supports_vision"] is False


def test_a_wait_false_unload_rechecks_active_requests_under_the_lock():
    """transcribe claims _active_requests while holding _lock, so a request starting between
    the unlocked probe and the acquire would have llama-server killed underneath it."""
    import threading

    from core.inference.stt_mtmd_sidecar import MtmdSttSidecar

    sidecar = MtmdSttSidecar.__new__(MtmdSttSidecar)
    sidecar._lock = threading.RLock()
    sidecar._active_requests = 0
    sidecar._loading = False
    sidecar.is_loading = lambda: False
    sidecar.cancel_pending_load = lambda: False
    sidecar.wait_for_load_to_settle = lambda: None
    released = []
    sidecar._release_locked = lambda: released.append(True)

    # The racing transcription claims the slot after the unlocked probe has already passed.
    real_lock = sidecar._lock

    class _RacingLock:
        def acquire(
            self,
            blocking = True,
            *args,
            **kwargs,
        ):
            sidecar._active_requests = 1
            return real_lock.acquire(blocking, *args, **kwargs)

        def release(self):
            return real_lock.release()

    sidecar._lock = _RacingLock()
    MtmdSttSidecar.unload(sidecar, wait = False)
    assert released == []


def test_a_downloaded_switch_releases_the_old_engine_before_allocating(monkeypatch):
    """Holding two engines across the load is what OOMs a device that fits either alone."""
    from core.inference import stt_registry

    order = []
    monkeypatch.setattr(stt_registry, "_model_is_downloaded", lambda _e, _m: True)
    monkeypatch.setattr(
        stt_registry,
        "unload",
        lambda engines = None, wait = True: order.append(("unload", tuple(engines or ()))) or [],
    )

    class _Fake:
        def load(
            self,
            model,
            request_cancel_event = None,
            device = None,
        ):
            order.append(("load", model))

    monkeypatch.setattr(stt_registry, "sidecar_for", lambda _engine: _Fake())
    stt_registry.load("mtmd", "some/asr-model")

    assert [step for step, _ in order] == ["unload", "load"]


def test_an_undownloaded_switch_keeps_the_resident_engine_until_the_load_succeeds(monkeypatch):
    """A 409 for a model that was never downloaded must not cost the engine in use."""
    from core.inference import stt_registry

    order = []
    monkeypatch.setattr(stt_registry, "_model_is_downloaded", lambda _e, _m: False)
    monkeypatch.setattr(
        stt_registry,
        "unload",
        lambda engines = None, wait = True: order.append(("unload", tuple(engines or ()))) or [],
    )

    class _Fake:
        def load(
            self,
            model,
            request_cancel_event = None,
            device = None,
        ):
            order.append(("load", model))

    monkeypatch.setattr(stt_registry, "sidecar_for", lambda _engine: _Fake())
    stt_registry.load("mtmd", "some/asr-model")

    assert [step for step, _ in order] == ["load", "unload"]


def _racing_sidecar():
    """An MtmdSttSidecar stripped to the fields unload() touches."""
    import threading

    from core.inference.stt_mtmd_sidecar import MtmdSttSidecar

    sidecar = MtmdSttSidecar.__new__(MtmdSttSidecar)
    sidecar._lock = threading.RLock()
    sidecar._active_requests = 0
    sidecar._loading = False
    sidecar.is_loading = lambda: False
    sidecar.cancel_pending_load = lambda: False
    sidecar.wait_for_load_to_settle = lambda: None
    sidecar.released = []
    sidecar._release_locked = lambda: sidecar.released.append(True)
    return sidecar


def test_a_blocking_unload_drains_a_request_that_started_during_the_acquire():
    """The under-lock recheck only guarded wait=False, so the blocking unload training
    uses could reap llama-server underneath a transcription that had just started, losing
    the recording rather than honoring the drain window."""
    import threading

    from core.inference.stt_mtmd_sidecar import MtmdSttSidecar

    sidecar = _racing_sidecar()
    real_lock = sidecar._lock
    acquires = []

    class _RacingLock:
        def acquire(
            self,
            blocking = True,
            *args,
            **kwargs,
        ):
            acquires.append(True)
            if len(acquires) == 1:
                # Claimed after the unlocked drain has already passed.
                sidecar._active_requests = 1
                threading.Timer(0.15, lambda: setattr(sidecar, "_active_requests", 0)).start()
            return real_lock.acquire(blocking, *args, **kwargs)

        def release(self):
            return real_lock.release()

    sidecar._lock = _RacingLock()
    MtmdSttSidecar.unload(sidecar, wait = True)
    # Released, but only once the transcription that raced in had finished.
    assert sidecar.released == [True]
    assert len(acquires) >= 2
    assert sidecar._active_requests == 0


def test_a_blocking_unload_still_gives_up_after_the_drain_window(monkeypatch):
    """Training claiming the VRAM cannot wait forever, so a request that never finishes
    must not turn the bounded window into a permanent block."""
    from core.inference import stt_mtmd_sidecar
    from core.inference.stt_mtmd_sidecar import MtmdSttSidecar

    monkeypatch.setattr(stt_mtmd_sidecar, "_ACTIVE_REQUEST_DRAIN_TIMEOUT", 0.3)
    sidecar = _racing_sidecar()
    sidecar._active_requests = 1
    MtmdSttSidecar.unload(sidecar, wait = True)
    assert sidecar.released == [True]


def test_an_implicit_transcribe_load_releases_the_other_engines(monkeypatch):
    """Each sidecar loads its own model, but only the registry frees the others.

    An API client alternating between engines through /v1/audio/transcriptions never
    calls /audio/stt/load, so without this both models stayed resident until their
    independent idle timers fired, which OOMs a device that fits either alone.
    """
    import asyncio

    import routes.inference as ri

    loaded: list[tuple] = []
    monkeypatch.setattr(ri, "_resolve_serving_stt_engine", lambda engine: "mtmd")
    monkeypatch.setattr(
        ri,
        "_stt_sidecar_for",
        lambda engine: type(
            "S",
            (),
            {"transcribe": staticmethod(lambda *a, **k: {"text": "hi", "model": "qwen3-asr-0.6b"})},
        )(),
    )
    monkeypatch.setattr(
        ri,
        "_stt_lifecycle",
        lambda: (
            lambda model, engine, cancel = None, device = None: loaded.append((model, engine)),
            lambda *a: [],
        ),
    )

    result = asyncio.run(
        ri._transcribe_audio_result(
            b"audio",
            model = "qwen3-asr-0.6b",
            language = None,
            fast = True,
            engine = "mtmd",
        )
    )
    assert result["text"] == "hi"
    assert loaded == [("qwen3-asr-0.6b", "mtmd")]


def test_the_registry_load_is_what_frees_the_other_engines(monkeypatch):
    """The lifecycle the route calls is the registry's, whose contract is single
    residency; this pins that it releases the others rather than only allocating."""
    from core.inference import stt_registry

    released: list = []
    monkeypatch.setattr(stt_registry, "_model_is_downloaded", lambda engine, model: True)
    monkeypatch.setattr(
        stt_registry, "unload", lambda engines, wait = True: released.append(list(engines))
    )
    monkeypatch.setattr(
        stt_registry,
        "sidecar_for",
        lambda engine: type("S", (), {"load": staticmethod(lambda *a, **k: None)})(),
    )

    stt_registry.load("qwen3-asr-0.6b", "mtmd")
    assert released and "mtmd" not in released[0]
    assert set(released[0]) == {e for e in stt_registry.STT_ENGINES if e != "mtmd"}


def test_a_scoped_unload_leaves_another_surfaces_newer_model_alone():
    """Ownership is decided by the caller, so a queued Eject can arrive after another
    surface switched the same engine. The comparison happens under the sidecar's own
    lock, which is the only place the answer cannot go stale."""
    import threading

    from core.inference.stt_sidecar import WhisperSttSidecar

    sidecar = WhisperSttSidecar.__new__(WhisperSttSidecar)
    sidecar._lock = threading.RLock()
    sidecar._model_id = "base"
    released = []
    sidecar._release_engine_locked = lambda: released.append(True)

    # The caller owned "small"; "base" belongs to whoever loaded it after.
    WhisperSttSidecar.unload(sidecar, expected_model = "small")
    assert released == []

    # Its own model still goes, and so does an unscoped release.
    WhisperSttSidecar.unload(sidecar, expected_model = "base")
    assert released == [True]
    WhisperSttSidecar.unload(sidecar)
    assert released == [True, True]


def test_a_scoped_unload_accepts_the_short_key_the_client_sends():
    """Clients name the sidecar key ("small"), which the sidecar stores resolved."""
    import threading

    from core.inference.stt_sidecar import WhisperSttSidecar, resolve_model_id

    sidecar = WhisperSttSidecar.__new__(WhisperSttSidecar)
    sidecar._lock = threading.RLock()
    sidecar._model_id = resolve_model_id("small")
    released = []
    sidecar._release_engine_locked = lambda: released.append(True)

    WhisperSttSidecar.unload(sidecar, expected_model = "small")
    assert released == [True]


def test_the_registry_passes_the_claimed_model_to_each_sidecar(monkeypatch):
    from core.inference import stt_registry

    seen: list[tuple] = []
    monkeypatch.setattr(
        stt_registry,
        "sidecar_for",
        lambda engine: type(
            "S",
            (),
            {
                "unload": staticmethod(
                    lambda wait = True, expected_model = None: seen.append((engine, expected_model))
                )
            },
        )(),
    )
    stt_registry.unload(["gguf"], expected_model = "small")
    assert seen == [("gguf", "small")]
