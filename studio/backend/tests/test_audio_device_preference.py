# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The user's choice of where an audio model's weights go.

Audio loads pick an accelerator on their own. These cover the option that
overrides that: CPU RAM must win over a working GPU, "auto" must still detect,
and a resident model loaded under the other preference must be reloaded rather
than reused where it is.
"""

import sys
import threading
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.inference.audio_device import (  # noqa: E402
    audio_device_default,
    audio_device_forces_cpu,
    normalize_audio_device,
)


# --- the preference itself -------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("cpu", "cpu"),
        ("CPU", "cpu"),
        ("  cpu  ", "cpu"),
        ("ram", "cpu"),
        ("gpu", "gpu"),
        ("cuda", "gpu"),
        ("mps", "gpu"),
        ("rocm", "gpu"),
        ("auto", "auto"),
        ("", "auto"),
        (None, "auto"),
    ],
)
def test_every_accepted_spelling_maps_onto_one_of_three_values(value, expected):
    assert normalize_audio_device(value) == expected


def test_an_unknown_preference_detects_rather_than_failing_the_load():
    """Detection is what the caller would have done without the option at all."""
    assert normalize_audio_device("gpu-2") == "auto"
    assert normalize_audio_device("nvidia rtx 4090") == "auto"
    assert not audio_device_forces_cpu("gpu-2")


def test_the_environment_supplies_the_default_for_a_request_that_names_none(monkeypatch):
    """A headless or CLI Studio sets this once instead of sending it per request."""
    monkeypatch.setenv("UNSLOTH_AUDIO_DEVICE", "cpu")
    assert audio_device_default() == "cpu"
    assert audio_device_forces_cpu(None)

    # An explicit request still outranks it, in both directions.
    assert not audio_device_forces_cpu("auto")
    assert audio_device_forces_cpu("cpu")


def test_without_the_environment_variable_nothing_is_forced_to_cpu(monkeypatch):
    monkeypatch.delenv("UNSLOTH_AUDIO_DEVICE", raising = False)
    assert audio_device_default() == "auto"
    assert not audio_device_forces_cpu(None)


# --- the Transformers dictation sidecar ------------------------------------


def _torch_with_cuda(monkeypatch):
    """A torch whose CUDA is available, so anything but CPU is a real choice."""
    torch = types.SimpleNamespace(
        float16 = "float16",
        float32 = "float32",
        cuda = types.SimpleNamespace(is_available = lambda: True),
        backends = types.SimpleNamespace(mps = types.SimpleNamespace(is_available = lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_cpu_is_honoured_on_a_machine_with_a_working_gpu(monkeypatch):
    """The whole point of the option: detection would have said cuda."""
    from core.inference import stt_sidecar

    _torch_with_cuda(monkeypatch)
    monkeypatch.setattr(stt_sidecar, "_training_active", lambda: False)

    assert stt_sidecar._pick_device("cpu") == ("cpu", "float32")


def test_auto_and_gpu_both_still_detect_the_accelerator(monkeypatch):
    """ "gpu" is not a separate placement: detection already prefers the card."""
    from core.inference import stt_sidecar

    _torch_with_cuda(monkeypatch)
    monkeypatch.setattr(stt_sidecar, "_training_active", lambda: False)

    assert stt_sidecar._pick_device("auto") == ("cuda", "float16")
    assert stt_sidecar._pick_device("gpu") == ("cuda", "float16")
    assert stt_sidecar._pick_device(None) == ("cuda", "float16")


def test_a_resident_model_on_the_other_device_is_reloaded_not_reused(monkeypatch):
    """A preference is a request about placement. Reusing the old one ignores it."""
    from core.inference import stt_sidecar

    monkeypatch.setattr(stt_sidecar, "_pick_device", lambda _preference = None: ("cpu", "float32"))
    monkeypatch.setattr(stt_sidecar, "ensure_stt_available", lambda: None)
    monkeypatch.setattr(stt_sidecar, "resolve_model_id", lambda model: model or "small")

    sidecar = stt_sidecar.WhisperSttSidecar(keep_alive_seconds = 0.0)
    sidecar._engine = object()
    sidecar._model_id = "small"
    sidecar._device = "cuda"
    sidecar._device_preference = "auto"

    builds: list[str] = []

    def _build(snapshot_path, device, dtype, cancel_event):
        builds.append(device)
        return object()

    monkeypatch.setattr(sidecar, "_build_model", _build)
    monkeypatch.setattr(
        sidecar,
        "_ensure_model_downloaded",
        lambda model_id, use_resident = True: stt_sidecar._CachedSttSnapshot(
            # The resident shortcut answers with no path; a replacement load needs one.
            path = None if use_resident else "/snapshots/small",
            is_multilingual = True,
        ),
    )
    monkeypatch.setattr(sidecar, "_release_engine_locked", lambda: True)

    sidecar.load("small", device = "cpu")

    assert builds == ["cpu"], "the CPU preference must have driven a fresh load"
    assert sidecar._device == "cpu"
    assert sidecar._device_preference == "cpu"


def test_the_same_preference_reuses_the_resident_model(monkeypatch):
    """Unchanged placement must stay a residency check, not a reload."""
    from core.inference import stt_sidecar

    monkeypatch.setattr(stt_sidecar, "ensure_stt_available", lambda: None)
    monkeypatch.setattr(stt_sidecar, "resolve_model_id", lambda model: model or "small")

    sidecar = stt_sidecar.WhisperSttSidecar(keep_alive_seconds = 0.0)
    resident = object()
    sidecar._engine = resident
    sidecar._model_id = "small"
    sidecar._device = "cpu"
    sidecar._device_preference = "cpu"

    def _never(*args, **kwargs):
        raise AssertionError("a matching preference must not rebuild the model")

    monkeypatch.setattr(sidecar, "_build_model", _never)

    assert sidecar.load("small", device = "cpu") is resident


def test_a_model_loaded_before_the_option_existed_is_not_reloaded(monkeypatch):
    """No recorded preference means the engine predates the choice; reusing it is
    what every caller got before, and a forced reload would cost a load for nothing."""
    from core.inference import stt_sidecar

    monkeypatch.setattr(stt_sidecar, "ensure_stt_available", lambda: None)
    monkeypatch.setattr(stt_sidecar, "resolve_model_id", lambda model: model or "small")

    sidecar = stt_sidecar.WhisperSttSidecar(keep_alive_seconds = 0.0)
    resident = object()
    sidecar._engine = resident
    sidecar._model_id = "small"
    sidecar._device = "cuda"
    sidecar._device_preference = None

    monkeypatch.setattr(
        sidecar,
        "_build_model",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not rebuild")),
    )

    assert sidecar.load("small", device = "auto") is resident


# --- the registry forwards it ----------------------------------------------


def test_the_registry_hands_the_preference_to_the_engines_sidecar(monkeypatch):
    from core.inference import stt_registry

    seen: dict = {}

    class _Sidecar:
        def load(
            self,
            model,
            request_cancel_event = None,
            device = None,
        ):
            seen["model"] = model
            seen["device"] = device

    monkeypatch.setattr(stt_registry, "sidecar_for", lambda _engine: _Sidecar())
    monkeypatch.setattr(stt_registry, "_model_is_downloaded", lambda _e, _m: True)
    monkeypatch.setattr(stt_registry, "unload", lambda *a, **k: [])

    stt_registry.load("small", "transformers", threading.Event(), device = "cpu")

    assert seen == {"model": "small", "device": "cpu"}


# --- native audio (TTS) ----------------------------------------------------


def test_native_audio_holds_tts_weights_in_cpu_ram_when_asked(monkeypatch):
    from core.inference import native_audio

    _torch_with_cuda(monkeypatch)

    assert native_audio.NativeAudioBackend(device_preference = "cpu").device == "cpu"
    assert native_audio.NativeAudioBackend(device_preference = "auto").device == "cuda"
    assert native_audio.NativeAudioBackend().device == "cuda"


def test_minimax_music_explains_that_cpu_was_chosen_rather_than_missing(monkeypatch):
    """The generic refusal reads like "your hardware cannot do this" and sends a
    user with a perfectly good card looking for one."""
    from core.inference import native_audio

    _torch_with_cuda(monkeypatch)
    backend = native_audio.NativeAudioBackend(device_preference = "cpu")
    config = types.SimpleNamespace(
        identifier = "MiniMaxAI/MiniMax-Music3",
        audio_type = "minimax_music3",
        path = None,
    )

    with pytest.raises(RuntimeError, match = "cannot be loaded into CPU RAM"):
        backend.load_model(config)


# --- a caller with no preference must not move a placed model ---------------


def test_a_caller_that_sends_no_device_leaves_the_placement_alone(monkeypatch):
    """``/v1/audio/transcriptions`` sends none. Treating that as "auto" pulled a
    CPU model back onto the GPU, and the next dictation pulled it off again:
    two full reloads per alternation, and VRAM the user asked us not to take."""
    from core.inference import stt_sidecar

    monkeypatch.setattr(stt_sidecar, "ensure_stt_available", lambda: None)
    monkeypatch.setattr(stt_sidecar, "resolve_model_id", lambda model: model or "small")
    monkeypatch.setattr(
        stt_sidecar,
        "_pick_device",
        lambda preference = None: ("cpu", "float32") if preference == "cpu" else ("cuda", "float16"),
    )

    sidecar = stt_sidecar.WhisperSttSidecar(keep_alive_seconds = 0.0)
    builds: list[str] = []
    monkeypatch.setattr(
        sidecar, "_build_model", lambda p, device, dtype, c: builds.append(device) or object()
    )
    monkeypatch.setattr(sidecar, "_release_engine_locked", lambda: True)
    monkeypatch.setattr(
        sidecar,
        "_ensure_model_downloaded",
        lambda model_id, use_resident = True: stt_sidecar._CachedSttSnapshot(
            path = None if (sidecar._engine is not None and use_resident) else "/snapshots/small",
            is_multilingual = True,
        ),
    )

    sidecar.load("small", device = "cpu")  # Voice settings: the user picked CPU
    sidecar.load("small", device = None)  # OpenAI-compatible route: no opinion
    sidecar.load("small", device = "cpu")  # the next dictation

    assert builds == ["cpu"], "only the first load should have built anything"
    assert sidecar._device == "cpu"


def test_an_explicit_change_still_reloads_after_a_no_opinion_call(monkeypatch):
    """No opinion must not also freeze the placement: the setting still moves it."""
    from core.inference import stt_sidecar

    monkeypatch.setattr(stt_sidecar, "ensure_stt_available", lambda: None)
    monkeypatch.setattr(stt_sidecar, "resolve_model_id", lambda model: model or "small")
    monkeypatch.setattr(
        stt_sidecar,
        "_pick_device",
        lambda preference = None: ("cpu", "float32") if preference == "cpu" else ("cuda", "float16"),
    )

    sidecar = stt_sidecar.WhisperSttSidecar(keep_alive_seconds = 0.0)
    builds: list[str] = []
    monkeypatch.setattr(
        sidecar, "_build_model", lambda p, device, dtype, c: builds.append(device) or object()
    )
    monkeypatch.setattr(sidecar, "_release_engine_locked", lambda: True)
    monkeypatch.setattr(
        sidecar,
        "_ensure_model_downloaded",
        lambda model_id, use_resident = True: stt_sidecar._CachedSttSnapshot(
            path = None if (sidecar._engine is not None and use_resident) else "/snapshots/small",
            is_multilingual = True,
        ),
    )

    sidecar.load("small", device = "cpu")
    sidecar.load("small", device = None)
    sidecar.load("small", device = "auto")

    assert builds == ["cpu", "cuda"]


def test_the_mtmd_reuse_branch_still_records_the_choice(monkeypatch):
    """Training makes an explicit "cpu" indistinguishable from the running
    server, so the early return has to keep the preference or the next
    device-less load sends the model back to the GPU once training ends."""
    from core.inference import stt_mtmd_sidecar

    sidecar = stt_mtmd_sidecar.MtmdSttSidecar.__new__(stt_mtmd_sidecar.MtmdSttSidecar)
    sidecar._lock = threading.RLock()
    sidecar._forced_cpu = False
    sidecar._gpu_disabled = True
    sidecar._model_id = "m"
    sidecar._binary_path_revision = 1
    sidecar._active_requests = 0
    monkeypatch.setattr(sidecar, "_process_alive", lambda: True)
    monkeypatch.setattr(sidecar, "_schedule_idle_unload_locked", lambda: None)
    monkeypatch.setattr(stt_mtmd_sidecar, "_training_active", lambda: True)

    sidecar._load_locked("m", "whisper-server", path_revision = 1, device = "cpu")

    assert sidecar._forced_cpu is True


def test_stt_unload_can_skip_a_sidecar_that_is_mid_transcription():
    """The Voice device switch is an early release, not a reclaim: draining a
    live transcription for 30s and then killing it loses the recording."""
    import inspect

    from core.inference import stt_registry
    from core.inference.orchestrator import InferenceOrchestrator
    from routes import inference as ri

    assert "wait" in inspect.signature(ri.stt_unload).parameters
    assert inspect.signature(ri.stt_unload).parameters["wait"].default is True
    # Both halves of _stt_lifecycle have to take it, or the route raises TypeError
    # for whichever one is live.
    assert "wait" in inspect.signature(InferenceOrchestrator.unload_stt_model).parameters
    assert "wait" in inspect.signature(stt_registry.unload).parameters
