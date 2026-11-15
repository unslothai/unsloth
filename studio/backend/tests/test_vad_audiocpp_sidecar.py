# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import time
from types import SimpleNamespace

import pytest
from types import SimpleNamespace

from core.inference import audio_cpp_clip_sidecar as clip
from core.inference import audio_cpp_models as acm
from core.inference import vad_audiocpp_sidecar as vad

# A Hub detector, which downloads and may take the GPU; the default is built in and does neither.
PULSEVAD = f"{acm.AUDIO_CPP_REPO}/PulseVAD-GGUF"


def test_segments_are_reported_in_seconds_with_the_voiced_total():
    result = vad.segments_from_result(
        {
            "segments": [
                {"start_sample": 1600, "end_sample": 8000, "confidence": 0.9},
                {"start_sample": 9600, "end_sample": 9600},
                {"start_sample": 16000, "end_sample": 24000},
            ],
            "sample_rate": 16000,
        },
        fallback_rate = 8000,
    )
    assert result["sample_rate"] == 16000
    assert [(s["start_s"], s["end_s"]) for s in result["segments"]] == [(0.1, 0.5), (1.0, 1.5)]
    assert result["speech_ms"] == 900.0


def test_a_clip_with_no_segments_has_no_speech():
    result = vad.segments_from_result({"timing": {"wall_ms": 3.0}}, fallback_rate = 16000)
    assert result == {"segments": [], "speech_ms": 0.0, "sample_rate": 16000}


def test_the_default_detector_is_the_earshot_copy_that_ships_with_studio():
    assert vad.DEFAULT_VAD_MODEL == "builtin/earshot-vad"
    model = vad.resolve_vad_model(None)
    assert (model.family, model.task, model.server_task) == ("earshot_vad", "vad", "vad")
    # The runtime loads the folder; the weights are in the repo, so there is no download.
    assert model.is_package and model.repo_id is None
    assert (vad._ASSETS / "earshot_vad" / "earshot_vad.safetensors").stat().st_size > 40_000
    assert model.local_path == str(vad._ASSETS / "earshot_vad")
    assert acm.family_policy("pulsevad").task == "vad"


def test_the_weights_studio_ships_are_not_git_ignored():
    """``*.safetensors`` is ignored repo-wide: without the exception the default detector is never committed."""
    import subprocess

    weights = vad._ASSETS / "earshot_vad" / "earshot_vad.safetensors"
    try:
        check = subprocess.run(
            ["git", "check-ignore", "-q", weights.name],
            cwd = weights.parent,
            capture_output = True,
            timeout = 30,
        )
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("git is not available")
    if check.returncode not in (0, 1):
        pytest.skip("not a git checkout")
    assert check.returncode == 1


def test_a_builtin_detector_loads_on_the_cpu_with_nothing_to_download(monkeypatch):
    from core.inference import audio_cpp_backend, stt_sidecar

    monkeypatch.setattr(
        audio_cpp_backend.AudioCppBackend,
        "_download_missing",
        staticmethod(lambda *a: pytest.fail("downloaded")),
    )
    monkeypatch.setattr(
        clip.audio_cpp_files, "materialize", lambda e: pytest.fail("read the Hub cache")
    )
    launched = []

    def start(
        model,
        path,
        force_cpu = False,
        max_threads = None,
    ):
        launched.append((model.family, path, force_cpu))
        return SimpleNamespace(process = SimpleNamespace(poll = lambda: None), stop = lambda: None)

    monkeypatch.setattr(clip.AudioCppServer, "start", staticmethod(start))
    # Not training: a Hub detector would go to the GPU here.
    monkeypatch.setattr(stt_sidecar, "_training_active", lambda: False)
    sidecar = vad.AudioCppVadSidecar()

    assert sidecar.download(None) == "builtin/earshot-vad"
    assert sidecar.load(None) == "builtin/earshot-vad"
    assert launched == [("earshot_vad", str(vad._ASSETS / "earshot_vad"), True)]
    assert sidecar.on_cpu is True
    listed = {d["id"]: d for d in sidecar.status()["detectors"]}
    assert listed["builtin/earshot-vad"] == {
        "id": "builtin/earshot-vad",
        "label": "Earshot VAD",
        "available": True,
    }
    # Every clip asks for a load: a live built-in answers without looking at the disk again.
    monkeypatch.setattr(vad, "resolve_vad_model", lambda *a, **k: pytest.fail("resolved again"))
    assert sidecar.load("Builtin/Earshot-VAD") == "builtin/earshot-vad"
    assert len(launched) == 1


def test_status_answers_while_a_clip_check_holds_the_lock():
    """The picker reads the built-in list from status; a slow detector must not hold it up."""
    import threading

    sidecar = vad.AudioCppVadSidecar()
    answered = []
    with sidecar._lock:
        reader = threading.Thread(target = lambda: answered.append(sidecar.status()))
        reader.start()
        reader.join(timeout = 10)
    assert answered and answered[0]["default_model"] == "builtin/earshot-vad"
    assert answered[0]["loaded_model"] is None


@pytest.mark.parametrize(
    "model, gated",
    [
        (None, None),
        ("builtin/earshot-vad", None),
        ("Builtin/Silero-VAD", None),
        # Unknown built-in: the load refuses it (422); it is not a Hub repo to check access for.
        ("builtin/no-such-detector", None),
        (PULSEVAD, acm.AUDIO_CPP_REPO),
    ],
)
def test_model_access_gates_hub_detectors_only(monkeypatch, model, gated):
    import asyncio

    from routes import inference

    asked = []
    monkeypatch.setattr(inference.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(inference.account_access, "require_model_access", asked.append)
    asyncio.run(inference._require_vad_model_access(model))
    assert asked == ([gated] if gated else [])


def test_only_vad_models_resolve(monkeypatch):
    kokoro = acm.family_policy("kokoro_tts")
    fake = acm.AudioCppModel(
        id = f"{acm.AUDIO_CPP_REPO}/Kokoro-82M-GGUF",
        repo_id = acm.AUDIO_CPP_REPO,
        folder = "Kokoro-82M-GGUF",
        display_name = "Kokoro-82M-GGUF",
        family = kokoro.family,
        task = kokoro.task,
        server_task = kokoro.default_server_task,
        variant = acm.AudioCppVariant("Q8_0", (), "x.gguf"),
        variants = (),
        default_variant = "Q8_0",
    )
    monkeypatch.setattr(clip, "resolve", lambda *a, **k: fake)
    with pytest.raises(vad.VadError, match = "not a voice activity detection model"):
        vad.resolve_vad_model(fake.id)
    with pytest.raises(vad.VadError, match = "not an audio GGUF"):
        vad.resolve_vad_model("not a model id at all!!")


def test_a_download_drops_the_cached_miss_so_the_next_load_sees_it(monkeypatch):
    """The load route downloads on a miss and loads again at once; that load must not reuse the miss."""
    entry = SimpleNamespace(id = PULSEVAD)
    monkeypatch.setattr(vad, "resolve_vad_model", lambda *a, **k: entry)
    from core.inference import audio_cpp_backend

    monkeypatch.setattr(
        audio_cpp_backend.AudioCppBackend, "_download_missing", staticmethod(lambda *a: True)
    )
    miss = (PULSEVAD.lower(), "", False, "", (), "")
    monkeypatch.setitem(acm._resolve_cache, miss, (time.monotonic(), None))

    assert vad.AudioCppVadSidecar().download(PULSEVAD) == PULSEVAD
    assert miss not in acm._resolve_cache


def test_the_detector_runs_on_the_cpu_while_training_holds_the_gpu(monkeypatch, tmp_path):
    """As the STT sidecar does: no second GPU runtime beside training, and back to the GPU after."""
    from core.inference import stt_sidecar

    entry = SimpleNamespace(id = PULSEVAD, display_name = "PulseVAD")
    monkeypatch.setattr(vad, "resolve_vad_model", lambda *a, **k: entry)
    monkeypatch.setattr(clip.audio_cpp_files, "materialize", lambda e: tmp_path / "vad.gguf")
    launched = []

    def start(
        model,
        path,
        force_cpu = False,
        max_threads = None,
    ):
        launched.append(force_cpu)
        return SimpleNamespace(process = SimpleNamespace(poll = lambda: None), stop = lambda: None)

    monkeypatch.setattr(clip.AudioCppServer, "start", staticmethod(start))
    training = {"on": True}
    monkeypatch.setattr(stt_sidecar, "_training_active", lambda: training["on"])
    sidecar = vad.AudioCppVadSidecar()

    sidecar.load(PULSEVAD)
    sidecar.load(PULSEVAD)
    training["on"] = False
    sidecar.load(PULSEVAD)
    assert launched == [True, False]


@pytest.mark.parametrize("on_cpu", [False, True])
def test_training_admission_frees_a_gpu_placed_detector(monkeypatch, on_cpu):
    """Training start releases the detector unless it already runs on the CPU, outside the STT list."""
    from routes import training_vram

    unloaded = []
    fake = SimpleNamespace(
        loaded_model = vad.DEFAULT_VAD_MODEL,
        on_cpu = on_cpu,
        unload = lambda: unloaded.append(True) or True,
    )
    monkeypatch.setattr(vad, "get_audio_cpp_vad_sidecar", lambda: fake)

    monkeypatch.setattr(training_vram, "free_prompt_upsampler_for_training", lambda reason: [])
    monkeypatch.setattr(training_vram, "_coordinate_resident_models", lambda can_keep: [])

    freed = training_vram.coordinate_models_for_training(lambda: (True, {}))

    assert (f"vad:{vad.DEFAULT_VAD_MODEL}" in freed) is (not on_cpu)
    # The STT release list, which admission reads as "STT let go", never carries the detector.
    assert not any(
        f.startswith("vad:") for f in training_vram.free_stt_model_for_training(reason = "test")
    )
    assert unloaded == ([] if on_cpu else [True])


def test_a_detect_keeps_the_server_it_loaded_until_the_run_ends(monkeypatch):
    import io
    import threading
    import wave

    sidecar = vad.AudioCppVadSidecar()
    held_during_load = []

    def lock_free_elsewhere() -> bool:
        box = []
        probe = threading.Thread(target = lambda: box.append(sidecar._lock.acquire(blocking = False)))
        probe.start()
        probe.join()
        if box[0]:
            sidecar._lock.release()
        return box[0]

    class Server:
        model_id = "pulsevad"
        process = type("P", (), {"poll": lambda self: None})()

        def post_json(self, path, body, timeout):
            held_during_load.append(not lock_free_elsewhere())
            return 200, b'{"segments": [], "sample_rate": 16000}'

    def load(model):
        held_during_load.append(not lock_free_elsewhere())
        sidecar._server = Server()
        return "pulsevad"

    monkeypatch.setattr(sidecar, "load", load)
    monkeypatch.setattr(vad, "_debug_dump", lambda *a: None)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    result = sidecar.detect(buf.getvalue(), "pulsevad")
    assert result["model"] == "pulsevad"
    # The bars travel with the answer, so the browser needs no table of detectors.
    assert result["bars"] == {"utterance_ms": 300.0, "barge_ms": 200.0}
    assert held_during_load == [True, True]


def test_a_download_resolves_the_repo_with_the_callers_token(monkeypatch):
    from core.inference import audio_cpp_backend

    seen: list = []
    entry = SimpleNamespace(id = "org/private-vad-GGUF")

    def _resolve(
        model,
        *,
        network = False,
        hf_token = None,
    ):
        seen.append(("resolve", hf_token))
        return entry

    monkeypatch.setattr(vad, "resolve_vad_model", _resolve)
    monkeypatch.setattr(
        audio_cpp_backend.AudioCppBackend,
        "_download_missing",
        staticmethod(lambda e, token: seen.append(("download", token))),
    )
    from core.inference import audio_cpp_clip_sidecar

    monkeypatch.setattr(
        audio_cpp_clip_sidecar,
        "forget",
        lambda identifier = None: seen.append(("forget", identifier)),
    )
    monkeypatch.setattr(vad.AudioCppVadSidecar, "__init__", lambda self: None)
    sidecar = vad.AudioCppVadSidecar()
    assert sidecar.download("org/private-vad-GGUF", "hf_secret") == "org/private-vad-GGUF"
    # The memoized "not downloaded" answer is dropped, so the load right after finds the files.
    assert seen == [
        ("resolve", "hf_secret"),
        ("download", "hf_secret"),
        ("forget", "org/private-vad-GGUF"),
    ]


def test_a_detector_refused_during_an_audio_runtime_update_is_retryable():
    from core.inference import audio_cpp_server
    from core.inference.vad_audiocpp_sidecar import VadUnavailableError
    from routes import inference

    audio_cpp_server.UPDATE_IN_PROGRESS.set()
    try:
        assert inference._clip_http_error(VadUnavailableError("updating"), "VAD").status_code == 503
    finally:
        audio_cpp_server.UPDATE_IN_PROGRESS.clear()
    assert inference._clip_http_error(VadUnavailableError("no runtime"), "VAD").status_code == 501
