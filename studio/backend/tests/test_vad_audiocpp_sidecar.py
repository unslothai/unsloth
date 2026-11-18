# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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


def test_a_load_right_after_a_download_finds_the_model_it_fetched(monkeypatch, tmp_path):
    """A load straight after a download must find the model the download fetched."""
    from core.inference import audio_cpp_backend, stt_sidecar
    from utils import hf_cache_settings

    hub = [acm.RepoFile("PulseVAD-GGUF/pulsevad.gguf", 1024)]
    # What the local cache lists: nothing until a download has fetched the repo.
    cache = []

    def list_files(repo_id, folder = "", hf_token = None, *, network = True):
        return (list(hub), True) if network else (list(cache), False)

    def download_missing(entry, hf_token):
        cache[:] = hub
        return True

    launched = []

    def start(model, path, force_cpu = False, max_threads = None):
        launched.append((model.id, path, force_cpu))
        return SimpleNamespace(process = SimpleNamespace(poll = lambda: None), stop = lambda: None)

    acm.forget()
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: str(tmp_path))
    monkeypatch.setattr(acm, "list_files", list_files)
    # No header read over the network: the family comes from the file names.
    monkeypatch.setattr(acm, "read_remote_header", lambda *a, **k: None)
    monkeypatch.setattr(
        audio_cpp_backend.AudioCppBackend, "_download_missing", staticmethod(download_missing)
    )
    monkeypatch.setattr(clip.audio_cpp_files, "materialize", lambda e: "pulsevad.gguf")
    monkeypatch.setattr(clip.AudioCppServer, "start", staticmethod(start))
    monkeypatch.setattr(stt_sidecar, "_training_active", lambda: False)
    sidecar = vad.AudioCppVadSidecar()

    with pytest.raises(clip.ClipModelNotDownloadedError):
        sidecar.load(PULSEVAD)
    assert sidecar.download(PULSEVAD) == PULSEVAD
    assert sidecar.load(PULSEVAD) == PULSEVAD
    assert sidecar.loaded_model == PULSEVAD
    assert launched == [(PULSEVAD, "pulsevad.gguf", False)]


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
    """An unload asked for during a clip waits for the clip to finish before it stops the server."""
    import io
    import threading
    import wave

    from core.inference import stt_sidecar

    monkeypatch.setattr(
        vad,
        "resolve_vad_model",
        lambda *a, **k: SimpleNamespace(id = "pulsevad", display_name = "PulseVAD"),
    )
    monkeypatch.setattr(clip.audio_cpp_files, "materialize", lambda e: "pulsevad.gguf")
    monkeypatch.setattr(stt_sidecar, "_training_active", lambda: False)
    monkeypatch.delenv("UNSLOTH_VAD_DEBUG_DIR", raising = False)
    in_run = threading.Event()
    answer = threading.Event()
    stopped = []

    class Server:
        model_id = "pulsevad"
        process = SimpleNamespace(poll = lambda: None)

        def post_json(self, path, body, timeout):
            in_run.set()
            answer.wait(10)
            return 200, b'{"segments": [], "sample_rate": 16000}'

        def stop(self):
            stopped.append(True)

    monkeypatch.setattr(clip.AudioCppServer, "start", staticmethod(lambda *a, **k: Server()))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    sidecar = vad.AudioCppVadSidecar()
    detected, unloaded = [], []
    clip_thread = threading.Thread(
        target = lambda: detected.append(sidecar.detect(buf.getvalue(), "pulsevad"))
    )
    unload_thread = None
    clip_thread.start()
    try:
        assert in_run.wait(10)
        unload_thread = threading.Thread(target = lambda: unloaded.append(sidecar.unload()))
        unload_thread.start()
        unload_thread.join(timeout = 0.5)
        # The unload is still waiting behind the clip, so the server has not been stopped.
        assert unload_thread.is_alive()
        assert stopped == []
    finally:
        answer.set()
        clip_thread.join(10)
        if unload_thread is not None:
            unload_thread.join(10)
    assert detected[0]["model"] == "pulsevad"
    assert detected[0]["bars"] == {"utterance_ms": 300.0, "barge_ms": 200.0}
    assert unloaded == [True]
    assert stopped == [True]


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
