# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The audio.cpp STT sidecar against a fake ``audiocpp_server`` that records requests."""

import json
import sys
import textwrap
from pathlib import Path

import pytest

from core.inference import audio_cpp_backend
from core.inference import audio_cpp_models as acm
from core.inference import audio_cpp_server as srv
from core.inference import stt_audiocpp_sidecar as stt
from core.inference.audio_cpp_models import AUDIO_CPP_REPO
from core.inference.stt_sidecar import SttAudioDecodeError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_audio_cpp_models import _gguf_bytes, _put, _snapshot, hub  # noqa: E402, F401
from test_audio_inputs import wav_bytes  # noqa: E402

QWEN3 = f"{AUDIO_CPP_REPO}/Qwen3-ASR-0.6B-GGUF"
MOSS = f"{AUDIO_CPP_REPO}/MOSS-Transcribe-Diarize-GGUF"
NIAGARA = f"{AUDIO_CPP_REPO}/Niagara-ASR-GGUF"
ALIGNER_FILE = "Qwen3-ForcedAligner-0.6B-GGUF/qwen3-forced-aligner-0.6b-q8_0.gguf"
ALIGNER_KEY = "qwen3_asr.forced_aligner_model_path"

FAKE_SERVER = textwrap.dedent(
    r"""
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    here = os.path.dirname(os.path.abspath(__file__))
    cfg = json.load(open(sys.argv[sys.argv.index("--config") + 1], encoding="utf-8"))
    entry = cfg["models"][0]
    with open(os.path.join(here, "starts.jsonl"), "a", encoding="utf-8") as log:
        log.write(json.dumps({"backend": cfg["backend"], "entry": entry}) + "\n")

    SPANS = [(1920, 16000, "Hello there.", "S01"), (17600, 32000, "General Kenobi.", "S02")]
    MOSS = {
        "text": "[0.12][S01] Hello there.[1.00][1.10][S02] General Kenobi.[2.00]",
        "segments": [{"start_sample": a, "end_sample": b, "text": t} for a, b, t, _ in SPANS],
        "speaker_turns": [{"start_sample": a, "end_sample": b, "speaker_id": s} for a, b, _, s in SPANS],
        "sample_rate": 16000,
    }

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def reply(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            if self.path == "/health":
                return self.reply(200, {"status": "ok"})
            if self.path == "/v1/models":
                return self.reply(200, {"data": [{"id": entry["id"]}]})
            self.reply(404, {})
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            audio = req.get("audio")
            record = {"path": self.path, "json": req, "audio_existed": bool(audio) and os.path.isfile(audio)}
            with open(os.path.join(here, "requests.jsonl"), "a", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")
            language = req.get("language")
            if language == "xx":
                return self.reply(400, {"error": {"message": "unsupported language"}})
            if language == "boom":
                return self.reply(500, {"error": {"message": "decoder exploded"}})
            if entry["family"] == "moss_transcribe_diarize":
                return self.reply(200, MOSS)
            payload = {"text": "Concord returned.", "language": "English"}
            if (req.get("options") or {}).get("return_timestamps") == "true":
                payload["words"] = [
                    {"word": "Concord", "start_sample": 9088, "end_sample": 19328},
                    {"word": "returned.", "start_sample": 19328, "end_sample": 25728},
                ]
            self.reply(200, payload)

    ThreadingHTTPServer(("127.0.0.1", cfg["port"]), H).serve_forever()
    """
)


def _add(hub, rel, family):
    path = _put(_snapshot(hub), rel, _gguf_bytes(family = family))
    acm.forget()
    with acm._resolve_lock:
        acm._downloaded_cache.clear()
    return path


class Fake:
    def __init__(self, root: Path):
        self.root = root

    def log(self, name):
        return [json.loads(line) for line in (self.root / name).read_text("utf-8").splitlines()]

    bodies = property(lambda self: [r["json"] for r in self.log("requests.jsonl")])
    starts = property(lambda self: self.log("starts.jsonl"))


@pytest.fixture
def fake(tmp_path, monkeypatch, hub):
    _add(hub, "Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf", "qwen3_asr")
    _add(hub, "MOSS-Transcribe-Diarize-GGUF/moss-q8_0.gguf", "moss_transcribe_diarize")
    _add(hub, "Niagara-ASR-GGUF/niagara-19m-batch.en-f32.gguf", "niagara_asr")
    root = tmp_path / "fake"
    root.mkdir()
    for log in ("starts.jsonl", "requests.jsonl"):
        (root / log).touch()
    script = root / "fake_server.py"
    script.write_text(FAKE_SERVER, encoding = "utf-8")
    binary = tmp_path / srv.BINARY_NAME
    binary.write_bytes(b"")
    real_popen = srv.subprocess.Popen

    def launch(command, *args, **kwargs):
        if command and command[0] == str(binary):
            command = [sys.executable, str(script), *command[1:]]
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(srv.subprocess, "Popen", launch)
    monkeypatch.setattr(srv, "ensure_binary", lambda: str(binary))
    monkeypatch.setattr(srv, "model_runtime_problem", lambda model, binary = None: None)
    monkeypatch.setattr(
        srv, "select_backend", lambda binary, force_cpu: "cpu" if force_cpu else "cuda"
    )
    monkeypatch.setattr(stt, "_training_active", lambda: False)
    monkeypatch.setenv("UNSLOTH_AUDIO_DEVICE", "auto")
    return Fake(root)


@pytest.fixture
def side():
    sidecar = stt.AudioCppSttSidecar()
    yield sidecar
    sidecar.unload()


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.wav"
    path.write_bytes(wav_bytes(2.0))
    return path


def test_bytes_go_as_json_naming_a_temp_file_removed_after_success_or_error(fake, side):
    result = side.transcribe(wav_bytes(2.0), QWEN3, "en")
    assert result["text"] == "Concord returned." and result["language"] == "en"
    assert result["duration"] == pytest.approx(2.0)
    assert not {"segments", "words", "speakers"} & set(result)
    (record,) = fake.log("requests.jsonl")
    body, audio = record["json"], Path(record["json"]["audio"])
    assert record["path"] == "/v1/audio/transcriptions/details"
    assert audio.is_absolute() and audio.parent == stt._stt_tmp_dir() and audio.suffix == ".wav"
    assert record["audio_existed"] and not list(audio.parent.glob("*.wav"))
    assert (body["language"], body["model"]) == ("en", fake.starts[0]["entry"]["id"])
    assert "options" not in body
    with pytest.raises(stt.SttEngineUnavailableError):
        side.transcribe(wav_bytes(), QWEN3, "boom")
    assert not list(audio.parent.glob("*.wav"))
    # A family that rejects the language is asked again without it.
    assert side.transcribe(wav_bytes(), QWEN3, "xx")["text"] == "Concord returned."
    assert fake.bodies[-2]["language"] == "xx" and "language" not in fake.bodies[-1]


def test_moss_reads_the_source_in_place_with_speakers_and_no_options(fake, side, source):
    result = side.transcribe_path(source, MOSS, None, timestamps = True)
    (body,) = fake.bodies
    assert body["audio"] == str(source) and "options" not in body
    assert source.exists()  # the caller's file is not ours to remove
    assert result["text"] == "Hello there. General Kenobi." and result["duration"] == 2.0
    assert result["speakers"] == ["S01", "S02"]
    assert [(s["start"], s["end"], s["speaker"]) for s in result["segments"]] == [
        (0.12, 1.0, "S01"),
        (1.1, 2.0, "S02"),
    ]
    # MOSS loads without the aligner whatever the request asked.
    assert ALIGNER_KEY not in json.dumps(fake.starts[0]["entry"])
    assert side.transcribe(wav_bytes(), MOSS, None)["text"] == "Hello there. General Kenobi."


def test_qwen3_aligner_is_downloaded_loaded_once_and_kept(fake, side, source, hub, monkeypatch):
    real_resolve = audio_cpp_backend._resolve_companion
    downloads, phases = [], []

    def resolve(
        model,
        companion,
        hf_token = None,
        *,
        network = True,
    ):
        if network:
            # Stands in for the Hub listing: the files arrive with the download below.
            _add(hub, ALIGNER_FILE, "qwen3_forced_aligner")
        return real_resolve(model, companion, hf_token, network = False)

    monkeypatch.setattr(audio_cpp_backend, "_resolve_companion", resolve)
    download = staticmethod(
        lambda model, token, cancel_event = None: downloads.append(model.id) or True
    )
    monkeypatch.setattr(audio_cpp_backend.AudioCppBackend, "_download_missing", download)

    def run(**kwargs):
        phases.clear()
        return side.transcribe_path(source, QWEN3, None, on_phase = phases.append, **kwargs)

    assert run() == {
        "text": "Concord returned.",
        "language": "English",
        "duration": 2.0,
        "model": QWEN3,
    }
    assert "options" not in fake.bodies[-1] and "session_options" not in fake.starts[0]["entry"]
    result = run(timestamps = True)
    assert phases == ["downloading_aligner", "loading", "transcribing"]
    assert downloads == [stt.QWEN3_ALIGNER.id] and len(fake.starts) == 2
    assert Path(fake.starts[1]["entry"]["session_options"][ALIGNER_KEY]).suffix == ".gguf"
    assert fake.bodies[-1]["options"] == {
        "return_timestamps": "true",
        "audio_chunk_mode": "fixed",
        "qwen3_asr.preserve_punctuation": "true",
    }
    assert result["words"] == [
        {"start": 0.568, "end": 1.208, "word": "Concord"},
        {"start": 1.208, "end": 1.608, "word": "returned."},
    ]
    assert result["segments"] == [{"start": 0.568, "end": 1.608, "text": "Concord returned."}]
    run(timestamps = True)
    assert phases == ["transcribing"] and len(downloads) == 1
    run()
    side.transcribe(wav_bytes(), QWEN3, None)
    assert len(fake.starts) == 2 and all("options" not in b for b in fake.bodies[-2:])


def test_stop_during_the_aligner_download_cancels_it(fake, side, source, hub, monkeypatch):
    """Stop while "Downloading the timing aligner" only cancelled the transcription that never
    started; the download itself read no cancel event and kept streaming."""
    import threading

    from core.inference.audio_cpp_backend import AudioCppRequestCancelledError
    from core.inference.audio_cpp_backend import AudioCppBackend as backend_cls

    import inspect

    assert "cancel_event.is_set()" in inspect.getsource(backend_cls._download_missing)
    real_resolve = audio_cpp_backend._resolve_companion

    def resolve(
        model,
        companion,
        hf_token = None,
        *,
        network = True,
    ):
        if network:
            _add(hub, ALIGNER_FILE, "qwen3_forced_aligner")
        return real_resolve(model, companion, hf_token, network = False)

    monkeypatch.setattr(audio_cpp_backend, "_resolve_companion", resolve)
    seen = []

    def download(
        model,
        token,
        cancel_event = None,
    ):
        seen.append(cancel_event)
        cancel_event.set()
        raise AudioCppRequestCancelledError("Request cancelled.")

    monkeypatch.setattr(backend_cls, "_download_missing", staticmethod(download))
    cancel = threading.Event()
    with pytest.raises(stt.SttTranscriptionCancelledError):
        side.transcribe_path(source, QWEN3, None, timestamps = True, cancel_event = cancel)
    assert seen == [cancel]


def test_the_aligner_preflight_carries_the_request_cancel(fake, side, source, hub, monkeypatch):
    """The route's preflight (``ensure_aligner``) is the call that does the first 1.1 GB download;
    without the event it ran uncancellable and the forwarding in ``load`` came too late."""
    import threading

    from core.inference.audio_cpp_backend import AudioCppBackend as backend_cls

    real_resolve = audio_cpp_backend._resolve_companion

    def resolve(
        model,
        companion,
        hf_token = None,
        *,
        network = True,
    ):
        if network:
            _add(hub, ALIGNER_FILE, "qwen3_forced_aligner")
        return real_resolve(model, companion, hf_token, network = False)

    monkeypatch.setattr(audio_cpp_backend, "_resolve_companion", resolve)
    seen = []
    monkeypatch.setattr(
        backend_cls,
        "_download_missing",
        staticmethod(lambda model, token, cancel_event = None: seen.append(cancel_event)),
    )
    cancel = threading.Event()
    side.ensure_aligner(QWEN3, None, cancel)
    assert seen == [cancel]


def test_stop_mid_file_returns_before_the_download_finishes(monkeypatch):
    """The aligner is one 1.1 GB file, so a check between files never fires while it streams. The
    download runs on its own thread and the caller returns on Stop while the file is still coming."""
    import threading
    from types import SimpleNamespace

    from core.inference.audio_cpp_backend import AudioCppRequestCancelledError
    from core.inference.audio_cpp_backend import AudioCppBackend as backend_cls

    entered, release = threading.Event(), threading.Event()

    def streaming(*_args, **_kwargs):
        entered.set()
        release.wait(10)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", streaming)
    monkeypatch.setattr(
        audio_cpp_backend.audio_cpp_files, "missing_files", lambda model: [("aligner.gguf", 1)]
    )
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    with pytest.raises(AudioCppRequestCancelledError):
        backend_cls._download_missing(SimpleNamespace(repo_id = "org/aligner"), None, cancel)
    assert entered.is_set() and not release.is_set()
    release.set()


def test_a_retry_joins_the_download_that_stop_left_running(monkeypatch):
    """Stop then retry: the file is still missing, so without sharing the retry would start a second
    transfer of the same 1.1 GB and each cycle would leave another thread behind."""
    import threading
    from types import SimpleNamespace

    from core.inference.audio_cpp_backend import AudioCppRequestCancelledError
    from core.inference.audio_cpp_backend import AudioCppBackend as backend_cls

    release, calls = threading.Event(), []

    def streaming(*_args, **_kwargs):
        calls.append(1)
        release.wait(10)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", streaming)
    monkeypatch.setattr(
        audio_cpp_backend.audio_cpp_files, "missing_files", lambda model: [("aligner.gguf", 1)]
    )
    model = SimpleNamespace(repo_id = "org/aligner")
    stopped = threading.Event()
    threading.Timer(0.3, stopped.set).start()
    with pytest.raises(AudioCppRequestCancelledError):
        backend_cls._download_missing(model, None, stopped)
    outcome = []
    retry = threading.Thread(
        target = lambda: outcome.append(backend_cls._download_missing(model, None, threading.Event()))
    )
    retry.start()
    threading.Timer(0.3, release.set).start()
    retry.join(10)
    assert outcome == [True]
    assert calls == [1]
    assert audio_cpp_backend._inflight == {}


def test_a_retry_into_a_relocated_cache_starts_its_own_download(monkeypatch):
    """Joining is keyed by cache root too: after Stop, a retry with the Hub cache moved in Settings
    would otherwise wait on a transfer writing into the old cache and report the new one filled."""
    import threading
    from pathlib import Path
    from types import SimpleNamespace

    from core.inference.audio_cpp_backend import AudioCppRequestCancelledError
    from core.inference.audio_cpp_backend import AudioCppBackend as backend_cls
    from utils import hf_cache_settings

    release, cache_dirs = threading.Event(), []

    def streaming(*_args, cache_dir, **_kwargs):
        cache_dirs.append(cache_dir)
        release.wait(10)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", streaming)
    monkeypatch.setattr(
        audio_cpp_backend.audio_cpp_files, "missing_files", lambda model: [("aligner.gguf", 1)]
    )
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: Path("/cache/old"))
    model = SimpleNamespace(repo_id = "org/aligner")
    stopped = threading.Event()
    threading.Timer(0.3, stopped.set).start()
    with pytest.raises(AudioCppRequestCancelledError):
        backend_cls._download_missing(model, None, stopped)
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: Path("/cache/new"))
    stopped = threading.Event()
    threading.Timer(0.3, stopped.set).start()
    with pytest.raises(AudioCppRequestCancelledError):
        backend_cls._download_missing(model, None, stopped)
    assert [Path(c).name for c in cache_dirs] == ["old", "new"]
    release.set()


def test_a_download_error_still_reaches_the_caller_with_a_cancel_event(monkeypatch):
    import threading
    from types import SimpleNamespace

    from core.inference.audio_cpp_backend import AudioCppBackend as backend_cls

    def failing(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", failing)
    monkeypatch.setattr(
        audio_cpp_backend.audio_cpp_files, "missing_files", lambda model: [("aligner.gguf", 1)]
    )
    with pytest.raises(OSError, match = "disk full"):
        backend_cls._download_missing(
            SimpleNamespace(repo_id = "org/aligner"), None, threading.Event()
        )


def test_an_aligner_download_failure_says_how_to_go_on(fake, side, source, monkeypatch):
    def offline(*_args, **_kwargs):
        raise RuntimeError(
            "Qwen3-ASR needs Qwen3-ForcedAligner-0.6B-GGUF, which Studio could not find."
        )

    monkeypatch.setattr(audio_cpp_backend, "_resolve_companion", offline)
    with pytest.raises(stt.SttModelNotDownloadedError, match = "Turn off Timestamps"):
        side.transcribe_path(source, QWEN3, None, timestamps = True)
    assert fake.starts == []


def test_niagara_runs_on_the_cpu_even_when_the_gpu_is_asked_for(fake, side):
    side.load(NIAGARA, device = "gpu")
    side.load(NIAGARA, device = "gpu")
    side.transcribe(wav_bytes(), NIAGARA, None)
    assert [s["backend"] for s in fake.starts] == ["cpu"]
    assert side.device == "cpu" and side._gpu_disabled is True


def test_an_unreadable_source_is_a_decode_error(fake, side, tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav")
    with pytest.raises(SttAudioDecodeError):
        side.transcribe_path(bad, QWEN3, None)
    assert fake.starts == []
