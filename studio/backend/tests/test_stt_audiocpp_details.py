# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The audio.cpp STT sidecar against a fake ``audiocpp_server`` that records requests."""

import io
import json
import struct
import sys
import textwrap
import threading
import time
import wave
from pathlib import Path

import pytest

from core.inference import audio_cpp_backend, audio_cpp_files
from core.inference import audio_cpp_models as acm
from core.inference import audio_cpp_server as srv
from core.inference import stt_audiocpp_sidecar as stt
from core.inference.audio_cpp_models import AUDIO_CPP_REPO
from core.inference.stt_sidecar import (
    SttAudioDecodeError,
    SttTranscriptionCancelledError,
)

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
        log.write(json.dumps({"backend": cfg["backend"], "entry": entry, "pid": os.getpid()}) + "\n")

    MOSS = {
        "text": "[0.12][S01] Hello there.[1.00][1.10][S02] General Kenobi.[2.00]",
        "segments": [
            {"start_sample": 1920, "end_sample": 16000, "text": "Hello there."},
            {"start_sample": 17600, "end_sample": 32000, "text": "General Kenobi."},
        ],
        "speaker_turns": [
            {"start_sample": 1920, "end_sample": 16000, "speaker_id": "S01"},
            {"start_sample": 17600, "end_sample": 32000, "speaker_id": "S02"},
        ],
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
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            req = {"path": self.path, "content_type": self.headers.get("Content-Type")}
            try:
                req["json"] = json.loads(body)
            except ValueError:
                req["json"] = None
            audio = (req["json"] or {}).get("audio")
            req["audio_existed"] = bool(audio) and os.path.isfile(audio)
            with open(os.path.join(here, "requests.jsonl"), "a", encoding="utf-8") as log:
                log.write(json.dumps(req) + "\n")
            if self.path != "/v1/audio/transcriptions/details" or req["json"] is None:
                return self.reply(404, {"error": "multipart is not served here"})
            language = req["json"].get("language")
            if language == "slow":
                time.sleep(30)
            if language == "xx":
                return self.reply(400, {"error": {"message": "unsupported language"}})
            if language == "boom":
                return self.reply(500, {"error": {"message": "decoder exploded"}})
            if entry["family"] == "moss_transcribe_diarize":
                return self.reply(200, MOSS)
            payload = {"text": "Concord returned.", "language": "English"}
            if (req["json"].get("options") or {}).get("return_timestamps") == "true":
                payload["words"] = [
                    {"word": "Concord", "start_sample": 9088, "end_sample": 19328},
                    {"word": "returned.", "start_sample": 19328, "end_sample": 25728},
                ]
                payload["sample_rate"] = 16000
            self.reply(200, payload)

    ThreadingHTTPServer(("127.0.0.1", cfg["port"]), H).serve_forever()
    """
)


def _gguf_bytes(family) -> bytes:
    kv = [
        ("general.architecture", 8, "audiocpp"),
        ("audiocpp.model_spec.version", 4, 1),
        ("audiocpp.model_spec.family", 8, family),
    ]
    out = bytearray(struct.pack("<IIQQ", 0x46554747, 3, 0, len(kv)))
    for key, vtype, value in kv:
        k = key.encode()
        out += struct.pack("<Q", len(k)) + k + struct.pack("<I", vtype)
        if vtype == 8:
            v = value.encode()
            out += struct.pack("<Q", len(v)) + v
        else:
            out += struct.pack("<I", value)
    return bytes(out) + b"\0" * 64


def _put(hub, rel, family):
    repo_dir = hub / ("models--" + AUDIO_CPP_REPO.replace("/", "--"))
    snap = repo_dir / "snapshots" / ("a" * 40)
    (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs" / "main").write_text("a" * 40)
    path = snap / rel
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(_gguf_bytes(family))
    acm.forget()
    with acm._resolve_lock:
        acm._downloaded_cache.clear()
    return path


def _wav(seconds = 2.0, rate = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(seconds * rate))
    return buf.getvalue()


class Fake:
    def __init__(self, root: Path):
        self.root = root

    def _lines(self, name):
        path = self.root / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding = "utf-8").splitlines()]

    @property
    def requests(self):
        return self._lines("requests.jsonl")

    @property
    def starts(self):
        return self._lines("starts.jsonl")


@pytest.fixture
def hub(tmp_path, monkeypatch):
    root = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setattr(acm, "_hub_cache", lambda: root)
    monkeypatch.setattr(audio_cpp_files, "_hub_cache", lambda: root)
    monkeypatch.setattr(acm, "runtime_spec", lambda family: None)
    monkeypatch.setattr(acm, "runtime_knows_family", lambda family: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    acm.forget()
    with acm._resolve_lock:
        acm._downloaded_cache.clear()
    _put(root, "Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf", "qwen3_asr")
    _put(
        root,
        "MOSS-Transcribe-Diarize-GGUF/moss-transcribe-diarize-q8_0.gguf",
        "moss_transcribe_diarize",
    )
    _put(root, "Niagara-ASR-GGUF/niagara-19m-batch.en-f32.gguf", "niagara_asr")
    yield root
    acm.forget()


@pytest.fixture
def fake(tmp_path, monkeypatch, hub):
    root = tmp_path / "fake"
    root.mkdir()
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


def _details(fake):
    return [r for r in fake.requests if r["path"] == "/v1/audio/transcriptions/details"]


def _source(
    tmp_path,
    seconds = 2.0,
    rate = 16000,
) -> Path:
    path = tmp_path / f"source-{rate}.wav"
    path.write_bytes(_wav(seconds, rate))
    return path


def test_bytes_go_as_json_naming_a_temp_file_that_is_removed_after(fake, side):
    result = side.transcribe(_wav(), QWEN3, "en")
    assert result["text"] == "Concord returned." and result["language"] == "en"
    assert result["duration"] == pytest.approx(2.0)
    assert "segments" not in result and "words" not in result and "speakers" not in result
    (request,) = _details(fake)
    assert request["content_type"] == "application/json"
    body = request["json"]
    audio = Path(body["audio"])
    assert audio.is_absolute() and audio.parent == stt._stt_tmp_dir() and audio.suffix == ".wav"
    # The server could read it while answering; it is gone now.
    assert request["audio_existed"] and not audio.exists()
    assert body["language"] == "en" and body["model"] == fake.starts[0]["entry"]["id"]
    assert "options" not in body
    assert not list(stt._stt_tmp_dir().glob("*.wav"))


def test_the_temp_file_is_removed_after_an_error_and_a_cancel(fake, side):
    with pytest.raises(stt.SttEngineUnavailableError):
        side.transcribe(_wav(), QWEN3, "boom")
    assert not list(stt._stt_tmp_dir().glob("*.wav"))
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(SttTranscriptionCancelledError):
        side.transcribe(_wav(), QWEN3, "slow", cancel_event = cancel)
    assert time.monotonic() - started < 10
    assert not list(stt._stt_tmp_dir().glob("*.wav"))
    # The cancelled server was stopped so the next request does not queue behind it.
    assert side.loaded_model is None


def test_a_rejected_language_is_retried_without_it(fake, side):
    result = side.transcribe(_wav(), QWEN3, "xx")
    assert result["text"] == "Concord returned."
    first, second = _details(fake)
    assert first["json"]["language"] == "xx" and "language" not in second["json"]


def test_a_path_transcription_names_the_source_and_moss_gets_no_options(fake, side, tmp_path):
    source = _source(tmp_path)
    result = side.transcribe_path(source, MOSS, None, timestamps = True)
    (request,) = _details(fake)
    assert request["json"]["audio"] == str(source) and "options" not in request["json"]
    assert source.exists()  # the caller's file is not ours to remove
    assert result["text"] == "Hello there. General Kenobi."
    assert result["speakers"] == ["S01", "S02"]
    assert [(s["start"], s["end"], s["speaker"]) for s in result["segments"]] == [
        (0.12, 1.0, "S01"),
        (1.1, 2.0, "S02"),
    ]
    assert result["duration"] == 2.0
    # MOSS loads without the aligner whatever the request asked.
    assert ALIGNER_KEY not in json.dumps(fake.starts[0]["entry"])
    # Through the bytes path (dictation) the markers are gone too.
    assert side.transcribe(_wav(), MOSS, None)["text"] == "Hello there. General Kenobi."


def test_qwen3_timestamps_load_the_aligner_once_and_keep_it(fake, side, tmp_path, hub):
    aligner = _put(hub, ALIGNER_FILE, "qwen3_forced_aligner")
    source = _source(tmp_path)
    phases = []

    # Off: plain text, no aligner, no options.
    result = side.transcribe_path(source, QWEN3, None, on_phase = phases.append)
    assert result["text"] == "Concord returned." and "words" not in result
    assert result["language"] == "English"
    assert "options" not in _details(fake)[-1]["json"]
    assert len(fake.starts) == 1
    assert ALIGNER_KEY not in (fake.starts[0]["entry"].get("session_options") or {})

    # On, with no aligner loaded: a restart with it, and the two request options.
    assert side.needs_reload_for(QWEN3, True) and not side.needs_reload_for(QWEN3, False)
    result = side.transcribe_path(source, QWEN3, None, timestamps = True, on_phase = phases.append)
    assert len(fake.starts) == 2
    served = fake.starts[1]["entry"]["session_options"][ALIGNER_KEY]
    assert Path(served).name == aligner.name
    assert _details(fake)[-1]["json"]["options"] == {
        "return_timestamps": "true",
        "audio_chunk_mode": "fixed",
        "qwen3_asr.preserve_punctuation": "true",
    }
    assert result["words"] == [
        {"start": 0.568, "end": 1.208, "word": "Concord"},
        {"start": 1.208, "end": 1.608, "word": "returned."},
    ]
    assert result["segments"] == [{"start": 0.568, "end": 1.608, "text": "Concord returned."}]
    # The aligner was already downloaded: no download phase.
    assert "downloading_aligner" not in phases and "loading" in phases
    assert phases[-1] == "transcribing"

    # Off again on the aligned server: no restart, and no options.
    assert not side.needs_reload_for(QWEN3, True)
    side.transcribe_path(source, QWEN3, None)
    side.transcribe(_wav(), QWEN3, None)
    assert len(fake.starts) == 2
    assert all("options" not in r["json"] for r in _details(fake)[-2:])


def test_a_missing_aligner_is_downloaded_first_with_its_phase(
    fake, side, tmp_path, hub, monkeypatch
):
    real_resolve = audio_cpp_backend._resolve_companion
    downloads = []

    def resolve(
        model,
        companion,
        hf_token = None,
        *,
        network = True,
    ):
        if network:
            # Stands in for the Hub listing: the files arrive with the download below.
            _put(hub, ALIGNER_FILE, "qwen3_forced_aligner")
        return real_resolve(model, companion, hf_token, network = False)

    monkeypatch.setattr(audio_cpp_backend, "_resolve_companion", resolve)
    monkeypatch.setattr(
        audio_cpp_backend.AudioCppBackend,
        "_download_missing",
        staticmethod(lambda model, token: downloads.append(model.id) or True),
    )
    phases = []
    side.transcribe_path(_source(tmp_path), QWEN3, None, timestamps = True, on_phase = phases.append)
    assert phases == ["downloading_aligner", "loading", "transcribing"]
    assert downloads == [stt.QWEN3_ALIGNER.id]
    assert ALIGNER_KEY in fake.starts[-1]["entry"]["session_options"]
    # Present now: a second timestamped run neither downloads nor restarts.
    phases.clear()
    side.transcribe_path(_source(tmp_path), QWEN3, None, timestamps = True, on_phase = phases.append)
    assert phases == ["transcribing"] and len(downloads) == 1 and len(fake.starts) == 1


def test_an_aligner_download_failure_says_how_to_go_on(fake, side, tmp_path, monkeypatch):
    def offline(
        model,
        companion,
        hf_token = None,
        *,
        network = True,
    ):
        raise RuntimeError(
            "Qwen3-ASR needs Qwen3-ForcedAligner-0.6B-GGUF, which Studio could not find."
        )

    monkeypatch.setattr(audio_cpp_backend, "_resolve_companion", offline)
    with pytest.raises(stt.SttModelNotDownloadedError, match = "Turn off Timestamps"):
        side.transcribe_path(_source(tmp_path), QWEN3, None, timestamps = True)
    assert fake.starts == []


def test_niagara_runs_on_the_cpu_even_when_the_gpu_is_asked_for(fake, side):
    side.load(NIAGARA, device = "gpu")
    side.load(NIAGARA, device = "gpu")
    assert [s["backend"] for s in fake.starts] == ["cpu"]
    assert side.device == "cpu" and side._gpu_disabled is True
    side.transcribe(_wav(), NIAGARA, None)
    assert len(fake.starts) == 1


def test_an_unreadable_source_is_a_decode_error(fake, side, tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav")
    with pytest.raises(SttAudioDecodeError):
        side.transcribe_path(bad, QWEN3, None)
    assert fake.starts == []
