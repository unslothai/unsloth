# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AudioCppServer against a real child process: a small fake ``audiocpp_server`` speaking the HTTP contract."""

import io
import json
import sys
import textwrap
import threading
import time
import wave

import pytest

from core.inference import audio_cpp_server as srv
from core.inference.audio_cpp_models import AudioCppModel, AudioCppVariant, RepoFile


def _model(folder: str, family: str, task: str) -> AudioCppModel:
    main = RepoFile(f"{folder}/{folder.lower()}-q8_0.gguf", 1)
    variant = AudioCppVariant("Q8_0", (main,), main.path)
    return AudioCppModel(
        id = f"audio-cpp/audio.cpp-gguf/{folder}",
        repo_id = "audio-cpp/audio.cpp-gguf",
        folder = folder,
        display_name = folder,
        family = family,
        task = task,
        server_task = task,
        variant = variant,
        variants = (variant,),
        default_variant = "Q8_0",
    )


CANARY = _model("Canary-180M-Flash-GGUF", "canary_asr", "asr")
KOKORO = _model("Kokoro-82M-GGUF", "kokoro_tts", "tts")

FAKE_SERVER = textwrap.dedent(
    r"""
    import email, io, json, sys, time, wave
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    cfg = json.load(open(sys.argv[sys.argv.index("--config") + 1], encoding="utf-8"))
    model_id = cfg["models"][0]["id"]
    listed = "wrong-id" if cfg["models"][0]["path"].endswith("wrong-id.gguf") else model_id

    def wav():
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\0\0" * 1600)
        return buf.getvalue()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def reply(self, code, body, ctype="application/json"):
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            if self.path == "/health":
                return self.reply(200, json.dumps({"status": "ok", "backend": cfg["backend"]}).encode())
            if self.path == "/v1/models":
                return self.reply(200, json.dumps({"data": [{"id": listed}]}).encode())
            self.reply(404, b"{}")
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path == "/v1/audio/speech":
                req = json.loads(body)
                if req.get("input") == "slow":
                    time.sleep(30)
                return self.reply(200, wav(), "audio/wav")
            if self.path == "/v1/audio/transcriptions":
                msg = email.message_from_bytes(b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
                fields = {}
                for part in msg.get_payload():
                    name = part.get_param("name", header="content-disposition")
                    fields[name] = part.get_filename() or part.get_payload(decode=True).decode()
                if fields.get("language") == "slow":
                    time.sleep(30)
                return self.reply(200, json.dumps({"text": json.dumps(fields, sort_keys=True)}).encode())
            if self.path == "/v1/audio/transcriptions/details":
                req = json.loads(body)
                if req.get("language") == "slow":
                    time.sleep(30)
                return self.reply(200, json.dumps({"text": json.dumps(req, sort_keys=True)}).encode())
            self.reply(404, b"{}")

    ThreadingHTTPServer(("127.0.0.1", cfg["port"]), H).serve_forever()
    """
)


@pytest.fixture
def fake_binary(tmp_path, monkeypatch):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER, encoding = "utf-8")
    binary = tmp_path / srv.BINARY_NAME
    binary.write_bytes(b"")
    real_popen = srv.subprocess.Popen

    def launch(command, *args, **kwargs):
        # The fake runs as the direct child (no shell wrapper), so stop() terminates the server itself.
        if command and command[0] == str(binary):
            command = [sys.executable, str(script), *command[1:]]
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(srv.subprocess, "Popen", launch)
    monkeypatch.setattr(srv, "ensure_binary", lambda: str(binary))
    monkeypatch.setattr(srv, "model_runtime_problem", lambda model, binary = None: None)
    monkeypatch.setattr(srv, "select_backend", lambda binary, force_cpu: "cpu")
    return binary


def test_ready_only_when_our_model_id_is_listed(fake_binary, tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_SERVER_START_TIMEOUT_SECONDS", 4.0)
    model = CANARY
    server = srv.AudioCppServer.start(model, str(tmp_path / "m.gguf"))
    try:
        assert server.alive() and server._probe()
    finally:
        server.stop()
    assert not server.alive()
    assert not server._config_dir.exists()
    # A responder that does not list this launch's id is not our child: never treated as ready.
    with pytest.raises(srv.AudioCppUnavailableError, match = "did not start in time"):
        srv.AudioCppServer.start(model, str(tmp_path / "wrong-id.gguf"))


def test_readiness_ignores_an_ambient_http_proxy(fake_binary, tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_SERVER_START_TIMEOUT_SECONDS", 6.0)
    for name in ("HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising = False)
    server = srv.AudioCppServer.start(KOKORO, str(tmp_path / "m.gguf"))
    try:
        assert server._probe()
    finally:
        server.stop()


def test_cancel_closes_the_socket_mid_request(fake_binary, tmp_path):
    server = srv.AudioCppServer.start(KOKORO, str(tmp_path / "m.gguf"))
    try:
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        started = time.monotonic()
        with pytest.raises(srv.AudioCppRequestCancelledError):
            server.post_json(
                "/v1/audio/speech",
                {"model": server.model_id, "input": "slow"},
                timeout = 20,
                cancel_event = cancel,
            )
        # Returns on the cancel, not on the server's reply or the timeout (Windows never woke the recv).
        assert time.monotonic() - started < 5
        # The server is untouched by the client-side cancel and still serves.
        _ctype, data = server.post_json(
            "/v1/audio/speech", {"model": server.model_id, "input": "hi"}
        )
        with wave.open(io.BytesIO(data)) as w:
            assert w.getframerate() == 16000
    finally:
        server.stop()


def test_transcription_details_carries_model_audio_language_and_options(fake_binary, tmp_path):
    from core.inference.stt_audiocpp_sidecar import AudioCppSttSidecar
    server = srv.AudioCppServer.start(CANARY, str(tmp_path / "m.gguf"))
    try:
        side = AudioCppSttSidecar()
        side._server = server
        audio = tmp_path / "a.wav"
        payload = side._post_details(audio, "en", None, {"timestamps": True})
        assert json.loads(payload["text"]) == {
            "audio": str(audio),
            "language": "en",
            "model": server.model_id,
            "options": {"timestamps": True},
        }
    finally:
        server.stop()


def test_cancelled_transcription_stops_the_busy_server(fake_binary, tmp_path):
    from core.inference.stt_audiocpp_sidecar import (
        AudioCppSttSidecar,
        SttTranscriptionCancelledError,
    )
    server = srv.AudioCppServer.start(CANARY, str(tmp_path / "m.gguf"))
    try:
        side = AudioCppSttSidecar()
        side._server = server
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        with side._lock, pytest.raises(SttTranscriptionCancelledError):
            side._post_details(tmp_path / "a.wav", "slow", cancel, {})
        # The child still decoding the abandoned clip is gone, so the next load starts a fresh one.
        assert side._server is None and not server.alive()
    finally:
        server.stop()


def test_threads_budget_reaches_the_command_line(fake_binary, tmp_path, monkeypatch):
    seen = {}
    launch = srv.subprocess.Popen  # the fixture's launcher

    def spy(command, *args, **kwargs):
        seen["command"] = command
        return launch(command, *args, **kwargs)

    monkeypatch.setattr(srv.subprocess, "Popen", spy)
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", "3")
    server = srv.AudioCppServer.start(KOKORO, str(tmp_path / "m.gguf"))
    server.stop()
    assert seen["command"][-2:] == ["--threads", "3"]
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", "zero")
    assert srv._cpu_threads() is None


def test_cuda_graphs_are_off_only_for_families_that_wedge_with_them(
    fake_binary, tmp_path, monkeypatch
):
    seen = []
    launch = srv.subprocess.Popen  # the fixture's launcher

    def spy(command, *args, **kwargs):
        seen.append(kwargs["env"])
        return launch(command, *args, **kwargs)

    monkeypatch.setattr(srv.subprocess, "Popen", spy)
    monkeypatch.delenv("GGML_CUDA_DISABLE_GRAPHS", raising = False)
    for model in (_model("Nemotron-3.5-ASR-Streaming-0.6B-GGUF", "nemotron_asr", "asr"), CANARY):
        srv.AudioCppServer.start(model, str(tmp_path / "m.gguf")).stop()
    assert seen[0].get("GGML_CUDA_DISABLE_GRAPHS") == "1"
    assert "GGML_CUDA_DISABLE_GRAPHS" not in seen[1]


def test_gpu_runs_cap_host_threads_unless_the_user_set_a_budget(fake_binary, tmp_path, monkeypatch):
    seen = []
    launch = srv.subprocess.Popen  # the fixture's launcher

    def spy(command, *args, **kwargs):
        seen.append(command)
        return launch(command, *args, **kwargs)

    monkeypatch.setattr(srv.subprocess, "Popen", spy)
    monkeypatch.delenv("UNSLOTH_CPU_THREADS", raising = False)
    monkeypatch.setattr(srv.os, "cpu_count", lambda: 192)
    srv.AudioCppServer.start(KOKORO, str(tmp_path / "m.gguf")).stop()  # the fixture's CPU backend
    monkeypatch.setattr(srv, "select_backend", lambda binary, force_cpu: "cuda")
    srv.AudioCppServer.start(KOKORO, str(tmp_path / "m.gguf")).stop()
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", "32")
    srv.AudioCppServer.start(KOKORO, str(tmp_path / "m.gguf")).stop()
    assert "--threads" not in seen[0]
    assert seen[1][-2:] == ["--threads", "8"] and seen[2][-2:] == ["--threads", "32"]


def test_a_custom_build_launches_on_a_backend_it_was_compiled_with(tmp_path, monkeypatch):
    """No install record: a CPU or Vulkan build on an NVIDIA host must not be asked for CUDA."""
    if sys.platform == "win32":
        pytest.skip("shell-script stand-in for the binary")
    from core.inference import audio_cpp_server as srv

    monkeypatch.delenv("UNSLOTH_AUDIO_CPP_BACKEND", raising = False)
    monkeypatch.setattr(srv.sys, "platform", "linux")
    monkeypatch.setattr(
        srv.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None
    )
    monkeypatch.setattr(srv, "_gpu_vendors", lambda: {"nvidia"})

    def build(name, backends):
        binary = tmp_path / name / "audiocpp_server"
        binary.parent.mkdir()
        binary.write_text(f"#!/bin/sh\necho 'audio.cpp custom'\necho 'backends: {backends}'\n")
        binary.chmod(0o755)
        return str(binary)

    assert srv.select_backend(build("cpu", "cpu"), False) == "cpu"
    assert srv.select_backend(build("vulkan", "cpu,vulkan"), False) == "vulkan"
    assert srv.select_backend(build("cuda", "cpu,cuda"), False) == "cuda"
    # A build too old to report keeps the host guess.
    assert srv.select_backend(build("silent", ""), False) == "cuda"


def _recorded_build(tmp_path, backend):
    binary = tmp_path / backend / srv.BINARY_NAME
    binary.parent.mkdir()
    binary.write_bytes(b"")
    (binary.parent / srv.INSTALL_RECORD).write_text(json.dumps({"backend": backend}))
    return str(binary)


@pytest.mark.parametrize(
    "backend, vendors, expected",
    [
        # CUDA_VISIBLE_DEVICES="" or a container without the NVIDIA device nodes.
        ("cuda", set(), "cpu"),
        ("cuda", {"nvidia"}, "cuda"),
        # Nothing detected at all: an unknown host keeps the build's backend.
        ("cuda", None, "cuda"),
        ("hip", {"nvidia"}, "cpu"),
        ("hip", {"amd"}, "hip"),
        ("vulkan", set(), "vulkan"),
        ("metal", {"apple"}, "metal"),
    ],
)
def test_a_gpu_build_launches_on_cpu_when_its_gpu_is_not_visible(
    tmp_path, monkeypatch, backend, vendors, expected
):
    monkeypatch.delenv("UNSLOTH_AUDIO_CPP_BACKEND", raising = False)
    monkeypatch.setattr(srv, "_gpu_vendors", lambda: vendors)
    assert srv.select_backend(_recorded_build(tmp_path, backend), False) == expected


def test_an_explicit_backend_or_cpu_request_is_not_second_guessed(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_gpu_vendors", lambda: set())
    binary = _recorded_build(tmp_path, "cuda")
    monkeypatch.setenv("UNSLOTH_AUDIO_CPP_BACKEND", "CUDA")
    assert srv.select_backend(binary, False) == "cuda"
    assert srv.select_backend(binary, True) == "cpu"
    monkeypatch.setattr(srv, "_gpu_vendors", lambda: {"nvidia"})
    monkeypatch.delenv("UNSLOTH_AUDIO_CPP_BACKEND")
    assert srv.select_backend(binary, True) == "cpu"


def test_nvidia_smi_on_path_is_not_a_visible_gpu(tmp_path, monkeypatch):
    """No install record: nvidia-smi stays on PATH when CUDA_VISIBLE_DEVICES hides every GPU."""
    monkeypatch.delenv("UNSLOTH_AUDIO_CPP_BACKEND", raising = False)
    monkeypatch.setattr(srv.sys, "platform", "linux")
    monkeypatch.setattr(
        srv.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None
    )
    monkeypatch.setattr(srv, "compiled_backends", lambda binary: frozenset({"cpu", "cuda"}))
    binary = str(tmp_path / srv.BINARY_NAME)
    monkeypatch.setattr(srv, "_gpu_vendors", lambda: set())
    assert srv.select_backend(binary, False) == "cpu"
    monkeypatch.setattr(srv, "_gpu_vendors", lambda: None)
    assert srv.select_backend(binary, False) == "cuda"


def test_gpu_vendor_probe_failure_reads_as_unknown(monkeypatch):
    from utils import llama_cpp_path_settings

    def boom():
        raise OSError("sysfs unreadable")

    monkeypatch.setattr(llama_cpp_path_settings, "host_gpu_vendors", boom)
    assert srv._gpu_vendors() is None


@pytest.mark.parametrize(
    "env, expected",
    [
        # With no HIP or ROCR mask set, CUDA_VISIBLE_DEVICES stands in for AMD too.
        ({"CUDA_VISIBLE_DEVICES": ""}, set()),
        ({"CUDA_VISIBLE_DEVICES": "-1", "HIP_VISIBLE_DEVICES": "0"}, {"amd"}),
        ({"HIP_VISIBLE_DEVICES": ""}, {"nvidia"}),
        ({"CUDA_VISIBLE_DEVICES": "0"}, {"nvidia", "amd"}),
        ({}, {"nvidia", "amd"}),
    ],
)
def test_visibility_masks_hide_vendors_found_by_driver_dlls(monkeypatch, env, expected):
    """Windows detection only globs driver DLLs, so the masks are applied on top of it."""
    from utils import llama_cpp_path_settings

    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising = False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(llama_cpp_path_settings, "host_gpu_vendors", lambda: {"nvidia", "amd"})
    assert srv._gpu_vendors() == expected


def test_a_masked_windows_gpu_build_launches_on_cpu(tmp_path, monkeypatch):
    from utils import llama_cpp_path_settings

    monkeypatch.delenv("UNSLOTH_AUDIO_CPP_BACKEND", raising = False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(llama_cpp_path_settings, "host_gpu_vendors", lambda: {"nvidia"})
    binary = _recorded_build(tmp_path, "cuda")
    assert srv.select_backend(binary, False) == "cpu"
    monkeypatch.setattr(llama_cpp_path_settings, "host_gpu_vendors", lambda: None)
    assert srv.select_backend(binary, False) == "cuda"


_EXITS_WITH = textwrap.dedent(
    """
    import sys
    print({output!r}, flush=True)
    sys.exit(1)
    """
)


def test_a_gpu_init_failure_says_so_instead_of_blaming_the_model(
    fake_binary, tmp_path, monkeypatch
):
    (tmp_path / "fake_server.py").write_text(
        _EXITS_WITH.format(
            output = "ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected\n"
            "audiocpp_server failed: CUDA backend requested but registry 'CUDA' has no device 0"
        ),
        encoding = "utf-8",
    )
    monkeypatch.setattr(srv, "select_backend", lambda binary, force_cpu: "cuda")
    with pytest.raises(srv.AudioCppUnavailableError) as raised:
        srv.AudioCppServer.start(CANARY, str(tmp_path / "m.gguf"))
    message = str(raised.value)
    assert "incomplete" not in message
    assert "CUDA backend" in message and "Unsloth Studio" in message
    assert "UNSLOTH_AUDIO_CPP_BACKEND=cpu" in message
    assert "has no device 0" in message


@pytest.mark.parametrize(
    "output",
    [
        "gguf: unknown tensor type 42",
        # A CUDA build on the CPU backend logs its CUDA init failure before the real one.
        "ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected\n"
        "audiocpp_server failed: unknown tensor type 42",
    ],
)
def test_other_early_exits_keep_pointing_at_the_model(fake_binary, tmp_path, output):
    (tmp_path / "fake_server.py").write_text(_EXITS_WITH.format(output = output), encoding = "utf-8")
    with pytest.raises(srv.AudioCppUnavailableError) as raised:
        srv.AudioCppServer.start(CANARY, str(tmp_path / "m.gguf"))
    message = str(raised.value)
    assert "the model file may be incomplete or unsupported by this build" in message
    assert "unknown tensor type 42" in message


_GPU_FAILS = textwrap.dedent(
    """
    import json, sys
    if json.load(open(sys.argv[sys.argv.index("--config") + 1], encoding="utf-8"))["backend"] != "cpu":
        print({output!r}, flush=True)
        sys.exit(1)
    """
)
_NO_GPU = "ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected"


def _backend_whose_gpu_fails(
    tmp_path,
    monkeypatch,
    output,
    preference = None,
):
    """The fake server exits with ``output`` unless launched on CPU."""
    from core.inference import audio_cpp_backend as acb

    (tmp_path / "fake_server.py").write_text(
        _GPU_FAILS.format(output = output) + FAKE_SERVER, encoding = "utf-8"
    )
    starts = []

    def select(binary, force_cpu):
        starts.append(force_cpu)
        return "cpu" if force_cpu else "cuda"

    monkeypatch.setattr(srv, "select_backend", select)
    monkeypatch.setattr(acb.audio_cpp_files, "materialize", lambda model: str(tmp_path / "m.gguf"))
    monkeypatch.delenv("UNSLOTH_AUDIO_DEVICE", raising = False)
    return acb.AudioCppBackend(preference), starts


def test_gpu_unavailable_start_retries_once_on_cpu(fake_binary, tmp_path, monkeypatch):
    backend, starts = _backend_whose_gpu_fails(tmp_path, monkeypatch, _NO_GPU)
    try:
        backend._start_server(KOKORO)
        assert starts == [False, True]
        assert (backend.device, backend._device_fallback, backend._server.backend) == (
            "cpu",
            "cpu",
            "cpu",
        )
        backend._start_server(KOKORO)
        assert starts == [False, True, True]
        backend._model = KOKORO
        fields = backend.runtime_fields()
        assert (fields["audio_cpu"], fields["audio_device_fallback"]) == (True, "cpu")
    finally:
        backend._stop_server_locked()


@pytest.mark.parametrize(
    "output, preference, code",
    [
        (_NO_GPU, "gpu", "gpu_unavailable"),
        ("failed to allocate CUDA0 backend weight buffer", None, "out_of_memory"),
    ],
)
def test_pinned_gpu_or_oom_does_not_retry(
    fake_binary, tmp_path, monkeypatch, output, preference, code
):
    from core.inference.audio_cpp_failures import failure_of

    backend, starts = _backend_whose_gpu_fails(tmp_path, monkeypatch, output, preference)
    with pytest.raises(srv.AudioCppUnavailableError) as raised:
        backend._start_server(KOKORO)
    assert starts == [False] and backend._device_fallback is None
    assert failure_of(raised.value).code == code


def test_cancelled_start_does_not_retry(monkeypatch):
    from core.inference import audio_cpp_backend as acb
    from core.inference.audio_cpp_failures import AudioCppFailure, attach_failure

    backend = acb.AudioCppBackend("auto")
    gpu = attach_failure(
        srv.AudioCppUnavailableError("x"), AudioCppFailure("gpu_unavailable", "m", 503)
    )
    assert backend._may_retry_on_cpu(gpu, threading.Event())
    cancelled = threading.Event()
    cancelled.set()
    assert not backend._may_retry_on_cpu(gpu, cancelled)
    monkeypatch.setattr("utils.process_lifetime.is_process_shutting_down", lambda: True)
    assert not backend._may_retry_on_cpu(gpu, None)


def test_stop_survives_a_server_slow_to_exit_after_sigkill(tmp_path):
    # A server switching task mid-request (Vevo2 Clone <-> Edit) failed the request when the old
    # process outlived the SIGKILL wait in GPU teardown.
    class Stuck:
        pid = 999999999
        waits: list = []

        def poll(self):
            return None

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout):
            self.waits.append(timeout)
            raise srv.subprocess.TimeoutExpired("audiocpp_server", timeout)

    server = object.__new__(srv.AudioCppServer)
    server.process = Stuck()
    server._config_dir = tmp_path / "cfg"
    server.stop()
    assert Stuck.waits == [10, 120]


def test_a_server_slow_to_exit_after_kill_does_not_fail_the_restart(tmp_path, monkeypatch):
    """Seed-VC exited ~54 s after SIGTERM following a run; a reload must not raise on it."""
    import subprocess as sp

    class SlowProcess:
        pid = 424242
        killed = False

        def poll(self):
            return None

        def terminate(self):
            pass

        def kill(self):
            self.killed = True

        def wait(self, timeout = None):
            raise sp.TimeoutExpired(["audiocpp_server"], timeout)

    forgotten = []
    monkeypatch.setattr(srv, "forget_pid", forgotten.append)
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    process = SlowProcess()
    server = srv.AudioCppServer(process, 1, KOKORO, "kokoro", "cuda", config_dir)
    server.stop()
    assert process.killed and not config_dir.exists() and forgotten == []
