# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import http.server
import io
import json
import os
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

import core.inference.stt_ggml_sidecar as ggml_module
from core.inference.stt_ggml_sidecar import (
    DEFAULT_GGML_STT_MODEL,
    GGML_STT_MODELS,
    GGML_STT_REPOS,
    GgmlSttSidecar,
    SttEngineUnavailableError,
    find_whisper_server_binary,
    resolve_ggml_model_id,
)
from core.inference.stt_sidecar import (
    SttLanguageError,
    SttLoadCancelledError,
    SttModelIdError,
    SttModelNotDownloadedError,
    SttUnavailableError,
)


@pytest.fixture(autouse = True)
def stub_audio_decoder(monkeypatch):
    """Unit tests exercise orchestration, not PyAV container parsing."""
    monkeypatch.setattr(
        ggml_module,
        "_decode_audio_bounded",
        lambda audio: np.zeros(16000, dtype = np.float32),
    )


# ---------------------------------------------------------------------------
# Model id resolution
# ---------------------------------------------------------------------------


def test_curated_ids_resolve():
    for model_id in GGML_STT_MODELS:
        assert resolve_ggml_model_id(model_id) == model_id


def test_default_model_resolves_from_none_and_blank():
    assert resolve_ggml_model_id(None) == DEFAULT_GGML_STT_MODEL
    assert resolve_ggml_model_id("  ") == DEFAULT_GGML_STT_MODEL


def test_custom_repo_ids_are_rejected():
    with pytest.raises(SttModelIdError):
        resolve_ggml_model_id("owner/model")
    with pytest.raises(SttModelIdError):
        resolve_ggml_model_id("large-v2")


def test_curated_ids_mirror_transformers_sidecar():
    from core.inference.stt_sidecar import STT_MODELS
    assert list(GGML_STT_MODELS.keys()) == list(STT_MODELS.keys())


def test_curated_filenames_match_repo_naming():
    # unslothai/whisper-<id>-GGUF hosts whisper-<id>.bin; keep the download
    # filename in lockstep with the repo so it resolves instead of 404ing.
    for model_id, repo in GGML_STT_REPOS.items():
        expected = repo.split("/", 1)[1].removesuffix("-GGUF") + ".bin"
        assert GGML_STT_MODELS[model_id] == expected


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------


def test_env_binary_override_wins(monkeypatch, tmp_path):
    binary = tmp_path / "whisper-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)  # find_whisper_server_binary requires an executable
    monkeypatch.setenv("WHISPER_SERVER_PATH", str(binary))
    assert find_whisper_server_binary() == str(binary)


def test_env_dir_override_scans_layouts(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_SERVER_PATH", raising = False)
    build_bin = tmp_path / "build" / "bin"
    build_bin.mkdir(parents = True)
    binary = build_bin / "whisper-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)  # find_whisper_server_binary requires an executable
    monkeypatch.setenv("UNSLOTH_WHISPER_CPP_PATH", str(tmp_path))
    assert find_whisper_server_binary() == str(binary)


def test_missing_binary_reports_unavailable(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_SERVER_PATH", raising = False)
    monkeypatch.setenv("UNSLOTH_WHISPER_CPP_PATH", str(tmp_path / "nope"))
    monkeypatch.setattr(ggml_module, "_managed_whisper_cpp_dir", lambda: tmp_path / "gone")
    monkeypatch.setattr(ggml_module.shutil, "which", lambda name: None)
    assert find_whisper_server_binary() is None
    assert not ggml_module.is_available()
    with pytest.raises(SttEngineUnavailableError):
        ggml_module.ensure_engine_available()


def test_non_executable_binary_is_not_runnable(monkeypatch, tmp_path):
    if sys.platform == "win32":
        pytest.skip("X_OK is an existence check on Windows")
    binary = tmp_path / "whisper-server"
    binary.write_text("#!/bin/sh\n")  # written but not chmod +x
    monkeypatch.setenv("WHISPER_SERVER_PATH", str(binary))
    monkeypatch.setattr(ggml_module.shutil, "which", lambda name: None)
    assert find_whisper_server_binary() is None


# ---------------------------------------------------------------------------
# Slim-install launch guard
# ---------------------------------------------------------------------------


def _slim_install(
    tmp_path,
    *,
    install_kind = "slim",
    with_ggml = True,
    linked_libraries = None,
    backend = "cpu",
    linked_runtime_directories = None,
    runtime_wiring_version = None,
) -> str:
    """A managed-looking install tree: marker at the root, server in build/bin."""
    install_dir = tmp_path / "whisper.cpp"
    bin_dir = install_dir / "build" / "bin"
    bin_dir.mkdir(parents = True)
    binary = bin_dir / "whisper-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    marker: dict = {
        "schema_version": 1,
        "component": "whisper.cpp",
        "release_tag": "v1.9.1-unsloth.1",
        "backend": backend,
        "paired_llama_tag": "b10069-mix-fb3d4ca",
    }
    if install_kind is not None:
        marker["install_kind"] = install_kind
    if linked_libraries is not None:
        marker["linked_libraries"] = linked_libraries
    if linked_runtime_directories is not None:
        marker["linked_runtime_directories"] = linked_runtime_directories
        for name in linked_runtime_directories:
            catalog = bin_dir / name
            catalog.mkdir()
            (catalog / "kernel.dat").write_bytes(b"kernel")
    if runtime_wiring_version is not None:
        marker["runtime_wiring_version"] = runtime_wiring_version
    (install_dir / "UNSLOTH_WHISPER_PREBUILT_INFO.json").write_text(json.dumps(marker))
    if with_ggml:
        names = (
            ("ggml.dll", "ggml-base.dll")
            if sys.platform == "win32"
            else ("libggml.so.0", "libggml-base.so.0")
        )
        for name in names:
            (bin_dir / name).write_bytes(b"ggml")
    return str(binary)


def test_slim_guard_flags_missing_ggml_links(monkeypatch, tmp_path):
    # A slim marker whose linked ggml runtime is gone must read as engine
    # unavailable (reinstall), never crash into a server launch.
    binary = _slim_install(tmp_path, with_ggml = False)
    assert ggml_module.slim_runtime_intact(binary) is False
    monkeypatch.setattr(ggml_module, "find_whisper_server_binary", lambda: binary)
    assert not ggml_module.is_available()
    with pytest.raises(SttEngineUnavailableError, match = "ggml"):
        ggml_module.ensure_engine_available()


def test_slim_guard_passes_with_links_in_place(monkeypatch, tmp_path):
    names = ["libggml.so.0", "libggml-base.so.0"]
    binary = _slim_install(tmp_path, with_ggml = True, linked_libraries = names)
    assert ggml_module.slim_runtime_intact(binary) is True
    monkeypatch.setattr(ggml_module, "find_whisper_server_binary", lambda: binary)
    assert ggml_module.ensure_engine_available() == binary


def test_slim_guard_verifies_the_marker_linked_libraries(monkeypatch, tmp_path):
    # New markers record the exact wired filenames; one missing name flips the
    # install to unavailable even when the legacy core ggml names are present.
    names = ["libggml.dylib", "libggml-base.dylib", "libggml-metal.dylib"]
    binary = _slim_install(tmp_path, with_ggml = True, linked_libraries = names)
    bin_dir = Path(binary).parent
    for name in names[:-1]:
        (bin_dir / name).write_bytes(b"ggml")
    assert ggml_module.slim_runtime_intact(binary) is False  # metal dylib absent
    (bin_dir / names[-1]).write_bytes(b"ggml")
    assert ggml_module.slim_runtime_intact(binary) is True
    monkeypatch.setattr(ggml_module, "find_whisper_server_binary", lambda: binary)
    assert ggml_module.ensure_engine_available() == binary


def test_slim_guard_malformed_authoritative_marker_fails_closed(tmp_path):
    for bad in ("not-a-list", [], [1, 2]):
        root = tmp_path / f"case_{type(bad).__name__}_{len(str(bad))}"
        root.mkdir()
        binary = _slim_install(root, with_ggml = True, linked_libraries = bad)
        assert ggml_module.slim_runtime_intact(binary) is False


def test_slim_guard_prefers_authoritative_root_marker(tmp_path):
    names = ["libggml.so.0", "libggml-base.so.0"]
    binary = _slim_install(tmp_path, with_ggml = True, linked_libraries = names)
    packaging_marker = Path(binary).parent / "UNSLOTH_WHISPER_PREBUILT_INFO.json"
    packaging_marker.write_text(json.dumps({"backend": "slim", "release_tag": "packaging"}))
    assert ggml_module._whisper_install_marker(binary)["install_kind"] == "slim"
    assert ggml_module.slim_runtime_intact(binary) is True


def test_slim_guard_rejects_invalid_root_even_with_inner_marker(tmp_path):
    binary = _slim_install(tmp_path, with_ggml = True, linked_libraries = ["libggml.so.0"])
    root_marker = Path(binary).parents[2] / "UNSLOTH_WHISPER_PREBUILT_INFO.json"
    root_marker.write_text("not json")
    (Path(binary).parent / root_marker.name).write_text(json.dumps({"backend": "slim"}))
    assert ggml_module.slim_runtime_intact(binary) is False


def test_slim_guard_rejects_missing_rocm_catalog(tmp_path):
    names = ["libggml.so.0", "libggml-base.so.0", "libggml-hip.so"]
    binary = _slim_install(
        tmp_path,
        linked_libraries = names,
        backend = "rocm",
        linked_runtime_directories = ["hipblaslt", "rocblas"],
        runtime_wiring_version = 2,
    )
    bin_dir = Path(binary).parent
    (bin_dir / "libggml-hip.so").write_bytes(b"ggml")
    assert ggml_module.slim_runtime_intact(binary) is True
    (bin_dir / "rocblas" / "kernel.dat").unlink()
    assert ggml_module.slim_runtime_intact(binary) is False


def test_slim_guard_accepts_windows_rocm_dll_overlay(monkeypatch, tmp_path):
    monkeypatch.setattr(ggml_module.sys, "platform", "win32")
    names = ["ggml.dll", "ggml-base.dll", "ggml-hip.dll", "amdhip64.dll"]
    binary = _slim_install(
        tmp_path,
        linked_libraries = names,
        backend = "rocm",
        linked_runtime_directories = [],
        runtime_wiring_version = 2,
    )
    for name in names:
        (Path(binary).parent / name).write_bytes(b"dll")
    assert ggml_module.slim_runtime_intact(binary) is True


def test_slim_guard_ignores_fat_and_markerless_installs(tmp_path):
    # Fat installs carry their own ggml; no marker means source/custom build.
    fat = _slim_install(tmp_path / "fat", install_kind = None, with_ggml = False)
    assert ggml_module.slim_runtime_intact(fat) is True
    bare = tmp_path / "bare" / "whisper-server"
    bare.parent.mkdir(parents = True)
    bare.write_text("#!/bin/sh\n")
    assert ggml_module.slim_runtime_intact(str(bare)) is True


# ---------------------------------------------------------------------------
# whisper-server child-process environment
# ---------------------------------------------------------------------------


def _loader_path_var() -> str:
    return {"win32": "PATH", "darwin": "DYLD_LIBRARY_PATH"}.get(sys.platform, "LD_LIBRARY_PATH")


def test_child_env_scrubs_secrets_and_adds_lib_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "secret-token")  # exact name
    monkeypatch.setenv("MY_API_KEY", "nope")  # marker substring
    monkeypatch.setenv("HTTPS_PROXY", "http://u:p@px:8080")  # url-name
    monkeypatch.setenv("SOME_REMOTE", "https://u:pw@host/repo")  # url-userinfo value
    monkeypatch.setenv("STT_KEEPME", "keep")  # benign
    binary = tmp_path / "whisper-server"
    binary.write_text("#!/bin/sh\n")
    env = ggml_module._whisper_server_child_env(str(binary))
    for scrubbed in ("HF_TOKEN", "MY_API_KEY", "HTTPS_PROXY", "SOME_REMOTE"):
        assert scrubbed not in env
    assert env.get("STT_KEEPME") == "keep"
    assert str(tmp_path.resolve()) in env[_loader_path_var()].split(os.pathsep)


def test_child_env_isolates_home_and_cred_locations(monkeypatch, tmp_path):
    # The downloaded server must not see the real home (token caches live
    # there) nor explicit cred-store pointers like HF_HOME / NETRC.
    monkeypatch.setenv("HOME", "/real/home")
    monkeypatch.setenv("HF_HOME", "/real/hf")
    monkeypatch.setenv("NETRC", "/real/.netrc")
    monkeypatch.setattr(ggml_module, "_managed_whisper_cpp_dir", lambda: tmp_path / "managed")
    binary = tmp_path / "whisper-server"
    binary.write_text("#!/bin/sh\n")
    env = ggml_module._whisper_server_child_env(str(binary))
    assert env["HOME"] == str(tmp_path / "managed" / ".child_home")
    assert "HF_HOME" not in env
    assert "NETRC" not in env
    assert (tmp_path / "managed" / ".child_home").is_dir()


def test_child_env_wsl_rocm_prepends_system_hip(monkeypatch, tmp_path):
    if sys.platform != "linux":
        pytest.skip("WSL ROCm library precedence is Linux-only")
    rocm = tmp_path / "rocm-lib"
    rocm.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binary = bindir / "whisper-server"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setattr(ggml_module, "_wsl_system_rocm_lib_dirs", lambda: [str(rocm)])
    env = ggml_module._whisper_server_child_env(str(binary))
    parts = env["LD_LIBRARY_PATH"].split(os.pathsep)
    assert parts[0] == str(rocm.resolve())  # system HIP wins
    assert str(bindir.resolve()) in parts  # bundle libs still present
    assert env.get("HSA_ENABLE_DXG_DETECTION") == "1"


def test_child_env_adds_cuda_runtime_dirs_for_cuda_bundle(monkeypatch, tmp_path):
    # Versioned CUDA backend modules are valid too. They still need the
    # CUDA-from-PyTorch wheel dirs for libcudart/libcublas at launch.
    if sys.platform == "darwin":
        pytest.skip("no CUDA on macOS")
    import utils.prebuilt.runtime_libs as rl

    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "whisper-server").write_text("#!/bin/sh\n")
    module_name = "ggml-cuda.dll" if sys.platform == "win32" else "libggml-cuda.so.0"
    (bindir / module_name).write_text("")
    cuda_dir = tmp_path / "nvidia" / "cuda_runtime" / "lib"
    cuda_dir.mkdir(parents = True)
    monkeypatch.setattr(rl, "python_runtime_dirs", lambda: [str(cuda_dir)])
    env = ggml_module._whisper_server_child_env(str(bindir / "whisper-server"))
    parts = env[_loader_path_var()].split(os.pathsep)
    assert str(bindir.resolve()) in parts
    assert str(cuda_dir.resolve()) in parts
    assert parts.index(str(bindir.resolve())) < parts.index(str(cuda_dir.resolve()))


def test_child_env_omits_cuda_runtime_dirs_for_cpu_bundle(monkeypatch, tmp_path):
    # No libggml-cuda.so beside the binary -> a static CPU/Metal bundle -> the CUDA
    # wheel discovery must not run and must not touch the loader path.
    if sys.platform == "darwin":
        pytest.skip("no CUDA on macOS")
    import utils.prebuilt.runtime_libs as rl

    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "whisper-server").write_text("#!/bin/sh\n")
    cuda_dir = tmp_path / "nvidia" / "cuda_runtime" / "lib"
    cuda_dir.mkdir(parents = True)
    called = {"n": 0}

    def _fake_dirs():
        called["n"] += 1
        return [str(cuda_dir)]

    monkeypatch.setattr(rl, "python_runtime_dirs", _fake_dirs)
    env = ggml_module._whisper_server_child_env(str(bindir / "whisper-server"))
    parts = env[_loader_path_var()].split(os.pathsep)
    assert str(cuda_dir.resolve()) not in parts
    assert called["n"] == 0


def test_engine_unavailable_is_stt_unavailable():
    # Routes map SttUnavailableError to HTTP 501; the engine error must share it.
    assert issubclass(SttEngineUnavailableError, SttUnavailableError)


# ---------------------------------------------------------------------------
# WAV packaging
# ---------------------------------------------------------------------------


def test_pcm_to_wav_bytes_shape_and_rate():
    pcm = np.zeros(3200, dtype = np.float32)
    data = ggml_module._pcm_to_wav_bytes(pcm)
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16000
        assert w.getnframes() == 3200


def test_pcm_to_wav_bytes_clips_out_of_range():
    pcm = np.array([2.0, -2.0], dtype = np.float32)
    data = ggml_module._pcm_to_wav_bytes(pcm)
    with wave.open(io.BytesIO(data)) as w:
        frames = np.frombuffer(w.readframes(2), dtype = "<i2")
    assert frames[0] == 32767
    assert frames[1] == -32767


# ---------------------------------------------------------------------------
# Sidecar orchestration
# ---------------------------------------------------------------------------


def _available(monkeypatch):
    monkeypatch.setattr(ggml_module, "find_whisper_server_binary", lambda: "/bin/echo")


def test_transcribe_requires_engine(monkeypatch):
    monkeypatch.setattr(ggml_module, "find_whisper_server_binary", lambda: None)
    sidecar = GgmlSttSidecar()
    with pytest.raises(SttEngineUnavailableError):
        sidecar.transcribe(b"RIFF")


def test_transcribe_rejects_unknown_language(monkeypatch):
    _available(monkeypatch)
    sidecar = GgmlSttSidecar()
    with pytest.raises(SttLanguageError):
        sidecar.transcribe(b"RIFF", model = "small", language = "xx-QQ")


def test_load_requires_downloaded_model(monkeypatch):
    _available(monkeypatch)
    monkeypatch.setattr(ggml_module, "_cached_model_path", lambda model_id: None)
    sidecar = GgmlSttSidecar()
    with pytest.raises(SttModelNotDownloadedError):
        sidecar.load("small")


def test_unloaded_sidecar_reports_nothing_resident():
    sidecar = GgmlSttSidecar()
    assert sidecar.loaded_model is None
    assert sidecar.device is None
    assert sidecar.is_loading() is False
    sidecar.unload()  # no-op, must not raise


def test_update_maintenance_unloads_and_blocks_new_loads(monkeypatch):
    class FakeProcess:
        pid = 4242

        def __init__(self):
            self.running = True

        def poll(self):
            return None if self.running else 0

        def terminate(self):
            self.running = False

        def wait(self, timeout = None):
            return 0

    monkeypatch.setattr(ggml_module, "forget_pid", lambda _pid: None)
    sidecar = GgmlSttSidecar()
    sidecar._process = FakeProcess()
    sidecar._model_id = "small"

    with sidecar.update_maintenance() as model_was_active:
        assert model_was_active is True
        assert sidecar.loaded_model is None
        with pytest.raises(SttEngineUnavailableError, match = "being updated"):
            sidecar.load("small")

    assert sidecar._update_in_progress is False


def test_server_pid_is_tracked_for_parent_lifetime(monkeypatch):
    # The spawned server must be adopted for the terminate_all backstop and
    # forgotten once this sidecar has reaped it.
    _available(monkeypatch)
    monkeypatch.setattr(ggml_module, "_cached_model_path", lambda model_id: "/tmp/ggml.bin")

    class FakeProcess:
        pid = 4242

        def __init__(self, *args, **kwargs):
            self.terminated = False

        def poll(self):
            return 1 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout = None):
            return 0

    events = []
    monkeypatch.setattr(ggml_module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(ggml_module, "adopt_pid", lambda pid: events.append(("adopt", pid)))
    monkeypatch.setattr(ggml_module, "forget_pid", lambda pid: events.append(("forget", pid)))
    monkeypatch.setattr(
        GgmlSttSidecar,
        "_wait_for_server",
        staticmethod(lambda process, port, cancel_event = None: None),
    )

    sidecar = GgmlSttSidecar()
    sidecar.load("small")
    assert events == [("adopt", 4242)]
    sidecar.unload()
    assert events == [("adopt", 4242), ("forget", 4242)]


def test_training_forces_whisper_server_off_gpu(monkeypatch):
    # Mirror the Transformers sidecar: keep whisper.cpp on CPU during training
    # so a mid-training dictation cannot reclaim the VRAM training just freed.
    _available(monkeypatch)
    monkeypatch.setattr(ggml_module, "_cached_model_path", lambda model_id: "/tmp/ggml.bin")
    commands: list[list[str]] = []

    class FakeProcess:
        pid = 4242

        def __init__(self, command, *args, **kwargs):
            commands.append(command)

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout = None):
            return 0

    monkeypatch.setattr(ggml_module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(ggml_module, "adopt_pid", lambda pid: None)
    monkeypatch.setattr(ggml_module, "forget_pid", lambda pid: None)
    monkeypatch.setattr(
        GgmlSttSidecar,
        "_wait_for_server",
        staticmethod(lambda process, port, cancel_event = None: None),
    )

    monkeypatch.setattr(ggml_module, "_training_active", lambda: False)
    idle = GgmlSttSidecar()
    idle.load("small")
    assert "--no-gpu" not in commands[0]
    assert idle.is_loading() is False
    idle.unload()

    monkeypatch.setattr(ggml_module, "_training_active", lambda: True)
    training = GgmlSttSidecar()
    training.load("small")
    assert "--no-gpu" in commands[1]
    training.unload()


def test_cpu_root_marker_forces_no_gpu_despite_inner_packaging_marker(monkeypatch, tmp_path):
    names = ["libggml.so.0", "libggml-base.so.0"]
    binary = _slim_install(tmp_path, with_ggml = True, linked_libraries = names)
    (Path(binary).parent / "UNSLOTH_WHISPER_PREBUILT_INFO.json").write_text(
        json.dumps({"backend": "slim"})
    )
    monkeypatch.setattr(ggml_module, "find_whisper_server_binary", lambda: binary)
    monkeypatch.setattr(ggml_module, "_cached_model_path", lambda model_id: "/tmp/ggml.bin")
    commands: list[list[str]] = []

    class FakeProcess:
        pid = 4244

        def __init__(self, command, *args, **kwargs):
            commands.append(command)

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout = None):
            return 0

    monkeypatch.setattr(ggml_module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(ggml_module, "adopt_pid", lambda pid: None)
    monkeypatch.setattr(ggml_module, "forget_pid", lambda pid: None)
    monkeypatch.setattr(ggml_module, "_training_active", lambda: False)
    monkeypatch.setattr(
        GgmlSttSidecar,
        "_wait_for_server",
        staticmethod(lambda process, port, cancel_event = None: None),
    )

    sidecar = GgmlSttSidecar()
    sidecar.load("small")
    assert "--no-gpu" in commands[0]
    sidecar.unload()


def test_startup_is_cancellable_before_training(monkeypatch):
    # A whisper-server still binding its (Metal/CUDA) backend must be preemptible
    # so training coordination can stop it before admitting the run, instead of
    # racing an allocating subprocess.
    _available(monkeypatch)
    monkeypatch.setattr(ggml_module, "_cached_model_path", lambda model_id: "/tmp/ggml.bin")

    class FakeProcess:
        pid = 4243

        def __init__(self, *args, **kwargs):
            self.terminated = False
            self.killed = False

        def poll(self):
            return -15 if (self.terminated or self.killed) else None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout = None):
            return 0

    monkeypatch.setattr(ggml_module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(ggml_module, "adopt_pid", lambda pid: None)
    monkeypatch.setattr(ggml_module, "forget_pid", lambda pid: None)

    # The server never reports ready, so _wait_for_server loops until cancelled.
    def never_ready(req, timeout = None):
        raise OSError("connection refused")

    monkeypatch.setattr(ggml_module.urllib.request, "urlopen", never_ready)

    sidecar = GgmlSttSidecar()
    result: dict = {}

    def _load():
        try:
            sidecar.load("small")
            result["ok"] = True
        except Exception as exc:  # noqa: BLE001 - recorded for the assertion below
            result["error"] = exc

    thread = threading.Thread(target = _load)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not sidecar.is_loading():
            time.sleep(0.01)
        assert sidecar.is_loading() is True
        assert sidecar.cancel_pending_load() is True
        # Blocks until the cancelled startup has been reaped and the lock freed.
        sidecar.wait_for_load_to_settle()
    finally:
        thread.join(timeout = 5)

    assert thread.is_alive() is False
    assert isinstance(result.get("error"), SttLoadCancelledError)
    assert sidecar.is_loading() is False
    assert sidecar.loaded_model is None


class _FakeWhisperHandler(http.server.BaseHTTPRequestHandler):
    """Stands in for whisper-server's /inference endpoint."""

    response_text = "Hello world.\n Second line."

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = json.dumps({"text": self.response_text}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def fake_whisper_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeWhisperHandler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()


def test_transcribe_joins_segments_one_line(monkeypatch, fake_whisper_server):
    _available(monkeypatch)
    monkeypatch.setattr(ggml_module, "_cached_model_path", lambda model_id: "/tmp/ggml.bin")
    sidecar = GgmlSttSidecar()

    def fake_load(model = None):
        sidecar._port = fake_whisper_server
        sidecar._model_id = ggml_module.resolve_ggml_model_id(model)

    monkeypatch.setattr(sidecar, "load", fake_load)
    result = sidecar.transcribe(b"RIFF", model = "small", language = "en", fast = True)
    assert result["text"] == "Hello world. Second line."
    assert result["language"] == "en"
    assert result["model"] == "small"
    assert result["duration"] == pytest.approx(1.0)


def test_transcribe_maps_bad_payload_to_decode_error(monkeypatch, fake_whisper_server):
    _available(monkeypatch)
    monkeypatch.setattr(ggml_module, "_cached_model_path", lambda model_id: "/tmp/ggml.bin")
    monkeypatch.setattr(_FakeWhisperHandler, "response_text", None)
    sidecar = GgmlSttSidecar()

    def fake_load(model = None):
        sidecar._port = fake_whisper_server
        sidecar._model_id = ggml_module.resolve_ggml_model_id(model)

    monkeypatch.setattr(sidecar, "load", fake_load)
    from core.inference.stt_sidecar import SttAudioDecodeError

    with pytest.raises(SttAudioDecodeError):
        sidecar.transcribe(b"RIFF", model = "small")


def test_beam_size_matches_fast_flag(monkeypatch, fake_whisper_server):
    _available(monkeypatch)
    monkeypatch.setattr(ggml_module, "_cached_model_path", lambda model_id: "/tmp/ggml.bin")
    seen: list[bytes] = []

    orig_post = _FakeWhisperHandler.do_POST

    def capture_post(handler):
        length = int(handler.headers.get("Content-Length", "0"))
        body = handler.rfile.read(length)
        seen.append(body)
        payload = json.dumps({"text": "ok"}).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    monkeypatch.setattr(_FakeWhisperHandler, "do_POST", capture_post)
    try:
        sidecar = GgmlSttSidecar()

        def fake_load(model = None):
            sidecar._port = fake_whisper_server
            sidecar._model_id = ggml_module.resolve_ggml_model_id(model)

        monkeypatch.setattr(sidecar, "load", fake_load)
        sidecar.transcribe(b"RIFF", model = "small", fast = True)
        sidecar.transcribe(b"RIFF", model = "small", fast = False)
    finally:
        _FakeWhisperHandler.do_POST = orig_post
    assert b'name="beam_size"\r\n\r\n1' in seen[0]
    assert b'name="beam_size"\r\n\r\n5' in seen[1]
    # Dictation defaults to deterministic decoding.
    assert b'name="temperature"\r\n\r\n0.0' in seen[0]


def test_download_rejects_custom_ids():
    with pytest.raises(SttModelIdError):
        ggml_module.start_model_download("owner/model")


def test_download_status_idle_shape():
    status = ggml_module.download_status()
    assert set(status) >= {"downloading", "model", "error"}
