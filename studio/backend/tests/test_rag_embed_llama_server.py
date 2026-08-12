# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama-server GGUF embedder tests, every boundary mocked."""

import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.rag import config, embeddings
from core.rag import embed_llama_server as mod
from core.rag.embed_llama_server import LlamaServerBackend


@pytest.fixture(autouse = True)
def _reset_backend_singleton():
    embeddings._reset_backend()
    yield
    embeddings._reset_backend()


class _FakeProc:
    """subprocess.Popen stand-in with controllable liveness."""

    def __init__(
        self,
        alive = True,
        returncode = 0,
    ):
        self._alive = alive
        self.returncode = returncode
        self.pid = 424242  # every real Popen has one; the lifetime record reads it
        self.stdout = iter(())  # drain thread exits immediately

    def poll(self):
        return None if self._alive else self.returncode

    def terminate(self):
        self._alive = False

    def kill(self):
        self._alive = False

    def wait(self, timeout = None):
        return self.returncode


def _mock_auto(monkeypatch, *, gpus, binary):
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(config, "EMBED_BACKEND", "auto")
    monkeypatch.setattr(LlamaCppBackend, "_get_gpu_free_memory", staticmethod(lambda: gpus))
    monkeypatch.setattr(LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: binary))


def _stub_st_load(monkeypatch):
    # Make the ST probe succeed without importing sentence-transformers (absent in
    # the torch-free backend CI); these tests assert selection, not a real load.
    monkeypatch.setattr(embeddings, "_get", lambda *a, **k: object())


def test_auto_uses_st_with_cuda(monkeypatch):
    _stub_st_load(monkeypatch)
    _mock_auto(monkeypatch, gpus = [(0, 40000)], binary = "/bin/llama-server")
    assert type(embeddings._get_backend()).__name__ == "_SentenceTransformersBackend"


def test_auto_uses_llama_without_cuda(monkeypatch):
    _mock_auto(monkeypatch, gpus = [], binary = "/bin/llama-server")
    assert isinstance(embeddings._get_backend(), LlamaServerBackend)


def test_auto_falls_back_to_st_without_binary(monkeypatch):
    _stub_st_load(monkeypatch)
    _mock_auto(monkeypatch, gpus = [], binary = None)
    assert type(embeddings._get_backend()).__name__ == "_SentenceTransformersBackend"


def test_llama_backend_selected_by_config(monkeypatch):
    monkeypatch.setattr(config, "EMBED_BACKEND", "llama-server")
    assert isinstance(embeddings._get_backend(), LlamaServerBackend)


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.setattr(config, "EMBED_BACKEND", "bogus")
    with pytest.raises(ValueError, match = "Unknown RAG_EMBED_BACKEND"):
        embeddings._get_backend()


def test_explicit_backend_overrides_auto(monkeypatch):
    _stub_st_load(monkeypatch)
    monkeypatch.setattr(config, "EMBED_BACKEND", "sentence-transformers")
    assert type(embeddings._get_backend()).__name__ == "_SentenceTransformersBackend"
    monkeypatch.setattr(config, "EMBED_BACKEND", "llama-server")
    assert isinstance(embeddings._get_backend(), LlamaServerBackend)


def _mock_rocm_probe(monkeypatch, *, is_rocm, probe_ok):
    """Pin ROCm-ness and the isolated allocation probe's verdict (#8474)."""
    import threading

    from utils import device_allocation_probe as probe_mod
    from utils.hardware import hardware as hardware_mod
    from utils.hardware.hardware import DeviceType

    monkeypatch.setattr(embeddings, "get_device", lambda: DeviceType.CUDA)
    monkeypatch.setattr(hardware_mod, "IS_ROCM", is_rocm)
    detected = threading.Event()
    detected.set()
    monkeypatch.setattr(hardware_mod, "DETECTION_COMPLETE", detected)
    monkeypatch.setattr(
        probe_mod,
        "probe_torch_device_allocation",
        lambda device = "cuda:0": probe_mod.DeviceAllocationProbeResult(
            ok = probe_ok,
            device = device,
            returncode = 0 if probe_ok else -11,
            reason = None if probe_ok else "killed by SIGSEGV",
            duration_seconds = 0.1,
        ),
    )


def test_auto_does_not_query_gpu_memory_after_a_failed_rocm_probe(monkeypatch):
    # _get_gpu_free_memory falls back to torch mem_get_info in THIS process on AMD, so on
    # a condemned host auto-resolution used to steer into the very crash it should avoid.
    # The probe has to be consulted first.
    _stub_st_load(monkeypatch)
    _mock_rocm_probe(monkeypatch, is_rocm = True, probe_ok = False)
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(config, "EMBED_BACKEND", "auto")

    def _must_not_run():
        raise AssertionError("_get_gpu_free_memory reached on a condemned ROCm host")

    monkeypatch.setattr(LlamaCppBackend, "_get_gpu_free_memory", staticmethod(_must_not_run))
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "/bin/llama-server")
    )

    # Sentence-transformers, not llama-server: the model and therefore the vector space
    # stays the same, so no knowledge base needs reindexing. The device moves, not the
    # backend.
    assert embeddings._resolve_auto() == "sentence-transformers"


def _forbid_gpu_query(monkeypatch):
    """Make the in-process AMD GPU-memory query a test failure if it is reached."""
    from core.inference.llama_cpp import LlamaCppBackend

    def _must_not_run():
        raise AssertionError("_get_gpu_free_memory reached; its AMD path runs torch here")

    monkeypatch.setattr(LlamaCppBackend, "_get_gpu_free_memory", staticmethod(_must_not_run))
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "/bin/llama-server")
    )


def _forbid_probe(monkeypatch):
    """Make the isolated allocation probe a test failure if it is reached."""
    from utils import device_allocation_probe as probe_mod

    def _must_not_run(device = "cuda:0"):
        raise AssertionError("allocation probe reached on a request path")

    monkeypatch.setattr(probe_mod, "probe_torch_device_allocation", _must_not_run)


def test_settings_gate_does_not_block_on_detection_or_probe(monkeypatch):
    # active_backend_is_llama() runs inside PUT /embedding-model. Before detection settles
    # it must not force detection, must not run the probe (a cold torch import is allowed
    # 120s), and must not reach the AMD GPU query. Answering False keeps the ST pickle gate
    # engaged, which is the safe direction for a security check.
    import threading

    from utils.hardware import hardware as hardware_mod

    monkeypatch.setattr(config, "EMBED_BACKEND", "auto")
    monkeypatch.setattr(hardware_mod, "DETECTION_COMPLETE", threading.Event())  # unset
    monkeypatch.setattr(
        embeddings, "get_device", lambda: (_ for _ in ()).throw(AssertionError("forced detection"))
    )
    _forbid_gpu_query(monkeypatch)
    _forbid_probe(monkeypatch)

    assert embeddings.active_backend_is_llama() is False


def _unsettled_detection(monkeypatch, *, rocm_possible):
    """Detection still running, with ROCm-possibility pinned torch-free."""
    import threading

    from utils.hardware import hardware as hardware_mod

    monkeypatch.setattr(config, "EMBED_BACKEND", "auto")
    monkeypatch.setattr(hardware_mod, "DETECTION_COMPLETE", threading.Event())  # unset
    monkeypatch.setattr(embeddings, "_rocm_is_possible", lambda: rocm_possible)


def test_a_cpu_host_keeps_its_gguf_classification_while_detection_runs(monkeypatch):
    # settings.py reads this to decide whether a local .gguf needs HF verification. A host
    # that can never be ROCm must keep its real answer during the detection window, or a
    # valid local .gguf transiently 409s on a machine that has no AMD GPU at all.
    _unsettled_detection(monkeypatch, rocm_possible = False)
    _mock_auto(monkeypatch, gpus = [], binary = "/bin/llama-server")
    assert embeddings.active_backend_is_llama() is True


def test_a_possibly_rocm_host_still_holds_back_while_detection_runs(monkeypatch):
    _unsettled_detection(monkeypatch, rocm_possible = True)
    _forbid_gpu_query(monkeypatch)
    _forbid_probe(monkeypatch)
    assert embeddings.active_backend_is_llama() is False


def test_resolver_keeps_the_gguf_answer_for_an_impossible_rocm_host(monkeypatch):
    _unsettled_detection(monkeypatch, rocm_possible = False)
    monkeypatch.setattr(embeddings, "get_device", lambda: None)  # detection stays unsettled
    _mock_auto(monkeypatch, gpus = [], binary = "/bin/llama-server")
    assert embeddings._resolve_auto() == "llama-server"


def test_settings_gate_never_waits_on_hardware_detection(monkeypatch):
    # ensure_hardware_detected() holds _DETECT_LOCK across a cold torch import, so a
    # request reaching get_device() while background detection runs parks for the whole
    # pass. ROCm is already ruled out here, and the rest of auto needs no detection.
    _unsettled_detection(monkeypatch, rocm_possible = False)
    monkeypatch.setattr(
        embeddings, "get_device", lambda: (_ for _ in ()).throw(AssertionError("forced detection"))
    )
    _mock_auto(monkeypatch, gpus = [], binary = "/bin/llama-server")
    assert embeddings.active_backend_is_llama() is True


def _no_torch_loaded(monkeypatch, *, installed = None):
    """torch not imported. ``installed`` is what the torch-free on-disk read reports."""
    monkeypatch.delitem(sys.modules, "torch", raising = False)
    embeddings._installed_torch_is_rocm.cache_clear()
    monkeypatch.setattr(embeddings, "_installed_torch_is_rocm", lambda: installed)


def test_rocm_is_possible_is_false_on_macos(monkeypatch):
    _no_torch_loaded(monkeypatch)
    monkeypatch.setattr(embeddings.sys, "platform", "darwin")
    assert embeddings._rocm_is_possible() is False


def test_rocm_is_possible_follows_the_kfd_node_on_linux(monkeypatch):
    _no_torch_loaded(monkeypatch)
    monkeypatch.setattr(embeddings.sys, "platform", "linux")
    seen: list = []

    def _isdir(path):
        seen.append(path)
        return False

    monkeypatch.setattr(embeddings.os.path, "isdir", _isdir)
    assert embeddings._rocm_is_possible() is False
    # ROCm's own kernel driver publishes this; hardware.py reads the same tree.
    assert seen == ["/sys/class/kfd/kfd/topology/nodes"]


def test_an_installed_hip_sdk_alone_is_not_a_rocm_host(monkeypatch):
    # main.py makes the same point where it refuses to let HIP_PATH/ROCM_PATH pick a
    # backend: a CUDA or CPU box can carry the SDK. What decides is the installed wheel,
    # which is readable before torch is imported.
    _no_torch_loaded(monkeypatch, installed = False)
    monkeypatch.setattr(embeddings.sys, "platform", "win32")
    assert embeddings._rocm_is_possible() is False


def test_a_windows_rocm_wheel_is_seen_before_torch_is_imported(monkeypatch):
    # The startup window, and the standing case when the torch warm is disabled: detection
    # unsettled and torch not yet imported. Windows has no KFD node, so the wheel on disk
    # is the only thing that can keep this host off the in-process AMD query.
    _no_torch_loaded(monkeypatch, installed = True)
    monkeypatch.setattr(embeddings.sys, "platform", "win32")
    assert embeddings._rocm_is_possible() is True


def test_windows_stays_cautious_when_the_wheel_cannot_be_read(monkeypatch):
    _no_torch_loaded(monkeypatch, installed = None)
    monkeypatch.setattr(embeddings.sys, "platform", "win32")
    assert embeddings._rocm_is_possible() is True


def test_installed_torch_is_read_without_importing_torch(monkeypatch, tmp_path):
    # torch/version.py is generated literals, and find_spec locates the package without
    # executing it, so this reads what detection reads long before torch is imported.
    import importlib.util

    embeddings._installed_torch_is_rocm.cache_clear()
    package = tmp_path / "torch"
    package.mkdir()
    (package / "__init__.py").write_text("raise AssertionError('torch was imported')")

    def _fake_find_spec(name):
        assert name == "torch"
        return SimpleNamespace(origin = str(package / "__init__.py"))

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)

    (package / "version.py").write_text(
        "__version__ = '2.9.1+rocm6.3'\nhip: Optional[str] = '6.3.42134'\n"
    )
    assert embeddings._installed_torch_is_rocm() is True

    embeddings._installed_torch_is_rocm.cache_clear()
    (package / "version.py").write_text("__version__ = '2.9.1+cu128'\nhip: Optional[str] = None\n")
    assert embeddings._installed_torch_is_rocm() is False

    # AMD SDK wheel: hip unset, rocm only in the version string.
    embeddings._installed_torch_is_rocm.cache_clear()
    (package / "version.py").write_text(
        "__version__ = '2.11.0+rocm7.1'\nhip: Optional[str] = None\n"
    )
    assert embeddings._installed_torch_is_rocm() is True

    # Absent is a definite answer, not an unknown: no torch, no ROCm torch path. A
    # --no-torch install resolves to llama-server, and calling this unknown would make
    # Windows cautious forever and reject the local GGUF that install is meant to take.
    embeddings._installed_torch_is_rocm.cache_clear()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert embeddings._installed_torch_is_rocm() is False

    # Present but unreadable stays unknown, which is what the caution is reserved for.
    embeddings._installed_torch_is_rocm.cache_clear()
    (package / "version.py").unlink()
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    assert embeddings._installed_torch_is_rocm() is None


def test_a_no_torch_windows_install_keeps_its_gguf_answer(monkeypatch):
    # --no-torch on Windows: detection never settles when the torch warm is disabled, and
    # auto really does resolve to llama-server there, so PUT /embedding-model must accept a
    # local GGUF rather than send it through the sentence-transformers verification path.
    _no_torch_loaded(monkeypatch, installed = False)
    monkeypatch.setattr(embeddings.sys, "platform", "win32")
    _mock_auto(monkeypatch, gpus = [], binary = "/bin/llama-server")
    assert embeddings._rocm_is_possible() is False
    assert embeddings.active_backend_is_llama() is True


def test_rocm_is_possible_reads_torch_hip_when_torch_is_already_loaded(monkeypatch):
    # A build attribute, so reading it initialises no driver, and it is the exact ROCm-ness
    # the dangerous query depends on. Only consulted when torch is already imported.
    monkeypatch.setattr(embeddings.sys, "platform", "win32")
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(version = SimpleNamespace(hip = "6.3.42134"))
    )
    assert embeddings._rocm_is_possible() is True

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(version = SimpleNamespace(hip = None), __version__ = "2.9.1+cu128"),
    )
    assert embeddings._rocm_is_possible() is False


def test_an_amd_sdk_wheel_without_hip_metadata_is_still_rocm(monkeypatch):
    # AMD SDK wheels leave torch.version.hip unset and only say rocm in the version string.
    # hardware.py's own ROCm test carries the same fallback (utils/hardware/hardware.py).
    monkeypatch.setattr(embeddings.sys, "platform", "linux")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(version = SimpleNamespace(hip = None), __version__ = "2.9.1+rocm6.3"),
    )
    assert embeddings._rocm_is_possible() is True


def test_a_half_imported_torch_is_treated_as_possibly_rocm(monkeypatch):
    # torch can be in sys.modules while still executing, so neither attribute is there yet.
    # That is absence of evidence, and the caution has to keep the safe side.
    monkeypatch.setattr(embeddings.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    assert embeddings._rocm_is_possible() is True
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(version = SimpleNamespace()))
    assert embeddings._rocm_is_possible() is True


def test_settings_gate_does_not_probe_on_a_settled_rocm_host(monkeypatch):
    _mock_rocm_probe(monkeypatch, is_rocm = True, probe_ok = False)
    monkeypatch.setattr(config, "EMBED_BACKEND", "auto")
    _forbid_gpu_query(monkeypatch)
    _forbid_probe(monkeypatch)

    assert embeddings.active_backend_is_llama() is False


def test_settings_gate_still_reports_llama_on_a_settled_non_rocm_host(monkeypatch):
    # The NVIDIA/CPU path keeps its old answer: nvidia-smi is torch-free and cheap.
    _mock_rocm_probe(monkeypatch, is_rocm = False, probe_ok = True)
    _mock_auto(monkeypatch, gpus = [], binary = "/bin/llama-server")
    assert embeddings.active_backend_is_llama() is True


def test_builder_settles_detection_rather_than_guessing(monkeypatch):
    # The backend builder caches its answer, so unlike the request path it must not commit
    # a provisional one: it settles detection first and then decides.
    import threading

    from utils.hardware import hardware as hardware_mod
    from utils.hardware.hardware import DeviceType

    monkeypatch.setattr(config, "EMBED_BACKEND", "auto")
    event = threading.Event()
    monkeypatch.setattr(hardware_mod, "DETECTION_COMPLETE", event)
    monkeypatch.setattr(hardware_mod, "IS_ROCM", False)

    settled = {"count": 0}

    def _detect():
        settled["count"] += 1
        event.set()
        return DeviceType.CPU

    monkeypatch.setattr(embeddings, "get_device", _detect)
    _mock_auto(monkeypatch, gpus = [], binary = "/bin/llama-server")

    assert embeddings._resolve_auto() == "llama-server"
    assert settled["count"] == 1


def test_auto_is_unchanged_after_a_passing_rocm_probe(monkeypatch):
    _stub_st_load(monkeypatch)
    _mock_rocm_probe(monkeypatch, is_rocm = True, probe_ok = True)
    _mock_auto(monkeypatch, gpus = [(0, 40000)], binary = "/bin/llama-server")
    assert embeddings._resolve_auto() == "sentence-transformers"


def test_auto_on_a_non_rocm_host_still_asks_gpu_memory(monkeypatch):
    # The NVIDIA path must be byte-identical to before: nvidia-smi decides, no probe.
    _stub_st_load(monkeypatch)
    _mock_rocm_probe(monkeypatch, is_rocm = False, probe_ok = False)
    _mock_auto(monkeypatch, gpus = [], binary = "/bin/llama-server")
    assert embeddings._resolve_auto() == "llama-server"


def test_llama_backend_imports_no_torch():
    # Clean subprocess so the parent's imports don't mask a regression.
    backend_dir = Path(__file__).resolve().parents[1]
    code = textwrap.dedent(
        """
        import sys
        from core.rag import embeddings
        b = embeddings._get_backend()
        assert type(b).__name__ == "LlamaServerBackend", type(b).__name__
        assert "torch" not in sys.modules, "torch was imported"
        assert "sentence_transformers" not in sys.modules, "ST was imported"
        print("OK")
        """
    )
    env = {
        **__import__("os").environ,
        "RAG_EMBED_BACKEND": "llama-server",
        "PYTHONPATH": str(backend_dir),
    }
    proc = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, env = env)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_build_cmd_cpu_flags():
    b = LlamaServerBackend()
    cmd = b._build_cmd("/bin/llama-server", "/m/bge.gguf", 9999, use_gpu = False)
    assert "--embedding" in cmd
    assert cmd[cmd.index("--pooling") + 1] == "cls"
    assert cmd[cmd.index("--fit") + 1] == "off"  # deterministic, no auto-resize
    assert cmd[cmd.index("-ngl") + 1] == "0"  # CPU keeps all off the GPU
    assert cmd[cmd.index("--port") + 1] == "9999"


def test_build_cmd_gpu_offloads():
    b = LlamaServerBackend()
    cmd = b._build_cmd("/bin/llama-server", "/m/bge.gguf", 1, use_gpu = True)
    assert cmd[cmd.index("-ngl") + 1] == "-1"  # offload all, matching the chat server


def test_build_env_cpu_hides_gpus():
    b = LlamaServerBackend()
    env = b._build_env("/bin/llama-server", use_gpu = False)
    assert env["CUDA_VISIBLE_DEVICES"] == ""  # never contend with the chat model
    assert env["LLAMA_SET_ROWS"] == "1"


def test_build_env_gpu_inherits_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    b = LlamaServerBackend()
    env = b._build_env("/bin/llama-server", use_gpu = True)
    assert env.get("CUDA_VISIBLE_DEVICES") == "0,1"  # inherit Unsloth's selection


def test_use_gpu_explicit_modes(monkeypatch):
    b = LlamaServerBackend()
    monkeypatch.setattr(config, "EMBED_DEVICE", "gpu")
    assert b._use_gpu() is True
    monkeypatch.setattr(config, "EMBED_DEVICE", "cpu")
    assert b._use_gpu() is False


def test_use_gpu_auto_follows_probe(monkeypatch):
    b = LlamaServerBackend()
    monkeypatch.setattr(config, "EMBED_DEVICE", "auto")
    monkeypatch.setattr(LlamaServerBackend, "_gpu_available", staticmethod(lambda: True))
    assert b._use_gpu() is True
    monkeypatch.setattr(LlamaServerBackend, "_gpu_available", staticmethod(lambda: False))
    assert b._use_gpu() is False


def test_use_gpu_sticky_cpu_fallback(monkeypatch):
    b = LlamaServerBackend()
    monkeypatch.setattr(config, "EMBED_DEVICE", "auto")
    monkeypatch.setattr(LlamaServerBackend, "_gpu_available", staticmethod(lambda: True))
    b._force_cpu = True  # a prior GPU start failed
    assert b._use_gpu() is False


def test_gpu_available_reuses_studio_probe(monkeypatch):
    import utils.hardware as uh
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(uh, "is_apple_silicon", lambda: False)
    # Ample free VRAM -> GPU; nearly full -> CPU; none -> CPU.
    monkeypatch.setattr(LlamaCppBackend, "_get_gpu_free_memory", staticmethod(lambda: [(0, 40000)]))
    assert LlamaServerBackend._gpu_available() is True
    monkeypatch.setattr(LlamaCppBackend, "_get_gpu_free_memory", staticmethod(lambda: [(0, 100)]))
    assert LlamaServerBackend._gpu_available() is False
    monkeypatch.setattr(LlamaCppBackend, "_get_gpu_free_memory", staticmethod(lambda: []))
    assert LlamaServerBackend._gpu_available() is False


def test_gpu_available_apple_metal(monkeypatch):
    import utils.hardware as uh
    monkeypatch.setattr(uh, "is_apple_silicon", lambda: True)
    assert LlamaServerBackend._gpu_available() is True


def _patch_spawn_deps(
    monkeypatch,
    proc,
    *,
    free_port = 54321,
):
    # Force CPU so spawn never depends on a host GPU.
    monkeypatch.setattr(config, "EMBED_DEVICE", "cpu")
    monkeypatch.setattr(LlamaServerBackend, "_resolve_binary", lambda self: "/bin/llama-server")
    monkeypatch.setattr(LlamaServerBackend, "_resolve_model_path", lambda self: "/m/bge.gguf")
    monkeypatch.setattr(LlamaServerBackend, "_find_free_port", staticmethod(lambda: free_port))
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: proc)


def test_spawn_uses_explicit_port(monkeypatch):
    monkeypatch.setattr(config, "EMBED_PORT", 8123)
    b = LlamaServerBackend()
    _patch_spawn_deps(monkeypatch, _FakeProc(alive = True))
    monkeypatch.setattr(b, "_wait_for_health", lambda *a, **k: True)
    b._spawn()
    assert b._port == 8123


def test_spawn_uses_free_port_when_auto(monkeypatch):
    monkeypatch.setattr(config, "EMBED_PORT", 0)
    b = LlamaServerBackend()
    _patch_spawn_deps(monkeypatch, _FakeProc(alive = True), free_port = 47000)
    monkeypatch.setattr(b, "_wait_for_health", lambda *a, **k: True)
    b._spawn()
    assert b._port == 47000


def test_spawn_fails_loud_on_early_exit(monkeypatch):
    monkeypatch.setattr(config, "EMBED_PORT", 8124)
    b = LlamaServerBackend()
    _patch_spawn_deps(monkeypatch, _FakeProc(alive = False, returncode = 1))
    with pytest.raises(RuntimeError, match = "failed to become healthy"):
        b._spawn()


def test_spawn_auto_falls_back_to_cpu_on_gpu_failure(monkeypatch):
    monkeypatch.setattr(config, "EMBED_DEVICE", "auto")
    monkeypatch.setattr(LlamaServerBackend, "_gpu_available", staticmethod(lambda: True))
    b = LlamaServerBackend()
    calls = []

    def fake_spawn_once(use_gpu):
        calls.append(use_gpu)
        if use_gpu:
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(b, "_spawn_once", fake_spawn_once)
    b._spawn()
    assert calls == [True, False]  # tried GPU, then fell back to CPU
    assert b._force_cpu is True  # sticky, so respawns stay on CPU


def test_spawn_explicit_gpu_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(config, "EMBED_DEVICE", "gpu")
    b = LlamaServerBackend()

    def fake_spawn_once(use_gpu):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(b, "_spawn_once", fake_spawn_once)
    with pytest.raises(RuntimeError, match = "out of memory"):
        b._spawn()
    assert b._force_cpu is False  # explicit gpu never silently downgrades


def _embed_response(vectors):
    # Reversed so the index sort is exercised.
    items = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    return {"data": list(reversed(items))}


def test_encode_orders_and_returns_float32(monkeypatch):
    b = LlamaServerBackend()
    monkeypatch.setattr(b, "_ensure_ready", lambda: None)
    captured = {}

    def fake_post(path, payload):
        captured["path"] = path
        captured["input"] = payload["input"]
        return _embed_response([[3.0, 4.0], [0.0, 5.0]])

    monkeypatch.setattr(b, "_post", fake_post)
    out = b.encode(["a", "b"], normalize = False)
    assert captured["path"] == "/v1/embeddings"
    assert out.dtype == np.float32
    assert out.shape == (2, 2)
    assert out[0].tolist() == [3.0, 4.0]  # index sort restored order


def test_encode_normalizes(monkeypatch):
    b = LlamaServerBackend()
    monkeypatch.setattr(b, "_ensure_ready", lambda: None)
    monkeypatch.setattr(b, "_post", lambda p, pl: _embed_response([[3.0, 4.0]]))
    out = b.encode(["a"], normalize = True)
    np.testing.assert_allclose(np.linalg.norm(out, axis = 1), [1.0], rtol = 1e-6)


def test_encode_empty_returns_zero_rows(monkeypatch):
    b = LlamaServerBackend()
    b._dim = 384
    monkeypatch.setattr(b, "_ensure_ready", lambda: None)
    out = b.encode([])
    assert out.shape == (0, 384)
    assert out.dtype == np.float32


def test_encode_rejects_count_mismatch(monkeypatch):
    b = LlamaServerBackend()
    monkeypatch.setattr(b, "_ensure_ready", lambda: None)
    monkeypatch.setattr(b, "_post", lambda p, pl: {"data": [{"index": 0, "embedding": [1.0]}]})
    with pytest.raises(RuntimeError, match = "vectors for"):
        b.encode(["a", "b"], normalize = False)


def test_encode_batches(monkeypatch):
    monkeypatch.setattr(config, "EMBED_BATCH", 2)
    b = LlamaServerBackend()
    monkeypatch.setattr(b, "_ensure_ready", lambda: None)
    calls = []

    def fake_post(path, payload):
        chunk = payload["input"]
        calls.append(len(chunk))
        return _embed_response([[1.0, 0.0]] * len(chunk))

    monkeypatch.setattr(b, "_post", fake_post)
    out = b.encode(["a", "b", "c"], normalize = False)
    assert out.shape == (3, 2)
    assert calls == [2, 1]  # batched at EMBED_BATCH=2


def test_dim_probes_once_and_caches(monkeypatch):
    b = LlamaServerBackend()
    monkeypatch.setattr(b, "_ensure_ready", lambda: None)
    n_calls = {"n": 0}

    def fake_post(path, payload):
        n_calls["n"] += 1
        return _embed_response([[0.1] * 384])

    monkeypatch.setattr(b, "_post", fake_post)
    assert b.dim() == 384
    assert b.dim() == 384
    assert n_calls["n"] == 1  # cached after the first probe


def test_token_counter_hits_tokenize(monkeypatch):
    b = LlamaServerBackend()
    monkeypatch.setattr(b, "_ensure_ready", lambda: None)
    seen = {}

    def fake_post(path, payload):
        seen["path"] = path
        seen["content"] = payload["content"]
        return {"tokens": [1, 2, 3, 4]}

    monkeypatch.setattr(b, "_post", fake_post)
    count = b.token_counter()
    assert count("hello world") == 4
    assert seen["path"] == "/tokenize"
    assert seen["content"] == "hello world"


def test_ensure_ready_respawns_dead_process(monkeypatch):
    b = LlamaServerBackend()
    b._process = _FakeProc(alive = False, returncode = 0)
    spawned = {"n": 0}

    def fake_spawn():
        spawned["n"] += 1
        b._process = _FakeProc(alive = True)
        # _current() now also checks the served repo, so mark it current.
        b._model_repo = config.effective_gguf_repo()

    monkeypatch.setattr(b, "_spawn", fake_spawn)
    b._ensure_ready()
    assert spawned["n"] == 1
    assert b._process_alive()
    # Already alive -> no second spawn.
    b._ensure_ready()
    assert spawned["n"] == 1


def test_post_restarts_once_on_connect_error(monkeypatch):
    import httpx

    b = LlamaServerBackend()
    b._port = 9000
    monkeypatch.setattr(b, "_ensure_ready", lambda: None)
    restarts = {"n": 0}
    monkeypatch.setattr(b, "_restart", lambda: restarts.__setitem__("n", restarts["n"] + 1))

    attempts = {"n": 0}

    class _Client:
        def post(self, url, json):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise httpx.ConnectError("boom")

            class _R:
                def raise_for_status(self_inner):
                    return None

                def json(self_inner):
                    return {"tokens": [1]}

            return _R()

    b._client = _Client()
    out = b._post("/tokenize", {"content": "x"})
    assert out == {"tokens": [1]}
    assert restarts["n"] == 1  # one self-heal restart, then success


def test_post_restarts_once_on_read_timeout(monkeypatch):
    # A wedged request (ReadTimeout) also triggers one restart-and-retry.
    import httpx

    b = LlamaServerBackend()
    b._port = 9000
    monkeypatch.setattr(b, "_ensure_ready", lambda: None)
    restarts = {"n": 0}
    monkeypatch.setattr(b, "_restart", lambda: restarts.__setitem__("n", restarts["n"] + 1))

    attempts = {"n": 0}

    class _Client:
        def post(self, url, json):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise httpx.ReadTimeout("timed out")

            class _R:
                def raise_for_status(self_inner):
                    return None

                def json(self_inner):
                    return {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}

            return _R()

    b._client = _Client()
    out = b._post("/v1/embeddings", {"input": ["x"]})
    assert out["data"][0]["embedding"] == [1.0, 0.0]
    assert restarts["n"] == 1  # timeout self-heals like a transport error
