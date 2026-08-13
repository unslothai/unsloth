# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama-server GGUF embedder tests, every boundary mocked."""

import subprocess
import sys
import textwrap
import types
from pathlib import Path

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
    monkeypatch.setattr(LlamaCppBackend, "_get_gpu_free_memory", staticmethod(lambda **_kw: gpus))
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


def test_build_env_gpu_on_macos_uses_the_dyld_search_path(monkeypatch, tmp_path):
    # dyld ignores LD_LIBRARY_PATH, and Apple Silicon is routed through the
    # use_gpu branch, so the embedding server needs the same treatment the chat
    # server got in #8566.
    import os
    import sys as _sys

    monkeypatch.setattr(_sys, "platform", "darwin")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/opt/inherited")
    # Pin an inherited value rather than asserting the key is absent: the child
    # env is a copy of os.environ, so an ambient LD_LIBRARY_PATH would be there
    # whatever this branch does. What matters is that it is left alone.
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/sentinel")
    # A host-native directory, not a POSIX literal: _llama_lib_dir resolves the
    # path, and this test simulates darwin on whatever host runs it, so on a
    # Windows runner "/opt/llama/bin" comes back drive-anchored as "D:\opt\...".
    bin_dir = tmp_path / "llama" / "bin"
    bin_dir.mkdir(parents = True)
    (bin_dir / "llama-server").write_bytes(b"\xcf\xfa\xed\xfe")
    b = LlamaServerBackend()
    env = b._build_env(str(bin_dir / "llama-server"), use_gpu = True)
    entries = env["DYLD_LIBRARY_PATH"].split(os.pathsep)
    assert entries == [str(bin_dir), "/opt/inherited"]
    assert env["LD_LIBRARY_PATH"] == "/opt/sentinel"


def test_build_env_cpu_on_macos_still_gets_the_dyld_search_path(monkeypatch, tmp_path):
    # EMBED_DEVICE=cpu, and the CPU retry after a failed GPU start, load the
    # same sibling dylibs; the loader path cannot be gated on use_gpu.
    import os
    import sys as _sys

    monkeypatch.setattr(_sys, "platform", "darwin")
    monkeypatch.delenv("DYLD_LIBRARY_PATH", raising = False)
    # Host-native, for the same reason as the GPU case above.
    bin_dir = tmp_path / "llama" / "bin"
    bin_dir.mkdir(parents = True)
    (bin_dir / "llama-server").write_bytes(b"\xcf\xfa\xed\xfe")
    b = LlamaServerBackend()
    env = b._build_env(str(bin_dir / "llama-server"), use_gpu = False)
    assert env["DYLD_LIBRARY_PATH"].split(os.pathsep)[0] == str(bin_dir)
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_build_env_resolves_a_wrapper_entrypoint(monkeypatch, tmp_path):
    # The managed install puts an entrypoint in front of the real server; the
    # dylibs sit next to the target, not next to the wrapper.
    import os
    import sys as _sys

    real_dir = tmp_path / "llama.cpp" / "build" / "bin"
    real_dir.mkdir(parents = True)
    (real_dir / "llama-server").write_text("")
    wrapper = tmp_path / "llama.cpp" / "llama-server"
    wrapper.write_text('#!/bin/sh\nexec "$(dirname "$0")/build/bin/llama-server" "$@"\n')
    monkeypatch.setattr(_sys, "platform", "darwin")
    monkeypatch.delenv("DYLD_LIBRARY_PATH", raising = False)
    b = LlamaServerBackend()
    env = b._build_env(str(wrapper), use_gpu = True)
    assert env["DYLD_LIBRARY_PATH"].split(os.pathsep)[0] == str(real_dir)


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
    monkeypatch.setattr(
        LlamaCppBackend, "_get_gpu_free_memory", staticmethod(lambda **_kw: [(0, 40000)])
    )
    assert LlamaServerBackend._gpu_available() is True
    monkeypatch.setattr(
        LlamaCppBackend, "_get_gpu_free_memory", staticmethod(lambda **_kw: [(0, 100)])
    )
    assert LlamaServerBackend._gpu_available() is False
    monkeypatch.setattr(LlamaCppBackend, "_get_gpu_free_memory", staticmethod(lambda **_kw: []))
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


class TestTheEmbeddingLoaderChangesAreMacOsOnly:
    """The RAG server's Linux and Windows launches must be as they were.

    The probe environment and the resolved library directory both started out
    unconditional, which moved the first LD_LIBRARY_PATH entry for every Linux
    GPU install whose llama-server is a symlink, and gave the capability probe
    an environment it never had.
    """

    def test_the_help_probe_still_inherits_the_environment_off_mac(self, monkeypatch):
        seen = {}

        def fake_run(cmd, **kwargs):
            seen.update(kwargs)
            return types.SimpleNamespace(stdout = "--embedding", stderr = "")

        monkeypatch.setattr(mod.sys, "platform", "linux")
        monkeypatch.setattr(mod.subprocess, "run", fake_run)
        mod.LlamaServerBackend._help_text.cache_clear()
        mod.LlamaServerBackend._help_text("/opt/mine/llama-server")
        assert seen.get("env") is None

    def test_the_help_probe_gets_the_loader_environment_on_mac(self, monkeypatch, tmp_path):
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"\xcf\xfa\xed\xfe")
        seen = {}

        def fake_run(cmd, **kwargs):
            seen.update(kwargs)
            return types.SimpleNamespace(stdout = "--embedding", stderr = "")

        monkeypatch.setattr(mod.sys, "platform", "darwin")
        monkeypatch.setattr(mod.subprocess, "run", fake_run)
        mod.LlamaServerBackend._help_text.cache_clear()
        mod.LlamaServerBackend._help_text(str(binary))
        assert seen.get("env") is not None

    def test_linux_gpu_libs_still_hang_off_the_given_path(self, monkeypatch, tmp_path):
        real = tmp_path / "build" / "bin" / "llama-server"
        real.parent.mkdir(parents = True)
        real.write_bytes(b"\xcf\xfa\xed\xfe")
        wrapper = tmp_path / "llama-server"
        wrapper.write_text('#!/bin/sh\nexec "$(dirname "$0")/build/bin/llama-server" "$@"\n')
        wrapper.chmod(0o755)
        seen = {}
        monkeypatch.setattr(mod.sys, "platform", "linux")
        monkeypatch.setattr(
            mod.LlamaServerBackend,
            "_add_linux_cuda_libs",
            staticmethod(lambda env, d: seen.setdefault("dir", d)),
        )
        mod.LlamaServerBackend()._build_env(str(wrapper), use_gpu = True)
        assert seen["dir"] == str(wrapper.parent)
