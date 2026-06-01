# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama-server GGUF embedder backend. Every external boundary (binary, GGUF
download, subprocess, HTTP) is mocked so no model or network is needed. Verifies
backend selection, import isolation (no torch), the spawn command, readiness
fail-loud, HTTP encode/tokenize parsing, and self-healing respawn."""

import subprocess
import sys
import textwrap
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
    """Minimal subprocess.Popen stand-in: drainable empty stdout, controllable
    liveness."""

    def __init__(self, alive = True, returncode = 0):
        self._alive = alive
        self.returncode = returncode
        self.stdout = iter(())  # drain thread finishes immediately

    def poll(self):
        return None if self._alive else self.returncode

    def terminate(self):
        self._alive = False

    def kill(self):
        self._alive = False

    def wait(self, timeout = None):
        return self.returncode


# ── Backend selection (facade) ───────────────────────────────────


def test_default_backend_is_sentence_transformers():
    assert type(embeddings._get_backend()).__name__ == "_SentenceTransformersBackend"


def test_llama_backend_selected_by_config(monkeypatch):
    monkeypatch.setattr(config, "EMBED_BACKEND", "llama-server")
    assert isinstance(embeddings._get_backend(), LlamaServerBackend)


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.setattr(config, "EMBED_BACKEND", "bogus")
    with pytest.raises(ValueError, match = "Unknown RAG_EMBED_BACKEND"):
        embeddings._get_backend()


def test_backend_rebuilds_when_name_changes(monkeypatch):
    assert type(embeddings._get_backend()).__name__ == "_SentenceTransformersBackend"
    monkeypatch.setattr(config, "EMBED_BACKEND", "llama-server")
    assert isinstance(embeddings._get_backend(), LlamaServerBackend)


def test_llama_backend_imports_no_torch():
    """Selecting the llama backend must not import torch / sentence_transformers.
    Runs in a clean subprocess so the parent's already-imported modules don't
    mask a regression."""
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
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output = True, text = True, env = env
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


# ── Spawn command / env ──────────────────────────────────────────


def test_build_cmd_cpu_flags():
    b = LlamaServerBackend()
    cmd = b._build_cmd("/bin/llama-server", "/m/bge.gguf", 9999)
    assert "--embedding" in cmd
    assert cmd[cmd.index("--pooling") + 1] == "cls"
    assert cmd[cmd.index("-ngl") + 1] == "0"  # CPU default
    assert cmd[cmd.index("--port") + 1] == "9999"


def test_build_cmd_gpu_offloads(monkeypatch):
    monkeypatch.setattr(config, "EMBED_DEVICE", "gpu")
    b = LlamaServerBackend()
    cmd = b._build_cmd("/bin/llama-server", "/m/bge.gguf", 1)
    assert cmd[cmd.index("-ngl") + 1] == "99"


def test_build_env_cpu_hides_gpus():
    b = LlamaServerBackend()
    env = b._build_env("/bin/llama-server")
    assert env["CUDA_VISIBLE_DEVICES"] == ""  # never contend with the chat model


# ── Spawn / readiness ────────────────────────────────────────────


def _patch_spawn_deps(monkeypatch, proc, *, free_port = 54321):
    monkeypatch.setattr(
        LlamaServerBackend, "_resolve_binary", lambda self: "/bin/llama-server"
    )
    monkeypatch.setattr(
        LlamaServerBackend, "_resolve_model_path", lambda self: "/m/bge.gguf"
    )
    monkeypatch.setattr(
        LlamaServerBackend, "_find_free_port", staticmethod(lambda: free_port)
    )
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
    # Dead process -> real _wait_for_health returns False on the first poll.
    _patch_spawn_deps(monkeypatch, _FakeProc(alive = False, returncode = 1))
    with pytest.raises(RuntimeError, match = "failed to become healthy"):
        b._spawn()


# ── encode / dim / token_counter (HTTP mocked) ───────────────────


def _embed_response(vectors):
    # Intentionally out-of-order indices to exercise the sort.
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
    # index sort restored original order despite reversed response.
    assert out[0].tolist() == [3.0, 4.0]


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
    monkeypatch.setattr(
        b, "_post", lambda p, pl: {"data": [{"index": 0, "embedding": [1.0]}]}
    )
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


# ── Self-healing respawn ─────────────────────────────────────────


def test_ensure_ready_respawns_dead_process(monkeypatch):
    b = LlamaServerBackend()
    b._process = _FakeProc(alive = False, returncode = 0)  # reaper killed us
    spawned = {"n": 0}

    def fake_spawn():
        spawned["n"] += 1
        b._process = _FakeProc(alive = True)

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
    monkeypatch.setattr(
        b, "_restart", lambda: restarts.__setitem__("n", restarts["n"] + 1)
    )

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
