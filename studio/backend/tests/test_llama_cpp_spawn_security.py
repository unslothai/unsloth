from __future__ import annotations

import json
from pathlib import Path
import threading

import pytest

from core.inference import llama_cpp as llama_cpp_module
from core.inference.llama_cpp import LlamaCppBackend
from core.inference.llama_server_args import LlamaServerArgsError


class _FakeProcess:
    pid = 4242


class _FakeThread:
    def start(self) -> None:
        return None


def _backend(monkeypatch, tmp_path: Path):
    backend = object.__new__(LlamaCppBackend)
    backend._port = 12345
    backend._process = None
    backend._stdout_lines = []
    backend._llama_log_fh = None
    backend._llama_log_path = None
    backend._stdout_thread = None
    monkeypatch.setattr(backend, "_kill_process", lambda: None)
    monkeypatch.setattr(backend, "_record_server_pid", lambda _pid: None)
    monkeypatch.setattr(backend, "_drain_stdout", lambda: None)
    monkeypatch.setattr(llama_cpp_module, "_swa_cache_path", lambda: tmp_path / "cache.json")
    monkeypatch.setattr(llama_cpp_module.threading, "Thread", lambda **_kwargs: _FakeThread())
    monkeypatch.setattr(llama_cpp_module, "_windows_hidden_subprocess_kwargs", lambda: {})
    monkeypatch.setattr(llama_cpp_module, "_child_popen_kwargs", lambda: {})
    return backend


def test_spawn_scrubs_env_and_emits_one_redacted_record(monkeypatch, tmp_path):
    backend = _backend(monkeypatch, tmp_path)
    spawns = []
    records = []
    monkeypatch.setattr(
        llama_cpp_module.subprocess,
        "Popen",
        lambda cmd, **kwargs: spawns.append((list(cmd), kwargs)) or _FakeProcess(),
    )

    def capture(message, *args):
        rendered = message % args if args else str(message)
        if rendered.startswith("llama_server_start "):
            records.append(rendered)

    monkeypatch.setattr(llama_cpp_module.logger, "info", capture)
    custom = [
        "--future-safe",
        "private-relative.txt",
        "--future-inline=secret.txt",
        "-xattached-secret",
        "--top-k",
        "20",
        "value\twith-tab",
    ]
    cmd = [
        str(tmp_path / "llama-server.exe"),
        "--model",
        str(tmp_path / "private.gguf"),
        "--api-key=secret",
        "--model-draft",
        "private-draft.gguf",
        *custom,
    ]
    env = {
        "PATH": "kept",
        "CUDA_VISIBLE_DEVICES": "0",
        "LLAMA_ARG_TOP_K": "99",
        "LLAMA_API_KEY": "secret",
    }
    backend._start_llama_process(cmd, env, custom)

    assert spawns[0][1]["env"] == {"PATH": "kept", "CUDA_VISIBLE_DEVICES": "0"}
    assert len(records) == 1
    assert not any(
        secret in records[0]
        for secret in ("private.gguf", "secret.txt", "private-draft.gguf", '"20"')
    )
    payload = json.loads(records[0].split(" ", 1)[1])
    assert payload["event"] == "llama_server_start"
    assert payload["argv"][0] == "<private-path>"
    assert "--api-key=<redacted>" in payload["argv"]
    assert "--future-inline=<redacted>" in payload["argv"]
    assert "<unknown-option>" in payload["argv"]


def test_invalid_final_argv_rejects_before_kill_or_spawn(monkeypatch, tmp_path):
    backend = _backend(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(backend, "_kill_process", lambda: calls.append("kill"))
    monkeypatch.setattr(
        llama_cpp_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: calls.append("spawn"),
    )
    forged = ["--future-safe", "forged\u2028line"]
    with pytest.raises(LlamaServerArgsError, match = "forbidden"):
        backend._start_llama_process(["llama-server", *forged], {"PATH": "kept"}, forged)
    assert calls == []


def test_runtime_reconciliation_and_scrubbed_spec_preflight(monkeypatch, tmp_path):
    backend = _backend(monkeypatch, tmp_path)
    backend._lock = threading.Lock()
    backend._process = _FakeProcess()
    backend._extra_args = ["--top-k", "20", "--grammar-file", "private.gbnf"]
    stopped = []
    monkeypatch.setattr(backend, "_kill_process", lambda: stopped.append(True))
    assert backend.reconcile_argument_policy() is False
    assert stopped == [True]

    monkeypatch.setenv("LLAMA_ARG_SPEC_TYPE", "draft-mtp")
    monkeypatch.setenv("LLAMA_ARG_SPEC_DRAFT_MODEL", "ambient.gguf")
    assert llama_cpp_module._extra_args_requests_mtp([]) is False
    assert llama_cpp_module._extra_args_mtp_draft_path([]) is None
    managed = {
        "LLAMA_ARG_SPEC_TYPE": "draft-mtp",
        "LLAMA_ARG_SPEC_DRAFT_MODEL": "managed.gguf",
    }
    assert llama_cpp_module._extra_args_requests_mtp([], env = managed) is True
    assert llama_cpp_module._extra_args_mtp_draft_path([], env = managed) == "managed.gguf"


def test_windows_length_checks_complete_command_before_execution(monkeypatch, tmp_path):
    backend = _backend(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(llama_cpp_module.sys, "platform", "win32")
    monkeypatch.setattr(llama_cpp_module, "_WINDOWS_CREATEPROCESS_MAX_UTF16_UNITS", 40)
    monkeypatch.setattr(backend, "_kill_process", lambda: calls.append("kill"))
    monkeypatch.setattr(
        llama_cpp_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: calls.append("spawn"),
    )
    cmd = [r"C:\Program Files\llama-server.exe", "--model", "x" * 80]
    with pytest.raises(LlamaServerArgsError, match = "CreateProcess"):
        backend._start_llama_process(cmd, {"PATH": "kept"}, [])
    assert calls == []
