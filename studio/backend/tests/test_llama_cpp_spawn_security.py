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


def _bare_backend(monkeypatch, tmp_path: Path):
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


def test_spawn_scrubs_all_llama_env_and_logs_one_redacted_structured_record(
    monkeypatch, tmp_path
):
    backend = _bare_backend(monkeypatch, tmp_path)
    popen_calls = []
    log_records = []

    def fake_popen(cmd, **kwargs):
        popen_calls.append((list(cmd), kwargs))
        return _FakeProcess()

    def fake_info(message, *args):
        rendered = message % args if args else str(message)
        if rendered.startswith("llama_server_start "):
            log_records.append(rendered)

    monkeypatch.setattr(llama_cpp_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(llama_cpp_module.logger, "info", fake_info)

    private_model = str(tmp_path / "private-model.gguf")
    private_unknown = str(tmp_path / "unknown-value.txt")
    cmd = [
        str(tmp_path / "llama-server.exe"),
        "--model",
        private_model,
        "--api-key=first-secret",
        "--api-key",
        "second-secret",
        "--rpc",
        "10.0.0.5:5000",
        "--model-draft",
        "relative-private-draft.gguf",
        "--future-safe",
        private_unknown,
        "--future-inline=relative-secret.txt",
        "-xattached-relative-secret.txt",
        "--top-k",
        "20",
        "value\twith-tab",
    ]
    env = {
        "PATH": "system-path",
        "CUDA_VISIBLE_DEVICES": "0",
        "LLAMA_ARG_TOP_K": "99",
        "LLAMA_ARG_FUTURE_CAPABILITY": "on",
        "LLAMA_API_KEY": "ambient-secret",
        "HF_TOKEN": "hf-secret",
    }

    backend._start_llama_process(
        cmd,
        env,
        [
            "--future-safe",
            private_unknown,
            "--future-inline=relative-secret.txt",
            "-xattached-relative-secret.txt",
            "--top-k",
            "20",
            "value\twith-tab",
        ],
    )

    assert len(popen_calls) == 1
    spawned_env = popen_calls[0][1]["env"]
    assert spawned_env == {"PATH": "system-path", "CUDA_VISIBLE_DEVICES": "0"}
    assert len(log_records) == 1
    raw_record = log_records[0]
    assert "first-secret" not in raw_record
    assert "second-secret" not in raw_record
    assert "10.0.0.5:5000" not in raw_record
    assert "relative-private-draft.gguf" not in raw_record
    assert private_model not in raw_record
    assert private_unknown not in raw_record
    assert "relative-secret.txt" not in raw_record
    assert "attached-relative-secret.txt" not in raw_record
    assert '"20"' not in raw_record
    assert "value\\twith-tab" not in raw_record
    payload = json.loads(raw_record.split(" ", 1)[1])
    assert payload["event"] == "llama_server_start"
    assert payload["argv"][0] == "<private-path>"
    assert payload["argv"][2] == "<redacted>"
    assert "--api-key=<redacted>" in payload["argv"]
    rpc_index = payload["argv"].index("--rpc")
    assert payload["argv"][rpc_index + 1] == "<redacted>"
    future_index = payload["argv"].index("--future-safe")
    assert payload["argv"][future_index + 1] == "<redacted>"
    assert "--future-inline=<redacted>" in payload["argv"]
    assert "<unknown-option>" in payload["argv"]
    top_k_index = payload["argv"].index("--top-k")
    assert payload["argv"][top_k_index + 1] == "<redacted>"


def test_spawn_rejects_forbidden_separator_before_kill_or_popen(monkeypatch, tmp_path):
    backend = _bare_backend(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(backend, "_kill_process", lambda: calls.append("kill"))
    monkeypatch.setattr(
        llama_cpp_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: calls.append("spawn"),
    )

    with pytest.raises(LlamaServerArgsError) as excinfo:
        backend._start_llama_process(
            ["llama-server", "--future-safe", "forged\u2028line"],
            {"PATH": "system-path"},
            ["--future-safe", "forged\u2028line"],
        )
    assert excinfo.value.code == "forbidden_character"
    assert calls == []


def test_spawn_rejects_invalid_managed_argv_token_before_kill_or_popen(monkeypatch, tmp_path):
    backend = _bare_backend(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(backend, "_kill_process", lambda: calls.append("kill"))
    monkeypatch.setattr(
        llama_cpp_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: calls.append("spawn"),
    )

    with pytest.raises(LlamaServerArgsError) as excinfo:
        backend._start_llama_process(
            ["llama-server", "--model", " managed-path.gguf"],
            {"PATH": "system-path"},
            [],
        )
    assert excinfo.value.code == "malformed"
    assert calls == []


def test_startup_source_has_no_raw_or_duplicate_custom_argument_log():
    source = Path(llama_cpp_module.__file__).read_text(encoding = "utf-8")
    assert "Appending user extra args" not in source
    assert "Starting llama-server:" not in source
    assert "llama-server stdout/stderr ->" not in source
    assert source.count("self._log_llama_start(") == 2


def test_runtime_reconciliation_stops_process_with_newly_blocked_saved_args(
    monkeypatch, tmp_path
):
    backend = _bare_backend(monkeypatch, tmp_path)
    backend._lock = threading.Lock()
    backend._process = _FakeProcess()
    backend._extra_args = ["--top-k", "20", "--grammar-file", "private.gbnf"]
    stopped = []
    monkeypatch.setattr(backend, "_kill_process", lambda: stopped.append(True))

    assert backend.reconcile_argument_policy() is False
    assert stopped == [True]


def test_speculative_preflight_ignores_ambient_llama_arg_variables(monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_SPEC_TYPE", "draft-mtp")
    monkeypatch.setenv("LLAMA_ARG_SPEC_DRAFT_MODEL", "ambient-private.gguf")
    monkeypatch.setenv("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K", "q4_0")
    monkeypatch.setenv("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V", "q5_0")
    monkeypatch.setenv("LLAMA_ARG_N_GPU_LAYERS_DRAFT", "0")

    assert llama_cpp_module._extra_args_requests_mtp([]) is False
    assert llama_cpp_module._extra_args_mtp_draft_path([]) is None
    assert llama_cpp_module._extra_args_draft_cache_types([]) == (None, None)
    assert llama_cpp_module._extra_args_draft_offloaded_to_cpu([]) is False

    studio_env = {
        "LLAMA_ARG_SPEC_TYPE": "draft-mtp",
        "LLAMA_ARG_SPEC_DRAFT_MODEL": "studio-managed.gguf",
        "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K": "q4_0",
        "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V": "q5_0",
        "LLAMA_ARG_N_GPU_LAYERS_DRAFT": "0",
    }
    assert llama_cpp_module._extra_args_requests_mtp([], env = studio_env) is True
    assert llama_cpp_module._extra_args_mtp_draft_path([], env = studio_env) == "studio-managed.gguf"
    assert llama_cpp_module._extra_args_draft_cache_types([], env = studio_env) == (
        "q4_0",
        "q5_0",
    )
    assert llama_cpp_module._extra_args_draft_offloaded_to_cpu([], env = studio_env) is True


def test_windows_command_length_counts_executable_quoting_and_nul(monkeypatch):
    monkeypatch.setattr(llama_cpp_module.sys, "platform", "win32")
    cmd = [r"C:\Program Files\llama-server.exe", "--model", r"C:\private model.gguf"]
    expected = len(llama_cpp_module.subprocess.list2cmdline(cmd).encode("utf-16-le")) // 2 + 1
    assert LlamaCppBackend._windows_command_line_utf16_units(cmd) == expected


def test_oversized_final_windows_command_is_rejected_before_kill_or_spawn(
    monkeypatch, tmp_path
):
    backend = _bare_backend(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(llama_cpp_module.sys, "platform", "win32")
    monkeypatch.setattr(llama_cpp_module, "_WINDOWS_CREATEPROCESS_MAX_UTF16_UNITS", 40)
    monkeypatch.setattr(backend, "_kill_process", lambda: calls.append("kill"))
    monkeypatch.setattr(
        llama_cpp_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: calls.append("spawn"),
    )

    with pytest.raises(LlamaServerArgsError) as excinfo:
        backend._start_llama_process(
            ["llama-server.exe", "--model", "x" * 80],
            {"PATH": "system-path"},
            [],
        )
    assert excinfo.value.code == "command_too_long"
    assert calls == []


def test_windows_preflight_counts_managed_and_custom_argv_before_teardown(monkeypatch):
    calls = []
    monkeypatch.setattr(llama_cpp_module.sys, "platform", "win32")
    monkeypatch.setattr(llama_cpp_module, "_WINDOWS_CREATEPROCESS_MAX_UTF16_UNITS", 80)
    monkeypatch.setattr(llama_cpp_module, "_WINDOWS_MANAGED_ARGV_RESERVE_UTF16_UNITS", 0)

    with pytest.raises(LlamaServerArgsError) as excinfo:
        LlamaCppBackend._preflight_windows_command_length(
            r"C:\llama-server.exe",
            model = "managed-model.gguf",
            mmproj = "managed-mmproj.gguf",
            draft_model = "managed-draft.gguf",
            model_alias = "private-alias",
            chat_template = None,
            port = 12345,
            extra_args = ["--future-safe", "custom-value"],
        )
        calls.append("teardown")
    assert excinfo.value.code == "command_too_long"
    assert calls == []
