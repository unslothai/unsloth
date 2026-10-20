# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("structlog")
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

import core.inference.llama_cpp as llama_cpp  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

# Verbatim from llama.cpp b11317 (load of a missing GGUF, then a normal start).
_NOISE = (
    "0.00.003.868 I srv  llama_server: initializing ...",
    "0.13.635.789 I srv    load_model: loading model '/nonexistent.gguf'",
    "0.34.561.161 I cmn          init: llama threadpool init, n_threads = 96",
    "0.13.636.121 I srv    operator(): operator(): cleaning up before exit...",
)
_NOTABLE = (
    "0.13.616.121 W srv  llama_server: security: no API key is set and CORS allows all origins",
    "0.13.635.929 E gguf_init_from_file: failed to open GGUF file '/nonexistent.gguf'",
    "0.13.650.267 E srv  llama_server: exiting due to model loading error",
)


class _Proc:
    def __init__(
        self,
        lines,
        returncode = None,
    ):
        self.stdout = iter(line + "\n" for line in lines)
        self.returncode = returncode

    def poll(self):
        return self.returncode


def _backend(lines):
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._process = _Proc(lines)
    b._stdout_lines = []
    return b


class _Recorder:
    def __init__(self):
        self.records = []

    def _at(level):
        def log(self, msg, *args, **kwargs):
            self.records.append((level, msg % args if args else msg))

        return log

    debug, info, warning, error = _at(10), _at(20), _at(30), _at(40)


def _logger(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(llama_cpp, "logger", rec)
    return rec


def _info_lines(rec, floor = 20):
    return [m for level, m in rec.records if level >= floor]


def test_only_warnings_errors_and_readiness_reach_info(monkeypatch):
    ready = "0.40.000.000 I srv  llama_server: server is listening on http://127.0.0.1:8080"
    log = _logger(monkeypatch)
    _backend(_NOISE + _NOTABLE + (ready,))._drain_stdout()
    info = _info_lines(log)
    assert info == [f"[llama-server] {line}" for line in _NOTABLE + (ready,)]


def test_builds_without_level_letters_fall_back_to_keywords(monkeypatch):
    log = _logger(monkeypatch)
    _backend(
        [
            "llama_model_loader: - kv 2: general.name str = test",
            "error: failed to load model",
            "warning: GPU backend unavailable",
        ]
    )._drain_stdout()
    assert _info_lines(log) == [
        "[llama-server] error: failed to load model",
        "[llama-server] warning: GPU backend unavailable",
    ]


def test_user_text_in_a_trace_dump_stays_debug(monkeypatch):
    log = _logger(monkeypatch)
    warning = "0.13.616.121 W srv  llama_server: security: no API key is set"
    _backend(
        [
            "0.29.142.020 D srv  params_from_: request: model loaded, it failed with an error",
            "and then model loaded",
            "an error on a continuation line",
            "hello W private prompt",
            warning,
            "the warning's own continuation",
        ]
    )._drain_stdout()
    assert _info_lines(log) == [
        f"[llama-server] {warning}",
        "[llama-server] the warning's own continuation",
    ]


def test_a_failing_tee_warns_once_and_a_closed_one_stays_quiet(monkeypatch):
    class _Broken:
        def __init__(self, exc):
            self.exc = exc

        def write(self, _):
            raise self.exc

    log = _logger(monkeypatch)
    b = _backend(_NOISE)
    b._llama_log_fh = _Broken(OSError("disk full"))
    b._drain_stdout()
    assert sum("disk full" in m for m in _info_lines(log)) == 1

    log.records.clear()
    b = _backend(_NOISE)
    b._llama_log_fh = _Broken(ValueError("I/O operation on closed file"))
    b._drain_stdout()
    assert not _info_lines(log, 30)


def test_close_writes_reason_and_exit_code_and_rearms_the_tee_warning(tmp_path, monkeypatch):
    log = _logger(monkeypatch)
    b = _backend([])
    b._process.returncode = 1
    b._llama_log_path = tmp_path / "llama.log"
    b._llama_log_fh = open(b._llama_log_path, "w", encoding = "utf-8")
    b._llama_log_tee_failed = True
    b._close_attempt_log(reason = "killed")
    assert b._llama_log_fh is None
    assert not b._llama_log_tee_failed
    assert b._llama_log_path.read_text(encoding = "utf-8").endswith(
        "attempt end reason=killed exit_code=1\n"
    )
    assert any("exit_code=1" in m for m in _info_lines(log))

    log.records.clear()
    b._close_attempt_log()
    assert not _info_lines(log)


def test_a_raising_logger_does_not_stop_the_drain():
    class _Raising:
        def __getattr__(self, name):
            def boom(*a, **k):
                raise ValueError("I/O operation on closed file")

            return boom

    class _Full:
        def write(self, _):
            raise OSError("disk full")

    lines = list(_NOTABLE) + list(_NOISE)
    b = _backend(lines)
    b._llama_log_fh = _Full()
    llama_cpp_logger = llama_cpp.logger
    llama_cpp.logger = _Raising()
    try:
        b._drain_stdout()
    finally:
        llama_cpp.logger = llama_cpp_logger
    assert b._stdout_lines == lines


def test_coloured_and_timestamp_free_prefixes_keep_their_level(monkeypatch):
    log = _logger(monkeypatch)
    # Verbatim from llama.cpp b11317 with --log-colors on / --no-log-timestamps.
    coloured = "\x1b[34m0.08.894.798\x1b[0m \x1b[35mW srv  llama_server: no API key is set"
    bare = "E gguf_init_from_file: cannot open GGUF file '/x.gguf'"
    _backend(
        [
            "\x1b[34m0.00.002.742\x1b[0m \x1b[32mI \x1b[0msrv  llama_server: initializing",
            coloured,
            bare,
        ]
    )._drain_stdout()
    assert _info_lines(log) == [f"[llama-server] {coloured}", f"[llama-server] {bare}"]
