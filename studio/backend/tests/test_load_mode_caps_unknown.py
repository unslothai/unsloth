# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A --help probe that fails must not turn into --no-mmap / --mlock on a build that dropped them."""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

from core.inference.llama_cpp import LlamaCppBackend
from core.inference.llama_server_args import apply_load_mode_policy, resolve_launch_load_mode

_NO_SETTINGS = (False, False)


def _fake_server(tmp_path, body: str) -> str:
    fake = tmp_path / "llama-server"
    fake.write_text("#!/bin/sh\n" + body + "\n")
    fake.chmod(0o755)
    return str(fake)


def _launch_flags(caps) -> list[str]:
    from core.inference.llama_server_args import load_mode_support

    selected, _ = apply_load_mode_policy(
        [],
        supports_load_mode = load_mode_support(caps),
        requested_load_mode = "none",
        settings = _NO_SETTINGS,
    )
    return selected


@pytest.mark.parametrize("mode", ["none", "mlock", "mmap+mlock"])
def test_unknown_support_emits_no_legacy_flag(mode):
    assert apply_load_mode_policy(
        [], supports_load_mode = None, requested_load_mode = mode, settings = _NO_SETTINGS
    ) == ([], [])


def test_unknown_support_keeps_the_extras():
    assert apply_load_mode_policy(
        ["--threads", "4"],
        supports_load_mode = None,
        requested_load_mode = "none",
        settings = _NO_SETTINGS,
    ) == ([], ["--threads", "4"])


def test_resolve_launch_load_mode_accepts_unknown_support():
    assert resolve_launch_load_mode(
        [],
        supports_load_mode = None,
        weights_in_host_memory = True,
        gpu_offload_confirmed = False,
        requested_load_mode = "none",
        env = {},
        settings = _NO_SETTINGS,
    ) == (False, False)


def test_a_failed_help_probe_emits_no_load_flag(tmp_path):
    caps = LlamaCppBackend.probe_server_capabilities(_fake_server(tmp_path, "exit 1"))
    assert caps["help_probe_ok"] is False and caps["supports_load_mode"] is False
    assert _launch_flags(caps) == []


def test_an_empty_help_probe_emits_no_load_flag(tmp_path):
    caps = LlamaCppBackend.probe_server_capabilities(_fake_server(tmp_path, "exit 0"))
    assert caps["help_probe_ok"] is True and not caps["flags"]
    assert _launch_flags(caps) == []


def test_a_build_listing_only_the_old_flags_still_gets_them(tmp_path):
    help_text = "--no-mmap                do not memory-map model\n--mlock                  lock model in RAM"
    caps = LlamaCppBackend.probe_server_capabilities(
        _fake_server(tmp_path, f"cat <<'X'\n{help_text}\nX\nexit 0")
    )
    assert caps["help_probe_ok"] is True and caps["supports_load_mode"] is False
    assert _launch_flags(caps) == ["--no-mmap"]


def test_a_build_listing_load_mode_gets_it(tmp_path):
    help_text = "--load-mode MODE         how to load the model"
    caps = LlamaCppBackend.probe_server_capabilities(
        _fake_server(tmp_path, f"cat <<'X'\n{help_text}\nX\nexit 0")
    )
    assert _launch_flags(caps) == ["--load-mode", "none"]
