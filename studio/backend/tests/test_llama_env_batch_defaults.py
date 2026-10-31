# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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

import httpx  # noqa: F401

from core.inference.llama_cpp import (
    _ENV_DEFAULT_N_BATCH,
    _ENV_DEFAULT_N_UBATCH,
    _env_positive_int,
    _launcher_batch_default,
)


@pytest.mark.parametrize("name", [_ENV_DEFAULT_N_BATCH, _ENV_DEFAULT_N_UBATCH])
def test_env_positive_int(monkeypatch, name):
    monkeypatch.delenv(name, raising = False)
    assert _env_positive_int(name) is None
    for raw, expected in [
        ("1024", 1024),
        (" 256 ", 256),
        ("999999", 65536),
        ("0", None),
        ("-5", None),
        ("abc", None),
        ("1.5", None),
        ("", None),
    ]:
        monkeypatch.setenv(name, raw)
        assert _env_positive_int(name) == expected, raw


def test_per_load_value_wins_over_env(monkeypatch):
    monkeypatch.setenv(_ENV_DEFAULT_N_BATCH, "1024")
    assert _launcher_batch_default(4096, _ENV_DEFAULT_N_BATCH) == 4096
    assert _launcher_batch_default(None, _ENV_DEFAULT_N_BATCH) == 1024


def test_unset_or_invalid_env_leaves_none(monkeypatch):
    monkeypatch.delenv(_ENV_DEFAULT_N_UBATCH, raising = False)
    assert _launcher_batch_default(None, _ENV_DEFAULT_N_UBATCH) is None
    monkeypatch.setenv(_ENV_DEFAULT_N_UBATCH, "nope")
    assert _launcher_batch_default(None, _ENV_DEFAULT_N_UBATCH) is None
