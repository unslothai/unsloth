# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two HF offline flags are not interchangeable.

``huggingface_hub`` honors only ``HF_HUB_OFFLINE``. ``TRANSFORMERS_OFFLINE``
expresses the same user intent but does NOT stop a Hub fetch, so it may be used
to pass ``local_files_only`` (a real guarantee) but must never be used to skip a
security gate -- doing so would wave through the very download the gate exists
to block.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def _maybe_stub(name: str, builder):
    # Stub only if the real module is unavailable, so this file never shadows
    # real packages for later tests in the same pytest process.
    try:
        importlib.import_module(name)
    except ImportError:
        sys.modules[name] = builder()


def _build_loggers_stub():
    m = types.ModuleType("loggers")
    m.get_logger = lambda name: __import__("logging").getLogger(name)
    return m


def _build_structlog_stub():
    m = types.ModuleType("structlog")
    m.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
    return m


_maybe_stub("loggers", _build_loggers_stub)
_maybe_stub("structlog", _build_structlog_stub)

from utils.utils import hf_env_offline, hf_hub_offline  # noqa: E402


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    yield


# ── flag semantics ──


def test_neither_flag_is_online():
    assert hf_hub_offline() is False
    assert hf_env_offline() is False


def test_transformers_offline_is_intent_but_not_a_fetch_guarantee(monkeypatch):
    # The distinction the security gate depends on: intent yes, guarantee no.
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert hf_env_offline() is True
    assert hf_hub_offline() is False


def test_hub_offline_sets_both(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert hf_hub_offline() is True
    assert hf_env_offline() is True


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "  1  "])
def test_truthy_values_parse(monkeypatch, value):
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    assert hf_hub_offline() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_non_truthy_values_do_not_parse(monkeypatch, value):
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    assert hf_hub_offline() is False


# ── the security gate must not be skipped on the weaker flag ──


def _fake_hub(monkeypatch, calls):
    fake = types.ModuleType("huggingface_hub")

    def _model_info(*a, **k):
        calls.append(1)
        raise RuntimeError("network unreachable")

    fake.model_info = _model_info
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


def test_security_scan_still_runs_under_transformers_offline(monkeypatch):
    # TRANSFORMERS_OFFLINE does NOT stop SentenceTransformer from fetching, so
    # the scan must still be attempted; skipping it here would let an unscanned
    # repo be downloaded and its pickle deserialized anyway.
    import utils.security.file_security as fs

    calls: list = []
    _fake_hub(monkeypatch, calls)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert fs._fetch_security_status("org/model", None) is None  # fails open
    assert calls, "the security scan must not be skipped for TRANSFORMERS_OFFLINE"


def test_security_scan_short_circuits_under_hub_offline(monkeypatch):
    # With HF_HUB_OFFLINE no fetch is possible, so this metadata-only lookup can
    # skip straight to its documented fail-open instead of burning both timeouts.
    import utils.security.file_security as fs

    calls: list = []
    _fake_hub(monkeypatch, calls)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert fs._fetch_security_status("org/model", None) is None
    assert calls == [], "no Hub call is possible when HF_HUB_OFFLINE is set"


def test_security_scan_runs_when_online(monkeypatch):
    import utils.security.file_security as fs

    calls: list = []
    _fake_hub(monkeypatch, calls)
    assert fs._fetch_security_status("org/model", None) is None  # fails open on error
    assert calls, "online must attempt the Hub"
