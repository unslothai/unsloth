# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Refcounted HF offline guard semantics.

Overlapping local-only validations/loads share one process-global env
override. The refcount means a request finishing first cannot restore the
environment while another local-only request still runs (which would let its
remaining metadata checks reach the Hub), and forced mode overrides an
explicitly falsy HF_HUB_OFFLINE=0 then restores it. The guard is extracted
from source and exercised with stubbed logging/DNS so no ML dependencies are
needed.
"""

from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path

import pytest

_LLAMA_CPP = Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_cpp.py"


class _NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _load_guard(dns_dead: bool = False):
    src = _LLAMA_CPP.read_text()
    start = src.index("# Overlapping offline guards")
    end = src.index("_SLOT_SAVE_MAX_BYTES")
    end = src.rindex("try:", start, end)
    block = src[start:end]
    ns = {
        "threading": threading,
        "contextlib": contextlib,
        "os": os,
        "logger": _NullLogger(),
        "_hf_env_offline": lambda: os.environ.get("HF_HUB_OFFLINE", "").strip().lower()
        in {"1", "true", "yes", "on"},
        "_probe_dns_dead": lambda: dns_dead,
    }
    exec(block, ns)
    return ns["_hf_offline_if_dns_dead"]


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)


def test_overlapping_guards_restore_only_after_last_exit(clean_env):
    guard = _load_guard()
    a = guard(force = True)
    b = guard(force = True)
    assert a.__enter__() is True
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert b.__enter__() is True
    a.__exit__(None, None, None)
    assert os.environ.get("HF_HUB_OFFLINE") == "1", (
        "first exit must not restore while another guard is active"
    )
    b.__exit__(None, None, None)
    assert "HF_HUB_OFFLINE" not in os.environ


def test_force_overrides_and_restores_falsy_env(clean_env):
    guard = _load_guard()
    os.environ["HF_HUB_OFFLINE"] = "0"
    g = guard(force = True)
    assert g.__enter__() is True
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    g.__exit__(None, None, None)
    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_falsy_env_stays_authoritative_for_ordinary_loads(clean_env):
    guard = _load_guard(dns_dead = True)
    os.environ["HF_HUB_OFFLINE"] = "0"
    g = guard(force = False)
    assert g.__enter__() is False
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    g.__exit__(None, None, None)


def test_truthy_user_env_is_a_noop(clean_env):
    guard = _load_guard()
    os.environ["HF_HUB_OFFLINE"] = "1"
    g = guard(force = True)
    assert g.__enter__() is False
    g.__exit__(None, None, None)
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_nonforce_joins_active_override(clean_env):
    """A DNS-alive non-force guard entering while a forced guard is active must
    JOIN the refcount (deferring the restore) rather than no-op."""
    guard = _load_guard()
    forced = guard(force = True)
    plain = guard(force = False)
    assert forced.__enter__() is True
    assert plain.__enter__() is True
    forced.__exit__(None, None, None)
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    plain.__exit__(None, None, None)
    assert "HF_HUB_OFFLINE" not in os.environ
