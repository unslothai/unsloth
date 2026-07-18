# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Effective-offline handling for the embedding preflight.

``huggingface_hub`` honors only ``HF_HUB_OFFLINE``; ``TRANSFORMERS_OFFLINE``
expresses the same user intent but does not itself stop a fetch. Studio treats
either as offline (``hf_env_offline``) and makes that real by passing
``local_files_only`` to the loader, which lets the metadata-only Hub security
scan skip straight to its documented fail-open instead of burning both request
timeouts on a session the user declared offline.

That skip is only sound while the loader really is pinned to the local cache, so
the coupling between the two is pinned here as an explicit invariant rather than
left to a comment.
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

from utils.utils import hf_env_offline  # noqa: E402


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    yield


# ── flag semantics ──


def test_neither_flag_is_online():
    assert hf_env_offline() is False


@pytest.mark.parametrize("var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_either_flag_means_offline(monkeypatch, var):
    monkeypatch.setenv(var, "1")
    assert hf_env_offline() is True


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "  1  "])
def test_truthy_values_parse(monkeypatch, value):
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    assert hf_env_offline() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_non_truthy_values_do_not_parse(monkeypatch, value):
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    assert hf_env_offline() is False


# ── the offline security short-circuit, and the invariant it rests on ──


def _fake_hub(monkeypatch, calls):
    fake = types.ModuleType("huggingface_hub")

    def _model_info(*a, **k):
        calls.append(1)
        raise RuntimeError("network unreachable")

    fake.model_info = _model_info
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


@pytest.mark.parametrize("var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_security_scan_short_circuits_when_offline(monkeypatch, var):
    # Metadata-only lookup with no local fallback: on a session the user declared
    # offline it must skip straight to its documented fail-open rather than burn
    # both request timeouts (10s + 20s) on every save and load.
    import utils.security.file_security as fs

    calls: list = []
    _fake_hub(monkeypatch, calls)
    monkeypatch.setenv(var, "1")
    assert fs._fetch_security_status("org/model", None) is None
    assert calls == [], f"the scan must not hit the Hub under {var}"


def test_security_scan_runs_when_online(monkeypatch):
    import utils.security.file_security as fs

    calls: list = []
    _fake_hub(monkeypatch, calls)
    assert fs._fetch_security_status("org/model", None) is None  # fails open on error
    assert calls, "online must attempt the Hub"


def test_embedding_loader_forces_local_only_when_offline():
    """The invariant the offline scan skip rests on.

    Skipping the Hub security scan offline is only sound while every loader
    behind that gate is pinned to the local cache by the SAME predicate -- if the
    loader could still fetch, an unscanned repo's pickle would be downloaded and
    deserialized. Pinned at source level because importing the loader would drag
    in sentence_transformers/torch.
    """
    src = (Path(__file__).resolve().parents[1] / "core" / "rag" / "embeddings.py").read_text(
        encoding = "utf-8"
    )
    assert "local_files_only = hf_env_offline()" in src, (
        "core/rag/embeddings.py must pin SentenceTransformer to local files when "
        "hf_env_offline(); without it the offline security-scan skip in "
        "utils/security/file_security.py is unsafe"
    )
