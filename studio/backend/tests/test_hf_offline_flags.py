# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Effective-offline handling for the embedding preflight.

``huggingface_hub`` honors only ``HF_HUB_OFFLINE``; ``TRANSFORMERS_OFFLINE`` expresses the
same intent but does not stop a fetch. Studio treats either as offline (``hf_env_offline``)
and makes it real by passing ``local_files_only`` to the loader, which lets the Hub security
scan skip to its fail-open instead of burning both timeouts. That skip is sound only while
the loader is pinned to the local cache, so the coupling is pinned here as an invariant.
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
    # Stub only if the real module is unavailable, so this file never shadows real packages.
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
def test_shared_gate_still_scans_when_offline_by_default(monkeypatch, var):
    # Shared gate: an offline env var alone must never disable the scan for a fetching loader.
    import utils.security.file_security as fs

    calls: list = []
    _fake_hub(monkeypatch, calls)
    monkeypatch.setenv(var, "1")
    assert fs._fetch_security_status("org/model", None) is None
    assert calls, f"{var} alone must NOT bypass the shared malware gate"


def test_security_scan_short_circuits_for_a_local_only_caller(monkeypatch):
    import utils.security.file_security as fs

    calls: list = []
    _fake_hub(monkeypatch, calls)
    assert fs._fetch_security_status("org/model", None, True) is None
    assert calls == [], "a local-only caller must not hit the Hub"


def test_security_scan_runs_when_online(monkeypatch):
    import utils.security.file_security as fs

    calls: list = []
    _fake_hub(monkeypatch, calls)
    assert fs._fetch_security_status("org/model", None) is None
    assert calls, "online must attempt the Hub"


def _read_backend(rel: str) -> str:
    return (Path(__file__).resolve().parents[1] / rel).read_text(encoding = "utf-8")


def test_embedding_loader_forces_local_only_when_offline():
    """The invariant the RAG opt-in rests on: embeddings.py may pass local_only_load
    because its loader is pinned to the local cache by the SAME value, read ONCE and shared
    (two hf_env_offline() reads can disagree since _hf_offline_if_dns_dead() flips the vars).
    Checked at source level because importing the loader drags in sentence_transformers.
    """
    src = _read_backend("core/rag/embeddings.py")
    assert (
        "local_only = hf_env_offline()" in src
    ), "core/rag/embeddings.py must capture the offline state once in _get()"
    assert (
        "_guard_model_security(load_name, local_only)" in src
    ), "the security guard must receive the captured value, not re-read the env"
    assert "local_files_only = local_only" in src, (
        "SentenceTransformer must be pinned with the SAME captured value; a second "
        "hf_env_offline() read can flip to False and fetch the unscanned repo"
    )
    guard = src.split("def _guard_model_security", 1)[1].split("\ndef ", 1)[0]
    assert "hf_env_offline()" not in guard, (
        "_guard_model_security must take local_only_load as an argument so it "
        "cannot observe a different offline state than the loader"
    )


def test_only_the_rag_embedding_path_opts_into_the_bypass():
    """No other loader may claim local-only without constraining its loader: the
    MLX/inference, training and export gates call from_pretrained without a local-only
    argument, so passing local_only_load there would disable the gate for a fetching path.
    """
    allowed = {"core/rag/embeddings.py", "routes/settings.py"}
    callers = [
        "core/inference/worker.py",
        "core/training/worker.py",
        "core/export/worker.py",
        "routes/models.py",
        "routes/inference.py",
    ]
    for rel in callers:
        assert rel not in allowed
        assert "local_only_load" not in _read_backend(rel), (
            f"{rel} passes local_only_load but does not pin its loader to the "
            "local cache; that would disable the malware gate for a fetching path"
        )
