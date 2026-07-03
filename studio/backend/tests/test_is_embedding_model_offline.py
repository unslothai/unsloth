# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""is_embedding_model must classify an already-downloaded model from the local
HF cache and honour HF_HUB_OFFLINE, instead of making a model_info() network call
that hangs on DNS retries when offline (#6817)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

import utils.models.model_config as mc  # noqa: E402


@pytest.fixture(autouse = True)
def _clean_state(monkeypatch):
    mc._embedding_detection_cache.clear()
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    yield
    mc._embedding_detection_cache.clear()


def _snapshot(tmp_path, name, *, sentence_transformer):
    d = tmp_path / name
    d.mkdir(parents = True, exist_ok = True)
    (d / "config.json").write_text("{}")
    if sentence_transformer:
        (d / "modules.json").write_text("[]")
    return d


def _fake_hf_model_info(monkeypatch, fn):
    fake = types.ModuleType("huggingface_hub")
    fake.model_info = fn
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


def _no_network(*a, **k):
    raise AssertionError("model_info() must not be called")


# ── _embedding_marker_in_hf_cache ──


def test_marker_true_when_modules_json_present(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path, "snap", sentence_transformer = True)
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter([snap]))
    assert mc._embedding_marker_in_hf_cache("org/emb") is True


def test_marker_false_when_cached_without_modules_json(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path, "snap", sentence_transformer = False)
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter([snap]))
    assert mc._embedding_marker_in_hf_cache("org/llm") is False


def test_marker_none_when_not_cached(monkeypatch):
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(()))
    assert mc._embedding_marker_in_hf_cache("org/llm") is None


# ── is_embedding_model ──


def test_cached_sentence_transformer_skips_network(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path, "snap", sentence_transformer = True)
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter([snap]))
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("unsloth/bge-small-en-v1.5") is True


def test_offline_cached_non_st_returns_false_without_network(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path, "snap", sentence_transformer = False)
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter([snap]))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("org/gemma-4-e4b") is False


def test_offline_not_cached_returns_false_without_network(monkeypatch):
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(()))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("org/never-downloaded") is False


def test_online_uncached_still_uses_network(monkeypatch):
    # Not offline, not cached: the network model_info path must still run so an
    # embedding model that lacks modules.json (feature-extraction tag) is caught.
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(()))
    calls = []

    def _info(model_name, token = None):
        calls.append(model_name)
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/gte-modernbert") is True
    assert calls == ["org/gte-modernbert"]
