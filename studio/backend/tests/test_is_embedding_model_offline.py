# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""is_embedding_model must classify an already-downloaded model from the local
HF cache and honour HF_HUB_OFFLINE, instead of making a model_info() network call
that hangs on DNS retries when offline (#6817)."""

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

import utils.models.model_config as mc  # noqa: E402


@pytest.fixture(autouse = True)
def _clean_state(monkeypatch):
    mc._embedding_detection_cache.clear()
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    yield
    mc._embedding_detection_cache.clear()


def _repo(
    tmp_path,
    *snapshots,
    main_ref = None,
):
    """Fake HF cache repo dir: snapshots/<name>[/modules.json] (+ refs/main).

    ``snapshots``: (name, sentence_transformer) tuples, oldest last (the
    iterator under test yields newest first, so pass them in that order).
    Returns the snapshot dirs in the given order.
    """
    repo = tmp_path / "models--org--model"
    dirs = []
    for name, is_st in snapshots:
        d = repo / "snapshots" / name
        d.mkdir(parents = True, exist_ok = True)
        (d / "config.json").write_text("{}")
        if is_st:
            (d / "modules.json").write_text("[]")
        dirs.append(d)
    if main_ref is not None:
        refs = repo / "refs"
        refs.mkdir(parents = True, exist_ok = True)
        (refs / "main").write_text(main_ref)
    return dirs


def _fake_hf_model_info(monkeypatch, fn):
    fake = types.ModuleType("huggingface_hub")
    fake.model_info = fn
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


def _no_network(*a, **k):
    raise AssertionError("model_info() must not be called")


# ── _embedding_marker_in_hf_cache ──


def test_marker_true_when_modules_json_present(tmp_path, monkeypatch):
    snaps = _repo(tmp_path, ("aaa", True))
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(snaps))
    assert mc._embedding_marker_in_hf_cache("org/emb") is True


def test_marker_false_when_cached_without_modules_json(tmp_path, monkeypatch):
    snaps = _repo(tmp_path, ("aaa", False))
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(snaps))
    assert mc._embedding_marker_in_hf_cache("org/llm") is False


def test_marker_none_when_not_cached(monkeypatch):
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(()))
    assert mc._embedding_marker_in_hf_cache("org/llm") is None


def test_marker_prefers_refs_main_revision(tmp_path, monkeypatch):
    # The repo USED to be a sentence-transformers model (old snapshot has
    # modules.json) but the revision refs/main points at no longer is. The
    # active revision must win: an any-snapshot scan would wrongly say True.
    snaps = _repo(tmp_path, ("new", False), ("old", True), main_ref = "new")
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(snaps))
    assert mc._embedding_marker_in_hf_cache("org/was-embedder") is False


def test_marker_refs_main_st_revision_is_true(tmp_path, monkeypatch):
    snaps = _repo(tmp_path, ("new", True), ("old", False), main_ref = "new")
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(snaps))
    assert mc._embedding_marker_in_hf_cache("org/is-embedder") is True


def test_marker_missing_ref_falls_back_to_snapshot_scan(tmp_path, monkeypatch):
    # No refs/main recorded: keep the newest-first any-snapshot behavior.
    snaps = _repo(tmp_path, ("new", False), ("old", True))
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(snaps))
    assert mc._embedding_marker_in_hf_cache("org/no-ref") is True


def test_marker_never_raises_when_cache_mutates(monkeypatch):
    # A snapshot vanishing mid-iteration (concurrent cached-model deletion)
    # must read as not-cached, not propagate a 500 out of the routes.
    def _exploding_iter(repo):
        raise FileNotFoundError("snapshot removed underneath")

    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", _exploding_iter)
    assert mc._embedding_marker_in_hf_cache("org/racing") is None


def test_is_embedding_model_survives_cache_race_online(monkeypatch):
    # With the cache probe failing, the online path must still resolve via the
    # Hub instead of erroring out.
    def _exploding_iter(repo):
        raise FileNotFoundError("snapshot removed underneath")

    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", _exploding_iter)
    _fake_hf_model_info(
        monkeypatch,
        lambda name, token = None: types.SimpleNamespace(
            tags = ["sentence-transformers"], pipeline_tag = None
        ),
    )
    assert mc.is_embedding_model("org/racing") is True


# ── is_embedding_model ──


def test_cached_sentence_transformer_skips_network(tmp_path, monkeypatch):
    snaps = _repo(tmp_path, ("aaa", True))
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(snaps))
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("unsloth/bge-small-en-v1.5") is True


def test_offline_cached_non_st_returns_false_without_network(tmp_path, monkeypatch):
    snaps = _repo(tmp_path, ("aaa", False))
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(snaps))
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
