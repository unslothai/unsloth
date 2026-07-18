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


def test_marker_ref_points_at_absent_snapshot_is_cache_miss(tmp_path, monkeypatch):
    # refs/main names a commit whose snapshot dir is absent (partial download /
    # pruning). The recorded ref is authoritative, so this is a cache miss
    # (None) -- NOT a fall-through to a stale historical snapshot that has
    # modules.json.
    snaps = _repo(tmp_path, ("old", True), main_ref = "missing_commit")
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(snaps))
    assert mc._embedding_marker_in_hf_cache("org/partial") is None


def test_marker_unreadable_ref_is_cache_miss(tmp_path, monkeypatch):
    # refs/main exists but cannot be read (transient I/O error / restrictive
    # permissions). Only a genuinely MISSING ref may enable the historical scan;
    # an unreadable ref is a cache miss (None), never a fall-through to a stale
    # snapshot that happens to carry modules.json.
    snaps = _repo(tmp_path, ("old", True))
    # Make refs/main a directory so read_text() raises IsADirectoryError -- an
    # OSError that is NOT FileNotFoundError, i.e. "exists but unreadable".
    refs_main = snaps[0].parent.parent / "refs" / "main"
    refs_main.parent.mkdir(parents = True, exist_ok = True)
    refs_main.mkdir()
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(snaps))
    assert mc._embedding_marker_in_hf_cache("org/unreadable-ref") is None


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


def test_offline_cached_st_detected_via_marker_no_network(tmp_path, monkeypatch):
    # Offline: a downloaded sentence-transformers repo is classified from its
    # modules.json marker with no model_info() network call that would hang on
    # DNS retries (#6817).
    snaps = _repo(tmp_path, ("aaa", True))
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(snaps))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("unsloth/bge-small-en-v1.5") is True


def test_online_defers_to_hub_over_stale_marker(tmp_path, monkeypatch):
    # Online: the Hub is authoritative for the current revision. Even with a
    # cached modules.json (the repo WAS an embedder), a Hub lookup that no longer
    # reports embedding signals wins -- the stale local marker must not
    # short-circuit model_info().
    snaps = _repo(tmp_path, ("aaa", True))
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(snaps))
    calls = []

    def _info(model_name, token = None):
        calls.append(model_name)
        return types.SimpleNamespace(tags = ["text-generation"], pipeline_tag = "text-generation")

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/was-embedder") is False
    assert calls == ["org/was-embedder"]  # Hub consulted, not skipped


def test_online_hub_failure_falls_back_to_marker_uncached(tmp_path, monkeypatch):
    # A transient model_info() failure falls back to the local marker WITHOUT
    # caching: the degraded result must not become sticky, so a later successful
    # Hub lookup can still override it.
    snaps = _repo(tmp_path, ("aaa", True))
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(snaps))
    _fake_hf_model_info(monkeypatch, _no_network)  # raises -> Hub "unreachable"
    assert mc.is_embedding_model("org/emb") is True  # marker fallback
    assert ("org/emb", None) not in mc._embedding_detection_cache  # not poisoned


def test_online_negative_does_not_block_later_offline_download(tmp_path, monkeypatch):
    # An online Hub lookup authoritatively reports non-embedding and is memoized.
    # The repo is then downloaded WITH modules.json and the session goes offline;
    # the offline path re-probes the marker (never consulting the online memo),
    # so the freshly downloaded embedder is detected instead of the stale False.
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(()))

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["text-generation"], pipeline_tag = "text-generation")

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/late-embedder") is False  # online: Hub says no
    assert mc._embedding_detection_cache[("org/late-embedder", None)] is False

    # Now the model is downloaded (marker appears) and the session goes offline.
    snaps = _repo(tmp_path, ("aaa", True))
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(snaps))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert mc.is_embedding_model("org/late-embedder") is True  # marker re-probed


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


def test_offline_negative_is_not_cached_then_online_detects(monkeypatch):
    # A tag-only embedder is not identifiable from modules.json. Offline returns
    # False WITHOUT caching, so once the env var clears the online model_info
    # lookup still runs and detects it -- the negative must not be sticky.
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: iter(()))
    calls = []

    def _info(model_name, token = None):
        calls.append(model_name)
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert mc.is_embedding_model("org/gte-modernbert") is False
    assert calls == []  # offline: no network
    assert ("org/gte-modernbert", None) not in mc._embedding_detection_cache

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    assert mc.is_embedding_model("org/gte-modernbert") is True  # now detected online
    assert calls == ["org/gte-modernbert"]
