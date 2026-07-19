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
        (d / "model.safetensors").write_bytes(b"\0")
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
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(snaps))
    assert mc._embedding_marker_in_hf_cache("org/emb") is True


def test_marker_false_when_cached_without_modules_json(tmp_path, monkeypatch):
    snaps = _repo(tmp_path, ("aaa", False))
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(snaps))
    assert mc._embedding_marker_in_hf_cache("org/llm") is False


def test_marker_none_when_not_cached(monkeypatch):
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(()))
    assert mc._embedding_marker_in_hf_cache("org/llm") is None


def test_marker_prefers_refs_main_revision(tmp_path, monkeypatch):
    # The repo USED to be a sentence-transformers model (old snapshot has
    # modules.json) but the revision refs/main points at no longer is. The
    # active revision must win: an any-snapshot scan would wrongly say True.
    snaps = _repo(tmp_path, ("new", False), ("old", True), main_ref = "new")
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(snaps))
    assert mc._embedding_marker_in_hf_cache("org/was-embedder") is False


def test_marker_refs_main_st_revision_is_true(tmp_path, monkeypatch):
    snaps = _repo(tmp_path, ("new", True), ("old", False), main_ref = "new")
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(snaps))
    assert mc._embedding_marker_in_hf_cache("org/is-embedder") is True


def test_marker_missing_ref_falls_back_to_snapshot_scan(tmp_path, monkeypatch):
    # No refs/main recorded: keep the newest-first any-snapshot behavior.
    snaps = _repo(tmp_path, ("new", False), ("old", True))
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(snaps))
    assert mc._embedding_marker_in_hf_cache("org/no-ref") is True


def test_marker_ref_points_at_absent_snapshot_is_cache_miss(tmp_path, monkeypatch):
    # refs/main names a commit whose snapshot dir is absent (partial download /
    # pruning). The recorded ref is authoritative, so this is a cache miss
    # (None) -- NOT a fall-through to a stale historical snapshot that has
    # modules.json.
    snaps = _repo(tmp_path, ("old", True), main_ref = "missing_commit")
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(snaps))
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
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(snaps))
    assert mc._embedding_marker_in_hf_cache("org/unreadable-ref") is None


def test_marker_empty_ref_is_cache_miss(tmp_path, monkeypatch):
    # refs/main exists but is empty / whitespace (a partial write or in-progress
    # truncate-and-rewrite): the active revision is unknown, so this is a cache
    # miss (None), NOT a fall-through to a stale snapshot that carries modules.json.
    snaps = _repo(tmp_path, ("old", True), main_ref = "   \n")
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(snaps))
    assert mc._embedding_marker_in_hf_cache("org/empty-ref") is None


def _fake_hf_cache(monkeypatch, root):
    fake = types.ModuleType("huggingface_hub")
    fake.constants = types.SimpleNamespace(HF_HUB_CACHE = str(root))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", fake.constants)


def _st_snapshot(
    root,
    repo_dir,
    commit = "aaa",
    loadable = True,
):
    snap = root / repo_dir / "snapshots" / commit
    snap.mkdir(parents = True)
    (snap / "modules.json").write_text("[]")
    if loadable:
        (snap / "config.json").write_text("{}")
        (snap / "model.safetensors").write_bytes(b"\0")
    return snap


def test_st_probe_uses_sentence_transformers_home(tmp_path, monkeypatch):
    # _get() builds SentenceTransformer without cache_folder, so with ST_HOME set
    # that is the ONLY cache the load searches. The probe must follow it, or a
    # model present there is called uncached and rejected with a 409 offline.
    hf_root, st_root = tmp_path / "hf", tmp_path / "st"
    hf_root.mkdir()
    _st_snapshot(st_root, "models--org--model")
    _fake_hf_cache(monkeypatch, hf_root)
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(st_root))
    assert [p.name for p in mc._iter_st_cache_snapshots("org/model")] == ["aaa"]
    assert mc._embedding_marker_in_hf_cache("org/model") is True


def test_st_probe_ignores_hub_cache_when_st_home_is_set(tmp_path, monkeypatch):
    # The loader searches ST_HOME only, so a repo cached ONLY in the Hub cache
    # must not validate -- otherwise validation passes and the load then fails.
    hf_root, st_root = tmp_path / "hf", tmp_path / "st"
    st_root.mkdir()
    _st_snapshot(hf_root, "models--org--model")
    _fake_hf_cache(monkeypatch, hf_root)
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(st_root))
    assert mc._embedding_marker_in_hf_cache("org/model") is None


def test_gguf_probe_never_follows_st_home(tmp_path, monkeypatch):
    # The GGUF path downloads with hf_hub_download and no cache_dir, so it uses
    # the Hub cache. Letting its probe see ST_HOME would select a file the GGUF
    # loader cannot find.
    hf_root, st_root = tmp_path / "hf", tmp_path / "st"
    hf_root.mkdir()
    _st_snapshot(st_root, "models--org--model")
    _fake_hf_cache(monkeypatch, hf_root)
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(st_root))
    assert list(mc._iter_hf_cache_snapshots("org/model")) == []


def test_st_probe_falls_back_to_hub_cache(tmp_path, monkeypatch):
    hf_root = tmp_path / "hf"
    hf_root.mkdir()
    _fake_hf_cache(monkeypatch, hf_root)
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising = False)
    assert mc._st_cache_roots() == [Path(str(hf_root))]


def test_marker_only_snapshot_is_not_loadable(tmp_path, monkeypatch):
    # The online security preflight downloads modules.json on its own via
    # hf_hub_download, and a partial download leaves it too. Accepting that
    # offline passes validation and then fails on the first RAG load.
    hf_root = tmp_path / "hf"
    _st_snapshot(hf_root, "models--org--model", loadable = False)
    _fake_hf_cache(monkeypatch, hf_root)
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising = False)
    assert mc._embedding_marker_in_hf_cache("org/model") is False


def test_marker_onnx_only_snapshot_is_not_loadable(tmp_path, monkeypatch):
    # A snapshot with the marker and config but ONLY an ONNX export cached is not
    # loadable: _get() builds SentenceTransformer with the default Torch backend
    # (no backend="onnx"), which reads model.safetensors / pytorch_model.bin, so
    # accepting the ONNX offline would pass validation and then fail on the first
    # RAG load -- the same validate-then-fail as the marker-only case.
    hf_root = tmp_path / "hf"
    snap = hf_root / "models--org--model" / "snapshots" / "aaa"
    snap.mkdir(parents = True)
    (snap / "modules.json").write_text("[]")
    (snap / "config.json").write_text("{}")
    (snap / "model.onnx").write_bytes(b"\0")
    _fake_hf_cache(monkeypatch, hf_root)
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising = False)
    assert mc._embedding_marker_in_hf_cache("org/model") is False


def _case_sensitive_fs(tmp_path) -> bool:
    probe = tmp_path / "_CaseProbe"
    probe.mkdir()
    return not (tmp_path / "_caseprobe").is_dir()


def test_st_casing_resolves_against_st_home(tmp_path, monkeypatch):
    # resolve_cached_repo_id_case scans only the Hub cache, so with ST_HOME set it
    # would persist the requested lower-case id while the exact-case offline load
    # looks for models--BAAI--bge-m3 in ST_HOME and misses it.
    if not _case_sensitive_fs(tmp_path):
        pytest.skip("casing only diverges on a case-sensitive filesystem")
    hf_root, st_root = tmp_path / "hf", tmp_path / "st"
    hf_root.mkdir()
    (st_root / "models--BAAI--bge-m3").mkdir(parents = True)
    _fake_hf_cache(monkeypatch, hf_root)
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(st_root))
    assert mc.resolve_st_cached_repo_id_case("baai/bge-m3") == "BAAI/bge-m3"


def test_st_casing_prefers_an_exact_match(tmp_path, monkeypatch):
    if not _case_sensitive_fs(tmp_path):
        pytest.skip("casing only diverges on a case-sensitive filesystem")
    hf_root, st_root = tmp_path / "hf", tmp_path / "st"
    hf_root.mkdir()
    (st_root / "models--baai--bge-m3").mkdir(parents = True)
    (st_root / "models--BAAI--bge-m3").mkdir(parents = True)
    _fake_hf_cache(monkeypatch, hf_root)
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(st_root))
    assert mc.resolve_st_cached_repo_id_case("baai/bge-m3") == "baai/bge-m3"


def test_st_casing_noop_when_uncached(tmp_path, monkeypatch):
    hf_root = tmp_path / "hf"
    hf_root.mkdir()
    _fake_hf_cache(monkeypatch, hf_root)
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising = False)
    assert mc.resolve_st_cached_repo_id_case("org/not-cached") == "org/not-cached"


def test_marker_never_raises_when_cache_mutates(monkeypatch):
    # A snapshot vanishing mid-iteration (concurrent cached-model deletion)
    # must read as not-cached, not propagate a 500 out of the routes.
    def _exploding_iter(repo):
        raise FileNotFoundError("snapshot removed underneath")

    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", _exploding_iter)
    assert mc._embedding_marker_in_hf_cache("org/racing") is None


def test_is_embedding_model_survives_cache_race_online(monkeypatch):
    # With the cache probe failing, the online path must still resolve via the
    # Hub instead of erroring out.
    def _exploding_iter(repo):
        raise FileNotFoundError("snapshot removed underneath")

    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", _exploding_iter)
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
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(snaps))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("unsloth/bge-small-en-v1.5") is True


def test_online_defers_to_hub_over_stale_marker(tmp_path, monkeypatch):
    # Online: the Hub is authoritative for the current revision. Even with a
    # cached modules.json (the repo WAS an embedder), a Hub lookup that no longer
    # reports embedding signals wins -- the stale local marker must not
    # short-circuit model_info().
    snaps = _repo(tmp_path, ("aaa", True))
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(snaps))
    calls = []

    def _info(model_name, token = None):
        calls.append(model_name)
        return types.SimpleNamespace(tags = ["text-generation"], pipeline_tag = "text-generation")

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/was-embedder") is False
    assert calls == ["org/was-embedder"]  # Hub consulted, not skipped


def test_online_permanent_hub_error_ignores_stale_marker(tmp_path, monkeypatch):
    # A permanent Hub error (deleted / gated / typo'd repo) is authoritative:
    # even with a cached modules.json, validation must NOT pass on the stale
    # marker -- return False so the settings route surfaces its 409, and the
    # persisted model can't fail later when the loader refreshes from the Hub.
    snaps = _repo(tmp_path, ("aaa", True))
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(snaps))

    class RepositoryNotFoundError(Exception):
        pass

    def _info(model_name, token = None):
        raise RepositoryNotFoundError("404 not found")

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/deleted-or-typo") is False


def test_online_hub_failure_falls_back_to_marker_uncached(tmp_path, monkeypatch):
    # A transient model_info() failure falls back to the local marker WITHOUT
    # caching: the degraded result must not become sticky, so a later successful
    # Hub lookup can still override it.
    snaps = _repo(tmp_path, ("aaa", True))
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(snaps))
    _fake_hf_model_info(monkeypatch, _no_network)  # raises -> Hub "unreachable"
    assert mc.is_embedding_model("org/emb") is True  # marker fallback
    assert ("org/emb", None) not in mc._embedding_detection_cache  # not poisoned


def test_online_negative_does_not_block_later_offline_download(tmp_path, monkeypatch):
    # An online Hub lookup authoritatively reports non-embedding and is memoized.
    # The repo is then downloaded WITH modules.json and the session goes offline;
    # the offline path re-probes the marker (never consulting the online memo),
    # so the freshly downloaded embedder is detected instead of the stale False.
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(()))

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["text-generation"], pipeline_tag = "text-generation")

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/late-embedder") is False  # online: Hub says no
    assert mc._embedding_detection_cache[("org/late-embedder", None)] is False

    # Now the model is downloaded (marker appears) and the session goes offline.
    snaps = _repo(tmp_path, ("aaa", True))
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(snaps))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert mc.is_embedding_model("org/late-embedder") is True  # marker re-probed


def test_offline_retains_online_confirmed_positive(monkeypatch):
    # A tag-only embedder (feature-extraction, no modules.json) is confirmed
    # online and cached True. _hf_offline_if_dns_dead() then flips the process to
    # offline mid-load; the offline path must RETAIN that positive, not re-probe
    # the absent marker and downgrade a model already verified this session.
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(()))

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/gte-modernbert") is True  # online: cached True

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)  # offline must not hit network
    assert mc.is_embedding_model("org/gte-modernbert") is True  # positive retained


def test_offline_cached_non_st_returns_false_without_network(tmp_path, monkeypatch):
    snaps = _repo(tmp_path, ("aaa", False))
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(snaps))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("org/gemma-4-e4b") is False


def test_offline_not_cached_returns_false_without_network(monkeypatch):
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(()))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("org/never-downloaded") is False


def test_online_uncached_still_uses_network(monkeypatch):
    # Not offline, not cached: the network model_info path must still run so an
    # embedding model that lacks modules.json (feature-extraction tag) is caught.
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(()))
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
    monkeypatch.setattr(mc, "_iter_st_cache_snapshots", lambda repo: iter(()))
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
