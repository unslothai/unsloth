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

import utils.models.model_config as mc  # noqa: E402


@pytest.fixture(autouse = True)
def _clean_state(tmp_path, monkeypatch):
    mc._embedding_detection_cache.clear()
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    # Per-test Studio home so the persisted allowlist never leaks into another test.
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "_studio_home"))
    yield
    mc._embedding_detection_cache.clear()


def _repo(
    tmp_path,
    monkeypatch,
    *snapshots,
    main_ref = None,
    repo_id = "org/model",
    root = None,
):
    """Build a real cache repo dir and point the ST cache root at it. ``snapshots``:
    (commit, sentence_transformer) tuples, each written fully loadable so only the marker
    varies. ``main_ref`` writes refs/main. Returns the snapshot dirs in order.
    """
    cache_root = root if root is not None else tmp_path / "cache"
    cache_root.mkdir(parents = True, exist_ok = True)
    repo = cache_root / f"models--{repo_id.replace('/', '--')}"
    dirs = []
    for name, is_st in snapshots:
        d = repo / "snapshots" / name
        d.mkdir(parents = True, exist_ok = True)
        (d / "config.json").write_text("{}")
        (d / "tokenizer.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"\0")
        if is_st:
            (d / "modules.json").write_text("[]")
        dirs.append(d)
    if main_ref is not None:
        refs = repo / "refs"
        refs.mkdir(parents = True, exist_ok = True)
        (refs / "main").write_text(main_ref)
    monkeypatch.setattr(mc, "_st_cache_roots", lambda: [cache_root])
    return dirs


def _no_cache(monkeypatch):
    monkeypatch.setattr(mc, "_st_cache_roots", lambda: [])


def _fake_hf_model_info(monkeypatch, fn):
    fake = types.ModuleType("huggingface_hub")
    fake.model_info = fn
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


def _no_network(*a, **k):
    raise AssertionError("model_info() must not be called")


# ── _embedding_marker_in_hf_cache ──


def test_marker_true_for_a_loadable_st_revision(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, ("aaa", True), main_ref = "aaa")
    assert mc._embedding_marker_in_hf_cache("org/model") is True


def test_marker_false_when_active_revision_is_not_st(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, ("aaa", False), main_ref = "aaa")
    assert mc._embedding_marker_in_hf_cache("org/model") is False


def test_marker_none_when_not_cached(monkeypatch):
    _no_cache(monkeypatch)
    assert mc._embedding_marker_in_hf_cache("org/model") is None


def test_marker_judges_the_revision_refs_main_points_at(tmp_path, monkeypatch):
    # The active refs/main revision wins over an older snapshot that still has modules.json.
    _repo(tmp_path, monkeypatch, ("new", False), ("old", True), main_ref = "new")
    assert mc._embedding_marker_in_hf_cache("org/model") is False


def test_marker_true_when_refs_main_revision_is_st(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, ("new", True), ("old", False), main_ref = "new")
    assert mc._embedding_marker_in_hf_cache("org/model") is True


def test_marker_missing_ref_is_a_cache_miss(tmp_path, monkeypatch):
    # local_files_only resolves the default revision through refs/main; a bare snapshot dir
    # is not discoverable.
    _repo(tmp_path, monkeypatch, ("new", False), ("old", True))
    assert mc._embedding_marker_in_hf_cache("org/model") is None


def test_marker_ref_points_at_absent_snapshot_is_cache_miss(tmp_path, monkeypatch):
    # refs/main naming an absent snapshot is a miss, not a fall-through to a stale one.
    _repo(tmp_path, monkeypatch, ("old", True), main_ref = "missing_commit")
    assert mc._embedding_marker_in_hf_cache("org/model") is None


def test_marker_unreadable_ref_is_cache_miss(tmp_path, monkeypatch):
    # refs/main present but unreadable: the loader cannot resolve it either.
    dirs = _repo(tmp_path, monkeypatch, ("old", True))
    refs_main = dirs[0].parent.parent / "refs" / "main"
    refs_main.parent.mkdir(parents = True, exist_ok = True)
    refs_main.mkdir()  # a directory -> read_text raises
    assert mc._embedding_marker_in_hf_cache("org/model") is None


def test_marker_empty_ref_is_cache_miss(tmp_path, monkeypatch):
    # empty / whitespace refs/main (partial write): the active revision is unknown.
    _repo(tmp_path, monkeypatch, ("old", True), main_ref = "   \n")
    assert mc._embedding_marker_in_hf_cache("org/model") is None


def test_marker_scopes_to_the_repo_dir_the_loader_opens(tmp_path, monkeypatch):
    # The loader opens the exact-case dir, so a complete model there validates even when the
    # other case variant holds a newer non-ST snapshot.
    if not _case_sensitive_fs(tmp_path):
        pytest.skip("duplicate case variants need a case-sensitive filesystem")
    root = tmp_path / "cache"
    _repo(tmp_path, monkeypatch, ("aaa", True), main_ref = "aaa", repo_id = "baai/bge-m3", root = root)
    # A newer variant that is NOT a sentence-transformers model.
    _repo(tmp_path, monkeypatch, ("zzz", False), main_ref = "zzz", repo_id = "BAAI/bge-m3", root = root)
    assert mc._embedding_marker_in_hf_cache("baai/bge-m3") is True


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
        (snap / "tokenizer.json").write_text("{}")
        (snap / "model.safetensors").write_bytes(b"\0")
    refs = root / repo_dir / "refs"
    refs.mkdir(parents = True, exist_ok = True)
    (refs / "main").write_text(commit)
    return snap


def test_st_probe_uses_sentence_transformers_home(tmp_path, monkeypatch):
    # ST_HOME is the only cache the load searches, so the probe must follow it.
    hf_root, st_root = tmp_path / "hf", tmp_path / "st"
    hf_root.mkdir()
    _st_snapshot(st_root, "models--org--model")
    _fake_hf_cache(monkeypatch, hf_root)
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(st_root))
    assert mc._st_cache_repo_dir("org/model") == st_root / "models--org--model"
    assert mc._embedding_marker_in_hf_cache("org/model") is True


def test_st_probe_ignores_hub_cache_when_st_home_is_set(tmp_path, monkeypatch):
    # Loader searches ST_HOME only; a repo cached only in the Hub cache must not validate.
    hf_root, st_root = tmp_path / "hf", tmp_path / "st"
    st_root.mkdir()
    _st_snapshot(hf_root, "models--org--model")
    _fake_hf_cache(monkeypatch, hf_root)
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(st_root))
    assert mc._embedding_marker_in_hf_cache("org/model") is None


def test_gguf_probe_never_follows_st_home(tmp_path, monkeypatch):
    # The GGUF path uses the Hub cache (hf_hub_download), so its probe must not follow ST_HOME.
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
    # The security preflight (and a partial download) leaves a bare modules.json.
    hf_root = tmp_path / "hf"
    _st_snapshot(hf_root, "models--org--model", loadable = False)
    _fake_hf_cache(monkeypatch, hf_root)
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising = False)
    assert mc._embedding_marker_in_hf_cache("org/model") is False


def test_marker_onnx_only_snapshot_is_not_loadable(tmp_path, monkeypatch):
    # Marker + config but only an ONNX export: the Torch backend needs safetensors/bin.
    hf_root = tmp_path / "hf"
    repo = hf_root / "models--org--model"
    snap = repo / "snapshots" / "aaa"
    snap.mkdir(parents = True)
    (snap / "modules.json").write_text("[]")
    (snap / "config.json").write_text("{}")
    (snap / "tokenizer.json").write_text("{}")
    (snap / "model.onnx").write_bytes(b"\0")
    # refs/main so the probe reaches the weight check.
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_text("aaa")
    _fake_hf_cache(monkeypatch, hf_root)
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising = False)
    assert mc._embedding_marker_in_hf_cache("org/model") is False


def test_marker_rejects_weights_without_tokenizer(tmp_path, monkeypatch):
    # Weights but no tokenizer asset: the Transformer module also builds an AutoTokenizer.
    _cache_repo_with_files(tmp_path, monkeypatch, "model.safetensors", tokenizer = False)
    assert mc._embedding_marker_in_hf_cache("org/model") is False


@pytest.mark.parametrize("tok_file", ["tokenizer_config.json", "vocab.txt", "spiece.model"])
def test_marker_accepts_alternate_tokenizer_assets(tmp_path, monkeypatch, tok_file):
    # Permissive union: any one recognized tokenizer asset suffices.
    _cache_repo_with_files(tmp_path, monkeypatch, "model.safetensors", tok_file, tokenizer = False)
    assert mc._embedding_marker_in_hf_cache("org/model") is True


def _cache_repo_with_files(
    tmp_path,
    monkeypatch,
    *files,
    commit = "aaa",
    tokenizer = True,
):
    """Cache repo whose active snapshot holds modules.json + config + a tokenizer + *files*.
    ``tokenizer=False`` omits the tokenizer asset."""
    hf_root = tmp_path / "hf"
    repo = hf_root / "models--org--model"
    snap = repo / "snapshots" / commit
    snap.mkdir(parents = True)
    (snap / "modules.json").write_text("[]")
    (snap / "config.json").write_text("{}")
    if tokenizer:
        (snap / "tokenizer.json").write_text("{}")
    for name in files:
        target = snap / name
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"\0")
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_text(commit)
    _fake_hf_cache(monkeypatch, hf_root)
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising = False)


def test_marker_rejects_non_base_weight_bins(tmp_path, monkeypatch):
    # A non-weight .bin or adapter-only artifact is not a loadable base model.
    _cache_repo_with_files(tmp_path, monkeypatch, "training_args.bin")
    assert mc._embedding_marker_in_hf_cache("org/model") is False
    _cache_repo_with_files(tmp_path / "b", monkeypatch, "adapter_model.safetensors")
    assert mc._embedding_marker_in_hf_cache("org/model") is False


@pytest.mark.parametrize(
    "weights",
    [
        ("model-00001-of-00002.safetensors", "model.safetensors.index.json"),  # shard 1 of 2
        ("model-00002-of-00003.bin", "pytorch_model.bin.index.json"),  # a lone middle shard
        (  # two shards + index but the set claims three
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model.safetensors.index.json",
        ),
        (  # a complete shard set WITHOUT the index map transformers loads it through
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ),
    ],
)
def test_marker_rejects_incomplete_shard_set(tmp_path, monkeypatch, weights):
    # An incomplete shard set (or one missing its index map) must not validate.
    _cache_repo_with_files(tmp_path, monkeypatch, *weights)
    assert mc._embedding_marker_in_hf_cache("org/model") is False


@pytest.mark.parametrize(
    "weights",
    [
        ("pytorch_model.bin",),  # torch .bin
        (  # sharded: every index plus the index map
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
        ),
        ("0_Transformer/model.safetensors",),  # weight in a module dir
    ],
)
def test_marker_accepts_recognized_torch_weights(tmp_path, monkeypatch, weights):
    # Must not over-reject: single bin, complete sharded set, and a weight in a module dir.
    _cache_repo_with_files(tmp_path, monkeypatch, *weights)
    assert mc._embedding_marker_in_hf_cache("org/model") is True


def _case_sensitive_fs(tmp_path) -> bool:
    probe = tmp_path / "_CaseProbe"
    probe.mkdir()
    return not (tmp_path / "_caseprobe").is_dir()


def test_st_casing_resolves_against_st_home(tmp_path, monkeypatch):
    # With ST_HOME set, casing must resolve against it, or the exact-case offline load misses.
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
    # A snapshot vanishing mid-iteration must read as not-cached, not raise a 500.
    def _exploding_iter(repo):
        raise FileNotFoundError("snapshot removed underneath")

    monkeypatch.setattr(mc, "_st_cache_repo_dir", _exploding_iter)
    assert mc._embedding_marker_in_hf_cache("org/racing") is None


def test_is_embedding_model_survives_cache_race_online(monkeypatch):
    # With the cache probe failing, the online path must still resolve via the Hub.
    def _exploding_iter(repo):
        raise FileNotFoundError("snapshot removed underneath")

    monkeypatch.setattr(mc, "_st_cache_repo_dir", _exploding_iter)
    _fake_hf_model_info(
        monkeypatch,
        lambda name, token = None: types.SimpleNamespace(
            tags = ["sentence-transformers"], pipeline_tag = None
        ),
    )
    assert mc.is_embedding_model("org/racing") is True


# ── is_embedding_model ──


def test_offline_cached_st_detected_via_marker_no_network(tmp_path, monkeypatch):
    # Offline: classified from the cached modules.json marker, no model_info() call that
    # would hang on DNS (#6817).
    _repo(tmp_path, monkeypatch, ("aaa", True), main_ref = "aaa", repo_id = "unsloth/bge-small-en-v1.5")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("unsloth/bge-small-en-v1.5") is True


def test_online_defers_to_hub_over_stale_marker(tmp_path, monkeypatch):
    # Online: the Hub is authoritative; a stale cached marker must not short-circuit a lookup
    # that no longer reports embedding signals.
    _repo(tmp_path, monkeypatch, ("aaa", True), main_ref = "aaa")
    calls = []

    def _info(model_name, token = None):
        calls.append(model_name)
        return types.SimpleNamespace(tags = ["text-generation"], pipeline_tag = "text-generation")

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/was-embedder") is False
    assert calls == ["org/was-embedder"]


def test_online_permanent_hub_error_ignores_stale_marker(tmp_path, monkeypatch):
    # A permanent Hub error is authoritative: return False even with a cached marker, so the
    # settings route surfaces its 409.
    _repo(tmp_path, monkeypatch, ("aaa", True), main_ref = "aaa")

    class RepositoryNotFoundError(Exception):
        pass

    def _info(model_name, token = None):
        raise RepositoryNotFoundError("404 not found")

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/deleted-or-typo") is False


def test_online_hub_failure_falls_back_to_marker_uncached(tmp_path, monkeypatch):
    # A transient failure falls back to the local marker WITHOUT caching, so a later Hub
    # lookup can override.
    _repo(tmp_path, monkeypatch, ("aaa", True), main_ref = "aaa", repo_id = "org/emb")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("org/emb") is True
    assert ("org/emb", None) not in mc._embedding_detection_cache


def test_online_negative_does_not_block_later_offline_download(tmp_path, monkeypatch):
    # A memoized online negative must not block a later offline detection: the offline path
    # re-probes the marker, never the memo.
    _no_cache(monkeypatch)

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["text-generation"], pipeline_tag = "text-generation")

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/late-embedder") is False
    assert mc._embedding_detection_cache[("org/late-embedder", None)] is False

    # Model now downloaded (marker appears); session goes offline.
    _repo(tmp_path, monkeypatch, ("aaa", True), main_ref = "aaa", repo_id = "org/late-embedder")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert mc.is_embedding_model("org/late-embedder") is True


def test_offline_retains_online_confirmed_positive(tmp_path, monkeypatch):
    # A tag-only embedder (no modules.json) cached True online, snapshot materialized. Offline
    # must retain the positive via the memo + present snapshot, since the marker reads False.
    _repo(tmp_path, monkeypatch, ("aaa", False), main_ref = "aaa", repo_id = "org/gte-modernbert")

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/gte-modernbert") is True

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("org/gte-modernbert") is True


def test_offline_metadata_only_positive_not_trusted_without_cache(monkeypatch):
    # An online positive only tags the repo; with nothing materialized the offline load would
    # fail, so the memo must NOT be trusted.
    _no_cache(monkeypatch)

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/uncached-embedder") is True

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("org/uncached-embedder") is False


def test_offline_detects_persisted_tag_only_embedder_after_restart(tmp_path, monkeypatch):
    # Tag-only embedder persisted online, snapshot materialized. After a restart (memo gone)
    # offline recognition rests on the persisted allowlist + present snapshot.
    _repo(tmp_path, monkeypatch, ("aaa", False), main_ref = "aaa", repo_id = "org/gte-modernbert")

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/gte-modernbert") is True
    assert "org/gte-modernbert" in mc._load_persisted_embedders()

    mc._embedding_detection_cache.clear()  # simulate a restart: memo lost
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("org/gte-modernbert") is True


def test_offline_persisted_verdict_not_trusted_when_uncached(tmp_path, monkeypatch):
    # The persisted allowlist records only the tag; after a restart with nothing materialized
    # it must NOT be trusted.
    _no_cache(monkeypatch)

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/uncached-embedder") is True
    assert "org/uncached-embedder" in mc._load_persisted_embedders()

    mc._embedding_detection_cache.clear()  # restart: memo lost, disk verdict remains
    _no_cache(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("org/uncached-embedder") is False


def test_offline_persisted_verdict_not_trusted_when_snapshot_partial(tmp_path, monkeypatch):
    # A persisted verdict is trusted only with a COMPLETE weight set, not mere materialization.
    # Here config but no weights (interrupted download) must be rejected.
    cache_root = tmp_path / "cache"
    snap = cache_root / "models--org--partial" / "snapshots" / "aaa"
    snap.mkdir(parents = True)
    (snap / "config.json").write_text("{}")
    refs = cache_root / "models--org--partial" / "refs"
    refs.mkdir(parents = True)
    (refs / "main").write_text("aaa")
    monkeypatch.setattr(mc, "_st_cache_roots", lambda: [cache_root])

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/partial") is True
    assert "org/partial" in mc._load_persisted_embedders()
    assert mc._embedding_marker_in_hf_cache("org/partial") is False  # materialized, not None

    mc._embedding_detection_cache.clear()  # restart: memo lost, disk verdict remains
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("org/partial") is False


def test_offline_persisted_verdict_matches_across_casing(tmp_path, monkeypatch):
    # The verdict is saved under the cache-resolved casing; the case-folded allowlist must
    # still match a differently-cased lookup.
    _repo(tmp_path, monkeypatch, ("aaa", False), main_ref = "aaa", repo_id = "BAAI/model")

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("baai/model") is True

    mc._embedding_detection_cache.clear()  # restart: memo lost, disk verdict remains
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("BAAI/model") is True


def test_persist_embedder_concurrent_writes_keep_every_verdict(tmp_path, monkeypatch):
    # Concurrent confirmations must not drop entries: serialized writes + per-thread temp files.
    import threading

    names = [f"org/emb-{i}" for i in range(24)]
    barrier = threading.Barrier(len(names))

    def _writer(name):
        barrier.wait()
        mc._persist_embedder(name)

    threads = [threading.Thread(target = _writer, args = (n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    persisted = mc._load_persisted_embedders()
    assert {n.casefold() for n in names} <= persisted


def test_persist_embedder_is_best_effort_when_home_unwritable(tmp_path, monkeypatch):
    # Persistence is best-effort: an unwritable Studio home must not raise; the online verdict
    # still returns.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(blocker / "studio"))
    _no_cache(monkeypatch)

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/emb") is True
    assert mc._load_persisted_embedders() == set()


def test_offline_cached_non_st_returns_false_without_network(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, ("aaa", False), main_ref = "aaa")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("org/gemma-4-e4b") is False


def test_offline_not_cached_returns_false_without_network(monkeypatch):
    _no_cache(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("org/never-downloaded") is False


def test_online_uncached_still_uses_network(monkeypatch):
    # Not offline, not cached: model_info still runs so a tag-only embedder is caught.
    _no_cache(monkeypatch)
    calls = []

    def _info(model_name, token = None):
        calls.append(model_name)
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/gte-modernbert") is True
    assert calls == ["org/gte-modernbert"]


def test_offline_negative_is_not_cached_then_online_detects(monkeypatch):
    # A tag-only embedder isn't identifiable from modules.json; offline returns False WITHOUT
    # caching so online can still detect it later.
    _no_cache(monkeypatch)
    calls = []

    def _info(model_name, token = None):
        calls.append(model_name)
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert mc.is_embedding_model("org/gte-modernbert") is False
    assert calls == []
    assert ("org/gte-modernbert", None) not in mc._embedding_detection_cache

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    assert mc.is_embedding_model("org/gte-modernbert") is True
    assert calls == ["org/gte-modernbert"]
