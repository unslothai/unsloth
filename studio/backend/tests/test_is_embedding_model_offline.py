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
    # The old snapshot has modules.json but the active refs/main revision does not: the
    # active revision must win, so scanning any snapshot would wrongly say True.
    _repo(tmp_path, monkeypatch, ("new", False), ("old", True), main_ref = "new")
    assert mc._embedding_marker_in_hf_cache("org/model") is False


def test_marker_true_when_refs_main_revision_is_st(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, ("new", True), ("old", False), main_ref = "new")
    assert mc._embedding_marker_in_hf_cache("org/model") is True


def test_marker_missing_ref_is_a_cache_miss(tmp_path, monkeypatch):
    # local_files_only resolves the default revision THROUGH refs/main, so a snapshot dir
    # alone is not discoverable and accepting one would fail at first indexing.
    _repo(tmp_path, monkeypatch, ("new", False), ("old", True))  # no refs/main
    assert mc._embedding_marker_in_hf_cache("org/model") is None


def test_marker_ref_points_at_absent_snapshot_is_cache_miss(tmp_path, monkeypatch):
    # refs/main names a commit whose snapshot dir is absent (partial download /
    # pruning): a miss, never a fall-through to a stale historical snapshot.
    _repo(tmp_path, monkeypatch, ("old", True), main_ref = "missing_commit")
    assert mc._embedding_marker_in_hf_cache("org/model") is None


def test_marker_unreadable_ref_is_cache_miss(tmp_path, monkeypatch):
    # refs/main exists but cannot be read (transient I/O / restrictive
    # permissions): the loader cannot resolve it either.
    dirs = _repo(tmp_path, monkeypatch, ("old", True))
    refs_main = dirs[0].parent.parent / "refs" / "main"
    refs_main.parent.mkdir(parents = True, exist_ok = True)
    refs_main.mkdir()  # a directory -> read_text raises IsADirectoryError
    assert mc._embedding_marker_in_hf_cache("org/model") is None


def test_marker_empty_ref_is_cache_miss(tmp_path, monkeypatch):
    # refs/main exists but is empty / whitespace (a partial write): the active
    # revision is unknown.
    _repo(tmp_path, monkeypatch, ("old", True), main_ref = "   \n")
    assert mc._embedding_marker_in_hf_cache("org/model") is None


def test_marker_scopes_to_the_repo_dir_the_loader_opens(tmp_path, monkeypatch):
    # Two cache dirs differing only by case: the loader opens the one persisted (exact case
    # first), so a complete model there must validate even when the OTHER variant holds a
    # newer, non-ST snapshot -- judging across both would reject it.
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
    # With ST_HOME set that is the ONLY cache the load searches, so the probe must follow
    # it or a model present there is called uncached and 409'd offline.
    hf_root, st_root = tmp_path / "hf", tmp_path / "st"
    hf_root.mkdir()
    _st_snapshot(st_root, "models--org--model")
    _fake_hf_cache(monkeypatch, hf_root)
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(st_root))
    assert mc._st_cache_repo_dir("org/model") == st_root / "models--org--model"
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
    # The GGUF path uses the Hub cache (hf_hub_download, no cache_dir), so letting its probe
    # see ST_HOME would select a file the GGUF loader cannot find.
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
    # The security preflight downloads modules.json on its own, and a partial download
    # leaves it too, so accepting it offline would fail on the first RAG load.
    hf_root = tmp_path / "hf"
    _st_snapshot(hf_root, "models--org--model", loadable = False)
    _fake_hf_cache(monkeypatch, hf_root)
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising = False)
    assert mc._embedding_marker_in_hf_cache("org/model") is False


def test_marker_onnx_only_snapshot_is_not_loadable(tmp_path, monkeypatch):
    # Marker + config but ONLY an ONNX export is not loadable: the default Torch backend
    # reads model.safetensors / pytorch_model.bin, so accepting the ONNX offline would fail
    # at load.
    hf_root = tmp_path / "hf"
    repo = hf_root / "models--org--model"
    snap = repo / "snapshots" / "aaa"
    snap.mkdir(parents = True)
    (snap / "modules.json").write_text("[]")
    (snap / "config.json").write_text("{}")
    (snap / "tokenizer.json").write_text("{}")  # isolate the failure to the weight format
    (snap / "model.onnx").write_bytes(b"\0")
    # refs/main so the probe reaches the weight check (without it the answer is None).
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_text("aaa")
    _fake_hf_cache(monkeypatch, hf_root)
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising = False)
    assert mc._embedding_marker_in_hf_cache("org/model") is False


def test_marker_rejects_weights_without_tokenizer(tmp_path, monkeypatch):
    # Marker + config + weights but NO tokenizer asset is not loadable: the Transformer
    # module also builds an AutoTokenizer, which fails offline without a tokenizer asset.
    _cache_repo_with_files(tmp_path, monkeypatch, "model.safetensors", tokenizer = False)
    assert mc._embedding_marker_in_hf_cache("org/model") is False


@pytest.mark.parametrize("tok_file", ["tokenizer_config.json", "vocab.txt", "spiece.model"])
def test_marker_accepts_alternate_tokenizer_assets(tmp_path, monkeypatch, tok_file):
    # The tokenizer check is a permissive union: any one recognized asset is enough, so a
    # valid non-tokenizer.json layout is not wrongly rejected.
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
    # A NON-weight .bin (training_args.bin) or an adapter-only artifact is not a loadable
    # base model: the Torch backend needs model.safetensors / pytorch_model.bin.
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
    # A partially downloaded sharded model must NOT validate: the Torch backend needs every
    # shard AND the index map, so an incomplete set (or a set missing its index) fails.
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
    # The recognizer must not over-reject real weights: single pytorch_model.bin, a complete
    # sharded set with its index, and weights inside a module dir all count as loadable.
    _cache_repo_with_files(tmp_path, monkeypatch, *weights)
    assert mc._embedding_marker_in_hf_cache("org/model") is True


def _case_sensitive_fs(tmp_path) -> bool:
    probe = tmp_path / "_CaseProbe"
    probe.mkdir()
    return not (tmp_path / "_caseprobe").is_dir()


def test_st_casing_resolves_against_st_home(tmp_path, monkeypatch):
    # resolve_cached_repo_id_case scans only the Hub cache, so with ST_HOME set it would
    # persist the lower-case id while the exact-case offline load misses it in ST_HOME.
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
    # Offline: a downloaded sentence-transformers repo is classified from its modules.json
    # marker with no model_info() network call that would hang on DNS retries (#6817).
    _repo(tmp_path, monkeypatch, ("aaa", True), main_ref = "aaa", repo_id = "unsloth/bge-small-en-v1.5")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("unsloth/bge-small-en-v1.5") is True


def test_online_defers_to_hub_over_stale_marker(tmp_path, monkeypatch):
    # Online: the Hub is authoritative. Even with a cached modules.json, a Hub lookup that
    # no longer reports embedding signals wins -- the stale marker must not short-circuit it.
    _repo(tmp_path, monkeypatch, ("aaa", True), main_ref = "aaa")
    calls = []

    def _info(model_name, token = None):
        calls.append(model_name)
        return types.SimpleNamespace(tags = ["text-generation"], pipeline_tag = "text-generation")

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/was-embedder") is False
    assert calls == ["org/was-embedder"]  # Hub consulted, not skipped


def test_online_permanent_hub_error_ignores_stale_marker(tmp_path, monkeypatch):
    # A permanent Hub error (deleted / gated / typo'd repo) is authoritative: even with a
    # cached modules.json, return False so the settings route surfaces its 409.
    _repo(tmp_path, monkeypatch, ("aaa", True), main_ref = "aaa")

    class RepositoryNotFoundError(Exception):
        pass

    def _info(model_name, token = None):
        raise RepositoryNotFoundError("404 not found")

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/deleted-or-typo") is False


def test_online_hub_failure_falls_back_to_marker_uncached(tmp_path, monkeypatch):
    # A transient model_info() failure falls back to the local marker WITHOUT caching, so a
    # later successful Hub lookup can still override the degraded result.
    _repo(tmp_path, monkeypatch, ("aaa", True), main_ref = "aaa", repo_id = "org/emb")
    _fake_hf_model_info(monkeypatch, _no_network)  # raises -> Hub "unreachable"
    assert mc.is_embedding_model("org/emb") is True  # marker fallback
    assert ("org/emb", None) not in mc._embedding_detection_cache  # not poisoned


def test_online_negative_does_not_block_later_offline_download(tmp_path, monkeypatch):
    # An online lookup reports non-embedding and is memoized. The repo is then downloaded
    # with modules.json and the session goes offline; the offline path re-probes the marker
    # (never the memo), so the fresh embedder is detected instead of the stale False.
    _no_cache(monkeypatch)

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["text-generation"], pipeline_tag = "text-generation")

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/late-embedder") is False  # online: Hub says no
    assert mc._embedding_detection_cache[("org/late-embedder", None)] is False

    # Now the model is downloaded (marker appears) and the session goes offline.
    _repo(tmp_path, monkeypatch, ("aaa", True), main_ref = "aaa", repo_id = "org/late-embedder")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert mc.is_embedding_model("org/late-embedder") is True  # marker re-probed


def test_offline_retains_online_confirmed_positive(tmp_path, monkeypatch):
    # A tag-only embedder (no modules.json) confirmed online and cached True, with its
    # snapshot materialized. Flipped offline mid-load, the offline path must RETAIN the
    # positive: the marker reads False, so retention rests on the memo plus a present
    # snapshot, not the marker.
    _repo(tmp_path, monkeypatch, ("aaa", False), main_ref = "aaa", repo_id = "org/gte-modernbert")

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/gte-modernbert") is True  # online: cached True

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)  # offline must not hit network
    assert mc.is_embedding_model("org/gte-modernbert") is True  # positive retained


def test_offline_metadata_only_positive_not_trusted_without_cache(monkeypatch):
    # An online positive proves only that the repo is tagged an embedder, not that files
    # were downloaded. With nothing materialized, the offline path must NOT trust the memo
    # (the local_files_only load would fail), so it returns False despite the cached True.
    _no_cache(monkeypatch)

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)
    assert (
        mc.is_embedding_model("org/uncached-embedder") is True
    )  # online: cached True (metadata only)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)  # offline must not hit network
    assert mc.is_embedding_model("org/uncached-embedder") is False  # not cached -> not trusted


def test_offline_detects_persisted_tag_only_embedder_after_restart(tmp_path, monkeypatch):
    # A tag-only embedder confirmed online durably records the verdict, snapshot
    # materialized. After a RESTART (memo gone) the process comes up offline: the marker
    # reads False, so recognition rests on the persisted allowlist plus the present snapshot.
    _repo(tmp_path, monkeypatch, ("aaa", False), main_ref = "aaa", repo_id = "org/gte-modernbert")

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/gte-modernbert") is True  # online: confirmed + persisted
    assert "org/gte-modernbert" in mc._load_persisted_embedders()  # written to disk

    mc._embedding_detection_cache.clear()  # simulate a process restart: memo lost
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)  # offline must not hit network
    assert mc.is_embedding_model("org/gte-modernbert") is True  # recovered from disk


def test_offline_persisted_verdict_not_trusted_when_uncached(tmp_path, monkeypatch):
    # The persisted allowlist records only the tag, not on-disk files. After a restart with
    # NOTHING materialized, the offline path must NOT trust it (gated on the active snapshot
    # being present), so an uncached repo returns False.
    _no_cache(monkeypatch)

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/uncached-embedder") is True  # online: confirmed + persisted
    assert "org/uncached-embedder" in mc._load_persisted_embedders()

    mc._embedding_detection_cache.clear()  # restart: memo lost, disk verdict remains
    _no_cache(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("org/uncached-embedder") is False  # nothing on disk -> not trusted


def test_offline_persisted_verdict_not_trusted_when_snapshot_partial(tmp_path, monkeypatch):
    # A persisted verdict is trusted only when the active snapshot carries a COMPLETE weight
    # set, not merely that it is materialized. Here config but no weights (interrupted
    # download): a bare "materialized" gate would wrongly return True, so the weight gate
    # must reject it.
    cache_root = tmp_path / "cache"
    snap = cache_root / "models--org--partial" / "snapshots" / "aaa"
    snap.mkdir(parents = True)
    (snap / "config.json").write_text("{}")  # config only -- weights never finished
    refs = cache_root / "models--org--partial" / "refs"
    refs.mkdir(parents = True)
    (refs / "main").write_text("aaa")
    monkeypatch.setattr(mc, "_st_cache_roots", lambda: [cache_root])

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/partial") is True  # online: confirmed + persisted
    assert "org/partial" in mc._load_persisted_embedders()
    assert mc._embedding_marker_in_hf_cache("org/partial") is False  # materialized, not None

    mc._embedding_detection_cache.clear()  # restart: memo lost, disk verdict remains
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    assert mc.is_embedding_model("org/partial") is False  # partial snapshot -> not trusted


def test_offline_persisted_verdict_matches_across_casing(tmp_path, monkeypatch):
    # The verdict is recorded under the request spelling but the settings route saves the
    # cache-resolved one, so an exact-string lookup would miss it. Verified as baai/model,
    # looked up as the saved BAAI/model, must still match: the allowlist is case-folded.
    _repo(tmp_path, monkeypatch, ("aaa", False), main_ref = "aaa", repo_id = "BAAI/model")

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("baai/model") is True  # online: confirmed as baai/model

    mc._embedding_detection_cache.clear()  # restart: memo lost, disk verdict remains
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _fake_hf_model_info(monkeypatch, _no_network)
    # Looked up under the cache-resolved casing the settings route persisted.
    assert mc.is_embedding_model("BAAI/model") is True


def test_persist_embedder_concurrent_writes_keep_every_verdict(tmp_path, monkeypatch):
    # Concurrent confirmations must not drop each other's entry: serialized read-modify-write
    # + per-thread temp files, so every model persisted from parallel threads survives.
    import threading

    names = [f"org/emb-{i}" for i in range(24)]
    barrier = threading.Barrier(len(names))

    def _writer(name):
        barrier.wait()  # maximize overlap on the shared file
        mc._persist_embedder(name)

    threads = [threading.Thread(target = _writer, args = (n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    persisted = mc._load_persisted_embedders()
    assert {n.casefold() for n in names} <= persisted


def test_persist_embedder_is_best_effort_when_home_unwritable(tmp_path, monkeypatch):
    # Persistence is an optimization: if the Studio home cannot be written, the online path
    # must still return its verdict rather than raise. Point the home at a path blocked by a file.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(blocker / "studio"))  # parent is a file
    _no_cache(monkeypatch)

    def _info(model_name, token = None):
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/emb") is True  # no exception despite unwritable home
    assert mc._load_persisted_embedders() == set()  # nothing recorded, silently


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
    # Not offline, not cached: the model_info path must still run so a feature-extraction
    # embedder lacking modules.json is caught.
    _no_cache(monkeypatch)
    calls = []

    def _info(model_name, token = None):
        calls.append(model_name)
        return types.SimpleNamespace(tags = ["feature-extraction"], pipeline_tag = None)

    _fake_hf_model_info(monkeypatch, _info)
    assert mc.is_embedding_model("org/gte-modernbert") is True
    assert calls == ["org/gte-modernbert"]


def test_offline_negative_is_not_cached_then_online_detects(monkeypatch):
    # A tag-only embedder is not identifiable from modules.json. Offline returns False
    # WITHOUT caching, so once the env clears the online lookup still detects it.
    _no_cache(monkeypatch)
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
