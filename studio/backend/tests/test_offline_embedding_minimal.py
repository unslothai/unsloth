# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Offline RAG embedding-model handling (issue #6817).

Offline, the studio must never call the Hub: a DNS-dead session hangs on model_info /
download retries. These tests build a fake HF cache under a temp ``HF_HUB_CACHE`` and
assert that, offline:
  * ``is_embedding_model`` classifies from the cached ``modules.json`` and never calls
    the Hub (``model_info`` patched to raise if reached);
  * the file-security gate fails CLOSED on an unscanned pickle weight with no safetensors
    alternative, and allows an inert (safetensors/gguf) cache;
  * the embedder threads ``local_files_only`` into the SentenceTransformer load.
Online behavior is unchanged: a bounded ``model_info`` timeout with a local-cache fallback.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from utils.security import evaluate_file_security
from utils.utils import hf_cache_snapshot_dir, hf_env_offline, st_repo_id_candidates

# A minimal sentence-transformers modules.json (the marker the local-path check keys on).
MODULES_JSON = (
    '[{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"}]'
)


def _modules_json(*paths):
    """A modules.json listing one Transformer module per ``path`` (a load root)."""
    import json

    return json.dumps(
        [
            {"idx": i, "name": str(i), "path": p, "type": "sentence_transformers.models.Transformer"}
            for i, p in enumerate(paths)
        ]
    )

_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _make_cache(
    root,
    repo_id,
    files,
    commit = _COMMIT,
):
    """Build a canonical HF-cache snapshot (``refs/main`` + ``snapshots/<commit>/``) for
    ``repo_id`` under ``root`` with ``{relpath: contents}``. Returns the snapshot dir."""
    from huggingface_hub.file_download import repo_folder_name

    repo_dir = Path(root) / repo_folder_name(repo_id = repo_id, repo_type = "model")
    (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs" / "main").write_text(commit)
    snapshot = repo_dir / "snapshots" / commit
    snapshot.mkdir(parents = True, exist_ok = True)
    for rel, contents in files.items():
        path = snapshot / rel
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(contents)
    return snapshot


def _no_network():
    """Patch the Hub metadata call to fail loudly if any offline path reaches it."""
    return patch("huggingface_hub.model_info", side_effect = AssertionError("hit the network"))


def _is_embedding_model(*args, **kwargs):
    from utils.models.model_config import is_embedding_model
    return is_embedding_model(*args, **kwargs)


@pytest.fixture
def hf_cache(tmp_path, monkeypatch):
    """Point the HF cache at a fresh temp dir the resolver reads at call time."""
    root = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    return root


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    """Each test starts online with an empty detection cache; offline tests opt in."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    from utils.models import model_config as mc

    mc._embedding_detection_cache.clear()
    yield
    mc._embedding_detection_cache.clear()


# ── hf_env_offline ───────────────────────────────────────────────


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "  On  "])
def test_hf_env_offline_true(monkeypatch, value):
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    assert hf_env_offline() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_hf_env_offline_false(monkeypatch, value):
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    assert hf_env_offline() is False


def test_hf_env_offline_honors_transformers_flag(monkeypatch):
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert hf_env_offline() is True


def test_hf_env_offline_default_false():
    assert hf_env_offline() is False


# ── st_repo_id_candidates ────────────────────────────────────────


def test_candidates_slashless_adds_st_alias():
    assert st_repo_id_candidates("all-MiniLM-L6-v2") == [
        "all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]


def test_candidates_with_org_is_verbatim():
    assert st_repo_id_candidates("org/model") == ["org/model"]


def test_candidates_empty_name():
    assert st_repo_id_candidates("   ") == []


# ── hf_cache_snapshot_dir ────────────────────────────────────────


def test_snapshot_dir_resolves_active_commit(hf_cache):
    snapshot = _make_cache(hf_cache, "org/emb", {"modules.json": MODULES_JSON})
    assert hf_cache_snapshot_dir("org/emb") == snapshot


def test_snapshot_dir_none_when_uncached(hf_cache):
    assert hf_cache_snapshot_dir("org/missing") is None


def test_snapshot_dir_uses_st_alias_for_slashless(hf_cache):
    snapshot = _make_cache(
        hf_cache, "sentence-transformers/all-MiniLM-L6-v2", {"modules.json": MODULES_JSON}
    )
    assert hf_cache_snapshot_dir("all-MiniLM-L6-v2") == snapshot


def test_snapshot_dir_none_when_snapshot_missing(hf_cache):
    from huggingface_hub.file_download import repo_folder_name

    repo_dir = hf_cache / repo_folder_name(repo_id = "org/broken", repo_type = "model")
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text("deadbeef")  # no snapshots/deadbeef dir
    assert hf_cache_snapshot_dir("org/broken") is None


def test_snapshot_dir_expands_env_vars_in_cache_path(tmp_path, monkeypatch):
    # HF_HUB_CACHE with an unexpanded $VAR must resolve to the same place the loader uses.
    real = tmp_path / "hub"
    real.mkdir()
    monkeypatch.setenv("MY_HF_CACHE", str(real))
    monkeypatch.setenv("HF_HUB_CACHE", "$MY_HF_CACHE")
    monkeypatch.delenv("HF_HOME", raising = False)
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising = False)
    snapshot = _make_cache(real, "org/emb", {"modules.json": MODULES_JSON})
    assert hf_cache_snapshot_dir("org/emb") == snapshot


def test_snapshot_dir_uses_sentence_transformers_home(tmp_path, monkeypatch):
    # SentenceTransformer uses SENTENCE_TRANSFORMERS_HOME as its cache_folder, so the gate must
    # inspect it too.
    st_home = tmp_path / "st_home"
    st_home.mkdir()
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(st_home))
    monkeypatch.delenv("HF_HUB_CACHE", raising = False)
    monkeypatch.delenv("HF_HOME", raising = False)
    snapshot = _make_cache(st_home, "org/emb", {"modules.json": MODULES_JSON})
    assert hf_cache_snapshot_dir("org/emb") == snapshot


def test_gate_blocks_pickle_in_sentence_transformers_home(tmp_path, monkeypatch):
    # A pickle cached under SENTENCE_TRANSFORMERS_HOME must still fail closed offline.
    st_home = tmp_path / "st_home"
    st_home.mkdir()
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(st_home))
    monkeypatch.delenv("HF_HUB_CACHE", raising = False)
    monkeypatch.delenv("HF_HOME", raising = False)
    _make_cache(st_home, "org/pk", {"config.json": "{}", "pytorch_model.bin": "x"})
    with _no_network():
        assert evaluate_file_security("org/pk", local_only_load = True).blocked is True


# ── is_embedding_model: offline (no network) ─────────────────────


def test_offline_true_for_cached_st_model(hf_cache, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _make_cache(hf_cache, "org/emb", {"modules.json": MODULES_JSON, "config.json": "{}"})
    with _no_network():
        assert _is_embedding_model("org/emb") is True


def test_offline_false_for_cached_non_st_model(hf_cache, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _make_cache(hf_cache, "org/plain", {"config.json": "{}", "model.safetensors": "x"})
    with _no_network():
        assert _is_embedding_model("org/plain") is False


def test_offline_false_when_uncached(hf_cache, monkeypatch):
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    with _no_network():
        assert _is_embedding_model("org/missing") is False


def test_offline_slashless_resolves_via_alias(hf_cache, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _make_cache(hf_cache, "sentence-transformers/all-MiniLM-L6-v2", {"modules.json": MODULES_JSON})
    with _no_network():
        assert _is_embedding_model("all-MiniLM-L6-v2") is True


# ── is_embedding_model: online (bounded + fallback) ──────────────


def test_online_passes_bounded_timeout(hf_cache):
    seen = {}

    def _mi(
        name,
        token = None,
        timeout = None,
        **kw,
    ):
        seen["timeout"] = timeout
        return SimpleNamespace(tags = ["sentence-transformers"], pipeline_tag = None)

    with patch("huggingface_hub.model_info", side_effect = _mi):
        assert _is_embedding_model("org/emb") is True
    assert seen["timeout"] == 15.0


def test_online_error_falls_back_to_cache_marker(hf_cache):
    _make_cache(hf_cache, "org/emb", {"modules.json": MODULES_JSON})
    with patch("huggingface_hub.model_info", side_effect = RuntimeError("dns dead")):
        assert _is_embedding_model("org/emb") is True


def test_online_error_without_cache_returns_false(hf_cache):
    with patch("huggingface_hub.model_info", side_effect = RuntimeError("dns dead")):
        assert _is_embedding_model("org/missing") is False


# ── evaluate_file_security: offline fail-closed gate ─────────────


def _offline_decision(name):
    return evaluate_file_security(name, local_only_load = True)


def test_gate_allows_safetensors_only(hf_cache):
    _make_cache(hf_cache, "org/st", {"modules.json": MODULES_JSON, "model.safetensors": "x"})
    with _no_network():
        assert _offline_decision("org/st").blocked is False


def test_gate_blocks_pickle_without_safetensors(hf_cache):
    _make_cache(hf_cache, "org/pk", {"config.json": "{}", "pytorch_model.bin": "x"})
    with _no_network():
        decision = _offline_decision("org/pk")
    assert decision.blocked is True
    assert any(u["path"] == "pytorch_model.bin" for u in decision.unsafe_files)


def test_gate_allows_pickle_with_safetensors_sibling(hf_cache):
    _make_cache(hf_cache, "org/both", {"pytorch_model.bin": "x", "model.safetensors": "y"})
    with _no_network():
        assert _offline_decision("org/both").blocked is False


def test_gate_blocks_sharded_pickle(hf_cache):
    _make_cache(
        hf_cache,
        "org/shard",
        {
            "pytorch_model-00001-of-00002.bin": "a",
            "pytorch_model-00002-of-00002.bin": "b",
        },
    )
    with _no_network():
        assert _offline_decision("org/shard").blocked is True


def test_gate_allows_nothing_cached(hf_cache):
    with _no_network():
        assert _offline_decision("org/missing").blocked is False


def test_gate_allows_gguf_only(hf_cache):
    _make_cache(hf_cache, "org/gg", {"model.gguf": "x"})
    with _no_network():
        assert _offline_decision("org/gg").blocked is False


def test_gate_blocks_pickle_in_module_subdir(hf_cache):
    # 0_Transformer is a module load root (listed in modules.json), so its pickle blocks.
    _make_cache(
        hf_cache,
        "org/mod",
        {"modules.json": _modules_json("0_Transformer"), "0_Transformer/pytorch_model.bin": "x"},
    )
    with _no_network():
        assert _offline_decision("org/mod").blocked is True


def test_gate_allows_pickle_in_subdir_with_safetensors(hf_cache):
    _make_cache(
        hf_cache,
        "org/mod2",
        {
            "modules.json": _modules_json("0_Transformer"),
            "0_Transformer/pytorch_model.bin": "x",
            "0_Transformer/model.safetensors": "y",
        },
    )
    with _no_network():
        assert _offline_decision("org/mod2").blocked is False


def test_gate_allows_unreferenced_nested_pickle(hf_cache):
    # A pickle in a dir NOT referenced by modules.json (e.g. nemo/) is never deserialized by
    # SentenceTransformer, so it must not block the offline load (matches the online gate).
    _make_cache(
        hf_cache,
        "org/aux",
        {
            "modules.json": MODULES_JSON,  # Transformer at the root only
            "model.safetensors": "w",
            "nemo/pytorch_model.bin": "x",
        },
    )
    with _no_network():
        assert _offline_decision("org/aux").blocked is False


def test_gate_blocks_adapter_pickle_without_safetensors(hf_cache):
    _make_cache(hf_cache, "org/ad", {"config.json": "{}", "adapter_model.bin": "x"})
    with _no_network():
        decision = _offline_decision("org/ad")
    assert decision.blocked is True
    assert any(u["path"] == "adapter_model.bin" for u in decision.unsafe_files)


def test_gate_allows_adapter_pickle_with_adapter_safetensors(hf_cache):
    _make_cache(hf_cache, "org/ad2", {"adapter_model.bin": "x", "adapter_model.safetensors": "y"})
    with _no_network():
        assert _offline_decision("org/ad2").blocked is False


def test_gate_blocks_base_pickle_with_only_adapter_safetensors_decoy(hf_cache):
    # A decoy adapter_model.safetensors must NOT suppress a base pytorch_model.bin: the base
    # loader would still deserialize the unscanned pickle.
    _make_cache(hf_cache, "org/decoy", {"pytorch_model.bin": "x", "adapter_model.safetensors": "y"})
    with _no_network():
        assert _offline_decision("org/decoy").blocked is True


def test_gate_blocks_adapter_pickle_with_only_base_safetensors_decoy(hf_cache):
    # Symmetric: a base model.safetensors must NOT suppress an adapter_model.bin.
    _make_cache(hf_cache, "org/decoy2", {"adapter_model.bin": "x", "model.safetensors": "y"})
    with _no_network():
        assert _offline_decision("org/decoy2").blocked is True


def test_gate_reports_snapshot_relative_path(hf_cache):
    _make_cache(
        hf_cache,
        "org/mod3",
        {"modules.json": _modules_json("0_Transformer"), "0_Transformer/pytorch_model.bin": "x"},
    )
    with _no_network():
        decision = _offline_decision("org/mod3")
    assert decision.blocked is True
    assert any(u["path"] == "0_Transformer/pytorch_model.bin" for u in decision.unsafe_files)


# ── evaluate_file_security: online path unchanged ────────────────


def test_online_default_blocks_unsafe():
    status = {
        "scansDone": True,
        "filesWithIssues": [{"path": "pytorch_model.bin", "level": "unsafe"}],
    }
    with patch(
        "huggingface_hub.model_info",
        side_effect = lambda *a, **k: SimpleNamespace(security_repo_status = status),
    ):
        assert evaluate_file_security("org/x").blocked is True


def test_online_default_allows_clean():
    status = {"scansDone": True, "filesWithIssues": []}
    with patch(
        "huggingface_hub.model_info",
        side_effect = lambda *a, **k: SimpleNamespace(security_repo_status = status),
    ):
        assert evaluate_file_security("org/x").blocked is False


# ── embeddings guard + loader ────────────────────────────────────


def test_guard_offline_blocks_pickle_only(hf_cache):
    from core.rag.embeddings import UnsafeEmbeddingModelError, _guard_model_security
    _make_cache(hf_cache, "org/pk", {"config.json": "{}", "pytorch_model.bin": "x"})
    with _no_network():
        with pytest.raises(UnsafeEmbeddingModelError):
            _guard_model_security("org/pk", local_only = True)


def test_guard_offline_allows_safetensors(hf_cache):
    from core.rag.embeddings import _guard_model_security
    _make_cache(hf_cache, "org/st", {"modules.json": MODULES_JSON, "model.safetensors": "x"})
    with _no_network():
        _guard_model_security("org/st", local_only = True)  # must not raise


def _install_fake_sentence_transformers(monkeypatch, captured):
    class FakeSentenceTransformer:
        def __init__(
            self,
            name,
            *,
            device = None,
            model_kwargs = None,
            local_files_only = False,
            **kw,
        ):
            captured["name"] = name
            captured["device"] = device
            captured["local_files_only"] = local_files_only

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_get_offline_threads_local_files_only(hf_cache, monkeypatch):
    from core.rag import embeddings

    _make_cache(hf_cache, "org/st", {"modules.json": MODULES_JSON, "model.safetensors": "x"})
    # TRANSFORMERS_OFFLINE only: local_files_only is passed per-call so the load is cache-only
    # regardless of the process-wide (import-frozen) HF_HUB_OFFLINE constant.
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setattr(embeddings, "_model", None, raising = False)
    monkeypatch.setattr(embeddings, "_name", None, raising = False)
    monkeypatch.setattr(embeddings, "_install_torchao_stub_once", lambda: None)
    monkeypatch.setattr(embeddings, "_device", lambda: "cpu")
    captured = {}
    _install_fake_sentence_transformers(monkeypatch, captured)
    with _no_network():
        embeddings._get("org/st")
    assert captured["name"] == "org/st"
    assert captured["local_files_only"] is True


def test_get_online_omits_local_files_only(monkeypatch):
    from core.rag import embeddings

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setattr(embeddings, "_model", None, raising = False)
    monkeypatch.setattr(embeddings, "_name", None, raising = False)
    monkeypatch.setattr(embeddings, "_install_torchao_stub_once", lambda: None)
    monkeypatch.setattr(embeddings, "_device", lambda: "cpu")
    # Isolate the loader wiring from the online guard's network calls.
    monkeypatch.setattr(embeddings, "_guard_model_security", lambda name, local_only = False: None)
    captured = {}
    _install_fake_sentence_transformers(monkeypatch, captured)
    embeddings._get("org/online")
    assert captured["local_files_only"] is False
