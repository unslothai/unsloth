"""Offline GGUF export must not probe the Hub for VLM tokenizer metadata (issue #7481).

Regression for ``PreTrainedTokenizerFast.from_pretrained`` on a repo id calling
``is_base_mistral()`` -> ``model_info()`` even with ``TRANSFORMERS_OFFLINE=1``.
Pure CPU, no network, no GPU.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from unsloth.models import loader_utils as L


_REPO = "llmfan46/gemma-4-E4B-it-ultra-uncensored-heretic"
_COMMIT = "5964fe4c7339c5974e879baba8982a09616f68ca"


def _write_gemma4_cache(
    root,
    repo_id = _REPO,
    commit = _COMMIT,
):
    """Minimal cached snapshot matching the reporter's layout."""
    org, name = repo_id.split("/")
    repo_root = root / f"models--{org}--{name}"
    snap = repo_root / "snapshots" / commit
    snap.mkdir(parents = True)
    refs = repo_root / "refs"
    refs.mkdir(parents = True, exist_ok = True)
    (refs / "main").write_text(commit, encoding = "utf-8")
    (snap / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "GemmaTokenizer", "model_max_length": 8192}),
        encoding = "utf-8",
    )
    (snap / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": None,
                "post_processor": None,
                "decoder": None,
                "model": {"type": "BPE", "vocab": {"<pad>": 0}, "merges": []},
            }
        ),
        encoding = "utf-8",
    )
    (snap / "processor_config.json").write_text("{}", encoding = "utf-8")
    (snap / "config.json").write_text(
        json.dumps({"model_type": "gemma4"}),
        encoding = "utf-8",
    )
    return snap


def _offline_env(monkeypatch, cache_root):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))


def test_resolve_hub_repo_cached_file_finds_tokenizer_model(tmp_path, monkeypatch):
    snap = _write_gemma4_cache(tmp_path)
    (snap / "tokenizer.model").write_bytes(b"sp-model")
    _offline_env(monkeypatch, tmp_path)

    got = L._resolve_hub_repo_cached_file(
        _REPO,
        "tokenizer.model",
        local_files_only=True,
        cache_dir=str(tmp_path),
    )
    assert got == str(snap / "tokenizer.model")


def test_resolve_hub_repo_local_dir_from_cached_snapshot(tmp_path, monkeypatch):
    snap = _write_gemma4_cache(tmp_path)
    _offline_env(monkeypatch, tmp_path)

    got = L._resolve_hub_repo_local_dir(_REPO, local_files_only=True, cache_dir=str(tmp_path))
    assert got == str(snap)


def test_hub_repo_or_local_path_prefers_snapshot_over_repo_id(tmp_path, monkeypatch):
    snap = _write_gemma4_cache(tmp_path)
    _offline_env(monkeypatch, tmp_path)

    got = L._hub_repo_or_local_path(_REPO, local_files_only=True, cache_dir=str(tmp_path))
    assert got == str(snap)
    assert got != _REPO


def test_hub_repo_or_local_path_keeps_repo_id_online(tmp_path, monkeypatch):
    snap = _write_gemma4_cache(tmp_path)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    got = L._hub_repo_or_local_path(_REPO, local_files_only=False, cache_dir=str(tmp_path))
    assert got == _REPO
    assert got != str(snap)


def test_has_tokenizer_model_offline_does_not_cache_negative(tmp_path, monkeypatch):
    from unsloth.save import _TOKENIZER_MODEL_CACHE, _has_tokenizer_model

    snap = _write_gemma4_cache(tmp_path)
    _offline_env(monkeypatch, tmp_path)
    _TOKENIZER_MODEL_CACHE.clear()

    tok = SimpleNamespace(name_or_path=_REPO)
    assert _has_tokenizer_model(tok, token=None) is False
    assert _REPO not in _TOKENIZER_MODEL_CACHE

    (snap / "tokenizer.model").write_bytes(b"sp-model")
    assert _has_tokenizer_model(tok, token=None) is True


def test_preserve_sentencepiece_offline_copies_cached_model(tmp_path, monkeypatch):
    from unsloth.save import _TOKENIZER_MODEL_CACHE, _preserve_sentencepiece_tokenizer_assets

    snap = _write_gemma4_cache(tmp_path)
    (snap / "tokenizer.model").write_bytes(b"cached-sp-model")
    _offline_env(monkeypatch, tmp_path)
    _TOKENIZER_MODEL_CACHE.clear()

    save_dir = tmp_path / "export"
    save_dir.mkdir()
    (save_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    tok = SimpleNamespace(name_or_path=_REPO)

    _preserve_sentencepiece_tokenizer_assets(tok, str(save_dir))

    assert (save_dir / "tokenizer.model").read_bytes() == b"cached-sp-model"


def test_load_pretrained_tokenizer_fast_passes_snapshot_not_repo_id(tmp_path, monkeypatch):
    snap = _write_gemma4_cache(tmp_path)
    _offline_env(monkeypatch, tmp_path)

    seen_paths = []

    class _FakeFast:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            seen_paths.append(path)
            assert kwargs.get("local_files_only") is True
            return SimpleNamespace(name_or_path = path)

    monkeypatch.setattr(
        "transformers.PreTrainedTokenizerFast",
        _FakeFast,
        raising = False,
    )

    with patch("huggingface_hub.HfApi.model_info") as model_info:
        model_info.side_effect = AssertionError("model_info must not run offline")
        tok = L._load_pretrained_tokenizer_fast(_REPO, cache_dir = str(tmp_path))

    assert seen_paths == [str(snap)]
    assert tok.name_or_path == str(snap)


def test_has_tokenizer_model_offline_skips_model_info(tmp_path, monkeypatch):
    from unsloth.save import _TOKENIZER_MODEL_CACHE, _has_tokenizer_model

    _write_gemma4_cache(tmp_path)
    _offline_env(monkeypatch, tmp_path)
    _TOKENIZER_MODEL_CACHE.clear()

    tok = SimpleNamespace(name_or_path = _REPO)

    with patch("huggingface_hub.HfApi.model_info") as model_info:
        model_info.side_effect = AssertionError("model_info must not run offline")
        assert _has_tokenizer_model(tok, token = None) is False
