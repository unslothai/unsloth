# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Offline GGUF export must not probe the Hub for VLM tokenizer metadata (issue #7481).

Regression for ``PreTrainedTokenizerFast.from_pretrained`` on a repo id calling
``is_base_mistral()`` -> ``model_info()`` even with ``TRANSFORMERS_OFFLINE=1``.
Pure CPU, no network, no GPU.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

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
        local_files_only = True,
        cache_dir = str(tmp_path),
    )
    assert got == str(snap / "tokenizer.model")


def test_resolve_hub_repo_local_dir_from_cached_snapshot(tmp_path, monkeypatch):
    snap = _write_gemma4_cache(tmp_path)
    _offline_env(monkeypatch, tmp_path)

    got = L._resolve_hub_repo_local_dir(_REPO, local_files_only = True, cache_dir = str(tmp_path))
    assert got == str(snap)


def test_hub_repo_or_local_path_prefers_snapshot_over_repo_id(tmp_path, monkeypatch):
    snap = _write_gemma4_cache(tmp_path)
    _offline_env(monkeypatch, tmp_path)

    got = L._hub_repo_or_local_path(_REPO, local_files_only = True, cache_dir = str(tmp_path))
    assert got == str(snap)
    assert got != _REPO


def test_hub_repo_or_local_path_keeps_repo_id_online(tmp_path, monkeypatch):
    snap = _write_gemma4_cache(tmp_path)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    got = L._hub_repo_or_local_path(_REPO, local_files_only = False, cache_dir = str(tmp_path))
    assert got == _REPO
    assert got != str(snap)


def test_has_tokenizer_model_offline_does_not_cache_negative(tmp_path, monkeypatch):
    from unsloth.save import _TOKENIZER_MODEL_CACHE, _has_tokenizer_model

    snap = _write_gemma4_cache(tmp_path)
    _offline_env(monkeypatch, tmp_path)
    _TOKENIZER_MODEL_CACHE.clear()

    tok = SimpleNamespace(name_or_path = _REPO)
    assert _has_tokenizer_model(tok, token = None) is False
    assert _REPO not in _TOKENIZER_MODEL_CACHE

    (snap / "tokenizer.model").write_bytes(b"sp-model")
    assert _has_tokenizer_model(tok, token = None) is True


def test_preserve_sentencepiece_offline_copies_cached_model(tmp_path, monkeypatch):
    from unsloth.save import _TOKENIZER_MODEL_CACHE, _preserve_sentencepiece_tokenizer_assets

    snap = _write_gemma4_cache(tmp_path)
    (snap / "tokenizer.model").write_bytes(b"cached-sp-model")
    _offline_env(monkeypatch, tmp_path)
    _TOKENIZER_MODEL_CACHE.clear()

    save_dir = tmp_path / "export"
    save_dir.mkdir()
    (save_dir / "tokenizer_config.json").write_text("{}", encoding = "utf-8")
    tok = SimpleNamespace(name_or_path = _REPO)

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

    # A raising side_effect proves nothing: _has_tokenizer_model wraps the call
    # in `except Exception: return False`, so it passes with the fix reverted.
    with patch("huggingface_hub.HfApi.model_info") as model_info:
        assert _has_tokenizer_model(tok, token = None) is False
    assert model_info.call_count == 0


def test_has_tokenizer_model_probes_cache_before_model_info(tmp_path, monkeypatch):
    from unsloth.save import _TOKENIZER_MODEL_CACHE, _has_tokenizer_model

    snap = _write_gemma4_cache(tmp_path)
    (snap / "tokenizer.model").write_bytes(b"sp-model")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    _TOKENIZER_MODEL_CACHE.clear()

    tok = SimpleNamespace(name_or_path = _REPO)

    with patch("huggingface_hub.HfApi.model_info") as model_info:
        model_info.side_effect = AssertionError("model_info must not run when cache hit")
        assert _has_tokenizer_model(tok, token = None) is True


def test_offline_aware_load_persists_local_only_for_saving(tmp_path, monkeypatch):
    """An explicit ``local_files_only = True`` load must still be local-only at save time.

    ``transformers`` takes ``local_files_only`` as an explicit ``from_pretrained``
    parameter, so it never reaches ``tokenizer.init_kwargs``, and
    ``_offline_aware_load`` restores the offline env vars once the load returns.
    Without the stamp the request is invisible by the time we save.
    """
    from unsloth.save import _TOKENIZER_MODEL_CACHE, _has_tokenizer_model

    # Snapshot has tokenizer metadata but deliberately no tokenizer.model, so the cache probe misses and only the
    # local-only stamp can stop the Hub request.
    _write_gemma4_cache(tmp_path)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    _TOKENIZER_MODEL_CACHE.clear()

    @L._offline_aware_load
    def _load(model_name, **kwargs):
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        # A processor keeps the Hub repo id and carries no local_files_only.
        return object(), SimpleNamespace(
            tokenizer = SimpleNamespace(name_or_path = model_name, init_kwargs = {}),
        )

    _model, processor = _load(_REPO, local_files_only = True)

    assert os.environ.get("HF_HUB_OFFLINE") is None
    assert processor.tokenizer.init_kwargs.get("local_files_only") is None
    assert L._tokenizer_wants_local_only(processor.tokenizer) is True

    with patch("huggingface_hub.HfApi.model_info") as model_info:
        model_info.return_value = SimpleNamespace(
            siblings = [SimpleNamespace(rfilename = "tokenizer.model")],
        )
        assert _has_tokenizer_model(processor, token = None) is False
    assert model_info.call_count == 0


def test_preserve_sentencepiece_after_local_only_load_never_downloads(tmp_path, monkeypatch):
    """The save path inherits the load's local-only mode: no metadata probe, no download."""
    import huggingface_hub

    from unsloth.save import _TOKENIZER_MODEL_CACHE, _preserve_sentencepiece_tokenizer_assets

    _write_gemma4_cache(tmp_path)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    _TOKENIZER_MODEL_CACHE.clear()

    @L._offline_aware_load
    def _load(model_name, **kwargs):
        return object(), SimpleNamespace(
            tokenizer = SimpleNamespace(name_or_path = model_name, init_kwargs = {}),
        )

    _model, processor = _load(_REPO, local_files_only = True)

    save_dir = tmp_path / "export"
    save_dir.mkdir()
    (save_dir / "tokenizer_config.json").write_text("{}", encoding = "utf-8")

    real_download = huggingface_hub.hf_hub_download
    seen_local_files_only = []

    def _recording_download(*args, **kwargs):
        seen_local_files_only.append(kwargs.get("local_files_only"))
        return real_download(*args, **kwargs)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _recording_download)

    with patch("huggingface_hub.HfApi.model_info") as model_info:
        model_info.return_value = SimpleNamespace(
            siblings = [SimpleNamespace(rfilename = "tokenizer.model")],
        )
        _preserve_sentencepiece_tokenizer_assets(processor, str(save_dir), token = None)

    assert model_info.call_count == 0
    # Every hf_hub_download here must be a cache probe, never a Hub fetch.
    assert seen_local_files_only and all(seen_local_files_only)
    assert not (save_dir / "tokenizer.model").exists()


def test_has_tokenizer_model_local_files_only_skips_model_info(tmp_path, monkeypatch):
    from unsloth.save import _TOKENIZER_MODEL_CACHE, _has_tokenizer_model

    _write_gemma4_cache(tmp_path)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    _TOKENIZER_MODEL_CACHE.clear()

    tok = SimpleNamespace(
        name_or_path = _REPO,
        init_kwargs = {"local_files_only": True},
    )

    with patch("huggingface_hub.HfApi.model_info") as model_info:
        assert _has_tokenizer_model(tok, token = None) is False
    assert model_info.call_count == 0


def test_custom_cache_dir_survives_to_saving(tmp_path, monkeypatch):
    """A local-only load with a caller-supplied cache_dir that no env var points
    at. Saving derives its cache from HF_HUB_CACHE / HF_HOME, so without the
    stamp it probes the wrong place, and the local-only marker then stops it
    falling back to the Hub, silently dropping tokenizer.model."""
    from unsloth.save import _TOKENIZER_MODEL_CACHE, _has_tokenizer_model

    custom_cache = tmp_path / "caller_cache"
    custom_cache.mkdir()
    snap = _write_gemma4_cache(custom_cache)
    (snap / "tokenizer.model").write_bytes(b"sp-model")

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "unrelated"))
    _TOKENIZER_MODEL_CACHE.clear()

    @L._offline_aware_load
    def _load(**kwargs):
        return SimpleNamespace(name_or_path = _REPO)

    tok = _load(local_files_only = True, cache_dir = str(custom_cache))

    assert L._tokenizer_cache_dir(tok) == str(custom_cache)
    with patch("huggingface_hub.HfApi.model_info") as model_info:
        assert _has_tokenizer_model(tok, token = None) is True
    assert model_info.call_count == 0
