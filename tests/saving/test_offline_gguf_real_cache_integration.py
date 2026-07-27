"""Integration tests for #7481 using real cached Gemma weights.

Requires a one-time online download:
  HF_HOME=/tmp/hf_offline_test_cache python -c \\
    "from huggingface_hub import snapshot_download; snapshot_download('unsloth/gemma-3-270m-it-bnb-4bit', cache_dir='/tmp/hf_offline_test_cache/hub')"

No GPU. No full unsloth import (CPU-only hosts cannot import the package graph).
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

REPO = "unsloth/gemma-3-270m-it-bnb-4bit"
CACHE_ROOT = Path(os.environ.get("HF_HOME", "/tmp/hf_offline_test_cache"))


def _require_cached_repo():
    from huggingface_hub import scan_cache_dir

    cache_dir = CACHE_ROOT / "hub"
    if not cache_dir.exists():
        pytest.skip(f"cache missing at {cache_dir}; run snapshot_download for {REPO}")
    repos = [r.repo_id for r in scan_cache_dir(str(cache_dir)).repos]
    if REPO not in repos:
        pytest.skip(f"{REPO} not in {cache_dir}")


def _block_network(monkeypatch):
    def _guard(*args, **kwargs):
        raise OSError("network blocked for offline integration test")

    monkeypatch.setattr(socket, "socket", _guard)
    monkeypatch.setattr(socket, "create_connection", _guard)
    monkeypatch.setattr(socket, "getaddrinfo", _guard)


def _offline_env(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(CACHE_ROOT))


def _resolve_snapshot(cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        REPO,
        "tokenizer_config.json",
        cache_dir = str(cache_dir),
        local_files_only = True,
    )
    return Path(path).parent


@pytest.mark.integration
def test_real_cached_snapshot_resolves_offline(monkeypatch):
    _require_cached_repo()
    _offline_env(monkeypatch)
    _block_network(monkeypatch)

    snap = _resolve_snapshot(CACHE_ROOT / "hub")
    assert (snap / "tokenizer.json").is_file()
    assert (snap / "tokenizer.model").is_file()


@pytest.mark.integration
def test_real_cached_tokenizer_loads_from_snapshot_not_repo_id(monkeypatch):
    """Mirrors the #7481 fix: load from snapshot dir, not Hub repo id."""
    _require_cached_repo()
    _offline_env(monkeypatch)
    _block_network(monkeypatch)

    from transformers import PreTrainedTokenizerFast

    snap = _resolve_snapshot(CACHE_ROOT / "hub")
    tok = PreTrainedTokenizerFast.from_pretrained(str(snap), local_files_only = True)
    assert tok.vocab_size > 0

    # Repo-id path is what triggered model_info() offline in #7481; snapshot path is the fix.
    assert str(snap) != REPO
    assert "/" not in Path(str(snap)).name


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("UNSLOTH_INTEGRATION_IMPORT") != "1",
    reason = "full unsloth import needs GPU host; set UNSLOTH_INTEGRATION_IMPORT=1 to enable",
)
def test_real_cached_unsloth_helpers_offline(monkeypatch):
    _require_cached_repo()
    _offline_env(monkeypatch)
    _block_network(monkeypatch)

    from unsloth.models.loader_utils import _load_pretrained_tokenizer_fast
    from unsloth.save import _has_tokenizer_model

    tok = _load_pretrained_tokenizer_fast(
        REPO,
        local_files_only = True,
        cache_dir = str(CACHE_ROOT / "hub"),
    )
    assert tok.vocab_size > 0
    assert _has_tokenizer_model(tok) is True
