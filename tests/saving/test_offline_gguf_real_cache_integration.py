# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Integration tests for #7481 using real cached Gemma weights.

Requires a one-time online download into ``$HF_HOME`` (defaults to a
``hf_offline_test_cache`` directory under the platform temp dir):

  HF_HOME=<cache> python -c \\
    "from huggingface_hub import snapshot_download; snapshot_download('unsloth/gemma-3-270m-it-bnb-4bit', cache_dir='<cache>/hub')"

Every test here drives unsloth's own resolver. Resolving through
``hf_hub_download`` directly would pass with the fix reverted, since that is
plain huggingface_hub behaviour rather than anything this change touches.

Importing unsloth pulls the whole package graph, which CPU-only hosts cannot
do, so the suite is gated behind ``UNSLOTH_INTEGRATION_IMPORT=1``.
"""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path

import pytest

REPO = "unsloth/gemma-3-270m-it-bnb-4bit"
CACHE_ROOT = Path(
    os.environ.get("HF_HOME") or os.path.join(tempfile.gettempdir(), "hf_offline_test_cache")
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("UNSLOTH_INTEGRATION_IMPORT") != "1",
        reason = "full unsloth import needs a GPU host; set UNSLOTH_INTEGRATION_IMPORT=1 to enable",
    ),
]


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

    # Patch the method, not the class: replacing socket.socket itself breaks any
    # isinstance(x, socket.socket) in the stack under test.
    monkeypatch.setattr(socket.socket, "connect", _guard)
    monkeypatch.setattr(socket, "create_connection", _guard)
    monkeypatch.setattr(socket, "getaddrinfo", _guard)


def _offline_env(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(CACHE_ROOT))


def test_real_cached_snapshot_resolves_offline(monkeypatch):
    _require_cached_repo()
    _offline_env(monkeypatch)
    _block_network(monkeypatch)

    from unsloth.models.loader_utils import _resolve_hub_repo_local_dir

    snap = Path(
        _resolve_hub_repo_local_dir(
            REPO,
            cache_dir = str(CACHE_ROOT / "hub"),
            local_files_only = True,
        )
    )
    assert (snap / "tokenizer.json").is_file()
    assert (snap / "tokenizer.model").is_file()


def test_real_cached_tokenizer_loads_from_snapshot_not_repo_id(monkeypatch):
    """The #7481 fix: the loader hands transformers a snapshot dir, not a repo id."""
    _require_cached_repo()
    _offline_env(monkeypatch)
    _block_network(monkeypatch)

    from unsloth.models.loader_utils import _load_pretrained_tokenizer_fast

    tok = _load_pretrained_tokenizer_fast(
        REPO,
        local_files_only = True,
        cache_dir = str(CACHE_ROOT / "hub"),
    )
    assert tok.vocab_size > 0
    # A repo id here means the Hub metadata probe was reached, which is the bug.
    assert tok.name_or_path != REPO
    assert Path(tok.name_or_path).is_dir()


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
