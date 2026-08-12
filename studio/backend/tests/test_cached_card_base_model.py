# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
import types
from pathlib import Path

import pytest

# Keep this test runnable without optional logging deps.
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

from hub.services.models import cache_inventory as CI


# Verbatim frontmatter shape of unsloth/Muse-Glimmer-30B-GGUF: the base model is a
# list under `base_model`, and `tags` never carries the Hub's synthesized entries.
MUSE_GLIMMER_CARD = """---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- unsloth
- meta
base_model:
- meta-models/Muse-Glimmer-30B
---

# Muse Glimmer
"""


def _snapshot(tmp_path: Path, card: str) -> Path:
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents = True)
    (snapshot / "README.md").write_text(card, encoding = "utf-8")
    return snapshot


def test_card_base_model_reads_a_list(tmp_path):
    snapshot = _snapshot(tmp_path, MUSE_GLIMMER_CARD)
    assert CI._cached_repo_base_model(snapshot) == "meta-models/Muse-Glimmer-30B"


def test_card_base_model_reads_a_bare_string():
    assert CI._card_base_model({"base_model": " meta-llama/Llama-3.1-8B "}) == (
        "meta-llama/Llama-3.1-8B"
    )


@pytest.mark.parametrize(
    "card",
    [{}, {"base_model": ""}, {"base_model": []}, {"base_model": [None, "  "]}, {"base_model": 7}],
)
def test_card_base_model_ignores_junk(card):
    assert CI._card_base_model(card) is None


def test_cached_repo_base_model_tolerates_a_missing_card(tmp_path):
    assert CI._cached_repo_base_model(None) is None
    assert CI._cached_repo_base_model(tmp_path) is None


def test_local_metadata_carries_the_base_model(tmp_path):
    repo_path = tmp_path / "models--unsloth--Muse-Glimmer-30B-GGUF"
    repo_path.mkdir()
    snapshot = _snapshot(repo_path, MUSE_GLIMMER_CARD)

    metadata = CI._cached_model_local_metadata(repo_path, snapshot)

    assert metadata["base_model"] == "meta-models/Muse-Glimmer-30B"
    # The card's own tag list stays untouched, and carries no base_model entry.
    assert metadata["tags"] == ["unsloth", "meta"]


def test_gguf_scan_emits_the_card_base_model(tmp_path, monkeypatch):
    """The GGUF scan skips the checkpoint metadata pass, so it must read the card itself."""
    from types import SimpleNamespace

    from hub.utils import inventory_scan

    snapshot_id = "a" * 40
    repo_dir = tmp_path / "models--unsloth--Muse-Glimmer-30B-GGUF"
    snapshot = repo_dir / "snapshots" / snapshot_id
    snapshot.mkdir(parents = True)
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text(snapshot_id, encoding = "utf-8")
    (snapshot / "Muse-Glimmer-30B-UD-Q4_K_XL.gguf").write_bytes(b"\0" * 32)
    (snapshot / "README.md").write_text(MUSE_GLIMMER_CARD, encoding = "utf-8")

    monkeypatch.setattr(inventory_scan, "hf_cache_roots", lambda **kw: [tmp_path])
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = tmp_path),
    )
    inventory_scan.invalidate_hf_cache_scans()
    try:
        rows = CI._scan_cached_gguf()
    finally:
        inventory_scan.invalidate_hf_cache_scans()

    assert [row["repo_id"] for row in rows] == ["unsloth/Muse-Glimmer-30B-GGUF"]
    assert rows[0]["base_model"] == "meta-models/Muse-Glimmer-30B"


def test_response_schemas_keep_the_base_model():
    """The routes serialize through these models, so an undeclared field is dropped."""
    from hub.schemas.inventory import CachedGgufResponse, CachedModelsResponse

    row = {"repo_id": "unsloth/Muse-Glimmer-30B-GGUF", "base_model": "meta-models/Muse-Glimmer-30B"}
    for response_model in (CachedGgufResponse, CachedModelsResponse):
        payload = response_model.model_validate({"cached": [row]}).model_dump()
        assert payload["cached"][0]["base_model"] == "meta-models/Muse-Glimmer-30B", response_model


def test_managed_gguf_download_keeps_the_card():
    """The card is the only on-disk source of base_model, so the GGUF fetch must allow it."""
    from hub.workers.hf_download import _gguf_allow_patterns

    targets = ["Muse-Glimmer-30B-UD-Q4_K_XL.gguf", "mmproj-F16.gguf"]
    patterns = _gguf_allow_patterns(targets)

    assert patterns[: len(targets)] == targets, "shards must still be fetched"
    assert "README.md" in patterns
