# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ollama rows keep source ollama on the compat models route, even from a registered custom folder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.models import LocalModelInfo


def _write_ollama_store(root: Path) -> None:
    digest_value = "b" * 64
    blob = root / "blobs" / f"sha256-{digest_value}"
    blob.parent.mkdir(parents = True)
    blob.write_bytes(b"GGUF-not-really")
    tag_file = root / "manifests" / "registry.ollama.ai" / "library" / "llama3" / "latest"
    tag_file.parent.mkdir(parents = True)
    tag_file.write_text(
        json.dumps(
            {
                "config": {},
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": f"sha256:{digest_value}",
                    }
                ],
            }
        ),
        encoding = "utf-8",
    )


def test_legacy_schema_admits_the_ollama_source():
    row = LocalModelInfo(
        id = "/links/llama3.gguf",
        display_name = "llama3:latest",
        path = "/links/llama3.gguf",
        source = "ollama",
    )
    assert row.source == "ollama"


def test_a_registered_ollama_store_lists_ollama_rows(tmp_path):
    from hub.services.models import local_inventory
    from routes import models as models_route

    root = tmp_path / "ollama"
    _write_ollama_store(root)
    empty = tmp_path / "empty"
    sources = local_inventory._LocalInventorySources(
        hf_cache_dir = empty,
        legacy_hf = empty,
        hf_default = empty,
        lm_dirs = (),
        ollama_dirs = (),
        hermes_dirs = (),
        known_hf_caches = (),
    )
    found = models_route.collect_local_models(
        tmp_path / "models", custom_folders = [{"path": str(root)}], sources = sources
    )
    assert [(m.source, m.model_id) for m in found] == [("ollama", "ollama/llama3:latest")]
    assert found[0].path.endswith(".gguf")


@pytest.mark.parametrize("source", ["hf_cache", "ollama", "hermes"])
def test_custom_folder_merge_keeps_real_sources(source):
    from hub.services.models.common import _local_model_info
    from hub.services.models.local_inventory import _promote_to_custom_source

    row = _local_model_info(
        scan_path = Path("/x"), load_path = Path("/x"), source = source, model_format = "gguf"
    )
    assert _promote_to_custom_source(row).source == source
