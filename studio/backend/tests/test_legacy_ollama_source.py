# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The compat /api/models/local path tags Ollama rows with their real source.

The chat and training pickers group and label rows by ``source``, and the
frontend branches on ``"ollama"`` explicitly. The legacy scanner in
routes/models.py stamped its rows ``"custom"`` — and the legacy LocalModelInfo
schema did not even admit ``"ollama"``, so correcting the stamp alone 500'd the
route. The custom-folder merge then re-stamped every row from a registered
folder, so a user who registered ``~/.ollama/models`` (Studio's own recommended
folder) lost the attribution a second way. (#9986)
"""

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


def test_legacy_scanner_tags_rows_ollama(tmp_path):
    from routes import models as models_route

    root = tmp_path / "ollama"
    _write_ollama_store(root)
    found = models_route._scan_ollama_dir(root)
    assert len(found) == 1
    assert found[0].source == "ollama"
    assert found[0].model_id == "ollama/llama3:latest"


@pytest.mark.parametrize("source", ["hf_cache", "ollama"])
def test_custom_folder_merge_keeps_real_sources(source):
    # The merge step mirrors _promote_to_custom_source() in
    # hub/services/models/local_inventory.py: only unattributed rows become
    # "custom"; a registered ~/.ollama/models or HF-cache shadow keeps its label.
    src = (Path(__file__).resolve().parent.parent / "routes" / "models.py").read_text(
        encoding = "utf-8"
    )
    assert 'm if m.source in ("hf_cache", "ollama") else' in src
    row = LocalModelInfo(
        id = "/x",
        display_name = "x",
        path = "/x",
        source = source,
    )
    kept = (
        row if row.source in ("hf_cache", "ollama") else row.model_copy(update = {"source": "custom"})
    )
    assert kept.source == source
