# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /load resolves opaque ``ollama-manifest:`` inventory references.

The read-only hub inventory scan returns Ollama rows whose load id is an
``ollama-manifest:`` reference (hub/services/models/ollama.py); the load path
owns the filesystem write that turns one into a loadable ``.gguf`` link. These
tests pin that hand-off at ``_resolve_model_identifier_for_request``, which both
the load and validate routes go through.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException


def _write_ollama_store(root: Path) -> Path:
    """A minimal Ollama layout: one manifest whose model layer resolves to a blob."""
    digest_value = "a" * 64
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
    return tag_file


def _manifest_ref(tmp_path: Path, monkeypatch) -> str:
    from hub.services.models import ollama

    root = tmp_path / "ollama"
    _write_ollama_store(root)
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])

    rows = ollama.scan_ollama_dir(root)
    assert len(rows) == 1
    ref = rows[0].load_id
    assert ollama.is_ollama_manifest_ref(ref)
    return ref


def test_load_resolves_a_manifest_ref_to_a_gguf_link(tmp_path, monkeypatch):
    from models.inference import LoadRequest
    from routes.inference import _resolve_model_identifier_for_request

    ref = _manifest_ref(tmp_path, monkeypatch)
    identifier, label, native_grant_backed = _resolve_model_identifier_for_request(
        LoadRequest(model_path = ref), operation = "load-model"
    )

    assert identifier.endswith(".gguf")
    assert Path(identifier).is_file(), "the .gguf link must be materialized"
    assert label == Path(identifier).name, "logs get the link name, not the opaque ref"
    assert native_grant_backed is False


def test_a_ref_outside_known_ollama_dirs_is_a_400(tmp_path, monkeypatch):
    from hub.services.models import ollama
    from models.inference import LoadRequest
    from routes.inference import _resolve_model_identifier_for_request

    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [tmp_path / "elsewhere"])

    outside = tmp_path / "not-ollama" / "manifests" / "x" / "y" / "latest"
    ref = f"ollama-manifest:{outside}"
    with pytest.raises(HTTPException) as excinfo:
        _resolve_model_identifier_for_request(LoadRequest(model_path = ref), operation = "load-model")
    assert excinfo.value.status_code == 400


def test_non_ollama_paths_take_the_existing_path(tmp_path):
    from models.inference import LoadRequest
    from routes.inference import _resolve_model_identifier_for_request

    identifier, label, native_grant_backed = _resolve_model_identifier_for_request(
        LoadRequest(model_path = "unsloth/model-GGUF"), operation = "load-model"
    )
    assert (identifier, label, native_grant_backed) == (
        "unsloth/model-GGUF",
        "unsloth/model-GGUF",
        False,
    )
