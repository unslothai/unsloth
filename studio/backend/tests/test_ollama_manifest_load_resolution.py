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
from urllib.parse import quote

import pytest
from fastapi import HTTPException


def _write_ollama_store(root: Path, *, extra_layers: tuple[str, ...] = ()) -> Path:
    """A minimal Ollama layout whose optional extra layers resolve to real blobs."""
    digest_value = "a" * 64
    blob = root / "blobs" / f"sha256-{digest_value}"
    blob.parent.mkdir(parents = True)
    blob.write_bytes(b"GGUF-not-really")

    layers = [
        {
            "mediaType": "application/vnd.ollama.image.model",
            "digest": f"sha256:{digest_value}",
        }
    ]
    for index, media_type in enumerate(extra_layers, start = 1):
        layer_digest = f"{index:064x}"
        (root / "blobs" / f"sha256-{layer_digest}").write_text("{}", encoding = "utf-8")
        layers.append(
            {
                "mediaType": media_type,
                "digest": f"sha256:{layer_digest}",
            }
        )

    tag_file = root / "manifests" / "registry.ollama.ai" / "library" / "llama3" / "latest"
    tag_file.parent.mkdir(parents = True)
    tag_file.write_text(
        json.dumps(
            {
                "config": {},
                "layers": layers,
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


def test_ollama_intent_loads_the_link_but_keeps_the_manifest_identity(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from models.inference import LoadRequest
    from routes.inference import (
        _LoadPlacement,
        _llama_status_model_ids,
        _resolve_gguf_load_intent,
        _resolve_model_identifier_for_request,
    )

    ref = _manifest_ref(tmp_path, monkeypatch)
    resolved, _, _ = _resolve_model_identifier_for_request(
        LoadRequest(model_path = ref), operation = "load-model"
    )
    config = SimpleNamespace(
        identifier = resolved,
        gguf_hf_repo = None,
        gguf_file = resolved,
        gguf_mmproj_file = None,
        gguf_mtp_file = None,
        gguf_dspark_file = None,
        gguf_dflash_file = None,
        gguf_variant = None,
        is_vision = False,
    )

    intent = _resolve_gguf_load_intent(
        config,
        LoadRequest(model_path = ref),
        native_grant_backed = False,
        chat_template_override = None,
        extra_args = None,
        placement = _LoadPlacement(None, None, False, None),
        n_parallel = 1,
    )

    assert intent.gguf_path == resolved
    assert intent.model_identifier == ref

    backend = SimpleNamespace(
        model_identifier = intent.model_identifier,
        _native_grant_backed = False,
        _native_display_label = None,
        _openai_advertised_id = None,
    )
    assert _llama_status_model_ids(backend) == (ref, ref)


_UNSUPPORTED_RUNTIME_LAYERS = (
    "application/vnd.ollama.image.params",
    "application/vnd.ollama.image.template",
    "application/vnd.ollama.image.system",
    "application/vnd.ollama.image.messages",
    "application/vnd.ollama.image.adapter",
    "application/vnd.ollama.image.prompt",
    "application/vnd.ollama.image.future-runtime",
)


def _rich_manifest_ref(tmp_path: Path, monkeypatch) -> tuple[Path, str]:
    from hub.services.models import ollama

    root = tmp_path / "ollama-rich"
    tag_file = _write_ollama_store(root, extra_layers = _UNSUPPORTED_RUNTIME_LAYERS)
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    ref = f"ollama-manifest:{quote(str(tag_file), safe = '')}"
    return root, ref


def test_rich_manifest_is_withheld_from_inventory(tmp_path, monkeypatch):
    from hub.services.models import ollama
    root, _ = _rich_manifest_ref(tmp_path, monkeypatch)

    assert ollama.scan_ollama_dir(root) == []


def test_rich_manifest_ref_is_rejected_without_creating_links(tmp_path, monkeypatch):
    from models.inference import LoadRequest
    from routes.inference import _resolve_model_identifier_for_request

    root, ref = _rich_manifest_ref(tmp_path, monkeypatch)

    with pytest.raises(HTTPException) as excinfo:
        _resolve_model_identifier_for_request(LoadRequest(model_path = ref), operation = "load-model")

    assert excinfo.value.status_code == 400
    assert "unsupported runtime layers" in str(excinfo.value.detail).lower()
    for media_type in _UNSUPPORTED_RUNTIME_LAYERS:
        assert media_type in str(excinfo.value.detail)
    links_root = root / ".studio_links"
    assert not links_root.exists() or not any(path.is_file() for path in links_root.rglob("*"))


def test_license_metadata_does_not_hide_a_plain_manifest(tmp_path, monkeypatch):
    from hub.services.models import ollama

    root = tmp_path / "ollama-licensed"
    _write_ollama_store(
        root,
        extra_layers = ("application/vnd.ollama.image.license",),
    )
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])

    rows = ollama.scan_ollama_dir(root)
    assert len(rows) == 1
    assert ollama.is_ollama_manifest_ref(rows[0].load_id)


def test_non_object_manifest_ref_is_a_400(tmp_path, monkeypatch):
    from hub.services.models import ollama
    from models.inference import LoadRequest
    from routes.inference import _resolve_model_identifier_for_request

    root = tmp_path / "ollama-non-object-manifest"
    tag_file = _write_ollama_store(root)
    tag_file.write_text("[]", encoding = "utf-8")
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    ref = f"ollama-manifest:{quote(str(tag_file), safe = '')}"

    assert ollama.scan_ollama_dir(root) == []
    with pytest.raises(HTTPException) as excinfo:
        _resolve_model_identifier_for_request(LoadRequest(model_path = ref), operation = "load-model")
    assert excinfo.value.status_code == 400
    assert "manifest" in str(excinfo.value.detail).lower()


def test_non_object_config_blob_ref_is_a_400(tmp_path, monkeypatch):
    from hub.services.models import ollama
    from models.inference import LoadRequest
    from routes.inference import _resolve_model_identifier_for_request

    root = tmp_path / "ollama-non-object-config"
    tag_file = _write_ollama_store(root)
    config_digest = "b" * 64
    (root / "blobs" / f"sha256-{config_digest}").write_text("[]", encoding = "utf-8")
    manifest = json.loads(tag_file.read_text(encoding = "utf-8"))
    manifest["config"] = {"digest": f"sha256:{config_digest}"}
    tag_file.write_text(json.dumps(manifest), encoding = "utf-8")
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    ref = f"ollama-manifest:{quote(str(tag_file), safe = '')}"

    assert ollama.scan_ollama_dir(root) == []
    with pytest.raises(HTTPException) as excinfo:
        _resolve_model_identifier_for_request(LoadRequest(model_path = ref), operation = "load-model")
    assert excinfo.value.status_code == 400
    assert "config blob" in str(excinfo.value.detail).lower()


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
