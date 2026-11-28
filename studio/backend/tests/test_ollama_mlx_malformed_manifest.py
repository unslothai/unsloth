# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A malformed Ollama MLX-format manifest (a field of the wrong JSON type) skips that one tag, as a
malformed GGUF manifest already does. It used to raise TypeError out of scan_ollama_dir: the hub
inventory lost every tag in that Ollama root, and the legacy /api/models/local failed outright."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _blob(root: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    path = root / "blobs" / f"sha256-{digest}"
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(data)
    return f"sha256:{digest}"


def _tag(root: Path, name: str, tag: str, manifest: dict) -> None:
    path = root / "manifests" / "registry.ollama.ai" / "library" / name / tag
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(manifest), encoding = "utf-8")


def _store(
    root: Path,
    *,
    mlx_layer_patch = None,
    mlx_config_patch = None,
    gguf_bad = False,
) -> None:
    # A healthy GGUF tag.
    gguf_layers = [
        {"mediaType": "application/vnd.ollama.image.model", "digest": _blob(root, b"GGUF")}
    ]
    _tag(root, "llama3", "latest", {"config": {}, "layers": gguf_layers})

    # An MLX-format tag.
    cfg = {"model_format": "safetensors", "capabilities": ["completion"]}
    cfg.update(mlx_config_patch or {})
    tensor = {
        "mediaType": "application/vnd.ollama.image.tensor",
        "name": "model.norm.weight",
        "digest": _blob(root, b"tensor"),
        "size": 6,
    }
    config_json = {
        "mediaType": "application/vnd.ollama.image.json",
        "name": "config.json",
        "digest": _blob(root, b"{}"),
        "size": 2,
    }
    layers = [tensor, config_json]
    if mlx_layer_patch is not None:
        mlx_layer_patch(layers)
    _tag(
        root,
        "gemma4",
        "e2b-mlx",
        {"config": {"digest": _blob(root, json.dumps(cfg).encode())}, "layers": layers},
    )

    if gguf_bad:
        bad = [
            {"mediaType": "application/vnd.ollama.image.model", "digest": _blob(root, b"GGUF2")},
            {"mediaType": ["application/vnd.ollama.image.license"], "digest": _blob(root, b"x")},
        ]
        _tag(root, "phi3", "latest", {"config": {}, "layers": bad})


def _string_size(layers):
    layers[0]["size"] = "6"


def _list_media_type(layers):
    layers.append(
        {"mediaType": ["application/vnd.ollama.image.license"], "digest": "sha256:" + "0" * 64}
    )


MALFORMED = {
    "tensor size is a string": dict(mlx_layer_patch = _string_size),
    "a layer mediaType is a list": dict(mlx_layer_patch = _list_media_type),
    "config capabilities is a number": dict(mlx_config_patch = {"capabilities": 1}),
}


@pytest.fixture(autouse = True)
def _apple_silicon(monkeypatch):
    from hub.services.models import ollama_mlx
    monkeypatch.setattr(ollama_mlx, "mlx_host", lambda: True)


def test_control_bad_gguf_manifest_is_skipped(tmp_path):
    from hub.services.models import ollama

    root = tmp_path / "ollama"
    _store(root, gguf_bad = True)
    ids = sorted(r.model_id for r in ollama.scan_ollama_dir(root))
    assert ids == ["ollama/gemma4:e2b-mlx", "ollama/llama3:latest"]


@pytest.mark.parametrize("case", list(MALFORMED))
def test_hub_scan_keeps_healthy_tags_beside_a_bad_mlx_manifest(tmp_path, case):
    from hub.services.models import ollama

    root = tmp_path / "ollama"
    _store(root, **MALFORMED[case])
    ids = [r.model_id for r in ollama.scan_ollama_dir(root)]
    assert ids == ["ollama/llama3:latest"]


@pytest.mark.parametrize("case", list(MALFORMED))
def test_legacy_inventory_survives_a_bad_mlx_manifest_in_a_scan_folder(tmp_path, case, monkeypatch):
    from routes import models as legacy

    root = tmp_path / "ollama"
    _store(root, **MALFORMED[case])
    missing = tmp_path / "missing"
    sources = legacy._CompatLocalInventorySources(missing, missing, missing, (), ())
    monkeypatch.setattr(legacy, "note_scan_folder_scanned", lambda *a, **k: None, raising = False)
    rows = legacy.collect_local_models(
        tmp_path / "models", custom_folders = [{"path": str(root)}], sources = sources
    )
    assert "ollama/llama3:latest" in {r.model_id for r in rows}
