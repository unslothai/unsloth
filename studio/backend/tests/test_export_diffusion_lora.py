# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image-generation LoRAs can be exported from the Images catalog to a folder."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import HTTPException

from core.inference import diffusion_lora as dl
from models import ExportDiffusionLoRARequest
from routes import export as export_routes


@pytest.fixture
def loras(tmp_path, monkeypatch):
    d = tmp_path / "loras"
    d.mkdir()
    (d / "mystyle.safetensors").write_bytes(b"weights")
    (d / "mystyle.json").write_text(json.dumps({"family": "sdxl", "source": "studio-trained"}))
    (d / "bare.safetensors").write_bytes(b"bare")
    monkeypatch.setattr(dl, "loras_dir", lambda: d)
    return d


def test_export_copies_weights_and_a_marked_sidecar(loras, tmp_path):
    out = dl.export_local_lora("mystyle", tmp_path / "out")
    assert out == tmp_path / "out" / "mystyle.safetensors"
    assert out.read_bytes() == b"weights"
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta == {"family": "sdxl", "source": "studio-trained", "kind": "diffusion-lora"}
    assert dl.is_image_lora_file(out)
    # The source stays in the catalog.
    assert (loras / "mystyle.safetensors").is_file()


def test_export_without_sidecar_still_writes_the_marker(loras, tmp_path):
    out = dl.export_local_lora("bare", tmp_path / "out")
    assert out.read_bytes() == b"bare"
    assert json.loads(out.with_suffix(".json").read_text()) == {"kind": "diffusion-lora"}


def test_export_into_the_catalog_itself_keeps_the_file(loras):
    out = dl.export_local_lora("mystyle", loras)
    assert out.read_bytes() == b"weights"
    assert json.loads(out.with_suffix(".json").read_text())["family"] == "sdxl"


def test_export_never_overwrites_other_files(loras, tmp_path):
    out_dir = tmp_path / "shared"
    out_dir.mkdir()
    (out_dir / "mystyle.safetensors").write_bytes(b"another install")
    (out_dir / "mystyle.json").write_text(json.dumps({"kind": "diffusion-lora"}))
    (out_dir / "bare.json").write_text('{"model_type": "llama"}')
    assert dl.export_local_lora("mystyle", out_dir) == out_dir / "mystyle-2.safetensors"
    assert (out_dir / "mystyle.safetensors").read_bytes() == b"another install"
    assert dl.export_local_lora("bare", out_dir) == out_dir / "bare-2.safetensors"
    assert json.loads((out_dir / "bare.json").read_text()) == {"model_type": "llama"}
    # Re-exporting the same bytes reuses its slot instead of piling up copies.
    assert dl.export_local_lora("mystyle", out_dir) == out_dir / "mystyle-2.safetensors"


def test_export_refuses_ids_outside_the_local_catalog(loras, tmp_path):
    secret = tmp_path / "secret.safetensors"
    secret.write_bytes(b"secret")
    for lora_id in ("missing", str(secret), "../secret", "krea/Krea-2-LoRA-retroanime"):
        with pytest.raises(FileNotFoundError):
            dl.export_local_lora(lora_id, tmp_path / "out")
    assert not (tmp_path / "out" / "secret.safetensors").exists()


def _export(lora_id, save_directory):
    request = ExportDiffusionLoRARequest(lora_id = lora_id, save_directory = str(save_directory))
    return asyncio.run(export_routes.export_diffusion_lora(request, "subject"))


def test_route_returns_the_saved_path(loras, tmp_path):
    response = _export("mystyle", tmp_path / "exported")
    assert response.success
    saved = tmp_path / "exported" / "mystyle.safetensors"
    assert saved.read_bytes() == b"weights"
    assert response.details == {"output_path": str(saved.resolve())}


def test_route_maps_an_unknown_lora_to_404(loras, tmp_path):
    with pytest.raises(HTTPException) as exc:
        _export("missing", tmp_path / "exported")
    assert exc.value.status_code == 404
