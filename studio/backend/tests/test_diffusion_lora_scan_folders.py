# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An exported image LoRA dropped into a custom models folder shows up in the Images LoRA picker, and
never as a model in the model pickers."""

from __future__ import annotations

import asyncio
import json
import struct

import pytest

from core.inference import diffusion_lora as dl
from storage.studio_db import add_scan_folder_with_status

_MARK = {"kind": "diffusion-lora"}


def _safetensors(path, sidecar = None):
    path.parent.mkdir(parents = True, exist_ok = True)
    header = json.dumps(
        {"unet.down.0.lora_A.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")
    if sidecar is not None:
        path.with_suffix(".json").write_text(json.dumps(sidecar))
    return path


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    d = tmp_path / "loras"
    d.mkdir()
    monkeypatch.setattr(dl, "loras_dir", lambda: d)
    return d


def _local():
    return {e.id: e for e in dl.list_loras() if e.source == "local"}


def test_an_export_in_a_custom_folder_is_listed(catalog, tmp_path):
    (catalog / "mystyle.safetensors").write_bytes(b"w")
    (catalog / "mystyle.json").write_text(
        json.dumps({"family": "sdxl", "source": "studio-trained"})
    )
    folder = tmp_path / "my-models"
    dl.export_local_lora("mystyle", folder / "image-loras")
    (catalog / "mystyle.safetensors").unlink()
    (catalog / "mystyle.json").unlink()
    add_scan_folder_with_status(str(folder))

    entry = _local()["mystyle"]
    assert entry.local_path == str(folder / "image-loras" / "mystyle.safetensors")
    assert entry.families == ("sdxl",) and entry.fine_tuned
    assert dl.resolve_one("mystyle", 1.0).path == entry.local_path


def test_unmarked_weights_in_a_custom_folder_are_ignored(catalog, tmp_path):
    folder = tmp_path / "my-models"
    _safetensors(folder / "model.safetensors")
    _safetensors(folder / "notes.safetensors", sidecar = {"family": "sdxl"})
    add_scan_folder_with_status(str(folder))
    assert _local() == {}


def test_a_name_taken_in_the_catalog_gets_a_suffix(catalog, tmp_path):
    (catalog / "style.safetensors").write_bytes(b"catalog")
    folder = tmp_path / "my-models"
    _safetensors(folder / "style.safetensors", sidecar = _MARK)
    add_scan_folder_with_status(str(folder))
    local = _local()
    assert local["style"].local_path == str(catalog / "style.safetensors")
    assert local["style-2"].local_path == str(folder / "style.safetensors")
    assert not local["style-2"].fine_tuned


def test_the_catalog_registered_as_a_custom_folder_is_not_listed_twice(catalog, tmp_path):
    _safetensors(catalog / "style.safetensors", sidecar = _MARK)
    add_scan_folder_with_status(str(catalog))
    assert list(_local()) == ["style"]


@pytest.mark.parametrize("scanner", ["routes", "inventory"])
def test_marked_loras_are_not_listed_as_models(tmp_path, scanner):
    if scanner == "routes":
        from routes.models import _scan_models_dir
    else:
        from hub.services.models.local_inventory import _scan_models_dir

    folder = tmp_path / "image-loras"
    _safetensors(folder / "a.safetensors", sidecar = _MARK)
    _safetensors(folder / "b.safetensors", sidecar = {"source": "studio-trained"})
    assert _scan_models_dir(folder) == []
    assert _scan_models_dir(tmp_path) == []

    _safetensors(tmp_path / "checkpoint" / "model.safetensors")
    assert [row.display_name for row in _scan_models_dir(tmp_path)] == ["checkpoint"]


def test_the_picker_route_reports_fine_tuned(catalog):
    from routes.models import scan_diffusion_loras

    _safetensors(catalog / "trained.safetensors", sidecar = {"source": "studio-trained", **_MARK})
    _safetensors(catalog / "downloaded.safetensors")
    rows = asyncio.run(scan_diffusion_loras(family = None, current_subject = "subject"))["loras"]
    flags = {row["id"]: row["fine_tuned"] for row in rows if row["source"] == "local"}
    assert flags == {"trained": True, "downloaded": False}


@pytest.mark.parametrize("scanner", ["routes", "inventory"])
def test_a_marked_gguf_in_a_subfolder_is_not_a_model(tmp_path, scanner):
    if scanner == "routes":
        from routes.models import _scan_models_dir
    else:
        from hub.services.models.local_inventory import _scan_models_dir

    folder = tmp_path / "image-loras"
    folder.mkdir()
    (folder / "style.gguf").write_bytes(b"GGUF" + b"\0" * 60)
    (folder / "style.json").write_text(json.dumps(_MARK))
    assert _scan_models_dir(tmp_path) == []


def test_a_managed_account_only_sees_custom_folder_loras_it_may_access(
    catalog, tmp_path, monkeypatch
):
    from hub.services.models import account_access

    folder = tmp_path / "shared"
    _safetensors(folder / "style.safetensors", sidecar = _MARK)
    add_scan_folder_with_status(str(folder))
    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    monkeypatch.setattr(account_access, "model_visible", lambda reference, **_: False)
    assert _local() == {}
    with pytest.raises(FileNotFoundError):
        dl.export_local_lora("style", tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        dl.resolve_one("style", 1.0)
