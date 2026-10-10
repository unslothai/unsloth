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
import storage.studio_db as studio_db

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


# The registered-folder table itself (normalisation, denylist) is covered elsewhere; macOS tmp_path
# lives under /private/var, which registration refuses, so the rows are faked here.
_FOLDERS: list = []


def register(folder):
    _FOLDERS.append({"path": str(folder)})


@pytest.fixture(autouse = True)
def _folders(monkeypatch):
    _FOLDERS.clear()
    monkeypatch.setattr(studio_db, "list_scan_folders", lambda: list(_FOLDERS))


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    d = tmp_path / "loras"
    d.mkdir()
    monkeypatch.setattr(dl, "loras_dir", lambda: d)
    return d


def _local():
    return {e.id: e for e in dl.list_loras() if e.source == "local"}


def _by_path(path):
    return next(e for e in _local().values() if e.local_path == str(path))


def test_an_export_in_a_custom_folder_is_listed(catalog, tmp_path):
    (catalog / "mystyle.safetensors").write_bytes(b"w")
    (catalog / "mystyle.json").write_text(
        json.dumps({"family": "sdxl", "source": "studio-trained"})
    )
    folder = tmp_path / "my-models"
    dl.export_local_lora("mystyle", folder / "image-loras")
    (catalog / "mystyle.safetensors").unlink()
    (catalog / "mystyle.json").unlink()
    register(str(folder))

    entry = _by_path(folder / "image-loras" / "mystyle.safetensors")
    assert entry.display_name == "mystyle" and entry.id.startswith("mystyle-")
    assert entry.families == ("sdxl",) and entry.fine_tuned
    assert dl.resolve_one(entry.id, 1.0).path == entry.local_path


def test_unmarked_weights_in_a_custom_folder_are_ignored(catalog, tmp_path):
    folder = tmp_path / "my-models"
    _safetensors(folder / "model.safetensors")
    _safetensors(folder / "notes.safetensors", sidecar = {"family": "sdxl"})
    register(str(folder))
    assert _local() == {}


def test_custom_folder_ids_stay_stable_when_folders_change(catalog, tmp_path):
    (catalog / "style.safetensors").write_bytes(b"catalog")
    folders = [tmp_path / name for name in ("a", "b", "c")]
    for folder in folders:
        _safetensors(folder / "style.safetensors", sidecar = _MARK)
        register(str(folder))
    assert _local()["style"].local_path == str(catalog / "style.safetensors")
    ids = {str(f): _by_path(f / "style.safetensors").id for f in folders}
    assert len(set(ids.values())) == 3 and not _by_path(folders[2] / "style.safetensors").fine_tuned

    _FOLDERS.pop(0)
    for folder in folders[1:]:
        assert _by_path(folder / "style.safetensors").id == ids[str(folder)]
    assert ids[str(folders[0])] not in _local()


def test_the_catalog_registered_as_a_custom_folder_is_not_listed_twice(catalog, tmp_path):
    _safetensors(catalog / "style.safetensors", sidecar = _MARK)
    register(str(catalog))
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
    register(str(folder))
    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    monkeypatch.setattr(account_access, "model_visible", lambda reference, **_: False)
    assert _local() == {}
    with pytest.raises(FileNotFoundError):
        dl.export_local_lora("style", tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        dl.resolve_one("style", 1.0)
