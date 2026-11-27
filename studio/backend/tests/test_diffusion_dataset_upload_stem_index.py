# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The upload's caption-stem check lists the dataset folder once per request, not once per media file."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import routes.training as training
from auth.authentication import get_current_subject


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buf, format = "PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(training.router, prefix = "/api/train")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


@pytest.fixture
def ds_root(monkeypatch, tmp_path):
    import utils.paths as up

    root = tmp_path / "assets" / "datasets"
    root.mkdir(parents = True)
    monkeypatch.setattr(up, "datasets_root", lambda: root)
    return root


def _upload(client, names):
    png = _png_bytes()
    parts = [("files", (n, png, "application/octet-stream")) for n in names]
    return client.post("/api/train/diffusion/dataset", data = {"name": "styleset"}, files = parts)


def _count_folder_listings(monkeypatch, folder: Path) -> list:
    calls = []
    real = Path.iterdir

    def counting(self):
        if self == folder:
            calls.append(1)
        return real(self)

    monkeypatch.setattr(Path, "iterdir", counting)
    return calls


def test_many_media_files_list_the_folder_once_before_commit(client, ds_root, monkeypatch):
    folder = ds_root / "styleset"
    folder.mkdir()
    for i in range(20):
        (folder / f"old_{i}.png").write_bytes(_png_bytes())
        (folder / f"old_{i}.txt").write_text("cap")
    calls = _count_folder_listings(monkeypatch, folder)
    # The last name clashes with an existing stem, so the request stops before writing anything.
    names = [f"new_{i}.png" for i in range(30)] + ["old_7.jpg"]
    r = _upload(client, names)
    assert r.status_code == 400
    assert "'old_7.png' is already in this dataset" in r.json()["detail"]
    assert len(calls) == 1
    assert sorted(p.name for p in folder.iterdir() if p.name.startswith("new_")) == []


def test_captions_only_batch_does_not_list_the_folder(client, ds_root, monkeypatch):
    folder = ds_root / "styleset"
    folder.mkdir()
    calls = _count_folder_listings(monkeypatch, folder)
    r = _upload(client, ["a.txt", "b.caption"])
    assert r.status_code == 200
    # The response summary lists the folder; the stem check adds nothing for caption files.
    assert len(calls) == 1


def test_existing_clash_reports_the_first_listed_match(client, ds_root):
    folder = ds_root / "styleset"
    folder.mkdir()
    (folder / "Photo.png").write_bytes(_png_bytes())
    r = _upload(client, ["photo.jpg"])
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail == (
        "Duplicate image name 'photo'. 'Photo.png' is already in this dataset; two files "
        "sharing a name would share one 'photo.txt' caption. Rename one before uploading."
    )


def test_casefold_equal_existing_name_with_other_stem_case_is_exempt(client, ds_root):
    folder = ds_root / "styleset"
    folder.mkdir()
    (folder / "Photo.PNG").write_bytes(_png_bytes())
    # Same name up to case: a replacement, not a sidecar clash, unless the stems match exactly.
    r = _upload(client, ["photo.png"])
    assert r.status_code == 200


def test_unicode_casefold_stem_clash(client, ds_root):
    folder = ds_root / "styleset"
    folder.mkdir()
    (folder / "STRASSE.webp").write_bytes(_png_bytes())
    r = _upload(client, ["straße.png"])
    assert r.status_code == 400
    assert "'STRASSE.webp' is already in this dataset" in r.json()["detail"]


def test_directory_and_appledouble_names_do_not_clash(client, ds_root):
    folder = ds_root / "styleset"
    folder.mkdir()
    (folder / "shot.png").mkdir()
    (folder / "._shot.png").write_bytes(b"\x00\x05\x16\x07")
    r = _upload(client, ["shot.jpg"])
    assert r.status_code == 200
    assert r.json()["uploaded"] == 1


def test_batch_clash_still_found_after_existing_index(client, ds_root):
    folder = ds_root / "styleset"
    folder.mkdir()
    (folder / "old.png").write_bytes(_png_bytes())
    r = _upload(client, ["x.png", "X.jpg"])
    assert r.status_code == 400
    assert "'x.png' is already in this dataset" in r.json()["detail"]
