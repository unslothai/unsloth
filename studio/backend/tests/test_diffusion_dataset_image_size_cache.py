# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The labeling grid listing reopens an image header only once the file changed."""

from __future__ import annotations

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import routes.training as training
from auth.authentication import get_current_subject


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(training.router, prefix = "/api/train")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


@pytest.fixture
def folder(monkeypatch, tmp_path):
    import utils.paths as up

    root = tmp_path / "datasets"
    root.mkdir()
    monkeypatch.setattr(up, "datasets_root", lambda: root)
    monkeypatch.setattr(training, "_image_size_cache", type(training._image_size_cache)())
    ds = root / "grid"
    ds.mkdir()
    return ds


@pytest.fixture
def opens(monkeypatch):
    seen: list[str] = []
    real = Image.open

    def counting(fp, *args, **kwargs):
        seen.append(os.path.basename(str(fp)))
        return real(fp, *args, **kwargs)

    monkeypatch.setattr(Image, "open", counting)
    return seen


def _dims(client) -> dict[str, tuple[int, int]]:
    r = client.get("/api/train/diffusion/dataset/grid/images")
    assert r.status_code == 200, r.text
    return {rec["filename"]: (rec["width"], rec["height"]) for rec in r.json()["images"]}


def test_unchanged_images_are_not_reopened(client, folder, opens):
    Image.new("RGB", (8, 6)).save(folder / "a.png")
    Image.new("RGB", (5, 7)).save(folder / "b.jpg")
    assert _dims(client) == {"a.png": (8, 6), "b.jpg": (5, 7)}
    assert sorted(opens) == ["a.png", "b.jpg"]
    opens.clear()
    assert _dims(client) == {"a.png": (8, 6), "b.jpg": (5, 7)}
    assert opens == []


def test_in_place_rewrite_is_reread(client, folder):
    path = folder / "a.png"
    Image.new("RGB", (8, 6)).save(path)
    st = path.stat()
    assert _dims(client) == {"a.png": (8, 6)}
    Image.new("RGB", (6, 8)).save(path)
    # A normal in-place save moves the mtime; pinned forward so a coarse clock cannot hide it.
    os.utime(path, ns = (st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert _dims(client) == {"a.png": (6, 8)}


@pytest.mark.skipif(
    sys.platform == "win32", reason = "st_ctime is the creation time on Windows, not the change time"
)
def test_rewrite_with_the_old_mtime_is_reread(client, folder, opens):
    path = folder / "a.png"
    Image.new("RGB", (8, 6)).save(path)
    st = path.stat()
    assert _dims(client) == {"a.png": (8, 6)}
    Image.new("RGB", (6, 8)).save(path)
    assert path.stat().st_size == st.st_size
    os.utime(path, ns = (st.st_atime_ns, st.st_mtime_ns))
    assert _dims(client) == {"a.png": (6, 8)}


def test_replaced_file_is_reread(client, folder):
    path = folder / "a.png"
    Image.new("RGB", (8, 6)).save(path)
    assert _dims(client) == {"a.png": (8, 6)}
    tmp = folder / "staged.bin"
    Image.new("RGB", (3, 4)).save(tmp, format = "PNG")
    os.replace(tmp, path)
    assert _dims(client) == {"a.png": (3, 4)}


def test_unreadable_image_is_retried_every_listing(client, folder, opens):
    (folder / "broken.png").write_bytes(b"not a png")
    assert _dims(client) == {"broken.png": (0, 0)}
    assert _dims(client) == {"broken.png": (0, 0)}
    assert opens == ["broken.png", "broken.png"]
    assert str(folder / "broken.png") not in training._image_size_cache


def test_clips_never_open_pillow(client, folder, opens):
    (folder / "clip.mp4").write_bytes(b"\x00" * 16)
    r = client.get("/api/train/diffusion/dataset/grid/images")
    assert r.json()["images"][0]["kind"] == "clip"
    assert opens == []


def test_cache_is_bounded(client, folder, monkeypatch, opens):
    monkeypatch.setattr(training, "_IMAGE_SIZE_CACHE_MAX", 2)
    for name in ("a.png", "b.png", "c.png"):
        Image.new("RGB", (4, 4)).save(folder / name)
    _dims(client)
    assert len(training._image_size_cache) == 2
    opens.clear()
    # Sorted listing: a.png was evicted by c.png and is the only one reopened.
    _dims(client)
    assert opens.count("a.png") == 1


def test_stat_failure_still_reads_the_header(folder, monkeypatch, opens):
    path = folder / "a.png"
    Image.new("RGB", (8, 6)).save(path)
    assert training._dataset_image_size(path, None) == (8, 6)
    assert training._dataset_image_size(path, None) == (8, 6)
    assert opens == ["a.png", "a.png"]
    assert len(training._image_size_cache) == 0
