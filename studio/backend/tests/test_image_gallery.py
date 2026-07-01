# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the disk-backed image gallery: PNG-embedded recipe round-trips,
listing order, safe id handling, and delete/clear."""

from __future__ import annotations

import base64
import io
import os

import pytest

import core.inference.image_gallery as gallery

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


@pytest.fixture(autouse = True)
def _tmp_gallery(monkeypatch, tmp_path):
    # Point the gallery at a throwaway root instead of ~/.unsloth/studio.
    monkeypatch.setattr(gallery, "studio_root", lambda: tmp_path)


def _img(color = (10, 20, 30)):
    return Image.new("RGB", (16, 16), color)


def _meta(**over):
    base = {
        "prompt": "a sloth",
        "negative_prompt": None,
        "width": 1024,
        "height": 1024,
        "steps": 9,
        "guidance": 0.0,
        "seed": 7,
        "model": "unsloth/Z-Image-Turbo-GGUF",
        "created_at": 100.0,
    }
    base.update(over)
    return base


def test_save_embeds_recipe_and_round_trips():
    record = gallery.save(_img(), _meta())
    assert record["id"] and record["url"].endswith(f"{record['id']}/file")

    # The recipe is embedded in the PNG itself (portable), not just in a sidecar.
    raw = base64.b64decode(gallery.image_b64(record["id"]))
    with Image.open(io.BytesIO(raw)) as im:
        assert im.text["unsloth"]
        assert "Negative prompt" not in im.text["parameters"]  # none given
        assert "Steps: 9" in im.text["parameters"]

    listed = gallery.list_images()
    assert len(listed) == 1
    assert listed[0]["prompt"] == "a sloth" and listed[0]["seed"] == 7


def _save_with_mtime(prompt: str, t: float) -> dict:
    record = gallery.save(_img(), _meta(prompt = prompt, created_at = t))
    # Listing orders by mtime; set it explicitly so a tight test loop can't tie it.
    os.utime(gallery.gallery_dir() / f"{record['id']}.png", (t, t))
    return record


def test_list_is_newest_first():
    old = _save_with_mtime("old", 100.0)
    new = _save_with_mtime("new", 200.0)
    assert [r["id"] for r in gallery.list_images()] == [new["id"], old["id"]]


def test_list_paginates_with_limit_offset():
    # 5 images, newest (t=4) first.
    for i in range(5):
        _save_with_mtime(f"p{i}", float(i))
    page1 = gallery.list_images(limit = 2, offset = 0)
    page2 = gallery.list_images(limit = 2, offset = 2)
    assert [r["prompt"] for r in page1] == ["p4", "p3"]
    assert [r["prompt"] for r in page2] == ["p2", "p1"]
    # limit=None still returns everything from the offset.
    assert len(gallery.list_images()) == 5
    assert len(gallery.list_images(offset = 4)) == 1


def test_negative_prompt_recorded_in_parameters():
    record = gallery.save(_img(), _meta(negative_prompt = "blurry"))
    raw = base64.b64decode(gallery.image_b64(record["id"]))
    with Image.open(io.BytesIO(raw)) as im:
        assert "Negative prompt: blurry" in im.text["parameters"]


def test_delete_and_clear():
    a = gallery.save(_img(), _meta(prompt = "a"))
    gallery.save(_img(), _meta(prompt = "b"))
    assert gallery.delete(a["id"]) is True
    assert gallery.delete(a["id"]) is False  # already gone
    assert len(gallery.list_images()) == 1
    assert gallery.clear() == 1
    assert gallery.list_images() == []


def test_image_path_rejects_unsafe_ids():
    # Traversal / bad chars never resolve to a path.
    assert gallery.image_path("../../etc/passwd") is None
    assert gallery.image_path("a/b") is None
    assert gallery.image_path("missing") is None


def test_list_skips_foreign_pngs(tmp_path):
    # A PNG without our recipe chunk (user dropped a file) is ignored.
    foreign = gallery.gallery_dir() / "foreign.png"
    _img().save(foreign, format = "PNG")
    gallery.save(_img(), _meta(prompt = "ours"))
    listed = gallery.list_images()
    assert [r["prompt"] for r in listed] == ["ours"]


def test_foreign_png_in_window_does_not_drop_valid_images():
    # A foreign PNG sorting INTO the requested page must not consume a window slot and
    # drop a valid image that sorts after it: paging is over readable records, not files.
    _save_with_mtime("p2", 100.0)
    foreign = gallery.gallery_dir() / "zzz_foreign.png"
    _img().save(foreign, format = "PNG")  # newest by mtime (set below), sorts first
    os.utime(foreign, (300.0, 300.0))
    _save_with_mtime("p1", 200.0)
    # First page of 2 must still return both real images, not [p1] (foreign eating a slot).
    page1 = gallery.list_images(limit = 2, offset = 0)
    assert [r["prompt"] for r in page1] == ["p1", "p2"]


def test_list_skips_recipe_missing_required_fields(tmp_path):
    # A PNG carrying our chunk but an incomplete/older-schema recipe (no seed etc.)
    # must be skipped, not crash the whole listing when the route builds GalleryImage.
    import json

    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text("unsloth", json.dumps({"prompt": "partial"}))  # missing width/seed/...
    _img().save(gallery.gallery_dir() / "partial.png", format = "PNG", pnginfo = info)
    gallery.save(_img(), _meta(prompt = "ours"))
    listed = gallery.list_images()
    assert [r["prompt"] for r in listed] == ["ours"]
