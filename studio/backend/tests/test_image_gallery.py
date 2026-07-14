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


def test_clear_preserves_foreign_png():
    # A hand-dropped PNG with no recipe chunk is invisible to list_images; clear must not destroy it.
    foreign = gallery.gallery_dir() / "family-photo.png"
    _img().save(foreign, format = "PNG")
    gallery.save(_img(), _meta(prompt = "ours"))
    assert gallery.clear() == 1
    assert foreign.exists()
    assert gallery.list_images() == []


def test_delete_ignores_foreign_png():
    # A per-id delete must refuse a file we do not own (no readable recipe chunk).
    foreign = gallery.gallery_dir() / "family-photo.png"
    _img().save(foreign, format = "PNG")
    assert gallery.delete("family-photo") is False
    assert foreign.exists()


def test_image_path_rejects_unsafe_ids():
    # Traversal / bad chars never resolve to a path.
    assert gallery.image_path("../../etc/passwd") is None
    assert gallery.image_path("a/b") is None
    assert gallery.image_path("missing") is None


def test_owned_image_path_serves_only_owned_pngs():
    # A hand-dropped foreign PNG resolves via image_path (safe stem, on disk) but must NOT be
    # served: owned_image_path applies the same recipe check as delete/clear, so the serve route
    # can't stream a file the listing hides.
    foreign = gallery.gallery_dir() / "family-photo.png"
    _img().save(foreign, format = "PNG")
    assert gallery.image_path("family-photo") is not None  # resolvable...
    assert gallery.owned_image_path("family-photo") is None  # ...but not ours to serve

    ours = gallery.save(_img(), _meta(prompt = "ours"))
    assert gallery.owned_image_path(ours["id"]) is not None
    # Unsafe / missing ids resolve to nothing, like image_path.
    assert gallery.owned_image_path("../../etc/passwd") is None
    assert gallery.owned_image_path("missing") is None


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


def test_valid_callback_paginates_over_accepted_records():
    # ``valid`` must filter before pagination, so offset/limit/has_more count over the accepted
    # domain; else a leading bad record returns a short page with more remaining and stalls scroll.
    _save_with_mtime("BAD", 300.0)  # newest, sorts first
    _save_with_mtime("g1", 200.0)
    _save_with_mtime("g2", 100.0)

    def _valid(rec):
        return rec.get("prompt") != "BAD"

    # First page of 2 returns both good records, not [g1] or [].
    page = gallery.list_images(limit = 2, offset = 0, valid = _valid)
    assert [r["prompt"] for r in page] == ["g1", "g2"]
    # The has_more probe (limit + 1) sees no extra VALID record beyond the two returned.
    assert len(gallery.list_images(limit = 3, offset = 0, valid = _valid)) == 2


def test_valid_callback_leading_bad_record_does_not_stall_at_offset_zero():
    # Every record in the first window is invalid; without in-pager filtering the route stalled.
    for i in range(3):
        _save_with_mtime(f"BAD{i}", 300.0 - i)  # newest three are all invalid
    _save_with_mtime("good", 10.0)

    def _valid(rec):
        return not str(rec.get("prompt", "")).startswith("BAD")

    # The pager must look past the invalid leaders and return the one good record.
    records = gallery.list_images(limit = 2, offset = 0, valid = _valid)
    assert [r["prompt"] for r in records] == ["good"]


def test_save_is_atomic_no_partial_png_on_publish_failure(monkeypatch):
    # A crash before publishing must leave neither a truncated {id}.png nor a leftover temp.
    def _boom(*a, **k):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(gallery.os, "replace", _boom)
    with pytest.raises(OSError, match = "simulated rename failure"):
        gallery.save(_img(), _meta())
    # No final PNG surfaced, and the hidden temp was cleaned up.
    assert list(gallery.gallery_dir().glob("*.png")) == []
    assert list(gallery.gallery_dir().iterdir()) == []
