# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the disk-backed image gallery: PNG-embedded recipe round-trips,
listing order, safe id handling, and delete/clear."""

from __future__ import annotations

import base64
import errno
import io
import os

import pytest

import core.inference.gallery_flags as gallery_flags
import core.inference.image_gallery as gallery

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402
import json as _json


@pytest.fixture(autouse = True)
def _tmp_gallery(monkeypatch, tmp_path):
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

    raw = base64.b64decode(gallery.image_b64(record["id"]))
    with Image.open(io.BytesIO(raw)) as im:
        assert im.text["unsloth"]
        assert "Negative prompt" not in im.text["parameters"]
        assert "Steps: 9" in im.text["parameters"]

    listed = gallery.list_images()
    assert len(listed) == 1
    assert listed[0]["prompt"] == "a sloth" and listed[0]["seed"] == 7


def test_listing_reads_the_recipe_without_decoding_pixels(monkeypatch):
    from PIL import PngImagePlugin

    for i in range(3):
        gallery.save(_img(), _meta(seed = i))
    loads = []
    real_load = PngImagePlugin.PngImageFile.load
    monkeypatch.setattr(
        PngImagePlugin.PngImageFile,
        "load",
        lambda self, *a, **k: loads.append(1) or real_load(self, *a, **k),
    )
    assert sorted(r["seed"] for r in gallery.list_images()) == [0, 1, 2]
    assert loads == []


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
    for i in range(5):
        _save_with_mtime(f"p{i}", float(i))
    page1 = gallery.list_images(limit = 2, offset = 0)
    page2 = gallery.list_images(limit = 2, offset = 2)
    assert [r["prompt"] for r in page1] == ["p4", "p3"]
    assert [r["prompt"] for r in page2] == ["p2", "p1"]
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
    assert gallery.delete(a["id"]) is False
    assert len(gallery.list_images()) == 1
    assert gallery.clear() == 1
    assert gallery.list_images() == []


@pytest.mark.parametrize("stage", ["read", "unlink"])
def test_delete_does_not_report_an_io_failure_as_a_missing_image(monkeypatch, stage):
    record = gallery.save(_img(), _meta())
    path = gallery.image_path(record["id"])

    def refuse(*args, **kwargs):
        raise PermissionError(errno.EACCES, "read-only gallery")

    if stage == "read":
        monkeypatch.setattr(Image, "open", refuse)
    else:
        monkeypatch.setattr(type(path), "unlink", refuse)
    with pytest.raises(PermissionError, match = "read-only gallery"):
        gallery.delete(record["id"])
    assert path.exists()


def test_clear_preserves_foreign_png():
    foreign = gallery.gallery_dir() / "family-photo.png"
    _img().save(foreign, format = "PNG")
    gallery.save(_img(), _meta(prompt = "ours"))
    assert gallery.clear() == 1
    assert foreign.exists()
    assert gallery.list_images() == []


def test_delete_ignores_foreign_png():
    foreign = gallery.gallery_dir() / "family-photo.png"
    _img().save(foreign, format = "PNG")
    assert gallery.delete("family-photo") is False
    assert foreign.exists()


def test_image_path_rejects_unsafe_ids():
    assert gallery.image_path("../../etc/passwd") is None
    assert gallery.image_path("a/b") is None
    assert gallery.image_path("missing") is None


def test_owned_image_path_serves_only_owned_pngs():
    # Resolvable via image_path, but owned_image_path applies the recipe check.
    foreign = gallery.gallery_dir() / "family-photo.png"
    _img().save(foreign, format = "PNG")
    assert gallery.image_path("family-photo") is not None
    assert gallery.owned_image_path("family-photo") is None

    ours = gallery.save(_img(), _meta(prompt = "ours"))
    assert gallery.owned_image_path(ours["id"]) is not None
    assert gallery.owned_image_path("../../etc/passwd") is None
    assert gallery.owned_image_path("missing") is None


def test_list_skips_foreign_pngs(tmp_path):
    foreign = gallery.gallery_dir() / "foreign.png"
    _img().save(foreign, format = "PNG")
    gallery.save(_img(), _meta(prompt = "ours"))
    listed = gallery.list_images()
    assert [r["prompt"] for r in listed] == ["ours"]


def test_foreign_png_in_window_does_not_drop_valid_images():
    # Paging counts readable records, not files.
    _save_with_mtime("p2", 100.0)
    foreign = gallery.gallery_dir() / "zzz_foreign.png"
    _img().save(foreign, format = "PNG")
    os.utime(foreign, (300.0, 300.0))
    _save_with_mtime("p1", 200.0)
    page1 = gallery.list_images(limit = 2, offset = 0)
    assert [r["prompt"] for r in page1] == ["p1", "p2"]


def test_list_skips_recipe_missing_required_fields(tmp_path):
    import json

    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text("unsloth", json.dumps({"prompt": "partial"}))
    _img().save(gallery.gallery_dir() / "partial.png", format = "PNG", pnginfo = info)
    gallery.save(_img(), _meta(prompt = "ours"))
    listed = gallery.list_images()
    assert [r["prompt"] for r in listed] == ["ours"]


def test_valid_callback_paginates_over_accepted_records():
    # `valid` must filter before pagination, else a bad leading record stalls scroll.
    _save_with_mtime("BAD", 300.0)  # newest, sorts first
    _save_with_mtime("g1", 200.0)
    _save_with_mtime("g2", 100.0)

    def _valid(rec):
        return rec.get("prompt") != "BAD"

    page = gallery.list_images(limit = 2, offset = 0, valid = _valid)
    assert [r["prompt"] for r in page] == ["g1", "g2"]
    assert len(gallery.list_images(limit = 3, offset = 0, valid = _valid)) == 2


def test_valid_callback_leading_bad_record_does_not_stall_at_offset_zero():
    for i in range(3):
        _save_with_mtime(f"BAD{i}", 300.0 - i)
    _save_with_mtime("good", 10.0)

    def _valid(rec):
        return not str(rec.get("prompt", "")).startswith("BAD")

    records = gallery.list_images(limit = 2, offset = 0, valid = _valid)
    assert [r["prompt"] for r in records] == ["good"]


def test_save_is_atomic_no_partial_png_on_publish_failure(monkeypatch):
    def _boom(*a, **k):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(gallery.os, "replace", _boom)
    with pytest.raises(OSError, match = "simulated rename failure"):
        gallery.save(_img(), _meta())
    assert list(gallery.gallery_dir().glob("*.png")) == []
    assert list(gallery.gallery_dir().iterdir()) == []


def test_records_carry_default_flags():
    _save_with_mtime("a", 100.0)
    record = gallery.list_images()[0]
    assert record["pinned"] is False and record["archived"] is False


def test_records_carry_the_listing_sort_key():
    a = gallery.save(_img(), _meta(created_at = 100.0))
    b = gallery.save(_img(), _meta(created_at = 100.0))
    os.utime(gallery.gallery_dir() / f"{a['id']}.png", (150.0, 150.0))
    os.utime(gallery.gallery_dir() / f"{b['id']}.png", (120.0, 120.0))
    assert [(r["id"], r["order_at"]) for r in gallery.list_images()] == [
        (a["id"], 150.0),
        (b["id"], 120.0),
    ]


def test_pinned_images_sort_ahead_of_newer_ones():
    old = _save_with_mtime("old", 100.0)
    _save_with_mtime("new", 200.0)
    gallery.set_flags(old["id"], pinned = True)
    assert [r["prompt"] for r in gallery.list_images()] == ["old", "new"]
    assert gallery.list_images()[0]["pinned"] is True


def test_most_recently_pinned_leads_the_pinned_group():
    first = _save_with_mtime("first", 100.0)
    second = _save_with_mtime("second", 200.0)
    gallery.set_flags(second["id"], pinned = True)
    gallery.set_flags(first["id"], pinned = True)
    assert [r["prompt"] for r in gallery.list_images()] == ["first", "second"]


def test_unpinning_returns_an_image_to_newest_first_order():
    old = _save_with_mtime("old", 100.0)
    _save_with_mtime("new", 200.0)
    gallery.set_flags(old["id"], pinned = True)
    gallery.set_flags(old["id"], pinned = False)
    assert [r["prompt"] for r in gallery.list_images()] == ["new", "old"]


def test_archived_images_leave_the_default_listing():
    keep = _save_with_mtime("keep", 100.0)
    shelved = _save_with_mtime("shelved", 200.0)
    gallery.set_flags(shelved["id"], archived = True)
    assert [r["id"] for r in gallery.list_images()] == [keep["id"]]
    archived = gallery.list_images(archived = True)
    assert [r["id"] for r in archived] == [shelved["id"]]
    assert archived[0]["archived"] is True


def test_restoring_puts_an_image_back_on_the_strip():
    record = _save_with_mtime("a", 100.0)
    gallery.set_flags(record["id"], archived = True)
    gallery.set_flags(record["id"], archived = False)
    assert [r["id"] for r in gallery.list_images()] == [record["id"]]


def test_archived_images_do_not_consume_a_page_slot():
    for i in range(4):
        record = _save_with_mtime(f"a{i}", 100.0 + i)
        if i % 2 == 0:
            gallery.set_flags(record["id"], archived = True)
    assert [r["prompt"] for r in gallery.list_images(limit = 2)] == ["a3", "a1"]
    assert len(gallery.list_images(limit = 3)) == 2
    assert [r["prompt"] for r in gallery.list_images(archived = True)] == ["a2", "a0"]


def test_pinning_survives_pagination():
    oldest = _save_with_mtime("oldest", 100.0)
    for i in range(1, 4):
        _save_with_mtime(f"a{i}", 100.0 + i)
    gallery.set_flags(oldest["id"], pinned = True)
    assert gallery.list_images(limit = 1, offset = 0)[0]["prompt"] == "oldest"


def test_set_flags_refuses_a_foreign_or_unknown_id():
    assert gallery.set_flags("does-not-exist", pinned = True) is None
    foreign = gallery.gallery_dir() / "foreign.png"
    _img().save(foreign, format = "PNG")
    assert gallery.set_flags("foreign", pinned = True) is None


@pytest.mark.parametrize("missing_at", [None, "lookup", "read", "unlink"])
def test_delete_prunes_the_flag_entry(monkeypatch, missing_at):
    record = _save_with_mtime("a", 100.0)
    other = _save_with_mtime("keep", 200.0)
    gallery.set_flags(record["id"], pinned = True, archived = True)
    gallery.set_flags(other["id"], archived = True)
    path = gallery.image_path(record["id"])
    if missing_at == "lookup":
        path.unlink()
    elif missing_at == "read":
        read_meta = gallery._read_meta

        def disappear_before_read(candidate, **kwargs):
            candidate.unlink()
            return read_meta(candidate, **kwargs)

        monkeypatch.setattr(gallery, "_read_meta", disappear_before_read)
    elif missing_at == "unlink":
        unlink = type(path).unlink

        def disappear_before_unlink(candidate, **kwargs):
            if candidate == path:
                unlink(candidate)
            return unlink(candidate, **kwargs)

        monkeypatch.setattr(type(path), "unlink", disappear_before_unlink)
    assert gallery.delete(record["id"]) is (missing_at is None)
    assert gallery_flags.read(gallery.gallery_dir()) == {other["id"]: {"archived": True}}
    assert gallery.image_path(other["id"]) is not None


def test_clear_spares_archived_images():
    active = _save_with_mtime("active", 100.0)
    shelved = _save_with_mtime("shelved", 200.0)
    gallery.set_flags(shelved["id"], archived = True)
    assert gallery.clear() == 1
    assert [r["id"] for r in gallery.list_images(archived = True)] == [shelved["id"]]
    assert set(gallery_flags.read(gallery.gallery_dir())) == {shelved["id"]}
    assert gallery.image_path(active["id"]) is None


def test_clear_can_include_archived_images():
    record = _save_with_mtime("shelved", 100.0)
    gallery.set_flags(record["id"], archived = True)
    assert gallery.clear(include_archived = True) == 1
    assert gallery.list_images(archived = True) == []
    assert gallery_flags.read(gallery.gallery_dir()) == {}


def test_flags_are_not_required_recipe_keys():
    record = _save_with_mtime("older-schema", 100.0)
    assert "pinned" not in gallery._read_meta(gallery.image_path(record["id"]))
    assert [r["id"] for r in gallery.list_images()] == [record["id"]]


def test_clear_refuses_when_the_flag_store_cannot_be_read():
    # Fail closed: an unreadable store reads as 'nothing archived', which would delete the archive.
    record = _save_with_mtime("shelved", 100.0)
    gallery.set_flags(record["id"], archived = True)
    (gallery.gallery_dir() / ".flags.json").write_text("corrupt", encoding = "utf-8")
    with pytest.raises(gallery_flags.FlagsUnavailable):
        gallery.clear()
    assert gallery.image_path(record["id"]) is not None


def test_clear_all_still_works_with_an_unreadable_store():
    record = _save_with_mtime("a", 100.0)
    (gallery.gallery_dir() / ".flags.json").write_text("corrupt", encoding = "utf-8")
    assert gallery.clear(include_archived = True) == 1
    assert gallery.image_path(record["id"]) is None


def test_clear_refuses_when_a_single_flag_entry_is_malformed():
    record = _save_with_mtime("shelved", 100.0)
    gallery.set_flags(record["id"], archived = True)
    (gallery.gallery_dir() / ".flags.json").write_text(
        _json.dumps({"version": 1, "items": {record["id"]: "hand edited"}}), encoding = "utf-8"
    )
    with pytest.raises(gallery_flags.FlagsUnavailable):
        gallery.clear()
    assert gallery.image_path(record["id"]) is not None


def test_clear_refuses_when_an_archived_flag_is_not_a_boolean():
    # `{"archived": null}` is still a dict but reads as 'not archived'.

    record = _save_with_mtime("shelved", 100.0)
    gallery.set_flags(record["id"], archived = True)
    (gallery.gallery_dir() / ".flags.json").write_text(
        _json.dumps({"version": 1, "items": {record["id"]: {"archived": None}}}), encoding = "utf-8"
    )
    with pytest.raises(gallery_flags.FlagsUnavailable):
        gallery.clear()
    assert gallery.image_path(record["id"]) is not None


def test_a_repair_never_makes_a_damaged_archive_deletable():
    shelved = _save_with_mtime("shelved", 100.0)
    other = _save_with_mtime("other", 200.0)
    gallery.set_flags(shelved["id"], archived = True)
    (gallery.gallery_dir() / ".flags.json").write_text(
        _json.dumps({"version": 1, "items": {shelved["id"]: {"archived": None}}}),
        encoding = "utf-8",
    )
    gallery.set_flags(other["id"], pinned = True)
    assert gallery.clear() == 1
    assert gallery.image_path(shelved["id"]) is not None
    assert gallery.image_path(other["id"]) is None


def test_a_repair_of_an_illegible_store_never_makes_the_archive_deletable():
    shelved = _save_with_mtime("shelved", 100.0)
    other = _save_with_mtime("other", 200.0)
    gallery.set_flags(shelved["id"], archived = True)
    (gallery.gallery_dir() / ".flags.json").write_text(
        '{"version": 1, "items": {"shel', encoding = "utf-8"
    )
    gallery.set_flags(other["id"], pinned = True)
    with pytest.raises(gallery_flags.FlagsUnavailable):
        gallery.clear()
    assert gallery.image_path(shelved["id"]) is not None
    assert gallery.image_path(other["id"]) is not None
    assert gallery.clear(include_archived = True) == 2


def test_clear_all_replaces_an_unreadable_store_so_the_gallery_recovers():
    _save_with_mtime("a", 100.0)
    _save_with_mtime("b", 200.0)
    (gallery.gallery_dir() / ".flags.json").write_text(
        '{"version": 1, "items": {"a": {"archi', encoding = "utf-8"
    )
    assert gallery.clear(include_archived = True) == 2
    later = _save_with_mtime("c", 300.0)
    assert gallery.clear() == 1
    assert gallery.image_path(later["id"]) is None


def test_archiving_during_a_clear_never_leaves_a_deleted_image_reported_as_archived():
    # Without a shared lock an archive racing clear() could report success then be deleted.
    import threading

    records = [_save_with_mtime(f"i{i}", float(i)) for i in range(30)]
    target = records[15]["id"]
    out = {}

    def _archive():
        out["result"] = gallery.set_flags(target, archived = True)

    worker = threading.Thread(target = _archive)
    worker.start()
    gallery.clear()
    worker.join(timeout = 10)

    said_ok = out.get("result") is not None
    survived = gallery.image_path(target) is not None
    assert said_ok == survived


def test_save_looks_up_the_folder_after_encoding(tmp_path, monkeypatch):
    order = []
    encode = gallery._png_bytes
    monkeypatch.setattr(gallery, "_png_bytes", lambda *a: order.append("encode") or encode(*a))
    monkeypatch.setattr(gallery, "gallery_dir", lambda: order.append("dir") or tmp_path)
    gallery.save(Image.new("RGB", (4, 4)), {"prompt": "p"})
    assert order[:2] == ["encode", "dir"]


def test_thumbnail_is_a_downscaled_webp_of_the_png():
    record = gallery.save(Image.new("RGB", (64, 32), (200, 10, 10)), _meta())
    data = gallery.thumbnail(gallery.image_path(record["id"]), 16)
    with Image.open(io.BytesIO(data)) as im:
        assert im.format == "WEBP"
        assert im.size == (16, 8)


def _noisy(size = 128):
    import random
    rng = random.Random(1234)
    return Image.frombytes(
        "RGB", (size, size), bytes(rng.randrange(256) for _ in range(size * size * 3))
    )


def _encode(image, level):
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text("unsloth", _json.dumps(_meta()))
    info.add_text("parameters", gallery._params_text(_meta()))
    buf = io.BytesIO()
    image.save(buf, format = "PNG", pnginfo = info, compress_level = level)
    return buf.getvalue()


def test_gallery_png_uses_fast_deflate_by_default(monkeypatch):
    # The encode is on the request path, so the default level must be fast.
    monkeypatch.delenv(gallery.PNG_COMPRESS_LEVEL_ENV, raising = False)
    image = _noisy()
    data = gallery._png_bytes(image, _meta())
    assert data == _encode(image, 1)
    assert data != _encode(image, 6)


def test_gallery_png_level_kill_switch_restores_pillow_default(monkeypatch):
    monkeypatch.setenv(gallery.PNG_COMPRESS_LEVEL_ENV, "6")
    image = _noisy()
    assert gallery._png_bytes(image, _meta()) == _encode(image, 6)


@pytest.mark.parametrize("raw", ["", "fast", "-1", "10"])
def test_gallery_png_level_rejects_invalid_values(monkeypatch, raw):
    monkeypatch.setenv(gallery.PNG_COMPRESS_LEVEL_ENV, raw)
    assert gallery.png_compress_level() == 1


@pytest.mark.parametrize("level", ["0", "1", "6", "9"])
def test_gallery_png_is_lossless_at_any_level(monkeypatch, level):
    monkeypatch.setenv(gallery.PNG_COMPRESS_LEVEL_ENV, level)
    image = _noisy(64)
    record = gallery.save(image, _meta())
    raw = base64.b64decode(gallery.image_b64(record["id"]))
    with Image.open(io.BytesIO(raw)) as im:
        assert im.convert("RGB").tobytes() == image.tobytes()
        assert _json.loads(im.text["unsloth"])["seed"] == 7
