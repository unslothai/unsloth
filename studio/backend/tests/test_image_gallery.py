# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the disk-backed image gallery: PNG-embedded recipe round-trips,
listing order, safe id handling, and delete/clear."""

from __future__ import annotations

import base64
import io
import os

import pytest

import core.inference.gallery_flags as gallery_flags
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
    # A hand-dropped foreign PNG resolves via image_path (safe stem, on disk) but must NOT be served: owned_image_path applies the same recipe check as delete/clear.
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
    # A foreign PNG sorting INTO the requested page must not consume a window slot: paging is over readable records, not files.
    _save_with_mtime("p2", 100.0)
    foreign = gallery.gallery_dir() / "zzz_foreign.png"
    _img().save(foreign, format = "PNG")  # newest by mtime (set below), sorts first
    os.utime(foreign, (300.0, 300.0))
    _save_with_mtime("p1", 200.0)
    # First page of 2 must still return both real images, not [p1].
    page1 = gallery.list_images(limit = 2, offset = 0)
    assert [r["prompt"] for r in page1] == ["p1", "p2"]


def test_list_skips_recipe_missing_required_fields(tmp_path):
    # A PNG carrying our chunk but an incomplete/older-schema recipe must be skipped, not crash the listing when the route builds GalleryImage.
    import json

    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text("unsloth", json.dumps({"prompt": "partial"}))  # missing width/seed/...
    _img().save(gallery.gallery_dir() / "partial.png", format = "PNG", pnginfo = info)
    gallery.save(_img(), _meta(prompt = "ours"))
    listed = gallery.list_images()
    assert [r["prompt"] for r in listed] == ["ours"]


def test_valid_callback_paginates_over_accepted_records():
    # ``valid`` must filter before pagination, else a leading bad record returns a short page with more remaining and stalls scroll.
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


# --- pin / archive flags ---------------------------------------------------------------------


def test_records_carry_default_flags():
    _save_with_mtime("a", 100.0)
    record = gallery.list_images()[0]
    assert record["pinned"] is False and record["archived"] is False


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
    gallery.set_flags(first["id"], pinned = True)  # pinned later, so it leads
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
    # The archived shelf is its own listing, not a superset of the active one.
    archived = gallery.list_images(archived = True)
    assert [r["id"] for r in archived] == [shelved["id"]]
    assert archived[0]["archived"] is True


def test_restoring_puts_an_image_back_on_the_strip():
    record = _save_with_mtime("a", 100.0)
    gallery.set_flags(record["id"], archived = True)
    gallery.set_flags(record["id"], archived = False)
    assert [r["id"] for r in gallery.list_images()] == [record["id"]]


def test_archived_images_do_not_consume_a_page_slot():
    # Pagination counts over the shelf being listed, so has_more probes stay truthful.
    for i in range(4):
        record = _save_with_mtime(f"a{i}", 100.0 + i)
        if i % 2 == 0:
            gallery.set_flags(record["id"], archived = True)
    assert [r["prompt"] for r in gallery.list_images(limit = 2)] == ["a3", "a1"]
    # Only two active records exist, so a limit+1 probe must not invent a third.
    assert len(gallery.list_images(limit = 3)) == 2
    assert [r["prompt"] for r in gallery.list_images(archived = True)] == ["a2", "a0"]


def test_pinning_survives_pagination():
    oldest = _save_with_mtime("oldest", 100.0)
    for i in range(1, 4):
        _save_with_mtime(f"a{i}", 100.0 + i)
    gallery.set_flags(oldest["id"], pinned = True)
    # The pin must reach page 0 rather than waiting for the page its mtime belongs to.
    assert gallery.list_images(limit = 1, offset = 0)[0]["prompt"] == "oldest"


def test_set_flags_refuses_a_foreign_or_unknown_id():
    assert gallery.set_flags("does-not-exist", pinned = True) is None
    foreign = gallery.gallery_dir() / "foreign.png"
    _img().save(foreign, format = "PNG")  # no recipe chunk, so not ours
    assert gallery.set_flags("foreign", pinned = True) is None


def test_delete_prunes_the_flag_entry():
    record = _save_with_mtime("a", 100.0)
    gallery.set_flags(record["id"], pinned = True)
    assert gallery.delete(record["id"]) is True
    assert gallery_flags.read(gallery.gallery_dir()) == {}


def test_clear_spares_archived_images():
    active = _save_with_mtime("active", 100.0)
    shelved = _save_with_mtime("shelved", 200.0)
    gallery.set_flags(shelved["id"], archived = True)
    assert gallery.clear() == 1
    assert [r["id"] for r in gallery.list_images(archived = True)] == [shelved["id"]]
    # The cleared image's flags go with it; the archived one keeps its own.
    assert set(gallery_flags.read(gallery.gallery_dir())) == {shelved["id"]}
    assert gallery.image_path(active["id"]) is None


def test_clear_can_include_archived_images():
    record = _save_with_mtime("shelved", 100.0)
    gallery.set_flags(record["id"], archived = True)
    assert gallery.clear(include_archived = True) == 1
    assert gallery.list_images(archived = True) == []
    assert gallery_flags.read(gallery.gallery_dir()) == {}


def test_flags_are_not_required_recipe_keys():
    # Flags live beside the PNG, so an image written before they existed must still list.
    record = _save_with_mtime("older-schema", 100.0)
    assert "pinned" not in gallery._read_meta(gallery.image_path(record["id"]))
    assert [r["id"] for r in gallery.list_images()] == [record["id"]]


def test_clear_refuses_when_the_flag_store_cannot_be_read():
    # Fail CLOSED: an unreadable store reads as "nothing archived", which would delete the archive.
    record = _save_with_mtime("shelved", 100.0)
    gallery.set_flags(record["id"], archived = True)
    (gallery.gallery_dir() / ".flags.json").write_text("corrupt", encoding = "utf-8")
    with pytest.raises(gallery_flags.FlagsUnavailable):
        gallery.clear()
    # Nothing was unlinked before the refusal.
    assert gallery.image_path(record["id"]) is not None


def test_clear_all_still_works_with_an_unreadable_store():
    # include_archived spares nothing, so it needs no flags and must not be blocked by them.
    record = _save_with_mtime("a", 100.0)
    (gallery.gallery_dir() / ".flags.json").write_text("corrupt", encoding = "utf-8")
    assert gallery.clear(include_archived = True) == 1
    assert gallery.image_path(record["id"]) is None


def test_clear_refuses_when_a_single_flag_entry_is_malformed():
    # The whole-file corruption case is not the only risk: one bad VALUE reads as "not archived",
    # which is enough to delete an archived image, so it must block the clear too.
    import json as _json

    record = _save_with_mtime("shelved", 100.0)
    gallery.set_flags(record["id"], archived = True)
    (gallery.gallery_dir() / ".flags.json").write_text(
        _json.dumps({"version": 1, "items": {record["id"]: "hand edited"}}), encoding = "utf-8"
    )
    with pytest.raises(gallery_flags.FlagsUnavailable):
        gallery.clear()
    assert gallery.image_path(record["id"]) is not None


def test_clear_refuses_when_an_archived_flag_is_not_a_boolean():
    # `{"archived": null}` is still a dict, so a container-only check trusted it, and every reader
    # turns it into "not archived" -- enough for the default clear to delete the archived image.
    import json as _json

    record = _save_with_mtime("shelved", 100.0)
    gallery.set_flags(record["id"], archived = True)
    (gallery.gallery_dir() / ".flags.json").write_text(
        _json.dumps({"version": 1, "items": {record["id"]: {"archived": None}}}), encoding = "utf-8"
    )
    with pytest.raises(gallery_flags.FlagsUnavailable):
        gallery.clear()
    assert gallery.image_path(record["id"]) is not None


def test_a_repair_never_makes_a_damaged_archive_deletable():
    # An unrelated pin rewrites the store. If that repair dropped the unreadable archived flag, the
    # store would come back trusted with the image active, and the next default clear() removes it.
    import json as _json

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
    # The sibling case to the test above, and the one it does not cover: when the CONTAINER is
    # unreadable rather than one entry, there is nothing to repair from. _load substitutes an empty
    # map, so an unrelated pin used to write a clean, trusted store saying nothing was archived --
    # and the next default clear() deleted the image the user had shelved.
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
    # The escape hatch still works, since it spares nothing and so needs no flags.
    assert gallery.clear(include_archived = True) == 2


def test_clear_all_replaces_an_unreadable_store_so_the_gallery_recovers():
    # The escape hatch has to actually escape. include_archived spares nothing, so once it has run
    # there is no file left for the taint to protect -- and leaving the corrupt store behind meant
    # every later default clear still refused, including for media generated afterwards.
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
    # clear() decides from a flag snapshot and then unlinks. Without a shared lock an archive
    # landing in that window returned success for a file the same clear went on to delete.
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
    # Either the archive won (reported success, file kept) or the clear won (reported gone, file
    # deleted). "Reported success but deleted" is the outcome this must never produce.
    assert said_ok == survived
