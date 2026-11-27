# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The video listing scans the folder once per request and reads a sidecar again only after it
changed, with the same records and order as a fresh read."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import core.inference.video_gallery as gallery
from core.inference import gallery_flags

import pytest


def _clear_cache():
    with gallery._sidecar_cache_lock:
        gallery._sidecar_cache.clear()
        gallery._sidecar_cache_chars = 0


@pytest.fixture(autouse = True)
def _tmp_gallery(monkeypatch, tmp_path):
    monkeypatch.setattr(gallery, "studio_root", lambda: tmp_path)
    _clear_cache()


def _meta(**over):
    base = {
        "prompt": "a sloth surfing",
        "negative_prompt": None,
        "width": 1024,
        "height": 576,
        "num_frames": 49,
        "fps": 24,
        "duration_s": 2.0,
        "steps": 30,
        "guidance": 6.0,
        "seed": 7,
        "model": "unsloth/some-video-model",
        "created_at": "2026-08-06T00:00:00Z",
        "extra": {"options": [1, 2]},
    }
    base.update(over)
    return base


def _clip(
    directory: Path,
    video_id: str,
    mtime: float,
    meta = None,
) -> Path:
    mp4 = directory / f"{video_id}.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    sidecar = directory / f"{video_id}.json"
    if meta is not False:
        sidecar.write_text(json.dumps(meta or _meta(prompt = video_id)), encoding = "utf-8")
    os.utime(mp4, (mtime, mtime))
    return sidecar


def _reads(monkeypatch):
    seen = []
    real = Path.read_text

    def counting(self, *args, **kwargs):
        if self.suffix == ".json" and not self.name.startswith("."):
            seen.append(self.stem)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting)
    return seen


def _fresh(*args, **kwargs):
    _clear_cache()
    return gallery.list_videos(*args, **kwargs)


def test_unchanged_sidecars_are_read_once(monkeypatch):
    directory = gallery.gallery_dir()
    for i in range(6):
        _clip(directory, f"c{i}", 1_700_000_000 + i)
    seen = _reads(monkeypatch)
    first = gallery.list_videos()
    assert sorted(seen) == [f"c{i}" for i in range(6)]
    seen.clear()
    assert gallery.list_videos() == first
    assert seen == []


@pytest.mark.parametrize(
    "change",
    [
        "rewrite",
        pytest.param(
            "same_size_same_mtime",
            marks = pytest.mark.skipif(
                sys.platform == "win32",
                reason = "st_ctime is the creation time on Windows, not the change time",
            ),
        ),
        "replace_same_mtime",
        "corrupt",
        "remove",
    ],
)
def test_a_changed_sidecar_is_read_again(change):
    directory = gallery.gallery_dir()
    sidecar = _clip(directory, "a1", 1_700_000_000, _meta(prompt = "pppp"))
    _clip(directory, "b2", 1_700_000_001)
    gallery.list_videos()
    info = os.stat(sidecar)
    if change == "rewrite":
        sidecar.write_text(json.dumps(_meta(prompt = "a longer prompt")), encoding = "utf-8")
    elif change == "same_size_same_mtime":
        sidecar.write_text(json.dumps(_meta(prompt = "qqqq")), encoding = "utf-8")
        os.utime(sidecar, ns = (info.st_atime_ns, info.st_mtime_ns))
    elif change == "replace_same_mtime":
        tmp = directory / ".a1.json.tmp"
        tmp.write_text(json.dumps(_meta(prompt = "rrrr")), encoding = "utf-8")
        os.utime(tmp, ns = (info.st_atime_ns, info.st_mtime_ns))
        os.replace(tmp, sidecar)
    elif change == "corrupt":
        sidecar.write_text("{", encoding = "utf-8")
    else:
        sidecar.unlink()
    assert gallery.list_videos() == _fresh()


def test_callers_cannot_change_a_later_listing():
    directory = gallery.gallery_dir()
    _clip(directory, "a1", 1_700_000_000)
    record = gallery.list_videos()[0]
    record["extra"]["options"].append(99)
    record["prompt"] = "changed"
    again = gallery.list_videos()[0]
    assert again["prompt"] == "a1"
    assert again["extra"] == {"options": [1, 2]}


def test_unreadable_sidecar_is_not_kept(monkeypatch):
    directory = gallery.gallery_dir()
    _clip(directory, "a1", 1_700_000_000)
    real = Path.read_text
    fail = {"on": True}

    def flaky(self, *args, **kwargs):
        if fail["on"] and self.name == "a1.json":
            raise PermissionError("locked")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky)
    assert gallery.list_videos() == []
    assert str(directory / "a1.json") not in gallery._sidecar_cache
    fail["on"] = False
    assert [r["id"] for r in gallery.list_videos()] == ["a1"]


def test_order_pages_and_flags_match_a_fresh_read():
    directory = gallery.gallery_dir()
    for i in range(12):
        # ties on mtime keep the folder's order, as before
        _clip(directory, f"c{i:02d}", 1_700_000_000 + (i // 3))
    _clip(directory, "orphan", 1_700_000_100, meta = False)
    (directory / "not an id.mp4").write_bytes(b"x")
    (directory / "not an id.json").write_text(json.dumps(_meta()), encoding = "utf-8")
    # Path.stem keeps a dot-led name whole: ".mp4" pairs with ".mp4.json".
    (directory / ".mp4").write_bytes(b"x")
    (directory / ".mp4.json").write_text(json.dumps(_meta()), encoding = "utf-8")
    (directory / "folder.mp4").mkdir()
    (directory / "folder.json").write_text(json.dumps(_meta()), encoding = "utf-8")
    with gallery_flags.exclusive(directory):
        gallery_flags.set_flags_locked(directory, "c03", pinned = True)
        gallery_flags.set_flags_locked(directory, "c07", archived = True)
    gallery.move("c00", "c03")
    for archived in (False, True):
        for limit, offset in ((None, 0), (4, 0), (4, 4), (51, 0), (3, 100)):
            assert gallery.list_videos(limit, offset, archived = archived) == _fresh(
                limit, offset, archived = archived
            )
    listed = gallery.list_videos()
    by_id = {r["id"]: r for r in listed}
    assert listed[0]["id"] == "c03" and listed[1]["id"] == "c00"
    assert "c07" not in by_id and "orphan" not in by_id
    # Entries video_path refuses keep the 0.0 they had: a bad id and a folder are not clips.
    assert by_id["not an id"]["order_at"] == 0.0
    assert by_id["folder"]["order_at"] == 0.0
    assert by_id[".mp4"]["order_at"] == 0.0
    assert by_id["c11"]["order_at"] == os.stat(directory / "c11.mp4").st_mtime


@pytest.mark.skipif(sys.platform == "win32", reason = "symlinks need privileges on Windows")
def test_symlinked_clips_keep_the_full_check(tmp_path):
    directory = gallery.gallery_dir()
    _clip(directory, "real", 1_700_000_000)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x")
    os.utime(outside, (1_700_000_500, 1_700_000_500))
    os.symlink(outside, directory / "escape.mp4")
    (directory / "escape.json").write_text(json.dumps(_meta()), encoding = "utf-8")
    os.symlink(directory / "real.mp4", directory / "inside.mp4")
    (directory / "inside.json").write_text(json.dumps(_meta()), encoding = "utf-8")
    listed = gallery.list_videos()
    assert listed == _fresh()
    by_id = {r["id"]: r for r in listed}
    # Sorted by the target's mtime, but an escaping link is not a clip, so its order_at is 0.0.
    assert listed[0]["id"] == "escape" and by_id["escape"]["order_at"] == 0.0
    assert by_id["inside"]["order_at"] == 1_700_000_000


def test_missing_folder_lists_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(gallery, "gallery_dir", lambda: tmp_path / "gone")
    assert gallery.list_videos() == []


def test_a_read_error_mid_scan_lists_nothing(monkeypatch):
    directory = gallery.gallery_dir()
    for i in range(3):
        _clip(directory, f"c{i}", 1_700_000_000 + i)
    real = os.scandir

    class Failing:
        def __init__(self, it):
            self._it = it
            self._seen = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._it.close()

        def __iter__(self):
            return self

        def __next__(self):
            self._seen += 1
            if self._seen > 1:
                raise OSError(5, "Input/output error")
            return next(self._it)

    monkeypatch.setattr(gallery.os, "scandir", lambda path: Failing(real(path)))
    # Path.glob listed a folder whose read failed part way as empty, never a 500.
    assert gallery.list_videos() == []


def test_only_the_requested_shelf_is_statted(monkeypatch):
    directory = gallery.gallery_dir()
    for i in range(4):
        _clip(directory, f"c{i}", 1_700_000_000 + i)
    gallery_flags.set_flags_locked(directory, "c0", pinned = None, archived = True)
    real = os.scandir
    statted = []

    class Entry:
        def __init__(self, entry):
            self._entry = entry
            self.name = entry.name

        def stat(self):
            statted.append(self.name)
            return self._entry.stat()

        def is_file(self, **kwargs):
            return self._entry.is_file(**kwargs)

    class Scan:
        def __init__(self, it):
            self._it = it

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._it.close()

        def __iter__(self):
            return (Entry(e) for e in self._it)

    monkeypatch.setattr(gallery.os, "scandir", lambda path: Scan(real(path)))
    assert [r["id"] for r in gallery.list_videos(archived = True)] == ["c0"]
    assert statted == ["c0.mp4"]


def test_an_uppercase_name_keeps_the_full_check(monkeypatch):
    directory = gallery.gallery_dir()
    _clip(directory, "c1", 1_700_000_000)
    (directory / "c1.mp4").rename(directory / "C2.MP4")
    (directory / "c1.json").rename(directory / "C2.json")
    monkeypatch.setattr(gallery, "_GLOB_FOLDS_CASE", True)
    expected = os.stat(directory / "C2.mp4").st_mtime if (directory / "C2.mp4").is_file() else 0.0
    [record] = gallery.list_videos()
    assert record["id"] == "C2"
    assert record["order_at"] == gallery_flags.order_rank({}, "C2", expected)


def test_cache_stays_within_its_budget(monkeypatch):
    directory = gallery.gallery_dir()
    monkeypatch.setattr(gallery, "_SIDECAR_CACHE_MAX", 3)
    for i in range(5):
        _clip(directory, f"c{i}", 1_700_000_000 + i)
    _clip(
        directory, "big", 1_699_000_000, _meta(prompt = "x" * (gallery._SIDECAR_CACHE_ITEM_MAX + 1))
    )
    assert len(gallery.list_videos()) == 6
    assert len(gallery._sidecar_cache) == 3
    assert str(directory / "big.json") not in gallery._sidecar_cache
    assert gallery._sidecar_cache_chars == sum(len(v[1]) for v in gallery._sidecar_cache.values())


def test_concurrent_writers_never_mix_records():
    directory = gallery.gallery_dir()
    for i in range(20):
        _clip(directory, f"c{i:02d}", 1_700_000_000 + i, _meta(prompt = "v0"))
    stop = threading.Event()
    errors = []

    def writer():
        n = 0
        while not stop.is_set():
            n += 1
            for i in range(0, 20, 3):
                tmp = directory / f".c{i:02d}.json.tmp"
                tmp.write_text(json.dumps(_meta(prompt = f"v{n}")), encoding = "utf-8")
                os.replace(tmp, directory / f"c{i:02d}.json")

    thread = threading.Thread(target = writer)
    thread.start()
    try:
        for _ in range(50):
            try:
                records = gallery.list_videos()
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)
                break
            for record in records:
                if int(record["id"][1:]) % 3:
                    assert record["prompt"] == "v0"
    finally:
        stop.set()
        thread.join()
    assert not errors
    assert gallery.list_videos() == _fresh()
