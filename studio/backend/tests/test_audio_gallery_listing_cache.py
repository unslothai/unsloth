# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The audio listing scans the folder once per request and reads a sidecar again only after it
changed, with the same records, order and cursors as a fresh read."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import core.inference.audio_gallery as gallery
from core.inference import gallery_flags

import pytest


@pytest.fixture(autouse = True)
def _tmp_gallery(monkeypatch, tmp_path):
    monkeypatch.setattr(gallery, "studio_root", lambda: tmp_path)
    with gallery._sidecar_cache_lock:
        gallery._sidecar_cache.clear()
        gallery._sidecar_cache_chars = 0


def _meta(**over):
    base = {
        "prompt": "hello from a sloth",
        "model": "unsloth/orpheus-3b-0.1-ft",
        "audio_type": "snac",
        "sample_rate": 24000,
        "duration_s": 1.5,
        "created_at": "2026-08-06T00:00:00Z",
        "settings": {"language": "en", "options": {"speed": [1, 2]}},
    }
    base.update(over)
    return base


def _clip(
    directory: Path,
    audio_id: str,
    mtime: float,
    meta = None,
) -> Path:
    (directory / f"{audio_id}.wav").write_bytes(b"RIFF")
    sidecar = directory / f"{audio_id}.json"
    if meta is not False:
        sidecar.write_text(json.dumps(meta or _meta(prompt = audio_id)), encoding = "utf-8")
    os.utime(directory / f"{audio_id}.wav", (mtime, mtime))
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


def _fresh(limit = 10_000, **kwargs):
    with gallery._sidecar_cache_lock:
        gallery._sidecar_cache.clear()
        gallery._sidecar_cache_chars = 0
    return gallery.list_audio_page(limit, **kwargs)


def test_unchanged_sidecars_are_read_once(monkeypatch):
    directory = gallery.gallery_dir()
    for i in range(6):
        _clip(directory, f"c{i}", 1_700_000_000 + i)
    seen = _reads(monkeypatch)
    first = gallery.list_audio()
    assert sorted(seen) == [f"c{i}" for i in range(6)]
    seen.clear()
    assert gallery.list_audio() == first
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
def test_a_changed_sidecar_is_read_again(monkeypatch, change):
    directory = gallery.gallery_dir()
    sidecar = _clip(directory, "a1", 1_700_000_000, _meta(prompt = "pppp"))
    _clip(directory, "b2", 1_700_000_001)
    gallery.list_audio()
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
    assert gallery.list_audio() == [r for r, _ in _fresh()]


def test_callers_cannot_change_a_later_listing():
    directory = gallery.gallery_dir()
    _clip(directory, "a1", 1_700_000_000)
    record = gallery.list_audio()[0]
    record["settings"]["options"]["speed"].append(99)
    record["prompt"] = "changed"
    again = gallery.list_audio()[0]
    assert again["prompt"] == "a1"
    assert again["settings"] == {"language": "en", "options": {"speed": [1, 2]}}


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
    assert gallery.list_audio() == []
    assert str(directory / "a1.json") not in gallery._sidecar_cache
    fail["on"] = False
    assert [r["id"] for r in gallery.list_audio()] == ["a1"]


def test_order_cursor_and_flags_match_a_fresh_read():
    directory = gallery.gallery_dir()
    for i in range(12):
        _clip(directory, f"c{i:02d}", 1_700_000_000 + (i // 3))
    _clip(directory, "orphan", 1_700_000_100, meta = False)
    (directory / "not an id.wav").write_bytes(b"x")
    (directory / ".hidden.wav").write_bytes(b"x")
    with gallery_flags.exclusive(directory):
        gallery_flags.set_flags_locked(directory, "c03", pinned = True)
        gallery_flags.set_flags_locked(directory, "c07", archived = True)
    gallery.move("c00", "c03")
    for archived in (False, True):
        listed = gallery.list_audio_page(50, archived = archived)
        assert listed == _fresh(limit = 50, archived = archived)
        for record, cursor in listed:
            assert record["order_at"] == cursor[1]
            assert record["id"] == cursor[2]
    ids = [r["id"] for r, _ in gallery.list_audio_page(50)]
    assert ids[0] == "c03" and ids[1] == "c00" and "c07" not in ids and "orphan" not in ids
    walked, cursor = [], None
    while True:
        page = gallery.list_audio_page(4, before = cursor)
        walked += [r["id"] for r, _ in page]
        if len(page) < 4:
            break
        cursor = page[-1][1]
    assert walked == ids


def test_missing_folder_lists_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(gallery, "gallery_dir", lambda: tmp_path / "gone")
    assert gallery.list_audio() == []


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
    assert gallery.list_audio() == []


def test_cache_stays_within_its_budget(monkeypatch):
    directory = gallery.gallery_dir()
    monkeypatch.setattr(gallery, "_SIDECAR_CACHE_MAX", 3)
    for i in range(5):
        _clip(directory, f"c{i}", 1_700_000_000 + i)
    _clip(
        directory, "big", 1_699_000_000, _meta(prompt = "x" * (gallery._SIDECAR_CACHE_ITEM_MAX + 1))
    )
    assert len(gallery.list_audio()) == 6
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
                records = gallery.list_audio()
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
    assert gallery.list_audio() == [r for r, _ in _fresh()]
