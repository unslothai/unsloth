# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image listing rereads a PNG chunk only on change; video listing resolves only symlinked clips."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

import core.inference.image_gallery as images
import core.inference.video_gallery as videos

PIL = pytest.importorskip("PIL")
from PIL import Image, PngImagePlugin  # noqa: E402


@pytest.fixture(autouse = True)
def _tmp_gallery(monkeypatch, tmp_path):
    monkeypatch.setattr(images, "studio_root", lambda: tmp_path)
    monkeypatch.setattr(videos, "studio_root", lambda: tmp_path)
    if hasattr(images, "_chunk_cache"):
        monkeypatch.setattr(images, "_chunk_cache", type(images._chunk_cache)())
        monkeypatch.setattr(images, "_chunk_cache_bytes", 0)


def _meta(**over):
    base = {
        "prompt": "a sloth",
        "width": 16,
        "height": 16,
        "steps": 4,
        "guidance": 0.0,
        "seed": 7,
        "created_at": 100.0,
    }
    base.update(over)
    return base


def _png(
    path: Path,
    meta = None,
    raw = None,
) -> None:
    info = PngImagePlugin.PngInfo()
    if raw is not None or meta is not None:
        info.add_text(images._META_KEY, raw if raw is not None else json.dumps(meta))
    Image.new("RGB", (8, 8), (1, 2, 3)).save(path, format = "PNG", pnginfo = info)


def _count_opens(monkeypatch) -> list:
    opened = []
    real = Image.open

    def counting(fp, *args, **kwargs):
        opened.append(fp)
        return real(fp, *args, **kwargs)

    monkeypatch.setattr(Image, "open", counting)
    return opened


def test_an_unchanged_png_is_parsed_once(monkeypatch):
    for k in range(3):
        images.save(Image.new("RGB", (8, 8)), _meta(seed = k))
    first = images.list_images()
    opened = _count_opens(monkeypatch)
    assert images.list_images() == first
    assert opened == []


def test_listed_records_are_fresh_dicts():
    images.save(Image.new("RGB", (8, 8)), _meta())
    first = images.list_images()
    first[0]["prompt"] = "changed"
    assert images.list_images()[0]["prompt"] == "a sloth"


def test_a_rewritten_png_is_read_again_even_at_the_same_mtime():
    record = images.save(Image.new("RGB", (8, 8)), _meta(prompt = "before"))
    path = images.gallery_dir() / f"{record['id']}.png"
    assert images.list_images()[0]["prompt"] == "before"
    before = path.stat()
    _png(path, _meta(prompt = "after, longer"))
    os.utime(path, ns = (before.st_atime_ns, before.st_mtime_ns))
    assert images.list_images()[0]["prompt"] == "after, longer"


def test_a_replaced_png_is_read_again():
    record = images.save(Image.new("RGB", (8, 8)), _meta(prompt = "before"))
    path = images.gallery_dir() / f"{record['id']}.png"
    assert images.list_images()[0]["prompt"] == "before"
    tmp = path.with_name(".swap.tmp")
    _png(tmp, _meta(prompt = "after"))
    os.replace(tmp, path)
    assert images.list_images()[0]["prompt"] == "after"


@pytest.mark.parametrize("content", ["no chunk", "not json", "missing key", "not a png"])
def test_a_foreign_png_stays_skipped_until_it_becomes_ours(content):
    directory = images.gallery_dir()
    path = directory / "foreign.png"
    if content == "no chunk":
        _png(path)
    elif content == "not json":
        _png(path, raw = "{oops")
    elif content == "missing key":
        _png(path, {k: v for k, v in _meta().items() if k != "seed"})
    else:
        path.write_bytes(b"\x89PNG not really")
    assert images.list_images() == [] and images.list_images() == []
    _png(path, _meta(prompt = "now ours"))
    assert [r["prompt"] for r in images.list_images()] == ["now ours"]


def test_a_deleted_png_leaves_the_listing():
    record = images.save(Image.new("RGB", (8, 8)), _meta())
    assert len(images.list_images()) == 1
    assert images.delete(record["id"])
    assert images.list_images() == []


def test_the_chunk_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(images, "_CHUNK_CACHE_MAX", 2)
    for k in range(5):
        images.save(Image.new("RGB", (8, 8)), _meta(seed = k))
    first = images.list_images()
    assert len(images._chunk_cache) == 2
    assert images.list_images() == first and len(first) == 5


def test_a_huge_chunk_is_read_but_not_kept(monkeypatch):
    monkeypatch.setattr(images, "_CHUNK_CACHE_ITEM_MAX", 600)
    images.save(Image.new("RGB", (8, 8)), _meta(prompt = "x" * 1000))
    images.save(Image.new("RGB", (8, 8)), _meta(prompt = "short"))
    assert sorted(r["prompt"] for r in images.list_images()) == ["short", "x" * 1000]
    assert len(images._chunk_cache) == 1
    assert sorted(r["prompt"] for r in images.list_images()) == ["short", "x" * 1000]


def test_the_chunk_cache_holds_a_byte_budget(monkeypatch):
    monkeypatch.setattr(images, "_CHUNK_CACHE_BYTES", 1000)
    for k in range(5):
        images.save(Image.new("RGB", (8, 8)), _meta(prompt = "p" * 300, seed = k))
    first = images.list_images()
    assert 0 < len(images._chunk_cache) < 5
    assert images._chunk_cache_bytes == sum(v[2] for v in images._chunk_cache.values()) <= 1000
    assert images.list_images() == first and len(first) == 5


def test_concurrent_listings_agree(monkeypatch):
    for k in range(20):
        images.save(Image.new("RGB", (8, 8)), _meta(seed = k))
    expected = images.list_images()
    if hasattr(images, "_chunk_cache"):
        monkeypatch.setattr(images, "_chunk_cache", type(images._chunk_cache)())
        monkeypatch.setattr(images, "_chunk_cache_bytes", 0)
    results, errors = [], []

    def run():
        try:
            for _ in range(5):
                results.append(images.list_images())
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target = run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and all(r == expected for r in results) and len(results) == 40


def test_delete_still_raises_on_an_unreadable_png(monkeypatch):
    record = images.save(Image.new("RGB", (8, 8)), _meta())
    images.list_images()

    def refuse(*args, **kwargs):
        raise PermissionError(13, "denied")

    monkeypatch.setattr(Image, "open", refuse)
    with pytest.raises(PermissionError):
        images.delete(record["id"])


def _clip(directory: Path, video_id: str) -> Path:
    path = directory / f"{video_id}.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    return path


def test_a_plain_clip_is_found_without_resolving(monkeypatch):
    directory = videos.gallery_dir()
    path = _clip(directory, "plain")

    def no_resolve(self, *args, **kwargs):
        raise AssertionError("resolve() on a plain clip")

    monkeypatch.setattr(Path, "resolve", no_resolve)
    monkeypatch.setattr(type(path), "resolve", no_resolve)
    assert videos.video_path("plain") == path
    assert videos.video_path("plain", directory) == path


def test_video_path_misses():
    directory = videos.gallery_dir()
    (directory / "folder.mp4").mkdir()
    assert videos.video_path("folder") is None
    assert videos.video_path("absent") is None
    assert videos.video_path("../escape") is None
    assert videos.video_path("") is None


def test_symlinked_clips_still_take_the_full_check(tmp_path):
    directory = videos.gallery_dir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "far.mp4").write_bytes(b"x")
    try:
        (directory / "escape.mp4").symlink_to(outside / "far.mp4")
    except (OSError, NotImplementedError):
        pytest.skip("this account cannot create symlinks")
    (directory / "inner.bin").write_bytes(b"y")
    (directory / "inside.mp4").symlink_to(directory / "inner.bin")
    (directory / "dangling.mp4").symlink_to(directory / "nothing")
    (directory / "subdir").mkdir()
    (directory / "todir.mp4").symlink_to(directory / "subdir")
    assert videos.video_path("escape") is None
    assert videos.video_path("inside") == directory / "inside.mp4"
    assert videos.video_path("dangling") is None
    assert videos.video_path("todir") is None
