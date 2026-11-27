# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Single-scan image listing matches the old per-file glob + stat listing."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import core.inference.image_gallery as images
from core.inference import gallery_flags

PIL = pytest.importorskip("PIL")
from PIL import Image, PngImagePlugin  # noqa: E402

_T0 = 1_700_000_000_000_000_000


@pytest.fixture(autouse = True)
def _tmp_gallery(monkeypatch, tmp_path):
    monkeypatch.setattr(images, "studio_root", lambda: tmp_path)
    monkeypatch.setattr(images, "_chunk_cache", type(images._chunk_cache)())
    monkeypatch.setattr(images, "_chunk_cache_bytes", 0)


def _meta(seed):
    return {
        "prompt": f"p{seed}",
        "width": 8,
        "height": 8,
        "steps": 4,
        "guidance": 0.0,
        "seed": seed,
        "created_at": 100.0 + seed,
    }


def _png(
    name,
    seed,
    mtime_ns,
    *,
    owned = True,
):
    path = images.gallery_dir() / name
    info = PngImagePlugin.PngInfo()
    if owned:
        info.add_text(images._META_KEY, json.dumps(_meta(seed)))
    Image.new("RGB", (8, 8)).save(path, format = "PNG", pnginfo = info)
    os.utime(path, ns = (mtime_ns, mtime_ns))
    return path


def _glob_order(archived = False):
    """The listing order as the per-file glob + stat listing computed it."""
    directory = images.gallery_dir()
    flags = gallery_flags.read(directory)
    paths = [
        p for p in directory.glob("*.png") if gallery_flags.is_archived(flags, p.stem) == archived
    ]

    def mtime(p):
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    paths.sort(
        key = lambda p: (
            gallery_flags.pin_rank(flags, p.stem),
            gallery_flags.order_rank(flags, p.stem, mtime(p)),
        ),
        reverse = True,
    )
    return [p.stem for p in paths if images._read_meta(p) is not None]


def _mixed_gallery():
    for i in range(12):
        _png(f"img{i:02d}.png", i, _T0 + (i // 3) * 1_000_000)
    _png(".hidden.png", 50, _T0 + 1_000_000)
    _png(".png", 51, _T0 + 2_000_000)
    _png("a..png", 52, _T0 + 3_000_000)
    _png("UPPER.PNG", 53, _T0 + 4_000_000)
    _png("foreign.png", 54, _T0 + 5_000_000, owned = False)
    _png("tmp.png.tmp", 55, _T0 + 6_000_000)
    (images.gallery_dir() / "folder.png").mkdir()
    directory = images.gallery_dir()
    gallery_flags.set_flags_locked(directory, "img03", pinned = True, archived = None)
    gallery_flags.set_flags_locked(directory, "img05", pinned = None, archived = True)


@pytest.mark.parametrize("full_entry_stat", [True, False])
@pytest.mark.parametrize("folds_case", [False, True])
def test_listing_matches_the_glob_listing(monkeypatch, full_entry_stat, folds_case):
    _mixed_gallery()
    monkeypatch.setattr(images, "_ENTRY_STAT_IS_FULL", full_entry_stat)
    monkeypatch.setattr(images, "_GLOB_FOLDS_CASE", folds_case)
    # Emulate the host's glob case rule either way, so both cells run on every OS.
    real_glob = Path.glob
    monkeypatch.setattr(
        Path,
        "glob",
        lambda self, pattern: [
            p
            for p in real_glob(self, "*")
            if (p.name.lower() if folds_case else p.name).endswith(pattern[1:])
        ],
    )
    for archived in (False, True):
        expected = _glob_order(archived)
        assert [r["id"] for r in images.list_images(archived = archived)] == expected
        assert [r["id"] for r in images.list_images(archived = archived)] == expected
    listed = [r["id"] for r in images.list_images()]
    assert ("UPPER" in listed) == folds_case
    assert {".hidden", ".png", "a."} <= set(listed)
    assert "foreign" not in listed and "folder" not in listed
    for offset in (0, 4, 100):
        assert [r["id"] for r in images.list_images(5, offset)] == listed[offset : offset + 5]


@pytest.mark.parametrize("full_entry_stat", [True, False])
def test_a_page_stats_only_the_records_it_reads(monkeypatch, full_entry_stat):
    for i in range(30):
        _png(f"img{i:02d}.png", i, _T0 + i * 1_000_000)
    monkeypatch.setattr(images, "_ENTRY_STAT_IS_FULL", full_entry_stat)
    images.list_images(4, 0)
    real = os.stat
    statted = []

    def counting(path, *args, **kwargs):
        if str(path).endswith(".png"):
            statted.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(images.os, "stat", counting)
    page = images.list_images(4, 2)
    assert [r["id"] for r in page] == [f"img{i:02d}" for i in (27, 26, 25, 24)]
    # A real stat keys the recipe cache on Windows only, and only for the offset + limit records read.
    assert len(statted) == (0 if full_entry_stat else 6)


@pytest.mark.parametrize("full_entry_stat", [True, False])
def test_a_rewrite_between_listings_is_reread(monkeypatch, full_entry_stat):
    monkeypatch.setattr(images, "_ENTRY_STAT_IS_FULL", full_entry_stat)
    _png("img.png", 1, _T0)
    assert images.list_images()[0]["seed"] == 1
    # Same size and mtime: POSIX ctime catches the rewrite; Windows ctime is creation time.
    _png("img.png", 2, _T0 if os.name != "nt" else _T0 + 1_000_000)
    assert images.list_images()[0]["seed"] == 2


@pytest.mark.parametrize("full_entry_stat", [True, False])
def test_a_png_deleted_before_its_read_is_skipped(monkeypatch, full_entry_stat):
    for i in range(6):
        _png(f"img{i}.png", i, _T0 + i * 1_000_000)
    monkeypatch.setattr(images, "_ENTRY_STAT_IS_FULL", full_entry_stat)
    real = images._listed_meta

    def deleting(path, info):
        if path.stem == "img5":
            (path.parent / "img3.png").unlink()
        return real(path, info)

    monkeypatch.setattr(images, "_listed_meta", deleting)
    assert [r["id"] for r in images.list_images(3, 0)] == ["img5", "img4", "img2"]


def test_a_read_error_mid_scan_lists_nothing(monkeypatch):
    for i in range(3):
        _png(f"img{i}.png", i, _T0 + i * 1_000_000)
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

    monkeypatch.setattr(images.os, "scandir", lambda path: Failing(real(path)))
    # Path.glob listed a folder whose read failed part way as empty, never a 500.
    assert images.list_images() == []
    assert images.move("img0", None) is not None


def test_an_unlistable_folder_lists_nothing(monkeypatch):
    def failing(path):
        raise PermissionError(13, "denied")

    monkeypatch.setattr(images.os, "scandir", failing)
    assert images.list_images() == []


def test_move_orders_the_shelf_like_the_listing():
    for i in range(6):
        _png(f"img{i}.png", i, _T0 + i * 1_000_000)
    directory = images.gallery_dir()
    gallery_flags.set_flags_locked(directory, "img1", pinned = None, archived = True)
    before = [r["id"] for r in images.list_images()]
    assert before == _glob_order()
    moved = images.move(before[0], before[2])
    assert moved is not None and moved["id"] == before[0]
    after = [r["id"] for r in images.list_images()]
    assert after.index(before[0]) == after.index(before[2]) + 1
    assert after == _glob_order()
    with pytest.raises(KeyError):
        images.move(before[1], "img1")


def test_a_broken_symlink_sorts_last_and_is_not_listed():
    for i in range(2):
        _png(f"img{i}.png", i, _T0 + i * 1_000_000)
    directory = images.gallery_dir()
    try:
        os.symlink(directory / "missing.png.gone", directory / "broken.png")
    except (OSError, NotImplementedError):
        pytest.skip(reason = "creating symlinks needs Developer Mode or admin on Windows")
    assert [r["id"] for r in images.list_images()] == ["img1", "img0"] == _glob_order()
