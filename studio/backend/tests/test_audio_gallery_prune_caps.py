# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os

import pytest

import core.inference.audio_gallery as gallery


@pytest.fixture(autouse = True)
def _tmp_gallery(monkeypatch, tmp_path):
    monkeypatch.setattr(gallery, "studio_root", lambda: tmp_path)


def _wav():
    return b"RIFF\x24\x00\x00\x00WAVEfmt "


def _meta(**over):
    base = {
        "prompt": "hello from a sloth",
        "model": "unsloth/orpheus-3b-0.1-ft",
        "audio_type": "snac",
        "sample_rate": 24000,
        "duration_s": 1.5,
        "created_at": "2026-08-06T00:00:00Z",
    }
    base.update(over)
    return base


def _save_with_mtime(prompt: str, t: float) -> dict:
    record = gallery.save(_wav(), _meta(prompt = prompt))
    os.utime(gallery.gallery_dir() / f"{record['id']}.wav", (t, t))
    return record


def test_prune_under_both_caps_reads_no_sidecar(monkeypatch):
    for i in range(5):
        gallery.save(_wav(), _meta(prompt = f"clip {i}"))
    monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_CLIPS", "5")
    reads = []
    monkeypatch.setattr(gallery, "_read_meta", lambda path: reads.append(path))
    assert gallery._prune_to_cap() == 0
    assert reads == []


@pytest.mark.parametrize("over", ["count", "bytes", "source_bytes"])
def test_prune_just_over_a_cap_still_prunes_the_oldest(monkeypatch, over):
    first = _save_with_mtime("first", 100.0)
    if over == "source_bytes":
        (gallery.gallery_dir() / f"{first['id']}.source.wav").write_bytes(b"s" * 64)
    second = _save_with_mtime("second", 200.0)
    size = (gallery.gallery_dir() / f"{second['id']}.wav").stat().st_size
    if over == "count":
        monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_CLIPS", "1")
    else:
        monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_CLIPS", "0")
        slack = 63 if over == "source_bytes" else -1
        monkeypatch.setenv("UNSLOTH_AUDIO_GALLERY_MAX_BYTES", str(2 * size + slack))
    assert gallery._prune_to_cap() == 1
    assert gallery.audio_path(first["id"]) is None
    assert gallery.audio_path(second["id"]) is not None
