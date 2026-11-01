# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Listing saved voices resolves the folder once and reads each sidecar once, with the same
skips and order as reading every voice twice."""

from __future__ import annotations

import json
import uuid

import pytest

import core.inference.audio_gallery as gallery
from core.inference import audio_voices


@pytest.fixture(autouse = True)
def _tmp_gallery(monkeypatch, tmp_path):
    monkeypatch.setattr(gallery, "studio_root", lambda: tmp_path)


def _voice(
    i,
    *,
    epoch = None,
    created = "2026-01-01T00:00:00Z",
    wav = True,
    body = None,
):
    vid = uuid.UUID(int = i + 1).hex
    folder = audio_voices.voices_dir()
    meta = {"name": f"v{i}", "duration_s": 1.0, "sample_rate": 24000, "created_at": created}
    if epoch is not None:
        meta["created_at_epoch"] = epoch
    (folder / f"{vid}.json").write_text(json.dumps(meta) if body is None else body, "utf-8")
    if wav:
        (folder / f"{vid}.wav").write_bytes(b"RIFF")
    return vid


def test_skips_corrupt_partial_and_lone_entries_and_keeps_order():
    newest = _voice(0, epoch = 3.0)
    tie_a = _voice(1, epoch = 2.0)
    tie_b = _voice(2, epoch = 2.0)
    no_epoch = _voice(3)
    _voice(4, body = '{"name": "x", "duration_s"')  # partially written
    _voice(5, body = "[]")
    _voice(6, body = json.dumps({"name": "x"}))  # missing required keys
    _voice(7, epoch = 9.0, wav = False)  # sidecar without its WAV
    (audio_voices.voices_dir() / f"{uuid.uuid4().hex}.wav").write_bytes(b"RIFF")  # lone WAV
    (audio_voices.voices_dir() / "not-an-id.json").write_text("{}", "utf-8")
    (audio_voices.voices_dir() / f"{uuid.UUID(int = 99).hex}.json").write_bytes(b"\xff\xfe")

    listed = [v["id"] for v in audio_voices.list_voices()]
    assert listed == [newest, max(tie_a, tie_b), min(tie_a, tie_b), no_epoch]
    # The count create() checks includes a sidecar whose WAV is gone, as before.
    assert len(audio_voices._ids()) == 5


def test_list_resolves_the_folder_once(monkeypatch):
    for i in range(20):
        _voice(i, epoch = float(i))
    calls = []
    real = audio_voices.voices_dir
    monkeypatch.setattr(audio_voices, "voices_dir", lambda: calls.append(1) or real())
    assert len(audio_voices.list_voices()) == 20
    assert len(calls) == 1


def test_unreadable_folder_lists_nothing(monkeypatch):
    def boom():
        raise OSError("drive gone")

    monkeypatch.setattr(audio_voices, "voices_dir", boom)
    assert audio_voices.list_voices() == []
    assert audio_voices._ids() == []
