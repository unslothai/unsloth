# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Clip datasets in the diffusion dataset routes.

The families that train from video read a folder of clips and the caption beside each one.
Before this, the routes only knew about images, so such a folder summarised as empty and
``/diffusion/info`` never listed it: the video trainer was unreachable from the UI.

The claim these tests exist to hold is that admitting clips is PURELY ADDITIVE. That rests on
two properties, both asserted below rather than asserted in prose:

* the image and clip extension sets are DISJOINT, so no file changes which branch it takes;
* the caption rules over them are IDENTICAL, so a clip resolves its caption through the very
  same function, with the same sidecar-over-metadata precedence and the same empty-sidecar
  tombstone, as an image does.

Everything else here is the behaviour those two properties buy: a clip folder is listed, a
clip uploads with its sidecar, and the stem-keyed sidecar is shared between kinds exactly as
it is shared between two images.
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from auth.authentication import get_current_subject
from core.training.diffusion_clip_formats import CLIP_EXTS
from routes.training import (
    _DIFFUSION_DATASET_CLIP_EXTS,
    _DIFFUSION_DATASET_IMAGE_EXTS,
    _DIFFUSION_DATASET_MEDIA_EXTS,
    _diffusion_dataset_summary,
    _resolve_dataset_caption,
)
from routes.training import router as training_router


def _write_png(
    path,
    color = (200, 100, 50),
    size = (8, 8),
) -> None:
    Image.new("RGB", size, color).save(path, format = "PNG")


def _png_bytes(color = (200, 100, 50), size = (8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format = "PNG")
    return buf.getvalue()


# An mp4 is never decoded by these routes, only counted and moved, so a recognisable ftyp box
# is enough to stand in for one. Nothing here opens it.
_MP4_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(training_router, prefix = "/api/train")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


@pytest.fixture
def ds_root(monkeypatch, tmp_path):
    import utils.paths as up

    root = tmp_path / "assets" / "datasets"
    root.mkdir(parents = True)
    monkeypatch.setattr(up, "datasets_root", lambda: root)
    return root


# ── the additive claim ───────────────────────────────────────────────────────
def test_image_and_clip_extension_sets_are_disjoint():
    """The whole design rests on this. Every site that widened from images to media assumes an
    extension belongs to exactly ONE kind: the summary picks its counter with an if/elif, so an
    overlapping extension would count as an image and never as a clip, and the two counts would
    stop summing to the number of trainable files in the folder."""
    assert _DIFFUSION_DATASET_IMAGE_EXTS & _DIFFUSION_DATASET_CLIP_EXTS == set()
    assert _DIFFUSION_DATASET_MEDIA_EXTS == (
        _DIFFUSION_DATASET_IMAGE_EXTS | _DIFFUSION_DATASET_CLIP_EXTS
    )
    assert len(_DIFFUSION_DATASET_MEDIA_EXTS) == len(_DIFFUSION_DATASET_IMAGE_EXTS) + len(
        _DIFFUSION_DATASET_CLIP_EXTS
    )
    # Every extension is lowercase and dotted, since every test site compares Path.suffix.lower().
    for ext in _DIFFUSION_DATASET_MEDIA_EXTS:
        assert ext == ext.lower() and ext.startswith(".") and len(ext) > 1


def test_routes_read_the_shared_clip_definition():
    """The routes must not carry their own copy: a container the upload accepts but the
    trainer's discovery does not read is a dataset that uploads fine and trains on nothing."""
    assert _DIFFUSION_DATASET_CLIP_EXTS == set(CLIP_EXTS)


def test_trainer_clip_discovery_uses_the_same_set_when_it_is_present():
    """A divergence tripwire for the video trainers.

    The clip trainer lands separately from this. Once it is in the tree its own extension set
    has to be the shared one, or the two halves can drift: a clip the upload accepts and the
    summary counts but the trainer skips is a run that starts and trains on nothing. Skipped
    while the module is absent, enforced the moment it appears."""
    clips = pytest.importorskip("core.training.diffusion_h3_clips")
    assert set(clips._VIDEO_EXTS) == set(CLIP_EXTS)


def test_clip_captions_resolve_through_the_same_rules_as_images(tmp_path):
    """Identical caption rules, asserted by running BOTH kinds through the one resolver over the
    same four cases. If any row ever differs, clip support has stopped being additive."""
    folder = tmp_path / "mixed"
    folder.mkdir()
    meta = {
        "sidecar.png": "from metadata",
        "sidecar.mp4": "from metadata",
        "meta.png": "from metadata",
        "meta.mp4": "from metadata",
        "tombstone.png": "from metadata",
        "tombstone.mp4": "from metadata",
    }
    for stem in ("sidecar", "meta", "tombstone", "bare"):
        _write_png(folder / f"{stem}.png")
        (folder / f"{stem}.mp4").write_bytes(_MP4_BYTES)
    # A sidecar wins over the metadata row; an EMPTY sidecar is a tombstone that shadows it.
    (folder / "sidecar.txt").write_text("edited sidecar", encoding = "utf-8")
    (folder / "tombstone.txt").write_text("   ", encoding = "utf-8")

    # sidecar.txt is shared by sidecar.png and sidecar.mp4 here on purpose: that sharing is the
    # point, and it is what the upload's stem check refuses to let a user create by accident.
    for stem, expected in (
        ("sidecar", "edited sidecar"),
        ("meta", "from metadata"),
        ("tombstone", ""),
        ("bare", None),
    ):
        image = _resolve_dataset_caption(folder, folder / f"{stem}.png", meta)
        clip = _resolve_dataset_caption(folder, folder / f"{stem}.mp4", meta)
        assert image == expected, stem
        assert clip == expected, stem
        assert image == clip, stem


def test_an_image_only_dataset_summarises_exactly_as_before(tmp_path):
    """The regression guard on the additive claim: nothing about an image dataset moves."""
    folder = tmp_path / "styleset"
    folder.mkdir()
    _write_png(folder / "a.png")
    _write_png(folder / "b.jpg")
    _write_png(folder / "c.webp")
    (folder / "a.txt").write_text("a cat", encoding = "utf-8")
    (folder / "metadata.jsonl").write_text(
        json.dumps({"file_name": "b.jpg", "text": "a dog"}) + "\n", encoding = "utf-8"
    )
    (folder / "notes.pdf").write_bytes(b"not a dataset file")

    summary = _diffusion_dataset_summary(folder)
    assert summary.image_count == 3
    assert summary.clip_count == 0
    assert summary.caption_count == 2


# ── summary + listing ────────────────────────────────────────────────────────
def test_summary_counts_clips_and_their_captions(tmp_path):
    folder = tmp_path / "clipset"
    folder.mkdir()
    for name in ("one.mp4", "two.MOV", "three.webm"):
        (folder / name).write_bytes(_MP4_BYTES)
    (folder / "one.txt").write_text("a waterfall", encoding = "utf-8")
    (folder / "captions.jsonl").write_text(
        json.dumps({"file_name": "two.MOV", "text": "a train"}) + "\n", encoding = "utf-8"
    )

    summary = _diffusion_dataset_summary(folder)
    assert summary.image_count == 0
    assert summary.clip_count == 3
    # Captions are one folder total over both kinds; three.webm has none from any source.
    assert summary.caption_count == 2


def test_summary_counts_a_mixed_folder_by_kind(tmp_path):
    folder = tmp_path / "both"
    folder.mkdir()
    _write_png(folder / "still.png")
    (folder / "moving.mp4").write_bytes(_MP4_BYTES)

    summary = _diffusion_dataset_summary(folder)
    assert (summary.image_count, summary.clip_count) == (1, 1)


def test_info_lists_a_clip_only_dataset(client, ds_root):
    """The bug this change exists to fix: a folder of clips was never offered in the picker,
    so the video trainer could not be pointed at anything."""
    folder = ds_root / "clipset"
    folder.mkdir()
    (folder / "one.mp4").write_bytes(_MP4_BYTES)
    (folder / "one.txt").write_text("a waterfall", encoding = "utf-8")

    r = client.get("/api/train/diffusion/info")
    assert r.status_code == 200, r.text
    datasets = {d["name"]: d for d in r.json()["datasets"]}
    assert "clipset" in datasets
    assert datasets["clipset"]["clip_count"] == 1
    assert datasets["clipset"]["image_count"] == 0
    assert datasets["clipset"]["caption_count"] == 1


def test_info_still_skips_a_folder_holding_neither(client, ds_root):
    """Admission widened to "an image OR a clip", not to "any folder"."""
    folder = ds_root / "captions-only"
    folder.mkdir()
    (folder / "metadata.jsonl").write_text(
        json.dumps({"file_name": "gone.png", "text": "x"}) + "\n", encoding = "utf-8"
    )
    (folder / "readme.txt").write_text("nothing to train on", encoding = "utf-8")

    r = client.get("/api/train/diffusion/info")
    assert r.status_code == 200, r.text
    assert [d["name"] for d in r.json()["datasets"]] == []


def test_list_images_marks_clips_and_leaves_images_unchanged(client, ds_root):
    """Clips list so a caller can see every name holding a stem-keyed sidecar open, but they
    carry no pixel dimensions and the grid filters them out on ``kind``."""
    folder = ds_root / "both"
    folder.mkdir()
    _write_png(folder / "still.png", size = (12, 9))
    (folder / "moving.mp4").write_bytes(_MP4_BYTES)
    (folder / "moving.txt").write_text("a train", encoding = "utf-8")

    r = client.get("/api/train/diffusion/dataset/both/images")
    assert r.status_code == 200, r.text
    recs = {rec["filename"]: rec for rec in r.json()["images"]}
    assert recs["still.png"]["kind"] == "image"
    assert (recs["still.png"]["width"], recs["still.png"]["height"]) == (12, 9)
    assert recs["moving.mp4"]["kind"] == "clip"
    assert (recs["moving.mp4"]["width"], recs["moving.mp4"]["height"]) == (0, 0)
    assert recs["moving.mp4"]["caption"] == "a train"
    assert recs["moving.mp4"]["caption_source"] == "sidecar"


# ── upload ───────────────────────────────────────────────────────────────────
def test_upload_accepts_a_clip_and_its_sidecar_as_a_pair(client, ds_root):
    r = client.post(
        "/api/train/diffusion/dataset",
        data = {"name": "clipset"},
        files = [
            ("files", ("one.mp4", _MP4_BYTES, "video/mp4")),
            ("files", ("one.txt", b"a waterfall", "text/plain")),
            ("files", ("two.webm", _MP4_BYTES, "video/webm")),
        ],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["uploaded"] == 3
    assert body["clip_count"] == 2
    assert body["image_count"] == 0
    assert body["caption_count"] == 1
    folder = ds_root / "clipset"
    assert (folder / "one.mp4").read_bytes() == _MP4_BYTES
    assert (folder / "one.txt").read_text(encoding = "utf-8") == "a waterfall"


def test_upload_still_refuses_an_extension_of_neither_kind(client, ds_root):
    r = client.post(
        "/api/train/diffusion/dataset",
        data = {"name": "clipset"},
        files = [("files", ("notes.pdf", b"%PDF-1.4", "application/pdf"))],
    )
    assert r.status_code == 400, r.text
    # All-or-nothing: the refused batch leaves nothing behind.
    assert not (ds_root / "clipset" / "notes.pdf").exists()


def test_upload_refuses_an_image_and_a_clip_sharing_a_stem(client, ds_root):
    """cat.png and cat.mp4 both resolve to cat.txt, exactly as two images do, so the same
    refusal has to cover the cross-kind pair."""
    r = client.post(
        "/api/train/diffusion/dataset",
        data = {"name": "mixed"},
        files = [
            ("files", ("cat.png", _png_bytes(), "image/png")),
            ("files", ("cat.mp4", _MP4_BYTES, "video/mp4")),
        ],
    )
    assert r.status_code == 400, r.text
    # The wording names the kind of the file being refused, so the existing image-vs-image
    # message is unchanged and the cross-kind one still reads truthfully.
    assert "Duplicate clip name 'cat'" in r.json()["detail"]
    assert "cat.png" in r.json()["detail"]
    assert list((ds_root / "mixed").iterdir()) == []


def test_upload_refuses_a_clip_whose_stem_is_already_in_the_folder(client, ds_root):
    folder = ds_root / "mixed"
    folder.mkdir()
    _write_png(folder / "cat.png")

    r = client.post(
        "/api/train/diffusion/dataset",
        data = {"name": "mixed"},
        files = [("files", ("cat.mp4", _MP4_BYTES, "video/mp4"))],
    )
    assert r.status_code == 400, r.text
    assert not (folder / "cat.mp4").exists()


def test_upload_does_not_decode_a_clip_as_an_image(client, ds_root):
    """The decompression-bomb check is about pixels and stays on images. A clip whose bytes
    Pillow cannot open must not be turned into a 400 by it."""
    r = client.post(
        "/api/train/diffusion/dataset",
        data = {"name": "clipset"},
        files = [("files", ("one.mkv", b"\x1a\x45\xdf\xa3" + b"\x00" * 32, "video/x-matroska"))],
    )
    assert r.status_code == 200, r.text
    assert r.json()["clip_count"] == 1


# ── delete ───────────────────────────────────────────────────────────────────
def test_deleting_an_image_keeps_a_clip_of_the_same_stem_captioned(client, ds_root):
    """Legacy folders can hold cat.png beside cat.mp4 (the upload refuses new ones). Deleting
    the image must not strip the sidecar the surviving clip still reads."""
    folder = ds_root / "legacy"
    folder.mkdir()
    _write_png(folder / "cat.png")
    (folder / "cat.mp4").write_bytes(_MP4_BYTES)
    (folder / "cat.txt").write_text("a cat", encoding = "utf-8")

    r = client.delete("/api/train/diffusion/dataset/legacy/image/cat.png")
    assert r.status_code == 200, r.text
    assert not (folder / "cat.png").exists()
    assert (folder / "cat.txt").read_text(encoding = "utf-8") == "a cat"
    assert _diffusion_dataset_summary(folder).caption_count == 1
