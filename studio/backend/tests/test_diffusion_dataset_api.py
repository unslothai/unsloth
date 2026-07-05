# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the diffusion dataset labeling + example-import routes.

The routes are hit with the FastAPI TestClient; the datasets root is redirected to a
tmp_path so nothing touches a real Studio home. The example importer is exercised with a
mocked datasets.load_dataset so no network / GPU is needed.
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from auth.authentication import get_current_subject
from routes.training import router as training_router


def _png_bytes(color = (200, 100, 50), size = (8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format = "PNG")
    return buf.getvalue()


def _write_png(
    path,
    color = (200, 100, 50),
    size = (8, 8),
) -> None:
    Image.new("RGB", size, color).save(path, format = "PNG")


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


# ── listing + caption precedence ─────────────────────────────────────────────
def test_list_images_caption_precedence(client, ds_root):
    folder = ds_root / "styleset"
    folder.mkdir()
    _write_png(folder / "a.png")
    _write_png(folder / "b.png")
    _write_png(folder / "c.png")
    # a.png -> sidecar (an explicit edit beats the metadata row), b.png -> metadata-only,
    # c.png -> none.
    (folder / "metadata.jsonl").write_text(
        json.dumps({"file_name": "a.png", "text": "from metadata"})
        + "\n"
        + json.dumps({"file_name": "b.png", "text": "from metadata"})
        + "\n",
        encoding = "utf-8",
    )
    (folder / "a.txt").write_text("edited sidecar", encoding = "utf-8")

    r = client.get("/api/train/diffusion/dataset/styleset/images")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "styleset"
    recs = {rec["filename"]: rec for rec in body["images"]}
    assert set(recs) == {"a.png", "b.png", "c.png"}
    # A sidecar edit overrides the metadata row for the same image.
    assert recs["a.png"]["caption"] == "edited sidecar"
    assert recs["a.png"]["caption_source"] == "sidecar"
    assert recs["b.png"]["caption"] == "from metadata"
    assert recs["b.png"]["caption_source"] == "metadata"
    assert recs["c.png"]["caption"] is None
    assert recs["c.png"]["caption_source"] == "none"
    assert recs["a.png"]["width"] == 8 and recs["a.png"]["height"] == 8


def test_list_images_missing_dataset_404(client, ds_root):
    assert client.get("/api/train/diffusion/dataset/nope/images").status_code == 404


# ── image serving + thumbnails ───────────────────────────────────────────────
def test_get_image_and_thumbnail_excluded_from_listing(client, ds_root):
    folder = ds_root / "pics"
    folder.mkdir()
    _write_png(folder / "one.png", size = (64, 48))

    full = client.get("/api/train/diffusion/dataset/pics/image/one.png")
    assert full.status_code == 200, full.text

    thumb = client.get("/api/train/diffusion/dataset/pics/image/one.png?thumb=32")
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/jpeg"
    assert (folder / ".thumbs").is_dir()

    # The .thumbs cache dir must not surface as a dataset image.
    listing = client.get("/api/train/diffusion/dataset/pics/images").json()
    assert [rec["filename"] for rec in listing["images"]] == ["one.png"]


def test_get_image_missing_404(client, ds_root):
    (ds_root / "pics").mkdir()
    assert client.get("/api/train/diffusion/dataset/pics/image/ghost.png").status_code == 404


# ── caption write / clear ────────────────────────────────────────────────────
def test_put_caption_roundtrip_and_clear(client, ds_root):
    folder = ds_root / "cap"
    folder.mkdir()
    _write_png(folder / "x.png")

    r = client.put(
        "/api/train/diffusion/dataset/cap/caption/x.png", json = {"caption": "a red apple"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["caption"] == "a red apple"
    assert r.json()["caption_source"] == "sidecar"
    assert (folder / "x.txt").read_text(encoding = "utf-8") == "a red apple"

    # Blank clears the sidecar.
    r = client.put("/api/train/diffusion/dataset/cap/caption/x.png", json = {"caption": "  "})
    assert r.status_code == 200
    assert r.json()["caption"] is None
    assert r.json()["caption_source"] == "none"
    assert not (folder / "x.txt").exists()


def test_put_caption_overrides_metadata_row(client, ds_root):
    # Editing a caption for an image that already has a metadata.jsonl row must take
    # effect: the sidecar edit wins over the metadata caption in the response (and in
    # the data the trainer reads), not the other way round.
    folder = ds_root / "cap"
    folder.mkdir()
    _write_png(folder / "x.png")
    (folder / "metadata.jsonl").write_text(
        json.dumps({"file_name": "x.png", "text": "from metadata"}) + "\n", encoding = "utf-8"
    )

    r = client.put(
        "/api/train/diffusion/dataset/cap/caption/x.png", json = {"caption": "edited caption"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["caption"] == "edited caption"
    assert r.json()["caption_source"] == "sidecar"


def test_put_caption_missing_image_404(client, ds_root):
    (ds_root / "cap").mkdir()
    r = client.put("/api/train/diffusion/dataset/cap/caption/ghost.png", json = {"caption": "hi"})
    assert r.status_code == 404


def test_put_caption_too_long_400(client, ds_root):
    folder = ds_root / "cap"
    folder.mkdir()
    _write_png(folder / "x.png")
    r = client.put("/api/train/diffusion/dataset/cap/caption/x.png", json = {"caption": "z" * 2001})
    assert r.status_code == 400


# ── delete ───────────────────────────────────────────────────────────────────
def test_delete_image_cleans_sidecar_and_thumb(client, ds_root):
    folder = ds_root / "d"
    folder.mkdir()
    _write_png(folder / "x.png")
    (folder / "x.txt").write_text("cap", encoding = "utf-8")
    # Generate a thumbnail so we can assert it is cleaned up too. Thumbs are keyed on
    # the full filename (stem + extension) to avoid same-stem collisions across formats.
    client.get("/api/train/diffusion/dataset/d/image/x.png?thumb=32")
    # Thumb cache key includes the extension (x.png_32.jpg), so png and jpg
    # siblings can't collide.
    assert list((folder / ".thumbs").glob("x.png_*.jpg"))

    r = client.delete("/api/train/diffusion/dataset/d/image/x.png")
    assert r.status_code == 200, r.text
    assert not (folder / "x.png").exists()
    assert not (folder / "x.txt").exists()
    assert not list((folder / ".thumbs").glob("x.png_*.jpg"))


def test_thumb_cache_key_distinguishes_same_stem_extensions(client, ds_root):
    # sample.png and sample.jpg share a stem; each must get its OWN thumbnail cache
    # file, so the labeling grid never serves one image's thumbnail for the other.
    folder = ds_root / "d"
    folder.mkdir()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(folder / "sample.png", format = "PNG")
    Image.new("RGB", (8, 8), (200, 210, 220)).save(folder / "sample.jpg", format = "JPEG")
    client.get("/api/train/diffusion/dataset/d/image/sample.png?thumb=32")
    client.get("/api/train/diffusion/dataset/d/image/sample.jpg?thumb=32")
    thumbs = sorted(p.name for p in (folder / ".thumbs").glob("*.jpg"))
    assert thumbs == ["sample.jpg_32.jpg", "sample.png_32.jpg"]


# ── traversal / validation ───────────────────────────────────────────────────
def test_dataset_name_traversal_rejected_over_http(client, ds_root):
    # A name that fails the folder-name validator returns 400, never touches disk.
    assert client.get("/api/train/diffusion/dataset/bad name!/images").status_code == 400


def test_image_filename_validation_rejects_traversal():
    from pathlib import Path

    from routes.training import _safe_dataset_image_path

    folder = Path("/tmp/some-dataset")
    for bad in ("../../etc/passwd", "/etc/passwd", "..", "sub/dir.png", "notimage.txt"):
        with pytest.raises(HTTPException) as exc:
            _safe_dataset_image_path(folder, bad)
        assert exc.value.status_code == 400
    # A plain image name resolves inside the folder.
    assert _safe_dataset_image_path(folder, "ok.png") == folder / "ok.png"


def test_clean_dataset_name_rejects_dotdot():
    from routes.training import _clean_diffusion_dataset_name
    for bad in ("../x", "a/b", "..", " "):
        with pytest.raises(HTTPException) as exc:
            _clean_diffusion_dataset_name(bad)
        assert exc.value.status_code == 400


# ── examples registry + import ───────────────────────────────────────────────
def test_list_dataset_examples(client, ds_root):
    r = client.get("/api/train/diffusion/dataset-examples")
    assert r.status_code == 200, r.text
    ids = {e["id"] for e in r.json()["examples"]}
    assert {
        "dreambooth-dog",
        "tuxemon",
        "tarot-1920",
        "smithsonian-butterflies",
        "pixel-nouns",
    } <= ids
    dog = next(e for e in r.json()["examples"] if e["id"] == "dreambooth-dog")
    assert dog["suggested_trigger"] == "a photo of sks dog"
    assert dog["license"]


def test_list_dataset_examples_large_sets(client, ds_root):
    # The two ~100-image sets: butterflies is a subject set (trigger, no caption column),
    # nouns is a captioned style set (caption column, no trigger). Both cap at 100.
    r = client.get("/api/train/diffusion/dataset-examples")
    examples = {e["id"]: e for e in r.json()["examples"]}
    butterflies = examples["smithsonian-butterflies"]
    assert butterflies["image_cap"] == 100
    assert butterflies["suggested_trigger"] == "a photo of a sks butterfly"
    assert "CC0" in butterflies["license"]
    nouns = examples["pixel-nouns"]
    assert nouns["image_cap"] == 100
    assert nouns["suggested_trigger"] is None
    assert nouns["license"] == "cc0-1.0"


class _FakeImageFeature:
    # Mimics datasets.Image so _detect_image_column matches by class name.
    pass


_FakeImageFeature.__name__ = "Image"


class _FakeDS:
    def __init__(self, rows, features):
        self._rows = rows
        self.features = features

    def __iter__(self):
        return iter(self._rows)


def _install_fake_load_dataset(monkeypatch, n_rows):
    calls = {"count": 0}
    rows = [
        {"image": Image.new("RGB", (8, 8), (i * 30 % 255, 60, 90)), "prompt": f"caption {i}"}
        for i in range(n_rows)
    ]
    features = {"image": _FakeImageFeature(), "prompt": object()}

    def fake_load(repo, **kwargs):
        calls["count"] += 1
        assert kwargs.get("split") == "train"
        return _FakeDS(rows, features)

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    return calls


def test_import_example_writes_images_and_captions(client, ds_root, monkeypatch):
    calls = _install_fake_load_dataset(monkeypatch, n_rows = 3)
    r = client.post(
        "/api/train/diffusion/dataset/import-example",
        json = {"id": "tuxemon", "name": "my-tux"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "my-tux"
    assert body["imported"] == 3
    assert body["image_count"] == 3
    assert body["caption_count"] == 3
    assert body["license"] == "cc-by-sa-3.0"
    assert body["source_repo"] == "linoyts/Tuxemon"
    folder = ds_root / "my-tux"
    assert sorted(p.name for p in folder.glob("*.png")) == [f"img_{i:04d}.png" for i in range(3)]
    assert (folder / "img_0000.txt").read_text(encoding = "utf-8") == "caption 0"

    # Idempotent: a second call does not reload or duplicate.
    r2 = client.post(
        "/api/train/diffusion/dataset/import-example",
        json = {"id": "tuxemon", "name": "my-tux"},
    )
    assert r2.status_code == 200
    assert r2.json()["imported"] == 0
    assert r2.json()["image_count"] == 3
    assert calls["count"] == 1


def test_import_example_respects_cap(client, ds_root, monkeypatch):
    _install_fake_load_dataset(monkeypatch, n_rows = 5)
    entry = next(
        e
        for e in __import__("routes.training", fromlist = ["_DATASET_EXAMPLES"])._DATASET_EXAMPLES
        if e["id"] == "tuxemon"
    )
    monkeypatch.setitem(entry, "image_cap", 2)
    r = client.post("/api/train/diffusion/dataset/import-example", json = {"id": "tuxemon"})
    assert r.status_code == 200, r.text
    assert r.json()["imported"] == 2
    assert r.json()["image_count"] == 2


def test_import_example_unknown_id_404(client, ds_root):
    r = client.post("/api/train/diffusion/dataset/import-example", json = {"id": "does-not-exist"})
    assert r.status_code == 404


def test_import_example_load_failure_maps_to_502(client, ds_root, monkeypatch):
    import datasets

    def boom(repo, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(datasets, "load_dataset", boom)
    r = client.post("/api/train/diffusion/dataset/import-example", json = {"id": "tuxemon"})
    assert r.status_code == 502
    assert "Could not import" in r.json()["detail"]
