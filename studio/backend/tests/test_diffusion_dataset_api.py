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
    # a.png has a sidecar (an explicit edit beats the metadata row), b.png is metadata-only, c.png has
    # none.
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


def test_list_images_tolerates_invalid_utf8_sidecar(client, ds_root):
    # The upload route stores .txt/.caption sidecars as raw bytes, so a sidecar can hold non-UTF-8
    # text. read_text then raises UnicodeDecodeError, a ValueError not an OSError, so an `except
    # OSError` around it 500s the whole labeling grid. One bad sidecar must read as no caption while
    # every other image still lists.
    folder = ds_root / "badutf8"
    folder.mkdir()
    _write_png(folder / "a.png")
    _write_png(folder / "b.png")
    (folder / "a.txt").write_bytes(b"\xff\xfe not valid utf-8")
    (folder / "b.txt").write_text("cap b", encoding = "utf-8")

    r = client.get("/api/train/diffusion/dataset/badutf8/images")
    assert r.status_code == 200, r.text
    recs = {rec["filename"]: rec for rec in r.json()["images"]}
    assert set(recs) == {"a.png", "b.png"}
    assert recs["a.png"]["caption"] in (None, "")
    assert recs["b.png"]["caption"] == "cap b"

    # The caption PUT returns the same record, so it must not 500 after writing either.
    put = client.put(
        "/api/train/diffusion/dataset/badutf8/caption/a.png", json = {"caption": "fixed"}
    )
    assert put.status_code == 200, put.text
    assert put.json()["caption"] == "fixed"


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
    # Editing a caption for an image that already has a metadata.jsonl row must take effect: the
    # sidecar edit wins over the metadata caption, in the response and in what the trainer reads.
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
    # Generate a thumbnail so we can assert it is cleaned up too. Thumbs are keyed on the full
    # filename to avoid same-stem collisions across formats.
    client.get("/api/train/diffusion/dataset/d/image/x.png?thumb=32")
    assert list((folder / ".thumbs").glob("x.png_*.jpg"))

    r = client.delete("/api/train/diffusion/dataset/d/image/x.png")
    assert r.status_code == 200, r.text
    assert not (folder / "x.png").exists()
    assert not (folder / "x.txt").exists()
    assert not list((folder / ".thumbs").glob("x.png_*.jpg"))


def test_thumb_cache_key_distinguishes_same_stem_extensions(client, ds_root):
    # sample.png and sample.jpg share a stem; each must get its OWN thumbnail cache file, so the grid
    # never serves one image's thumbnail for the other.
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
    # The two ~100-image sets: butterflies is a subject set (trigger, no caption column), nouns a
    # captioned style set. Both cap at 100.
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


def _install_fake_load_dataset(
    monkeypatch,
    n_rows,
    features = "default",
    streamable = True,
):
    calls = {"count": 0, "streaming": [], "features": features}
    rows = [
        {"image": Image.new("RGB", (8, 8), (i * 30 % 255, 60, 90)), "prompt": f"caption {i}"}
        for i in range(n_rows)
    ]
    if features == "default":
        features = {"image": _FakeImageFeature(), "prompt": object()}

    def fake_load(repo, **kwargs):
        calls["count"] += 1
        calls["streaming"].append(bool(kwargs.get("streaming")))
        assert kwargs.get("split") == "train"
        if kwargs.get("streaming") and not streamable:
            raise ValueError("Loading a dataset cached in a LocalFileSystem is not supported")
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


def test_import_example_streams_instead_of_preparing_the_whole_split(client, ds_root, monkeypatch):
    # The cap keeps 10-100 rows, while the curated repos run to 49,859 rows / 328 MB
    # (m1guelpf/nouns), all of which a prepared load downloads and converts before the first row is
    # read. The import must ask for a streamed split.
    calls = _install_fake_load_dataset(monkeypatch, n_rows = 3)
    r = client.post("/api/train/diffusion/dataset/import-example", json = {"id": "tuxemon"})
    assert r.status_code == 200, r.text
    assert r.json()["imported"] == 3
    assert calls["streaming"] == [True]


def test_import_example_falls_back_when_the_repo_cannot_stream(client, ds_root, monkeypatch):
    # A repo with a loading script or no listed data files cannot stream; the one-click import must
    # still work through the prepared load rather than 502.
    calls = _install_fake_load_dataset(monkeypatch, n_rows = 3, streamable = False)
    r = client.post("/api/train/diffusion/dataset/import-example", json = {"id": "tuxemon"})
    assert r.status_code == 200, r.text
    assert r.json()["imported"] == 3
    assert calls["streaming"] == [True, False]


def test_import_example_resolves_columns_from_the_first_row_without_features(
    client, ds_root, monkeypatch
):
    # A streamed dataset can arrive with no feature metadata to inspect, so the image and caption
    # columns come from the first row instead.
    _install_fake_load_dataset(monkeypatch, n_rows = 2, features = None)
    r = client.post("/api/train/diffusion/dataset/import-example", json = {"id": "tuxemon"})
    assert r.status_code == 200, r.text
    assert r.json()["imported"] == 2
    assert r.json()["caption_count"] == 2


def test_import_example_without_an_image_column_maps_to_502(client, ds_root, monkeypatch):
    # No image anywhere in the row: still a clean 502, not a KeyError 500.
    _install_fake_load_dataset(monkeypatch, n_rows = 2, features = {"prompt": object()})
    r = client.post("/api/train/diffusion/dataset/import-example", json = {"id": "tuxemon"})
    assert r.status_code == 502


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


# ── upload: same-stem image collision ────────────────────────────────────────
def _jpg_bytes(color = (30, 120, 200), size = (8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format = "JPEG")
    return buf.getvalue()


def _upload(client, name, files):
    # files: list of (filename, bytes). Content type is irrelevant to the route (it keys off the
    # extension), so send everything as octet-stream.
    parts = [("files", (fn, data, "application/octet-stream")) for fn, data in files]
    return client.post("/api/train/diffusion/dataset", data = {"name": name}, files = parts)


def test_upload_rejects_same_stem_different_extension(client, ds_root):
    # sample.png and sample.jpg share the stem "sample", so both map to one sample.txt sidecar and
    # keeping both would silently corrupt captions. The second must 400.
    assert _upload(client, "styleset", [("sample.png", _png_bytes())]).status_code == 200
    dup = _upload(client, "styleset", [("sample.jpg", _jpg_bytes())])
    assert dup.status_code == 400
    assert "Duplicate image name" in dup.json()["detail"]
    # The rejected image never landed on disk: only sample.png survives.
    folder = ds_root / "styleset"
    assert sorted(p.name for p in folder.iterdir() if p.suffix != ".txt") == ["sample.png"]


def test_upload_same_stem_collision_within_one_batch(client, ds_root):
    # The scan must cover files uploaded earlier IN THE SAME batch, not just those already on disk.
    r = _upload(client, "styleset", [("sample.png", _png_bytes()), ("sample.jpg", _jpg_bytes())])
    assert r.status_code == 400
    assert "Duplicate image name" in r.json()["detail"]


def test_upload_rejects_exact_duplicate_name_within_one_batch(client, ds_root):
    # Two parts with the SAME name in ONE multipart batch are distinct files, and the staged commit
    # would let the later replace silently discard the earlier one while `uploaded` counts both. The
    # batch must be rejected whole; re-sending a name in a SEPARATE upload stays a deliberate
    # overwrite.
    r = _upload(
        client,
        "styleset",
        [("sample.png", _png_bytes((10, 20, 30))), ("sample.png", _png_bytes((90, 90, 90)))],
    )
    assert r.status_code == 400
    assert "more than once" in r.json()["detail"]
    assert not (ds_root / "styleset" / "sample.png").exists()  # all-or-nothing
    # Caption files collide at one destination the same way.
    r = _upload(client, "styleset", [("sample.txt", b"a"), ("sample.txt", b"b")])
    assert r.status_code == 400
    assert "more than once" in r.json()["detail"]
    # A STEM case variant pair (Cat.png vs cat.png) stays exempt: it is one file / an overwrite on
    # case-insensitive filesystems, and on Linux the two write separate sidecars.
    r = _upload(client, "styleset", [("Cat.png", _png_bytes()), ("cat.png", _png_bytes())])
    assert r.status_code == 200


def test_upload_rejects_extension_case_variant_sidecar_collision(client, ds_root):
    # An EXTENSION-case variant pair (dog.PNG vs dog.png) has exactly equal stems: on a case-sensitive
    # filesystem both land and both resolve to ONE dog.txt sidecar. Must 400 within one batch and
    # against a file already on disk.
    r = _upload(client, "styleset", [("dog.PNG", _png_bytes()), ("dog.png", _png_bytes())])
    assert r.status_code == 400
    assert "Duplicate image name" in r.json()["detail"]
    folder = ds_root / "styleset"
    assert not any(p.suffix.lower() == ".png" for p in folder.iterdir())  # all-or-nothing
    assert _upload(client, "styleset", [("dog.png", _png_bytes())]).status_code == 200
    dup = _upload(client, "styleset", [("dog.PNG", _png_bytes())])
    assert dup.status_code == 400
    assert "Duplicate image name" in dup.json()["detail"]
    assert sorted(p.name for p in folder.iterdir() if p.suffix != ".txt") == ["dog.png"]


def test_upload_allows_exact_name_overwrite_and_caption_sidecar(client, ds_root):
    # Re-uploading the EXACT same name is an allowed overwrite, and a .txt caption for the same stem
    # is the intended kohya flow: neither is a same-stem image collision.
    assert (
        _upload(client, "styleset", [("sample.png", _png_bytes((10, 20, 30)))]).status_code == 200
    )
    assert (
        _upload(client, "styleset", [("sample.png", _png_bytes((90, 90, 90)))]).status_code == 200
    )
    assert _upload(client, "styleset", [("sample.txt", b"a caption")]).status_code == 200
    folder = ds_root / "styleset"
    assert (folder / "sample.png").is_file()
    assert (folder / "sample.txt").read_text(encoding = "utf-8") == "a caption"


# ── import: promotion is all-or-nothing ──────────────────────────────────────
def test_import_promotion_leaves_no_partial_dataset_on_failure(ds_root, monkeypatch):
    # The staging dir is promoted into the dataset folder in one atomic rename. If that rename fails,
    # the folder must be left with NO images rather than a half-filled dataset the image_count>0
    # idempotency check would accept as complete on retry. Simulate the rename failing, assert nothing
    # partial is left, and assert a retry re-imports cleanly.
    import os

    # A client that returns the 500 (as production does) instead of re-raising.
    app = FastAPI()
    app.include_router(training_router, prefix = "/api/train")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    noraise = TestClient(app, raise_server_exceptions = False)

    calls = _install_fake_load_dataset(monkeypatch, n_rows = 3)
    folder = ds_root / "my-tux"
    real_replace = os.replace

    def flaky_replace(src, dst, *a, **k):
        # Only sabotage the staging to folder promotion; leave every other rename working.
        if str(dst) == str(folder):
            raise OSError("simulated crash during promotion")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", flaky_replace)
    r = noraise.post(
        "/api/train/diffusion/dataset/import-example",
        json = {"id": "tuxemon", "name": "my-tux"},
    )
    assert r.status_code == 500
    # No half-filled dataset: the folder holds zero images and no stray staging dir.
    assert list(ds_root.glob("my-tux/*.png")) == []
    assert not any(p.name.startswith(".my-tux.import-") for p in ds_root.iterdir())

    # Retry with the promotion working: a clean, complete import (idempotency did not short-circuit).
    monkeypatch.setattr(os, "replace", real_replace)
    r2 = noraise.post(
        "/api/train/diffusion/dataset/import-example",
        json = {"id": "tuxemon", "name": "my-tux"},
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["imported"] == 3
    assert calls["count"] == 2  # the failed attempt did not leave a dataset that blocks a reload


def test_upload_rolls_back_when_a_later_promotion_fails(ds_root, monkeypatch):
    # Re-uploading a.txt and b.txt where the SECOND commit fails must roll back the first overwrite,
    # so both originals survive and no stray temp/backup files remain.
    from pathlib import Path

    app = FastAPI()
    app.include_router(training_router, prefix = "/api/train")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    noraise = TestClient(app, raise_server_exceptions = False)

    folder = ds_root / "styleset"
    folder.mkdir()
    (folder / "a.txt").write_bytes(b"ORIGINAL-A")
    (folder / "b.txt").write_bytes(b"ORIGINAL-B")

    real_replace = Path.replace
    state = {"failed": False}

    def flaky_replace(self, target, *a, **k):
        # Fail once on the tmp to b.txt promotion only (not the backup restore), so rollback works.
        if (
            not state["failed"]
            and str(target).endswith("b.txt")
            and self.name.startswith(".upload-")
            and not self.name.startswith(".upload-backup-")
        ):
            state["failed"] = True
            raise OSError("simulated second commit failure")
        return real_replace(self, target, *a, **k)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    parts = [
        ("files", ("a.txt", b"NEW-A", "application/octet-stream")),
        ("files", ("b.txt", b"NEW-B", "application/octet-stream")),
    ]
    r = noraise.post("/api/train/diffusion/dataset", data = {"name": "styleset"}, files = parts)
    assert r.status_code == 500
    monkeypatch.setattr(Path, "replace", real_replace)

    # Both originals are intact: no partial overwrite of the live dataset.
    assert (folder / "a.txt").read_bytes() == b"ORIGINAL-A"
    assert (folder / "b.txt").read_bytes() == b"ORIGINAL-B"
    # No staging or backup artifacts left behind.
    assert not list(folder.glob(".upload-*.part"))
    assert not list(folder.glob(".upload-backup-*.part"))


def test_upload_rechecks_training_state_before_commit(ds_root, monkeypatch):
    # A /diffusion/start that reserves the training slot AFTER the upload passed its entry guard but
    # BEFORE the commit must not have its dataset mutated: the recheck just before the promotion
    # catches the now-active run, 409s, and leaves the on-disk dataset untouched.
    import routes.training as tr

    folder = ds_root / "styleset"
    folder.mkdir()
    (folder / "a.png").write_bytes(_png_bytes())  # a pre-existing image the overwrite would clobber

    calls = {"n": 0}

    def fake_active():
        # Inactive at the entry guard (call 1), active by the pre-commit recheck (call 2+): the training
        # run started while the upload was streaming.
        calls["n"] += 1
        return calls["n"] >= 2

    monkeypatch.setattr(tr, "_diffusion_training_active", fake_active)

    app = FastAPI()
    app.include_router(training_router, prefix = "/api/train")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    noraise = TestClient(app, raise_server_exceptions = False)

    parts = [("files", ("a.png", _png_bytes(color = (1, 2, 3)), "image/png"))]
    r = noraise.post("/api/train/diffusion/dataset", data = {"name": "styleset"}, files = parts)
    assert r.status_code == 409
    assert calls["n"] >= 2  # both the entry guard and the pre-commit recheck ran
    # The dataset is untouched: the original image survives and no staged temp lingers.
    assert (folder / "a.png").read_bytes() == _png_bytes()
    assert not list(folder.glob(".upload-*.part"))
    assert not list(folder.glob(".upload-backup-*.part"))


def test_resolve_dataset_folder_rejects_symlink(ds_root, tmp_path):
    # A dataset dir that is a symlink outside the datasets root must be rejected, else delete /
    # caption / read could operate on external files through the link.
    from routes.training import _resolve_dataset_folder

    external = tmp_path / "external"
    external.mkdir()
    (external / "victim.png").write_bytes(_png_bytes())
    (ds_root / "linked").symlink_to(external, target_is_directory = True)

    with pytest.raises(HTTPException) as exc:
        _resolve_dataset_folder("linked")
    assert exc.value.status_code == 400


def test_upload_through_symlinked_dataset_cannot_escape_root(client, ds_root, tmp_path):
    # An upload to a dataset name that is a symlink to an external directory must be refused (400)
    # before any bytes are written.
    external = tmp_path / "external"
    external.mkdir()
    (ds_root / "linked").symlink_to(external, target_is_directory = True)

    r = _upload(client, "linked", [("intruder.png", _png_bytes())])
    assert r.status_code == 400
    assert "symbolic link" in r.json()["detail"]
    # Nothing was written through the link into the external directory.
    assert not (external / "intruder.png").exists()
    assert not any(external.iterdir())


def test_delete_through_symlinked_dataset_cannot_escape_root(client, ds_root, tmp_path):
    # A DELETE inside a symlinked dataset dir is refused (400) and the external file survives.
    external = tmp_path / "external"
    external.mkdir()
    victim = external / "victim.png"
    _write_png(victim)
    (ds_root / "linked").symlink_to(external, target_is_directory = True)

    r = client.delete("/api/train/diffusion/dataset/linked/image/victim.png")
    assert r.status_code == 400
    assert victim.exists()  # the external file survives


def test_delete_image_with_glob_chars_only_removes_own_thumbs(client, ds_root):
    # Deleting a filename with glob metacharacters (e.g. "[ab].png") must remove only its own
    # thumbnails, not a sibling's that the raw glob would spuriously match.
    from urllib.parse import quote

    folder = ds_root / "d"
    folder.mkdir()
    _write_png(folder / "[ab].png")
    _write_png(folder / "a.png")
    thumbs = folder / ".thumbs"
    thumbs.mkdir()
    (thumbs / "[ab].png_32.jpg").write_bytes(b"own")
    (thumbs / "a.png_32.jpg").write_bytes(b"sibling")

    r = client.delete("/api/train/diffusion/dataset/d/image/" + quote("[ab].png", safe = ""))
    assert r.status_code == 200, r.text
    # Its own thumbnail is gone; the sibling a.png's thumbnail is untouched.
    assert not (thumbs / "[ab].png_32.jpg").exists()
    assert (thumbs / "a.png_32.jpg").exists()
    assert not (folder / "[ab].png").exists()
    assert (folder / "a.png").exists()


def test_import_preserves_unrelated_files_when_folder_not_empty(client, ds_root, monkeypatch):
    # If the target folder already holds unrelated NON-image files (so image_count is still 0 and the
    # import runs), the atomic rmdir refuses and the code falls back to a per-file move: the images
    # are imported AND the pre-existing file is preserved.
    _install_fake_load_dataset(monkeypatch, n_rows = 3)
    folder = ds_root / "my-tux"
    folder.mkdir(parents = True)
    (folder / "notes.md").write_text("keep me", encoding = "utf-8")
    r = client.post(
        "/api/train/diffusion/dataset/import-example",
        json = {"id": "tuxemon", "name": "my-tux"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["imported"] == 3
    assert sorted(p.name for p in folder.glob("*.png")) == [f"img_{i:04d}.png" for i in range(3)]
    assert (folder / "notes.md").read_text(encoding = "utf-8") == "keep me"


def test_a_second_concurrent_import_of_the_same_name_is_refused(client, ds_root, monkeypatch):
    """The training interlock counts mutations rather than excluding them, so two imports into the
    same empty name both got through; the loser then merged its files into the winner's folder and
    produced a dataset built from two sources. The second request is refused instead."""
    import routes.training as training_route

    _install_fake_load_dataset(monkeypatch, n_rows = 3)
    folder = ds_root / "my-tux"
    folder.mkdir(parents = True)
    # Stand in for the in-flight import: the lock is held for the whole materialize + promote.
    held = training_route._dataset_import_lock(folder)
    assert held.acquire(blocking = False)
    try:
        r = client.post(
            "/api/train/diffusion/dataset/import-example",
            json = {"id": "tuxemon", "name": "my-tux"},
        )
    finally:
        held.release()
    assert r.status_code == 409, r.text
    assert "already running" in r.json()["detail"]
    # Nothing was written into the folder the other import owns.
    assert list(folder.glob("*.png")) == []

    # Once it is free the same request imports normally.
    ok = client.post(
        "/api/train/diffusion/dataset/import-example",
        json = {"id": "tuxemon", "name": "my-tux"},
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["imported"] == 3


def test_an_import_into_a_folder_filled_meanwhile_does_not_merge(client, ds_root, monkeypatch):
    """The re-check under the lock: a request that waited while another import promoted its
    staging dir must return the folder as-is, not move its own files on top."""
    import routes.training as training_route

    _install_fake_load_dataset(monkeypatch, n_rows = 3)
    folder = ds_root / "my-tux"
    folder.mkdir(parents = True)
    (folder / "img_0000.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    (folder / "img_0000.txt").write_text("from the first import", encoding = "utf-8")

    r = client.post(
        "/api/train/diffusion/dataset/import-example",
        json = {"id": "tuxemon", "name": "my-tux"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["imported"] == 0
    assert (folder / "img_0000.txt").read_text(encoding = "utf-8") == "from the first import"
    assert not training_route._dataset_import_lock(folder).locked()
