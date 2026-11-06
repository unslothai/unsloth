# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's dataset media cells and image strings open only files it may read, and never reach
the Hub with the server's own token; the owner decodes as before."""

from __future__ import annotations

import io
import os

import pytest
from fastapi import HTTPException

from auth import policy
from hub.schemas.datasets import CheckFormatRequest
from hub.services.datasets import formatting
from utils import hf_cache_settings
from utils.account_context import AccountContext, run_as
from utils.paths import storage_roots

ALICE = AccountContext("a" * 32, "alice")


def _png(color) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buf, format = "PNG")
    return buf.getvalue()


@pytest.fixture(autouse = True)
def guarded(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(formatting, "ensure_audio_decoding", lambda: True)
    cache = tmp_path / "hub-cache"
    cache.mkdir()
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: str(cache))
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [cache])
    try:
        from utils.security import dataset_media
    except ImportError:  # a tree without the guard: the tests below show what it lets through
        return
    dataset_media.install()
    dataset_media.install()


@pytest.fixture
def outside_png(tmp_path):
    path = tmp_path / "owner-private" / "secret.png"
    path.parent.mkdir()
    path.write_bytes(_png((255, 0, 0)))
    return path


def _upload(rows, account = ALICE) -> str:
    import datasets
    def write():
        root = storage_roots.dataset_uploads_root()
        root.mkdir(parents = True, exist_ok = True)
        target = root / "pictures.parquet"
        features = datasets.Features({"image": datasets.Image(), "text": datasets.Value("string")})
        datasets.Dataset.from_list(rows, features = features).to_parquet(str(target))
        return str(target)

    return run_as(account, write)


def _check(name, account = ALICE):
    return run_as(
        account, formatting.check_format_response, CheckFormatRequest(dataset_name = name), None
    )


def test_managed_image_cell_naming_a_host_file_is_refused(outside_png):
    name = _upload([{"image": {"bytes": None, "path": str(outside_png)}, "text": "a cat"}])
    with pytest.raises(HTTPException) as exc:
        _check(name)
    assert 400 <= exc.value.status_code < 500


def test_managed_chained_cell_whose_first_hop_reads_a_host_file_is_refused(outside_png):
    # fsspec opens the first hop of `file://x::hf://...`; only archive hops read through.
    # Windows: tmp_path (C:) and the checkout (D:) can sit on different drives, so relpath
    # raises; the drive-letter form `file://C:/...::hf://...` exercises the same guard.
    try:
        first_hop = os.path.relpath(outside_png)
    except ValueError:
        first_hop = outside_png.as_posix()
    path = f"file://{first_hop}::hf://datasets/a/b@main/c.png"
    name = _upload([{"image": {"bytes": None, "path": path}, "text": "a cat"}])
    with pytest.raises(HTTPException) as exc:
        _check(name)
    assert 400 <= exc.value.status_code < 500


def test_managed_zip_member_of_own_archive_still_previews():
    import zipfile

    def own_zip():
        path = storage_roots.dataset_uploads_root() / "own.zip"
        path.parent.mkdir(parents = True, exist_ok = True)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("in.png", _png((0, 0, 255)))
        return f"zip://in.png::{path}"

    name = _upload([{"image": {"bytes": None, "path": run_as(ALICE, own_zip)}, "text": "x"}])
    assert _check(name).preview_samples[0]["image"]["type"] == "image"


def test_managed_image_cell_with_embedded_bytes_still_previews():
    name = _upload([{"image": {"bytes": _png((0, 255, 0)), "path": None}, "text": "a cat"}])
    response = _check(name)
    assert response.preview_samples
    assert response.preview_samples[0]["image"]["type"] == "image"


def test_managed_image_cell_inside_own_workspace_still_previews():
    def own_png():
        path = storage_roots.dataset_uploads_root() / "own.png"
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(_png((0, 0, 255)))
        return str(path)

    name = _upload([{"image": {"bytes": None, "path": run_as(ALICE, own_png)}, "text": "x"}])
    assert _check(name).preview_samples[0]["image"]["type"] == "image"


def test_owner_image_cell_naming_a_host_file_is_unchanged(outside_png):
    from utils.account_context import OWNER

    name = _upload([{"image": {"bytes": None, "path": str(outside_png)}, "text": "a cat"}], OWNER)
    response = _check(name, OWNER)
    assert response.preview_samples[0]["image"]["type"] == "image"


@pytest.fixture
def hub_fetches(monkeypatch):
    from datasets.features import image as image_module

    seen = []

    def fake_xopen(
        path,
        mode = "r",
        *args,
        download_config = None,
        **kwargs,
    ):
        seen.append((path, download_config.token if download_config else None))
        return io.BytesIO(_png((1, 2, 3)))

    monkeypatch.setattr(image_module, "xopen", fake_xopen)
    return seen


def _decode(
    value,
    tokens,
    account = ALICE,
):
    from datasets import Image
    return run_as(account, Image().decode_example, value, token_per_repo_id = tokens)


@pytest.mark.parametrize(
    "path",
    [
        "hf://datasets/other/private@main/secret.png",
        "{endpoint}/datasets/other/private/resolve/main/secret.png",
    ],
)
def test_managed_hub_cell_of_another_repo_never_uses_the_server_token(hub_fetches, path):
    from datasets import config

    path = path.format(endpoint = config.HF_ENDPOINT)
    _decode({"bytes": None, "path": path}, {"alice/own": "alice-token"})
    assert hub_fetches == [(path, False)]


def test_managed_hub_cell_of_its_own_repo_keeps_its_token(hub_fetches):
    path = "hf://datasets/alice/own@main/a.png"
    _decode({"bytes": None, "path": path}, {"alice/own": "alice-token"})
    assert hub_fetches == [(path, "alice-token")]


@pytest.mark.parametrize(
    "path",
    [
        "http://127.0.0.1:9/secret.png",
        "hf://datasets/other/private/secret.png",
        "s3://bucket/x.png",
    ],
)
def test_managed_cell_naming_another_url_is_never_fetched(hub_fetches, path):
    with pytest.raises(PermissionError):
        _decode({"bytes": None, "path": path}, {})
    assert hub_fetches == []


def test_owner_hub_cell_is_unchanged(hub_fetches):
    from utils.account_context import OWNER

    path = "hf://datasets/other/private@main/secret.png"
    _decode({"bytes": None, "path": path}, {}, OWNER)
    assert hub_fetches == [(path, None)]


def test_managed_vlm_string_image_paths_stay_in_the_account(outside_png):
    from utils.account_context import OWNER
    from utils.datasets.format_conversion import convert_llava_to_vlm_format

    def rows(path):
        return [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "index": 0}, {"type": "text", "text": "hi"}],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "a red square"}]},
                ],
                "images": [path],
            }
        ]

    with pytest.raises(Exception) as exc:
        run_as(ALICE, convert_llava_to_vlm_format, rows(str(outside_png)))
    assert isinstance(exc.value, (PermissionError, HTTPException))
    assert run_as(OWNER, convert_llava_to_vlm_format, rows(str(outside_png)))

    def own_png():
        path = storage_roots.dataset_uploads_root() / "own.png"
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(_png((0, 0, 255)))
        return str(path)

    assert run_as(ALICE, convert_llava_to_vlm_format, rows(run_as(ALICE, own_png)))


def test_install_is_idempotent():
    from datasets import Image

    from utils.security import dataset_media

    decode = Image.decode_example
    dataset_media.install()
    assert Image.decode_example is decode
    assert dataset_media.guard_decoder(decode) is decode


def test_managed_snapshot_files_of_its_training_dataset_still_decode(tmp_path, monkeypatch):
    from hub.services.models import account_access
    from utils.security import dataset_media

    monkeypatch.setattr(dataset_media, "_dataset_repos", set())
    monkeypatch.setattr(account_access, "model_grants", lambda: set())
    monkeypatch.setattr(account_access, "repo_is_public", lambda *args, **kwargs: False)
    image = tmp_path / "hub-cache" / "datasets--alice--private" / "snapshots" / "rev" / "a.png"
    image.parent.mkdir(parents = True)
    image.write_bytes(_png((9, 9, 9)))
    cell = {"bytes": None, "path": str(image)}

    with pytest.raises(PermissionError):
        _decode(cell, {})
    dataset_media.allow_dataset_repos("alice/private")
    assert _decode(cell, {}).size == (4, 4)
    dataset_media.allow_dataset_repos("someone/else")
    other = tmp_path / "hub-cache" / "datasets--bob--private" / "snapshots" / "rev" / "b.png"
    other.parent.mkdir(parents = True)
    other.write_bytes(_png((9, 9, 9)))
    with pytest.raises(PermissionError):
        _decode({"bytes": None, "path": str(other)}, {})


@pytest.mark.parametrize(
    "path",
    [
        "http://127.0.0.1:9/hf://datasets/other/private@main/secret.png",
        "http://127.0.0.1:9/{endpoint}/datasets/other/private/resolve/main/secret.png",
        "{endpoint}.example.invalid/datasets/other/private/resolve/main/secret.png",
        "zip://secret.png::http://127.0.0.1:9/hf://datasets/other/private@main/a.zip",
        "http://127.0.0.1:9/secret.png::hf://datasets/other/private@main/a.zip",
        "hf://datasets/other/private@main/secret.png\nhttp://127.0.0.1:9/",
    ],
)
def test_managed_url_merely_containing_a_hub_path_is_never_fetched(hub_fetches, path):
    from datasets import config
    with pytest.raises(PermissionError):
        _decode({"bytes": None, "path": path.format(endpoint = config.HF_ENDPOINT)}, {})
    assert hub_fetches == []


def test_managed_archive_member_on_the_hub_keeps_its_token(hub_fetches):
    path = "zip://a.png::hf://datasets/alice/own@main/images.zip"
    _decode({"bytes": None, "path": path}, {"alice/own": "alice-token"})
    assert hub_fetches == [(path, "alice-token")]


@pytest.fixture
def extracted_root(tmp_path, monkeypatch):
    from datasets import config

    from utils.security import dataset_media

    root = tmp_path / "datasets-cache" / "downloads" / "extracted"
    root.mkdir(parents = True)
    monkeypatch.setattr(config, "EXTRACTED_DATASETS_PATH", str(root))
    monkeypatch.setattr(dataset_media, "_extracted_sources", {}, raising = False)
    return root


def _zip(path, color):
    import zipfile

    path.parent.mkdir(parents = True, exist_ok = True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("member.png", _png(color))
    return path


def _extract(archive, account):
    from datasets.utils.extract import ExtractManager
    return run_as(account, ExtractManager().extract, str(archive))


def test_managed_member_of_its_own_extracted_archive_still_decodes(extracted_root):
    from utils.security import dataset_media

    archive = run_as(
        ALICE, lambda: _zip(storage_roots.dataset_uploads_root() / "own.zip", (5, 6, 7))
    )
    member = {"bytes": None, "path": str(extracted_root / _extract(archive, ALICE) / "member.png")}
    assert _decode(member, {}).getpixel((0, 0)) == (5, 6, 7)
    # A later process reads the record left beside the extraction.
    getattr(dataset_media, "_extracted_sources", {}).clear()
    assert _decode(member, {}).getpixel((0, 0)) == (5, 6, 7)


def test_managed_member_of_another_accounts_extracted_archive_is_refused(extracted_root, tmp_path):
    from utils.account_context import OWNER

    from utils.security import dataset_media

    archive = _zip(tmp_path / "owner-private" / "private.zip", (1, 1, 1))
    member = {"bytes": None, "path": str(extracted_root / _extract(archive, OWNER) / "member.png")}
    with pytest.raises(PermissionError):
        _decode(member, {})
    assert _decode(member, {}, OWNER).getpixel((0, 0)) == (1, 1, 1)
    getattr(dataset_media, "_extracted_sources", {}).clear()
    with pytest.raises(PermissionError):
        _decode(member, {})

    # An extraction nothing recorded (or whose record names a different archive) has no readable source.
    unrecorded = extracted_root / ("0" * 64) / "member.png"
    unrecorded.parent.mkdir()
    unrecorded.write_bytes(_png((2, 2, 2)))
    own = run_as(ALICE, lambda: _zip(storage_roots.dataset_uploads_root() / "own.zip", (3, 3, 3)))
    (extracted_root / ("0" * 64 + ".unsloth-source")).write_text(str(own))
    with pytest.raises(PermissionError):
        _decode({"bytes": None, "path": str(unrecorded)}, {})


def test_managed_extraction_record_is_read_once_per_process(extracted_root, monkeypatch):
    from utils.security import dataset_media

    archive = run_as(
        ALICE, lambda: _zip(storage_roots.dataset_uploads_root() / "own.zip", (5, 6, 7))
    )
    member = {"bytes": None, "path": str(extracted_root / _extract(archive, ALICE) / "member.png")}
    dataset_media._extracted_sources.clear()
    reads = []
    real_open = os.open

    def counting_open(file, *args, **kwargs):
        if str(file).endswith(".unsloth-source"):
            reads.append(file)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(dataset_media.os, "open", counting_open)
    for _ in range(3):
        assert _decode(member, {}).getpixel((0, 0)) == (5, 6, 7)
    assert len(reads) == 1


def test_managed_extraction_record_is_published_whole(extracted_root):
    from utils.security import dataset_media

    archive = run_as(
        ALICE, lambda: _zip(storage_roots.dataset_uploads_root() / "own.zip", (5, 6, 7))
    )
    from datasets.utils.extract import ExtractManager

    top = ExtractManager(cache_dir = None)._get_output_path(str(archive))
    # A record left empty by an interrupted writer is replaced by the next extraction.
    (extracted_root / (os.path.basename(top) + ".unsloth-source")).write_text("")
    member = {"bytes": None, "path": str(extracted_root / _extract(archive, ALICE) / "member.png")}
    dataset_media._extracted_sources.clear()
    assert _decode(member, {}).getpixel((0, 0)) == (5, 6, 7)
    assert not [p for p in os.listdir(extracted_root) if ".unsloth-source." in p]


def test_managed_oversized_extraction_record_is_refused_unread(extracted_root, monkeypatch):
    from utils.security import dataset_media

    archive = run_as(
        ALICE, lambda: _zip(storage_roots.dataset_uploads_root() / "own.zip", (5, 6, 7))
    )
    top = _extract(archive, ALICE)
    member = {"bytes": None, "path": str(extracted_root / top / "member.png")}
    sidecar = extracted_root / (top + ".unsloth-source")
    # A record padded far past any path length is refused without being read whole.
    sidecar.write_text(str(archive) + "\0" * (16 << 20))
    dataset_media._extracted_sources.clear()
    import tracemalloc

    tracemalloc.start()
    try:
        assert dataset_media._extracted_source(member["path"]) == ""
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < (1 << 20)
    with pytest.raises(PermissionError):
        _decode(member, {})


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason = "needs FIFOs")
def test_managed_extraction_record_planted_as_fifo_is_refused_without_blocking(
    extracted_root, monkeypatch
):
    import threading

    from utils.security import dataset_media

    archive = run_as(
        ALICE, lambda: _zip(storage_roots.dataset_uploads_root() / "own.zip", (5, 6, 7))
    )
    top = _extract(archive, ALICE)
    member = {"bytes": None, "path": str(extracted_root / top / "member.png")}
    sidecar = extracted_root / (top + ".unsloth-source")
    sidecar.unlink()
    os.mkfifo(sidecar)
    dataset_media._extracted_sources.clear()
    result = []
    reader = threading.Thread(
        target = lambda: result.append(dataset_media._extracted_source(member["path"])),
        daemon = True,
    )
    reader.start()
    reader.join(5)
    if reader.is_alive():
        # Unblock the stuck open so the thread can end, then fail.
        with open(sidecar, "wb"):
            pass
        reader.join(5)
        pytest.fail("reading a FIFO extraction record blocked")
    assert result == [""]
    with pytest.raises(PermissionError):
        _decode(member, {})
