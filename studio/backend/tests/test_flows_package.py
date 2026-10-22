# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sharing: export modes strip secrets and local ids; import refuses hostile packages and never trusts names."""

import asyncio
import hashlib
import io
import json
import stat
import zipfile

import pytest

from core.flows import assets as asset_store
from core.flows import package
from core.flows.package import PackageError, read_package

from ._flows_helpers import doc, node, run_async


def _png() -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGBA", (4, 4), (200, 30, 90, 255)).save(out, "PNG")
    return out.getvalue()


PNG = _png()
PNG_SHA = hashlib.sha256(PNG).hexdigest()


def _zip(
    entries: dict[str, bytes],
    *,
    link: str | None = None,
    method = zipfile.ZIP_STORED,
) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", method) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
        if link:
            info = zipfile.ZipInfo(link)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, "/etc/passwd")
    return out.getvalue()


def _flow(**extra) -> bytes:
    return json.dumps(doc([node("a", "input.text", value = "x")], **extra)).encode()


def test_plain_json_imports():
    parsed, files, report = read_package(_flow())
    assert parsed["nodes"][0]["id"] == "a" and files == {} and report["warnings"] == []


def test_package_with_matching_asset_imports_and_stores():
    media = {"kind": "image", "ref": f"sha256:{PNG_SHA}", "url": "x"}
    data = _zip(
        {
            "flow.json": json.dumps(doc([node("i", "input.image", asset = media)])).encode(),
            f"assets/{PNG_SHA}.png": PNG,
        }
    )
    parsed, files, report = read_package(data)
    assert PNG_SHA in files
    assert parsed["assets"][0]["id"] == f"sha256:{PNG_SHA}"
    package.store_assets(files)
    assert asset_store.read(PNG_SHA)[0] == PNG


@pytest.mark.parametrize(
    "name",
    [
        "../evil.png",
        "/abs/flow.json",
        "assets/../../x",
        "C:/x.png",
        "assets\\x.png",
        "notes.txt",
        "assets/abc.png",
    ],
)
def test_unexpected_or_unsafe_names_are_refused(name):
    with pytest.raises(PackageError):
        read_package(_zip({"flow.json": _flow(), name: b"x"}))


def test_symlink_entry_is_refused():
    with pytest.raises(PackageError, match = "link"):
        read_package(_zip({"flow.json": _flow()}, link = "README.md"))


def test_duplicate_entries_are_refused():
    import warnings

    warnings.simplefilter("ignore", UserWarning)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("flow.json", _flow())
        zf.writestr("README.md", b"a")
        zf.writestr("README.md", b"b")
    with pytest.raises(PackageError, match = "twice"):
        read_package(out.getvalue())


def test_bad_checksum_and_wrong_type_are_refused():
    with pytest.raises(PackageError, match = "checksum"):
        read_package(_zip({"flow.json": _flow(), f"assets/{'0' * 64}.png": PNG}))
    wav_named_png = b"RIFF\x24\x00\x00\x00WAVEfmt "
    digest = hashlib.sha256(wav_named_png).hexdigest()
    with pytest.raises(PackageError, match = "media type"):
        read_package(_zip({"flow.json": _flow(), f"assets/{digest}.png": wav_named_png}))
    text = b"#!/bin/sh\necho hi"
    digest = hashlib.sha256(text).hexdigest()
    with pytest.raises(PackageError):
        read_package(_zip({"flow.json": _flow(), f"assets/{digest}.png": text}))


def test_zip_bomb_ratio_is_refused():
    huge = b"\x00" * (8 * 1024 * 1024)
    with pytest.raises(PackageError, match = "expands"):
        read_package(_zip({"flow.json": _flow(), "README.md": huge}, method = zipfile.ZIP_DEFLATED))


def test_oversized_and_missing_flow_are_refused(monkeypatch):
    with pytest.raises(PackageError, match = "no flow.json"):
        read_package(_zip({"README.md": b"hi"}))
    monkeypatch.setattr(package.schema, "MAX_PACKAGE_BYTES", 100)
    with pytest.raises(PackageError, match = "larger"):
        read_package(b"x" * 200)


def test_invalid_document_and_unknown_types():
    with pytest.raises(PackageError):
        read_package(b'{"format": "unsloth-flow", "id": "x", "name": "n", "evil": 1}')
    parsed, _, report = read_package(json.dumps(doc([node("x", "custom.node")])).encode())
    assert report["unknown_types"] == ["custom.node"]


def test_secrets_are_scrubbed_on_import():
    raw = doc(
        [
            node(
                "a",
                "input.text",
                value = "key sk-abcdefghijklmnopqrstuvwx here",
                api_key = "secret",
                max_tokens = 5,
            )
        ],
        meta = {"hf_token": "hf_abcdefghijklmnopqrstuvwxyz"},
    )
    parsed, _, _ = read_package(json.dumps(raw).encode())
    params = parsed["nodes"][0]["params"]
    assert "api_key" not in params and params["max_tokens"] == 5
    assert "sk-" not in params["value"]
    assert "hf_token" not in parsed["meta"]


@run_async
async def test_export_modes_embed_media_and_strip_local_ids():
    gallery = {"kind": "image", "ref": "image:abc", "url": "/api/inference/images/gallery/abc/file"}
    d = doc(
        [node("i", "input.image", asset = gallery, api_key = "x")],
        baked = {"g": {"image": gallery}},
        editor = {"viewport": [0, 0, 1]},
    )

    async def read_media(media):
        assert media["ref"] == "image:abc"
        return PNG, "image/png"

    data, name, mime = await package.export_flow(d, "json", read_media)
    exported = json.loads(data)
    assert (
        name.endswith(".json")
        and exported["baked"] == {}
        and exported["nodes"][0]["params"].get("asset") is None
    )
    assert "api_key" not in exported["nodes"][0]["params"]

    data, name, mime = await package.export_flow(d, "baked", read_media)
    assert name.endswith(".unslothflow")
    parsed, files, report = read_package(data)
    assert set(files) == {PNG_SHA}
    assert parsed["nodes"][0]["params"]["asset"]["ref"] == f"sha256:{PNG_SHA}"
    assert parsed["baked"]["g"]["image"]["ref"] == f"sha256:{PNG_SHA}"
    assert "image:abc" not in data.decode("latin-1")

    data, _, _ = await package.export_flow(d, "package", read_media)
    parsed, _, _ = read_package(data)
    assert parsed["baked"] == {}


def test_import_records_lineage_and_export_sets_license():
    original = doc([node("a", "input.text", value = "hi")])
    original["meta"] = {"author": "ana", "derived_from": [{"id": "root", "name": "Root"}]}
    package.record_lineage(original)
    assert original["meta"]["derived_from"][0] == {
        "id": original["id"],
        "name": original["name"],
        "author": "ana",
    }
    assert original["meta"]["derived_from"][1]["id"] == "root"

    data, _name, _mime = asyncio.run(package.export_flow(original, "json", None))
    meta = json.loads(data)["meta"]
    assert meta["license"] == package.DEFAULT_LICENSE
    assert meta["ai_generated"] is False
