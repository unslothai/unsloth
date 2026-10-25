# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import io
import json
import struct
import zipfile

import pytest

from core.flows import assets as asset_store
from core.flows import glb, package

TRI = {
    "asset": {"version": "2.0"},
    "buffers": [{"byteLength": 36}],
    "bufferViews": [{"buffer": 0, "byteLength": 36}],
    "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
}


def make_glb(doc: dict, bin_: bytes = b"\0" * 36) -> bytes:
    text = json.dumps(doc).encode()
    text += b" " * (-len(text) % 4)
    chunks = struct.pack("<II", len(text), 0x4E4F534A) + text
    if bin_:
        chunks += struct.pack("<II", len(bin_), 0x004E4942) + bin_
    return struct.pack("<III", 0x46546C67, 2, 12 + len(chunks)) + chunks


def test_a_self_contained_model_passes():
    data = make_glb(TRI)
    assert glb.inspect_glb(data) == {"bytes": len(data), "triangles": 1}
    assert asset_store.sniff_mime(data) == "model/gltf-binary"
    assert asset_store.kind_of("model/gltf-binary") == "model"


@pytest.mark.parametrize(
    "data, reason",
    [
        (
            make_glb({**TRI, "buffers": [{"byteLength": 36, "uri": "https://x.test/a.bin"}]}),
            "outside files",
        ),
        (make_glb({**TRI, "images": [{"uri": "data:image/png;base64,AAAA"}]}), "outside files"),
        (make_glb({**TRI, "accessors": [{"count": 3_000_000}]}), "1,000,000 triangles"),
        (make_glb({**TRI, "extensionsUsed": ["EXT_meshopt_compression"]}), "meshopt"),
        (
            make_glb(
                {**TRI, "accessors": [{"count": 300_000}], "nodes": [{"mesh": 0}, {"mesh": 0}]}
            ),
            "200,000 triangles",
        ),
        (make_glb({**TRI, "extensionsUsed": ["EXT_mesh_gpu_instancing"]}), "instancing"),
        (b"glTF" + struct.pack("<II", 1, 20) + b"\0" * 8, "version 1"),
        (make_glb(TRI)[:-4], "cut short"),
        (b"glTF" + b"\0" * (glb.MAX_GLB_BYTES + 1), "at most 20 MB"),
    ],
    # Short ids: an id built from a 20 MB byte string overflows the environment on Windows (PYTEST_CURRENT_TEST).
    ids = lambda v: v if isinstance(v, str) else "glb",
)
def test_bad_models_are_refused_with_a_reason(data, reason):
    with pytest.raises(glb.GlbError, match = reason):
        glb.inspect_glb(data)


def test_the_asset_store_refuses_a_bad_model(tmp_path, monkeypatch):
    monkeypatch.setattr(asset_store, "assets_dir", lambda: tmp_path)
    digest, mime = asset_store.store(make_glb(TRI))
    assert mime == "model/gltf-binary" and (tmp_path / f"{digest}.glb").is_file()
    assert asset_store.read(digest)[1] == "model/gltf-binary"
    with pytest.raises(ValueError, match = "outside files"):
        asset_store.store(make_glb({**TRI, "buffers": [{"uri": "x.bin"}]}))
    assert len(list(tmp_path.glob("*.glb"))) == 1


def _package(blob: bytes) -> bytes:
    import hashlib

    digest = hashlib.sha256(blob).hexdigest()
    doc = {
        "id": "f",
        "name": "w",
        "kind": "world",
        "nodes": [
            {
                "id": "play",
                "type": "world.play",
                "pos": [0, 0],
                "params": {
                    "props3d": [
                        {
                            "id": "m-1",
                            "kind": "model",
                            "x": 1,
                            "y": 1,
                            "asset": {"kind": "model", "ref": f"sha256:{digest}"},
                        }
                    ]
                },
            }
        ],
        "edges": [],
    }
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("flow.json", json.dumps(doc))
        zf.writestr(f"assets/{digest}.glb", blob)
    return out.getvalue()


def test_packages_carry_models_and_refuse_bad_ones():
    doc, files, report = package.read_package(_package(make_glb(TRI)))
    assert report["assets"] == 1 and not report["warnings"]
    assert next(iter(files.values()))[1] == "model/gltf-binary"
    with pytest.raises(package.PackageError, match = "outside files"):
        package.read_package(_package(make_glb({**TRI, "buffers": [{"uri": "x.bin"}]})))
