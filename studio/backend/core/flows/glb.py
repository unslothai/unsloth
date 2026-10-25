# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checks for a 3D model (binary glTF, ``.glb``) a world places as a prop. Only the JSON chunk is read: a model must
carry every buffer and texture inside the file (no URI, so nothing is fetched from elsewhere when it is drawn), stay
under the size and triangle budgets, and need no decoder the 3D player does not ship. The browser runs the same
checks (``props3d.ts`` ``inspectGlb``) for a readable error before it uploads."""

from __future__ import annotations

import json
import struct
from typing import Optional

MAX_GLB_BYTES = 20 * 1024 * 1024
MAX_TRIANGLES = 150_000
GLB_MIME = "model/gltf-binary"
# Draco / meshopt geometry needs a decoder the player does not load; WebP / Basis textures are fine in browsers but
# KTX2 needs a transcoder.
UNSUPPORTED_EXTENSIONS = {
    "KHR_draco_mesh_compression": "Draco compression",
    "EXT_meshopt_compression": "meshopt compression",
    "KHR_texture_basisu": "KTX2 textures",
    "EXT_mesh_gpu_instancing": "GPU instancing",
}


class GlbError(ValueError):
    pass


def is_glb(data: bytes) -> bool:
    return data[:4] == b"glTF"


def _triangles(doc: dict) -> int:
    accessors = doc.get("accessors") or []

    def count(i: object) -> int:
        if isinstance(i, int) and 0 <= i < len(accessors) and isinstance(accessors[i], dict):
            c = accessors[i].get("count")
            return c if isinstance(c, int) and c > 0 else 0
        return 0

    per_mesh = []
    for mesh in doc.get("meshes") or []:
        total = 0
        prims = mesh.get("primitives") if isinstance(mesh, dict) else None
        for prim in prims if isinstance(prims, list) else []:
            if not isinstance(prim, dict):
                continue
            mode = prim.get("mode", 4)
            attrs = prim.get("attributes") if isinstance(prim.get("attributes"), dict) else {}
            n = count(prim["indices"]) if "indices" in prim else count(attrs.get("POSITION"))
            if mode == 4:
                total += n // 3
            elif mode in (5, 6):
                total += max(0, n - 2)
        per_mesh.append(total)
    # A mesh is drawn once per node that places it, so reusing one mesh many times is not free.
    placed = sum(
        per_mesh[n["mesh"]]
        for n in doc.get("nodes") or []
        if isinstance(n, dict)
        and isinstance(n.get("mesh"), int)
        and not isinstance(n.get("mesh"), bool)
        and 0 <= n["mesh"] < len(per_mesh)
    )
    return max(placed, sum(per_mesh))


def inspect_glb(data: bytes) -> dict:
    """``{"bytes", "triangles"}`` for a model a world may use; raises GlbError with a readable reason otherwise."""
    if len(data) > MAX_GLB_BYTES:
        raise GlbError(f"The model is {len(data) / 2**20:.1f} MB; models can be at most 20 MB")
    if len(data) < 20 or not is_glb(data):
        raise GlbError("Not a binary glTF (.glb) model")
    version, length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise GlbError(f"glTF version {version} is not supported; export glTF 2.0")
    if length != len(data):
        raise GlbError("The model file is cut short or padded")
    chunk_len, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != 0x4E4F534A or 20 + chunk_len > len(data):
        raise GlbError("The model has no glTF JSON chunk")
    try:
        doc = json.loads(data[20 : 20 + chunk_len])
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise GlbError("The model's glTF JSON is not valid") from None
    if not isinstance(doc, dict):
        raise GlbError("The model's glTF JSON is not valid")
    for key in ("buffers", "images"):
        for entry in doc.get(key) or []:
            if isinstance(entry, dict) and "uri" in entry:
                raise GlbError(
                    "The model points at outside files; export it as a single .glb with everything embedded"
                )
    used = set(doc.get("extensionsRequired") or []) | set(doc.get("extensionsUsed") or [])
    for ext, what in UNSUPPORTED_EXTENSIONS.items():
        if ext in used:
            raise GlbError(
                f"The model uses {what}, which the 3D player cannot read; export without it"
            )
    triangles = _triangles(doc)
    if triangles > MAX_TRIANGLES:
        raise GlbError(
            f"The model has {triangles:,} triangles; props can have at most {MAX_TRIANGLES:,} (decimate it in Blender)"
        )
    return {"bytes": len(data), "triangles": triangles}


def check(data: bytes) -> Optional[str]:
    """The reason a model is refused, or None when it is fine."""
    try:
        inspect_glb(data)
    except GlbError as exc:
        return str(exc)
    return None
