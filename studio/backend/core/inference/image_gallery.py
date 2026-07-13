# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Disk-backed persistence for generated images.

Each image is a PNG under ``studio_root()/images`` with its full recipe embedded as PNG text
chunks: a structured ``unsloth`` JSON blob (the source of truth) plus an Automatic1111-style
``parameters`` string for interop. So a downloaded PNG carries its own settings.

Dumb storage: the route owns the metadata schema and passes a plain dict; this only reads/writes/
sorts files.
"""

from __future__ import annotations

import base64
import json
import os
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger
from utils.paths import ensure_dir, studio_root

logger = get_logger(__name__)

# PNG text-chunk key holding our structured recipe JSON.
_META_KEY = "unsloth"
# Image ids are file stems; restrict to safe chars so a crafted id can't escape the directory.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def gallery_dir() -> Path:
    return ensure_dir(studio_root() / "images")


def _params_text(meta: dict[str, Any]) -> str:
    """Automatic1111-style ``parameters`` string for cross-tool interop."""
    lines = [str(meta.get("prompt", ""))]
    negative = meta.get("negative_prompt")
    if negative:
        lines.append(f"Negative prompt: {negative}")
    lines.append(
        f"Steps: {meta.get('steps')}, CFG scale: {meta.get('guidance')}, "
        f"Seed: {meta.get('seed')}, Size: {meta.get('width')}x{meta.get('height')}, "
        f"Model: {meta.get('model', '')}"
    )
    return "\n".join(lines)


def _png_bytes(image: Any, meta: dict[str, Any]) -> bytes:
    import io

    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text(_META_KEY, json.dumps(meta))
    info.add_text("parameters", _params_text(meta))
    buf = io.BytesIO()
    image.save(buf, format = "PNG", pnginfo = info)
    return buf.getvalue()


def save(image: Any, meta: dict[str, Any]) -> dict[str, Any]:
    """Persist a PIL image with its recipe embedded; return the gallery record."""
    image_id = uuid.uuid4().hex
    directory = gallery_dir()
    final_path = directory / f"{image_id}.png"
    # Write to a dotted temp (skipped by the *.png glob) then atomically rename, so a crash mid-write
    # never leaves a truncated {id}.png that the listing would surface as a corrupt record.
    tmp_path = directory / f".{image_id}.png.tmp"
    try:
        tmp_path.write_bytes(_png_bytes(image, meta))
        os.replace(tmp_path, final_path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok = True)
        except OSError:
            pass
        raise
    return _record(image_id, meta)


def _record(image_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "id": image_id,
        "url": f"/api/inference/images/gallery/{image_id}/file",
    }


def image_path(image_id: str) -> Optional[Path]:
    """Resolve an id to its on-disk PNG, or None if missing / unsafe."""
    if not _ID_RE.match(image_id):
        return None
    path = gallery_dir() / f"{image_id}.png"
    # Defence in depth: confirm the resolved path is still inside the gallery.
    try:
        path.resolve().relative_to(gallery_dir().resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def image_b64(image_id: str) -> Optional[str]:
    path = image_path(image_id)
    if path is None:
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


# Required recipe keys (GalleryImage fields minus id/url). A PNG missing any is skipped as
# foreign, so a hand-dropped or older-schema file can't 500 the listing.
_REQUIRED_META = ("prompt", "width", "height", "steps", "guidance", "seed", "created_at")


def _read_meta(path: Path) -> Optional[dict[str, Any]]:
    from PIL import Image

    try:
        with Image.open(path) as im:
            raw = im.text.get(_META_KEY)  # type: ignore[attr-defined]
    except Exception:
        return None
    if not raw:
        return None
    try:
        meta = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(meta, dict) or any(k not in meta for k in _REQUIRED_META):
        return None
    return meta


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def list_images(
    limit: Optional[int] = None,
    offset: int = 0,
    *,
    valid: Optional[Callable[[dict[str, Any]], bool]] = None,
) -> list[dict[str, Any]]:
    """A newest-first window of images for infinite scroll.

    Ordered by file mtime (a cheap stat ~= generation order), so a large gallery isn't opened in
    full just to sort; only the window's recipes are read. limit=None returns everything from
    ``offset`` on.

    ``valid`` (optional) filters records BEFORE pagination, so ``offset`` / ``limit`` and has_more
    all count over the accepted-record domain. Pass the route's schema validator: a record with
    every required key (so ``_read_meta`` accepts it) but a wrong value type would otherwise be
    counted here yet dropped after slicing, stalling infinite scroll at offset 0."""
    try:
        paths = list(gallery_dir().glob("*.png"))
    except OSError:
        return []
    paths.sort(key = _mtime, reverse = True)
    # Page over READABLE records, not raw files: filtering a foreign PNG out of an already-sliced
    # window would drop valid images and make has_more wrong. Read only as far as needed.
    # Known limit: this re-reads headers from newest down to `offset+limit` per page, so a deep
    # scroll is O(offset) header-opens. PIL opens are lazy (header only) and off the event loop, so
    # no freeze; a later phase can switch to cursor-based paging if it bites.
    want = None if limit is None else offset + limit
    records = []
    for path in paths:
        meta = _read_meta(path)
        if meta is None:  # not one of ours (no recipe chunk)
            continue
        record = _record(path.stem, meta)
        if valid is not None and not valid(record):  # present but schema-invalid
            continue
        records.append(record)
        if want is not None and len(records) >= want:
            break
    return records[offset:] if limit is None else records[offset : offset + limit]


def delete(image_id: str) -> bool:
    path = image_path(image_id)
    if path is None:
        return False
    try:
        path.unlink()
        return True
    except OSError as exc:
        logger.warning("image_gallery.delete_failed: %s", exc)
        return False


def clear() -> int:
    """Delete every gallery PNG; return how many were removed."""
    removed = 0
    try:
        paths = list(gallery_dir().glob("*.png"))
    except OSError:
        return 0
    for path in paths:
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed
