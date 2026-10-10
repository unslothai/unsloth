# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Disk-backed persistence for generated images.

Each image is a PNG under ``workspace_root()/images`` with its recipe embedded as PNG text chunks:
an ``unsloth`` JSON blob (the source of truth) plus an Automatic1111-style ``parameters`` string,
so a downloaded PNG carries its own settings. The route owns the schema; this only stores files.
"""

from __future__ import annotations

import base64
import errno
import json
import os
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from core.inference import gallery_flags
from loggers import get_logger
from utils.account_context import is_owner_context
from utils.paths import ensure_account_dir, studio_root
from utils.paths.relocations import location_dir
from utils.paths.storage_roots import account_path

logger = get_logger(__name__)

_META_KEY = "unsloth"
RECIPE_SCHEMA_VERSION = 1
# Restrict ids to safe chars so a crafted id cannot escape the directory.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def gallery_dir() -> Path:
    if is_owner_context():
        return location_dir("images", studio_root() / "images")
    return ensure_account_dir(account_path("images"))


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


# zlib level 6 costs ~0.4 s per 1024x1024 PNG; level 1 ~0.1 s, ~10% larger.
PNG_COMPRESS_LEVEL_ENV = "UNSLOTH_IMAGE_PNG_COMPRESS_LEVEL"
_DEFAULT_PNG_COMPRESS_LEVEL = 1


def png_compress_level() -> int:
    raw = os.environ.get(PNG_COMPRESS_LEVEL_ENV, "").strip()
    try:
        level = int(raw)
    except ValueError:
        return _DEFAULT_PNG_COMPRESS_LEVEL
    return level if 0 <= level <= 9 else _DEFAULT_PNG_COMPRESS_LEVEL


def _png_bytes(image: Any, meta: dict[str, Any]) -> bytes:
    import io

    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text(_META_KEY, json.dumps(meta))
    info.add_text("parameters", _params_text(meta))
    buf = io.BytesIO()
    image.save(buf, format = "PNG", pnginfo = info, compress_level = png_compress_level())
    return buf.getvalue()


def save(image: Any, meta: dict[str, Any]) -> dict[str, Any]:
    """Persist a PIL image with its recipe embedded; return the gallery record."""
    image_id = uuid.uuid4().hex
    meta = {**meta, "schema_version": RECIPE_SCHEMA_VERSION}
    # Encode before resolving the folder, so a concurrent Settings move cannot be overtaken.
    data = _png_bytes(image, meta)
    directory = gallery_dir()
    final_path = directory / f"{image_id}.png"
    # Dotted temp (skipped by the *.png glob) then atomic rename: no truncated PNG in the listing.
    tmp_path = directory / f".{image_id}.png.tmp"
    try:
        tmp_path.write_bytes(data)
        os.replace(tmp_path, final_path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok = True)
        except OSError:
            pass
        raise
    return _record(image_id, meta)


def _record(
    image_id: str,
    meta: dict[str, Any],
    flags: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    if flags is None:
        flags = gallery_flags.read(gallery_dir())
    return {
        **meta,
        "id": image_id,
        "url": f"/api/inference/images/gallery/{image_id}/file",
        **gallery_flags.flags_for(flags, image_id),
        "order_at": gallery_flags.order_rank(
            flags, image_id, _mtime(gallery_dir() / f"{image_id}.png")
        ),
    }


def image_path(image_id: str) -> Optional[Path]:
    """Resolve an id to its on-disk PNG, or None if missing / unsafe."""
    if not _ID_RE.match(image_id):
        return None
    path = gallery_dir() / f"{image_id}.png"
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


# A PNG missing any required key is skipped as foreign so it cannot 500 the listing.
_REQUIRED_META = ("prompt", "width", "height", "steps", "guidance", "seed", "created_at")


def _read_meta(path: Path, *, strict_io: bool = False) -> Optional[dict[str, Any]]:
    from PIL import Image

    try:
        with Image.open(path) as im:
            # The chunk is written before IDAT; im.text would decode every pixel to find later chunks.
            raw = im.info.get(_META_KEY) or im.text.get(_META_KEY)  # type: ignore[attr-defined]
    except OSError as exc:
        if strict_io and exc.errno not in (None, errno.ENOENT):
            raise
        return None
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


def owned_image_path(image_id: str) -> Optional[Path]:
    """Resolve an id to its PNG only when it is an Unsloth-owned image (a readable recipe chunk),
    else None. The serve route uses this instead of image_path() so a guessed stem for a
    hand-dropped foreign PNG -- which list_images/delete/clear already treat as not ours -- can't
    be streamed out. Mirrors the delete/clear ownership guard."""
    path = image_path(image_id)
    if path is None or _read_meta(path) is None:
        return None
    return path


def thumbnail(path: Path, size: int) -> bytes:
    """Return a WebP thumbnail with its longest side at most ``size`` pixels."""
    import io

    from PIL import Image

    with Image.open(path) as im:
        im.thumbnail((size, size), Image.LANCZOS)
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGBA" if "A" in im.getbands() or "transparency" in im.info else "RGB")
        buf = io.BytesIO()
        im.save(buf, format = "WEBP", quality = 85, method = 4)
    return buf.getvalue()


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
    archived: bool = False,
) -> list[dict[str, Any]]:
    """A window of images for infinite scroll: pinned first (most recently pinned leading), then
    newest-first by file mtime (or the manual key once dragged).

    mtime is a cheap stat ~= generation order, so a large gallery isn't opened in full just to
    sort; only the window's recipes are read. limit=None returns everything from ``offset`` on.

    ``archived`` selects WHICH shelf to page over, it does not widen one: False lists only active
    images, True lists only archived ones. The archived section needs its own scrollable page, so
    a chat-style "include archived" flag would not do.

    ``valid`` (optional) filters records BEFORE pagination, so ``offset`` / ``limit`` and has_more
    all count over the accepted-record domain. Pass the route's schema validator: a record with
    every required key (so ``_read_meta`` accepts it) but a wrong value type would otherwise be
    counted here yet dropped after slicing, stalling infinite scroll at offset 0."""
    try:
        paths = list(gallery_dir().glob("*.png"))
    except OSError:
        return []
    flags = gallery_flags.read(gallery_dir())
    paths = [p for p in paths if gallery_flags.is_archived(flags, p.stem) == archived]
    paths.sort(
        key = lambda p: (
            gallery_flags.pin_rank(flags, p.stem),
            gallery_flags.order_rank(flags, p.stem, _mtime(p)),
        ),
        reverse = True,
    )
    # Page over readable records, not raw files, or has_more is wrong. Deep scroll is O(offset).
    want = None if limit is None else offset + limit
    records = []
    for path in paths:
        meta = _read_meta(path)
        if meta is None:
            continue
        record = _record(path.stem, meta, flags)
        if valid is not None and not valid(record):
            continue
        records.append(record)
        if want is not None and len(records) >= want:
            break
    return records[offset:] if limit is None else records[offset : offset + limit]


def set_flags(
    image_id: str,
    *,
    pinned: Optional[bool] = None,
    archived: Optional[bool] = None,
) -> Optional[dict[str, Any]]:
    """Patch one image's pin/archive flags and return its updated record, or None when the id is
    not an Unsloth-owned image. Ownership-gated like delete: a guessed stem for a hand-dropped
    foreign PNG must not become flaggable (and so listable under a shelf we own)."""
    # Check and write under one lock so a concurrent clear cannot delete the file between them.
    with gallery_flags.exclusive(gallery_dir()):
        path = owned_image_path(image_id)
        if path is None:
            return None
        gallery_flags.set_flags_locked(gallery_dir(), image_id, pinned = pinned, archived = archived)
        meta = _read_meta(path)
    if meta is None:
        return None
    return _record(image_id, meta)


def move(image_id: str, after_id: Optional[str]) -> Optional[dict[str, Any]]:
    """Move an active image to just after ``after_id`` (None = front) and return its record.

    None if the id is not an owned, active image. Raises KeyError if ``after_id`` is not on the shelf."""
    with gallery_flags.exclusive(gallery_dir()):
        path = owned_image_path(image_id)
        if path is None:
            return None
        flags = gallery_flags.read(gallery_dir())
        if gallery_flags.is_archived(flags, image_id):
            return None
        try:
            paths = [
                p
                for p in gallery_dir().glob("*.png")
                if not gallery_flags.is_archived(flags, p.stem)
            ]
        except OSError:
            paths = []
        keyed = [(p.stem, _mtime(p)) for p in paths]
        keyed.sort(
            key = lambda pair: (
                gallery_flags.pin_rank(flags, pair[0]),
                gallery_flags.order_rank(flags, pair[0], pair[1]),
            ),
            reverse = True,
        )
        gallery_flags.place_locked(gallery_dir(), image_id, keyed, after_id = after_id)
        meta = _read_meta(path)
    if meta is None:
        return None
    return _record(image_id, meta)


def delete(image_id: str) -> bool:
    path = image_path(image_id)
    # A foreign PNG is invisible to list_images, so a guessed id must not destroy it.
    if path is None or _read_meta(path, strict_io = True) is None:
        if _ID_RE.fullmatch(image_id):
            try:
                (gallery_dir() / f"{image_id}.png").lstat()
            except FileNotFoundError:
                gallery_flags.forget(gallery_dir(), [image_id])
        return False
    removed = True
    try:
        path.unlink()
    except FileNotFoundError:
        removed = False
    except OSError as exc:
        logger.warning("image_gallery.delete_failed: %s", exc)
        raise
    gallery_flags.forget(gallery_dir(), [image_id])
    return removed


def clear(include_archived: bool = False) -> int:
    """Delete Unsloth-owned gallery PNGs (readable recipe chunk); return how many were removed.

    Archived images are SPARED by default: archiving is how a user sets something aside, so a
    "clear the gallery" action that destroyed the archive would defeat it. Pass
    include_archived=True to remove those too.

    Raises FlagsUnavailable when the archive has to be spared but the flag store cannot be read.
    Fail CLOSED: read() answers "nothing is archived" for an unreadable store, which here would
    quietly delete the very archive this promises to keep.

    Foreign PNGs are preserved: list_images already hides them, so clear must not destroy them."""
    removed = 0
    directory = gallery_dir()
    # Hold the flag lock across read-then-delete, or an archive landing mid-loop gets deleted.
    with gallery_flags.exclusive(directory):
        # Read flags BEFORE listing: nothing is unlinked if the store is untrusted.
        flags = {} if include_archived else gallery_flags.read_trusted(directory)
        try:
            paths = list(directory.glob("*.png"))
        except OSError:
            return 0
        cleared: list[str] = []
        for path in paths:
            if _read_meta(path) is None:
                continue
            if not include_archived and gallery_flags.is_archived(flags, path.stem):
                continue
            try:
                path.unlink()
                removed += 1
                cleared.append(path.stem)
            except OSError:
                continue
        # Every owned image is gone, so replace the untrusted store or later clears keep refusing.
        if include_archived and not gallery_flags.is_trusted(directory):
            gallery_flags.reset_locked(directory)
        else:
            gallery_flags.forget_locked(directory, cleared)
    return removed
