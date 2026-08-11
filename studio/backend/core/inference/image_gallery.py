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

from core.inference import gallery_flags
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
    # Write to a dotted temp (skipped by the *.png glob) then atomically rename, so a crash mid-write never leaves a truncated {id}.png in the listing.
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


def _record(
    image_id: str,
    meta: dict[str, Any],
    flags: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    # Flags are library state, not recipe: they come from the sidecar store, never the PNG chunk.
    return {
        **meta,
        "id": image_id,
        "url": f"/api/inference/images/gallery/{image_id}/file",
        **gallery_flags.flags_for(
            flags if flags is not None else gallery_flags.read(gallery_dir()), image_id
        ),
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


# Required recipe keys (GalleryImage fields minus id/url). A PNG missing any is skipped as foreign, so a hand-dropped or older-schema file cannot 500 the listing.
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


def owned_image_path(image_id: str) -> Optional[Path]:
    """Resolve an id to its PNG only when it is a Studio-owned image (a readable recipe chunk),
    else None. The serve route uses this instead of image_path() so a guessed stem for a
    hand-dropped foreign PNG -- which list_images/delete/clear already treat as not ours -- can't
    be streamed out. Mirrors the delete/clear ownership guard."""
    path = image_path(image_id)
    if path is None or _read_meta(path) is None:
        return None
    return path


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
    newest-first by file mtime.

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
    # Both the shelf split and the pin sort run on file stems, BEFORE any recipe is read, so they
    # cost one dict lookup per file and leave the early break below intact.
    paths = [p for p in paths if gallery_flags.is_archived(flags, p.stem) == archived]
    paths.sort(key = lambda p: (gallery_flags.pin_rank(flags, p.stem), _mtime(p)), reverse = True)
    # Page over READABLE records, not raw files: filtering a foreign PNG out of an already-sliced window would drop valid images and make has_more
    # wrong. Known limit: this re-reads headers from newest down to `offset+limit` per page, so a deep scroll is O(offset) header-opens.
    want = None if limit is None else offset + limit
    records = []
    for path in paths:
        meta = _read_meta(path)
        if meta is None:  # not one of ours (no recipe chunk)
            continue
        record = _record(path.stem, meta, flags)
        if valid is not None and not valid(record):  # present but schema-invalid
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
    not a Studio-owned image. Ownership-gated like delete: a guessed stem for a hand-dropped
    foreign PNG must not become flaggable (and so listable under a shelf we own)."""
    # Ownership check and write under one lock, so a concurrent clear cannot delete the file
    # between them and leave this reporting success for an image that is already gone.
    with gallery_flags.exclusive(gallery_dir()):
        path = owned_image_path(image_id)
        if path is None:
            return None
        gallery_flags.set_flags_locked(gallery_dir(), image_id, pinned = pinned, archived = archived)
        meta = _read_meta(path)
    if meta is None:  # raced a delete between the guard and the read
        return None
    return _record(image_id, meta)


def delete(image_id: str) -> bool:
    path = image_path(image_id)
    if path is None:
        return False
    # Only delete files we own (a readable recipe chunk): a hand-dropped foreign PNG is invisible to list_images, so a guessed id must not destroy it.
    if _read_meta(path) is None:
        return False
    try:
        path.unlink()
    except OSError as exc:
        logger.warning("image_gallery.delete_failed: %s", exc)
        return False
    # Drop the flags with the file, so the id cannot hand a stale pin to anything and the store
    # cannot grow forever.
    gallery_flags.forget(gallery_dir(), [image_id])
    return True


def clear(include_archived: bool = False) -> int:
    """Delete Studio-owned gallery PNGs (readable recipe chunk); return how many were removed.

    Archived images are SPARED by default: archiving is how a user sets something aside, so a
    "clear the gallery" action that destroyed the archive would defeat it. Pass
    include_archived=True to remove those too.

    Raises FlagsUnavailable when the archive has to be spared but the flag store cannot be read.
    Fail CLOSED: read() answers "nothing is archived" for an unreadable store, which here would
    quietly delete the very archive this promises to keep.

    Foreign PNGs are preserved: list_images already hides them, so clear must not destroy them."""
    removed = 0
    directory = gallery_dir()
    # Hold the flag lock across the whole read-then-delete: an archive landing mid-loop would
    # otherwise be judged active from the stale snapshot and deleted, after its PATCH had already
    # reported success.
    with gallery_flags.exclusive(directory):
        # Read flags BEFORE listing: nothing is unlinked if the store turns out to be untrusted.
        flags = {} if include_archived else gallery_flags.read_trusted(directory)
        try:
            paths = list(directory.glob("*.png"))
        except OSError:
            return 0
        cleared: list[str] = []
        for path in paths:
            if _read_meta(path) is None:  # foreign / not ours
                continue
            if not include_archived and gallery_flags.is_archived(flags, path.stem):
                continue
            try:
                path.unlink()
                removed += 1
                cleared.append(path.stem)
            except OSError:
                continue
        # An unreadable store has nothing left to protect once every image we own is gone, so this
        # is where the escape hatch actually escapes: replace it, or the corrupt file survives the
        # wipe and every later default clear still refuses, new media included.
        if include_archived and not gallery_flags.is_trusted(directory):
            gallery_flags.reset_locked(directory)
        else:
            gallery_flags.forget_locked(directory, cleared)
    return removed
