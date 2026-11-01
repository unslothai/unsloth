# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image PNGs in workspace_root()/images; recipe in an unsloth JSON text chunk (source of truth)"""

from __future__ import annotations

import base64
import errno
import functools
import json
import os
import re
import sys
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Optional

from core.inference import gallery_flags, png_encode
from loggers import get_logger
from utils.account_context import is_owner_context
from utils.paths import ensure_account_dir, studio_root
from utils.paths.relocations import location_dir
from utils.paths.storage_roots import account_path

logger = get_logger(__name__)

# PNG text-chunk key holding our structured recipe JSON.
_META_KEY = "unsloth"
# Absent on PNGs written before it existed.
RECIPE_SCHEMA_VERSION = 1
# Image ids are file stems; restrict to safe chars so a crafted id can't escape the directory.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def gallery_dir() -> Path:
    if is_owner_context():
        return location_dir("images", studio_root() / "images")
    return ensure_account_dir(account_path("images"))


_IMG2IMG_WORKFLOWS = ("img2img", "inpaint", "outpaint", "edit", "upscale")


def _params_text(meta: dict[str, Any]) -> str:
    """Automatic1111-style ``parameters`` string for cross-tool interop."""
    from core.inference.diffusion_hires import infotext_fields as hires_infotext, infotext_meta

    # A Hires fix image is written the way A1111 writes one: Size is the base pass, Hires upscale the factor.
    meta = infotext_meta(meta)
    # The caption the model encoded (FLUX.3 prompt upsampler): other tools reproduce the image from this line. The
    # typed prompt stays in the Unsloth chunk.
    lines = [str(meta.get("enhanced_prompt") or meta.get("prompt", ""))]
    negative = meta.get("negative_prompt")
    if negative:
        lines.append(f"Negative prompt: {negative}")
    from core.inference.diffusion_sampler import sampler_label, schedule_label

    if meta.get("model_kind") == "connection":
        lines.append(
            f"Size: {meta.get('width')}x{meta.get('height')}, Model: {meta.get('model', '')}"
        )
        return "\n".join(lines)
    sampler = sampler_label(meta.get("sampler"))
    schedule = schedule_label(meta.get("schedule_type"))
    fields = (
        f"Steps: {meta.get('steps')}, "
        + (f"Sampler: {sampler}, " if sampler else "")
        + (f"Schedule type: {schedule}, " if schedule else "")
        + f"CFG scale: {meta.get('guidance')}, "
        f"Seed: {meta.get('seed')}, Size: {meta.get('width')}x{meta.get('height')}, "
        f"Model: {meta.get('model', '')}"
    )
    # webui infotext names (processing.create_infotext).
    if meta.get("variation_strength"):
        fields += (
            f", Variation seed: {meta.get('variation_seed')}, "
            f"Variation seed strength: {meta['variation_strength']}"
        )
    if meta.get("clip_skip"):
        fields += f", Clip skip: {meta['clip_skip']}"
    if meta.get("cfg_rescale"):
        fields += f", CFG rescale: {meta['cfg_rescale']}"
    # webui opts.emphasis infotext: a prompt run with weighting off replays literally there too.
    if meta.get("prompt_weighting") is False:
        fields += ", Emphasis: None"
    if meta.get("workflow") in _IMG2IMG_WORKFLOWS and meta.get("strength") is not None:
        fields += f", Denoising strength: {meta['strength']}"
    # A1111 inpaint keys (modules/processing.py), only when this inpaint applied them.
    if meta.get("mask_blur"):
        fields += f", Mask blur: {meta['mask_blur']}"
    if meta.get("inpaint_only_masked"):
        fields += ", Inpaint area: Only masked"
        if meta.get("inpaint_padding") is not None:
            fields += f", Masked area padding: {meta['inpaint_padding']}"
    fields += hires_infotext(meta)
    # Static flow shift, read back by the paste parser; base / max shift stay in the recipe chunk only.
    if meta.get("shift") is not None:
        fields += f", Shift: {meta['shift']}"
    # ADetailer's own keys (adetailer/args.py), so a paste into Studio or a webui with ADetailer restores the pass.
    if meta.get("detailer"):
        from core.inference.diffusion_detailer import infotext_fields
        fields += infotext_fields(meta["detailer"])
    if meta.get("generator"):
        fields += f", Version: {meta['generator']}"
    lines.append(fields)
    return "\n".join(lines)


# Pillow's default zlib level 6 costs ~0.4 s per 1024x1024 PNG on the request path; level 1 ~0.1 s, ~10% larger, lossless.
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
    data = png_encode.encode(image, pnginfo = info, compress_level = png_compress_level())
    if data is not None:
        return data
    buf = io.BytesIO()
    image.save(buf, format = "PNG", pnginfo = info, compress_level = png_compress_level())
    return buf.getvalue()


@functools.lru_cache(maxsize = 1)
def _generator() -> str:
    try:
        from utils.studio_version import get_studio_version
        return f"Unsloth {get_studio_version()}"
    except Exception:
        return "Unsloth"


def _engine() -> Optional[str]:
    # Read only when a generation already loaded the router: diffusers and the native engine draw
    # different pixels from the same seed, so a replay needs to know which one ran.
    router = sys.modules.get("core.inference.diffusion_engine_router")
    try:
        return router.active_engine_name() if router is not None else None
    except Exception:
        return None


def _recipe(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        # A Connection run never touched the local engine.
        "engine": None if meta.get("model_kind") == "connection" else _engine(),
        "generator": _generator(),
        **meta,
        "schema_version": RECIPE_SCHEMA_VERSION,
    }


def save(image: Any, meta: dict[str, Any]) -> dict[str, Any]:
    """Persist a PIL image with its recipe embedded; return the gallery record."""
    image_id = uuid.uuid4().hex
    meta = _recipe(meta)
    # Encoded before the folder is looked up: a Settings move during the encode would otherwise
    # finish first, and the image land in the folder it left.
    data = _encoded_ahead(image, meta)
    if data is None:
        data = _png_bytes(image, meta)
    return _publish(image_id, meta, data)


# Pillow releases the GIL while it filters and deflates, so a batch's PNGs encode side by side.
_MAX_ENCODE_WORKERS = 8
_ahead = threading.local()


def _encoded_ahead(image: Any, meta: dict[str, Any]) -> Optional[bytes]:
    """The PNG ``save_many`` already encoded from this image and this exact recipe, else None."""
    pending = getattr(_ahead, "pending", None)
    entry = pending.pop(id(image), None) if pending else None
    if entry is None or entry[0] is not image or list(entry[1].items()) != list(meta.items()):
        return None
    return entry[2].result()


def save_many(items: Iterable[tuple[Any, dict[str, Any]]]) -> Iterator[dict[str, Any]]:
    """``save`` each (image, meta) in order, with every PNG encoded concurrently up front.

    Each ``save`` still writes its file and builds its record in turn, so the bytes, the gallery
    order and what a failure leaves behind are those of a plain loop of ``save``."""
    items = list(items)
    cpus = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    workers = min(len(items), cpus, _MAX_ENCODE_WORKERS)
    # One image object twice would be loaded by two threads at once.
    if workers < 2 or len({id(image) for image, _ in items}) < len(items):
        for image, meta in items:
            yield save(image, meta)
        return
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers = workers, thread_name_prefix = "gallery-png") as pool:
        pending = {}
        for image, meta in items:
            recipe = _recipe(meta)
            pending[id(image)] = (image, recipe, pool.submit(_png_bytes, image, recipe))
        futures = [entry[2] for entry in pending.values()]
        _ahead.pending = pending
        # The gallery sorts by mtime and back-to-back writes can tie (Windows ~15.6 ms tick): keep each strictly newer.
        _ahead.floor_ns = None
        try:
            for image, meta in items:
                yield save(image, meta)
        finally:
            if getattr(_ahead, "pending", None) is pending:
                _ahead.pending = None
                _ahead.floor_ns = None
            for future in futures:
                future.cancel()


def _publish(image_id: str, meta: dict[str, Any], data: bytes) -> dict[str, Any]:
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
    if getattr(_ahead, "pending", None) is not None:
        _keep_after_previous(final_path)
    return _record(image_id, meta, directory = directory)


def _keep_after_previous(path: Path) -> None:
    """In a ``save_many`` batch, move ``path``'s mtime 1 us past the previous image's when it is not already later."""
    try:
        stat = path.stat()
        mtime_ns = stat.st_mtime_ns
        floor = getattr(_ahead, "floor_ns", None)
        if floor is not None and mtime_ns <= floor:
            mtime_ns = floor + 1000
            os.utime(path, ns = (stat.st_atime_ns, mtime_ns))
        _ahead.floor_ns = mtime_ns
    except OSError:
        pass


def _record(
    image_id: str,
    meta: dict[str, Any],
    flags: Optional[dict[str, dict[str, Any]]] = None,
    *,
    directory: Optional[Path] = None,
) -> dict[str, Any]:
    if directory is None:
        directory = gallery_dir()
    # flags are library state, not recipe: they come from the sidecar store, never the PNG chunk
    if flags is None:
        flags = gallery_flags.read(directory)
    return {
        **meta,
        "id": image_id,
        "url": f"/api/inference/images/gallery/{image_id}/file",
        **gallery_flags.flags_for(flags, image_id),
        # The server's own sort key, so a client re-sort agrees with it (created_at can differ).
        "order_at": gallery_flags.order_rank(
            flags, image_id, _mtime(directory / f"{image_id}.png")
        ),
    }


def image_path(image_id: str) -> Optional[Path]:
    """Resolve an id to its on-disk PNG, or None if missing / unsafe."""
    if not _ID_RE.match(image_id):
        return None
    directory = gallery_dir()
    path = directory / f"{image_id}.png"
    try:
        path.resolve().relative_to(directory.resolve())
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
    """Returns None unless the PNG is Unsloth-owned, so a guessed id cannot stream a hand-dropped file."""
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
    """Pinned first, then mtime; archived selects a shelf, never widens; valid filter runs before paging."""
    try:
        directory = gallery_dir()
        paths = list(directory.glob("*.png"))
    except OSError:
        return []
    flags = gallery_flags.read(directory)
    # Shelf split and pin sort use file stems before any recipe read, keeping the early break intact.
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
        record = _record(path.stem, meta, flags, directory = directory)
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
    if meta is None:  # raced a delete between the guard and the read
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
        # Full shelf in listing order, so neighbours past the client's loaded window are known.
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
    if meta is None:  # raced a delete between the guard and the read
        return None
    return _record(image_id, meta)


def delete(image_id: str) -> bool:
    path = image_path(image_id)
    # a hand-dropped foreign PNG is invisible to list_images, so a guessed id must not destroy it
    if path is None or _read_meta(path, strict_io = True) is None:
        if _ID_RE.fullmatch(image_id):
            # Only prune absent files, preserving foreign files and symlinks.
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
        # Propagate I/O failures instead of reporting a missing image.
        raise
    # drop the flags with the file, so the id cannot hand out a stale pin and the store cannot grow forever
    gallery_flags.forget(gallery_dir(), [image_id])
    return removed


def clear(include_archived: bool = False) -> int:
    """Owned PNGs only; archived spared unless include_archived; fails closed if flags unreadable."""
    removed = 0
    directory = gallery_dir()
    # Hold the flag lock across read-then-delete, or an archive landing mid-loop gets deleted.
    with gallery_flags.exclusive(directory):
        # read flags BEFORE listing: nothing is unlinked if the store turns out to be untrusted
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
