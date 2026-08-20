# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# web_search image results: a registry keyed by opaque ids plus a thumbnail proxy,
# so neither the model nor the browser sees an image URL.

from __future__ import annotations

import io
import json
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from loggers import get_logger

logger = get_logger(__name__)

SEARCH_IMAGES_SENTINEL = "\n__WEB_IMAGES__:"
IMAGE_ID_RE = re.compile(r"^[0-9a-f]{12}$")

MAX_IMAGES_PER_SEARCH = 6
MAX_THUMBNAIL_BYTES = 3 * 1024 * 1024
THUMBNAIL_EDGE_PX = 320
THUMBNAIL_FETCH_TIMEOUT_S = 10
# Read from the header before decoding. Low because draft() only subsamples JPEG:
# a 77 KB 6000x4000 PNG decodes to ~100 MB of RGB.
MAX_IMAGE_PIXELS = 6_000_000
_ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "GIF", "WEBP"})

_REGISTRY_TTL_S = 24 * 3600
_REGISTRY_MAX_ENTRIES = 2000
# The cache outlives the registry, so reopened chats keep their pictures.
_CACHE_MAX_FILES = 2000
_CACHE_DIRNAME = "search_thumbs"
_MAX_CONCURRENT_FETCHES = 4

_registry: dict[str, dict[str, Any]] = {}
_registry_lock = threading.Lock()
_fetch_slots = threading.BoundedSemaphore(_MAX_CONCURRENT_FETCHES)
_inflight: dict[str, threading.Lock] = {}
_inflight_lock = threading.Lock()


def search_images_enabled() -> bool:
    try:
        from storage.studio_db import list_chat_settings
        return list_chat_settings().get("searchImages") is True
    except Exception:  # noqa: BLE001 - a settings read must never break a tool call
        return False


def _clean_text(value: Any, limit: int = 200) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _domain_of(url: str) -> str:
    from urllib.parse import urlparse
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _names_public_host(url: str) -> bool:
    # Pre-filter only; the fetch re-resolves and pins the host regardless.
    import ipaddress
    from urllib.parse import urlparse

    try:
        host = (urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    if not host or host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        return False
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        return True


def _prune_registry_locked(now: float) -> None:
    expired = [key for key, entry in _registry.items() if now - entry["created"] > _REGISTRY_TTL_S]
    for key in expired:
        _registry.pop(key, None)
    if len(_registry) > _REGISTRY_MAX_ENTRIES:
        oldest = sorted(_registry.items(), key = lambda item: item[1]["created"])
        for key, _entry in oldest[: len(_registry) - _REGISTRY_MAX_ENTRIES]:
            _registry.pop(key, None)


def register_images(
    raw_results: list[dict[str, Any]],
    website_policy: dict | None = None,
    max_images: int = MAX_IMAGES_PER_SEARCH,
    subject: str | None = None,
) -> list[dict[str, str]]:
    # Public entries only; the URLs stay in this process.
    from .web_access_policy import check_url_access

    public: list[dict[str, str]] = []
    now = time.monotonic()
    with _registry_lock:
        _prune_registry_locked(now)
        for item in raw_results:
            if len(public) >= max_images:
                break
            if not isinstance(item, dict):
                continue
            thumbnail = str(item.get("thumbnail") or item.get("image") or "").strip()
            source = str(item.get("url") or "").strip()
            if not thumbnail or not source:
                continue
            ok_thumb, _r1, _h1 = check_url_access(thumbnail, website_policy)
            ok_source, _r2, _h2 = check_url_access(source, website_policy)
            if not (ok_thumb and ok_source):
                continue
            if not (_names_public_host(thumbnail) and _names_public_host(source)):
                continue
            image_id = secrets.token_hex(6)
            _registry[image_id] = {
                "thumbnail": thumbnail,
                "source": source,
                "created": now,
            }
            entry = {
                "id": image_id,
                "title": _clean_text(item.get("title")),
                "domain": _domain_of(source)[:100],
                "source": source[:2048],
            }
            if subject:
                entry["subject"] = _clean_text(subject, 80)
            public.append(entry)
    return public


def lookup_image(image_id: str) -> dict[str, Any] | None:
    if not IMAGE_ID_RE.fullmatch(image_id or ""):
        return None
    with _registry_lock:
        entry = _registry.get(image_id)
        if entry is None:
            return None
        if time.monotonic() - entry["created"] > _REGISTRY_TTL_S:
            _registry.pop(image_id, None)
            return None
        return dict(entry)


def format_images_for_model(entries: list[dict[str, str]]) -> str:
    if not entries:
        return ""
    lines = [
        "Images from this search. To show one, write its token exactly as given, e.g. "
        "[[img:" + entries[0]["id"] + "]], on its own line after the text it illustrates. "
        "Use only these tokens, and only where the image clearly matches. For a picture of "
        "a specific thing you name, call web_search with image_queries instead.",
    ]
    for entry in entries:
        label = entry["title"] or "(untitled)"
        domain = f" — {entry['domain']}" if entry["domain"] else ""
        lines.append(f"- [[img:{entry['id']}]] {label}{domain}")
    return "\n".join(lines)


def images_envelope(entries: list[dict[str, str]]) -> str:
    if not entries:
        return ""
    return SEARCH_IMAGES_SENTINEL + json.dumps(entries, ensure_ascii = True, separators = (",", ":"))


def is_image_entry(entry: object) -> bool:
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("id"), str)
        and bool(IMAGE_ID_RE.match(entry["id"]))
        and isinstance(entry.get("title"), str)
        and isinstance(entry.get("domain"), str)
        and isinstance(entry.get("source"), str)
        and (entry.get("subject") is None or isinstance(entry.get("subject"), str))
    )


def split_images_envelope(result: str) -> tuple[str, list[dict[str, str]]]:
    # Payload ends at the next "\n__", as _strip_files_sentinel and the frontend do,
    # so a sibling sentinel after ours does not make it unreadable.
    start = result.rfind(SEARCH_IMAGES_SENTINEL)
    if start == -1:
        return result, []
    payload_start = start + len(SEARCH_IMAGES_SENTINEL)
    end = result.find("\n__", payload_start)
    if end == -1:
        end = len(result)
    try:
        entries = json.loads(result[payload_start:end])
    except (ValueError, RecursionError):
        return result, []
    if not isinstance(entries, list) or not entries or not all(is_image_entry(e) for e in entries):
        return result, []
    return (result[:start] + result[end:]).rstrip(), entries


def strip_images_suffix(result: str) -> str:
    return split_images_envelope(result)[0]


def _cache_dir() -> Path:
    from utils.paths import ensure_dir, studio_root
    return ensure_dir(studio_root() / _CACHE_DIRNAME)


def _cache_path(image_id: str) -> Path:
    return _cache_dir() / f"{image_id}.jpg"


def _evict_cache() -> None:
    try:
        files = sorted(_cache_dir().glob("*.jpg"), key = lambda p: p.stat().st_mtime)
    except OSError:
        return
    for path in files[: max(0, len(files) - _CACHE_MAX_FILES)]:
        try:
            path.unlink()
        except OSError:
            pass
    try:
        for stale in _cache_dir().glob("*.tmp"):
            if time.time() - stale.stat().st_mtime > 300:
                stale.unlink(missing_ok = True)
    except OSError:
        pass


def _encode_thumbnail(raw: bytes) -> bytes | None:
    from PIL import Image

    # Pillow's own process-wide bomb check is left at its default.
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.format not in _ALLOWED_FORMATS:
                return None
            width, height = im.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                return None
            if getattr(im, "n_frames", 1) > 1:
                im.seek(0)
            im.draft("RGB", (THUMBNAIL_EDGE_PX * 2, THUMBNAIL_EDGE_PX * 2))
            converted = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
            if converted.mode == "RGBA":
                flat = Image.new("RGB", converted.size, (255, 255, 255))
                flat.paste(converted, mask = converted.getchannel("A"))
                converted = flat
            converted.thumbnail((THUMBNAIL_EDGE_PX, THUMBNAIL_EDGE_PX), Image.LANCZOS)
            out = io.BytesIO()
            converted.save(out, format = "JPEG", quality = 82, optimize = True)
            return out.getvalue()
    except Exception as exc:  # noqa: BLE001 - provider bytes; any decode failure is a miss
        logger.debug("search thumbnail decode failed (%s)", type(exc).__name__)
        return None


def _fetch_thumbnail_bytes(url: str) -> bytes | None:
    from . import tools

    error, body, _content_type = tools._fetch_url_raw(
        url,
        timeout = THUMBNAIL_FETCH_TIMEOUT_S,
        extra_headers = {"Accept": "image/*"},
        deadline = time.monotonic() + THUMBNAIL_FETCH_TIMEOUT_S * 2,
        raw_bytes_max = MAX_THUMBNAIL_BYTES,
    )
    if error is not None or not isinstance(body, (bytes, bytearray)) or not body:
        if error is not None:
            logger.debug("search thumbnail fetch refused: %s", error)
        return None
    return _encode_thumbnail(bytes(body))


def thumbnail_bytes(image_id: str) -> bytes | None:
    # Cache first: it survives the restart the in-memory registry does not.
    if not IMAGE_ID_RE.fullmatch(image_id or ""):
        return None
    path = _cache_path(image_id)
    try:
        if path.is_file():
            return path.read_bytes()
    except OSError:
        pass
    entry = lookup_image(image_id)
    if entry is None:
        return None

    with _inflight_lock:
        gate = _inflight.setdefault(image_id, threading.Lock())
    try:
        with gate:
            try:
                if path.is_file():
                    return path.read_bytes()
            except OSError:
                pass
            with _fetch_slots:
                data = _fetch_thumbnail_bytes(entry["thumbnail"])
            if data is None:
                return None
            try:
                # Writer-unique: racing writers must not publish a torn JPEG.
                tmp = path.with_suffix(f".{secrets.token_hex(4)}.tmp")
                tmp.write_bytes(data)
                tmp.replace(path)
                _evict_cache()
            except OSError as exc:
                logger.debug("search thumbnail cache write failed: %s", exc)
                try:
                    tmp.unlink(missing_ok = True)
                except OSError:
                    pass
            return data
    finally:
        with _inflight_lock:
            # Only drop the gate this call owns.
            if _inflight.get(image_id) is gate and not gate.locked():
                _inflight.pop(image_id, None)


def clear_cache() -> None:
    """Drop every registered image and its cached bytes. Called when the user
    clears all chats: the thumbnails say what was searched for."""
    with _registry_lock:
        _registry.clear()
    try:
        for pattern in ("*.jpg", "*.tmp"):
            for path in _cache_dir().glob(pattern):
                path.unlink(missing_ok = True)
    except OSError:
        pass
