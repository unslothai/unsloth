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
from utils import chat_history_policy

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
# Bumped by clear_cache. A fetch that started before the clear must not publish its
# thumbnail after it: the write is done under _registry_lock and skipped if this moved.
_cache_generation = 0
# The generation at which a CLEAR-EVERYTHING last ran, and the generation at which each
# individually reaped id was taken. A selective clear must not abort an in-flight fetch for
# an id it deliberately spared: thumbnail_bytes would answer None, the endpoint 404s, and
# SearchImageThumb renders nothing and never retries -- its effect depends only on (id,
# nearViewport), so "re-fetches on the next request" is not true, there is no next request.
# Bounded; on overflow the per-id record is dropped and the full-clear generation is moved
# instead, which over-aborts rather than republishing a thumbnail a clear removed.
_full_clear_generation = 0
_reaped_at: dict[str, int] = {}
_REAPED_AT_MAX = 4096
# The newest generation whose per-id records have been dropped to stay under that cap. A fetch
# that started at or after this is still answered exactly, because nothing covering it was
# dropped; only one older than every record we still hold has to be given up on. Fetches are
# bounded by THUMBNAIL_FETCH_TIMEOUT_S, so outliving 4096 reaped images is not a real case.
_reaped_floor_generation = 0
# Ids whose files a clear could not unlink -- on Windows another process holding the
# JPEG open is enough. The cache-first read and the sidecar read both go around the
# registry, so without this they would go on serving a picture the user had cleared.
_cleared_unservable: set[str] = set()
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
    expected_generation: int | None = None,
) -> list[dict[str, str]]:
    # Public entries only; the URLs stay in this process.
    from .web_access_policy import check_url_access

    memory_only = chat_history_policy.disabled()
    public: list[dict[str, str]] = []
    persist: list[tuple[str, dict[str, Any], int]] = []
    now = time.monotonic()
    with _registry_lock:
        generation = _cache_generation
        if expected_generation is not None and expected_generation != generation:
            return []
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
                # Kept with the entry: the proxy fetch happens on a later request, and
                # without it every redirect hop would be re-checked against no policy.
                "policy": website_policy,
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
            if not memory_only:
                persist.append((image_id, _registry[image_id], generation))
    for image_id, stored, registered_generation in persist:
        _persist_entry(image_id, stored, registered_generation)
    if persist:
        # Here too, not only after a thumbnail write: a user who never opens a picture
        # would otherwise accumulate sidecars with nothing ever bounding them.
        _evict_cache()
    return public


def cache_generation() -> int:
    with _registry_lock:
        return _cache_generation


def _lookup_locked(image_id: str) -> dict[str, Any] | None:
    """Registry read with the TTL applied. The caller holds ``_registry_lock``."""
    entry = _registry.get(image_id)
    if entry is None:
        return None
    if time.monotonic() - entry["created"] > _REGISTRY_TTL_S:
        _registry.pop(image_id, None)
        return None
    return dict(entry)


def lookup_image(image_id: str) -> dict[str, Any] | None:
    if not IMAGE_ID_RE.fullmatch(image_id or ""):
        return None
    with _registry_lock:
        return _lookup_locked(image_id)


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
        and bool(IMAGE_ID_RE.fullmatch(entry["id"]))
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


def _meta_path(image_id: str) -> Path:
    return _cache_dir() / f"{image_id}.json"


def _persist_entry(image_id: str, entry: dict[str, Any], generation: int) -> None:
    """Keep an id resolvable across a restart.

    The envelope in saved chat history carries ids, not URLs, and the browser only
    asks for a thumbnail once it nears the viewport -- so a picture that was never
    scrolled to has no bytes on disk, and the in-memory registry does not survive the
    process. Reopening that chat used to 404 forever. This is the same information the
    cached JPEG already reveals, and `clear_cache` removes both together.
    """
    if chat_history_policy.disabled():
        return
    try:
        payload = json.dumps(
            {
                "thumbnail": entry["thumbnail"],
                "source": entry["source"],
                "policy": entry.get("policy"),
            },
            ensure_ascii = True,
        )
        with _registry_lock:
            # Per id, like the thumbnail write: a selective clear bumps the generation
            # without touching this image, and dropping its sidecar then costs the id its
            # only way back after a restart.
            if _reaped_since_locked(image_id, generation):
                return
            # writer-unique, like the JPEG: a torn read must not be possible.
            tmp = _meta_path(image_id).with_suffix(f".{secrets.token_hex(4)}.tmp")
            tmp.write_text(payload, encoding = "utf-8")
            tmp.replace(_meta_path(image_id))
    except (OSError, TypeError, ValueError) as exc:
        # Best effort: losing this costs a 404 on an unseen picture, never the search.
        logger.debug("search image metadata write failed: %s", exc)


def _load_persisted_entry(image_id: str) -> dict[str, Any] | None:
    if chat_history_policy.disabled():
        return None
    try:
        raw = json.loads(_meta_path(image_id).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    thumbnail = raw.get("thumbnail")
    source = raw.get("source")
    if not isinstance(thumbnail, str) or not isinstance(source, str):
        return None
    # Re-checked on the way back in: what was public when it was written is not
    # necessarily public now, and this bypasses register_images' own gate.
    if not (_names_public_host(thumbnail) and _names_public_host(source)):
        return None
    policy = raw.get("policy")
    return {
        "thumbnail": thumbnail,
        "source": source,
        # No TTL: a disk entry follows the cache beside it, which is capped by file
        # count rather than age. time.monotonic() from a previous process is meaningless.
        "created": time.monotonic(),
        "policy": policy if isinstance(policy, dict) else None,
    }


def _evict_cache() -> None:
    # Capped per kind. Metadata is written for every registered image but bytes only
    # for the ones actually viewed, so the sidecars outnumber the JPEGs and need their
    # own bound; and an evicted JPEG keeps its sidecar, which is what lets it be
    # fetched again rather than 404.
    for pattern in ("*.jpg", "*.json"):
        try:
            files = sorted(_cache_dir().glob(pattern), key = lambda p: p.stat().st_mtime)
        except OSError:
            continue
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


def _fetch_thumbnail_bytes(url: str, website_policy: dict | None = None) -> bytes | None:
    from . import tools

    error, body, _content_type = tools._fetch_url_raw(
        url,
        timeout = THUMBNAIL_FETCH_TIMEOUT_S,
        extra_headers = {"Accept": "image/*"},
        deadline = time.monotonic() + THUMBNAIL_FETCH_TIMEOUT_S * 2,
        raw_bytes_max = MAX_THUMBNAIL_BYTES,
        website_policy = website_policy,
    )
    if error is not None or not isinstance(body, (bytes, bytearray)) or not body:
        if error is not None:
            logger.debug("search thumbnail fetch refused: %s", error)
        return None
    return _encode_thumbnail(bytes(body))


def _drop_if_cleared(image_id: str) -> bool:
    """False while a clear's leftover files for this id are still on disk.

    The unlink is retried on the way through: the process that had the JPEG open
    has usually let go by the next request, and once both files are gone the id is
    ordinary again -- nothing resolves it, so it 404s like any other unknown one.
    """
    with _registry_lock:
        if image_id not in _cleared_unservable:
            return True
        for path in (_cache_path(image_id), _meta_path(image_id)):
            try:
                path.unlink(missing_ok = True)
            except OSError:
                return False
        _cleared_unservable.discard(image_id)
        return True


def thumbnail_bytes(image_id: str) -> bytes | None:
    if not IMAGE_ID_RE.fullmatch(image_id or ""):
        return None
    memory_only = chat_history_policy.disabled()
    path = None
    if not memory_only:
        # Cache first: it survives the restart the in-memory registry does not.
        # Ahead of that read and of the sidecar below, which both go around the registry.
        if not _drop_if_cleared(image_id):
            return None
        path = _cache_path(image_id)
        try:
            if path.is_file():
                return path.read_bytes()
        except OSError:
            pass
    # Generation and entry in ONE acquisition, generation first. Taking them
    # separately let a clear land in the gap: this call would then read the POST-clear
    # generation, the check before the write would match, and the thumbnail the clear
    # had just deleted would be written back. Reading it first is the safe order --
    # a clear after this point leaves us holding a stale value, which fails the check.
    with _registry_lock:
        generation = _cache_generation
        entry = _lookup_locked(image_id)
    if entry is None:
        if memory_only:
            return None
        # Not in memory: the process may have restarted since the search. The metadata
        # on disk outlives it, the same way the cached bytes do.
        entry = _load_persisted_entry(image_id)
        if entry is None:
            return None

    with _inflight_lock:
        gate = _inflight.setdefault(image_id, threading.Lock())
    try:
        with gate:
            if path is not None:
                try:
                    if path.is_file():
                        return path.read_bytes()
                except OSError:
                    pass
            with _fetch_slots:
                data = _fetch_thumbnail_bytes(entry["thumbnail"], entry.get("policy"))
            if data is None:
                return None
            with _registry_lock:
                if _reaped_since_locked(image_id, generation):
                    # A clear that covered THIS id landed while the fetch was in flight.
                    # The chat that asked for it is gone, so publish nothing and hand back
                    # nothing -- writing here would restore a thumbnail the clear had
                    # removed, and the cache-first path above would keep serving it.
                    #
                    # Asked per id, not off the bare generation: a selective clear bumps
                    # that too, and aborting on it took down fetches for the images the
                    # clear had gone out of its way to spare.
                    return None
                if path is not None:
                    try:
                        # Writer-unique: racing writers must not publish a torn JPEG.
                        tmp = path.with_suffix(f".{secrets.token_hex(4)}.tmp")
                        tmp.write_bytes(data)
                        tmp.replace(path)
                    except OSError as exc:
                        logger.debug("search thumbnail cache write failed: %s", exc)
                        try:
                            tmp.unlink(missing_ok = True)
                        except OSError:
                            pass
            # Outside the lock: a glob plus a stat per file, and nothing here needs it.
            if not memory_only:
                _evict_cache()
            return data
    finally:
        with _inflight_lock:
            # Only drop the gate this call owns.
            if _inflight.get(image_id) is gate and not gate.locked():
                _inflight.pop(image_id, None)


def registered_image_ids() -> set[str] | None:
    """Every id a clear starting now would be responsible for, in memory and on disk.

    Taken before the caller's slow work so the reap that follows can be limited to it.
    A clear is global while the chat delete it accompanies is not, so anything that
    registers after this snapshot belongs to a chat the delete is keeping.
    """
    ids: set[str] = set()
    with _registry_lock:
        ids.update(_registry)
    for pattern in ("*.jpg", "*.json"):
        try:
            ids.update(path.stem for path in _cache_dir().glob(pattern))
        except OSError:
            # Unreadable dir, so the snapshot is incomplete and cannot bound a reap.
            # None is the sentinel clear_cache reads as "clear everything", which is
            # what a clear did before this snapshot existed. An empty set is NOT that
            # sentinel: it is a selective reap of nothing, and would leave the
            # registry populated and its images still fetchable after a clear.
            return None
    return ids


def snapshot_and_fence_registrations() -> set[str] | None:
    """``registered_image_ids``, plus a fence closing the window it opens.

    Bounding the reap to a snapshot means anything registered after it is spared. The
    reasoning was that such an image belongs to a chat created since, which the clear is
    keeping -- but a lookup already running when the clear started belongs to an answer the
    clear is DELETING, and it publishes into that same window. `/search-images/lookup` is
    the plain case: it carries no thread, so no cancellation reaches it, and it samples the
    cache generation on entry, before the reap moves it. Its images then survived Clear all
    with their sidecars, which say what was searched for.

    Bumping the generation HERE, at the clear boundary rather than at the reap seconds
    later, refuses exactly those: ``register_images`` compares against the generation its
    caller sampled, so work that started before this moment publishes nothing. Work that
    starts after samples the new value, registers normally, and is spared as intended.

    The bump is inside the same lock as the registry read, so a registration cannot slip
    between being missed by the snapshot and being caught by the fence. It does not abort
    in-flight FETCHES: those are keyed per id now, and a bare generation move is not one of
    the signals they read.
    """
    global _cache_generation
    ids: set[str] = set()
    with _registry_lock:
        ids.update(_registry)
        _cache_generation += 1
    for pattern in ("*.jpg", "*.json"):
        try:
            ids.update(path.stem for path in _cache_dir().glob(pattern))
        except OSError:
            # Same fallback as registered_image_ids: an incomplete snapshot cannot bound a
            # reap, and None is the sentinel clear_cache reads as "clear everything".
            return None
    return ids


def _reaped_since_locked(image_id: str, generation: int) -> bool:
    """Whether a clear covering ``image_id`` landed after ``generation``. Caller holds the lock."""
    if _full_clear_generation > generation:
        return True
    if generation < _reaped_floor_generation:
        # Older than every record still held, so this cannot be shown to have been spared.
        return True
    return _reaped_at.get(image_id, 0) > generation


def clear_cache(only_ids: set[str] | None = None) -> None:
    """Drop registered images and their cached bytes. Called when the user clears all
    chats: the thumbnails say what was searched for.

    ``only_ids`` limits the reap to a snapshot taken before the caller's slow work, so
    an image registered meanwhile -- by another tab or the LAN listener, for a chat the
    clear is not deleting -- keeps its bytes instead of 404ing out of ``thumbnail_bytes``.
    The generation still bumps either way -- ``register_images`` compares against it to
    refuse a registration racing a clear -- but the in-flight fetch check is per id, so a
    spared image's fetch is left alone. Aborting it would 404 a card that never retries.
    """
    global _cache_generation, _full_clear_generation, _reaped_floor_generation
    # The unlinks are under the lock too, so an in-flight fetch cannot slip its write
    # in between the bump and the delete and leave a cleared thumbnail on disk.
    with _registry_lock:
        if only_ids is None:
            _registry.clear()
        else:
            for image_id in only_ids:
                _registry.pop(image_id, None)
        _cache_generation += 1
        if only_ids is None:
            # Nothing survives, so every in-flight fetch has to abort. One number says so
            # for all of them, including ids this process has never seen.
            _full_clear_generation = _cache_generation
            _reaped_at.clear()
        else:
            if len(_reaped_at) + len(only_ids) > _REAPED_AT_MAX:
                # Out of room. Drop the OLDEST records rather than promoting this to a
                # full clear: doing that aborts every fetch in flight, including ones for
                # images this clear deliberately spared, and an aborted fetch is not a
                # cheap retry -- the card 404s and useSearchThumbnail never asks again.
                # Raising the floor instead gives up only on fetches older than every
                # record still held, which the fetch timeout makes unreachable in practice.
                keep_from = sorted(_reaped_at.values())[len(_reaped_at) // 2 :]
                floor = keep_from[0] - 1 if keep_from else _cache_generation
                for stale_id in [key for key, at in _reaped_at.items() if at <= floor]:
                    _reaped_at.pop(stale_id, None)
                _reaped_floor_generation = max(_reaped_floor_generation, floor)
            for image_id in only_ids:
                _reaped_at[image_id] = _cache_generation
        for pattern in ("*.jpg", "*.json", "*.tmp"):
            try:
                paths = list(_cache_dir().glob(pattern))
            except OSError:
                continue
            for path in paths:
                # `.tmp` stems carry a writer suffix, so they never match a snapshot id
                # and are always swept: one was never servable, and a torn write left
                # behind by a crashed fetch has no owner to spare it for.
                if only_ids is not None and pattern != "*.tmp" and path.stem not in only_ids:
                    continue
                # Per file, as _evict_cache does: one that cannot be unlinked --
                # a JPEG another process holds open on Windows -- must not leave
                # every later one on disk, where the cache-first read in
                # thumbnail_bytes would go on serving it after a clear.
                try:
                    path.unlink(missing_ok = True)
                except OSError:
                    # Still on disk, so remember the id and refuse to serve it until
                    # the unlink does land. `.jpg`/`.json` share a stem; a `.tmp` was
                    # never servable, and its stem carries the writer suffix.
                    if pattern != "*.tmp":
                        _cleared_unservable.add(path.stem)
