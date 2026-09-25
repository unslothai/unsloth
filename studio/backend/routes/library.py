# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Library API: one view over uploads, chat attachments, generated images and audio, fine-tuned
models and sandbox files, plus folders, favorites and renames. See ``core.library`` for the sources."""

import base64
import hashlib
import hmac
import os
import re
import secrets
import time
from typing import Literal, Optional
from urllib.parse import quote, urlencode

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from auth.authentication import (
    get_current_subject,
    request_admitted_without_credential,
    subject_for_header_or_query_token,
)
from core import library
from hub.services.models import account_access
from loggers import get_logger
from storage import library_db
from storage.studio_db import ChatMessageProtectedError
from utils.account_context import run_as
from utils.upload_limits import LIBRARY_UPLOAD_MAX_BYTES
from utils.utils import log_and_http_error, safe_curated_detail

logger = get_logger(__name__)

router = APIRouter()

_MAX_UPLOAD_BYTES = LIBRARY_UPLOAD_MAX_BYTES
_CHUNK_BYTES = 1024 * 1024
# Raster images, audio and video render inline; anything else (svg and html included) downloads as
# opaque bytes, as the sandbox route does, so a crafted upload cannot run script on the app origin.
# Exact types, never a prefix: a stored "audio/x, text/html" is not audio.
_INLINE_IMAGE_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/webp", "image/avif"}
)
_NOSNIFF = {"X-Content-Type-Options": "nosniff"}


def _inline_type(content_type: str) -> Optional[str]:
    value = library.media_type(content_type)
    if value in _INLINE_IMAGE_TYPES:
        return value
    playable = {
        known
        for known in library.content_types().values()
        if known.startswith(("audio/", "video/"))
    }
    return value if value in playable else None


class ItemPatch(BaseModel):
    id: str = Field(max_length = 4096)
    name: Optional[str] = Field(default = None, min_length = 1, max_length = 255)
    favorite: Optional[bool] = None
    # Present-and-null moves the item back to the Library root.
    folderId: Optional[str] = Field(default = None, max_length = 64)


class ItemRef(BaseModel):
    id: str = Field(max_length = 4096)


class LocationRef(BaseModel):
    key: str = Field(max_length = 32)


class ProjectCopy(BaseModel):
    id: str = Field(max_length = 4096)
    projectId: str = Field(max_length = 256)


class FolderCreate(BaseModel):
    name: str = Field(min_length = 1, max_length = 255)
    parentId: Optional[str] = Field(default = None, max_length = 64)


class FolderPatch(BaseModel):
    name: Optional[str] = Field(default = None, min_length = 1, max_length = 255)
    parentId: Optional[str] = Field(default = None, max_length = 64)


class TextContent(BaseModel):
    text: str = Field(max_length = 5_000_000)
    # The note's own encoding, so a UTF-16 file (Windows PowerShell, Notepad "Unicode") stays one.
    encoding: Literal["utf-8", "utf-16le", "utf-16be"] = "utf-8"


@router.get("")
async def get_library(current_subject: str = Depends(get_current_subject)) -> dict:
    items = await run_in_threadpool(library.list_items)
    return {
        "items": items,
        "folders": library_db.list_folders(),
        "disk": await run_in_threadpool(library.disk_usage),
    }


@router.get("/favorites")
def get_favorites(current_subject: str = Depends(get_current_subject)) -> dict:
    """Favorite item ids alone, for pages that mark favorites without listing every source. Only
    those the source still has: a gallery or chat can delete one without the Library."""
    return {
        "ids": [
            item_id
            for item_id, entry in library_db.list_entries().items()
            if entry["favorite"] and _still_there(item_id, entry["fingerprint"])
        ]
    }


def _still_there(item_id: str, fingerprint: Optional[str]) -> bool:
    try:
        return library.item_exists(item_id, fingerprint)
    except Exception:
        # A store that cannot be read right now keeps its stars rather than dropping them.
        logger.debug("library.favorite_check_failed: %s", item_id, exc_info = True)
        return True


# ── Items ────────────────────────────────────────────────────────


@router.patch("/items")
def patch_item(body: ItemPatch, current_subject: str = Depends(get_current_subject)) -> dict:
    try:
        library_db.update_entry(
            body.id,
            name = body.name.strip() if body.name else None,
            favorite = body.favorite,
            folder_id = body.folderId,
            move = "folderId" in body.model_fields_set,
            # A path can name another file later; the row is kept for this one.
            fingerprint = library.fingerprint(body.id),
        )
    except KeyError:
        raise HTTPException(status_code = 404, detail = "Folder not found")
    return {"ok": True}


@router.post("/items/opened")
def mark_item_opened(body: ItemRef, current_subject: str = Depends(get_current_subject)) -> dict:
    library_db.mark_opened(body.id)
    return {"ok": True}


@router.post("/items/delete")
async def delete_item(body: ItemRef, current_subject: str = Depends(get_current_subject)) -> dict:
    try:
        deleted = await run_in_threadpool(library.delete_item, body.id)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    except library.DeleteIncomplete as exc:
        raise HTTPException(status_code = 500, detail = str(exc))
    except ChatMessageProtectedError as exc:
        raise log_and_http_error(
            exc,
            409,
            safe_curated_detail(exc),
            event = "library.delete_protected",
            log = logger,
        ) from exc
    except OSError as exc:
        raise _file_error(exc, "library.delete_failed", "Could not delete the file.") from exc
    if not deleted:
        raise HTTPException(status_code = 404, detail = "Item not found")
    return {"ok": True}


def _file_error(exc: OSError, event: str, detail: str) -> HTTPException:
    """Windows refuses to replace or delete a file another program holds open, so that one says
    what to do; anything else is logged and answered without the path."""
    if isinstance(exc, PermissionError):
        logger.info("%s: %s", event, exc)
        return HTTPException(status_code = 409, detail = "The file is in use. Close it and try again.")
    logger.warning("%s: %s", event, exc, exc_info = True)
    return HTTPException(status_code = 500, detail = detail)


@router.post("/items/project")
async def add_item_to_project(
    body: ProjectCopy, current_subject: str = Depends(get_current_subject)
) -> dict:
    """Copy an item's file into a chat project's folder. The Library keeps its item."""
    from core.inference.gallery_projects import ProjectNotFound, copy_into_project

    def _copy() -> dict:
        with library.open_item(body.id) as item:
            copied = copy_into_project(item.handle, body.projectId, item.folder, item.project_name)
        # The copy is a project file of its own, listed from the next listing on.
        library.invalidate_listing()
        return copied

    try:
        result = await run_in_threadpool(_copy)
    except ProjectNotFound:
        raise HTTPException(status_code = 404, detail = "Project not found")
    except LookupError:
        raise HTTPException(status_code = 404, detail = "Item not found")
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    except OSError as exc:
        logger.warning("library.add_to_project_failed: %s", exc)
        raise HTTPException(status_code = 500, detail = "Could not copy the file into the project.")
    return {"already": result["already"]}


def _reveal(path) -> None:
    from utils.paths.file_manager import file_manager_kind
    from utils.paths.path_utils import reveal_in_file_manager

    # The UI hides Reveal on such a host; a direct call must not open a window nobody sees.
    if file_manager_kind() is None:
        raise HTTPException(status_code = 503, detail = "No file manager is available on this machine")
    try:
        reveal_in_file_manager(path)
    except FileNotFoundError:
        # Two things raise this, as the sandbox reveal knows: the file going before it is shown,
        # and Popen not finding a file manager at all (a headless Linux has no xdg-open).
        if os.path.exists(path):
            logger.error("library.reveal_failed: %s", path, exc_info = True)
            raise HTTPException(
                status_code = 503, detail = "No file manager is available on this machine"
            )
        raise HTTPException(status_code = 404, detail = "File not found")
    except Exception:
        logger.error("library.reveal_failed: %s", path, exc_info = True)
        raise HTTPException(status_code = 500, detail = "Failed to open file manager")


@router.post("/items/reveal")
async def reveal_item(body: ItemRef, current_subject: str = Depends(get_current_subject)) -> dict:
    """Show the item's file in the OS file manager. The backend host's, so the desktop app's."""
    account_access.require_installation_owner()
    try:
        path = await run_in_threadpool(library.local_path, body.id)
    except LookupError:
        raise HTTPException(status_code = 404, detail = "Item not found")
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    await run_in_threadpool(_reveal, path)
    return {"ok": True}


@router.get("/items/thumbnail")
def get_item_thumbnail(
    id: str = Query(max_length = 4096), current_subject: str = Depends(get_current_subject)
):
    """An image or video item's card picture. The client adds the item's version to the URL, so it
    can cache."""
    try:
        data = library.thumbnail(id)
    except LookupError:
        raise HTTPException(status_code = 404, detail = "Item not found")
    except (RuntimeError, OSError) as exc:
        logger.info("library.thumbnail_unavailable: %s", exc)
        raise HTTPException(status_code = 501, detail = "No thumbnail for this item")
    return Response(
        content = data,
        media_type = "image/webp",
        headers = {"Cache-Control": "private, max-age=31536000, immutable", **_NOSNIFF},
    )


@router.api_route("/items/download", methods = ["GET", "HEAD"])
async def download_item(
    request: Request,
    id: str = Query(max_length = 4096),
    token: Optional[str] = Query(default = None, max_length = 8192),
):
    """An item's file as an attachment. Takes ``?token=`` as well: the desktop app streams it
    straight to disk, and its native save sends no header."""
    await subject_for_header_or_query_token(request, token)
    try:
        # Opened once and streamed from that descriptor: a sandbox name can be a link by the time
        # it would be opened again.
        item = await run_in_threadpool(library.open_item, id)
    except LookupError:
        raise HTTPException(status_code = 404, detail = "Item not found")
    except ValueError:
        raise HTTPException(status_code = 400, detail = "This item has no file to download")
    headers = {
        **_attachment_headers(item.name),
        "Content-Length": str(item.size),
        "Cache-Control": "private, no-store",
    }
    if request.method.upper() == "HEAD":
        item.close()
        return Response(status_code = 200, media_type = "application/octet-stream", headers = headers)
    return StreamingResponse(
        _read_exactly(item), media_type = "application/octet-stream", headers = headers
    )


# ── Streamed previews ────────────────────────────────────────────

# An audio or video preview plays from a link the element fetches itself, with range requests, so
# a long file is never buffered whole and can seek. The bearer mints the link; the link alone then
# serves that one item, for that one account, until it expires. A secret and domain of its own, so
# a gallery's media link never works here, nor one of these there.
_STREAM_LINK_TTL = 6 * 3600
_STREAM_LINK_SECRET = secrets.token_bytes(32)
_STREAM_LINK_DOMAIN = b"unsloth-library-stream\0"


def _stream_signature(payload: str) -> str:
    return hmac.new(
        _STREAM_LINK_SECRET, _STREAM_LINK_DOMAIN + payload.encode(), hashlib.sha256
    ).hexdigest()


def _sign_stream_id(item_id: str) -> str:
    """A link token for ``item_id`` in the calling account."""
    target = base64.urlsafe_b64encode(account_access.media_link_target(item_id).encode())
    payload = f"{target.decode().rstrip('=')}.{int(time.time()) + _STREAM_LINK_TTL}"
    return f"{payload}.{_stream_signature(payload)}"


def _stream_link_account(token: str, item_id: str):
    """The account a valid, unexpired token for ``item_id`` was minted in, else None."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    target, expires, signature = parts
    if not hmac.compare_digest(signature, _stream_signature(f"{target}.{expires}")):
        return None
    if not expires.isdigit() or int(expires) < time.time():
        return None
    try:
        signed = base64.urlsafe_b64decode(target + "=" * (-len(target) % 4)).decode()
    except ValueError:
        return None
    return account_access.media_link_account(signed, item_id)


def _stream_type(item_id: str) -> Optional[str]:
    """The exact audio or video type an item plays as, from the Library's fixed map; None for
    anything else, which this route never serves. Read in the item's own account; LookupError for
    an upload it does not have."""
    kind, _, ref = item_id.partition(":")
    if kind == "audio":
        value = "audio/wav"
    elif kind == "video":
        value = "video/mp4"
    elif kind == "upload":
        record = library_db.get_upload(ref)
        if record is None:
            raise LookupError(item_id)
        value = library.media_type(record["contentType"])
    elif kind == "sandbox":
        value = library.content_types().get(os.path.splitext(ref)[1].lower())
    else:
        return None
    playable = {
        known
        for known in library.content_types().values()
        if known.startswith(("audio/", "video/"))
    }
    return value if value in playable else None


class _Unsatisfiable(Exception):
    pass


def _byte_range(request: Request, size: int) -> Optional[tuple[int, int]]:
    """The one ``bytes=`` range asked for, as (first, last) inclusive; None for the whole file.

    What RFC 9110 lets a server ignore is ignored (another unit, several ranges, a malformed one,
    an ``If-Range`` this route has no validator to match); a range past the end is unsatisfiable."""
    header = request.headers.get("range")
    if not header or request.headers.get("if-range"):
        return None
    unit, _, spec = header.partition("=")
    if unit.strip().lower() != "bytes" or "," in spec:
        return None
    first, dash, last = spec.strip().partition("-")
    if (
        not dash
        or not (first.isdigit() or last.isdigit())
        or not all(part.isdigit() for part in (first, last) if part)
    ):
        return None
    if not first:
        suffix = int(last)
        if suffix == 0 or size == 0:
            raise _Unsatisfiable
        return max(0, size - suffix), size - 1
    start = int(first)
    end = int(last) if last else size - 1
    if last and end < start:
        return None
    if start >= size:
        raise _Unsatisfiable
    return start, min(end, size - 1)


def _read_range(item, start: int, length: int):
    """``length`` bytes from ``start`` of the open file, never past them."""
    with item:
        item.handle.seek(start)
        remaining = length
        while remaining > 0:
            chunk = item.handle.read(min(_CHUNK_BYTES, remaining))
            if not chunk:
                return
            remaining -= len(chunk)
            yield chunk


@router.get("/items/stream-url")
async def get_item_stream_url(
    id: str = Query(max_length = 4096),
    current_subject: str = Depends(get_current_subject),
    no_credential: bool = Depends(request_admitted_without_credential),
) -> dict:
    """A short-lived link an ``<audio>`` or ``<video>`` element plays an item from, with range
    requests. Relative, so it works behind any proxy the page is served through."""
    if no_credential:
        raise HTTPException(
            status_code = 403,
            detail = "Media links can only be created from the Unsloth UI or with an API key.",
        )
    try:
        content_type = await run_in_threadpool(_stream_type, id)
    except LookupError:
        raise HTTPException(status_code = 404, detail = "Item not found")
    if content_type is None:
        raise HTTPException(status_code = 400, detail = "Only audio and video items can be streamed")
    if not await run_in_threadpool(library.item_exists, id):
        raise HTTPException(status_code = 404, detail = "Item not found")
    query = urlencode({"id": id, "token": _sign_stream_id(id)})
    return {"url": f"/api/library/items/stream?{query}"}


@router.api_route("/items/stream", methods = ["GET", "HEAD"])
async def stream_item(
    request: Request,
    id: str = Query(max_length = 4096),
    token: str = Query(max_length = 8192),
):
    """An audio or video item, gated by its link rather than the bearer so it can be a plain
    ``src``, and range-capable so it seeks. Opened once in the link's account and read from that
    descriptor, so a sandbox name swapped for a link after the check is never followed."""
    account = _stream_link_account(token, id)
    if account is None:
        raise HTTPException(status_code = 401, detail = "Invalid or expired media link.")

    def _open():
        content_type = _stream_type(id)
        if content_type is None:
            raise LookupError(id)
        return content_type, library.open_item(id)

    try:
        content_type, item = await run_in_threadpool(run_as, account, _open)
    except (LookupError, ValueError):
        raise HTTPException(status_code = 404, detail = "Item not found")
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "private", **_NOSNIFF}
    try:
        requested = _byte_range(request, item.size)
    except _Unsatisfiable:
        item.close()
        return Response(
            status_code = 416, headers = {"Content-Range": f"bytes */{item.size}", **headers}
        )
    start, end = requested or (0, item.size - 1)
    length = max(0, end - start + 1)
    headers["Content-Length"] = str(length)
    if requested is not None:
        headers["Content-Range"] = f"bytes {start}-{end}/{item.size}"
    status_code = 206 if requested is not None else 200
    if request.method.upper() == "HEAD":
        item.close()
        return Response(status_code = status_code, media_type = content_type, headers = headers)
    return StreamingResponse(
        _read_range(item, start, length),
        status_code = status_code,
        media_type = content_type,
        headers = headers,
        # A player that hangs up mid-range leaves the generator unfinished: the file still closes.
        background = BackgroundTask(item.close),
    )


def _attachment_headers(name: str) -> dict:
    # RFC 5987, as the sandbox route sends it: an ASCII fallback plus the UTF-8 name.
    ascii_name = name.encode("ascii", "replace").decode("ascii").replace('"', "_")
    return {
        "Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name)}",
        **_NOSNIFF,
    }


def _read_exactly(item):
    """The file's bytes up to the length sent in the header, as the sandbox route streams: a file
    still being appended to must not send a body longer than Content-Length."""
    remaining = item.size
    with item:
        while remaining > 0:
            chunk = item.handle.read(min(_CHUNK_BYTES, remaining))
            if not chunk:
                return
            remaining -= len(chunk)
            yield chunk


@router.get("/locations")
async def get_locations(current_subject: str = Depends(get_current_subject)) -> dict:
    return {"locations": await run_in_threadpool(library.locations)}


@router.post("/locations/reveal")
async def reveal_location(
    body: LocationRef, current_subject: str = Depends(get_current_subject)
) -> dict:
    account_access.require_installation_owner()
    from pathlib import Path

    entries = {entry["key"]: entry for entry in await run_in_threadpool(library.locations)}
    if body.key not in entries:
        raise HTTPException(status_code = 404, detail = "Unknown location")
    path = Path(entries[body.key]["path"])
    # A chosen folder that is gone is on a drive that is not there: making it would put the next
    # saves beneath the mount point. Only a default is made.
    if not entries[body.key]["available"]:
        raise HTTPException(
            status_code = 409,
            detail = f"{path} is not available. Reconnect its drive, or reset the folder.",
        )
    await run_in_threadpool(lambda: path.mkdir(parents = True, exist_ok = True))
    await run_in_threadpool(_reveal, path)
    return {"ok": True}


class LocationMove(BaseModel):
    key: str = Field(max_length = 32)
    # None moves the files back to the default folder.
    path: Optional[str] = Field(default = None, max_length = 4096)


@router.post("/locations/move")
async def move_location(
    body: LocationMove, current_subject: str = Depends(get_current_subject)
) -> dict:
    """Move one kind of file to another folder, files and all. Installation owner only, like the
    other storage settings: other accounts keep their files in their own workspace. `leftBehind`
    names the folder whose files stayed on an unplugged drive when a Reset let go of it."""
    account_access.require_installation_owner()
    try:
        left_behind = await run_in_threadpool(library.move_location, body.key, body.path)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    except RuntimeError as exc:
        logger.warning("library.move_location_failed: %s", exc)
        raise HTTPException(status_code = 500, detail = str(exc))
    return {
        "locations": await run_in_threadpool(library.locations),
        "leftBehind": left_behind,
    }


# ── Library-owned uploads ────────────────────────────────────────


def _chunks(stream, name: str):
    total = 0
    while chunk := stream.read(_CHUNK_BYTES):
        total += len(chunk)
        if total > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code = 413, detail = f"{name} is too large")
        yield chunk


def _base_name(name: Optional[str]) -> str:
    # Some clients send a full path, with either separator.
    return re.split(r"[\\/]", name or "")[-1][:255] or "Untitled"


@router.post("/uploads")
async def upload_files(
    files: Optional[list[UploadFile]] = File(None),
    # Desktop drops: signed grants for paths the OS handed the app, read here rather than
    # through the webview, which cannot read documents.
    nativePathLeases: Optional[list[str]] = Form(None),
    folderId: Optional[str] = Form(None),
    current_subject: str = Depends(get_current_subject),
) -> dict:
    if not files and not nativePathLeases:
        raise HTTPException(status_code = 400, detail = "No files were provided.")
    if folderId and library_db.get_folder(folderId) is None:
        raise HTTPException(status_code = 404, detail = "Folder not found")

    def _check_native(lease: str) -> None:
        try:
            name, size = library.check_native_upload(lease)
        except (ValueError, OSError) as exc:
            # A bad or expired grant, a path outside this account's workspace, or a file gone.
            # The reason can name the path, so it goes to the log alone.
            logger.info("library.native_drop_refused: %s", exc)
            raise HTTPException(
                status_code = 400, detail = "The dropped file could not be read. Drop it again."
            ) from exc
        if size > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code = 413, detail = f"{name} is too large")

    consumed: list[str] = []

    def _save_native(lease: str) -> dict:
        try:
            name, content_type, handle = library.open_native_upload(lease)
        except (ValueError, OSError) as exc:
            logger.info("library.native_drop_refused: %s", exc)
            raise HTTPException(
                status_code = 400, detail = "The dropped file could not be read. Drop it again."
            ) from exc
        consumed.append(lease)
        with handle:
            # Refused before a byte is copied; _chunks still caps a file that grows meanwhile.
            if os.fstat(handle.fileno()).st_size > _MAX_UPLOAD_BYTES:
                raise HTTPException(status_code = 413, detail = f"{name} is too large")
            return library.save_upload(name, content_type, _chunks(handle, name))

    # Every grant and its size first, and nothing spent: a batch refused here is retried whole.
    for lease in nativePathLeases or []:
        await run_in_threadpool(_check_native, lease)

    records = []
    try:
        for upload in files or []:
            name = _base_name(upload.filename)
            records.append(
                await run_in_threadpool(
                    library.save_upload,
                    name,
                    library.upload_content_type(name, upload.content_type),
                    _chunks(upload.file, name),
                )
            )
        for lease in nativePathLeases or []:
            records.append(await run_in_threadpool(_save_native, lease))
        ids = [f"upload:{record['id']}" for record in records]
        # Placement is part of the batch too: a folder deleted meanwhile undoes the upload.
        if folderId:
            for item_id in ids:
                library_db.update_entry(item_id, folder_id = folderId, move = True)
    except BaseException as exc:
        # All or nothing, so a retry never duplicates the files that did make it, and the grants
        # this batch spent are given back, or its retry would be refused as a replay.
        for record in records:
            await run_in_threadpool(library.delete_item, f"upload:{record['id']}")
        _release_leases(consumed)
        if isinstance(exc, KeyError):
            raise HTTPException(status_code = 404, detail = "Folder not found") from exc
        raise
    return {"ids": ids}


def _release_leases(leases: list[str]) -> None:
    from utils.native_path_leases import release_native_path_lease
    for lease in leases:
        try:
            release_native_path_lease(lease)
        except ValueError:
            logger.debug("library.native_lease_release_failed", exc_info = True)


@router.get("/uploads/{upload_id}/file")
def get_upload_file(upload_id: str, current_subject: str = Depends(get_current_subject)):
    record = library_db.get_upload(upload_id)
    path = library.upload_path(upload_id)
    if record is None or path is None or not path.is_file():
        raise HTTPException(status_code = 404, detail = "File not found")
    content_type = _inline_type(record["contentType"])
    if content_type:
        return FileResponse(path, media_type = content_type, headers = _NOSNIFF)
    return FileResponse(
        path,
        media_type = "application/octet-stream",
        filename = library.safe_file_name(record["name"]),
        headers = _NOSNIFF,
    )


@router.put("/uploads/{upload_id}/text")
def put_upload_text(
    upload_id: str,
    body: TextContent,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    try:
        written = library.write_upload_text(upload_id, body.text, body.encoding)
    except UnicodeEncodeError:
        raise HTTPException(status_code = 400, detail = "The note has characters it cannot hold.")
    except OSError as exc:
        raise _file_error(exc, "library.note_save_failed", "Could not save the note.") from exc
    if not written:
        raise HTTPException(status_code = 404, detail = "File not found")
    return {"ok": True}


# ── Folders ──────────────────────────────────────────────────────


@router.post("/folders")
def create_folder(body: FolderCreate, current_subject: str = Depends(get_current_subject)) -> dict:
    try:
        return library_db.create_folder(body.name.strip(), body.parentId)
    except KeyError:
        raise HTTPException(status_code = 404, detail = "Parent folder not found")


@router.patch("/folders/{folder_id}")
def patch_folder(
    folder_id: str,
    body: FolderPatch,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    try:
        folder = library_db.update_folder(
            folder_id,
            name = body.name.strip() if body.name else None,
            parent_id = body.parentId,
            move = "parentId" in body.model_fields_set,
        )
    except KeyError:
        raise HTTPException(status_code = 404, detail = "Parent folder not found")
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    if folder is None:
        raise HTTPException(status_code = 404, detail = "Folder not found")
    return folder


@router.delete("/folders/{folder_id}")
def delete_folder(folder_id: str, current_subject: str = Depends(get_current_subject)) -> dict:
    if not library_db.delete_folder(folder_id):
        raise HTTPException(status_code = 404, detail = "Folder not found")
    return {"ok": True}
