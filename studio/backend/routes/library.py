# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Library API: one view over uploads, chat attachments, generated images and audio, fine-tuned
models and sandbox files, plus folders, favorites and renames. See ``core.library`` for the sources."""

import os
import re
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from auth.authentication import get_current_subject, subject_for_header_or_query_token
from core import library
from hub.services.models import account_access
from loggers import get_logger
from storage import library_db
from storage.studio_db import ChatMessageProtectedError
from utils.upload_limits import LIBRARY_UPLOAD_MAX_BYTES
from utils.utils import log_and_http_error, safe_curated_detail

logger = get_logger(__name__)

router = APIRouter()

_MAX_UPLOAD_BYTES = LIBRARY_UPLOAD_MAX_BYTES
_CHUNK_BYTES = 1024 * 1024
# Raster images, audio and video render inline; anything else (svg and html included) downloads as
# opaque bytes, as the sandbox route does, so a crafted upload cannot run script on the app origin.
# Exact types, never a prefix: a stored "audio/x, text/html" is not audio.
_INLINE_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp", "image/avif"})
_NOSNIFF = {"X-Content-Type-Options": "nosniff"}


def _inline_type(content_type: str) -> Optional[str]:
    value = library.media_type(content_type)
    if value in _INLINE_IMAGE_TYPES:
        return value
    playable = {
        known for known in library.content_types().values() if known.startswith(("audio/", "video/"))
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


@router.get("")
async def get_library(current_subject: str = Depends(get_current_subject)) -> dict:
    items = await run_in_threadpool(library.list_items)
    return {"items": items, "folders": library_db.list_folders()}


@router.get("/favorites")
def get_favorites(current_subject: str = Depends(get_current_subject)) -> dict:
    """Favorite item ids alone, for pages that mark favorites without listing every source."""
    return {
        "ids": [
            item_id for item_id, entry in library_db.list_entries().items() if entry["favorite"]
        ]
    }


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
        )
    except KeyError:
        raise HTTPException(status_code = 404, detail = "Folder not found")
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
            return copy_into_project(item.handle, body.projectId, item.folder, item.project_name)

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
    from utils.paths.path_utils import reveal_in_file_manager
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

    paths = {entry["key"]: entry["path"] for entry in await run_in_threadpool(library.locations)}
    if body.key not in paths:
        raise HTTPException(status_code = 404, detail = "Unknown location")
    path = Path(paths[body.key])
    await run_in_threadpool(lambda: path.mkdir(parents = True, exist_ok = True))
    await run_in_threadpool(_reveal, path)
    return {"ok": True}


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
        written = library.write_upload_text(upload_id, body.text)
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
