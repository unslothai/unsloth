# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Library API: one view over uploads, chat attachments, generated images and audio, fine-tuned
models and sandbox files, plus folders, favorites and renames. See ``core.library`` for the sources."""

import re
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from auth.authentication import get_current_subject
from core import library
from loggers import get_logger
from storage import library_db
from storage.studio_db import ChatMessageProtectedError
from utils.utils import log_and_http_error, safe_curated_detail

logger = get_logger(__name__)

router = APIRouter()

_MAX_UPLOAD_BYTES = 512 * 1024 * 1024
_CHUNK_BYTES = 1024 * 1024
# Raster images render inline; anything else (svg and html included) downloads as opaque bytes, as
# the sandbox route does, so a crafted upload cannot run script on the app origin.
_INLINE_PREFIXES = (
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/avif",
    "audio/",
    "video/",
)


class ItemPatch(BaseModel):
    id: str = Field(max_length = 4096)
    name: Optional[str] = Field(default = None, min_length = 1, max_length = 255)
    favorite: Optional[bool] = None
    # Present-and-null moves the item back to the Library root.
    folderId: Optional[str] = Field(default = None, max_length = 64)


class ItemRef(BaseModel):
    id: str = Field(max_length = 4096)


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
    except ChatMessageProtectedError as exc:
        raise log_and_http_error(
            exc,
            409,
            safe_curated_detail(exc),
            event = "library.delete_protected",
            log = logger,
        ) from exc
    if not deleted:
        raise HTTPException(status_code = 404, detail = "Item not found")
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

    def _save_native(lease: str) -> dict:
        try:
            name, content_type, handle = library.open_native_upload(lease)
        except ValueError as exc:
            # A bad or expired grant, or a path outside this account's workspace.
            raise HTTPException(status_code = 400, detail = str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code = 400, detail = "Dropped file could not be read.") from exc
        with handle:
            return library.save_upload(name, content_type, _chunks(handle, name))

    records = []
    try:
        for upload in files or []:
            name = _base_name(upload.filename)
            records.append(
                await run_in_threadpool(
                    library.save_upload,
                    name,
                    upload.content_type or "application/octet-stream",
                    _chunks(upload.file, name),
                )
            )
        for lease in nativePathLeases or []:
            records.append(await run_in_threadpool(_save_native, lease))
    except BaseException:
        # All or nothing, so a retry never duplicates the files that did make it.
        for record in records:
            await run_in_threadpool(library.delete_item, f"upload:{record['id']}")
        raise

    ids = [f"upload:{record['id']}" for record in records]
    if folderId:
        for item_id in ids:
            library_db.update_entry(item_id, folder_id = folderId, move = True)
    return {"ids": ids}


@router.get("/uploads/{upload_id}/file")
def get_upload_file(upload_id: str, current_subject: str = Depends(get_current_subject)):
    record = library_db.get_upload(upload_id)
    path = library.upload_path(upload_id)
    if record is None or path is None or not path.is_file():
        raise HTTPException(status_code = 404, detail = "File not found")
    content_type = record["contentType"].lower()
    if content_type.startswith(_INLINE_PREFIXES):
        return FileResponse(path, media_type = content_type)
    return FileResponse(
        path,
        media_type = "application/octet-stream",
        filename = record["name"],
        headers = {"X-Content-Type-Options": "nosniff"},
    )


@router.put("/uploads/{upload_id}/text")
def put_upload_text(
    upload_id: str,
    body: TextContent,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    if not library.write_upload_text(upload_id, body.text):
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
