# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One listing over every place a Studio file can live.

Sources, each with its own id prefix so an item id says where its bytes are:

- ``upload:<id>``                         a file uploaded straight into the Library
- ``attachment:<message_id>:<attachment_id>``  a file attached to a chat message
- ``image:<id>``                          an image from the Images page gallery
- ``sandbox:<session_id>:<relative path>``     a file a chat tool wrote into its sandbox
- ``video:<id>``                          a clip from the Video page gallery
- ``audio:<id>``                          a clip from the Audio page's TTS gallery
- ``model:<training|exported>:<path>``    a fine-tuned model under outputs/ or exports/ (no bytes:
                                          a model is a directory, opened in chat instead)

Bytes are always served by the source's own route (``fileUrl``); the Library only lists, deletes
and layers its overlay (name, favorite, folder) on top.
"""

from __future__ import annotations

import functools
import os
import re
import threading
import uuid
from pathlib import Path
from typing import BinaryIO, Optional, Union
from urllib.parse import quote

from loggers import get_logger
from storage import library_db
from utils.paths import ensure_dir
from utils.paths.path_utils import is_path_within
from utils.paths.storage_roots import account_path

logger = get_logger(__name__)

_UPLOAD_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_PROJECT_SESSION_PREFIX = "project-"


def _to_ms(value: object) -> int:
    """Sources store seconds or milliseconds; the Library speaks milliseconds."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return int(number if number > 1e12 else number * 1000)


def _item(
    item_id: str,
    *,
    name: str,
    source: str,
    content_type: str,
    size_bytes: Optional[int],
    created_at: int,
    file_url: str,
    thread_id: Optional[str] = None,
    thread_title: Optional[str] = None,
    text_only: bool = False,
    model: Optional[dict] = None,
    archived: bool = False,
) -> dict:
    return {
        "id": item_id,
        "name": name,
        # The source's own name, which a rename leaves alone, so the type never follows it.
        "fileName": name,
        "source": source,
        "contentType": content_type,
        "sizeBytes": size_bytes,
        "createdAt": created_at,
        "updatedAt": created_at,
        "fileUrl": file_url,
        "threadId": thread_id,
        "threadTitle": thread_title,
        # Chat uploads of documents keep only their extracted text, so that is what downloads.
        "textOnly": text_only,
        "model": model,
        # Off its gallery page's active shelf, so that page cannot open it.
        "archived": archived,
    }


# ── Library uploads ──────────────────────────────────────────────


def uploads_dir() -> Path:
    return ensure_dir(account_path("library"))


def upload_path(upload_id: str) -> Optional[Path]:
    if not _UPLOAD_ID_RE.match(upload_id):
        return None
    return uploads_dir() / upload_id


def save_upload(name: str, content_type: str, chunks) -> dict:
    """Stream ``chunks`` to disk and record the upload."""
    upload_id = uuid.uuid4().hex
    final_path = uploads_dir() / upload_id
    tmp_path = uploads_dir() / f".{upload_id}.tmp"
    size = 0
    try:
        with open(tmp_path, "wb") as handle:
            for chunk in chunks:
                size += len(chunk)
                handle.write(chunk)
        os.replace(tmp_path, final_path)
        return library_db.insert_upload(upload_id, name, content_type, size)
    except BaseException:
        tmp_path.unlink(missing_ok = True)
        final_path.unlink(missing_ok = True)
        raise


def open_native_upload(lease: str):
    """Verify a desktop drop's signed path grant and open the file it names.

    Returns (name, content type, binary handle). The webview never names a path itself: the app
    signs the one the OS handed it, and the grant is re-checked here before a byte is read.
    """
    from utils.native_path_leases import verify_native_path_lease

    grant = verify_native_path_lease(
        lease,
        operation = "attach",
        expected_kind = "attachment",
        expected_path_type = "file",
    )
    # Refuses a path outside the acting account's workspace for a non-owner account.
    account_path(str(grant.canonical_path))
    name = grant.canonical_path.name
    return name, _guess_type(name), open(grant.canonical_path, "rb")


# Deleting an upload and rewriting a note each check its row, then change the file and the row.
# One at a time, or a delete could set the file aside mid-write, or drop a row another delete is
# still relying on, and leave a file with no row.
_upload_lock = threading.Lock()


def _swap_in(path: Path, data: bytes) -> None:
    """Replace the file whole: staged beside it and renamed over, so a failed write leaves it be."""
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_bytes(data)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok = True)
        raise


def write_upload_text(upload_id: str, text: str) -> bool:
    path = upload_path(upload_id)
    if path is None:
        return False
    data = text.encode("utf-8")
    with _upload_lock:
        if library_db.get_upload(upload_id) is None:
            return False
        previous = path.read_bytes() if path.exists() else None
        _swap_in(path, data)
        try:
            library_db.touch_upload(upload_id, len(data))
        except BaseException:
            # The row still describes the old note, so the old note goes back.
            if previous is not None:
                _swap_in(path, previous)
            raise
    return True


def _upload_items() -> list[dict]:
    items = []
    for upload in library_db.list_uploads():
        item = _item(
            f"upload:{upload['id']}",
            name = upload["name"],
            source = "uploaded",
            content_type = upload["contentType"],
            size_bytes = upload["sizeBytes"],
            created_at = upload["createdAt"],
            file_url = f"/api/library/uploads/{upload['id']}/file",
        )
        item["updatedAt"] = upload["updatedAt"]
        items.append(item)
    return items


# ── Chat attachments ─────────────────────────────────────────────


def _attachment_items() -> list[dict]:
    from storage.studio_db import list_chat_attachments

    items = []
    for attachment in list_chat_attachments():
        content_type = attachment.get("contentType") or ""
        has_bytes = attachment.get("type") == "image" or content_type.startswith(
            ("image/", "audio/", "video/")
        )
        message_id, attachment_id = attachment["messageId"], attachment["id"]
        items.append(
            _item(
                f"attachment:{message_id}:{attachment_id}",
                name = attachment.get("name") or "Attachment",
                source = "uploaded",
                content_type = content_type or "application/octet-stream",
                size_bytes = attachment.get("sizeBytes"),
                created_at = _to_ms(attachment.get("createdAt")),
                file_url = f"/api/chat/attachments/{quote(message_id, safe = '')}/{quote(attachment_id, safe = '')}/file",
                thread_id = attachment.get("threadId"),
                thread_title = attachment.get("threadTitle"),
                text_only = not has_bytes,
            )
        )
    return items


# ── Generated images ─────────────────────────────────────────────


def _prompt_name(prompt: str, fallback: str, extension: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|\s]+", " ", prompt).strip()
    if len(cleaned) > 60:
        cleaned = cleaned[:60].rsplit(" ", 1)[0] or cleaned[:60]
    return f"{cleaned or fallback}.{extension}"


def _file_size(path: Optional[Path]) -> Optional[int]:
    if path is None:
        return None
    try:
        return path.stat().st_size
    except OSError:
        return None


def _both_shelves(list_records) -> list[tuple[dict, bool]]:
    """(record, archived) for both shelves: archiving tidies a gallery page, the file is still
    Studio's."""
    return [(record, False) for record in list_records()] + [
        (record, True) for record in list_records(archived = True)
    ]


def _image_items() -> list[dict]:
    from core.inference import image_gallery

    items = []
    for record, archived in _both_shelves(image_gallery.list_images):
        items.append(
            _item(
                f"image:{record['id']}",
                name = _prompt_name(str(record.get("prompt") or ""), "Image", "png"),
                source = "generated",
                content_type = "image/png",
                size_bytes = _file_size(image_gallery.image_path(record["id"])),
                created_at = _to_ms(record.get("created_at")),
                file_url = record["url"],
                archived = archived,
            )
        )
    return items


# ── Generated video ──────────────────────────────────────────────


def _video_items() -> list[dict]:
    from core.inference import video_gallery
    return [
        _item(
            f"video:{record['id']}",
            name = _prompt_name(str(record.get("prompt") or ""), "Video", "mp4"),
            source = "generated",
            content_type = "video/mp4",
            size_bytes = _file_size(video_gallery.video_path(record["id"])),
            created_at = _to_ms(record.get("created_at")),
            file_url = record["url"],
            archived = archived,
        )
        for record, archived in _both_shelves(video_gallery.list_videos)
    ]


# ── Generated audio ──────────────────────────────────────────────


def _audio_items() -> list[dict]:
    from core.inference import audio_gallery
    return [
        _item(
            f"audio:{record['id']}",
            name = _prompt_name(str(record.get("prompt") or ""), "Audio", "wav"),
            source = "generated",
            content_type = "audio/wav",
            size_bytes = _file_size(audio_gallery.audio_path(record["id"])),
            created_at = _to_ms(record.get("created_at")),
            file_url = record["url"],
            archived = archived,
        )
        for record, archived in _both_shelves(audio_gallery.list_audio)
    ]


# ── Fine-tuned models ────────────────────────────────────────────

MODEL_CONTENT_TYPE = "application/x-unsloth-model"


def _tree_stats(path: Path) -> tuple[int, float]:
    """(total bytes, newest mtime) of a model file or directory, symlinks not followed."""
    if path.is_file():
        stat = path.stat()
        return stat.st_size, stat.st_mtime
    total, newest = 0, path.stat().st_mtime
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                stat = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            total += stat.st_size
            newest = max(newest, stat.st_mtime)
    return total, newest


def _model_items() -> list[dict]:
    from utils.models.model_config import (
        get_base_model_from_checkpoint,
        scan_exported_models,
        scan_trained_models,
    )
    from utils.paths.storage_roots import exports_root, outputs_root

    found = [
        (name, path, "training", model_type, None)
        for name, path, model_type in scan_trained_models(str(outputs_root()))
    ]
    found.extend(
        (name, path, "exported", export_type, base_model)
        for name, path, export_type, base_model in scan_exported_models(str(exports_root()))
    )
    items = []
    for name, path, origin, model_type, base_model in found:
        try:
            size, modified = _tree_stats(Path(path))
        except OSError:
            continue
        if base_model is None and origin == "training":
            try:
                base_model = get_base_model_from_checkpoint(path)
            except Exception:
                base_model = None
        items.append(
            _item(
                f"model:{origin}:{path}",
                name = name,
                source = "generated",
                content_type = MODEL_CONTENT_TYPE,
                size_bytes = size,
                created_at = _to_ms(modified),
                file_url = "",
                model = {
                    "path": path,
                    "origin": origin,
                    "exportType": model_type,
                    "baseModel": base_model,
                },
            )
        )
    return items


# ── Chat sandbox files ───────────────────────────────────────────


def _sandbox_sessions() -> list[tuple[str, Optional[str], Optional[str]]]:
    """(session id, thread id, title) for every chat or project that can own a sandbox."""
    from storage.studio_db import get_connection

    conn = get_connection()
    try:
        threads = conn.execute(
            "SELECT id, title FROM chat_threads WHERE project_id IS NULL"
        ).fetchall()
        projects = conn.execute("SELECT id, name, root_path FROM chat_projects").fetchall()
    finally:
        conn.close()
    sessions: list[tuple[str, Optional[str], Optional[str]]] = [
        (row["id"], row["id"], row["title"]) for row in threads
    ]
    workspaces = _project_workspaces()
    sessions.extend(
        (f"{_PROJECT_SESSION_PREFIX}{row['id']}", None, row["name"])
        for row in projects
        if _studio_project_root(row["root_path"], workspaces)
    )
    return sessions


def _project_workspaces() -> str:
    from utils.paths.storage_roots import project_workspaces_root
    return os.path.realpath(project_workspaces_root())


def _studio_project_root(root_path: Optional[str], workspaces: str) -> bool:
    """Whether a project works in Studio's own workspace. Every project is given a folder under
    the workspaces root when it is made, so the column is never empty; one pointed at a folder of
    the user's own works in that folder, and those files are not Studio's."""
    if not root_path:
        return True
    return is_path_within(os.path.realpath(os.path.expanduser(root_path)), workspaces)


def _sandbox_file(ref: str) -> Path:
    """The file a ``sandbox:`` id names, only if the Library lists it: an eligible session, a
    servable file, no dotfile. Raises LookupError otherwise, so a crafted id reaches nothing else."""
    from core.inference.tools import resolve_sandbox_workdir
    from routes.inference import _sandbox_listing

    session_id, _, relative = ref.partition(":")
    if session_id not in {session for session, _, _ in _sandbox_sessions()}:
        raise LookupError(ref)
    if not relative or any(part.startswith(".") for part in relative.split("/")):
        raise LookupError(ref)
    directory = os.path.realpath(resolve_sandbox_workdir(session_id))
    listed = {entry["name"].replace(os.sep, "/") for entry in _sandbox_listing(directory)}
    path = os.path.realpath(os.path.join(directory, *relative.split("/")))
    if relative not in listed or not path.startswith(directory + os.sep):
        raise LookupError(ref)
    if not os.path.isfile(path) or os.path.islink(path):
        raise LookupError(ref)
    return Path(path)


def _sandbox_items() -> list[dict]:
    from core.inference.tools import resolve_sandbox_workdir
    from routes.inference import _sandbox_listing

    items = []
    for session_id, thread_id, title in _sandbox_sessions():
        try:
            directory = os.path.realpath(resolve_sandbox_workdir(session_id))
            listing = _sandbox_listing(directory)
        except Exception:
            logger.debug("library.sandbox_listing_failed", exc_info = True)
            continue
        for entry in listing:
            relative = entry["name"].replace(os.sep, "/")
            if os.path.basename(relative).startswith("."):
                continue
            items.append(
                _item(
                    f"sandbox:{session_id}:{relative}",
                    name = os.path.basename(relative),
                    source = "generated",
                    content_type = _guess_type(relative),
                    size_bytes = entry["size"],
                    created_at = _to_ms(entry["modified"]),
                    file_url = f"/api/inference/sandbox/{quote(session_id, safe = '')}/{quote(relative)}",
                    thread_id = thread_id,
                    thread_title = title,
                )
            )
    return items


# By extension and the same on every OS: mimetypes reads the Windows registry, where any installed
# app can remap a type, and it names `.ts` an MPEG transport stream. The sandbox route's raster
# types are merged in, so the two agree on every image.
_CONTENT_TYPES = {
    ".svg": "image/svg+xml",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".ico": "image/x-icon",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".weba": "audio/webm",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".ogv": "video/ogg",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".log": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".jsonl": "application/jsonl",
    ".xml": "application/xml",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".toml": "text/plain",
    ".ini": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
    ".css": "text/css",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".cjs": "text/javascript",
    ".jsx": "text/javascript",
    ".ts": "text/typescript",
    ".tsx": "text/typescript",
    ".mts": "text/typescript",
    ".cts": "text/typescript",
    ".py": "text/x-python",
    ".ipynb": "application/x-ipynb+json",
    ".sh": "text/x-shellscript",
    ".sql": "text/plain",
    ".rtf": "application/rtf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".odp": "application/vnd.oasis.opendocument.presentation",
    ".epub": "application/epub+zip",
    ".zip": "application/zip",
    ".gz": "application/gzip",
    ".tar": "application/x-tar",
    ".parquet": "application/vnd.apache.parquet",
    ".safetensors": "application/octet-stream",
    ".gguf": "application/octet-stream",
}

_MEDIA_TYPE_RE = re.compile(r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*")

# Types a browser runs or renders as a document of its own. A client that declares one for a file
# whose extension says otherwise does not get it stored.
_ACTIVE_TYPES = frozenset(
    {
        "text/html",
        "application/xhtml+xml",
        "image/svg+xml",
        "application/pdf",
        "text/xml",
        "application/xml",
        "text/javascript",
        "application/javascript",
        "application/x-shockwave-flash",
    }
)


@functools.lru_cache(maxsize = None)
def content_types() -> dict[str, str]:
    from routes.inference import _SANDBOX_MEDIA_TYPES
    return {**_CONTENT_TYPES, **_SANDBOX_MEDIA_TYPES}


def media_type(value: object) -> Optional[str]:
    """The one media type ``value`` names, lower case and without parameters; None when it names
    none, or several (``audio/x, text/html``)."""
    essence = str(value or "").split(";", 1)[0].strip().lower()
    return essence if _MEDIA_TYPE_RE.fullmatch(essence) else None


def _guess_type(name: str) -> str:
    return content_types().get(os.path.splitext(name)[1].lower(), "application/octet-stream")


def upload_content_type(name: str, declared: Optional[str]) -> str:
    """What an upload is stored as: its extension's type when the map knows it, else the type the
    client declared, as long as it is one type and nothing a browser would run."""
    known = content_types().get(os.path.splitext(name)[1].lower())
    if known:
        return known
    declared_type = media_type(declared)
    if declared_type is None or declared_type in _ACTIVE_TYPES:
        return "application/octet-stream"
    return declared_type


# ── Public API ───────────────────────────────────────────────────

_SOURCES = (
    _upload_items,
    _attachment_items,
    _image_items,
    _video_items,
    _audio_items,
    _model_items,
    _sandbox_items,
)


def list_items() -> list[dict]:
    """Every item with its overlay applied, newest activity first. A failing source is skipped so
    one broken store cannot empty the whole Library."""
    overlay = library_db.list_entries()
    items: list[dict] = []
    for source in _SOURCES:
        try:
            items.extend(source())
        except Exception:
            logger.warning("library.source_failed: %s", source.__name__, exc_info = True)
    for item in items:
        entry = overlay.get(item["id"])
        item["favorite"] = bool(entry and entry["favorite"])
        item["folderId"] = entry["folderId"] if entry else None
        if entry and entry["name"]:
            item["name"] = entry["name"]
    items.sort(key = lambda item: item["updatedAt"], reverse = True)
    return items


def _project_name(name: str, item_id: str) -> str:
    """A readable name that stays unique per item, so adding it twice is a no-op."""
    import hashlib

    stem, ext = os.path.splitext(os.path.basename(name).lstrip(".") or "file")
    return f"{stem[:80]}-{hashlib.sha1(item_id.encode()).hexdigest()[:8]}{ext[:16]}"


def project_source(item_id: str) -> tuple[Path, str, str]:
    """(file, project folder, file name) for copying an item into a project.

    Raises LookupError when the item is gone and ValueError for items with no file of their own
    (chat attachments live inside messages, fine-tunes are folders)."""
    kind, _, ref = item_id.partition(":")
    path: Optional[Path] = None
    if kind == "upload":
        record = library_db.get_upload(ref)
        path = upload_path(ref)
        if record is None or path is None or not path.is_file():
            raise LookupError(item_id)
        return path, "files", _project_name(record["name"], item_id)
    if kind in ("image", "video", "audio"):
        from core.inference import audio_gallery, image_gallery, video_gallery

        owned = {
            "image": (image_gallery.owned_image_path, "images"),
            "video": (video_gallery.owned_video_path, "videos"),
            "audio": (audio_gallery.owned_audio_path, "audio"),
        }
        resolve, folder = owned[kind]
        path = resolve(ref)
        if path is None:
            raise LookupError(item_id)
        # Same name as the gallery's own Add to project, so either one finds the other's copy.
        return path, folder, path.name
    if kind == "sandbox":
        path = _sandbox_file(ref)
        return path, "files", _project_name(path.name, item_id)
    raise ValueError("This item cannot be added to a project.")


def local_path(item_id: str) -> Path:
    """The file, or model folder, behind an item, for Reveal in Finder.

    Same errors as ``project_source``. A model path comes from the id, so it must sit inside the
    outputs or exports root."""
    kind, _, ref = item_id.partition(":")
    if kind != "model":
        return project_source(item_id)[0]
    from utils.paths.storage_roots import exports_root, outputs_root

    _origin, _, path = ref.partition(":")
    resolved = os.path.realpath(path) if path else ""
    roots = [os.path.realpath(root) for root in (outputs_root(), exports_root())]
    if (
        not resolved
        or not any(is_path_within(resolved, root) for root in roots)
        or not os.path.exists(resolved)
    ):
        raise LookupError(item_id)
    return Path(resolved)


# Cards are up to a few hundred CSS pixels wide, so twice that for high-density screens.
_THUMBNAIL_WIDTH = 640
# The grid crops a picture past these heights (as a share of its width), so the rest is never sent.
_THUMBNAIL_MIN_RATIO = 2 / 3
_THUMBNAIL_MAX_RATIO = 3 / 2
# A small file can still decode to an enormous bitmap; past this many pixels a card shows its icon.
_THUMBNAIL_MAX_PIXELS = 64_000_000


def _attachment_media(ref: str) -> tuple[str, bytes]:
    """The image or clip stored in a chat attachment, with its type. LookupError when it holds none."""
    from routes.chat_history import _decode_attachment_base64
    from storage.studio_db import get_chat_attachment

    message_id, _, attachment_id = ref.partition(":")
    attachment = get_chat_attachment(message_id, attachment_id) or {}
    for part in attachment.get("content") or []:
        if not isinstance(part, dict) or part.get("type") != "file":
            continue
        data = part.get("data")
        mime_type = str(part.get("mimeType") or attachment.get("contentType") or "").lower()
        if isinstance(data, str) and data and mime_type.startswith(("image/", "video/")):
            return mime_type, _decode_attachment_base64(data)
    raise LookupError(ref)


def _image_thumbnail(source: Union[Path, BinaryIO]) -> bytes:
    """The picture cropped as a card shows it and at most `_THUMBNAIL_WIDTH` wide, as WebP."""
    import io

    try:
        from PIL import Image, ImageOps
    except Exception as exc:  # noqa: BLE001 -- a missing dependency makes thumbnails unavailable
        raise RuntimeError("Thumbnail generation needs the 'Pillow' package.") from exc
    try:
        with Image.open(source) as opened:
            # Read from the header, before anything is decoded.
            if opened.width * opened.height > _THUMBNAIL_MAX_PIXELS:
                raise RuntimeError(f"{opened.width}x{opened.height} is too large to thumbnail.")
            # A JPEG decodes straight at a fraction of its size, so a huge photo stays cheap.
            opened.draft("RGB", (_THUMBNAIL_WIDTH, _THUMBNAIL_WIDTH))
            image = ImageOps.exif_transpose(opened)
            width, height = image.size
            if height > width * _THUMBNAIL_MAX_RATIO:
                # From the top, where a screenshot or a page starts.
                image = image.crop((0, 0, width, round(width * _THUMBNAIL_MAX_RATIO)))
            elif height < width * _THUMBNAIL_MIN_RATIO:
                keep = round(height / _THUMBNAIL_MIN_RATIO)
                left = (width - keep) // 2
                image = image.crop((left, 0, left + keep, height))
            image.thumbnail(
                (_THUMBNAIL_WIDTH, round(_THUMBNAIL_WIDTH * _THUMBNAIL_MAX_RATIO)), Image.LANCZOS
            )
            if image.mode not in ("RGB", "RGBA"):
                transparent = "A" in image.getbands() or "transparency" in image.info
                image = image.convert("RGBA" if transparent else "RGB")
            buf = io.BytesIO()
            image.save(buf, format = "WEBP", quality = 85, method = 4)
            return buf.getvalue()
    except Exception as exc:  # noqa: BLE001 -- surface decoder / WebP encoder failures
        raise RuntimeError(f"Thumbnail generation failed to decode the image: {exc}") from exc


def thumbnail(item_id: str) -> bytes:
    """A card's picture: an image cropped and scaled down, or a video's first frame, as WebP.

    LookupError when the item is gone or has no picture; RuntimeError when it cannot be decoded."""
    import io

    from core.inference import video_gallery

    kind, _, ref = item_id.partition(":")
    source: Union[Path, BinaryIO]
    if kind == "attachment":
        mime_type, data = _attachment_media(ref)
        source = io.BytesIO(data)
    elif kind in ("upload", "image", "video", "sandbox"):
        source = project_source(item_id)[0]
        if kind == "upload":
            record = library_db.get_upload(ref) or {}
            types = (
                str(record.get("contentType") or ""),
                _guess_type(str(record.get("name") or "")),
            )
        elif kind == "sandbox":
            types = (_guess_type(source.name),)
        else:
            types = (f"{kind}/",)
        mime_type = next(
            (value.lower() for value in types if value.lower().startswith(("image/", "video/"))), ""
        )
    else:
        raise LookupError(item_id)
    if mime_type.startswith("video/"):
        return video_gallery.first_frame_webp(source, width = _THUMBNAIL_WIDTH)
    if mime_type.startswith("image/") and mime_type != "image/svg+xml":
        return _image_thumbnail(source)
    raise LookupError(item_id)


def locations() -> list[dict]:
    """Where each kind of Library file lives, for Settings > Library."""
    from core.inference import audio_gallery, image_gallery, video_gallery
    from utils.paths.storage_roots import exports_root, outputs_root

    return [
        {"key": key, "path": str(resolve())}
        for key, resolve in (
            ("uploads", uploads_dir),
            ("images", image_gallery.gallery_dir),
            ("videos", video_gallery.gallery_dir),
            ("audio", audio_gallery.gallery_dir),
            ("fineTunes", outputs_root),
            ("exports", exports_root),
        )
    ]


def _delete_upload(upload_id: str, path: Path) -> bool:
    with _upload_lock:
        return _delete_upload_locked(upload_id, path)


def _delete_upload_locked(upload_id: str, path: Path) -> bool:
    """Set the file aside before dropping its row, so a failure at either step leaves both."""
    staged = path.with_name(f".{upload_id}.deleting")
    try:
        os.replace(path, staged)
    except FileNotFoundError:
        return library_db.delete_upload(upload_id)
    try:
        deleted = library_db.delete_upload(upload_id)
    except BaseException:
        os.replace(staged, path)
        raise
    if deleted:
        staged.unlink(missing_ok = True)
    else:
        os.replace(staged, path)
    return deleted


class DeleteIncomplete(RuntimeError):
    """A delete that stopped part way; the item is still there to delete again."""


def delete_item(item_id: str) -> bool:
    """Delete an item from its source. Returns False when the source no longer has it."""
    kind, _, ref = item_id.partition(":")
    deleted = False
    if kind == "upload":
        path = upload_path(ref)
        if path is not None:
            deleted = _delete_upload(ref, path)
    elif kind == "attachment":
        from storage.studio_db import delete_chat_attachment
        message_id, _, attachment_id = ref.partition(":")
        deleted = delete_chat_attachment(message_id, attachment_id)
    elif kind == "image":
        from core.inference import image_gallery
        deleted = image_gallery.delete(ref)
    elif kind == "video":
        from core.inference import video_gallery
        from routes.video import _forget_openai_job, _forget_terminal_video

        # Only for a clip of ours, so a guessed id cannot drop a running generation's job.
        if video_gallery.get_record(ref) is not None:
            # The job goes first, as the Video page's cleanup, so /v1/videos never keeps a ghost
            # of the clip. Failing there leaves the clip listed, and deleting it again finishes.
            if not _forget_openai_job(ref):
                raise DeleteIncomplete("Could not delete the video job; try again.")
            deleted = video_gallery.delete(ref)
            if deleted:
                _forget_terminal_video(ref)
    elif kind == "audio":
        from core.inference import audio_gallery
        deleted = audio_gallery.delete(ref)
    elif kind == "sandbox":
        try:
            os.unlink(_sandbox_file(ref))
            deleted = True
        except LookupError:
            deleted = False
    elif kind == "model":
        # Deleting a model has load and training guards of its own; the model picker owns that.
        raise ValueError("Fine-tuned models are deleted from the model picker.")
    else:
        raise ValueError("Unknown library item")
    if deleted:
        library_db.delete_entry(item_id)
    return deleted
