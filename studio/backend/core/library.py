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
import hashlib
import importlib
import io
import os
import re
import stat
import threading
import time
import uuid
import zlib
from collections import OrderedDict
from pathlib import Path
from types import ModuleType
from typing import BinaryIO, Callable, NamedTuple, Optional, Union
from urllib.parse import quote, unquote

from core.inference.gallery_projects import UNSAFE_NAME_CHARS, _bad_name
from loggers import get_logger
from storage import library_db
from utils.paths import ensure_dir
from utils.paths.path_utils import is_path_within, same_path
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
    fingerprint: Optional[str] = None,
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
        # Which file a path-derived id found; the listing takes it off before answering.
        "_fingerprint": fingerprint,
    }


def _fingerprint(info: os.stat_result) -> str:
    """Which file a path-derived id (``sandbox:``, ``model:``) found: its inode (file id on Windows)
    and birth time where the OS keeps one, so an edit in place keeps it and a path made again does
    not. Not the device, which a remount changes. Linux keeps no birth time and ext4 reuses inodes,
    so there a file recreated at once can pass for the old one."""
    birth = getattr(info, "st_birthtime", None)
    if birth is None and os.name == "nt":
        # Before Python 3.12, Windows kept the creation time in st_ctime.
        birth = info.st_ctime
    return str(info.st_ino) if birth is None else f"{info.st_ino}:{birth!r}"


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


def _verify_native(lease: str, *, consume: bool):
    from utils.native_path_leases import verify_native_path_lease

    grant = verify_native_path_lease(
        lease,
        operation = "attach",
        expected_kind = "attachment",
        expected_path_type = "file",
        consume = consume,
    )
    # Refuses a path outside the acting account's workspace for a non-owner account.
    account_path(str(grant.canonical_path))
    return grant


def check_native_upload(lease: str) -> tuple[str, int]:
    """(name, size) of a desktop drop, its grant checked but not spent, so a batch can check all."""
    grant = _verify_native(lease, consume = False)
    return grant.canonical_path.name, os.stat(grant.canonical_path).st_size


def open_native_upload(lease: str):
    """(name, content type, binary handle) of a desktop drop: the app signs the path the OS handed
    it, never the webview, and the grant is spent here before a byte is read."""
    grant = _verify_native(lease, consume = True)
    name = grant.canonical_path.name
    return name, _guess_type(name), open(grant.canonical_path, "rb")


# Deletes and note saves check a row, then change the file and the row: one at a time, or a file
# can be left with no row.
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


def write_upload_text(
    upload_id: str,
    text: str,
    encoding: str = "utf-8",
) -> bool:
    """Write an edited note back in `encoding`. A BOM, if the note had one, is the text's first
    character, which each of these codecs writes as that encoding's own BOM."""
    path = upload_path(upload_id)
    if path is None:
        return False
    # A lone surrogate has no encoding in any of them; the route answers that with a 400.
    data = text.encode(encoding)
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


# A crash can leave a staged upload, note or delete behind; nothing that old is still being written.
_LEFTOVER_AGE_SECONDS = 3600
_swept_at: dict[str, float] = {}


def _sweep_leftovers() -> None:
    """Clear what a crash left in the uploads folder, at most once an hour per account. A delete
    that stopped between setting its file aside and dropping the row put the file back instead:
    the row still lists it."""
    directory = uploads_dir()
    now = time.time()
    if now - _swept_at.get(str(directory), 0) < _LEFTOVER_AGE_SECONDS:
        return
    _swept_at[str(directory)] = now
    # Deletes and note saves stage under this lock, so none of theirs is mid-way here.
    with _upload_lock:
        with os.scandir(directory) as entries:
            leftovers = [
                entry
                for entry in entries
                if entry.name.startswith(".") and entry.name.endswith((".tmp", ".deleting"))
            ]
        for entry in leftovers:
            try:
                info = entry.stat(follow_symlinks = False)
                if not stat.S_ISREG(info.st_mode) or now - info.st_mtime < _LEFTOVER_AGE_SECONDS:
                    continue
                upload_id = entry.name[1 : -len(".deleting")]
                if (
                    entry.name.endswith(".deleting")
                    and _UPLOAD_ID_RE.match(upload_id)
                    and library_db.get_upload(upload_id) is not None
                    and not (directory / upload_id).exists()
                ):
                    os.replace(entry.path, directory / upload_id)
                else:
                    os.unlink(entry.path)
            except OSError:
                logger.debug("library.leftover_sweep_failed: %s", entry.name, exc_info = True)


def _upload_items() -> list[dict]:
    try:
        _sweep_leftovers()
    except OSError:
        logger.debug("library.leftover_sweep_failed", exc_info = True)
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


def _attachment_id(message_id: str, attachment_id: str) -> str:
    """The item id of a chat attachment. A message id can be any string, so it is encoded: the
    attachment id is what follows its first colon."""
    return f"attachment:{quote(message_id, safe = '')}:{attachment_id}"


def _attachment_ref(ref: str) -> tuple[str, str]:
    """The message and attachment ids an ``attachment:`` item id's ref names."""
    message, _, attachment = ref.partition(":")
    return unquote(message), attachment


def _attachment_items() -> list[dict]:
    from storage.studio_db import list_chat_attachments

    items = []
    for attachment in list_chat_attachments():
        # The essence, as the attachment route reads it: a type is case-insensitive, and a
        # recorded clip's carries parameters (video/webm;codecs=vp9).
        content_type = str(attachment.get("contentType") or "").split(";", 1)[0].strip().lower()
        has_bytes = attachment.get("type") in ("image", "audio") or content_type.startswith(
            ("image/", "audio/", "video/")
        )
        message_id, attachment_id = attachment["messageId"], attachment["id"]
        items.append(
            _item(
                _attachment_id(message_id, attachment_id),
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


# ── Generated images, video and audio ────────────────────────────


# A run of what Windows refuses in a file name, or of whitespace: a prompt's line breaks make no name.
_NAME_BREAK_RE = re.compile(f"[{UNSAFE_NAME_CHARS}\\s]+")


def _prompt_name(prompt: str, fallback: str, extension: str) -> str:
    cleaned = _NAME_BREAK_RE.sub(" ", prompt).strip()
    if len(cleaned) > 60:
        cleaned = cleaned[:60].rsplit(" ", 1)[0] or cleaned[:60]
    return f"{cleaned or fallback}.{extension}"


class _Gallery(NamedTuple):
    module: ModuleType
    records: Callable
    resolve: Callable
    label: str  # what an item with no prompt is called
    extension: str
    content_type: str
    folder: str  # the project folder Add to project copies into


# kind: its module in core.inference, its list and path functions, then the rest of _Gallery.
_GALLERIES = {
    "image": ("image_gallery", "list_images", "image_path", "Image", "png", "image/png", "images"),
    "video": ("video_gallery", "list_videos", "video_path", "Video", "mp4", "video/mp4", "videos"),
    "audio": ("audio_gallery", "list_audio", "audio_path", "Audio", "wav", "audio/wav", "audio"),
}


def _gallery(kind: str) -> _Gallery:
    module_name, records, resolve, *rest = _GALLERIES[kind]
    module = importlib.import_module(f"core.inference.{module_name}")
    return _Gallery(module, getattr(module, records), getattr(module, resolve), *rest)


def _gallery_items(kind: str) -> list[dict]:
    gallery = _gallery(kind)
    items = []
    # Both shelves: archiving tidies a gallery page, the file is still Studio's.
    for archived in (False, True):
        for record in gallery.records(archived = archived):
            path = gallery.resolve(record["id"])
            try:
                size = path.stat().st_size if path is not None else None
            except OSError:
                size = None
            items.append(
                _item(
                    f"{kind}:{record['id']}",
                    name = _prompt_name(
                        str(record.get("prompt") or ""), gallery.label, gallery.extension
                    ),
                    source = "generated",
                    content_type = gallery.content_type,
                    size_bytes = size,
                    created_at = _to_ms(record.get("created_at")),
                    file_url = record["url"],
                    archived = archived,
                )
            )
    return items


def _image_items() -> list[dict]:
    return _gallery_items("image")


def _video_items() -> list[dict]:
    return _gallery_items("video")


def _audio_items() -> list[dict]:
    return _gallery_items("audio")


# ── Fine-tuned models ────────────────────────────────────────────

MODEL_CONTENT_TYPE = "application/x-unsloth-model"


def _tree_stats(path: Path) -> tuple[int, float, os.stat_result]:
    """(total bytes, newest mtime, the path's own stat) of a model file or directory, symlinks
    not followed."""
    info = path.stat()
    if stat.S_ISREG(info.st_mode):
        return info.st_size, info.st_mtime, info
    total, newest = 0, info.st_mtime
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                entry = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            total += entry.st_size
            newest = max(newest, entry.st_mtime)
    return total, newest, info


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
            size, modified, info = _tree_stats(Path(path))
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
                fingerprint = _fingerprint(info),
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
    return [(row["id"], row["id"], row["title"]) for row in threads] + [
        (f"{_PROJECT_SESSION_PREFIX}{row['id']}", None, row["name"])
        for row in projects
        if _studio_project_root(row["root_path"])
    ]


def _studio_project_root(root_path: Optional[str]) -> bool:
    """Whether a project works in Studio's own workspace (every project is given a folder there),
    not a folder of the user's own, whose files are not Studio's."""
    from utils.paths.storage_roots import project_workspaces_root

    if not root_path:
        return True
    return is_path_within(
        os.path.realpath(os.path.expanduser(root_path)),
        os.path.realpath(project_workspaces_root()),
    )


def _sandbox_session_eligible(session_id: str) -> bool:
    """Whether one session id is among ``_sandbox_sessions``, asked of that row alone."""
    from storage.studio_db import get_connection

    conn = get_connection()
    try:
        if conn.execute(
            "SELECT 1 FROM chat_threads WHERE id = ? AND project_id IS NULL", (session_id,)
        ).fetchone():
            return True
        if not session_id.startswith(_PROJECT_SESSION_PREFIX):
            return False
        row = conn.execute(
            "SELECT root_path FROM chat_projects WHERE id = ?",
            (session_id[len(_PROJECT_SESSION_PREFIX) :],),
        ).fetchone()
    finally:
        conn.close()
    return row is not None and _studio_project_root(row["root_path"])


def _sandbox_names(directory: str) -> frozenset:
    """The files one sandbox's listing holds, remembered with the listing, so a grid of cards walks
    a sandbox once rather than once a card."""
    from routes.inference import _sandbox_listing_names
    return _LISTING.get(
        ("names", directory),
        lambda: frozenset(_sandbox_listing_names(directory) if os.path.isdir(directory) else []),
        _SANDBOX_TTL_SECONDS,
    )


def _sandbox_path(ref: str) -> str:
    """The file a ``sandbox:`` id names, by the listing walk's rules: an eligible session, servable
    segments, no dotfile, not too deep, no link on the way, and within the walk's cap. Checked for
    this one file, so a card never lists every chat. LookupError otherwise."""
    from core.inference.tools import (
        _MAX_SANDBOX_PATH_SEGMENTS,
        _servable_segment,
        _user_path_parts,
        resolve_sandbox_workdir,
    )

    session_id, _, relative = ref.partition(":")
    parts = relative.split("/")
    if not relative or any(part.startswith(".") or not _servable_segment(part) for part in parts):
        raise LookupError(ref)
    if not _sandbox_session_eligible(session_id):
        raise LookupError(ref)
    directory = os.path.realpath(resolve_sandbox_workdir(session_id))
    if len(_user_path_parts(parts, directory)) > _MAX_SANDBOX_PATH_SEGMENTS:
        raise LookupError(ref)
    path = os.path.join(directory, *parts)
    if not same_path(os.path.realpath(path), path) or not is_path_within(path, directory):
        raise LookupError(ref)
    # Past the listing's cap, or skipped by its walk: never listed, so not reachable by id either.
    if relative not in _sandbox_names(directory):
        raise LookupError(ref)
    return path


def _open_sandbox_file(ref: str) -> tuple[BinaryIO, str]:
    """(open file, path) for a ``sandbox:`` id, opened once and checked by its descriptor.

    Tool code runs here, so a name checked then reopened can be a link by then. O_NOFOLLOW where the
    OS has it, then the path must still resolve to itself and to this same regular file, which also
    refuses a parent swapped for a link (the only way in on Windows)."""
    path = _sandbox_path(ref)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        raise LookupError(ref) from None
    try:
        info = os.fstat(fd)
        checked = os.stat(path, follow_symlinks = False)
        if (
            not stat.S_ISREG(info.st_mode)
            or not same_path(os.path.realpath(path), path)
            or (checked.st_dev, checked.st_ino) != (info.st_dev, info.st_ino)
        ):
            raise LookupError(ref)
    except OSError:
        os.close(fd)
        raise LookupError(ref) from None
    except BaseException:
        os.close(fd)
        raise
    return os.fdopen(fd, "rb"), path


def _delete_sandbox_file(ref: str) -> bool:
    """Unlink the file an id names, only while its name is still the file checked."""
    from core.inference.gallery_projects import _DIR_FLAGS, _USE_DIR_FD

    try:
        handle, path = _open_sandbox_file(ref)
    except LookupError:
        return False
    parent, name = os.path.split(path)
    with handle:
        info = os.fstat(handle.fileno())
        if _USE_DIR_FD:
            # By the folder's descriptor, so a parent swapped for a link since is not followed.
            dir_fd = os.open(parent, _DIR_FLAGS)
            try:
                entry = os.stat(name, dir_fd = dir_fd, follow_symlinks = False)
                if (entry.st_dev, entry.st_ino) != (info.st_dev, info.st_ino):
                    return False
                os.unlink(name, dir_fd = dir_fd)
            finally:
                os.close(dir_fd)
            return True
    # Windows cannot delete an open file; there a link needs privileges tool code lacks.
    os.unlink(path)
    return True


def _sandbox_items() -> list[dict]:
    from core.inference.tools import resolve_sandbox_workdir
    from routes.inference import _sandbox_listing_names

    items = []
    generation = _LISTING.generation
    for session_id, thread_id, title in _sandbox_sessions():
        try:
            directory = os.path.realpath(resolve_sandbox_workdir(session_id))
            names = _sandbox_listing_names(directory) if os.path.isdir(directory) else []
        except Exception:
            logger.debug("library.sandbox_listing_failed", exc_info = True)
            continue
        # Left for the per-item routes, so a card just listed is checked without walking again.
        _LISTING.put(("names", directory), frozenset(names), generation)
        for relative in names:
            if os.path.basename(relative).startswith("."):
                continue
            try:
                info = os.stat(os.path.join(directory, relative))
            except OSError:
                continue
            items.append(
                _item(
                    f"sandbox:{session_id}:{relative}",
                    name = os.path.basename(relative),
                    source = "generated",
                    content_type = _guess_type(relative),
                    size_bytes = info.st_size,
                    # To the millisecond: a card's thumbnail is cached by it, and a tool can write
                    # the file again within a second.
                    created_at = info.st_mtime_ns // 1_000_000,
                    file_url = f"/api/inference/sandbox/{quote(session_id, safe = '')}/{quote(relative)}",
                    thread_id = thread_id,
                    thread_title = title,
                    fingerprint = _fingerprint(info),
                )
            )
    return items


# By extension and the same on every OS: mimetypes reads the Windows registry, where any installed
# app can remap a type, and it names `.ts` an MPEG transport stream. The sandbox route's raster
# types are merged in, so the two agree on every image.
_CONTENT_TYPES = {
    f".{extension}": content_type
    for content_type, extensions in (
        ("image/svg+xml", "svg"),
        ("image/tiff", "tif tiff"),
        ("image/x-icon", "ico"),
        ("image/heic", "heic"),
        ("image/heif", "heif"),
        ("audio/mpeg", "mp3"),
        ("audio/wav", "wav"),
        ("audio/ogg", "ogg oga opus"),
        ("audio/flac", "flac"),
        ("audio/mp4", "m4a"),
        ("audio/aac", "aac"),
        ("audio/webm", "weba"),
        ("video/mp4", "mp4 m4v"),
        ("video/quicktime", "mov"),
        ("video/webm", "webm"),
        ("video/x-matroska", "mkv"),
        ("video/x-msvideo", "avi"),
        ("video/ogg", "ogv"),
        ("application/pdf", "pdf"),
        ("text/plain", "txt log toml ini sql"),
        ("text/markdown", "md markdown"),
        ("text/csv", "csv"),
        ("text/tab-separated-values", "tsv"),
        ("application/json", "json"),
        ("application/jsonl", "jsonl"),
        ("application/xml", "xml"),
        ("text/yaml", "yaml yml"),
        ("text/html", "html htm"),
        ("text/css", "css"),
        ("text/javascript", "js mjs cjs jsx"),
        ("text/typescript", "ts tsx mts cts"),
        ("text/x-python", "py"),
        ("application/x-ipynb+json", "ipynb"),
        ("text/x-shellscript", "sh"),
        ("application/rtf", "rtf"),
        ("application/msword", "doc"),
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"),
        ("application/vnd.ms-excel", "xls"),
        ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"),
        ("application/vnd.ms-powerpoint", "ppt"),
        ("application/vnd.openxmlformats-officedocument.presentationml.presentation", "pptx"),
        ("application/vnd.oasis.opendocument.text", "odt"),
        ("application/vnd.oasis.opendocument.spreadsheet", "ods"),
        ("application/vnd.oasis.opendocument.presentation", "odp"),
        ("application/epub+zip", "epub"),
        ("application/zip", "zip"),
        ("application/gzip", "gz"),
        ("application/x-tar", "tar"),
        ("application/vnd.apache.parquet", "parquet"),
        ("application/octet-stream", "safetensors gguf"),
    )
    for extension in extensions.split()
}

_MEDIA_TYPE_RE = re.compile(r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*")

# Types a browser runs or renders as a document of its own. A client that declares one for a file
# whose extension says otherwise does not get it stored.
_ACTIVE_TYPES = frozenset(
    "text/html application/xhtml+xml image/svg+xml application/pdf text/xml application/xml"
    " text/javascript application/javascript application/x-shockwave-flash".split()
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


def item_type(item_id: str) -> str:
    """The media type an item's bytes are, by the Library's fixed map; "" for an item it has none
    for. LookupError for an upload the account does not have."""
    kind, _, ref = item_id.partition(":")
    if kind in _GALLERIES:
        return _gallery(kind).content_type
    if kind == "upload":
        record = library_db.get_upload(ref)
        if record is None:
            raise LookupError(item_id)
        # Stored as its extension's type when the map knows it, so this is the name's type too.
        return media_type(record["contentType"]) or ""
    return _guess_type(ref) if kind == "sandbox" else ""


# ── Caches ───────────────────────────────────────────────────────


def _account_key() -> str:
    """Which account's stores a cached answer came from: each has a studio.db of its own."""
    from utils.paths import studio_db_path
    return str(studio_db_path())


class _Memo:
    """Answers kept per account and key, each for ``ttl`` seconds while its ``stamp`` agrees, and
    only the ``size`` most recent when that is set. ``forget`` drops them all, and an answer worked
    out across a forget is not kept: it can predate the write that forgot."""

    def __init__(self, size: int = 0):
        self.size = size
        self.generation = 0
        self._entries: OrderedDict[tuple, tuple[float, object, object]] = OrderedDict()
        self._lock = threading.Lock()

    def forget(self) -> None:
        with self._lock:
            self.generation += 1
            self._entries.clear()

    def get(
        self,
        key: tuple,
        compute: Callable,
        ttl: float = float("inf"),
        stamp = None,
    ):
        full, now = (_account_key(), *key), time.monotonic()
        with self._lock:
            generation, hit = self.generation, self._entries.get(full)
            if hit is not None and now - hit[0] < ttl and hit[1] == stamp:
                self._entries.move_to_end(full)
                return hit[2]
        value = compute()
        self.put(key, value, generation, stamp)
        return value

    def put(
        self,
        key: tuple,
        value: object,
        generation: int,
        stamp = None,
    ) -> None:
        with self._lock:
            if generation == self.generation:
                self._entries[(_account_key(), *key)] = (time.monotonic(), stamp, value)
                while self.size and len(self._entries) > self.size:
                    self._entries.popitem(last = False)


# Walking every fine-tune and every chat's sandbox is most of a listing's time, and the page lists
# again after each change: remembered a few seconds per account, a fine-tune's a minute while its
# roots' folders keep their mtimes. The Library's own writes forget them.
_SANDBOX_TTL_SECONDS = 5.0
_MODEL_TTL_SECONDS = 60.0
_LISTING = _Memo()
invalidate_listing = _LISTING.forget


def _model_stamp() -> tuple:
    """The mtimes of the fine-tune roots and the folders in them: a run added, removed or renamed
    changes one."""
    from utils.paths.storage_roots import exports_root, outputs_root

    stamp = []
    for root in (outputs_root(), exports_root()):
        try:
            with os.scandir(root) as entries:
                stamp.append((str(root), os.stat(root).st_mtime_ns))
                stamp.extend(
                    (entry.name, entry.stat(follow_symlinks = False).st_mtime_ns)
                    for entry in entries
                    if entry.is_dir(follow_symlinks = False)
                )
        except OSError:
            stamp.append((str(root), None))
    return tuple(sorted(stamp, key = repr))


def _remembered(
    name: str,
    ttl: float,
    stamp: Optional[Callable] = None,
) -> Callable:
    """The source ``name``, remembered. Found by name, so a test that swaps the source swaps it."""

    def remembered() -> list[dict]:
        items = _LISTING.get((name,), globals()[name], ttl, stamp() if stamp else None)
        # Copies: the overlay is written onto each item.
        return [dict(item) for item in items]

    remembered.__name__ = name
    return remembered


# ── Public API ───────────────────────────────────────────────────

_SOURCES = (
    _upload_items,
    _attachment_items,
    _image_items,
    _video_items,
    _audio_items,
    _remembered("_model_items", _MODEL_TTL_SECONDS, _model_stamp),
    _remembered("_sandbox_items", _SANDBOX_TTL_SECONDS),
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
    adopt: list[tuple[str, str]] = []
    stale: list[tuple[str, str]] = []
    for item in items:
        entry = overlay.get(item["id"])
        fingerprint = item.pop("_fingerprint", None)
        if entry and fingerprint is not None and entry["fingerprint"] != fingerprint:
            if entry["fingerprint"] is None:
                # Written before rows were fingerprinted: this file is taken to be the one.
                adopt.append((item["id"], fingerprint))
            else:
                # Made for a file since deleted; this one only shares its path.
                stale.append((item["id"], entry["fingerprint"]))
                entry = None
        item["favorite"] = bool(entry and entry["favorite"])
        item["folderId"] = entry["folderId"] if entry else None
        if entry and entry["name"]:
            item["name"] = entry["name"]
    try:
        library_db.reconcile_entries(adopt, stale)
    except Exception:
        # Answered as reconciled either way; the next listing tries the write again.
        logger.warning("library.overlay_reconcile_failed", exc_info = True)
    items.sort(key = lambda item: item["updatedAt"], reverse = True)
    return items


def fingerprint(item_id: str) -> Optional[str]:
    """The fingerprint of the file a path-derived id names now, for its overlay row; None for other
    ids (never given to another file) and for a file that is gone."""
    kind, _, ref = item_id.partition(":")
    if kind not in ("sandbox", "model"):
        return None
    try:
        path = _sandbox_path(ref) if kind == "sandbox" else local_path(item_id)
        return _fingerprint(os.stat(path))
    except (LookupError, ValueError, OSError):
        return None


def safe_file_name(
    name: str,
    fallback: str = "file",
    item_id: Optional[str] = None,
) -> str:
    """``name`` as a file name every OS can hold: its last path segment, nothing Windows refuses, no
    leading dot (hidden), no trailing dot or space (Windows drops them), never a device name such as
    ``CON``. With ``item_id``, a hash of it keeps each item's project copy apart, so adding one
    twice is a no-op."""
    stem, ext = os.path.splitext(re.split(r"[\\/]", name or "")[-1])
    if not ext and stem.startswith("."):
        stem, ext = os.path.splitext(stem.lstrip("."))
    stem = _NAME_BREAK_RE.sub(" ", stem).strip(" .") or fallback
    ext = _utf8_prefix(re.sub(f"[{UNSAFE_NAME_CHARS}]", "", ext).rstrip(" ."), 16)
    ext = ext if len(ext) > 1 else ""
    # Cut by UTF-8 bytes, as a file system counts its 255, leaving room for a copy's temp name.
    stem = _utf8_prefix(stem, 160 if item_id else 200).rstrip(" .") or fallback
    # Cleaned, only a device name can still be refused.
    if _bad_name(stem):
        stem = f"_{stem}"
    if item_id:
        return f"{stem}-{hashlib.sha1(item_id.encode()).hexdigest()[:8]}{ext}"
    return f"{stem}{ext}"


def _utf8_prefix(text: str, limit: int) -> str:
    """The longest start of ``text`` at most ``limit`` bytes long in UTF-8, never half a character."""
    return text.encode("utf-8", "ignore")[:limit].decode("utf-8", "ignore")


class ItemFile:
    """An item's file, open. Every consumer reads this one descriptor, so what was checked is what
    is read, and nothing reopens the name."""

    def __init__(self, handle: BinaryIO, name: str, folder: str, project_name: str):
        self.handle = handle
        info = os.fstat(handle.fileno())
        self.size = info.st_size
        self.modified_ns = info.st_mtime_ns
        # What it downloads as, and where and as what Add to project copies it.
        self.name = name
        self.folder = folder
        self.project_name = project_name

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "ItemFile":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _gallery_file(kind: str, ref: str) -> tuple[Path, str]:
    """(path, prompt) of a gallery item Studio made, by the gallery's own ownership test (a file
    with a readable recipe); LookupError otherwise. An image's recipe sits in its PNG, which Pillow
    decodes whole."""
    gallery = _gallery(kind)
    path = gallery.resolve(ref)
    if path is not None:
        meta = gallery.module._read_meta(
            path if kind == "image" else gallery.module._sidecar_path(ref)
        )
    if path is None or meta is None:
        raise LookupError(f"{kind}:{ref}")
    return path, str(meta.get("prompt") or "")


def _open_owned(path: Path, item_id: str) -> BinaryIO:
    try:
        return open(path, "rb")
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        raise LookupError(item_id) from None


def open_item(item_id: str) -> ItemFile:
    """An item's file, opened, with the names it downloads and copies into a project as.

    Raises LookupError when the item is gone and ValueError for items with no file of their own
    (chat attachments live inside messages, fine-tunes are folders)."""
    kind, _, ref = item_id.partition(":")
    if kind == "upload":
        record = library_db.get_upload(ref)
        path = upload_path(ref)
        if record is None or path is None:
            raise LookupError(item_id)
        # Stored under its id alone, so the name the user gave it is the one it downloads as.
        return ItemFile(
            _open_owned(path, item_id),
            safe_file_name(record["name"]),
            "files",
            safe_file_name(record["name"], item_id = item_id),
        )
    if kind in _GALLERIES:
        path, prompt = _gallery_file(kind, ref)
        # Downloads under its prompt (its id with none). A project copy keeps the stored name, as
        # the gallery pages copy it, so either place sees the other's copy as already there.
        name = safe_file_name(_prompt_name(prompt, ref, path.suffix.lstrip(".")), ref)
        return ItemFile(_open_owned(path, item_id), name, _gallery(kind).folder, path.name)
    if kind == "sandbox":
        handle, path = _open_sandbox_file(ref)
        name = os.path.basename(path)
        return ItemFile(
            handle, safe_file_name(name), "files", safe_file_name(name, item_id = item_id)
        )
    raise ValueError("This item cannot be added to a project.")


def local_path(item_id: str) -> Path:
    """The file, or model folder, behind an item, for Reveal in Finder.

    Same errors as ``open_item``. A model path comes from the id, so it must sit inside the
    outputs or exports root."""
    kind, _, ref = item_id.partition(":")
    if kind == "upload":
        path = upload_path(ref)
        if library_db.get_upload(ref) is None or path is None or not path.is_file():
            raise LookupError(item_id)
        return path
    if kind in _GALLERIES:
        return _gallery_file(kind, ref)[0]
    if kind == "sandbox":
        path = Path(_sandbox_path(ref))
        if not path.is_file():
            raise LookupError(item_id)
        return path
    if kind != "model":
        raise ValueError("This item cannot be added to a project.")
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


# ── Thumbnails ───────────────────────────────────────────────────

# Cards are up to a few hundred CSS pixels wide, so twice that for high-density screens.
_THUMBNAIL_WIDTH = 640
# The grid crops a picture past these heights (as a share of its width), so the rest is never sent.
_THUMBNAIL_MIN_RATIO = 2 / 3
_THUMBNAIL_MAX_RATIO = 3 / 2
# A small file can still decode to an enormous bitmap; past this many pixels a card shows its icon.
_THUMBNAIL_MAX_PIXELS = 64_000_000
# Decoders run on whatever a user uploads or a tool writes, so only what a card shows: no
# PostScript (EPS hands the file to Ghostscript), PDF or long tail of legacy image plugins.
_THUMBNAIL_IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF", "AVIF")
# The container a clip is read as, forced rather than probed: a probe can settle on HLS or concat,
# whose playlists name other files, and those would be read too.
_THUMBNAIL_VIDEO_CONTAINERS = {
    "video/mp4": "mp4",
    "video/quicktime": "mov",
    "video/webm": "webm",
    "video/x-matroska": "matroska",
    "video/ogg": "ogg",
    "video/x-msvideo": "avi",
}
# A grid of cards asks for dozens at once; each decode holds a full frame in memory.
_THUMBNAIL_DECODES = threading.BoundedSemaphore(2)
_THUMBNAILS = _Memo(size = 256)


def _attachment_media(ref: str) -> tuple[str, bytes]:
    """The image or clip stored in a chat attachment, with its type. LookupError when it holds none.

    An image is stored as a data URL (``{"type": "image", "image": "data:image/png;base64,..."}``),
    a clip as raw base64 (``{"type": "file", "data", "mimeType"}``), as the attachment route reads
    them."""
    import urllib.parse

    from fastapi import HTTPException

    from routes.chat_history import _decode_attachment_base64
    from storage.studio_db import get_chat_attachment

    message_id, attachment_id = _attachment_ref(ref)
    attachment = get_chat_attachment(message_id, attachment_id) or {}
    try:
        for part in attachment.get("content") or []:
            if not isinstance(part, dict):
                continue
            image = part.get("image")
            if isinstance(image, str) and image[:5].lower() == "data:":
                header, _, payload = image.partition(",")
                mime_type = media_type(header[5:].split(";", 1)[0]) or ""
                if not mime_type.startswith("image/"):
                    continue
                if "base64" not in header.lower():
                    return mime_type, urllib.parse.unquote_to_bytes(payload)
                return mime_type, _decode_attachment_base64(payload)
            if part.get("type") != "file":
                continue
            data = part.get("data")
            mime_type = media_type(part.get("mimeType") or attachment.get("contentType")) or ""
            if isinstance(data, str) and data and mime_type.startswith(("image/", "video/")):
                return mime_type, _decode_attachment_base64(data)
    except HTTPException as exc:
        # A corrupt stored payload: no picture, rather than the attachment route's 422.
        raise RuntimeError("The attachment's data is corrupt.") from exc
    raise LookupError(ref)


def _image_thumbnail(source: Union[Path, BinaryIO]) -> bytes:
    """The picture cropped as a card shows it and at most `_THUMBNAIL_WIDTH` wide, as WebP."""
    try:
        from PIL import Image, ImageOps
    except Exception as exc:  # noqa: BLE001 -- a missing dependency makes thumbnails unavailable
        raise RuntimeError("Thumbnail generation needs the 'Pillow' package.") from exc
    Image.init()
    formats = [name for name in _THUMBNAIL_IMAGE_FORMATS if name in Image.OPEN]
    try:
        with Image.open(source, formats = formats) as opened:
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


def _decode(mime_type: str, source: BinaryIO) -> bytes:
    from core.inference import video_gallery
    with _THUMBNAIL_DECODES:
        if mime_type.startswith("video/") and mime_type in _THUMBNAIL_VIDEO_CONTAINERS:
            return video_gallery.first_frame_webp(
                source,
                width = _THUMBNAIL_WIDTH,
                container = _THUMBNAIL_VIDEO_CONTAINERS[mime_type],
                max_pixels = _THUMBNAIL_MAX_PIXELS,
            )
        if mime_type.startswith("image/") and mime_type != "image/svg+xml":
            return _image_thumbnail(source)
    raise LookupError(mime_type)


def thumbnail(item_id: str) -> bytes:
    """A card's picture: an image cropped and scaled down, or a video's first frame, as WebP.
    Remembered by the file's version, so an edit makes a new one.

    LookupError when the item is gone or has no picture; RuntimeError when it cannot be decoded."""
    kind, _, ref = item_id.partition(":")
    if kind == "attachment":
        mime_type, data = _attachment_media(ref)
        version = (item_id, len(data), zlib.crc32(data))
        return _THUMBNAILS.get(version, lambda: _decode(mime_type, io.BytesIO(data)))
    mime_type = item_type(item_id)
    if not mime_type.startswith(("image/", "video/")):
        raise LookupError(item_id)
    with open_item(item_id) as item:
        version = (item_id, item.size, item.modified_ns)
        return _THUMBNAILS.get(version, lambda: _decode(mime_type, item.handle))


def item_exists(item_id: str, recorded: Optional[str] = None) -> bool:
    """Whether the source still has the item, without listing the source. Errors other than its
    absence propagate, so a store that cannot be read is never taken for an empty one. With the
    fingerprint an overlay row recorded, a new file at the item's path is not the item."""
    kind, _, ref = item_id.partition(":")
    try:
        if kind == "attachment":
            from storage.studio_db import get_chat_attachment
            message_id, attachment_id = _attachment_ref(ref)
            return get_chat_attachment(message_id, attachment_id) is not None
        if kind in _GALLERIES:
            # The file being there is enough: reading its recipe decodes a whole PNG, and the
            # galleries ask this for every star when they open.
            return _gallery(kind).resolve(ref) is not None
        path = local_path(item_id)
    except (LookupError, ValueError):
        return False
    if recorded is None or kind not in ("sandbox", "model"):
        return True
    try:
        return _fingerprint(os.stat(path)) == recorded
    except OSError:
        return False


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
    """Set the file aside before dropping its row, so a failure at either step leaves both."""
    staged = path.with_name(f".{upload_id}.deleting")
    with _upload_lock:
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
        message_id, attachment_id = _attachment_ref(ref)
        deleted = delete_chat_attachment(message_id, attachment_id)
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
    elif kind in _GALLERIES:
        deleted = _gallery(kind).module.delete(ref)
    elif kind == "sandbox":
        deleted = _delete_sandbox_file(ref)
    elif kind == "model":
        # Deleting a model has load and training guards of its own; the model picker owns that.
        raise ValueError("Fine-tuned models are deleted from the model picker.")
    else:
        raise ValueError("Unknown library item")
    invalidate_listing()
    # A gallery keeps a file it could not unlink (open in another app on Windows) listed, so the
    # delete can be tried again: its name, star and folder stay with it.
    if not deleted and kind in _GALLERIES and item_exists(item_id):
        raise DeleteIncomplete("Could not delete the file. Close any app using it and try again.")
    # Gone either way, so its name, star and folder go too, or they would sit in the overlay and
    # be counted among the favorites for good.
    library_db.delete_entry(item_id)
    return deleted
