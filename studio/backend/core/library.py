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

import os
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from loggers import get_logger
from storage import library_db
from utils.paths import ensure_dir
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
    from utils.paths.relocations import relocated

    # Settings > Library can move the owner's folder elsewhere; other accounts keep theirs.
    return ensure_dir(relocated("uploads", account_path("library")))


def _device(path) -> Optional[int]:
    """The filesystem holding ``path``, or its nearest existing parent."""
    path = Path(path)
    for candidate in (path, *path.parents):
        try:
            return os.stat(candidate).st_dev
        except OSError:
            continue
    return None


def _gallery_root(name: str) -> list:
    import importlib
    return [importlib.import_module(f"core.inference.{name}").gallery_dir()]


def _sandbox_root() -> list:
    from core.inference.tools import sandbox_root
    return [sandbox_root()]


def _attachment_root() -> list:
    from utils.paths.storage_roots import studio_db_path
    return [studio_db_path().parent]


def _training_root() -> list:
    from utils.paths.storage_roots import outputs_root
    return [outputs_root()]


def _export_root() -> list:
    from utils.paths.storage_roots import exports_root
    return [exports_root()]


# Where each source keeps its bytes. Any of them can be configured onto another disk. Fine-tunes
# and exports are told apart (`model:<origin>`, the start of their ids): either can be elsewhere.
_SOURCE_ROOTS = {
    "upload": lambda: [uploads_dir()],
    "attachment": _attachment_root,
    "image": lambda: _gallery_root("image_gallery"),
    "video": lambda: _gallery_root("video_gallery"),
    "audio": lambda: _gallery_root("audio_gallery"),
    "model:training": _training_root,
    "model:exported": _export_root,
    "sandbox": _sandbox_root,
}


def disk_usage() -> Optional[dict]:
    """Capacity of the disk the Library's own files live on, which need not be the system disk,
    and the sources stored on it, so the bar only counts bytes that disk actually holds. A source
    that cannot say where it lives is left out rather than failing the listing."""
    try:
        usage = shutil.disk_usage(uploads_dir())
        device = _device(uploads_dir())
    except OSError:
        return None
    sources = []
    for source, roots in _SOURCE_ROOTS.items():
        try:
            if all(_device(root) == device for root in roots()):
                sources.append(source)
        except Exception:
            logger.warning("library.source_root_failed: %s", source, exc_info = True)
    return {"totalBytes": usage.total, "freeBytes": usage.free, "sources": sources}


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
    import mimetypes

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
    content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
    return name, content_type, open(grant.canonical_path, "rb")


def write_upload_text(upload_id: str, text: str) -> bool:
    path = upload_path(upload_id)
    if path is None or library_db.get_upload(upload_id) is None:
        return False
    data = text.encode("utf-8")
    # Staged and swapped in, so a failed write leaves the previous note whole.
    tmp_path = path.with_name(f".{upload_id}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_bytes(data)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok = True)
        raise
    library_db.touch_upload(upload_id, len(data))
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
        # A GGUF export is listed by one of its files, but every quantization beside it is on disk.
        stats_path = Path(path)
        if model_type == "gguf" and stats_path.is_file():
            stats_path = stats_path.parent
        try:
            size, modified = _tree_stats(stats_path)
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
        # A project pointed at a folder of the user's own works in that folder; its files are not Studio's.
        projects = conn.execute(
            "SELECT id, name FROM chat_projects WHERE root_path IS NULL OR root_path = ''"
        ).fetchall()
    finally:
        conn.close()
    sessions: list[tuple[str, Optional[str], Optional[str]]] = [
        (row["id"], row["id"], row["title"]) for row in threads
    ]
    sessions.extend(
        (f"{_PROJECT_SESSION_PREFIX}{row['id']}", None, row["name"]) for row in projects
    )
    return sessions


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


def _guess_type(name: str) -> str:
    import mimetypes
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


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
        item["openedAt"] = entry["openedAt"] if entry else None
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
    if not any(resolved.startswith(root + os.sep) for root in roots) or not os.path.exists(
        resolved
    ):
        raise LookupError(item_id)
    return Path(resolved)


# Cards are up to a few hundred CSS pixels wide, so twice that for high-density screens.
_VIDEO_THUMBNAIL_WIDTH = 640


def _attachment_clip(ref: str) -> bytes:
    """The clip stored in a chat attachment. LookupError when it holds none."""
    from routes.chat_history import _decode_attachment_base64
    from storage.studio_db import get_chat_attachment

    message_id, _, attachment_id = ref.partition(":")
    attachment = get_chat_attachment(message_id, attachment_id) or {}
    for part in attachment.get("content") or []:
        if not isinstance(part, dict) or part.get("type") != "file":
            continue
        data = part.get("data")
        mime_type = str(part.get("mimeType") or attachment.get("contentType") or "").lower()
        if isinstance(data, str) and data and mime_type.startswith("video/"):
            return _decode_attachment_base64(data)
    raise LookupError(ref)


def video_thumbnail(item_id: str) -> bytes:
    """The first frame of a video item, as WebP, for its card.

    LookupError when the item is gone or is not a video; RuntimeError when it cannot be decoded."""
    import io

    from core.inference import video_gallery

    kind, _, ref = item_id.partition(":")
    if kind == "attachment":
        return video_gallery.first_frame_webp(
            io.BytesIO(_attachment_clip(ref)), width = _VIDEO_THUMBNAIL_WIDTH
        )
    if kind not in ("upload", "video", "sandbox"):
        raise LookupError(item_id)
    path = project_source(item_id)[0]
    if kind == "upload":
        record = library_db.get_upload(ref) or {}
        types = (str(record.get("contentType") or ""), _guess_type(str(record.get("name") or "")))
    else:
        types = ("video/" if kind == "video" else _guess_type(path.name),)
    if not any(value.lower().startswith("video/") for value in types):
        raise LookupError(item_id)
    return video_gallery.first_frame_webp(path, width = _VIDEO_THUMBNAIL_WIDTH)


def _location_resolvers() -> dict:
    from core.inference import audio_gallery, image_gallery, video_gallery
    from utils.paths.storage_roots import exports_root, outputs_root
    return {
        "uploads": uploads_dir,
        "images": image_gallery.gallery_dir,
        "videos": video_gallery.gallery_dir,
        "audio": audio_gallery.gallery_dir,
        "fineTunes": outputs_root,
        "exports": exports_root,
    }


def locations() -> list[dict]:
    """Where each kind of Library file lives, for Settings > Library. `movable` kinds can be moved
    with ``move_location``; `custom` says the owner already has."""
    from utils.paths.relocations import MOVABLE, chosen
    return [
        {
            "key": key,
            "path": str(resolve()),
            "movable": key in MOVABLE,
            "custom": chosen(key) is not None,
        }
        for key, resolve in _location_resolvers().items()
    ]


# Left behind by a move and ignored in an "empty" target: the OS writes them into any folder it shows.
_OS_CLUTTER = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})
_move_lock = threading.Lock()


def _location_default(key: str) -> Path:
    from utils.paths.storage_roots import studio_root
    return account_path("library") if key == "uploads" else studio_root() / key


def _move_target(raw: str) -> Path:
    """An absolute, ordinary folder whose parent exists. Same rules as the model download folder."""
    from hub.storage.scan_folders import (
        contains_sensitive_path_component,
        is_denied_system_path,
    )

    value = raw.strip()
    if not value:
        raise ValueError("Choose a folder.")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise ValueError("Choose an absolute folder path.")
    try:
        resolved = candidate.resolve(strict = False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("That folder path is invalid.") from exc
    if resolved.parent == resolved:
        raise ValueError("Choose a folder inside the drive, not the drive itself.")
    if is_denied_system_path(str(resolved)) or contains_sensitive_path_component(str(resolved)):
        raise ValueError("System, credential and config folders cannot hold these files.")
    if not resolved.parent.is_dir():
        raise ValueError("The parent folder does not exist.")
    return resolved


# A picked folder that already holds files gets one of these made inside it instead.
_SUBFOLDERS = {
    "uploads": "Unsloth Library",
    "images": "Unsloth Images",
    "videos": "Unsloth Videos",
    "audio": "Unsloth Audio",
}


def _is_empty(folder: Path) -> bool:
    return not any(entry.name not in _OS_CLUTTER for entry in folder.iterdir())


def _prepare_target(target: Path, key: str) -> Path:
    """The folder the files go to: `target` itself when empty, else a named folder made inside it.
    Created if needed and checked writable."""
    import tempfile

    try:
        target.mkdir(exist_ok = True)
        if not target.is_dir():
            raise ValueError("That path is a file, not a folder.")
        if not _is_empty(target):
            target = target / _SUBFOLDERS[key]
            target.mkdir(exist_ok = True)
            if not target.is_dir() or not _is_empty(target):
                raise ValueError(f"{target} already holds files. Choose another folder.")
        with tempfile.NamedTemporaryFile(prefix = ".unsloth-write-test-", dir = target):
            pass
    except PermissionError as exc:
        raise ValueError("Unsloth cannot write to that folder.") from exc
    except OSError as exc:
        raise ValueError(f"Unsloth cannot use that folder: {exc.strerror or exc}") from exc
    return target


def _move_entries(source: Path, target: Path, moved: list[tuple[Path, Path]]) -> None:
    """Move everything in `source` into `target`, across drives too, noting each move in `moved`
    as it lands so a failure part way can be undone."""
    for entry in list(source.iterdir()):
        if entry.name in _OS_CLUTTER:
            continue
        dest = target / entry.name
        shutil.move(str(entry), str(dest))
        moved.append((entry, dest))


def _refuse_overlap(target: Path, resolvers) -> None:
    from core.inference.tools import sandbox_root

    # Chat sandboxes too: their listing would show the files as tool output, and clearing the chat
    # with its files would delete them.
    roots = [Path(resolve()) for resolve in resolvers.values()] + [Path(sandbox_root())]
    for root in roots:
        root = root.resolve()
        if target == root or root in target.parents:
            raise ValueError("That folder is inside another Unsloth folder.")


def move_location(key: str, path: Optional[str]) -> None:
    """Move one kind of file to another folder, files and all, and keep saving there. `path` None
    moves it back to the default. Owner only; the caller checks.

    The switch is recorded first, so a file saved during the move already lands in the new folder;
    a second pass then picks up any save that was writing to the old one. On a failure everything
    moved so far goes back and the old folder stays in use. Raises ValueError for a folder that
    cannot be used, RuntimeError when the move itself fails."""
    from utils.paths.relocations import MOVABLE, chosen, set_chosen

    if key not in MOVABLE:
        raise ValueError(
            "These files stay where they are: training and chats remember them by path."
        )
    resolvers = _location_resolvers()
    with _move_lock:
        current = resolvers[key]().resolve()
        target = _move_target(path) if path is not None else _location_default(key).resolve()
        if target == current:
            return
        _refuse_overlap(target, resolvers)
        target = _prepare_target(target, key)
        if target == current:
            return
        # Again for the named subfolder, which can be another kind's folder.
        _refuse_overlap(target, resolvers)
        previous = chosen(key)
        # A default already holding files gets a subfolder, which has to be recorded to be used.
        set_chosen(key, None if target == _location_default(key).resolve() else target)
        moved: list[tuple[Path, Path]] = []
        try:
            _move_entries(current, target, moved)
        except OSError as exc:
            for source, dest in reversed(moved):
                try:
                    shutil.move(str(dest), str(source))
                except OSError:
                    logger.error("library.move_rollback_failed: %s", dest, exc_info = True)
            set_chosen(key, previous)
            raise RuntimeError(f"Could not move the files: {exc.strerror or exc}") from exc
        try:
            _move_entries(current, target, [])
        except OSError:
            logger.warning("library.move_straggler_failed: %s", current, exc_info = True)


def _delete_upload(upload_id: str, path: Path) -> bool:
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

        deleted = video_gallery.delete(ref)
        if deleted:
            # Same cleanup as the Video page, so the clip does not come back as a ghost card.
            _forget_terminal_video(ref)
            if not _forget_openai_job(ref):
                logger.warning("library.delete_video_job_failed: %s", ref)
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
        # The models route deletes the files, with its own load and training guards. Once they are
        # gone, this drops what the Library kept about the model.
        _origin, _, path = ref.partition(":")
        if not path or os.path.exists(path):
            raise ValueError("Fine-tuned models are deleted from the model picker.")
        deleted = True
    else:
        raise ValueError("Unknown library item")
    if deleted:
        library_db.delete_entry(item_id)
    return deleted
