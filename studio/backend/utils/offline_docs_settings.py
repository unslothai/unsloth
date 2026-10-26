# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A local copy of the Unsloth docs (``llms-full.txt``) for offline RAG.

The owner picks a folder in Settings; the full docs are written there as one dated Markdown file
and, when auto-update is on, refreshed the first time a new Studio release starts. Linking the
folder to a knowledge base keeps the index current through the normal folder sync. Off by default:
startup never performs this network work unless the owner opted in.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

OFFLINE_DOCS_DIR_SETTING_KEY = "offline_docs_dir"
OFFLINE_DOCS_AUTO_UPDATE_SETTING_KEY = "offline_docs_auto_update"
# One record, so a reader never pairs one refresh's file with another's release.
OFFLINE_DOCS_STATE_SETTING_KEY = "offline_docs_state"
DEFAULT_OFFLINE_DOCS_AUTO_UPDATE = False
MAX_OFFLINE_DOCS_PATH_LENGTH = 4096

DOCS_URL = "https://unsloth.ai/docs/llms-full.txt"
# The docs are paginated; every part but the last ends with a link to the next one.
_NEXT_PAGE_RE = re.compile(r"\[Next Page\]\((https://unsloth\.ai/docs/llms-full\.txt/\d+)\)")
MAX_DOCS_PARTS = 32
MAX_DOCS_BYTES = 64 * 1024 * 1024
DOCS_TIMEOUT_S = 60.0

DOCS_FILE_PREFIX = "Unsloth Docs LLMs Full Text [Documentation]"
_DOCS_FILE_RE = re.compile(
    r"^Unsloth Docs LLMs Full Text \[Documentation\] \(\d{4}-\d{2}-\d{2}\)\.md$"
)

logger = logging.getLogger(__name__)
_refresh_lock = threading.Lock()


class OfflineDocsBusyError(RuntimeError):
    """A refresh is already running."""


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return None


def _read(key: str, fallback: Any = None) -> Any:
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as
        return run_as(OWNER, get_app_setting, key, fallback)
    except Exception:
        return fallback


def _write(values: dict[str, Any]) -> None:
    from storage.studio_db import upsert_app_settings
    from utils.account_context import OWNER, run_as
    run_as(OWNER, upsert_app_settings, values)


def docs_file_name(day: date) -> str:
    return f"{DOCS_FILE_PREFIX} ({day.isoformat()}).md"


def get_offline_docs_dir() -> Optional[Path]:
    value = _read(OFFLINE_DOCS_DIR_SETTING_KEY)
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > MAX_OFFLINE_DOCS_PATH_LENGTH:
        return None
    return Path(os.path.expanduser(value))


def get_offline_docs_auto_update() -> bool:
    parsed = _coerce_bool(_read(OFFLINE_DOCS_AUTO_UPDATE_SETTING_KEY))
    return parsed if parsed is not None else DEFAULT_OFFLINE_DOCS_AUTO_UPDATE


def _get_state() -> dict[str, Any]:
    state = _read(OFFLINE_DOCS_STATE_SETTING_KEY)
    return dict(state) if isinstance(state, dict) else {}


def _canonical_directory(value: str) -> Path:
    raw = value.strip()
    if not raw:
        raise ValueError("Choose a folder for the offline docs.")
    if len(raw) > MAX_OFFLINE_DOCS_PATH_LENGTH:
        raise ValueError("The offline docs folder path is too long.")
    try:
        directory = Path(os.path.expanduser(raw)).resolve(strict = True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("The offline docs folder does not exist or cannot be accessed.") from exc
    if not directory.is_dir():
        raise ValueError("The offline docs location must be a folder.")
    if not os.access(directory, os.W_OK):
        raise ValueError("Unsloth cannot write to that folder.")
    return directory


def set_offline_docs_settings(*, path: Any = ..., auto_update: Any = ...) -> None:
    """Persist the folder and/or the auto-update flag; an omitted argument is left unchanged.

    ``path=None`` forgets the folder. Files already written there are left alone.
    """
    values: dict[str, Any] = {}
    if path is not ...:
        values[OFFLINE_DOCS_DIR_SETTING_KEY] = (
            str(_canonical_directory(path)) if path is not None else None
        )
    if auto_update is not ...:
        parsed = _coerce_bool(auto_update)
        if parsed is None:
            raise ValueError("Auto-update for the offline docs must be true or false.")
        values[OFFLINE_DOCS_AUTO_UPDATE_SETTING_KEY] = parsed
    if values:
        _write(values)


def _current_release() -> str:
    try:
        from utils.studio_version import get_studio_version
        return get_studio_version()
    except Exception:
        return "dev"


def offline_docs_status() -> dict[str, Any]:
    directory = get_offline_docs_dir()
    state = _get_state()
    file_name = state.get("file") if isinstance(state.get("file"), str) else None
    stored_dir = state.get("dir") if isinstance(state.get("dir"), str) else None
    file_path = None
    # A copy written to a folder the owner has since moved away from is not this folder's copy.
    if directory is not None and file_name and stored_dir == str(directory):
        candidate = directory / file_name
        if candidate.is_file():
            file_path = str(candidate)
    return {
        "path": str(directory) if directory is not None else None,
        "auto_update": get_offline_docs_auto_update(),
        "file": file_path,
        "updated_at": state.get("updated_at") if file_path else None,
        "release": state.get("release") if file_path else None,
        "current_release": _current_release(),
        "error": state.get("error") if isinstance(state.get("error"), str) else None,
        "refreshing": _refresh_lock.locked(),
    }


def fetch_docs_text(client: Any = None) -> str:
    """Every part of ``llms-full.txt``, in order, joined into one document."""
    import httpx

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout = DOCS_TIMEOUT_S, follow_redirects = True)
    try:
        parts: list[str] = []
        seen: set[str] = set()
        total = 0
        url: Optional[str] = DOCS_URL
        while url is not None:
            if url in seen or len(parts) >= MAX_DOCS_PARTS:
                break
            seen.add(url)
            response = client.get(url)
            response.raise_for_status()
            text = response.text
            total += len(response.content)
            if total > MAX_DOCS_BYTES:
                raise ValueError("The Unsloth docs are larger than expected; not saving them.")
            parts.append(text)
            links = _NEXT_PAGE_RE.findall(text[-2048:])
            url = links[-1] if links else None
    finally:
        if own_client:
            client.close()
    document = "\n\n".join(part.strip("\n") for part in parts).strip()
    if not document:
        raise ValueError("The Unsloth docs download was empty.")
    return document + "\n"


def _write_docs(directory: Path, name: str, text: str) -> Path:
    target = directory / name
    fd, temp_name = tempfile.mkstemp(prefix = ".unsloth-docs-", suffix = ".tmp", dir = directory)
    try:
        with os.fdopen(fd, "w", encoding = "utf-8", newline = "\n") as handle:
            handle.write(text)
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    # Older dated copies would be indexed next to the new one in a linked knowledge base.
    for entry in directory.iterdir():
        if entry.name != name and _DOCS_FILE_RE.match(entry.name) and entry.is_file():
            try:
                entry.unlink()
            except OSError:
                pass
    return target


def refresh_offline_docs(
    *, fetch: Callable[[], str] = fetch_docs_text, today: Callable[[], date] = date.today
) -> dict[str, Any]:
    """Download the docs into the configured folder now and return the new status."""
    if not _refresh_lock.acquire(blocking = False):
        raise OfflineDocsBusyError("The offline docs are already being updated.")
    try:
        directory = get_offline_docs_dir()
        if directory is None:
            raise ValueError("Choose a folder for the offline docs first.")
        state = _get_state()
        try:
            if not directory.is_dir():
                raise ValueError(
                    "The offline docs folder no longer exists. Choose it again in Settings."
                )
            text = fetch()
            target = _write_docs(directory, docs_file_name(today()), text)
        except Exception as exc:
            state["error"] = _error_text(exc)
            _write({OFFLINE_DOCS_STATE_SETTING_KEY: state})
            raise
        _write(
            {
                OFFLINE_DOCS_STATE_SETTING_KEY: {
                    "dir": str(directory),
                    "file": target.name,
                    "release": _current_release(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "error": None,
                }
            }
        )
    finally:
        _refresh_lock.release()
    return offline_docs_status()


def _error_text(exc: Exception) -> str:
    try:
        import httpx
        if isinstance(exc, httpx.HTTPStatusError):
            return f"The docs server answered {exc.response.status_code}."
        if isinstance(exc, httpx.HTTPError):
            return "Could not reach unsloth.ai to download the docs."
    except ImportError:
        pass
    if isinstance(exc, OSError):
        return f"Could not write the docs file: {exc.strerror or exc}"
    return str(exc) or exc.__class__.__name__


def offline_docs_refresh_due() -> bool:
    """Auto-update is on, a folder is set, and this release has no copy there yet."""
    if not get_offline_docs_auto_update():
        return False
    directory = get_offline_docs_dir()
    if directory is None:
        return False
    state = _get_state()
    if state.get("dir") != str(directory) or not state.get("file"):
        return True
    if not (directory / str(state["file"])).is_file():
        return True
    return state.get("release") != _current_release()


def refresh_offline_docs_if_due() -> bool:
    """Startup hook: refresh once per new release. A failure is recorded and retried next start."""
    try:
        if not offline_docs_refresh_due():
            return False
        refresh_offline_docs()
        logger.info("Offline Unsloth docs updated for release %s", _current_release())
        return True
    except Exception as exc:
        logger.warning("Offline Unsloth docs update failed: %s", _error_text(exc))
        return False
