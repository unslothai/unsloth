# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owner-chosen folder holding a dated copy of the Unsloth docs for offline RAG, refreshed once per release when opted in."""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

DIR_KEY = "offline_docs_dir"
AUTO_UPDATE_KEY = "offline_docs_auto_update"
STATE_KEY = "offline_docs_state"  # one record, so file / build / error never mix across refreshes
MAX_PATH_LENGTH = 4096

DOCS_URL = "https://unsloth.ai/docs/llms-full.txt"
_NEXT_PAGE_RE = re.compile(r"\[Next Page\]\((https://unsloth\.ai/docs/llms-full\.txt/\d+)\)")
MAX_DOCS_PARTS = 32
MAX_DOCS_BYTES = 64 * 1024 * 1024
DOCS_TIMEOUT_S = 60.0
_DOCS_FILE_RE = re.compile(
    r"^Unsloth Docs LLMs Full Text \[Documentation\] \(\d{4}-\d{2}-\d{2}\)\.md$"
)

logger = logging.getLogger(__name__)
_refresh_lock = threading.Lock()


class OfflineDocsBusyError(RuntimeError):
    pass


def _read(key: str) -> Any:
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as
        return run_as(OWNER, get_app_setting, key, None)
    except Exception:
        return None


def _write(values: dict[str, Any]) -> None:
    from storage.studio_db import upsert_app_settings
    from utils.account_context import OWNER, run_as
    run_as(OWNER, upsert_app_settings, values)


def _state() -> dict[str, Any]:
    state = _read(STATE_KEY)
    return dict(state) if isinstance(state, dict) else {}


def docs_file_name(day: date) -> str:
    return f"Unsloth Docs LLMs Full Text [Documentation] ({day.isoformat()}).md"


def get_offline_docs_dir() -> Optional[Path]:
    value = _read(DIR_KEY)
    return Path(value) if isinstance(value, str) and value else None


def get_offline_docs_auto_update() -> bool:
    return _read(AUTO_UPDATE_KEY) is True


def _canonical_directory(value: str) -> Path:
    raw = value.strip()
    if not raw or len(raw) > MAX_PATH_LENGTH:
        raise ValueError("Choose a folder for the offline docs.")
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
    """Omitted arguments stay unchanged; ``path=None`` forgets the folder (its files are kept)."""
    values: dict[str, Any] = {}
    if path is not ...:
        values[DIR_KEY] = str(_canonical_directory(path)) if path is not None else None
    if auto_update is not ...:
        if not isinstance(auto_update, bool):
            raise ValueError("Auto-update for the offline docs must be true or false.")
        values[AUTO_UPDATE_KEY] = auto_update
    if values:
        _write(values)


def _current_release() -> str:
    try:
        from utils.studio_version import get_studio_version
        return get_studio_version()
    except Exception:
        return "dev"


def _package_version() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version("unsloth")
    except Exception:
        return None


def _current_build() -> str:
    """Refresh key: Docker images and source checkouts report only ``dev`` or a branch, so add the package version."""
    release = _current_release()
    try:
        from utils.studio_version import is_valid_studio_release_version
        if is_valid_studio_release_version(release):
            return release
    except Exception:
        pass
    package = _package_version()
    return f"{release} {package}" if package else release


def offline_docs_status() -> dict[str, Any]:
    directory = get_offline_docs_dir()
    state = _state()
    file_path = None
    # A copy left in a previously chosen folder is not this folder's copy.
    if directory is not None and state.get("file") and state.get("dir") == str(directory):
        candidate = directory / str(state["file"])
        try:
            if candidate.is_file():
                file_path = str(candidate)
        except (
            OSError
        ):  # is_file() re-raises EACCES; the rows must still load so the folder can be changed
            pass
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
    """Every part of ``llms-full.txt``, following each part's trailing Next Page link, joined in order."""
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
            # A partial copy would replace the complete one and suppress retries.
            if url in seen or len(parts) >= MAX_DOCS_PARTS:
                raise ValueError("The Unsloth docs pagination did not end; not saving them.")
            seen.add(url)
            response = client.get(url)
            response.raise_for_status()
            total += len(response.content)
            if total > MAX_DOCS_BYTES:
                raise ValueError("The Unsloth docs are larger than expected; not saving them.")
            parts.append(response.text)
            links = _NEXT_PAGE_RE.findall(response.text[-2048:])
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
    # A linked knowledge base would index older dated copies next to the new one.
    for entry in directory.iterdir():
        if entry.name != name and _DOCS_FILE_RE.match(entry.name) and entry.is_file():
            try:
                entry.unlink()
            except OSError:
                pass
    return target


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


def refresh_offline_docs(
    *, fetch: Callable[[], str] = fetch_docs_text, today: Callable[[], date] = date.today
) -> dict[str, Any]:
    """Download the docs into the configured folder now; a failure keeps the previous copy and records the error."""
    if not _refresh_lock.acquire(blocking = False):
        raise OfflineDocsBusyError("The offline docs are already being updated.")
    try:
        directory = get_offline_docs_dir()
        if directory is None:
            raise ValueError("Choose a folder for the offline docs first.")
        state = _state()
        try:
            if not directory.is_dir():
                raise ValueError(
                    "The offline docs folder no longer exists. Choose it again in Settings."
                )
            target = _write_docs(directory, docs_file_name(today()), fetch())
        except Exception as exc:
            _write({STATE_KEY: {**state, "error": _error_text(exc)}})
            raise
        _write(
            {
                STATE_KEY: {
                    "dir": str(directory),
                    "file": target.name,
                    "release": _current_release(),
                    "build": _current_build(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "error": None,
                }
            }
        )
    finally:
        _refresh_lock.release()
    return offline_docs_status()


def offline_docs_refresh_due() -> bool:
    directory = get_offline_docs_dir()
    if directory is None or not get_offline_docs_auto_update():
        return False
    state = _state()
    if state.get("dir") != str(directory) or not state.get("file"):
        return True
    if not (directory / str(state["file"])).is_file():
        return True
    return state.get("build") != _current_build()


def refresh_offline_docs_if_due() -> bool:
    """Startup hook; the build is recorded only on success, so a failure retries on the next start."""
    try:
        if not offline_docs_refresh_due():
            return False
        refresh_offline_docs()
        logger.info("Offline Unsloth docs updated for %s", _current_build())
        return True
    except Exception as exc:
        logger.warning("Offline Unsloth docs update failed: %s", _error_text(exc))
        return False
