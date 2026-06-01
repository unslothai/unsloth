# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model folder recommendation and browsing services."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from loggers import get_logger

from hub.schemas.inventory import BrowseEntry, BrowseFoldersResponse
from hub.storage.scan_folders import (
    contains_sensitive_path_component,
    list_scan_folders,
)
from hub.utils.paths import (
    exports_root,
    hf_default_cache_dir,
    legacy_hf_cache_dir,
    lmstudio_model_dirs,
    normalize_path,
    outputs_root,
    studio_root,
    well_known_model_dirs,
)
from hub.services.models.common import _safe_is_dir
from hub.services.models.local_inventory import _resolve_hf_cache_dir

logger = get_logger(__name__)


def get_recommended_folders_response() -> dict:
    """Return well-known model directories that exist on this machine."""
    folders: list[str] = []
    seen: set[str] = set()

    def _add(p: Optional[Path]) -> None:
        if p is None:
            return
        try:
            resolved = str(p.resolve())
        except OSError:
            return
        if resolved in seen:
            return
        if _safe_is_dir(resolved) and os.access(resolved, os.R_OK | os.X_OK):
            seen.add(resolved)
            folders.append(resolved)

    try:
        for p in lmstudio_model_dirs():
            _add(p)
    except Exception as e:
        logger.warning("Failed to scan for LM Studio model directories: %s", e)

    ollama_env = os.environ.get("OLLAMA_MODELS")
    if ollama_env:
        _add(Path(normalize_path(ollama_env)).expanduser())
    for candidate in (
        Path.home() / ".ollama" / "models",
        Path("/usr/share/ollama/.ollama/models"),
        Path("/var/lib/ollama/.ollama/models"),
    ):
        _add(candidate)

    return {"folders": folders}


# Heuristic ceiling on how many children to stat when checking whether a
# directory "looks like" it contains models. Keeps the browser snappy
# even when a directory has thousands of unrelated entries.
_BROWSE_MODEL_HINT_PROBE = 64
# Hard cap on how many subdirectory entries we send back. Pointing the
# browser at something like ``/usr/lib`` or ``/proc`` must not stat-storm
# the process or send tens of thousands of rows to the client.
_BROWSE_ENTRY_CAP = 2000


def _count_model_files(directory: Path, cap: int = 200) -> int:
    """Count GGUF/safetensors files immediately inside *directory*.

    Bounded by *visited entries*, not by *match count*: in directories
    with many non-model files (or many subdirectories) the scan still
    stops after ``cap`` entries so a UI hint never costs more than a
    bounded directory walk.
    """
    n = 0
    visited = 0
    try:
        for f in directory.iterdir():
            visited += 1
            if visited > cap:
                break
            try:
                if f.is_file():
                    low = f.name.lower()
                    if low.endswith((".gguf", ".safetensors")):
                        n += 1
            except OSError:
                continue
    except PermissionError as e:
        logger.debug("browse-folders: permission denied counting %s: %s", directory, e)
        return 0
    except OSError as e:
        logger.debug("browse-folders: OS error counting %s: %s", directory, e)
        return 0
    return n


def _has_direct_model_signal(directory: Path) -> bool:
    """Return True if *directory* has an immediate child that signals
    it holds a model: a GGUF/safetensors/config.json file, or a
    `models--*` subdir (HF hub cache). Bounded by
    ``_BROWSE_MODEL_HINT_PROBE`` to stay fast."""
    try:
        it = directory.iterdir()
    except OSError:
        return False
    try:
        for i, child in enumerate(it):
            if i >= _BROWSE_MODEL_HINT_PROBE:
                break
            try:
                name = child.name
                if child.is_file():
                    low = name.lower()
                    if low.endswith((".gguf", ".safetensors")):
                        return True
                    if low in ("config.json", "adapter_config.json"):
                        return True
                elif child.is_dir() and name.startswith("models--"):
                    return True
            except OSError:
                continue
    except OSError:
        return False
    return False


def _looks_like_model_dir(directory: Path) -> bool:
    """Bounded heuristic used by the folder browser to flag directories
    worth exploring. False negatives are fine; the real scanner is
    authoritative.

    Three signals, cheapest first:

    1. Directory name itself: ``models--*`` is the HuggingFace hub cache
       layout (``blobs``/``refs``/``snapshots`` children wouldn't match
       the file-level probes below).
    2. An immediate child is a weight file or config (handled by
       :func:`_has_direct_model_signal`).
    3. A grandchild has a direct signal -- this catches the
       ``publisher/model/weights.gguf`` layout used by LM Studio and
       Ollama. We probe at most the first
       ``_BROWSE_MODEL_HINT_PROBE`` child directories, each of which is
       checked with a bounded :func:`_has_direct_model_signal` call,
       so the total cost stays O(PROBE^2) worst-case.
    """
    if directory.name.startswith("models--"):
        return True
    if _has_direct_model_signal(directory):
        return True
    # Grandchild probe: LM Studio / Ollama publisher/model layout.
    try:
        it = directory.iterdir()
    except OSError:
        return False
    try:
        for i, child in enumerate(it):
            if i >= _BROWSE_MODEL_HINT_PROBE:
                break
            try:
                if not child.is_dir():
                    continue
            except OSError:
                continue
            if child.name.startswith("models--"):
                return True
            if _has_direct_model_signal(child):
                return True
    except OSError:
        return False
    return False


def _build_browse_allowlist() -> list[Path]:
    """Return the list of root directories the folder browser is allowed
    to walk. The same list is used to seed the sidebar suggestion chips,
    so chip targets are always reachable.

    Roots include the current user's HOME, the resolved HF cache dirs,
    Studio's own outputs/exports/studio root, registered scan folders,
    and well-known third-party local-LLM dirs (LM Studio, Ollama,
    `~/models`). Each is added only if it currently resolves to a real
    directory, so we never produce a "dead" sandbox boundary the user
    can't navigate into.
    """
    from hub.storage.scan_folders import list_scan_folders

    candidates: list[Path] = []

    def _add(p: Optional[Path | str]) -> None:
        if p is None:
            return
        try:
            p = Path(normalize_path(str(p))).expanduser()
            resolved = p.resolve()
        except (OSError, RuntimeError, ValueError):
            return
        if _safe_is_dir(resolved):
            candidates.append(resolved)

    _add(Path.home())
    _add(_resolve_hf_cache_dir())
    try:
        _add(hf_default_cache_dir())
    except Exception:  # noqa: BLE001 -- best-effort
        pass
    try:
        _add(legacy_hf_cache_dir())
    except Exception:  # noqa: BLE001 -- best-effort
        pass
    try:
        _add(studio_root())
        _add(outputs_root())
        _add(exports_root())
    except Exception as exc:  # noqa: BLE001 -- best-effort
        logger.debug("browse-folders: studio roots unavailable: %s", exc)
    try:
        for folder in list_scan_folders():
            p = folder.get("path")
            if p:
                _add(p)
    except Exception as exc:  # noqa: BLE001 -- best-effort
        logger.debug("browse-folders: could not load scan folders: %s", exc)
    try:
        for p in well_known_model_dirs():
            _add(p)
    except Exception as exc:  # noqa: BLE001 -- best-effort
        logger.debug("browse-folders: well-known dirs unavailable: %s", exc)

    # Dedupe while preserving order.
    seen: set[str] = set()
    deduped: list[Path] = []
    for p in candidates:
        key = os.path.normcase(os.path.realpath(str(p)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def _is_path_inside_allowlist(target: Path, allowed_roots: list[Path]) -> bool:
    """Return True if *target* equals or is a descendant of any allowed
    root. The comparison uses ``os.path.realpath`` so symlinks cannot be
    used to escape the sandbox.
    """
    try:
        target_real = os.path.normcase(os.path.realpath(str(target)))
    except OSError:
        return False
    for root in allowed_roots:
        try:
            root_real = os.path.normcase(os.path.realpath(str(root)))
        except OSError:
            continue
        try:
            if os.path.commonpath([target_real, root_real]) == root_real:
                return True
        except ValueError:
            continue
        if target_real == root_real:
            return True
    return False


def _normalize_browse_request_path(path: Optional[str], *, relative_root: Path) -> str:
    """Normalize the browse request path lexically, without touching the FS."""
    if path is None or not path.strip():
        return os.path.normpath(str(Path.home()))

    expanded = os.path.expanduser(normalize_path(path.strip()))
    if not os.path.isabs(expanded):
        expanded = os.path.join(str(relative_root), expanded)
    return os.path.normpath(expanded)


def _browse_relative_parts(requested_path: str, root: Path) -> Optional[list[str]]:
    """Return validated relative path components under ``root``."""
    if "\x00" in requested_path:
        raise HTTPException(
            status_code = 400,
            detail = "Path cannot contain null bytes",
        )
    root_text = os.path.normcase(os.path.normpath(str(root)))
    requested_text = os.path.normcase(os.path.normpath(requested_path))
    try:
        rel_text = os.path.relpath(requested_text, root_text)
    except ValueError:
        return None

    if rel_text == ".":
        return []
    if rel_text == ".." or rel_text.startswith(f"..{os.sep}"):
        return None

    parts = [part for part in rel_text.split(os.sep) if part not in ("", ".")]
    altsep = os.altsep
    for part in parts:
        if (
            part == ".."
            or "\x00" in part
            or os.sep in part
            or (altsep and altsep in part)
        ):
            return None
    return parts


def _match_browse_child(current: Path, name: str) -> Optional[Path]:
    """Return the immediate child named ``name`` under ``current``, or None
    if it does not exist.

    ``name`` is already validated as a single, safe path component (no
    separators, ``..``, or null bytes) by :func:`_browse_relative_parts`,
    so a direct ``current / name`` join is safe and O(1) -- there is no
    need to enumerate the whole directory per component. On case-insensitive
    filesystems (Windows, default macOS) the OS resolves a differently-cased
    component transparently, and the caller's subsequent ``.resolve()``
    canonicalizes the on-disk casing; on case-sensitive filesystems an exact
    match is required, matching the OS's own semantics.
    """
    child = current / name
    try:
        child.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    except PermissionError:
        raise HTTPException(
            status_code = 403,
            detail = f"Permission denied reading {current}",
        ) from None
    except OSError as exc:
        raise HTTPException(
            status_code = 500,
            detail = f"Could not read {current}: {exc}",
        ) from exc
    return child


def _resolve_browse_target(path: Optional[str], allowed_roots: list[Path]) -> Path:
    """Resolve a requested browse path by walking from trusted allowlist roots."""
    requested_path = _normalize_browse_request_path(path, relative_root = Path.home())
    resolved_roots: list[Path] = []
    seen_roots: set[str] = set()
    for root in sorted(allowed_roots, key = lambda p: len(str(p)), reverse = True):
        try:
            resolved = root.resolve()
        except OSError:
            continue
        key = os.path.normcase(os.path.realpath(str(resolved)))
        if key in seen_roots:
            continue
        seen_roots.add(key)
        resolved_roots.append(resolved)

    for root in resolved_roots:
        parts = _browse_relative_parts(requested_path, root)
        if parts is None:
            continue

        current = root
        for part in parts:
            child = _match_browse_child(current, part)
            if child is None:
                raise HTTPException(
                    status_code = 404,
                    detail = f"Path does not exist: {requested_path}",
                )
            try:
                resolved_child = child.resolve()
            except OSError as exc:
                raise HTTPException(
                    status_code = 400,
                    detail = f"Invalid path: {exc}",
                ) from exc
            if not _is_path_inside_allowlist(resolved_child, resolved_roots):
                raise HTTPException(
                    status_code = 403,
                    detail = (
                        "Path is not in the browseable allowlist. Register it via "
                        "POST /api/hub/scan-folders first, or pick a directory "
                        "under your home folder."
                    ),
                )
            # Same credential/config denylist scan-folder registration enforces:
            # HOME is in the allowlist, so without this a user could browse into
            # ~/.ssh, ~/.aws, etc. even though they can't be registered.
            if contains_sensitive_path_component(str(resolved_child)):
                raise HTTPException(
                    status_code = 403,
                    detail = "Credential or configuration directories are not browseable.",
                )
            current = resolved_child

        if not current.is_dir():
            raise HTTPException(
                status_code = 400,
                detail = f"Not a directory: {current}",
            )
        return current

    raise HTTPException(
        status_code = 403,
        detail = (
            "Path is not in the browseable allowlist. Register it via "
            "POST /api/hub/scan-folders first, or pick a directory "
            "under your home folder."
        ),
    )


def browse_folders_response(
    path: Optional[str] = None, show_hidden: bool = False
) -> BrowseFoldersResponse:
    """
    List immediate subdirectories of *path* for the Custom Folders picker.

    Sandbox: requests are bounded to the allowlist returned by
    :func:`_build_browse_allowlist` (HOME, HF cache, Studio dirs,
    registered scan folders, well-known model dirs). Paths outside the
    allowlist return 403 so users cannot probe ``/etc``, ``/proc``,
    ``/root`` (when not HOME), or other sensitive system locations
    even if the server process can read them. Symlinks are resolved
    via ``os.path.realpath`` before the check, so symlink traversal
    cannot escape the sandbox either.

    Sorting: directories that look like they hold models come first, then
    plain directories, then hidden entries (if `show_hidden=true`).
    """
    from hub.storage.scan_folders import list_scan_folders

    # Build the allowlist once -- both the sandbox check below and the
    # suggestion chips use the same set, so chips are always navigable.
    allowed_roots = _build_browse_allowlist()

    try:
        target = _resolve_browse_target(path, allowed_roots)
    except HTTPException:
        requested_path = _normalize_browse_request_path(
            path,
            relative_root = Path.home(),
        )
        if path is not None and path.strip():
            logger.warning(
                "browse-folders: rejected path %r (normalized=%s)",
                path,
                requested_path,
            )
        raise

    # Enumerate immediate subdirectories with a bounded cap so a stray
    # query against ``/usr/lib`` or ``/proc`` can't stat-storm the process.
    entries: list[BrowseEntry] = []
    truncated = False
    visited = 0
    try:
        it = target.iterdir()
    except PermissionError:
        raise HTTPException(
            status_code = 403,
            detail = f"Permission denied reading {target}",
        )
    except OSError as exc:
        raise HTTPException(
            status_code = 500,
            detail = f"Could not read {target}: {exc}",
        )

    try:
        for child in it:
            # Bound by *visited entries*, not by *appended entries*: in
            # directories full of files (or hidden subdirs when
            # ``show_hidden=False``) the cap on ``len(entries)`` would
            # never trigger and we'd still stat every child. Counting
            # visits keeps the worst-case work to ``_BROWSE_ENTRY_CAP``
            # iterdir/is_dir calls regardless of how many of them
            # survive the filters below.
            visited += 1
            if visited > _BROWSE_ENTRY_CAP:
                truncated = True
                break
            try:
                if not child.is_dir():
                    continue
            except OSError:
                continue
            name = child.name
            is_hidden = name.startswith(".")
            if is_hidden and not show_hidden:
                continue
            # Don't surface credential/config dirs as navigable options even
            # with show_hidden: descending into them is refused anyway, and
            # registration rejects them. Keeps the two surfaces consistent.
            if contains_sensitive_path_component(name):
                continue
            entries.append(
                BrowseEntry(
                    name = name,
                    has_models = _looks_like_model_dir(child),
                    hidden = is_hidden,
                )
            )
    except PermissionError as exc:
        logger.debug(
            "browse-folders: permission denied during enumeration of %s: %s",
            target,
            exc,
        )
    except OSError as exc:
        # Rare: iterdir succeeded but reading a specific entry failed.
        logger.warning("browse-folders: partial enumeration of %s: %s", target, exc)

    # Model-bearing dirs first, then plain, then hidden; case-insensitive
    # alphabetical within each bucket.
    def _sort_key(e: BrowseEntry) -> tuple[int, str]:
        bucket = 0 if e.has_models else (2 if e.hidden else 1)
        return (bucket, e.name.lower())

    entries.sort(key = _sort_key)

    # Parent is None at the filesystem root (`p.parent == p`) AND when
    # the parent would step outside the sandbox -- otherwise the up-row
    # would 403 on click. Users can still hop to other allowed roots
    # via the suggestion chips below.
    parent: Optional[str]
    if target.parent == target or not _is_path_inside_allowlist(
        target.parent, allowed_roots
    ):
        parent = None
    else:
        parent = str(target.parent)

    # Handy starting points for the quick-pick chips.
    suggestions: list[str] = []
    seen_sug: set[str] = set()

    def _add_sug(p: Optional[Path | str]) -> None:
        if p is None:
            return
        try:
            p = Path(normalize_path(str(p))).expanduser()
            resolved = str(p.resolve())
        except (OSError, RuntimeError, ValueError):
            return
        if resolved in seen_sug:
            return
        if _safe_is_dir(resolved):
            seen_sug.add(resolved)
            suggestions.append(resolved)

    # Home always comes first -- it's the safe fallback when everything
    # else is cold.
    _add_sug(Path.home())
    # The HF cache root the process is actually using (honors HF_HOME /
    # HF_HUB_CACHE), then the platform default as a secondary chip.
    try:
        _add_sug(_resolve_hf_cache_dir())
    except Exception:
        pass
    try:
        _add_sug(hf_default_cache_dir())
    except Exception:
        pass
    # Already-registered scan folders (what the user has curated).
    try:
        for folder in list_scan_folders():
            _add_sug(folder.get("path", ""))
    except Exception as exc:
        logger.debug("browse-folders: could not load scan folders: %s", exc)
    # Directories commonly used by other local-LLM tools: LM Studio
    # (`~/.lmstudio/models` + legacy `~/.cache/lm-studio/models` +
    # user-configured downloadsFolder from LM Studio's settings.json),
    # Ollama (`~/.ollama/models` + common system paths + OLLAMA_MODELS
    # env var), and generic user-choice spots (`~/models`, `~/Models`).
    # Each helper only returns paths that currently exist so we never
    # show dead chips.
    try:
        for p in well_known_model_dirs():
            _add_sug(p)
    except Exception as exc:
        logger.debug("browse-folders: could not load well-known dirs: %s", exc)

    return BrowseFoldersResponse(
        current = str(target),
        parent = parent,
        entries = entries,
        suggestions = suggestions,
        truncated = truncated,
        model_files_here = _count_model_files(target),
    )
