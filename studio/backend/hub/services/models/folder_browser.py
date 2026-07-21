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
    is_denied_system_path,
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
from utils.paths.external_media import (
    linux_run_media_mount_roots,
    macos_volume_roots,
    windows_drive_roots,
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


# Ceiling on children to stat when guessing if a directory holds models.
_BROWSE_MODEL_HINT_PROBE = 64
# Hard cap on returned subdirectory entries so pointing at ``/usr/lib`` or
# ``/proc`` can't stat-storm the process or flood the client.
_BROWSE_ENTRY_CAP = 2000


def _count_model_files(directory: Path, cap: int = 200) -> int:
    """Count GGUF/safetensors files immediately inside *directory*, bounded by visited entries (not matches) so the hint never costs more than ``cap`` stats."""
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
    """True if an immediate child signals a model (GGUF/safetensors/config file or ``models--*`` HF-cache subdir); bounded by the hint probe."""
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
    """Bounded heuristic flagging dirs worth exploring (false negatives are fine; the scanner is authoritative). Three signals, cheapest first: a ``models--*`` name, a direct child signal, or a grandchild signal (LM Studio / Ollama ``publisher/model/weights.gguf`` layout)."""
    if directory.name.startswith("models--"):
        return True
    if _has_direct_model_signal(directory):
        return True
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


def _build_browse_allowlist(
    media_roots: Optional[list[Path]] = None, drive_roots: Optional[list[Path]] = None
) -> list[Path]:
    """Root directories the browser may walk (also seeds the suggestion chips): HOME, resolved HF cache dirs, Unsloth outputs/exports/root, registered scan folders, and well-known local-LLM dirs. Each is added only if it resolves to a real directory so the sandbox has no dead boundary.

    *media_roots* / *drive_roots* let the caller pass already-probed
    removable-media and Windows drive roots so they aren't scanned again (a
    disconnected mapped drive can make each probe slow); probed here when ``None``."""
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
    if media_roots is None:
        media_roots = [*linux_run_media_mount_roots(), *macos_volume_roots()]
    if drive_roots is None:
        drive_roots = windows_drive_roots()
    for p in media_roots:
        _add(p)
    for p in drive_roots:
        _add(p)
    _add(_resolve_hf_cache_dir())
    try:
        from utils.hf_cache_settings import known_hf_cache_homes

        for cache_home in known_hf_cache_homes():
            _add(cache_home)
    except Exception:  # noqa: BLE001 -- best-effort
        pass
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
    """True if *target* equals or descends from any allowed root; uses ``os.path.realpath`` so symlinks cannot escape the sandbox.

    A Windows drive root (``D:\\``) authorizes its descendants, but a bare POSIX
    root (``/``) must NOT: a single ``/`` allowlist entry (e.g. a legacy scan
    folder) would otherwise authorize every absolute path, reaching ``/var``,
    ``/root``, etc. the denylist does not cover. Mirrors the legacy browser so
    both treat ``/`` identically.
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
        if target_real == root_real:
            return True
        drive, tail = os.path.splitdrive(root_real)
        if os.path.dirname(root_real) == root_real and not drive:
            # Bare POSIX filesystem root ("/"): equality above is the only
            # match; do not let it authorize arbitrary descendants.
            continue
        if drive.startswith(("\\\\", "//")) and not tail:
            # Bare UNC share root (\\server\share): os.path.commonpath raises
            # "can't mix absolute and relative" on it, so authorize its
            # descendants with a boundary-safe prefix test (normcase applied).
            if target_real.startswith(root_real.rstrip("\\/") + os.sep):
                return True
            continue
        try:
            if os.path.commonpath([target_real, root_real]) == root_real:
                return True
        except ValueError:
            continue
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
        if part == ".." or "\x00" in part or os.sep in part or (altsep and altsep in part):
            return None
    return parts


def _match_browse_child(current: Path, name: str) -> Optional[Path]:
    """Immediate child named ``name`` under ``current``, or None. ``name`` is pre-validated as a safe single component, so the join is O(1); case resolution follows OS filesystem semantics."""
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
            # HOME is in the allowlist, so without this denylist (same one
            # registration enforces) a user could browse into ~/.ssh, ~/.aws, etc.
            if contains_sensitive_path_component(str(resolved_child)):
                raise HTTPException(
                    status_code = 403,
                    detail = "Credential or configuration directories are not browseable.",
                )
            if is_denied_system_path(str(resolved_child)):
                raise HTTPException(
                    status_code = 403,
                    detail = "System directories are not browseable.",
                )
            current = resolved_child

        if contains_sensitive_path_component(str(current)):
            raise HTTPException(
                status_code = 403,
                detail = "Credential or configuration directories are not browseable.",
            )
        # Zero-component case: the requested path IS an allowlist root
        # (e.g. a legacy-registered "/" or a Windows drive root).
        if is_denied_system_path(str(current)):
            raise HTTPException(
                status_code = 403,
                detail = "System directories are not browseable.",
            )
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
    """List immediate subdirectories of *path* for the Custom Folders picker.

    Requests are bounded to the :func:`_build_browse_allowlist` roots; paths
    outside it return 403 (symlinks resolved via realpath first, so traversal
    can't escape). Sorting: model-bearing dirs first, then plain, then hidden.
    """
    from hub.storage.scan_folders import list_scan_folders

    # Probe removable-media and Windows drive roots once; the allowlist and
    # chips reuse the result so a disconnected mapped drive isn't scanned twice.
    media_roots = [*linux_run_media_mount_roots(), *macos_volume_roots()]
    drive_roots = windows_drive_roots()
    # Build the allowlist once -- the sandbox check and suggestion chips share
    # it so chips are always navigable.
    allowed_roots = _build_browse_allowlist(media_roots, drive_roots)

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
            # Bound by visited entries, not appended ones, so a directory full
            # of files still caps work at ``_BROWSE_ENTRY_CAP`` stats.
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
            # Don't surface credential/config dirs even with show_hidden:
            # descending into them is refused and registration rejects them.
            if contains_sensitive_path_component(name):
                continue
            # Same for denied system dirs (C:\Windows, /etc, ...): descent 403s,
            # so don't render them as clickable rows. Resolve first so a
            # symlink/junction into a denied dir is hidden too, not just a literal name.
            try:
                resolved_child = os.path.realpath(str(child))
            except (OSError, ValueError):
                resolved_child = str(child)
            if is_denied_system_path(resolved_child):
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

    # Parent is None at the FS root and when it would step outside the sandbox,
    # so the up-row never 403s on click.
    parent: Optional[str]
    if target.parent == target or not _is_path_inside_allowlist(target.parent, allowed_roots):
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
        # Drop a denied system dir (e.g. a stale scan-folder row) so it never
        # becomes a chip that 403s on click. Drive roots stay: only their
        # system subdirectories are denied, not the root itself.
        if is_denied_system_path(resolved):
            return
        if _safe_is_dir(resolved):
            seen_sug.add(resolved)
            suggestions.append(resolved)

    # Home first as the safe fallback.
    _add_sug(Path.home())
    # Reuse the roots probed for the allowlist above (no second drive scan).
    for p in media_roots:
        _add_sug(p)
    # Windows drive roots so the user can hop between C:, D:, E: ...
    for p in drive_roots:
        _add_sug(p)
    # The HF cache root in use (honors HF_HOME / HF_HUB_CACHE), then the default.
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
    # Well-known third-party dirs (LM Studio, Ollama, ~/models). Each helper
    # only returns existing paths so we never show dead chips.
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
