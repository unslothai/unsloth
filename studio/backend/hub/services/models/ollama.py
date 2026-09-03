# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ollama model inventory: manifest parsing and writable-symlink materialization.

Ollama stores models content-addressed under ``<root>/manifests/`` and
``<root>/blobs/``. Inventory scans read the manifests directly (no writes),
returning rows whose ``id`` is an opaque ``ollama-manifest:`` reference. The
load path then calls :func:`materialize_ollama_model_ref`, which creates a
``.gguf``-named symlink (or hardlink) so that downstream loaders see a path
with the GGUF suffix without copying multi-GB blobs inside an API request.
"""

from __future__ import annotations

import hashlib
import json
import os

import threading
import uuid
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote, unquote

from loggers import get_logger

from hub.schemas.inventory import LocalModelInfo
from hub.services.models.common import (
    _capabilities_for_format,
    _local_inventory_id,
)
from hub.utils.paths import (
    cache_root,
    ollama_model_dirs,
    path_is_same_or_child,
    tmp_root,
)

logger = get_logger(__name__)

_OLLAMA_MANIFEST_REF_PREFIX = "ollama-manifest:"
_OLLAMA_BLOB_NAME_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._+-"
)

_OLLAMA_LOADABLE_LAYER_MEDIA_TYPES = frozenset(
    {
        "application/vnd.ollama.image.model",
        "application/vnd.ollama.image.projector",
        # License text does not affect model behavior and does not need to be carried into llama.cpp.
        "application/vnd.ollama.image.license",
    }
)

_OLLAMA_MATERIALIZE_LOCKS: dict[str, threading.Lock] = {}
_OLLAMA_MATERIALIZE_LOCKS_GUARD = threading.Lock()


class OllamaModelLease:
    def __init__(self, path: str, lock: threading.Lock):
        self.path = path
        self._lock = lock
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._lock.release()


def _ollama_manifest_ref(tag_file: Path) -> str:
    return f"{_OLLAMA_MANIFEST_REF_PREFIX}{quote(str(tag_file), safe = '')}"


def is_ollama_manifest_ref(ref: str) -> bool:
    """True when *ref* is an opaque ``ollama-manifest:`` inventory reference."""
    return ref.startswith(_OLLAMA_MANIFEST_REF_PREFIX)


def _unsupported_ollama_layer_media_types(layers: list[object]) -> tuple[str, ...]:
    """Layer types whose behavior the direct llama.cpp load cannot preserve."""
    unsupported: set[str] = set()
    for layer in layers:
        if not isinstance(layer, dict):
            unsupported.add("<invalid layer>")
            continue
        media_type = layer.get("mediaType")
        if not isinstance(media_type, str) or not media_type:
            unsupported.add("<missing mediaType>")
        elif media_type not in _OLLAMA_LOADABLE_LAYER_MEDIA_TYPES:
            unsupported.add(media_type)
    return tuple(sorted(unsupported))


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _ollama_blob_path(blobs_dir: Path, digest: object) -> Optional[Path]:
    if not isinstance(digest, str):
        return None
    algorithm, separator, value = digest.partition(":")
    if separator != ":" or not algorithm or not value:
        return None
    name = f"{algorithm}-{value}"
    if (
        not name
        or name in (".", "..")
        or any(char not in _OLLAMA_BLOB_NAME_CHARS for char in name)
        or not name.isprintable()
    ):
        return None
    return blobs_dir / name


def _contained_link_path(link_dir: Path, link_name: str) -> Optional[Path]:
    """Resolve *link_name* to a direct child of *link_dir*, or ``None``. ``link_name`` derives from manifest fields, so requiring a direct child keeps a crafted value with separators, ``..``, or a drive prefix from escaping the links dir."""
    if not link_name or link_name in (".", ".."):
        return None
    link_path = link_dir / link_name
    try:
        if link_path.parent.resolve() != link_dir.resolve():
            return None
    except OSError:
        return None
    return link_path


def _ollama_links_dir(ollama_dir: Path) -> Optional[Path]:
    """Writable directory for Ollama ``.gguf`` symlinks. Prefers ``<ollama_dir>/.studio_links/`` next to the blobs; falls back to Unsloth's cache (read-only system installs), then the temp dir (sandboxed installs)."""

    def _ensure_writable_dir(path: Path) -> Optional[Path]:
        try:
            path.mkdir(parents = True, exist_ok = True)
            probe = path / f".write-test-{uuid.uuid4().hex[:8]}"
            probe.mkdir()
            probe.rmdir()
            return path
        except OSError as e:
            logger.debug("Ollama link dir %s is not writable: %s", path, e)
            return None

    primary = ollama_dir / ".studio_links"
    if _ensure_writable_dir(primary) is not None:
        return primary

    # Namespace by a hash of the ollama_dir so two different Ollama roots do not collide. A cache path,
    # not a security boundary.
    try:
        digest = hashlib.sha256(str(ollama_dir.resolve()).encode()).hexdigest()[:12]
    except (OSError, RuntimeError):
        digest = "default"

    fallback = cache_root() / "ollama_links" / digest
    if _ensure_writable_dir(fallback) is not None:
        return fallback

    tmp_fallback = tmp_root() / "ollama_links" / digest
    if _ensure_writable_dir(tmp_fallback) is not None:
        return tmp_fallback

    logger.warning(
        "Could not create a writable Ollama link directory for %s",
        ollama_dir,
    )
    return None


def _make_ollama_blob_link(link_dir: Path, link_name: str, target: Path) -> Optional[str]:
    """Create a .gguf-named link to an Ollama blob: tries symlink then hardlink, skips the model if neither works (a full multi-GB copy would block the API). Idempotent."""
    try:
        link_dir.mkdir(parents = True, exist_ok = True)
    except OSError as e:
        logger.warning(
            "Could not create Ollama link directory %s: %s",
            link_dir,
            e,
        )
        return None
    link_path = _contained_link_path(link_dir, link_name)
    if link_path is None:
        logger.warning("Refusing unsafe Ollama link name %r under %s", link_name, link_dir)
        return None
    try:
        resolved = target.resolve()
    except OSError as e:
        logger.debug("Could not resolve Ollama blob %s: %s", target, e)
        return None

    # samefile, not size: `ollama pull` can swap a tag to a same-sized blob, leaving a stale link.
    try:
        if link_path.exists() and os.path.samefile(str(link_path), str(resolved)):
            return str(link_path)
    except OSError as e:
        logger.debug("Error checking existing link %s: %s", link_path, e)

    tmp_path = link_dir / f".{link_name}.tmp-{uuid.uuid4().hex[:8]}"
    try:
        if tmp_path.is_symlink() or tmp_path.exists():
            tmp_path.unlink()
        try:
            tmp_path.symlink_to(resolved)
        except OSError:
            try:
                os.link(str(resolved), str(tmp_path))
            except OSError:
                logger.warning(
                    "Could not create link for Ollama blob %s "
                    "(symlinks and hardlinks both failed). "
                    "Skipping model to avoid blocking the API.",
                    target,
                )
                return None
        os.replace(str(tmp_path), str(link_path))
        return str(link_path)
    except OSError as e:
        logger.debug("Could not create Ollama link %s: %s", link_path, e)
        try:
            if tmp_path.is_symlink() or tmp_path.exists():
                tmp_path.unlink()
        except OSError as cleanup_err:
            logger.debug("Could not clean up tmp path %s: %s", tmp_path, cleanup_err)
        return None


def _ollama_model_info_from_manifest(
    ollama_dir: Path,
    tag_file: Path,
    *,
    materialize_links: bool = False,
    links_root: Optional[Path] = None,
    reject_unsupported_layers: bool = False,
) -> Optional[LocalModelInfo]:
    manifests_root = ollama_dir / "manifests"
    blobs_dir = ollama_dir / "blobs"

    try:
        rel = tag_file.relative_to(manifests_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 3:
        return None

    host = parts[0]
    repo_parts = list(parts[1:-1])
    tag = parts[-1]

    if host == "registry.ollama.ai" and repo_parts and repo_parts[0] == "library":
        repo_name = "/".join(repo_parts[1:])
    elif host == "registry.ollama.ai":
        repo_name = "/".join(repo_parts)
    else:
        repo_name = "/".join([host] + repo_parts)

    if not repo_name:
        return None

    def invalid_manifest(reason: str) -> Optional[LocalModelInfo]:
        message = f"Invalid Ollama manifest: {reason}"
        if reject_unsupported_layers:
            raise ValueError(message)
        logger.debug("Skipping %s (%s)", tag_file, message)
        return None

    try:
        manifest = json.loads(tag_file.read_text(encoding = "utf-8-sig"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        return invalid_manifest(str(e))
    if not isinstance(manifest, dict):
        return invalid_manifest("top level must be a JSON object")

    config = manifest.get("config", {})
    if not isinstance(config, dict):
        return invalid_manifest("config must be a JSON object")
    config_digest = config.get("digest", "")
    model_type = ""
    file_type = ""
    if config_digest and blobs_dir.is_dir():
        config_blob = _ollama_blob_path(blobs_dir, config_digest)
        if config_blob is not None and _safe_is_file(config_blob):
            try:
                cfg = json.loads(config_blob.read_text(encoding = "utf-8-sig"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
                return invalid_manifest(f"config blob could not be parsed: {e}")
            if not isinstance(cfg, dict):
                return invalid_manifest("config blob must be a JSON object")
            model_type = cfg.get("model_type", "")
            file_type = cfg.get("file_type", "")

    layers = manifest.get("layers") or []
    if not isinstance(layers, list):
        return None

    unsupported_layers = _unsupported_ollama_layer_media_types(layers)
    if unsupported_layers:
        rendered_layers = ", ".join(unsupported_layers)
        if reject_unsupported_layers:
            raise ValueError(
                "Ollama manifest contains unsupported runtime layers that Unsloth cannot preserve: "
                f"{rendered_layers}"
            )
        logger.debug(
            "Skipping Ollama manifest %s with unsupported runtime layers: %s",
            tag_file,
            rendered_layers,
        )
        return None

    model_blob: Optional[Path] = None
    projector_blob: Optional[Path] = None
    gguf_link_path: Optional[str] = None
    stem_hash = hashlib.sha256(rel.as_posix().encode()).hexdigest()[:10]
    model_link_dir = links_root / stem_hash if links_root is not None else None
    safe_name = repo_name.replace("/", "-")

    for layer in layers:
        if not isinstance(layer, dict):
            continue
        media = layer.get("mediaType", "")
        digest = layer.get("digest", "")
        if media not in {
            "application/vnd.ollama.image.model",
            "application/vnd.ollama.image.projector",
        }:
            continue
        candidate = _ollama_blob_path(blobs_dir, digest) if digest else None
        if candidate is None or not _safe_is_file(candidate):
            layer_name = "model" if media.endswith(".model") else "projector"
            return invalid_manifest(f"{layer_name} blob is missing")
        if media == "application/vnd.ollama.image.model":
            model_blob = candidate
        else:
            projector_blob = candidate

    if model_blob is None:
        return invalid_manifest("model blob is missing")

    if materialize_links:
        if model_link_dir is None:
            return invalid_manifest("link directory is unavailable")
        link_name = f"{safe_name}-{tag}.gguf"
        mmproj_name = f"{safe_name}-{tag}-mmproj.gguf"
        projector_link = _contained_link_path(model_link_dir, mmproj_name)
        if projector_link is None:
            return invalid_manifest("projector link name is unsafe")
        previous_projector: Optional[Path] = None
        previous_projector_hardlink: Optional[Path] = None
        try:
            if projector_link.is_symlink():
                try:
                    previous_projector = projector_link.resolve(strict = True)
                except FileNotFoundError:
                    pass
            elif projector_link.exists():
                previous_projector_hardlink = model_link_dir / (
                    f".{mmproj_name}.rollback-{uuid.uuid4().hex[:8]}"
                )
                os.link(str(projector_link), str(previous_projector_hardlink))
        except (OSError, RuntimeError) as e:
            return invalid_manifest(f"existing projector link could not be preserved: {e}")

        try:
            if projector_blob is not None:
                if not _make_ollama_blob_link(model_link_dir, mmproj_name, projector_blob):
                    return invalid_manifest("could not materialize projector blob")
            else:
                try:
                    if projector_link.is_symlink() or projector_link.exists():
                        projector_link.unlink()
                except OSError as e:
                    return invalid_manifest(f"stale projector link could not be removed: {e}")

            gguf_link_path = _make_ollama_blob_link(model_link_dir, link_name, model_blob)
            if not gguf_link_path:
                restored = False
                if previous_projector_hardlink is not None:
                    try:
                        os.replace(str(previous_projector_hardlink), str(projector_link))
                        restored = True
                    except OSError:
                        pass
                elif previous_projector is not None:
                    restored = bool(
                        _make_ollama_blob_link(model_link_dir, mmproj_name, previous_projector)
                    )
                else:
                    try:
                        if projector_link.is_symlink() or projector_link.exists():
                            projector_link.unlink()
                        restored = True
                    except OSError:
                        pass
                if not restored:
                    return invalid_manifest(
                        "could not materialize model blob or restore the previous projector"
                    )
                return invalid_manifest("could not materialize model blob")
        finally:
            if previous_projector_hardlink is not None:
                try:
                    if previous_projector_hardlink.exists():
                        previous_projector_hardlink.unlink()
                except OSError as e:
                    logger.debug(
                        "Could not clean up Ollama projector rollback link %s: %s",
                        previous_projector_hardlink,
                        e,
                    )

    suffix = ""
    if model_type:
        suffix += f" ({model_type}"
        if file_type:
            suffix += f" {file_type}"
        suffix += ")"

    try:
        updated_at = tag_file.stat().st_mtime
    except OSError:
        updated_at = None

    display = f"{repo_name}:{tag}"
    model_id = f"ollama/{repo_name}:{tag}"
    path = gguf_link_path if materialize_links and gguf_link_path else str(model_blob)
    load_id = path if materialize_links else _ollama_manifest_ref(tag_file)
    return LocalModelInfo(
        id = load_id,
        inventory_id = _local_inventory_id("ollama", "gguf", model_id),
        load_id = load_id,
        model_id = model_id,
        display_name = display + suffix,
        path = path,
        source = "ollama",
        updated_at = updated_at,
        model_format = "gguf",
        runtime = "llama_cpp",
        capabilities = _capabilities_for_format("gguf", "ollama"),
    )


def scan_ollama_dir(
    ollama_dir: Path,
    *,
    limit: Optional[int] = None,
    materialize_links: bool = False,
) -> List[LocalModelInfo]:
    """Scan an Ollama models directory for downloaded models.

    Ollama uses a content-addressable layout
    (``manifests/<host>/<namespace>/<model>/<tag>`` + ``blobs/sha256-...``),
    iterated via ``rglob`` to find every depth. Each manifest's ``model`` layer
    holds the GGUF weights (vision models add a projector layer).

    Scans are read-only by default and return an opaque manifest reference;
    the load route later calls :func:`materialize_ollama_model_ref` to create a
    ``.gguf`` symlink/hardlink, keeping GET /local free of filesystem writes.
    """
    manifests_root = ollama_dir / "manifests"
    if not manifests_root.is_dir():
        return []

    found: List[LocalModelInfo] = []
    links_root = _ollama_links_dir(ollama_dir) if materialize_links else None
    if materialize_links and links_root is None:
        logger.warning(
            "Skipping Ollama scan for %s: no writable location for .gguf links",
            ollama_dir,
        )
        return []

    try:
        for tag_file in manifests_root.rglob("*"):
            if not _safe_is_file(tag_file):
                continue

            info = _ollama_model_info_from_manifest(
                ollama_dir,
                tag_file,
                materialize_links = materialize_links,
                links_root = links_root,
            )
            if info is None:
                continue
            found.append(info)
            if limit is not None and len(found) >= limit:
                return found
    except OSError as e:
        logger.warning("Error scanning Ollama directory %s: %s", ollama_dir, e)
    return found


def _ollama_dir_for_manifest(tag_file: Path) -> Optional[Path]:
    """Return a discovered or registered Ollama root containing *tag_file*."""
    known_dirs = list(ollama_model_dirs())
    try:
        from hub.storage.scan_folders import list_scan_folders
        known_dirs.extend(
            Path(folder["path"]).expanduser()
            for folder in list_scan_folders()
            if folder.get("path")
        )
    except Exception as e:
        logger.debug("Could not load registered Ollama roots: %s", e)
    for ollama_dir in known_dirs:
        if path_is_same_or_child(tag_file, ollama_dir / "manifests"):
            return ollama_dir
    return None


def _validated_ollama_manifest_location(ref: str) -> tuple[Path, Path]:
    if not ref.startswith(_OLLAMA_MANIFEST_REF_PREFIX):
        raise ValueError("Not an Ollama manifest reference")
    try:
        tag_file = Path(os.path.realpath(unquote(ref[len(_OLLAMA_MANIFEST_REF_PREFIX) :])))
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid Ollama manifest reference: {e}") from e
    ollama_dir = _ollama_dir_for_manifest(tag_file)
    if ollama_dir is None:
        raise ValueError("Reference is outside any known Ollama models directory")
    try:
        canonical_ollama_dir = Path(os.path.realpath(str(ollama_dir)))
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid Ollama models directory: {e}") from e
    return tag_file, canonical_ollama_dir


def _materialization_lock(tag_file: Path) -> threading.Lock:
    key = os.path.normcase(str(tag_file))
    with _OLLAMA_MATERIALIZE_LOCKS_GUARD:
        return _OLLAMA_MATERIALIZE_LOCKS.setdefault(key, threading.Lock())


def _materialize_ollama_model_ref_unlocked(tag_file: Path, ollama_dir: Path) -> str:
    links_root = _ollama_links_dir(ollama_dir)
    if links_root is None:
        raise ValueError("No writable location for Ollama .gguf links")

    info = _ollama_model_info_from_manifest(
        ollama_dir,
        tag_file,
        materialize_links = True,
        links_root = links_root,
        reject_unsupported_layers = True,
    )
    if info is None or not info.path:
        raise ValueError("Could not materialize Ollama model from manifest")
    return info.path


def materialize_ollama_model_ref(ref: str) -> str:
    """Resolve an Ollama ref while serializing updates to its model/projector pair."""
    tag_file, ollama_dir = _validated_ollama_manifest_location(ref)
    with _materialization_lock(tag_file):
        return _materialize_ollama_model_ref_unlocked(tag_file, ollama_dir)


def acquire_ollama_model_ref(ref: str) -> OllamaModelLease:
    """Materialize and keep the pair stable until the caller releases the lease."""
    tag_file, ollama_dir = _validated_ollama_manifest_location(ref)
    lock = _materialization_lock(tag_file)
    lock.acquire()
    try:
        return OllamaModelLease(_materialize_ollama_model_ref_unlocked(tag_file, ollama_dir), lock)
    except BaseException:
        lock.release()
        raise
