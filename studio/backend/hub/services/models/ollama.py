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

from core.inference.scan_incidents import note_scan_incident
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
_OLLAMA_LINK_DIR_NAMES = frozenset((".studio_links", "ollama_links"))
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

# Modelfile metadata nearly every pulled model carries. llama.cpp uses the GGUF's own template
# and defaults instead, so none of it reaches the load and none of it should hide the row.
# `image.adapter` is deliberately absent: it changes the weights.
_OLLAMA_METADATA_LAYER_MEDIA_TYPES = frozenset(
    {
        "application/vnd.ollama.image.template",
        "application/vnd.ollama.image.params",
        "application/vnd.ollama.image.system",
        "application/vnd.ollama.image.messages",
        "application/vnd.ollama.image.prompt",
    }
)

# Layers the load drops without changing what it returns. `image.draft` only feeds
# speculative decoding, which is output-equivalent by construction, so losing it costs
# speed and nothing else; `image.embed` has been deprecated since ollama 0.1.2 and
# ollama's own load ignores it too.
_OLLAMA_IGNORABLE_LAYER_MEDIA_TYPES = frozenset(
    {
        "application/vnd.ollama.image.draft",
        "application/vnd.ollama.image.embed",
    }
)

_OLLAMA_ADMITTED_LAYER_MEDIA_TYPES = (
    _OLLAMA_LOADABLE_LAYER_MEDIA_TYPES
    | _OLLAMA_METADATA_LAYER_MEDIA_TYPES
    | _OLLAMA_IGNORABLE_LAYER_MEDIA_TYPES
)

_OLLAMA_MATERIALIZE_LOCKS: dict[tuple, threading.Lock] = {}
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
    """Layer types that would make the row describe something the load cannot deliver."""
    unsupported: set[str] = set()
    for layer in layers:
        if not isinstance(layer, dict):
            unsupported.add("<invalid layer>")
            continue
        media_type = layer.get("mediaType")
        if not isinstance(media_type, str) or not media_type:
            unsupported.add("<missing mediaType>")
        elif media_type not in _OLLAMA_ADMITTED_LAYER_MEDIA_TYPES:
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


def _ollama_links_roots(ollama_dir: Path) -> tuple[Path, ...]:
    """Where *ollama_dir*'s ``.gguf`` links can live, best first: beside the blobs, then Unsloth's cache (read-only installs), then the temp dir (sandboxed installs)."""
    # Hashed so two Ollama roots cannot collide. A cache path, not a security boundary.
    try:
        digest = hashlib.sha256(str(ollama_dir.resolve()).encode()).hexdigest()[:12]
    except (OSError, RuntimeError):
        digest = "default"
    return (
        ollama_dir / ".studio_links",
        cache_root() / "ollama_links" / digest,
        tmp_root() / "ollama_links" / digest,
    )


def _ollama_links_dir(ollama_dir: Path) -> Optional[Path]:
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

    for candidate in _ollama_links_roots(ollama_dir):
        if _ensure_writable_dir(candidate) is not None:
            return candidate

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


def _manifest_rel_path(tag_file: Path, manifests_root: Path) -> Optional[Path]:
    try:
        return tag_file.relative_to(manifests_root)
    except ValueError:
        # ``manifests`` can be a symlink: a reference carries the canonical path, a scan its own.
        try:
            return Path(os.path.realpath(tag_file)).relative_to(os.path.realpath(manifests_root))
        except (OSError, ValueError):
            return None


def _manifest_stem_hash(rel: Path) -> str:
    return hashlib.sha256(rel.as_posix().encode()).hexdigest()[:10]


def _ollama_model_info_from_manifest(
    ollama_dir: Path,
    tag_file: Path,
    *,
    materialize_links: bool = False,
    links_root: Optional[Path] = None,
    reject_unsupported_layers: bool = False,
    existing_links_only: bool = False,
    manifest: Optional[dict] = None,
) -> Optional[LocalModelInfo]:
    manifests_root = ollama_dir / "manifests"
    blobs_dir = ollama_dir / "blobs"

    rel = _manifest_rel_path(tag_file, manifests_root)
    if rel is None:
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

    if manifest is None:
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
    model_link_dir = links_root / _manifest_stem_hash(rel) if links_root is not None else None
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

    if materialize_links and existing_links_only:
        link = (
            _contained_link_path(model_link_dir, f"{safe_name}-{tag}.gguf")
            if model_link_dir is not None
            else None
        )
        if link is None or not _safe_is_file(link):
            return None
        gguf_link_path = str(link)
    elif materialize_links:
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

            lock = _materialization_lock(tag_file, ollama_dir) if materialize_links else None
            # A load holds this tag; its lease keeps the link it made, so report that, never block.
            leased = lock is not None and not lock.acquire(blocking = False)
            try:
                info = _ollama_model_info_from_manifest(
                    ollama_dir,
                    tag_file,
                    materialize_links = materialize_links,
                    links_root = links_root,
                    existing_links_only = leased,
                )
            finally:
                if lock is not None and not leased:
                    lock.release()
            if info is None:
                continue
            found.append(info)
            if limit is not None and len(found) >= limit:
                return found
    except OSError as e:
        logger.warning("Error scanning Ollama directory %s: %s", ollama_dir, e)
        # A partial list reads exactly like a complete one from outside, so say so for the
        # caller that treats a miss as a confirmed absence.
        note_scan_incident(f"ollama dir unreadable: {ollama_dir}")
    return found


def _known_ollama_dirs() -> List[Path]:
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
    return known_dirs


def ollama_manifest_ref_for_path(model_path: str) -> Optional[str]:
    """The manifest reference *model_path* names, or None when it names no Ollama tag."""
    if is_ollama_manifest_ref(model_path):
        return model_path
    link = Path(model_path)
    if not any(part in _OLLAMA_LINK_DIR_NAMES for part in link.parts):
        return None
    for ollama_dir in _known_ollama_dirs():
        if not any(
            path_is_same_or_child(link.parent, root) for root in _ollama_links_roots(ollama_dir)
        ):
            continue
        manifests_root = ollama_dir / "manifests"
        try:
            for tag_file in manifests_root.rglob("*"):
                if not _safe_is_file(tag_file):
                    continue
                rel = _manifest_rel_path(tag_file, manifests_root)
                if rel is not None and _manifest_stem_hash(rel) == link.parent.name:
                    return _ollama_manifest_ref(tag_file)
        except OSError as e:
            logger.debug("Could not walk Ollama manifests under %s: %s", manifests_root, e)
    return None


def _ollama_dir_for_manifest(tag_file: Path) -> Optional[Path]:
    for ollama_dir in _known_ollama_dirs():
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


def _ollama_model_ref_info(ref: str) -> tuple[Path, dict, LocalModelInfo]:
    """Ollama root, parsed manifest and model row for *ref*, from a single read: a pull between two
    reads would pair one version's weights with another's projector. Raises on nothing loadable."""
    tag_file, ollama_dir = _validated_ollama_manifest_location(ref)
    try:
        manifest = json.loads(tag_file.read_text(encoding = "utf-8-sig"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        raise ValueError(f"Could not read Ollama manifest: {e}") from e
    # A manifest of JSON ``null`` is indistinguishable from passing none below, which re-reads.
    if not isinstance(manifest, dict):
        raise ValueError("Invalid Ollama manifest: top level must be a JSON object")
    info = _ollama_model_info_from_manifest(
        ollama_dir,
        tag_file,
        materialize_links = False,
        reject_unsupported_layers = True,
        manifest = manifest,
    )
    if info is None:
        raise ValueError("Could not resolve Ollama model from manifest")
    return ollama_dir, manifest, info


def ollama_model_ref_public_id(ref: str) -> str:
    """The ``ollama/<repo>:<tag>`` id clients see for *ref*, whose own form is internal."""
    _, _, info = _ollama_model_ref_info(ref)
    return info.model_id


def ollama_model_ref_files(ref: str) -> tuple[str, Optional[str]]:
    """``(weights blob, projector blob or None)`` for *ref*: the blobs, not the links a load makes."""
    ollama_dir, manifest, info = _ollama_model_ref_info(ref)
    projector = None
    for layer in manifest.get("layers") or []:
        if layer.get("mediaType") == "application/vnd.ollama.image.projector":
            projector = _ollama_blob_path(ollama_dir / "blobs", layer.get("digest"))
    return info.path, str(projector) if projector is not None else None


def _materialization_lock(tag_file: Path, ollama_dir: Path) -> threading.Lock:
    """Per-tag lock, keyed by the store's inode and the manifest's path inside it. Not by pathname:
    one store reached through a symlink or spelled in another case would hand out two locks."""
    try:
        stat = os.stat(ollama_dir)
        root: object = (stat.st_dev, stat.st_ino)
        rel = os.path.relpath(os.path.realpath(str(tag_file)), os.path.realpath(str(ollama_dir)))
    except (OSError, ValueError):
        root, rel = None, os.path.realpath(str(tag_file))
    key = (root, os.path.normcase(rel))
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
    with _materialization_lock(tag_file, ollama_dir):
        return _materialize_ollama_model_ref_unlocked(tag_file, ollama_dir)


def acquire_ollama_model_ref(ref: str) -> OllamaModelLease:
    """Materialize and keep the pair stable until the caller releases the lease."""
    tag_file, ollama_dir = _validated_ollama_manifest_location(ref)
    lock = _materialization_lock(tag_file, ollama_dir)
    lock.acquire()
    try:
        return OllamaModelLease(_materialize_ollama_model_ref_unlocked(tag_file, ollama_dir), lock)
    except BaseException:
        lock.release()
        raise
