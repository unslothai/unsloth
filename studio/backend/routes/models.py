# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model management API routes."""

import asyncio
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import structlog
from loggers import get_logger
from utils.utils import log_and_http_error

import re as _re

_VALID_REPO_ID = _re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


class CachedModelRepo(BaseModel):
    repo_id: str
    size_bytes: int
    last_modified: Optional[float] = None


class CachedModelsResponse(BaseModel):
    cached: List[CachedModelRepo]


def _is_valid_repo_id(repo_id: str) -> bool:
    return bool(_VALID_REPO_ID.fullmatch(repo_id))


def _normalize_hf_token(hf_token) -> Optional[str]:
    if not isinstance(hf_token, str):
        return None
    token = hf_token.strip()
    return token or None


def _safe_is_dir(path) -> bool:
    """``Path.is_dir()`` returning ``False`` instead of raising.

    Python >= 3.12 propagates ``PermissionError`` from ``is_dir()``;
    folder-scan endpoints probe system locations (e.g. root-owned
    ``/usr/share/ollama``) and must treat un-stat-able paths as "not a
    directory", never 500.
    """
    try:
        return Path(path).is_dir()
    except OSError:
        return False


# Shared with the hub inventory scans; keep the private aliases so existing
# importers (core.inference.local_model_resolver, tests) stay valid.
from utils.hidden_models import (
    _safe_resolve,
    is_hidden_model as _is_hidden_model,
)


backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from auth.authentication import get_current_subject
from hub.dependencies import get_hf_token

try:
    from utils.models import (
        scan_trained_models,
        scan_exported_models,
        get_base_model_from_checkpoint,
        load_model_defaults,
        get_base_model_from_lora,
        is_vision_model,
        is_embedding_model,
        scan_checkpoints,
        list_gguf_variants,
        ModelConfig,
    )
    from utils.models.model_config import (
        _pick_best_gguf,
        _extract_quant_label,
        _is_big_endian_gguf_path,
        is_audio_input_type,
    )
    from core.inference import get_inference_backend
    from utils.paths import (
        is_local_path,
        outputs_root,
        exports_root,
        resolve_cached_repo_id_case,
        resolve_output_dir,
        resolve_export_dir,
    )
except ImportError:
    # Fallback: import from parent directory.
    parent_backend = backend_path.parent / "backend"
    if str(parent_backend) not in sys.path:
        sys.path.insert(0, str(parent_backend))
    from utils.models import (
        scan_trained_models,
        scan_exported_models,
        get_base_model_from_checkpoint,
        load_model_defaults,
        get_base_model_from_lora,
        is_vision_model,
        is_embedding_model,
        scan_checkpoints,
        list_gguf_variants,
        ModelConfig,
    )
    from utils.models.model_config import (
        _pick_best_gguf,
        _extract_quant_label,
        _is_big_endian_gguf_path,
        is_audio_input_type,
    )
    from core.inference import get_inference_backend
    from utils.paths import (
        is_local_path,
        outputs_root,
        exports_root,
        resolve_cached_repo_id_case,
        resolve_output_dir,
        resolve_export_dir,
    )

from models import (
    CheckpointInfo,
    CheckpointListResponse,
    LocalModelInfo,
    LocalModelListResponse,
    ModelCheckpoints,
    ModelDetails,
    LoRAScanResponse,
    LoRAInfo,
    ModelListResponse,
)
from models.models import (
    BrowseEntry,
    BrowseFoldersResponse,
    ExportSizeResponse,
    GgufVariantDetail,
    GgufVariantsResponse,
    ModelType,
    ScanFolderInfo,
    AddScanFolderRequest,
)
from models.responses import (
    LoRABaseModelResponse,
    VisionCheckResponse,
    EmbeddingCheckResponse,
)

router = APIRouter()
logger = get_logger(__name__)


def derive_model_type(
    is_vision: bool,
    audio_type: Optional[str],
    is_embedding: bool = False,
) -> ModelType:
    """Collapse individual capability flags into a single model modality string."""
    if is_embedding:
        return "embeddings"
    if audio_type is not None:
        return "audio"
    if is_vision:
        return "vision"
    return "text"


def _resolve_hf_cache_dir() -> Path:
    """Resolve local HF cache root used by hub downloads."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def _is_model_directory(d: Path) -> bool:
    """Return ``True`` when *d* looks like a model directory.

    Requires both a config (``config.json``/``adapter_config.json``) and
    weight files. Excludes ``mmproj`` GGUFs (vision projectors) and
    non-weight ``.bin`` files (``tokenizer.bin`` etc.) to avoid false
    positives.
    """

    def _is_weight_file(f: Path) -> bool:
        suffix = f.suffix.lower()
        if suffix == ".safetensors":
            return True
        if suffix == ".gguf":
            return "mmproj" not in f.name.lower()
        if suffix == ".bin":
            name = f.name.lower()
            return (
                name.startswith("pytorch_model")
                or name.startswith("model")
                or name.startswith("adapter_model")
                or name.startswith("consolidated")
            )
        return False

    try:
        has_config = (d / "config.json").exists() or (d / "adapter_config.json").exists()
        if not has_config:
            return False
        return any(_is_weight_file(f) for f in d.iterdir() if f.is_file())
    except OSError:
        return False


# Weight ``.bin`` files the local scanners accept (PyTorch checkpoints), as
# opposed to companion ``.bin`` files like ``tokenizer.bin``. Mirrors the gating
# in ``_is_weight_file`` so every weight check classifies the same files.
_WEIGHT_BIN_PREFIXES = ("pytorch_model", "model", "adapter_model", "consolidated")


def _is_weight_bin(name: str) -> bool:
    low = name.lower()
    return low.endswith(".bin") and low.startswith(_WEIGHT_BIN_PREFIXES)


def _has_non_gguf_weights(path: Path) -> bool:
    """True if *path* holds non-GGUF weight files (``.safetensors`` or a weight
    ``.bin``), ignoring companion ``.bin`` files such as ``tokenizer.bin`` so a
    GGUF-only folder is not misread as a plain checkpoint."""
    try:
        if any(path.glob("*.safetensors")):
            return True
        return any(_is_weight_bin(f.name) for f in path.glob("*.bin"))
    except OSError:
        return False


def _scan_models_dir(models_dir: Path, *, limit: int | None = None) -> List[LocalModelInfo]:
    if not models_dir.exists() or not models_dir.is_dir():
        return []

    _is_self_model = _is_model_directory(models_dir)

    if _is_self_model:
        try:
            updated_at = models_dir.stat().st_mtime
        except OSError:
            updated_at = None
        return [
            LocalModelInfo(
                id = str(models_dir),
                display_name = models_dir.name,
                path = str(models_dir),
                source = "models_dir",
                model_format = _dir_model_format(models_dir),
                updated_at = updated_at,
            ),
        ]

    found: List[LocalModelInfo] = []
    for child in models_dir.iterdir():
        if limit is not None and len(found) >= limit:
            break
        try:
            if not child.is_dir():
                continue
            has_gguf = any(child.glob("*.gguf"))
            has_non_gguf_weights = _has_non_gguf_weights(child)
            has_config = (child / "config.json").exists() or (
                child / "adapter_config.json"
            ).exists()
            has_model_files = has_gguf or has_non_gguf_weights or has_config
        except OSError:
            # Skip unreadable children rather than failing the scan.
            continue
        if not has_model_files:
            continue
        try:
            updated_at = child.stat().st_mtime
        except OSError:
            updated_at = None
        # A folder whose only weights are .gguf is GGUF-format even when it also
        # ships a config.json (common for HF GGUF repos); such folders often lack
        # a -GGUF suffix, so surface the format for the UI's GGUF classification.
        model_format = "gguf" if has_gguf and not has_non_gguf_weights else None
        found.append(
            LocalModelInfo(
                id = str(child),
                display_name = child.name,
                path = str(child),
                source = "models_dir",
                model_format = model_format,
                updated_at = updated_at,
            ),
        )
    # Also scan standalone .gguf files in the models directory.
    if limit is None or len(found) < limit:
        for gguf_file in models_dir.glob("*.gguf"):
            if limit is not None and len(found) >= limit:
                break
            if gguf_file.is_file():
                try:
                    updated_at = gguf_file.stat().st_mtime
                except OSError:
                    updated_at = None
                found.append(
                    LocalModelInfo(
                        id = str(gguf_file),
                        display_name = gguf_file.stem,
                        path = str(gguf_file),
                        source = "models_dir",
                        model_format = "gguf",
                        updated_at = updated_at,
                    ),
                )

    return found


def _scan_hf_cache(cache_dir: Path) -> List[LocalModelInfo]:
    if not cache_dir.exists() or not cache_dir.is_dir():
        return []

    found: List[LocalModelInfo] = []
    for repo_dir in cache_dir.glob("models--*"):
        if not repo_dir.is_dir():
            continue

        repo_name = repo_dir.name[len("models--") :]
        if not repo_name:
            continue
        model_id = repo_name.replace("--", "/")

        try:
            updated_at = repo_dir.stat().st_mtime
        except OSError:
            updated_at = None

        found.append(
            LocalModelInfo(
                id = model_id,
                model_id = model_id,
                display_name = model_id.split("/")[-1],
                path = str(repo_dir),
                source = "hf_cache",
                updated_at = updated_at,
            ),
        )
    return found


def _dir_model_format(path: Path) -> Optional[str]:
    """Return ``"gguf"`` for a directory whose only weights are ``.gguf`` files.

    LM Studio and custom GGUF folders frequently lack a ``-GGUF`` name suffix,
    so the UI relies on this hint to route them through the GGUF load path
    rather than treating them as plain local checkpoints.
    """
    try:
        if not any(path.glob("*.gguf")):
            return None
        return None if _has_non_gguf_weights(path) else "gguf"
    except OSError:
        return None


def _scan_lmstudio_dir(lm_dir: Path) -> List[LocalModelInfo]:
    """Scan an LM Studio models directory for model files.

    LM Studio uses a ``publisher/model-name`` folder structure with GGUF
    files, or standalone GGUF files at the top level.
    """
    if not lm_dir.exists() or not lm_dir.is_dir():
        return []

    # If lm_dir is itself a model directory (not a publisher structure),
    # return it as a single entry rather than skipping it silently.
    if _is_model_directory(lm_dir):
        try:
            updated_at = lm_dir.stat().st_mtime
        except OSError:
            updated_at = None
        return [
            LocalModelInfo(
                id = str(lm_dir),
                display_name = lm_dir.name,
                path = str(lm_dir),
                source = "lmstudio",
                model_format = _dir_model_format(lm_dir),
                updated_at = updated_at,
            ),
        ]

    found: List[LocalModelInfo] = []
    for child in lm_dir.iterdir():
        try:
            if not child.is_dir():
                if child.suffix == ".gguf" and child.is_file():
                    try:
                        updated_at = child.stat().st_mtime
                    except OSError:
                        updated_at = None
                    found.append(
                        LocalModelInfo(
                            id = str(child),
                            display_name = child.stem,
                            path = str(child),
                            source = "lmstudio",
                            model_format = "gguf",
                            updated_at = updated_at,
                        ),
                    )
                continue

            # Surface a model-directory child directly instead of
            # descending into it as a publisher.
            if _is_model_directory(child):
                try:
                    updated_at = child.stat().st_mtime
                except OSError:
                    updated_at = None
                found.append(
                    LocalModelInfo(
                        id = str(child),
                        display_name = child.name,
                        path = str(child),
                        source = "lmstudio",
                        model_format = _dir_model_format(child),
                        updated_at = updated_at,
                    ),
                )
                continue

            # child is a publisher directory; scan its subdirectories.
            for model_dir in child.iterdir():
                try:
                    if model_dir.is_dir():
                        has_model = (
                            any(model_dir.glob("*.gguf"))
                            or (model_dir / "config.json").exists()
                            or any(model_dir.glob("*.safetensors"))
                        )
                        if not has_model:
                            continue
                        model_id = f"{child.name}/{model_dir.name}"
                        try:
                            updated_at = model_dir.stat().st_mtime
                        except OSError:
                            updated_at = None
                        found.append(
                            LocalModelInfo(
                                id = str(model_dir),
                                model_id = model_id,
                                display_name = model_dir.name,
                                path = str(model_dir),
                                source = "lmstudio",
                                model_format = _dir_model_format(model_dir),
                                updated_at = updated_at,
                            ),
                        )
                    elif model_dir.suffix == ".gguf" and model_dir.is_file():
                        try:
                            updated_at = model_dir.stat().st_mtime
                        except OSError:
                            updated_at = None
                        found.append(
                            LocalModelInfo(
                                id = str(model_dir),
                                model_id = f"{child.name}/{model_dir.stem}",
                                display_name = model_dir.stem,
                                path = str(model_dir),
                                source = "lmstudio",
                                model_format = "gguf",
                                updated_at = updated_at,
                            ),
                        )
                except OSError:
                    continue
        except OSError:
            continue
    return found


def _ollama_links_dir(ollama_dir: Path) -> Optional[Path]:
    """Return a writable directory for Ollama ``.gguf`` symlinks.

    Prefers ``<ollama_dir>/.studio_links/`` so links sit next to their
    blobs; falls back to a per-ollama-dir namespace under Unsloth's cache
    when the models dir is read-only (common for system installs).
    """
    from utils.paths.storage_roots import cache_root

    primary = ollama_dir / ".studio_links"
    try:
        primary.mkdir(exist_ok = True)
        return primary
    except OSError as e:
        logger.debug(
            "Ollama dir %s not writable for .studio_links (%s); falling back to Unsloth cache",
            ollama_dir,
            e,
        )

    # Fallback: namespace by a hash of ollama_dir so two roots don't
    # collide. Cache path, not a security boundary.
    try:
        digest = hashlib.sha256(str(ollama_dir.resolve()).encode()).hexdigest()[:12]
    except OSError:
        digest = "default"
    fallback = cache_root() / "ollama_links" / digest
    try:
        fallback.mkdir(parents = True, exist_ok = True)
        return fallback
    except OSError as e:
        logger.warning(
            "Could not create Ollama symlink cache at %s: %s",
            fallback,
            e,
        )
        return None


def _scan_ollama_dir(ollama_dir: Path, limit: Optional[int] = None) -> List[LocalModelInfo]:
    """Scan an Ollama models directory for downloaded models.

    Ollama uses a content-addressable layout
    (``manifests/<host>/<namespace>/<model>/<tag>`` + ``blobs/sha256-...``);
    we ``rglob`` all manifests so every layout depth is found. Each
    manifest is JSON with a ``layers`` array: the
    ``application/vnd.ollama.image.model`` layer holds the GGUF weights
    and ``...image.projector`` is the vision adapter.

    Ollama blobs lack the ``.gguf`` extension the loading pipeline
    requires, so we create ``.gguf``-named links to them (one subdir per
    model, keyed by a short hash of the manifest path, so
    ``detect_mmproj_file`` only sees that model's projector). Links are
    symlinks when possible, else hardlinks; the link dir is
    ``.studio_links/`` when writable, else Unsloth's cache.
    """
    manifests_root = ollama_dir / "manifests"
    if not manifests_root.is_dir():
        return []

    found: List[LocalModelInfo] = []
    blobs_dir = ollama_dir / "blobs"
    links_root = _ollama_links_dir(ollama_dir)
    if links_root is None:
        logger.warning(
            "Skipping Ollama scan for %s: no writable location for .gguf links",
            ollama_dir,
        )
        return []

    def _make_link(link_dir: Path, link_name: str, target: Path) -> Optional[str]:
        """Create a .gguf-named link to an Ollama blob.

        Tries symlink, then hardlink; skips the model if neither works
        (a multi-GB copy in a sync request would block the backend).
        Idempotent: skips recreation when a valid link already exists.
        """
        link_dir.mkdir(parents = True, exist_ok = True)
        link_path = link_dir / link_name
        resolved = target.resolve()

        # Skip if the link already points at the same blob. Use samefile
        # only; size checks can reuse stale links after `ollama pull`.
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

    try:
        for tag_file in manifests_root.rglob("*"):
            if not tag_file.is_file():
                continue

            rel = tag_file.relative_to(manifests_root)
            parts = rel.parts
            if len(parts) < 3:
                continue

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
                continue

            display = f"{repo_name}:{tag}"

            manifest_key = rel.as_posix()
            stem_hash = hashlib.sha256(manifest_key.encode()).hexdigest()[:10]

            try:
                manifest = json.loads(tag_file.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.debug(
                    "Skipping unreadable/invalid Ollama manifest %s: %s",
                    tag_file,
                    e,
                )
                continue

            config_digest = manifest.get("config", {}).get("digest", "")
            model_type = ""
            file_type = ""
            if config_digest and blobs_dir.is_dir():
                config_blob = blobs_dir / config_digest.replace(":", "-")
                if config_blob.is_file():
                    try:
                        cfg = json.loads(config_blob.read_text())
                        model_type = cfg.get("model_type", "")
                        file_type = cfg.get("file_type", "")
                    except (json.JSONDecodeError, OSError) as e:
                        logger.debug(
                            "Could not parse Ollama config blob %s: %s",
                            config_blob,
                            e,
                        )

            model_link_dir = links_root / stem_hash

            gguf_link_path: Optional[str] = None
            quant = f"-{file_type}" if file_type else ""
            safe_name = repo_name.replace("/", "-")
            for layer in manifest.get("layers") or []:
                media = layer.get("mediaType", "")
                digest = layer.get("digest", "")
                if not digest:
                    continue

                if media == "application/vnd.ollama.image.model":
                    candidate = blobs_dir / digest.replace(":", "-")
                    if candidate.is_file():
                        link_name = f"{safe_name}-{tag}{quant}.gguf"
                        gguf_link_path = _make_link(model_link_dir, link_name, candidate)

                elif media == "application/vnd.ollama.image.projector":
                    candidate = blobs_dir / digest.replace(":", "-")
                    if candidate.is_file():
                        mmproj_name = f"{safe_name}-{tag}-mmproj.gguf"
                        _make_link(model_link_dir, mmproj_name, candidate)

            if not gguf_link_path:
                continue

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

            found.append(
                LocalModelInfo(
                    id = gguf_link_path,
                    model_id = f"ollama/{repo_name}:{tag}",
                    display_name = display + suffix,
                    path = gguf_link_path,
                    source = "custom",
                    updated_at = updated_at,
                ),
            )
            if limit is not None and len(found) >= limit:
                return found
    except OSError as e:
        logger.warning("Error scanning Ollama directory %s: %s", ollama_dir, e)
    return found


def collect_local_models(models_root: Path) -> List[LocalModelInfo]:
    """Scan ``models_root``, the HF caches, LM Studio dirs, and user scan folders,
    returning a deduplicated, hidden-filtered list of discovered local models.

    Shared by ``GET /models/local`` (the model picker) and the OpenAI-compatible
    catalog (``GET /v1/models``) so the UI and the API never drift. ``models_root``
    must already be validated/trusted by the caller.
    """
    from storage.studio_db import list_scan_folders
    from utils.paths import (
        hf_default_cache_dir,
        legacy_hf_cache_dir,
        lmstudio_model_dirs,
    )

    hf_cache_dir = _resolve_hf_cache_dir()
    legacy_hf = legacy_hf_cache_dir()
    hf_default = hf_default_cache_dir()
    lm_dirs = lmstudio_model_dirs()

    local_models = _scan_models_dir(models_root) + _scan_hf_cache(hf_cache_dir)

    # Resolve once; an inaccessible aux cache must skip that scan, not 500.
    hf_cache_real = _safe_resolve(hf_cache_dir)
    legacy_real = _safe_resolve(legacy_hf)
    default_real = _safe_resolve(hf_default)

    # Scan legacy Unsloth HF cache for backward compatibility.
    if _safe_is_dir(legacy_hf) and legacy_real != hf_cache_real:
        local_models += _scan_hf_cache(legacy_hf)

    # Scan HF system default cache (may differ under env overrides).
    if _safe_is_dir(hf_default) and default_real != hf_cache_real and default_real != legacy_real:
        local_models += _scan_hf_cache(hf_default)

    # Scan LM Studio directories.
    for lm_dir in lm_dirs:
        local_models += _scan_lmstudio_dir(lm_dir)

    # Scan user-added custom folders (per-folder cap).
    _MAX_MODELS_PER_FOLDER = 200
    try:
        custom_folders = list_scan_folders()
    except Exception as e:
        logger.warning("Could not load custom scan folders: %s", e)
        custom_folders = []
    for folder in custom_folders:
        folder_path = Path(folder["path"])
        try:
            # Filter Ollama .studio_links/ from generic scanners to
            # avoid duplicates and leaking internal paths into the UI.
            _generic = [
                m
                for m in (
                    _scan_models_dir(folder_path, limit = _MAX_MODELS_PER_FOLDER)
                    + _scan_hf_cache(folder_path)
                    + _scan_lmstudio_dir(folder_path)
                )
                if not any(p in (".studio_links", "ollama_links") for p in Path(m.path).parts)
            ]
            custom_models = _generic
            if len(custom_models) < _MAX_MODELS_PER_FOLDER:
                custom_models += _scan_ollama_dir(
                    folder_path,
                    limit = _MAX_MODELS_PER_FOLDER - len(custom_models),
                )
        except OSError as e:
            logger.warning("Skipping unreadable scan folder %s: %s", folder_path, e)
            continue
        local_models += [m.model_copy(update = {"source": "custom"}) for m in custom_models]

    # Deduplicate, but always keep custom folder entries (keyed by
    # (id, source)) so they show in the "Custom Folders" UI section
    # even when the model is also in the HF cache.
    deduped: dict[str, LocalModelInfo] = {}
    for model in local_models:
        key = f"{model.id}\x00custom" if model.source == "custom" else model.id
        if key not in deduped:
            deduped[key] = model

    models = sorted(
        deduped.values(),
        key = lambda item: item.updated_at or 0,
        reverse = True,
    )
    return [m for m in models if not _is_hidden_model(m.id, m.model_id, m.path)]


@router.get("/local", response_model = LocalModelListResponse)
async def list_local_models(
    models_dir: str = Query(
        default = "./models", description = "Directory to scan for local model folders"
    ),
    current_subject: str = Depends(get_current_subject),
):
    """List local model candidates from the models dir, HF caches, and LM Studio dirs."""
    from utils.paths import (
        legacy_hf_cache_dir,
        hf_default_cache_dir,
        lmstudio_model_dirs,
    )

    # Resolve all scan directories up front.
    hf_cache_dir = _resolve_hf_cache_dir()
    legacy_hf = legacy_hf_cache_dir()
    hf_default = hf_default_cache_dir()
    lm_dirs = lmstudio_model_dirs()

    # Validate models_dir against an allowlist of trusted dirs. Only the
    # trusted Path objects are used for FS access; the user string is
    # used for matching only, never for path construction.
    allowed_roots: list[Path] = [Path("./models").resolve(), hf_cache_dir]
    if _safe_is_dir(legacy_hf):
        allowed_roots.append(legacy_hf)
    if _safe_is_dir(hf_default):
        allowed_roots.append(hf_default)
    try:
        from utils.paths import studio_root, outputs_root
        allowed_roots.extend([studio_root(), outputs_root()])
    except Exception:
        pass

    requested = os.path.realpath(os.path.expanduser(models_dir))
    models_root = None
    for root in allowed_roots:
        root_str = os.path.realpath(str(root))
        if requested == root_str or requested.startswith(root_str + os.sep):
            models_root = root  # trusted root, not the user-supplied path
            break
    if models_root is None:
        raise HTTPException(
            status_code = 403,
            detail = "Directory not allowed",
        )

    try:
        models = collect_local_models(models_root)

        return LocalModelListResponse(
            models_dir = str(models_root),
            hf_cache_dir = str(hf_cache_dir),
            lmstudio_dirs = [str(d) for d in lm_dirs],
            models = models,
        )
    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to list local models",
            event = "models.list_local_models_failed",
            log = logger,
        )


@router.get("/scan-folders")
async def get_scan_folders(current_subject: str = Depends(get_current_subject)):
    """List all registered custom model scan folders."""
    from storage.studio_db import list_scan_folders
    return {"folders": list_scan_folders()}


@router.post("/scan-folders", response_model = ScanFolderInfo, status_code = 201)
async def add_scan_folder_endpoint(
    body: AddScanFolderRequest, current_subject: str = Depends(get_current_subject)
):
    """Register a new directory to scan for local models."""
    from storage.studio_db import add_scan_folder

    try:
        folder = add_scan_folder(body.path)
    except ValueError as e:
        logger.warning("Scan folder rejected: %s (path=%s)", e, body.path)
        # Forward the curated, path-free validation message.
        rejection_message = str(e)
        raise HTTPException(status_code = 400, detail = rejection_message)
    logger.info("Scan folder added: %s", folder.get("path"))
    return folder


@router.delete("/scan-folders/{folder_id}")
async def remove_scan_folder_endpoint(
    folder_id: int, current_subject: str = Depends(get_current_subject)
):
    """Remove a registered custom scan folder."""
    from storage.studio_db import remove_scan_folder

    remove_scan_folder(folder_id)
    logger.info("Scan folder removed: id=%s", folder_id)
    return {"ok": True}


def _dir_has_downloaded_model(directory: Path, max_entries: int = 4000) -> bool:
    """True if *directory* actually holds a downloaded model.

    Recommended-folder chips should only appear once the well-known dir
    has real weights, not just an empty LM Studio/Ollama scaffold. Two
    layouts: a GGUF/safetensors/PyTorch-bin weight file anywhere in the
    tree (LM Studio, plain dirs) or the Ollama content-addressable store
    (a non-empty ``manifests/`` beside ``blobs/``, whose blobs carry no
    extension). Weight detection mirrors the local scanner so a folder the
    chip leads to is one the scanner would actually surface a model from.
    Bounded by *max_entries* so a huge tree can't stall the request.
    """
    # Ollama layout: each manifest is JSON referencing content-addressable
    # blobs. A manifest file alone is not enough -- a failed or pruned pull
    # leaves the manifest behind with its model blob missing, so we resolve the
    # ``application/vnd.ollama.image.model`` layer to an on-disk blob before
    # counting it, mirroring _scan_ollama_dir (which only surfaces a model once
    # its blob resolves). Otherwise the chip leads to an empty picker.
    visited = 0
    manifests = directory / "manifests"
    blobs = directory / "blobs"
    try:
        if _safe_is_dir(manifests) and _safe_is_dir(blobs):
            for m in manifests.rglob("*"):
                visited += 1
                if visited > max_entries:
                    break
                if not m.is_file():
                    continue
                try:
                    manifest = json.loads(m.read_text())
                except (json.JSONDecodeError, OSError, ValueError):
                    continue
                for layer in manifest.get("layers") or []:
                    if layer.get("mediaType") != "application/vnd.ollama.image.model":
                        continue
                    digest = layer.get("digest", "")
                    if digest and (blobs / digest.replace(":", "-")).is_file():
                        return True
    except OSError:
        pass
    # Generic weights: any GGUF/safetensors in a bounded BFS that skips hidden
    # directories (``.git``/``.cache``/venvs). ``rglob`` walks in arbitrary order
    # and counts every entry, so a large hidden subtree could exhaust the budget
    # before reaching real weights and falsely report "no model".
    queue = [directory]
    visited = 0
    while queue:
        current = queue.pop(0)
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            visited += 1
            if visited > max_entries:
                return False
            try:
                if entry.is_dir():
                    if not entry.name.startswith("."):
                        queue.append(entry)
                else:
                    low = entry.name.lower()
                    if low.endswith((".gguf", ".safetensors")):
                        return True
                    # PyTorch checkpoints the scanner also accepts; gate by name
                    # so tokenizer.bin and friends don't count as weights.
                    if _is_weight_bin(entry.name):
                        return True
            except OSError:
                continue
    return False


@router.get("/recommended-folders")
async def get_recommended_folders(current_subject: str = Depends(get_current_subject)):
    """Return well-known model directories that hold a downloaded model.

    Lightweight alternative to ``browse-folders`` for the frontend's
    one-click "Recommended" chips. Only paths that actually contain
    weights are returned, so an empty LM Studio/Ollama scaffold no longer
    shows up as a suggestion.
    """
    from utils.paths.storage_roots import lmstudio_model_dirs

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
        if (
            _safe_is_dir(resolved)
            and os.access(resolved, os.R_OK | os.X_OK)
            and _dir_has_downloaded_model(Path(resolved))
        ):
            seen.add(resolved)
            folders.append(resolved)

    # LM Studio model directories.
    try:
        for p in lmstudio_model_dirs():
            _add(p)
    except Exception as e:
        logger.warning("Failed to scan for LM Studio model directories: %s", e)

    # Ollama model directories.
    ollama_env = os.environ.get("OLLAMA_MODELS")
    if ollama_env:
        _add(Path(ollama_env).expanduser())
    for candidate in (
        Path.home() / ".ollama" / "models",
        Path("/usr/share/ollama/.ollama/models"),
        Path("/var/lib/ollama/.ollama/models"),
    ):
        _add(candidate)

    return {"folders": folders}


# Max children to stat when checking if a directory "looks like" it
# holds models; keeps the browser snappy on huge dirs.
_BROWSE_MODEL_HINT_PROBE = 64
# Hard cap on subdirectory entries returned, so browsing ``/usr/lib``
# can't stat-storm the process or flood the client.
_BROWSE_ENTRY_CAP = 2000


def _count_model_files(directory: Path, cap: int = 200) -> int:
    """Count GGUF/safetensors files immediately inside *directory*.

    Surfaces a count-hint so the UI can mark a weights-only leaf dir as a
    valid "Use this folder" target. Bounded by *visited entries* (stops
    after ``cap``), so the hint never costs more than a bounded walk.
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
    """Return True if an immediate child signals a model: a
    GGUF/safetensors/config.json file or a ``models--*`` subdir (HF
    cache). Bounded by ``_BROWSE_MODEL_HINT_PROBE``."""
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
    """Bounded heuristic to flag dirs worth exploring in the browser.

    False negatives are fine (the real scanner is authoritative). Three
    signals, cheapest first: (1) name ``models--*`` (HF cache layout),
    (2) an immediate child weight/config file, (3) a grandchild with a
    direct signal (LM Studio / Ollama ``publisher/model`` layout, probing
    the first ``_BROWSE_MODEL_HINT_PROBE`` child dirs).
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


def _build_browse_allowlist(
    media_roots: Optional[list[Path]] = None, drive_roots: Optional[list[Path]] = None
) -> list[Path]:
    """Return the root directories the folder browser may walk.

    The same list seeds the sidebar suggestion chips, so chip targets are
    always reachable. Roots: HOME, resolved HF cache dirs, Unsloth's
    outputs/exports/studio root, registered scan folders, and well-known
    local-LLM dirs (LM Studio, Ollama, ``~/models``); each added only if
    it resolves to a real directory.

    *media_roots* / *drive_roots* let the caller pass already-probed
    removable-media and Windows drive roots so they aren't scanned again (a
    disconnected mapped drive can make each probe slow); probed here when ``None``.
    """
    from utils.paths import (
        hf_default_cache_dir,
        legacy_hf_cache_dir,
        well_known_model_dirs,
    )
    from utils.paths.external_media import (
        linux_run_media_mount_roots,
        windows_drive_roots,
    )
    from storage.studio_db import list_scan_folders

    candidates: list[Path] = []

    def _add(p: Optional[Path]) -> None:
        if p is None:
            return
        try:
            resolved = p.resolve()
        except OSError:
            return
        if _safe_is_dir(resolved):
            candidates.append(resolved)

    _add(Path.home())
    if media_roots is None:
        media_roots = linux_run_media_mount_roots()
    if drive_roots is None:
        drive_roots = windows_drive_roots()
    for p in media_roots:
        _add(p)
    for p in drive_roots:
        _add(p)
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
        from utils.paths import (
            exports_root,
            outputs_root,
            studio_root,
        )

        _add(studio_root())
        _add(outputs_root())
        _add(exports_root())
    except Exception as exc:  # noqa: BLE001 -- best-effort
        logger.debug("browse-folders: studio roots unavailable: %s", exc)
    try:
        for folder in list_scan_folders():
            p = folder.get("path")
            if p:
                _add(Path(p))
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
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def _is_path_inside_allowlist(target: Path, allowed_roots: list[Path]) -> bool:
    """True if *target* equals or descends from any allowed root.

    Uses ``os.path.realpath`` (symlinks can't escape the sandbox) and
    ``os.path.commonpath`` for a component-wise containment test, so a string
    prefix like ``/home/u`` never matches a sibling ``/home/user2`` while a
    drive root ``D:\\`` still contains ``D:\\models``. A Windows drive root
    authorizes its descendants, but a bare POSIX root ``/`` must NOT, else one
    ``/`` allowlist entry would authorize every absolute path. ``normcase`` keeps
    the drive-letter comparison case-insensitive, matching the hub browser.
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
            # Different drives / mixed absolute-relative: not contained.
            continue
    return False


def _normalize_browse_request_path(path: Optional[str]) -> str:
    """Normalize the browse request path lexically, without touching the FS."""
    if path is None or not path.strip():
        return os.path.normpath(str(Path.home()))

    expanded = os.path.expanduser(path.strip())
    if not os.path.isabs(expanded):
        expanded = os.path.join(str(Path.cwd()), expanded)
    return os.path.normpath(expanded)


def _browse_relative_parts(requested_path: str, root: Path) -> Optional[list[str]]:
    """Return validated relative path components under ``root``."""
    root_text = os.path.normpath(str(root))
    try:
        rel_text = os.path.relpath(requested_path, root_text)
    except ValueError:
        return None

    if rel_text == ".":
        return []
    if rel_text == ".." or rel_text.startswith(f"..{os.sep}"):
        return None

    parts = [part for part in rel_text.split(os.sep) if part not in ("", ".")]
    altsep = os.altsep
    for part in parts:
        if part == ".." or os.sep in part or (altsep and altsep in part):
            return None
    return parts


def _match_browse_child(current: Path, name: str) -> Optional[Path]:
    """Return the immediate child named ``name`` under ``current``."""
    try:
        for child in current.iterdir():
            if child.name == name:
                return child
    except PermissionError:
        raise HTTPException(
            status_code = 403,
            detail = f"Permission denied reading {current.name}",
        ) from None
    except OSError as exc:
        logger.warning("browse-folders: could not read %s: %s", current, exc, exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Could not read {os.path.basename(str(current))}",
        ) from exc
    return None


def _resolve_browse_target(path: Optional[str], allowed_roots: list[Path]) -> Path:
    """Resolve a requested browse path by walking from trusted allowlist roots."""
    from storage.studio_db import (
        contains_sensitive_path_component,
        is_denied_system_path,
    )

    requested_path = _normalize_browse_request_path(path)
    resolved_roots: list[Path] = []
    seen_roots: set[str] = set()
    for root in sorted(allowed_roots, key = lambda p: len(str(p)), reverse = True):
        try:
            resolved = root.resolve()
        except OSError:
            continue
        key = str(resolved)
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
                    detail = f"Path does not exist: {os.path.basename(requested_path)}",
                )
            try:
                resolved_child = child.resolve()
            except OSError as exc:
                logger.warning(
                    "browse-folders: invalid path component %r under %s: %s",
                    part,
                    current,
                    exc,
                    exc_info = True,
                )
                raise HTTPException(
                    status_code = 400,
                    detail = "Invalid path",
                ) from exc
            if not _is_path_inside_allowlist(resolved_child, resolved_roots):
                raise HTTPException(
                    status_code = 403,
                    detail = (
                        "Path is not in the browseable allowlist. Register it via "
                        "POST /api/models/scan-folders first, or pick a directory "
                        "under your home folder."
                    ),
                )
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
                detail = f"Not a directory: {os.path.basename(str(current))}",
            )
        return current

    raise HTTPException(
        status_code = 403,
        detail = (
            "Path is not in the browseable allowlist. Register it via "
            "POST /api/models/scan-folders first, or pick a directory "
            "under your home folder."
        ),
    )


# Sync (def, not async) so FastAPI runs the blocking filesystem I/O (drive
# probes, iterdir, realpath) in the threadpool: a disconnected mapped drive can
# make the probe wait out its timeout, which on the event loop would stall every
# other request. Matches the hub browse endpoint.
@router.get("/browse-folders", response_model = BrowseFoldersResponse)
def browse_folders(
    path: Optional[str] = Query(
        None,
        description = (
            "Directory to list. If omitted, defaults to the current user's "
            "home directory. Tilde (`~`) and relative paths are expanded. "
            "Must resolve inside the allowlist of browseable roots (HOME, "
            "HF cache, Unsloth dirs, registered scan folders, well-known "
            "model dirs)."
        ),
    ),
    show_hidden: bool = Query(
        False,
        description = "Include entries whose name starts with a dot",
    ),
    current_subject: str = Depends(get_current_subject),
):
    """List immediate subdirectories of *path* for the Custom Folders picker.

    Lets the frontend render a modal folder browser without a native OS
    dialog. Read-only: enumerates visible subdirectories so the user can
    click to a folder and hand the string to POST /api/models/scan-folders.

    Sandbox: bounded to :func:`_build_browse_allowlist`; paths outside it
    return 403, and symlinks are resolved via ``os.path.realpath`` first
    so traversal can't escape. Sorting: model-bearing dirs, then plain,
    then hidden (if ``show_hidden=true``).
    """
    from utils.paths import hf_default_cache_dir, well_known_model_dirs
    from utils.paths.external_media import (
        linux_run_media_mount_roots,
        windows_drive_roots,
    )
    from storage.studio_db import (
        contains_sensitive_path_component,
        is_denied_system_path,
        list_scan_folders,
    )

    # Probe removable-media and Windows drive roots once; the allowlist and
    # chips reuse the result so a disconnected mapped drive isn't scanned twice.
    media_roots = linux_run_media_mount_roots()
    drive_roots = windows_drive_roots()
    # Build once; the sandbox check and suggestion chips share it.
    allowed_roots = _build_browse_allowlist(media_roots, drive_roots)

    try:
        target = _resolve_browse_target(path, allowed_roots)
    except HTTPException:
        requested_path = _normalize_browse_request_path(path)
        if path is not None and path.strip():
            logger.warning(
                "browse-folders: rejected path %r (normalized=%s)",
                path,
                requested_path,
            )
        raise

    # Enumerate immediate subdirectories with a bounded cap.
    entries: list[BrowseEntry] = []
    truncated = False
    visited = 0
    try:
        it = target.iterdir()
    except PermissionError:
        raise HTTPException(
            status_code = 403,
            detail = f"Permission denied reading {os.path.basename(str(target))}",
        )
    except OSError as exc:
        logger.warning("browse-folders: could not read %s: %s", target, exc, exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Could not read {os.path.basename(str(target))}",
        )

    try:
        for child in it:
            # Bound by *visited*, not *appended*: a cap on len(entries)
            # would never trigger in dirs full of files. Counting visits
            # caps worst-case work at ``_BROWSE_ENTRY_CAP`` calls.
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
            if contains_sensitive_path_component(name):
                continue
            # Hide denied system dirs (C:\Windows, /etc, ...) so they don't
            # render as clickable rows that then 403 on descent. Resolve first
            # so a symlink/junction into a denied dir is hidden too, not just a literal name.
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
        # Rare: iterdir succeeded but reading an entry failed.
        logger.warning("browse-folders: partial enumeration of %s: %s", target, exc)

    # Model-bearing first, then plain, then hidden; case-insensitive
    # alphabetical within each bucket.
    def _sort_key(e: BrowseEntry) -> tuple[int, str]:
        bucket = 0 if e.has_models else (2 if e.hidden else 1)
        return (bucket, e.name.lower())

    entries.sort(key = _sort_key)

    # Parent is None at the filesystem root and when it would leave the
    # sandbox (else the up-row would 403 on click); users can still hop
    # to other allowed roots via the suggestion chips.
    parent: Optional[str]
    if target.parent == target or not _is_path_inside_allowlist(target.parent, allowed_roots):
        parent = None
    else:
        parent = str(target.parent)

    # Handy starting points for the quick-pick chips.
    suggestions: list[str] = []
    seen_sug: set[str] = set()

    def _add_sug(p: Optional[Path]) -> None:
        if p is None:
            return
        try:
            resolved = str(p.resolve())
        except OSError:
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

    # Home first -- the safe fallback when everything else is cold.
    _add_sug(Path.home())
    # Reuse the roots probed for the allowlist above (no second drive scan).
    for p in media_roots:
        _add_sug(p)
    # Windows drive roots so the user can hop between C:, D:, E: ...
    for p in drive_roots:
        _add_sug(p)
    # The HF cache root the process is actually using.
    try:
        _add_sug(hf_default_cache_dir())
    except Exception:
        pass
    # Already-registered scan folders (user-curated).
    try:
        for folder in list_scan_folders():
            _add_sug(Path(folder.get("path", "")))
    except Exception as exc:
        logger.debug("browse-folders: could not load scan folders: %s", exc)
    # Dirs used by other local-LLM tools (LM Studio, Ollama, ~/models);
    # the helper returns only existing paths, so no dead chips.
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


def _looks_like_mlx_repo(model_id: str) -> bool:
    """Name heuristic for unloaded models (mirrors the -GGUF suffix check);
    tokenized so MLX only matches as a whole name segment."""
    if model_id.lower().startswith("mlx-community/"):
        return True
    tail = model_id.split("/")[-1]
    return "MLX" in _re.split(r"[-_.]", tail.upper())


@router.get("/list")
async def list_models(current_subject: str = Depends(get_current_subject)):
    """List available models: default plus currently loaded."""
    try:
        inference_backend = get_inference_backend()

        default_models = inference_backend.default_models

        loaded_models = []
        for model_name, model_data in inference_backend.models.items():
            _is_vision = model_data.get("is_vision", False)
            _audio_type = model_data.get("audio_type")
            model_info = ModelDetails(
                id = model_name,
                name = model_name.split("/")[-1] if "/" in model_name else model_name,
                is_vision = _is_vision,
                is_lora = model_data.get("is_lora", False),
                is_mlx = model_data.get("is_mlx", False),
                is_audio = model_data.get("is_audio", False),
                audio_type = _audio_type,
                has_audio_input = model_data.get("has_audio_input", False),
                model_type = derive_model_type(_is_vision, _audio_type),
            )
            loaded_models.append(model_info)

        # Include active GGUF model (loaded via llama-server).
        from routes.inference import get_llama_cpp_backend

        llama_backend = get_llama_cpp_backend()
        if llama_backend.is_loaded and llama_backend.model_identifier:
            loaded_models.append(
                ModelDetails(
                    id = llama_backend.model_identifier,
                    name = llama_backend.model_identifier.split("/")[-1],
                    is_gguf = True,
                    is_vision = llama_backend.is_vision,
                    is_audio = getattr(llama_backend, "_is_audio", False),
                    audio_type = getattr(llama_backend, "_audio_type", None),
                )
            )

        # Combine default and loaded; prefer loaded entries for duplicate
        # ids so runtime flags survive.
        all_models = []
        seen_ids = set()
        loaded_by_id = {model_info.id: model_info for model_info in loaded_models}

        for model_id in default_models:
            if model_id not in seen_ids:
                model_info = loaded_by_id.get(model_id) or ModelDetails(
                    id = model_id,
                    name = model_id.split("/")[-1] if "/" in model_id else model_id,
                    is_gguf = model_id.upper().endswith("-GGUF"),
                    is_mlx = _looks_like_mlx_repo(model_id),
                )
                all_models.append(model_info)
                seen_ids.add(model_id)

        for model_info in loaded_models:
            if model_info.id not in seen_ids:
                all_models.append(model_info)
                seen_ids.add(model_info.id)

        return ModelListResponse(models = all_models, default_models = default_models)

    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to list models",
            event = "models.list_models_failed",
            log = logger,
        )


def _get_max_position_embeddings(config) -> Optional[int]:
    """Extract max_position_embeddings from a config, with text_config fallback."""
    if hasattr(config, "max_position_embeddings"):
        return config.max_position_embeddings
    if hasattr(config, "text_config") and hasattr(config.text_config, "max_position_embeddings"):
        return config.text_config.max_position_embeddings
    return None


def _get_model_size_bytes(model_name: str, hf_token: Optional[str] = None) -> Optional[int]:
    """Total size of model weight files from HF Hub."""
    try:
        from huggingface_hub import HfApi

        api = HfApi(token = hf_token)
        info = api.repo_info(model_name, repo_type = "model", token = hf_token)
        if not info.siblings:
            return None

        weight_exts = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
        total = 0
        for sibling in info.siblings:
            if sibling.rfilename and any(sibling.rfilename.endswith(ext) for ext in weight_exts):
                if sibling.size is not None:
                    total += sibling.size

        return total if total > 0 else None
    except Exception as e:
        logger.warning(f"Could not get model size for {model_name}: {e}")
        return None


@router.get("/config/{model_name:path}")
async def get_model_config(
    model_name: str,
    hf_token: Optional[str] = Query(None),
    current_subject: str = Depends(get_current_subject),
):
    """Get configuration for a specific model (wraps load_model_defaults)."""
    try:
        if not is_local_path(model_name):
            resolved = resolve_cached_repo_id_case(model_name)
            if resolved != model_name:
                logger.info(
                    "Using cached repo_id casing '%s' for requested '%s'",
                    resolved,
                    model_name,
                )
            model_name = resolved

        logger.info(f"Getting model config for: {model_name}")
        from utils.models.model_config import detect_audio_type

        config_dict = load_model_defaults(model_name)

        # Detect capabilities (HF token for gated models).
        is_vision = is_vision_model(model_name, hf_token = hf_token)
        is_embedding = is_embedding_model(model_name, hf_token = hf_token)
        audio_type = detect_audio_type(model_name, hf_token = hf_token)

        is_lora = False
        base_model = None
        max_position_embeddings = None
        try:
            model_config = ModelConfig.from_identifier(model_name)
            is_lora = model_config.is_lora
            base_model = model_config.base_model if is_lora else None
            max_position_embeddings = _get_max_position_embeddings(model_config)
        except Exception:
            pass

        # Fallback: read raw config.json (declarative fields only) -- a selection-time
        # metadata probe that must never execute a repo's auto_map Python.
        if max_position_embeddings is None:
            try:
                from utils.transformers_version import _load_config_json
                from types import SimpleNamespace

                _cfg = _load_config_json(model_name, hf_token = hf_token)
                if _cfg is not None:

                    def _to_ns(d):
                        if isinstance(d, dict):
                            return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
                        return d

                    max_position_embeddings = _get_max_position_embeddings(_to_ns(_cfg))
            except Exception:
                pass

        logger.info(
            f"Model config result for {model_name}: is_vision={is_vision}, is_embedding={is_embedding}, audio_type={audio_type}, is_lora={is_lora}, max_position_embeddings={max_position_embeddings}"
        )
        return ModelDetails(
            id = model_name,
            model_name = model_name,
            config = config_dict,
            is_vision = is_vision,
            is_embedding = is_embedding,
            is_lora = is_lora,
            is_audio = audio_type is not None,
            audio_type = audio_type,
            has_audio_input = is_audio_input_type(audio_type),
            model_type = derive_model_type(is_vision, audio_type, is_embedding),
            base_model = base_model,
            max_position_embeddings = max_position_embeddings,
            model_size_bytes = _get_model_size_bytes(model_name, hf_token),
        )

    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to get model config",
            event = "models.get_model_config_failed",
            log = logger,
        )


def _consent_provider(
    model_name: str,
    scanned_targets: List[str],
    external_refs: Optional[List[str]] = None,
) -> Optional[str]:
    """HF org for the consent dialog's `from "<provider>"` tag, or None.

    Returns the owner only for a single, non-local, canonical ``owner/repo`` id; a LoRA's
    extra base, a local path, or an external ``auto_map`` ref yields None so the dialog
    never misattributes scanned code.
    """
    if len(scanned_targets) != 1 or external_refs or is_local_path(model_name):
        return None
    parts = model_name.split("/")
    return parts[0] if len(parts) == 2 and all(parts) else None


@router.post("/remote-code-scan")
async def scan_model_remote_code(
    model_name: str = Body(..., embed = True),
    hf_token: Optional[str] = Body(None, embed = True),
    current_subject: str = Depends(get_current_subject),
):
    """Scan a model's ``auto_map`` custom code so the UI can show findings before
    the user enables ``trust_remote_code``. Code-free: reads ``config.json`` and
    statically scans the repo ``.py`` (never loads the model). Returns
    ``has_remote_code`` plus the severity-tagged findings + a pinning fingerprint.

    POST (not GET) so the ``hf_token`` for gated repos travels in the body and
    never lands in a URL, browser history, or access log.
    """
    try:
        from utils.security import preflight_remote_code_consent_for_targets

        if not is_local_path(model_name):
            model_name = resolve_cached_repo_id_case(model_name)
        # Scan the adapter AND the base together (a LoRA runs both repos' code; a pickle
        # can live in either), pinned by one combined fingerprint. Snapshot the primary's
        # cache state BEFORE resolving the base: for a remote adapter that resolve
        # downloads adapter_config.json, which would otherwise hide the adapter from
        # cleanup on decline. On error treat as pre-existing so a decline never deletes it.
        try:
            _primary_preexisting = is_local_path(model_name) or _repo_in_any_hf_cache(model_name)
        except Exception:
            _primary_preexisting = True
        security_targets = [model_name]
        try:
            from utils.models.model_config import get_base_model_from_lora_identifier

            # Resolve a LOCAL or REMOTE adapter's base so its code/weights are scanned too.
            _base = get_base_model_from_lora_identifier(model_name, hf_token)
            if _base:
                security_targets.append(_base)
        except Exception:
            pass
        security_targets = list(dict.fromkeys(security_targets))
        # Record every repo OUR scan is first to pull into the cache (adapter, base, and
        # external auto_map repos like owner/name--module.Class), so a decline purges
        # exactly what was downloaded. Computed BEFORE the preflight downloads, against
        # every cache the discard searches, so a repo the user already had is not deleted.
        from utils.security.remote_code_scan import external_auto_map_repos

        scan_created_repos: list = []
        _seen_created: set = set()

        def _mark_scan_created(repo: str, *, preexisting: Optional[bool] = None) -> None:
            if not repo or repo in _seen_created:
                return
            _seen_created.add(repo)
            try:
                already = (
                    preexisting
                    if preexisting is not None
                    else (is_local_path(repo) or _repo_in_any_hf_cache(repo))
                )
                if not already:
                    scan_created_repos.append(repo)
            except Exception:
                pass

        external_refs: list = []
        for _target in security_targets:
            # Use the pre-base-resolution snapshot for the primary (see above).
            _mark_scan_created(
                _target, preexisting = _primary_preexisting if _target == model_name else None
            )
            for _ext in external_auto_map_repos(_target, hf_token):
                external_refs.append(_ext)
                _mark_scan_created(_ext)
        decision = preflight_remote_code_consent_for_targets(
            security_targets, hf_token = hf_token, subject = current_subject
        )
        payload = decision.response_payload()
        payload["requires_trust_remote_code"] = decision.has_remote_code
        # Prior approval for the unchanged repo lets the dialog be skipped; the scan still
        # ran, so this is a real fingerprint match under the current ruleset.
        payload["already_approved"] = (
            decision.has_remote_code
            and not decision.blocked
            and decision.reason == "approved by fingerprint"
        )
        # created_by_scan = primary flag (older clients); scan_created_repos drives cleanup.
        payload["created_by_scan"] = model_name in scan_created_repos
        payload["scan_created_repos"] = scan_created_repos
        # Provider tag decided here, where locality/scan scope/external refs are known.
        payload["provider"] = _consent_provider(model_name, security_targets, external_refs)

        # Malware gate (metadata-only): surface HF-flagged unsafe files so the dialog can
        # hard-block. Orthogonal to remote code -- a poisoned pickle needs no auto_map.
        from utils.security import evaluate_file_security, security_load_subdirs

        unsafe_files: list = []
        security_blocked = False
        for _target in security_targets:
            _sec = evaluate_file_security(
                _target, hf_token = hf_token, load_subdirs = security_load_subdirs(_target, hf_token)
            )
            security_blocked = security_blocked or _sec.blocked
            unsafe_files.extend(_sec.unsafe_files)
        payload["unsafe_files"] = unsafe_files
        payload["security_blocked"] = security_blocked
        if security_blocked:
            # Non-approvable hard block: approvable False hides "Enable and continue", and
            # requires_trust_remote_code forces the dialog open even with no custom code.
            payload["approvable"] = False
            payload["requires_trust_remote_code"] = True
            payload["error_kind"] = "malware_blocked"
        return payload
    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to scan model remote code",
            event = "models.remote_code_scan_failed",
            log = logger,
        )


@router.post("/discard-remote-code")
async def discard_remote_code_download(
    model_name: str = Body(..., embed = True), current_subject: str = Depends(get_current_subject)
):
    """Purge a repo the consent scan downloaded after the user DECLINED its custom
    code, so untrusted code is not left on disk.

    Safety: only ever deletes a metadata-only cache entry the scan created. It
    refuses a local path (never touches user files), a currently-loaded model, and
    any repo that has weight files cached (``*.safetensors`` / ``*.bin`` /
    ``*.gguf``) -- i.e. a model the user actually downloaded. The frontend only
    calls this when the scan reported ``created_by_scan``.
    """
    if is_local_path(model_name):
        return {"deleted": False, "reason": "local"}
    if not _is_valid_repo_id(model_name):
        return {"deleted": False, "reason": "invalid"}

    # Never delete a model that is loaded for inference.
    try:
        from routes.inference import get_llama_cpp_backend
        llama_backend = get_llama_cpp_backend()
        if llama_backend.is_loaded and llama_backend.model_identifier:
            loaded = llama_backend.model_identifier.lower()
            if loaded == model_name.lower() or loaded.startswith(model_name.lower()):
                return {"deleted": False, "reason": "loaded"}
    except Exception:
        pass
    try:
        inference_backend = get_inference_backend()
        if inference_backend.active_model_name:
            active = inference_backend.active_model_name.lower()
            if active == model_name.lower() or active.startswith(model_name.lower()):
                return {"deleted": False, "reason": "loaded"}
    except Exception:
        pass

    _WEIGHTS = (
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".h5",
        ".msgpack",
        ".gguf",
        ".onnx",
        ".ckpt",
    )
    try:
        target_repo = None
        hf_cache = None
        for cache in _all_hf_cache_scans():
            for repo_info in cache.repos:
                if repo_info.repo_type != "model":
                    continue
                if repo_info.repo_id.lower() == model_name.lower():
                    target_repo, hf_cache = repo_info, cache
                    break
            if target_repo is not None:
                break

        if target_repo is None:
            return {"deleted": False, "reason": "not_cached"}

        # Hard guard: a repo with weights is a real model the user has -- leave it.
        for rev in target_repo.revisions:
            for f in rev.files:
                if f.file_name.lower().endswith(_WEIGHTS):
                    return {"deleted": False, "reason": "has_weights"}

        revision_hashes = [rev.commit_hash for rev in target_repo.revisions]
        if not revision_hashes:
            return {"deleted": False, "reason": "not_cached"}
        hf_cache.delete_revisions(*revision_hashes).execute()
        logger.info("Discarded declined remote-code download: %s", model_name)
        return {"deleted": True}
    except Exception as e:
        logger.warning("Could not discard remote-code download for %s: %s", model_name, e)
        return {"deleted": False, "reason": "error"}


@router.get("/loras")
async def scan_loras(
    outputs_dir: str = Query(
        default = str(outputs_root()), description = "Directory to scan for LoRA adapters"
    ),
    exports_dir: str = Query(
        default = str(exports_root()), description = "Directory to scan for exported models"
    ),
    current_subject: str = Depends(get_current_subject),
):
    """Scan for trained LoRA adapters and exported models.

    Returns training outputs (outputs_dir) and exported models
    (exports_dir) in one list, distinguished by the source field.
    """
    try:
        resolved_outputs_dir = str(resolve_output_dir(outputs_dir))
        resolved_exports_dir = str(resolve_export_dir(exports_dir))
        lora_list = []

        trained_models = scan_trained_models(outputs_dir = resolved_outputs_dir)
        for display_name, model_path, model_type in trained_models:
            base_model = get_base_model_from_checkpoint(model_path)
            lora_list.append(
                LoRAInfo(
                    display_name = display_name,
                    adapter_path = model_path,
                    base_model = base_model,
                    source = "training",
                    export_type = model_type,
                )
            )

        # Scan exported models (merged, LoRA, base — skips GGUF)
        exported = scan_exported_models(exports_dir = resolved_exports_dir)
        for display_name, model_path, export_type, base_model in exported:
            lora_list.append(
                LoRAInfo(
                    display_name = display_name,
                    adapter_path = model_path,
                    base_model = base_model,
                    source = "exported",
                    export_type = export_type,
                )
            )

        return LoRAScanResponse(loras = lora_list, outputs_dir = resolved_outputs_dir)

    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to scan LoRA adapters",
            event = "models.scan_loras_failed",
            log = logger,
        )


def _is_path_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _is_path_under_lexically(path: Path, root: Path) -> bool:
    """Check containment without resolving the final path's symlink."""
    try:
        absolute_path = Path(os.path.abspath(str(path)))
        absolute_root = Path(os.path.abspath(str(root)))
        absolute_path.relative_to(absolute_root)
        return True
    except ValueError:
        return False


def _loaded_model_matches_deleted_path(active_model: str, deleted_path: Path) -> bool:
    try:
        active = Path(active_model).expanduser().resolve()
        target = deleted_path.resolve()
        return active == target or (target.is_dir() and active.is_relative_to(target))
    except (OSError, RuntimeError, ValueError) as e:
        logger.debug(
            "Could not resolve loaded/deleted model paths; falling back to string comparison: %s",
            e,
        )
        active_lower = active_model.lower()
        target_lower = str(deleted_path).lower()
        return active_lower == target_lower or active_lower.startswith(f"{target_lower}{os.sep}")


def _loading_model_matches_deleted_path(loading_model: object, deleted_path: Path) -> bool:
    if not loading_model:
        return False
    return _loaded_model_matches_deleted_path(str(loading_model), deleted_path)


def _prune_empty_parents(start: Path, stop_at: Path) -> None:
    """Remove empty ancestors of ``start`` up to (not including) ``stop_at``.

    Used after deleting a checkpoint so the enclosing run dir doesn't
    linger as an empty entry in scan results.
    """
    try:
        stop_resolved = stop_at.resolve()
    except OSError:
        return
    parent = start.parent
    while True:
        try:
            parent_resolved = parent.resolve()
        except OSError:
            return
        if parent_resolved == stop_resolved:
            return
        try:
            parent_resolved.relative_to(stop_resolved)
        except ValueError:
            return
        try:
            parent.rmdir()
        except OSError:
            return
        parent = parent.parent


def _delete_gguf_variant_files(root: Path, variant: str) -> tuple[int, int]:
    deleted_count = 0
    deleted_bytes = 0
    for path in root.rglob("*"):
        if not path.is_file() or not _is_main_gguf_filename(path.name):
            continue
        if _extract_quant_label(path.name).lower() != variant.lower():
            continue
        try:
            deleted_bytes += path.stat().st_size
        except OSError:
            pass
        path.unlink()
        deleted_count += 1
    return deleted_count, deleted_bytes


@router.delete("/delete-finetuned")
async def delete_finetuned_model(
    model_path: str = Body(...),
    source: str = Body(...),
    export_type: Optional[str] = Body(None),
    gguf_variant: Optional[str] = Body(None),
    current_subject: str = Depends(get_current_subject),
):
    """Delete an Unsloth-trained or exported model from disk.

    Only paths under Unsloth's outputs/exports roots are accepted.
    Exported GGUF entries can delete one quant variant at a time.
    """
    if source not in {"training", "exported"}:
        raise HTTPException(
            status_code = 400,
            detail = "Only trained or exported Unsloth models can be deleted",
        )

    if not model_path or not model_path.strip():
        raise HTTPException(status_code = 400, detail = "model_path is required")

    if export_type == "gguf" and not gguf_variant:
        raise HTTPException(
            status_code = 400,
            detail = "gguf_variant is required when export_type is 'gguf'",
        )

    raw_path = Path(model_path).expanduser()
    if source == "training":
        target_path = raw_path
        allowed_root = outputs_root()
    else:
        allowed_root = exports_root()
        target_path = (
            raw_path.parent
            if export_type == "gguf" and raw_path.suffix.lower() == ".gguf"
            else raw_path
        )

    allowed_root = allowed_root.resolve()
    delete_path = Path(os.path.abspath(str(target_path)))
    delete_path_is_symlink = delete_path.is_symlink()

    if delete_path_is_symlink:
        if not _is_path_under_lexically(delete_path, allowed_root):
            raise HTTPException(
                status_code = 400,
                detail = "Model path is outside Unsloth storage",
            )
        if export_type == "gguf" and gguf_variant:
            target_path = delete_path.resolve()
            if not _is_path_under(target_path, allowed_root):
                raise HTTPException(
                    status_code = 400,
                    detail = "Model path is outside Unsloth storage",
                )
        else:
            target_path = delete_path
    else:
        target_path = target_path.resolve()

    should_check_resolved_path = not delete_path_is_symlink or (
        export_type == "gguf" and gguf_variant
    )
    if should_check_resolved_path and not _is_path_under(target_path, allowed_root):
        raise HTTPException(
            status_code = 400,
            detail = "Model path is outside Unsloth storage",
        )
    if target_path == allowed_root:
        raise HTTPException(
            status_code = 400,
            detail = "Refusing to delete storage root",
        )
    if not target_path.exists() and not target_path.is_symlink():
        raise HTTPException(status_code = 404, detail = "Model not found on disk")

    if source == "training":
        try:
            from core.training import get_training_backend
            training_backend = get_training_backend()
            if training_backend.is_training_active():
                raise HTTPException(
                    status_code = 409,
                    detail = "Cannot delete trained models while training is running",
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning("Could not check training status before delete: %s", e)
            raise HTTPException(
                status_code = 500,
                detail = "Could not verify training status before deleting",
            ) from e

    try:
        from routes.inference import get_llama_cpp_backend

        llama_backend = get_llama_cpp_backend()
        if (
            llama_backend.is_active
            and not llama_backend.is_loaded
            and llama_backend.model_identifier
            and _loaded_model_matches_deleted_path(
                llama_backend.model_identifier,
                target_path,
            )
            and (
                not gguf_variant
                or not llama_backend.hf_variant
                or llama_backend.hf_variant.lower() == gguf_variant.lower()
            )
        ):
            raise HTTPException(
                status_code = 409,
                detail = "Cannot delete a model while it is loading",
            )
        if (
            llama_backend.is_loaded
            and llama_backend.model_identifier
            and _loaded_model_matches_deleted_path(
                llama_backend.model_identifier,
                target_path,
            )
            and (
                not gguf_variant
                or not llama_backend.hf_variant
                or llama_backend.hf_variant.lower() == gguf_variant.lower()
            )
        ):
            raise HTTPException(
                status_code = 400,
                detail = "Unload the model before deleting",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Could not check llama.cpp loaded model before delete: %s", e)
        raise HTTPException(
            status_code = 503,
            detail = "Could not verify model load status before deleting",
        ) from e

    try:
        inference_backend = get_inference_backend()
        loading_models = getattr(inference_backend, "loading_models", set())
        if any(
            _loading_model_matches_deleted_path(loading_model, target_path)
            for loading_model in loading_models
        ):
            raise HTTPException(
                status_code = 409,
                detail = "Cannot delete a model while it is loading",
            )
        if inference_backend.active_model_name:
            if _loaded_model_matches_deleted_path(
                inference_backend.active_model_name,
                target_path,
            ):
                raise HTTPException(
                    status_code = 400,
                    detail = "Unload the model before deleting",
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Could not check inference backend loaded model before delete: %s", e)
        raise HTTPException(
            status_code = 503,
            detail = "Could not verify model load status before deleting",
        ) from e

    try:
        if export_type == "gguf" and gguf_variant:
            if not target_path.is_dir():
                raise HTTPException(
                    status_code = 400,
                    detail = "GGUF variant deletion requires an export directory",
                )
            deleted_count, deleted_bytes = _delete_gguf_variant_files(
                target_path,
                gguf_variant,
            )
            if deleted_count == 0:
                raise HTTPException(
                    status_code = 404,
                    detail = f"Variant {gguf_variant} not found on disk",
                )
            try:
                if not any(target_path.iterdir()):
                    target_path.rmdir()
                    _prune_empty_parents(target_path, allowed_root)
            except OSError:
                pass
            logger.info(
                "Deleted %s GGUF file(s) for exported model at %s variant %s (%0.1f MB freed)",
                deleted_count,
                target_path,
                gguf_variant,
                deleted_bytes / (1024 * 1024),
            )
            return {
                "status": "deleted",
                "path": str(target_path),
                "gguf_variant": gguf_variant,
            }

        if target_path.is_symlink() or target_path.is_file():
            target_path.unlink()
        else:
            shutil.rmtree(target_path)

        if target_path.exists() or target_path.is_symlink():
            raise HTTPException(
                status_code = 500,
                detail = "Deletion incomplete; some files could not be removed",
            )

        _prune_empty_parents(target_path, allowed_root)

        logger.info("Deleted fine-tuned model at %s", target_path)
        return {"status": "deleted", "path": str(target_path)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error deleting fine-tuned model %s: %s",
            target_path,
            e,
            exc_info = True,
        )
        raise HTTPException(
            status_code = 500,
            detail = "Failed to delete fine-tuned model",
        )


@router.get("/loras/{lora_path:path}/base-model", response_model = LoRABaseModelResponse)
async def get_lora_base_model(lora_path: str, current_subject: str = Depends(get_current_subject)):
    """
    Get the base model for a LoRA adapter.

    This endpoint wraps the backend get_base_model_from_lora function.
    """
    try:
        base_model = get_base_model_from_lora(lora_path)

        if base_model is None:
            raise HTTPException(
                status_code = 404,
                detail = f"Could not determine base model for LoRA: {lora_path}",
            )

        return LoRABaseModelResponse(
            lora_path = lora_path,
            base_model = base_model,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to get base model",
            event = "models.get_lora_base_model_failed",
            log = logger,
        )


@router.get("/check-vision/{model_name:path}", response_model = VisionCheckResponse)
async def check_vision_model(
    model_name: str,
    hf_token: Optional[str] = Query(None),
    current_subject: str = Depends(get_current_subject),
):
    """
    Check if a model is a vision model.

    This endpoint wraps the backend is_vision_model function.
    """
    try:
        logger.info(f"Checking if vision model: {model_name}")
        # Authenticate so a gated/private VLM classifies correctly (else 404 -> non-vision).
        is_vision = is_vision_model(model_name, hf_token = hf_token)

        logger.info(f"Vision check result for {model_name}: is_vision={is_vision}")
        return VisionCheckResponse(
            model_name = model_name,
            is_vision = is_vision,
        )

    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to check vision model",
            event = "models.check_vision_model_failed",
            log = logger,
        )


@router.get("/check-embedding/{model_name:path}", response_model = EmbeddingCheckResponse)
async def check_embedding_model(
    model_name: str,
    hf_token: Optional[str] = Query(None),
    current_subject: str = Depends(get_current_subject),
):
    """
    Check if a model is an embedding model.

    This endpoint wraps the backend is_embedding_model function.
    """
    try:
        logger.info(f"Checking if embedding model: {model_name}")
        is_embedding = is_embedding_model(model_name, hf_token = hf_token)

        logger.info(f"Embedding check result for {model_name}: is_embedding={is_embedding}")
        return EmbeddingCheckResponse(
            model_name = model_name,
            is_embedding = is_embedding,
        )

    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to check embedding model",
            event = "models.check_embedding_model_failed",
            log = logger,
        )


def _read_native_context_length(repo_id: str, is_local: bool) -> Optional[int]:
    """Native max context from a downloaded GGUF for this repo, or None.

    The value is identical across quants, so reading one non-mmproj shard's
    header is enough. Only resolves once a file is on disk. Never raises.
    """
    try:
        from utils.models.gguf_metadata import read_gguf_context_length
        if is_local:
            roots = [Path(repo_id)]
        else:
            from huggingface_hub import constants as hf_constants

            if not _is_valid_repo_id(repo_id):
                return None
            cache_dir = Path(hf_constants.HF_HUB_CACHE)
            target = f"models--{repo_id.replace('/', '--')}".lower()
            roots = [e for e in cache_dir.iterdir() if e.name.lower() == target]

        for root in roots:
            for f in _iter_gguf_paths(root):
                if _is_mmproj_filename(f.name):
                    continue
                n = read_gguf_context_length(str(f))
                if n:
                    return n
    except Exception:
        pass
    return None


def _resolve_quant_gguf(repo_id: str, quant: str, is_local: bool) -> tuple[Optional[str], int]:
    """Primary shard path and total weight bytes for a downloaded quant, or
    (None, 0). Metadata lives in shard 1, so the lexicographically first file of
    the matching quant is returned. Scoped to one snapshot to avoid summing the
    same quant across revisions; when several snapshots hold the quant the most
    complete one (largest total) wins so a partial revision can't shadow it.
    Mirrors list_local_gguf_variants: quant labels are read from the snapshot-
    relative path (so layouts like ``BF16/model.gguf`` resolve) and MTP drafter
    files are skipped (so a ``...-Q8_0-MTP.gguf`` drafter can't be picked as the
    Q8_0 weights). Never raises.
    """
    try:
        from utils.models.model_config import (
            _extract_quant_label,
            _is_big_endian_gguf_path,
            _is_mtp_drafter,
        )

        if is_local:
            roots = [Path(repo_id)]
        else:
            from huggingface_hub import constants as hf_constants

            if not _is_valid_repo_id(repo_id):
                return None, 0
            cache_dir = Path(hf_constants.HF_HUB_CACHE)
            target = f"models--{repo_id.replace('/', '--')}".lower()
            roots = []
            for entry in cache_dir.iterdir():
                if entry.name.lower() == target:
                    snaps = entry / "snapshots"
                    if snaps.is_dir():
                        roots.extend(s for s in snaps.iterdir() if s.is_dir())

        want = quant.lower().replace("-", "").replace("_", "")
        best_total = 0
        best_first: Optional[str] = None
        for root in roots:
            matches: list[tuple[str, Path]] = []
            total = 0
            for f in _iter_gguf_paths(root):
                if _is_mmproj_filename(f.name):
                    continue
                try:
                    rel = f.relative_to(root).as_posix()
                except ValueError:
                    rel = f.name
                if _is_mtp_drafter(rel):
                    continue
                q = _extract_quant_label(rel)
                if _is_big_endian_gguf_path(rel, q):
                    continue
                if q.lower().replace("-", "").replace("_", "") != want:
                    continue
                try:
                    total += f.stat().st_size
                except OSError:
                    continue
                matches.append((rel, f))
            # Prefer the most complete snapshot so a partial older revision can't
            # shadow a newer complete one and underestimate the weight bytes.
            if matches and total > best_total:
                matches.sort(key = lambda m: m[0])
                best_total = total
                best_first = str(matches[0][1])
        if best_first is not None:
            return best_first, best_total
    except Exception:
        pass
    return None, 0


@router.get("/kv-cache-estimate")
async def get_kv_cache_estimate(
    repo_id: str = Query(..., description = "HF repo ID or local path"),
    quant: str = Query(..., description = "Quantization label (e.g. Q4_K_M)"),
    n_ctx: int = Query(..., ge = 1, description = "Context length to size the KV cache for"),
    cache_type_kv: Optional[str] = Query(None, description = "KV cache dtype (e.g. q8_0)"),
    current_subject: str = Depends(get_current_subject),
):
    """Estimate KV cache + weight bytes for a downloaded GGUF at n_ctx.

    Powers the load dialog's "exceeds memory" warning using the same
    architecture-aware estimator as load. Best-effort: returns nulls when the
    metadata is unavailable so the UI simply shows no warning.
    """
    null = {"kv_bytes": None, "weights_bytes": None, "native_context": None}
    try:
        from utils.models.model_config import is_local_path

        is_local = is_local_path(repo_id)
        path, weights_bytes = _resolve_quant_gguf(repo_id, quant, is_local)
        if not path:
            return null

        from core.inference.llama_cpp import LlamaCppBackend

        be = LlamaCppBackend.__new__(LlamaCppBackend)
        for attr in (
            "_context_length",
            "_n_layers",
            "_n_kv_heads",
            "_n_heads",
            "_embedding_length",
            "_kv_key_length",
            "_kv_value_length",
            "_kv_lora_rank",
            "_sliding_window",
            "_sliding_window_pattern",
            "_ssm_inner_size",
            "_full_attention_interval",
            "_key_length_mla",
            "_n_kv_heads_by_layer",
            "_kv_key_length_swa",
            "_kv_value_length_swa",
            "_shared_kv_layers",
            "_nextn_predict_layers",
        ):
            setattr(be, attr, None)
        be._model_identifier = "kv-estimate"
        be._read_gguf_metadata(path)

        kv = be._estimate_kv_cache_bytes(n_ctx, cache_type_kv)
        return {
            "kv_bytes": int(kv) if kv else None,
            "weights_bytes": weights_bytes or None,
            "native_context": be._context_length,
        }
    except Exception as e:
        logger.debug(f"kv-cache-estimate failed for '{repo_id}' {quant}: {e}")
        return null


@router.get("/gguf-variants", response_model = GgufVariantsResponse)
async def get_gguf_variants(
    repo_id: str = Query(
        ..., description = "HuggingFace repo ID (e.g. 'unsloth/gemma-3-4b-it-GGUF')"
    ),
    hf_token: Optional[str] = Query(None, description = "HuggingFace token for private repos"),
    hf_token_header: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    """List GGUF quantization variants for a HF repo or local directory."""
    try:
        hf_token = _normalize_hf_token(hf_token_header) or _normalize_hf_token(hf_token)
        from hub.services.models import gguf_variants as hub_gguf_variants

        response = await hub_gguf_variants.get_gguf_variants_response(
            repo_id,
            hf_token = hf_token,
        )
        local = is_local_path(repo_id)

        return GgufVariantsResponse(
            repo_id = response.repo_id,
            variants = [
                GgufVariantDetail(
                    filename = v.filename,
                    quant = v.quant,
                    size_bytes = v.size_bytes,
                    download_size_bytes = int(
                        getattr(v, "download_size_bytes", v.size_bytes) or v.size_bytes
                    ),
                    downloaded = bool(v.downloaded),
                    update_available = bool(getattr(v, "update_available", False)),
                )
                for v in response.variants
            ],
            has_vision = response.has_vision,
            default_variant = response.default_variant,
            context_length = _read_native_context_length(repo_id, is_local = local),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing GGUF variants for '{repo_id}': {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = "Failed to list GGUF variants",
        )


@router.get("/gguf-download-progress")
async def get_gguf_download_progress(
    repo_id: str = Query(..., description = "HuggingFace repo ID"),
    variant: str = Query("", description = "Quantization variant (e.g. UD-TQ1_0)"),
    expected_bytes: int = Query(0, description = "Expected total download size in bytes"),
    current_subject: str = Depends(get_current_subject),
):
    """Download progress from cached GGUF files for a specific variant.

    Tracks completed shards in snapshots and in-progress (.incomplete)
    downloads in the blobs directory.
    """
    try:
        if not _is_valid_repo_id(repo_id):
            return {
                "downloaded_bytes": 0,
                "expected_bytes": expected_bytes,
                "progress": 0,
            }

        from huggingface_hub import constants as hf_constants

        cache_dir = Path(hf_constants.HF_HUB_CACHE)
        target = f"models--{repo_id.replace('/', '--')}".lower()
        variant_lower = variant.lower().replace("-", "").replace("_", "")
        downloaded_bytes = 0
        in_progress_bytes = 0
        for entry in cache_dir.iterdir():
            if entry.name.lower() == target:
                # Completed .gguf files for this variant in snapshots.
                # Exclude mmproj so a vision adapter can't satisfy a same-label
                # main variant (e.g. mmproj-F16 vs an F16 weight).
                for f in _iter_gguf_paths(entry):
                    if _is_mmproj_filename(f.name):
                        continue
                    rel = f.relative_to(entry).as_posix()
                    quant = _extract_quant_label(rel)
                    if _is_big_endian_gguf_path(rel, quant):
                        continue
                    rel_key = rel.lower().replace("-", "").replace("_", "")
                    if not variant_lower or variant_lower in rel_key:
                        try:
                            downloaded_bytes += f.stat().st_size
                        except OSError:
                            continue  # broken symlink / unreadable: skip
                # In-progress (.incomplete) downloads in blobs.
                blobs_dir = entry / "blobs"
                if blobs_dir.is_dir():
                    for f in blobs_dir.iterdir():
                        if f.is_file() and f.name.endswith(".incomplete"):
                            try:
                                in_progress_bytes += f.stat().st_size
                            except OSError:
                                continue
                break

        total_progress_bytes = downloaded_bytes + in_progress_bytes
        progress = min(total_progress_bytes / expected_bytes, 0.99) if expected_bytes > 0 else 0
        # Report 1.0 only when all bytes are in completed files.
        if expected_bytes > 0 and downloaded_bytes >= expected_bytes:
            progress = 1.0
        return {
            "downloaded_bytes": total_progress_bytes,
            "expected_bytes": expected_bytes,
            "progress": round(progress, 3),
        }
    except Exception:
        return {"downloaded_bytes": 0, "expected_bytes": expected_bytes, "progress": 0}


def _resolve_hf_cache_realpath(repo_dir: Path) -> Optional[str]:
    """Pick the most useful on-disk path for a HF cache repo.

    Prefers the most-recent snapshot dir (what ``from_pretrained`` uses),
    falling back to the cache repo root. Returns the resolved realpath so
    snapshot symlinks follow back to blobs/.
    """
    try:
        snapshots_dir = repo_dir / "snapshots"
        if snapshots_dir.is_dir():
            snaps = [s for s in snapshots_dir.iterdir() if s.is_dir()]
            if snaps:
                latest = max(snaps, key = lambda s: s.stat().st_mtime)
                return str(latest.resolve())
        return str(repo_dir.resolve())
    except Exception:
        return None


@router.get("/download-progress")
async def get_download_progress(
    repo_id: str = Query(..., description = "HuggingFace repo ID"),
    current_subject: str = Depends(get_current_subject),
):
    """Return download progress for any HuggingFace model repo.

    Checks the local HF cache for completed blobs and in-progress
    (.incomplete) downloads. Gets the expected total size from the HF API
    on the first call, then caches it for later polls. Also returns
    ``cache_path``: the realpath of the snapshot dir (or cache repo root
    if no snapshot yet) so the UI can show where weights live on disk.
    """
    _empty = {
        "downloaded_bytes": 0,
        "expected_bytes": 0,
        "progress": 0,
        "cache_path": None,
    }
    try:
        if not _is_valid_repo_id(repo_id):
            return _empty

        from huggingface_hub import constants as hf_constants

        cache_dir = Path(hf_constants.HF_HUB_CACHE)
        target = f"models--{repo_id.replace('/', '--')}".lower()
        completed_bytes = 0
        in_progress_bytes = 0
        cache_path: Optional[str] = None

        for entry in cache_dir.iterdir():
            if entry.name.lower() != target:
                continue
            cache_path = _resolve_hf_cache_realpath(entry)
            blobs_dir = entry / "blobs"
            if not blobs_dir.is_dir():
                break
            for f in blobs_dir.iterdir():
                if not f.is_file():
                    continue
                if f.name.endswith(".incomplete"):
                    in_progress_bytes += f.stat().st_size
                else:
                    completed_bytes += f.stat().st_size
            break

        downloaded_bytes = completed_bytes + in_progress_bytes
        if downloaded_bytes == 0:
            return {**_empty, "cache_path": cache_path}

        expected_bytes = _get_repo_size_cached(repo_id)
        if expected_bytes <= 0:
            # Total unknown; report bytes only, no percentage.
            return {
                "downloaded_bytes": downloaded_bytes,
                "expected_bytes": 0,
                "progress": 0,
                "cache_path": cache_path,
            }

        # 95% threshold (blob dedup can skew completed_bytes). Do NOT
        # treat "no .incomplete files" as done: HF downloads sequentially,
        # so none exist between files even when far from finished.
        if completed_bytes >= expected_bytes * 0.95:
            progress = 1.0
        else:
            progress = min(downloaded_bytes / expected_bytes, 0.99)
        return {
            "downloaded_bytes": downloaded_bytes,
            "expected_bytes": expected_bytes,
            "progress": round(progress, 3),
            "cache_path": cache_path,
        }
    except Exception as e:
        logger.warning(f"Error checking download progress for {repo_id}: {e}")
        return _empty


_repo_size_cache: dict[str, int] = {}


def _get_repo_size_cached(repo_id: str) -> int:
    if repo_id in _repo_size_cache:
        return _repo_size_cache[repo_id]
    try:
        from huggingface_hub import model_info as hf_model_info

        info = hf_model_info(repo_id, token = None, files_metadata = True)
        total = sum(s.size for s in info.siblings if s.size)
        _repo_size_cache[repo_id] = total
        return total
    except Exception as e:
        logger.warning(f"Failed to get repo size for {repo_id}: {e}")
        return 0


def _repo_in_any_hf_cache(model_name: str) -> bool:
    """Whether ``model_name`` already exists in ANY HF cache the discard searches
    (active, legacy, default).

    ``created_by_scan`` must be True only when the scan itself first pulled the repo;
    checking just the active cache (``get_cache_path``) would mark a repo the user
    already had in a legacy/default cache as scan-created, so declining the consent
    would delete a model they did not download via the scan. Mirrors the cache set in
    ``_all_hf_cache_scans`` but only probes for the one repo dir (cheap, no full scan).
    """
    from utils.paths import (
        hf_default_cache_dir,
        legacy_hf_cache_dir,
        resolve_cached_repo_id_case,
    )

    dirname = f"models--{resolve_cached_repo_id_case(model_name).replace('/', '--')}"
    dirname_lower = dirname.lower()
    candidates = []
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        candidates.append(Path(HF_HUB_CACHE))
    except Exception:
        pass
    for fn in (legacy_hf_cache_dir, hf_default_cache_dir):
        try:
            candidates.append(fn())
        except Exception:
            continue
    # resolve_cached_repo_id_case only normalizes the ACTIVE cache, but discard deletes
    # case-insensitively across all caches, so detect case-insensitively too -- else a
    # pre-existing case-variant repo is misreported as scan-created and deleted on decline.
    for cache in candidates:
        try:
            if (cache / dirname).exists():
                return True
            if cache.is_dir():
                for entry in cache.iterdir():
                    if entry.name.lower() == dirname_lower and entry.is_dir():
                        return True
        except Exception:
            continue
    return False


def _all_hf_cache_scans():
    """scan_cache_dir for the active, legacy, and default HF caches.

    Each probe is isolated: an unreadable auxiliary cache (permission denied,
    broken symlink, OS-redirected ~/.cache) is skipped, not fatal, so the
    Downloaded list never blanks out and downloads never leak into Recommended.
    """
    from huggingface_hub import scan_cache_dir
    from utils.paths import legacy_hf_cache_dir, hf_default_cache_dir

    scans = []
    # Guard the active cache too: degrade to "no downloads" instead of raising.
    try:
        scans.append(scan_cache_dir())
    except Exception as exc:
        logger.warning("Could not scan active HF cache: %s", exc)

    seen: set[str] = set()
    try:
        # Resolve the active cache dir for dedup.
        from huggingface_hub.constants import HF_HUB_CACHE
        seen.add(str(Path(HF_HUB_CACHE).resolve()))
    except Exception:
        pass

    for extra_fn in (legacy_hf_cache_dir, hf_default_cache_dir):
        try:
            extra = extra_fn()
            # is_dir()/resolve() can raise on an inaccessible path; skip it.
            if not extra.is_dir():
                continue
            resolved = str(extra.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            scans.append(scan_cache_dir(cache_dir = str(extra)))
        except Exception as exc:
            logger.warning("Could not scan HF cache %s: %s", extra_fn.__name__, exc)
    return scans


def _is_gguf_filename(name: str) -> bool:
    return name.lower().endswith(".gguf")


def _is_mmproj_filename(name: str) -> bool:
    """Match GGUF vision-adapter (mmproj) files. Consistent with
    ``utils.models.model_config._is_mmproj``."""
    return "mmproj" in name.lower()


def _is_main_gguf_filename(name: str) -> bool:
    """A GGUF file that is a primary weight, not an mmproj vision
    adapter."""
    return _is_gguf_filename(name) and not _is_mmproj_filename(name)


def _repo_has_mmproj(repo_info) -> bool:
    """True if the repo ships a GGUF vision adapter (mmproj), so it can
    take image inputs. Cheap: scans already-listed file names only."""
    return any(
        _is_mmproj_filename(f.file_name) for revision in repo_info.revisions for f in revision.files
    )


def _iter_gguf_paths(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and _is_gguf_filename(path.name):
            yield path


def _repo_gguf_size_bytes(repo_info) -> int:
    """Total on-disk size of primary GGUF weight files across all
    revisions, excluding mmproj vision-adapter files.

    Hugging Face hardlinks blobs shared between revisions, so this
    deduplicates by blob path (or revision commit hash + filename as a
    fallback) to avoid double-counting. Unknown sizes (``size_on_disk is
    None``, e.g. a partial download) count as zero. mmproj files are
    excluded so repos whose only ``.gguf`` artifact is a vision adapter
    aren't classed as GGUF repos: the variant selector filters mmproj
    out and would otherwise show zero pickable variants.
    """
    unique_blobs: dict[str, int] = {}
    for revision in repo_info.revisions:
        rev_id = getattr(revision, "commit_hash", None) or str(id(revision))
        for f in revision.files:
            if _is_main_gguf_filename(f.file_name):
                blob_path = getattr(f, "blob_path", None)
                size = f.size_on_disk or 0
                if blob_path:
                    unique_blobs[str(blob_path)] = size
                else:
                    unique_blobs[f"{rev_id}:{f.file_name}"] = size
    return sum(unique_blobs.values())


def _repo_has_gguf_files(repo_info) -> bool:
    """True when any revision in a cached repo has a primary GGUF weight
    file. Repos whose only ``.gguf`` artifact is an mmproj vision adapter
    are not treated as GGUF here."""
    return _repo_gguf_size_bytes(repo_info) > 0


def _blob_mtime(f) -> float:
    """Blob modification time in epoch seconds (0.0 if unknown).

    Prefers HF metadata ``blob_last_modified``, falls back to stat(); uses
    only mtimes (portable across Windows, macOS, Linux), never path parsing.
    """
    ts = getattr(f, "blob_last_modified", None)
    if isinstance(ts, (int, float)) and ts > 0:
        return float(ts)
    blob_path = getattr(f, "blob_path", None)
    if blob_path:
        try:
            return float(Path(blob_path).stat().st_mtime)
        except OSError:
            pass
    return 0.0


def _repo_gguf_last_modified(repo_info) -> float:
    """Newest mtime among a repo's primary (non-mmproj) GGUF blobs.

    Drives the Downloaded list's "last downloaded" ordering and groups a
    multi-quant repo by its most recently downloaded quant.
    """
    latest = 0.0
    for revision in repo_info.revisions:
        for f in revision.files:
            if _is_main_gguf_filename(f.file_name):
                latest = max(latest, _blob_mtime(f))
    return latest


@router.get("/cached-gguf")
async def list_cached_gguf(current_subject: str = Depends(get_current_subject)):
    """List GGUF repos downloaded to HF cache, legacy Unsloth cache, and HF default cache."""
    try:
        cache_scans = _all_hf_cache_scans()

        seen_lower: dict[str, dict] = {}
        for hf_cache in cache_scans:
            for repo_info in hf_cache.repos:
                try:
                    if repo_info.repo_type != "model":
                        continue
                    repo_id = repo_info.repo_id
                    # Pass the snapshot path too so the config-based check hides a
                    # custom Whisper checkpoint here, not just curated repo ids.
                    if _is_hidden_model(repo_id, str(repo_info.repo_path)):
                        continue
                    total_size = _repo_gguf_size_bytes(repo_info)
                    if total_size == 0:
                        continue
                    key = repo_id.lower()
                    existing = seen_lower.get(key)
                    last_modified = _repo_gguf_last_modified(repo_info)
                    if existing is None or total_size > existing["size_bytes"]:
                        row = {
                            "repo_id": repo_id,
                            "size_bytes": total_size,
                            "cache_path": str(repo_info.repo_path),
                            "has_vision": _repo_has_mmproj(repo_info),
                        }
                        # Keep the newest timestamp across duplicate caches;
                        # attach only when known so absent rows sort as oldest.
                        lm = max(last_modified, (existing or {}).get("last_modified", 0.0))
                        if lm > 0:
                            row["last_modified"] = lm
                        seen_lower[key] = row
                    elif last_modified > existing.get("last_modified", 0.0):
                        existing["last_modified"] = last_modified
                except Exception as e:
                    repo_label = getattr(repo_info, "repo_id", "<unknown>")
                    logger.warning(f"Skipping cached GGUF repo {repo_label}: {e}")
                    continue
        # Newest download first; stable repo_id tie-break for equal/missing mtimes.
        cached = sorted(
            seen_lower.values(),
            key = lambda c: (-(c.get("last_modified") or 0.0), c["repo_id"].lower()),
        )
        return {"cached": cached}
    except Exception as e:
        logger.error(f"Error listing cached GGUF repos: {e}", exc_info = True)
        return {"cached": []}


@router.get("/cached-models", response_model = CachedModelsResponse)
async def list_cached_models(
    current_subject: str = Depends(get_current_subject),
    hf_token: Optional[str] = Depends(get_hf_token),
):
    """List non-GGUF model repos downloaded to HF cache, legacy Unsloth cache, and HF default cache."""
    _WEIGHT_EXTENSIONS = (".safetensors", ".bin")
    hf_token = _normalize_hf_token(hf_token)

    try:
        cache_scans = _all_hf_cache_scans()

        seen_lower: dict[str, dict] = {}
        for hf_cache in cache_scans:
            for repo_info in hf_cache.repos:
                try:
                    if repo_info.repo_type != "model":
                        continue
                    repo_id = repo_info.repo_id
                    # Pass the snapshot path too so the config-based check hides a
                    # custom Whisper checkpoint here, not just curated repo ids.
                    if _is_hidden_model(repo_id, str(repo_info.repo_path)):
                        continue
                    if _repo_has_gguf_files(repo_info):
                        continue
                    total_size = sum(
                        (f.size_on_disk or 0) for rev in repo_info.revisions for f in rev.files
                    )
                    if total_size == 0:
                        continue
                    weight_files = [
                        f
                        for rev in repo_info.revisions
                        for f in rev.files
                        if f.file_name.endswith(_WEIGHT_EXTENSIONS)
                    ]
                    if not weight_files:
                        continue
                    last_modified = max(
                        (_blob_mtime(f) for f in weight_files),
                        default = 0.0,
                    )
                    key = repo_id.lower()
                    existing = seen_lower.get(key)
                    if existing is None or total_size > existing["size_bytes"]:
                        row = {
                            "repo_id": repo_id,
                            "size_bytes": total_size,
                        }
                        # Keep the newest timestamp across duplicate caches;
                        # attach only when known so absent rows sort as oldest.
                        lm = max(last_modified, (existing or {}).get("last_modified", 0.0))
                        if lm > 0:
                            row["last_modified"] = lm
                        seen_lower[key] = row
                    elif last_modified > existing.get("last_modified", 0.0):
                        existing["last_modified"] = last_modified
                except Exception as e:
                    repo_label = getattr(repo_info, "repo_id", "<unknown>")
                    logger.warning(f"Skipping cached model repo {repo_label}: {e}")
                    continue

        rows = list(seen_lower.values())
        # Local-only list path: update checks are GGUF-only and happen lazily
        # when a repo's variants are viewed.
        cached = sorted(
            rows,
            key = lambda c: (-(c.get("last_modified") or 0.0), c["repo_id"].lower()),
        )
        return {"cached": cached}
    except Exception as e:
        logger.error(f"Error listing cached models: {e}", exc_info = True)
        return {"cached": []}


@router.delete("/delete-cached")
async def delete_cached_model(
    repo_id: str = Body(...),
    variant: Optional[str] = Body(None),
    current_subject: str = Depends(get_current_subject),
):
    """Delete a cached model repo (or a specific GGUF variant) from the HF cache.

    With *variant*, only GGUF files matching that quant label are removed
    (e.g. ``UD-Q4_K_XL``); otherwise the whole repo is deleted. Refuses
    if the model is currently loaded for inference.
    """
    if not _is_valid_repo_id(repo_id):
        raise HTTPException(status_code = 400, detail = "Invalid repo_id format")

    # Refuse if the model is currently loaded.
    try:
        from routes.inference import get_llama_cpp_backend
        llama_backend = get_llama_cpp_backend()
        if llama_backend.is_loaded and llama_backend.model_identifier:
            loaded_id = llama_backend.model_identifier.lower()
            if loaded_id == repo_id.lower() or loaded_id.startswith(repo_id.lower()):
                raise HTTPException(
                    status_code = 400,
                    detail = "Unload the model before deleting",
                )
    except HTTPException:
        raise
    except Exception:
        pass

    try:
        inference_backend = get_inference_backend()
        if inference_backend.active_model_name:
            active = inference_backend.active_model_name.lower()
            if active == repo_id.lower() or active.startswith(repo_id.lower()):
                raise HTTPException(
                    status_code = 400,
                    detail = "Unload the model before deleting",
                )
    except HTTPException:
        raise
    except Exception:
        pass

    try:
        cache_scans = _all_hf_cache_scans()

        target_repo = None
        for hf_cache in cache_scans:
            for repo_info in hf_cache.repos:
                if repo_info.repo_type != "model":
                    continue
                if repo_info.repo_id.lower() == repo_id.lower():
                    target_repo = repo_info
                    break
            if target_repo is not None:
                break

        if target_repo is None:
            raise HTTPException(status_code = 404, detail = "Model not found in cache")

        # ── Per-variant GGUF deletion ────────────────────────────
        if variant:
            deleted_bytes = 0
            deleted_count = 0
            for rev in target_repo.revisions:
                for f in rev.files:
                    if not _is_gguf_filename(f.file_name):
                        continue
                    quant = _extract_quant_label(f.file_name)
                    if quant.lower() != variant.lower():
                        continue
                    # Delete the blob (data) and the snapshot symlink.
                    try:
                        blob = Path(f.blob_path)
                        snap = Path(f.file_path)
                        size = blob.stat().st_size if blob.exists() else 0
                        if snap.exists() or snap.is_symlink():
                            snap.unlink()
                        if blob.exists():
                            blob.unlink()
                        deleted_bytes += size
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {f.file_name}: {e}")

            if deleted_count == 0:
                raise HTTPException(
                    status_code = 404,
                    detail = f"Variant {variant} not found in cache for {repo_id}",
                )

            freed_mb = deleted_bytes / (1024 * 1024)
            logger.info(
                f"Deleted {deleted_count} file(s) for {repo_id} variant {variant}: "
                f"{freed_mb:.1f} MB freed"
            )
            return {"status": "deleted", "repo_id": repo_id, "variant": variant}

        # ── Full repo deletion ───────────────────────────────────
        revision_hashes = [rev.commit_hash for rev in target_repo.revisions]
        if not revision_hashes:
            raise HTTPException(status_code = 404, detail = "No revisions found for model")

        delete_strategy = hf_cache.delete_revisions(*revision_hashes)
        logger.info(
            f"Deleting cached model {repo_id}: "
            f"{delete_strategy.expected_freed_size_str} will be freed"
        )
        delete_strategy.execute()

        return {"status": "deleted", "repo_id": repo_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting cached model {repo_id}: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = "Failed to delete cached model",
        )


@router.get("/checkpoints", response_model = CheckpointListResponse)
async def list_checkpoints(
    outputs_dir: str = Query(
        default = str(outputs_root()),
        description = "Directory to scan for checkpoints",
    ),
    current_subject: str = Depends(get_current_subject),
):
    """List checkpoints in the outputs directory.

    Scans the outputs folder for training runs and their checkpoints.
    """
    try:
        resolved_outputs_dir = str(resolve_output_dir(outputs_dir))
        raw_models = scan_checkpoints(outputs_dir = resolved_outputs_dir)

        models = [
            ModelCheckpoints(
                name = model_name,
                checkpoints = [
                    CheckpointInfo(display_name = display_name, path = path, loss = loss)
                    for display_name, path, loss in checkpoints
                ],
                base_model = metadata.get("base_model"),
                peft_type = metadata.get("peft_type"),
                lora_rank = metadata.get("lora_rank"),
                is_quantized = metadata.get("is_quantized", False),
            )
            for model_name, checkpoints, metadata in raw_models
        ]

        return CheckpointListResponse(
            outputs_dir = resolved_outputs_dir,
            models = models,
        )
    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to list checkpoints",
            event = "models.list_checkpoints_failed",
            log = logger,
        )


# Successful estimates only, keyed by model id (token-independent, never stored).
# Failures are not cached so a transient offline/gated error can recover later.
_EXPORT_SIZE_CACHE: dict[str, tuple[int, int, str]] = {}


def _is_sizable_local_path(model: str) -> bool:
    """True only for local paths under an Unsloth data root.

    Containment is decided lexically (no filesystem access) before the path is
    touched, then the path is symlink-resolved and re-checked so a symlink
    inside a root can't point the sizer outside it. A user-controlled path thus
    can't trigger a scan of an arbitrary dir.
    """
    from utils.paths import outputs_root, exports_root, studio_root
    from utils.paths.storage_roots import cache_root

    def _lexical(p: str) -> str:
        # Lexical only (no filesystem read); normpath collapses '..'.
        return os.path.normpath(os.path.abspath(os.path.expanduser(p)))

    raw_roots = [studio_root(), outputs_root(), exports_root(), cache_root()]
    roots = []
    for root in raw_roots:
        try:
            roots.append(_lexical(str(root)))
        except (OSError, RuntimeError, ValueError):
            continue

    try:
        candidate = _lexical(model)
    except (OSError, RuntimeError, ValueError):
        return False
    for root in roots:
        if candidate == root or candidate.startswith(root + os.sep):
            # Contained lexically; resolve symlinks and re-verify the real path
            # is still under a root before touching the filesystem.
            try:
                real = os.path.realpath(candidate)
            except (OSError, RuntimeError, ValueError):
                return False
            for raw in raw_roots:
                try:
                    real_root = os.path.realpath(str(raw))
                except (OSError, RuntimeError, ValueError):
                    continue
                if real == real_root or real.startswith(real_root + os.sep):
                    return os.path.exists(real)
            return False
    return False


def _export_size_cached(
    model: str, hf_token: Optional[str]
) -> tuple[Optional[int], Optional[int], str]:
    """Estimate a model's fp16/bf16-equivalent size in bytes (+ total params).

    Memoizes successful results by model id; never raises (failures return
    (None, None, "unavailable") and are not cached). Blocking I/O; call off-thread.
    """
    cached = _EXPORT_SIZE_CACHE.get(model)
    if cached is not None:
        return cached
    try:
        from utils.hardware.hardware import (
            _resolve_model_identifier_for_gpu_estimate,
            estimate_fp16_model_size_bytes,
        )

        # A local LoRA adapter is sized via its base model, which the sizer
        # reads from the adapter config; re-validate that resolved base so a
        # crafted adapter can't redirect the local scan outside the roots.
        if is_local_path(model):
            base = _resolve_model_identifier_for_gpu_estimate(model, hf_token = hf_token)
            if is_local_path(base) and not _is_sizable_local_path(base):
                return None, None, "unavailable"

        fp16_bytes, source = estimate_fp16_model_size_bytes(model, hf_token = hf_token)
        if not fp16_bytes or fp16_bytes <= 0:
            return None, None, source or "unavailable"
        result = (int(fp16_bytes), int(fp16_bytes) // 2, source)
        _EXPORT_SIZE_CACHE[model] = result
        return result
    except Exception as e:  # a size hint must never break export
        logger.warning("Could not estimate export size for '%s': %s", model, e)
        return None, None, "unavailable"


@router.get("/export-size", response_model = ExportSizeResponse)
async def get_export_size(
    model: str = Query(..., description = "Base model id or local model path to size"),
    hf_token: Optional[str] = Header(None, alias = "X-HF-Token"),
    current_subject: str = Depends(get_current_subject),
):
    """Estimate a model's fp16/bf16-equivalent size for the Export page.

    Returns nulls with HTTP 200 when the size can't be determined. The HF token
    (for gated repos) comes from the X-HF-Token header so it never hits URLs/logs.
    """
    if is_local_path(model):
        if not _is_sizable_local_path(model):
            return ExportSizeResponse(
                model = model, fp16_bytes = None, total_params = None, source = "unavailable"
            )
        resolved = model
    else:
        resolved = resolve_cached_repo_id_case(model)
    # Blocking network/disk I/O: run off the event loop.
    fp16_bytes, total_params, source = await asyncio.to_thread(
        _export_size_cached, resolved, hf_token
    )
    return ExportSizeResponse(
        model = resolved,
        fp16_bytes = fp16_bytes,
        total_params = total_params,
        source = source,
    )
