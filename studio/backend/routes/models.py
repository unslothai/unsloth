# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Model Management API routes
"""

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from typing import List, Optional
import structlog
from loggers import get_logger

import re as _re

_VALID_REPO_ID = _re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def _is_valid_repo_id(repo_id: str) -> bool:
    return bool(_VALID_REPO_ID.fullmatch(repo_id))


# Add backend directory to path
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from auth.authentication import get_current_subject

# Import backend functions
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
    # Fallback: try to import from parent directory
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
    is_vision: bool, audio_type: Optional[str], is_embedding: bool = False
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

    A model directory must have **both** a config file (``config.json`` or
    ``adapter_config.json``) **and** actual model weight files.  Both
    conditions are required: a bare directory with only loose ``.gguf``
    files (no config) might be a mixed collection, and a ``config.json``
    alone (no weights) is not a model directory.

    Excludes ``mmproj`` GGUF files (vision projectors) and non-weight
    ``.bin`` files (``tokenizer.bin``, ``vocab.bin``, etc.) from the
    weight check to avoid false positives.
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
        has_config = (d / "config.json").exists() or (
            d / "adapter_config.json"
        ).exists()
        if not has_config:
            return False
        return any(_is_weight_file(f) for f in d.iterdir() if f.is_file())
    except OSError:
        return False


def _scan_models_dir(
    models_dir: Path,
    *,
    limit: int | None = None,
) -> List[LocalModelInfo]:
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
            has_model_files = (
                (child / "config.json").exists()
                or (child / "adapter_config.json").exists()
                or any(child.glob("*.safetensors"))
                or any(child.glob("*.bin"))
                or any(child.glob("*.gguf"))
            )
        except OSError:
            # Skip individual children that are unreadable (permissions, broken
            # symlinks, etc.) rather than failing the entire scan.
            continue
        if not has_model_files:
            continue
        try:
            updated_at = child.stat().st_mtime
        except OSError:
            updated_at = None
        found.append(
            LocalModelInfo(
                id = str(child),
                display_name = child.name,
                path = str(child),
                source = "models_dir",
                updated_at = updated_at,
            ),
        )
    # Also scan for standalone .gguf files directly in the models directory
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


def _scan_lmstudio_dir(lm_dir: Path) -> List[LocalModelInfo]:
    """Scan an LM Studio models directory for model files.

    LM Studio uses a ``publisher/model-name`` folder structure containing
    GGUF files, or standalone GGUF files at the top level.
    """
    if not lm_dir.exists() or not lm_dir.is_dir():
        return []

    # If the directory itself is a model directory (has config AND weight
    # files), it is not an LM Studio publisher structure -- return it as a
    # single model entry.  We cannot skip it silently because this function
    # is the only scanner called for default LM Studio roots.
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
                            updated_at = updated_at,
                        ),
                    )
                continue

            # If the child directory itself looks like a model directory
            # (has config AND weight files), surface it directly instead
            # of descending into it as a publisher.
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
                        updated_at = updated_at,
                    ),
                )
                continue

            # child is a publisher directory -- scan its sub-directories
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

    Prefers ``<ollama_dir>/.studio_links/`` so the links sit next to the
    blobs they point at. Falls back to a per-ollama-dir namespace under
    Studio's own cache when the models directory is read-only (common
    for system installs under ``/usr/share/ollama`` or ``/var/lib/ollama``)
    so we still surface Ollama models in those environments.
    """
    from utils.paths.storage_roots import cache_root

    primary = ollama_dir / ".studio_links"
    try:
        primary.mkdir(exist_ok = True)
        return primary
    except OSError as e:
        logger.debug(
            "Ollama dir %s not writable for .studio_links (%s); "
            "falling back to Studio cache",
            ollama_dir,
            e,
        )

    # Fallback: namespace by a hash of the ollama_dir so two different
    # Ollama roots don't collide. This is a cache path, not a security
    # boundary.
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


def _scan_ollama_dir(
    ollama_dir: Path, limit: Optional[int] = None
) -> List[LocalModelInfo]:
    """Scan an Ollama models directory for downloaded models.

    Ollama stores models in a content-addressable layout::

        <ollama_dir>/manifests/<host>/<namespace>/<model>/<tag>
        <ollama_dir>/blobs/sha256-...

    The default host is ``registry.ollama.ai`` with namespace
    ``library`` (official models), but users can pull from custom
    namespaces (``mradermacher/llama3``) or entirely different hosts
    (``hf.co/org/repo:tag``).  We iterate all manifest files via
    ``rglob`` so every layout depth is discovered.

    Each manifest is JSON with a ``layers`` array. The layer with
    ``mediaType == "application/vnd.ollama.image.model"`` contains the
    GGUF weights. Vision models also have a projector layer
    (``application/vnd.ollama.image.projector``). We read the config
    layer to extract family/size info.

    Since Ollama blobs lack a ``.gguf`` extension (which the GGUF
    loading pipeline requires), we create ``.gguf``-named links
    pointing at the blobs so the existing ``detect_gguf_model`` and
    ``llama-server -m`` paths work unchanged. Each model gets its
    own subdirectory under the links dir (keyed by a short hash of
    the manifest path) so that ``detect_mmproj_file`` only sees the
    projector for *that* model.  Links are created as symlinks when
    possible, falling back to hardlinks (Windows without Developer
    Mode) as a last resort.  The link dir lives under
    ``<ollama_dir>/.studio_links/`` when writable, otherwise under
    Studio's own cache directory.
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

        Tries symlink first, then hardlink (works on Windows without
        Developer Mode when target is on the same filesystem).  Skips
        the model if neither works -- a full file copy of a multi-GB
        GGUF inside a synchronous API request would block the backend.

        Idempotent: skips recreation when a valid link already exists.
        """
        link_dir.mkdir(parents = True, exist_ok = True)
        link_path = link_dir / link_name
        resolved = target.resolve()

        # Skip if the link already points at the exact same blob.
        # Only use samefile -- size-based checks can reuse stale links
        # after `ollama pull` updates a tag to a same-sized blob.
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
                logger.debug(
                    "Could not clean up tmp path %s: %s", tmp_path, cleanup_err
                )
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

            if (
                host == "registry.ollama.ai"
                and repo_parts
                and repo_parts[0] == "library"
            ):
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
            for layer in manifest.get("layers", []):
                media = layer.get("mediaType", "")
                digest = layer.get("digest", "")
                if not digest:
                    continue

                if media == "application/vnd.ollama.image.model":
                    candidate = blobs_dir / digest.replace(":", "-")
                    if candidate.is_file():
                        link_name = f"{safe_name}-{tag}{quant}.gguf"
                        gguf_link_path = _make_link(
                            model_link_dir, link_name, candidate
                        )

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


@router.get("/local", response_model = LocalModelListResponse)
async def list_local_models(
    models_dir: str = Query(
        default = "./models", description = "Directory to scan for local model folders"
    ),
    current_subject: str = Depends(get_current_subject),
):
    """
    List local model candidates from custom models dir, HF cache,
    legacy Unsloth HF cache, and LM Studio directories.
    """
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

    # Validate models_dir against an allowlist of trusted directories.
    # Only the trusted Path objects are used for filesystem access -- the
    # user-supplied string is only used for matching, never for path construction.
    allowed_roots: list[Path] = [Path("./models").resolve(), hf_cache_dir]
    if legacy_hf.is_dir():
        allowed_roots.append(legacy_hf)
    if hf_default.is_dir():
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
            models_root = root  # Use the trusted root, not the user-supplied path
            break
    if models_root is None:
        raise HTTPException(
            status_code = 403,
            detail = "Directory not allowed",
        )

    try:
        local_models = _scan_models_dir(models_root) + _scan_hf_cache(hf_cache_dir)

        # Scan legacy Unsloth HF cache for backward compatibility
        if legacy_hf.is_dir() and legacy_hf.resolve() != hf_cache_dir.resolve():
            local_models += _scan_hf_cache(legacy_hf)

        # Scan HF system default cache (may differ when env vars are overridden)
        if (
            hf_default.is_dir()
            and hf_default.resolve() != hf_cache_dir.resolve()
            and hf_default.resolve() != legacy_hf.resolve()
        ):
            local_models += _scan_hf_cache(hf_default)

        # Scan LM Studio directories
        for lm_dir in lm_dirs:
            local_models += _scan_lmstudio_dir(lm_dir)

        # Scan user-added custom folders (cap per-folder to avoid unbounded scans)
        from storage.studio_db import list_scan_folders

        _MAX_MODELS_PER_FOLDER = 200
        try:
            custom_folders = list_scan_folders()
        except Exception as e:
            logger.warning("Could not load custom scan folders: %s", e)
            custom_folders = []
        for folder in custom_folders:
            folder_path = Path(folder["path"])
            try:
                # Ollama scanner creates .studio_links/ with .gguf symlinks.
                # Filter those from the generic scanners to avoid duplicates
                # and leaking internal paths into the UI.
                _generic = [
                    m
                    for m in (
                        _scan_models_dir(folder_path, limit = _MAX_MODELS_PER_FOLDER)
                        + _scan_hf_cache(folder_path)
                        + _scan_lmstudio_dir(folder_path)
                    )
                    if not any(
                        p in (".studio_links", "ollama_links")
                        for p in Path(m.path).parts
                    )
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
            local_models += [
                m.model_copy(update = {"source": "custom"}) for m in custom_models
            ]

        # Deduplicate models, but always keep custom folder entries so they
        # appear in the "Custom Folders" UI section even when the same model
        # also exists in the HF cache or default models directory.  Use a
        # (id, source) key for custom entries to avoid collisions.
        deduped: dict[str, LocalModelInfo] = {}
        for model in local_models:
            key = f"{model.id}\x00custom" if model.source == "custom" else model.id
            if key not in deduped:
                deduped[key] = model

        models = sorted(
            deduped.values(),
            key = lambda item: (item.updated_at or 0),
            reverse = True,
        )

        return LocalModelListResponse(
            models_dir = str(models_root),
            hf_cache_dir = str(hf_cache_dir),
            lmstudio_dirs = [str(d) for d in lm_dirs],
            models = models,
        )
    except Exception as e:
        logger.error(f"Error listing local models: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to list local models: {str(e)}",
        )


@router.get("/scan-folders")
async def get_scan_folders(
    current_subject: str = Depends(get_current_subject),
):
    """List all registered custom model scan folders."""
    from storage.studio_db import list_scan_folders

    return {"folders": list_scan_folders()}


@router.post("/scan-folders", response_model = ScanFolderInfo, status_code = 201)
async def add_scan_folder_endpoint(
    body: AddScanFolderRequest,
    current_subject: str = Depends(get_current_subject),
):
    """Register a new directory to scan for local models."""
    from storage.studio_db import add_scan_folder

    try:
        folder = add_scan_folder(body.path)
    except ValueError as e:
        logger.warning("Scan folder rejected: %s (path=%s)", e, body.path)
        raise HTTPException(status_code = 400, detail = str(e))
    logger.info("Scan folder added: %s", folder.get("path"))
    return folder


@router.delete("/scan-folders/{folder_id}")
async def remove_scan_folder_endpoint(
    folder_id: int,
    current_subject: str = Depends(get_current_subject),
):
    """Remove a registered custom scan folder."""
    from storage.studio_db import remove_scan_folder

    remove_scan_folder(folder_id)
    logger.info("Scan folder removed: id=%s", folder_id)
    return {"ok": True}


@router.get("/recommended-folders")
async def get_recommended_folders(
    current_subject: str = Depends(get_current_subject),
):
    """Return well-known model directories that exist on this machine.

    Lightweight alternative to ``browse-folders`` for showing quick-pick
    chips without the overhead of enumerating a directory tree.  Returns
    paths that actually exist on disk (HF cache, LM Studio, Ollama,
    ``~/models``, etc.) so the frontend can offer them as one-click
    "Recommended" shortcuts in the Custom Folders section.
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
        if Path(resolved).is_dir() and os.access(resolved, os.R_OK | os.X_OK):
            seen.add(resolved)
            folders.append(resolved)

    # LM Studio model directories
    try:
        for p in lmstudio_model_dirs():
            _add(p)
    except Exception as e:
        logger.warning("Failed to scan for LM Studio model directories: %s", e)

    # Ollama model directories
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
    Used to surface a count-hint on the response so the UI can tell
    users that a leaf directory (no subdirs, only weights) is a valid
    "Use this folder" target.

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
            # Fast name check first
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
    from utils.paths import (
        hf_default_cache_dir,
        legacy_hf_cache_dir,
        well_known_model_dirs,
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
        if resolved.is_dir():
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
    """Return True if *target* equals or is a descendant of any allowed
    root. The comparison uses ``os.path.realpath`` so symlinks cannot be
    used to escape the sandbox.
    """
    try:
        target_real = os.path.realpath(str(target))
    except OSError:
        return False
    for root in allowed_roots:
        try:
            root_real = os.path.realpath(str(root))
        except OSError:
            continue
        if target_real == root_real or target_real.startswith(root_real + os.sep):
            return True
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
            detail = f"Permission denied reading {current}",
        ) from None
    except OSError as exc:
        raise HTTPException(
            status_code = 500,
            detail = f"Could not read {current}: {exc}",
        ) from exc
    return None


def _resolve_browse_target(path: Optional[str], allowed_roots: list[Path]) -> Path:
    """Resolve a requested browse path by walking from trusted allowlist roots."""
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
                        "POST /api/models/scan-folders first, or pick a directory "
                        "under your home folder."
                    ),
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
            "POST /api/models/scan-folders first, or pick a directory "
            "under your home folder."
        ),
    )


@router.get("/browse-folders", response_model = BrowseFoldersResponse)
async def browse_folders(
    path: Optional[str] = Query(
        None,
        description = (
            "Directory to list. If omitted, defaults to the current user's "
            "home directory. Tilde (`~`) and relative paths are expanded. "
            "Must resolve inside the allowlist of browseable roots (HOME, "
            "HF cache, Studio dirs, registered scan folders, well-known "
            "model dirs)."
        ),
    ),
    show_hidden: bool = Query(
        False,
        description = "Include entries whose name starts with a dot",
    ),
    current_subject: str = Depends(get_current_subject),
):
    """
    List immediate subdirectories of *path* for the Custom Folders picker.

    The frontend uses this to render a modal folder browser without needing
    a native OS dialog (Studio is served over HTTP, so the browser can't
    reveal absolute paths on the host). The endpoint is read-only and does
    not create, move, or delete anything. It simply enumerates visible
    subdirectories so the user can click their way to a folder and hand
    the resulting string back to POST `/api/models/scan-folders`.

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
    from utils.paths import hf_default_cache_dir, well_known_model_dirs
    from storage.studio_db import list_scan_folders

    # Build the allowlist once -- both the sandbox check below and the
    # suggestion chips use the same set, so chips are always navigable.
    allowed_roots = _build_browse_allowlist()

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

    def _add_sug(p: Optional[Path]) -> None:
        if p is None:
            return
        try:
            resolved = str(p.resolve())
        except OSError:
            return
        if resolved in seen_sug:
            return
        if Path(resolved).is_dir():
            seen_sug.add(resolved)
            suggestions.append(resolved)

    # Home always comes first -- it's the safe fallback when everything
    # else is cold.
    _add_sug(Path.home())
    # The HF cache root the process is actually using.
    try:
        _add_sug(hf_default_cache_dir())
    except Exception:
        pass
    # Already-registered scan folders (what the user has curated).
    try:
        for folder in list_scan_folders():
            _add_sug(Path(folder.get("path", "")))
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


@router.get("/list")
async def list_models(
    current_subject: str = Depends(get_current_subject),
):
    """
    List available models (default models and loaded models).

    This endpoint returns the default models and any currently loaded models.
    """
    try:
        inference_backend = get_inference_backend()

        # Get default models
        default_models = inference_backend.default_models

        # Get loaded models
        loaded_models = []
        for model_name, model_data in inference_backend.models.items():
            _is_vision = model_data.get("is_vision", False)
            _audio_type = model_data.get("audio_type")
            model_info = ModelDetails(
                id = model_name,
                name = model_name.split("/")[-1] if "/" in model_name else model_name,
                is_vision = _is_vision,
                is_lora = model_data.get("is_lora", False),
                is_audio = model_data.get("is_audio", False),
                audio_type = _audio_type,
                has_audio_input = model_data.get("has_audio_input", False),
                model_type = derive_model_type(_is_vision, _audio_type),
            )
            loaded_models.append(model_info)

        # Include active GGUF model (loaded via llama-server)
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

        # Combine default and loaded models
        all_models = []
        seen_ids = set()

        # Add default models
        for model_id in default_models:
            if model_id not in seen_ids:
                model_info = ModelDetails(
                    id = model_id,
                    name = model_id.split("/")[-1] if "/" in model_id else model_id,
                    is_gguf = model_id.upper().endswith("-GGUF"),
                )
                all_models.append(model_info)
                seen_ids.add(model_id)

        # Add loaded models
        for model_info in loaded_models:
            if model_info.id not in seen_ids:
                all_models.append(model_info)
                seen_ids.add(model_info.id)

        return ModelListResponse(models = all_models, default_models = default_models)

    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = f"Failed to list models: {str(e)}")


def _get_max_position_embeddings(config) -> Optional[int]:
    """Extract max_position_embeddings from a model config, checking text_config fallback."""
    if hasattr(config, "max_position_embeddings"):
        return config.max_position_embeddings
    if hasattr(config, "text_config") and hasattr(
        config.text_config, "max_position_embeddings"
    ):
        return config.text_config.max_position_embeddings
    return None


def _get_model_size_bytes(
    model_name: str, hf_token: Optional[str] = None
) -> Optional[int]:
    """Get total size of model weight files from HF Hub."""
    try:
        from huggingface_hub import HfApi

        api = HfApi(token = hf_token)
        info = api.repo_info(model_name, repo_type = "model", token = hf_token)
        if not info.siblings:
            return None

        weight_exts = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
        total = 0
        for sibling in info.siblings:
            if sibling.rfilename and any(
                sibling.rfilename.endswith(ext) for ext in weight_exts
            ):
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
    """
    Get configuration for a specific model.

    This endpoint wraps the backend load_model_defaults function.
    """
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

        # Load model defaults from backend
        config_dict = load_model_defaults(model_name)

        # Detect model capabilities (pass HF token for gated models)
        is_vision = is_vision_model(model_name, hf_token = hf_token)
        is_embedding = is_embedding_model(model_name, hf_token = hf_token)
        audio_type = detect_audio_type(model_name, hf_token = hf_token)

        # Check if it's a LoRA adapter
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

        # Fallback: try AutoConfig directly if not found yet
        if max_position_embeddings is None:
            try:
                from transformers import AutoConfig as _AutoConfig

                _trust = model_name.lower().startswith("unsloth/")
                _ac = _AutoConfig.from_pretrained(
                    model_name, trust_remote_code = _trust, token = hf_token
                )
                max_position_embeddings = _get_max_position_embeddings(_ac)
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
        logger.error(f"Error getting model config: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500, detail = f"Failed to get model config: {str(e)}"
        )


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
    """
    Scan for trained LoRA adapters and exported models.

    Returns both training outputs (from outputs_dir) and exported models
    (from exports_dir) in a single list, distinguished by source field.
    """
    try:
        resolved_outputs_dir = str(resolve_output_dir(outputs_dir))
        resolved_exports_dir = str(resolve_export_dir(exports_dir))
        lora_list = []

        # Scan training outputs
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
        logger.error(f"Error scanning LoRAs: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500, detail = f"Failed to scan LoRA adapters: {str(e)}"
        )


@router.get("/loras/{lora_path:path}/base-model", response_model = LoRABaseModelResponse)
async def get_lora_base_model(
    lora_path: str,
    current_subject: str = Depends(get_current_subject),
):
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
        logger.error(f"Error getting LoRA base model: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500, detail = f"Failed to get base model: {str(e)}"
        )


@router.get("/check-vision/{model_name:path}", response_model = VisionCheckResponse)
async def check_vision_model(
    model_name: str,
    current_subject: str = Depends(get_current_subject),
):
    """
    Check if a model is a vision model.

    This endpoint wraps the backend is_vision_model function.
    """
    try:
        logger.info(f"Checking if vision model: {model_name}")
        is_vision = is_vision_model(model_name)

        logger.info(f"Vision check result for {model_name}: is_vision={is_vision}")
        return VisionCheckResponse(
            model_name = model_name,
            is_vision = is_vision,
        )

    except Exception as e:
        logger.error(f"Error checking vision model: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500, detail = f"Failed to check vision model: {str(e)}"
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

        logger.info(
            f"Embedding check result for {model_name}: is_embedding={is_embedding}"
        )
        return EmbeddingCheckResponse(
            model_name = model_name,
            is_embedding = is_embedding,
        )

    except Exception as e:
        logger.error(f"Error checking embedding model: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500, detail = f"Failed to check embedding model: {str(e)}"
        )


@router.get("/gguf-variants", response_model = GgufVariantsResponse)
async def get_gguf_variants(
    repo_id: str = Query(
        ..., description = "HuggingFace repo ID (e.g. 'unsloth/gemma-3-4b-it-GGUF')"
    ),
    hf_token: Optional[str] = Query(
        None, description = "HuggingFace token for private repos"
    ),
    current_subject: str = Depends(get_current_subject),
):
    """
    List available GGUF quantization variants for a HuggingFace repo
    or a local directory (e.g. LM Studio model folder).

    Returns all available quantization variants (Q4_K_M, Q8_0, BF16, etc.)
    with file sizes, whether the model supports vision, and the recommended
    default variant.
    """
    try:
        from utils.models.model_config import is_local_path, list_local_gguf_variants

        # Local directory path (e.g. LM Studio models) — scan filesystem
        if is_local_path(repo_id):
            variants, has_vision = list_local_gguf_variants(repo_id)

            filenames = [v.filename for v in variants]
            best = _pick_best_gguf(filenames)
            default_variant = _extract_quant_label(best) if best else None

            return GgufVariantsResponse(
                repo_id = repo_id,
                variants = [
                    GgufVariantDetail(
                        filename = v.filename,
                        quant = v.quant,
                        size_bytes = v.size_bytes,
                        downloaded = True,  # all local variants are downloaded
                    )
                    for v in variants
                ],
                has_vision = has_vision,
                default_variant = default_variant,
            )

        # Remote HuggingFace repo — query HF API
        variants, has_vision = list_gguf_variants(repo_id, hf_token = hf_token)

        # Determine default variant
        filenames = [v.filename for v in variants]
        best = _pick_best_gguf(filenames)
        default_variant = _extract_quant_label(best) if best else None

        # Check which variants are fully downloaded in the HF cache.
        # For split GGUFs, ALL shards must be present -- sum cached bytes
        # per variant and compare against the expected total.
        # HF cache dir uses the exact case from the repo_id at download time,
        # which may differ from the canonical HF repo_id, so do a
        # case-insensitive match.
        cached_bytes_by_quant: dict[str, int] = {}
        try:
            import re as _re
            from huggingface_hub import constants as hf_constants

            # Sanitize repo_id: must be "owner/name" with safe chars only
            if not _is_valid_repo_id(repo_id):
                raise ValueError(f"Invalid repo_id format: {repo_id}")

            cache_dir = Path(hf_constants.HF_HUB_CACHE)
            target = f"models--{repo_id.replace('/', '--')}".lower()
            for entry in cache_dir.iterdir():
                if entry.name.lower() == target:
                    snapshots = entry / "snapshots"
                    if snapshots.is_dir():
                        for snap in snapshots.iterdir():
                            for f in _iter_gguf_paths(snap):
                                q = _extract_quant_label(f.name)
                                cached_bytes_by_quant[q] = (
                                    cached_bytes_by_quant.get(q, 0) + f.stat().st_size
                                )
                    break
        except Exception:
            pass

        def _is_fully_downloaded(variant) -> bool:
            cached = cached_bytes_by_quant.get(variant.quant, 0)
            if cached == 0 or variant.size_bytes == 0:
                return False
            # Allow small rounding tolerance (symlinks vs real sizes)
            return cached >= variant.size_bytes * 0.99

        return GgufVariantsResponse(
            repo_id = repo_id,
            variants = [
                GgufVariantDetail(
                    filename = v.filename,
                    quant = v.quant,
                    size_bytes = v.size_bytes,
                    downloaded = _is_fully_downloaded(v),
                )
                for v in variants
            ],
            has_vision = has_vision,
            default_variant = default_variant,
        )

    except Exception as e:
        logger.error(f"Error listing GGUF variants for '{repo_id}': {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to list GGUF variants: {str(e)}",
        )


@router.get("/gguf-download-progress")
async def get_gguf_download_progress(
    repo_id: str = Query(..., description = "HuggingFace repo ID"),
    variant: str = Query("", description = "Quantization variant (e.g. UD-TQ1_0)"),
    expected_bytes: int = Query(0, description = "Expected total download size in bytes"),
    current_subject: str = Depends(get_current_subject),
):
    """Return download progress by checking cached GGUF files for a specific variant.

    Tracks completed shard downloads in snapshots and in-progress downloads
    in the blobs directory (incomplete files).
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
                # Count completed .gguf files matching this variant in snapshots
                for f in _iter_gguf_paths(entry):
                    fname = f.name.lower().replace("-", "").replace("_", "")
                    if not variant_lower or variant_lower in fname:
                        downloaded_bytes += f.stat().st_size
                # Check blobs for in-progress downloads (.incomplete files)
                blobs_dir = entry / "blobs"
                if blobs_dir.is_dir():
                    for f in blobs_dir.iterdir():
                        if f.is_file() and f.name.endswith(".incomplete"):
                            in_progress_bytes += f.stat().st_size
                break

        total_progress_bytes = downloaded_bytes + in_progress_bytes
        progress = (
            min(total_progress_bytes / expected_bytes, 0.99)
            if expected_bytes > 0
            else 0
        )
        # Only report 1.0 when all bytes are in completed files (not in-progress)
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

    Prefers the most-recent snapshot dir (what `from_pretrained` actually
    points at). Falls back to the cache repo root. Returns the resolved
    realpath so symlinks under snapshots/ are followed back to blobs/.
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
    (.incomplete) downloads. Uses the HF API to determine the expected
    total size on the first call, then caches it for subsequent polls.
    Also returns ``cache_path``: the realpath of the snapshot directory
    (or the cache repo root if no snapshot exists yet) so the UI can
    show users where the weights actually live on disk.
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

        # Get expected size from HF API (cached per repo_id)
        expected_bytes = _get_repo_size_cached(repo_id)
        if expected_bytes <= 0:
            # Cannot determine total; report bytes only, no percentage
            return {
                "downloaded_bytes": downloaded_bytes,
                "expected_bytes": 0,
                "progress": 0,
                "cache_path": cache_path,
            }

        # Use 95% threshold for completion (blob deduplication can make
        # completed_bytes differ slightly from expected_bytes).
        # Do NOT use "no .incomplete files" as a completion signal --
        # HF downloads files sequentially, so between files there are
        # no .incomplete files even though the download is far from done.
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


def _all_hf_cache_scans():
    """Return scan_cache_dir results for the active, legacy, and default HF caches."""
    from huggingface_hub import scan_cache_dir
    from utils.paths import legacy_hf_cache_dir, hf_default_cache_dir

    scans = [scan_cache_dir()]
    seen: set[str] = set()
    try:
        # Resolve the active cache dir so we can dedup
        from huggingface_hub.constants import HF_HUB_CACHE

        seen.add(str(Path(HF_HUB_CACHE).resolve()))
    except Exception:
        pass

    for extra_fn in (legacy_hf_cache_dir, hf_default_cache_dir):
        extra = extra_fn()
        if extra.is_dir() and str(extra.resolve()) not in seen:
            seen.add(str(extra.resolve()))
            try:
                scans.append(scan_cache_dir(cache_dir = str(extra)))
            except Exception as exc:
                logger.warning("Could not scan HF cache %s: %s", extra, exc)
    return scans


def _is_gguf_filename(name: str) -> bool:
    return name.lower().endswith(".gguf")


def _is_mmproj_filename(name: str) -> bool:
    """Match GGUF vision-adapter (mmproj) files. Kept consistent with
    ``utils.models.model_config._is_mmproj``."""
    return "mmproj" in name.lower()


def _is_main_gguf_filename(name: str) -> bool:
    """A GGUF file that is a primary weight artifact, not an mmproj
    vision adapter."""
    return _is_gguf_filename(name) and not _is_mmproj_filename(name)


def _iter_gguf_paths(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and _is_gguf_filename(path.name):
            yield path


def _repo_gguf_size_bytes(repo_info) -> int:
    """Return the total on-disk size of primary GGUF weight files across
    all revisions, excluding mmproj vision-adapter files.

    Hugging Face hardlinks blobs shared between revisions, so this
    deduplicates by blob path (or, as a fallback, by revision commit
    hash + filename) to avoid double-counting the same bytes. Files
    with an unknown size (``size_on_disk is None``, e.g. a partial or
    interrupted download) are treated as zero bytes. mmproj files are
    excluded so that repos whose only ``.gguf`` artifact is a vision
    adapter are not classified as GGUF repos: the variant selector
    filters mmproj out and would otherwise show zero pickable variants.
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
    """Return True when any revision in a cached repo contains a
    primary GGUF weight file. Repos whose only ``.gguf`` artifact is
    an mmproj vision adapter are not treated as GGUF here."""
    return _repo_gguf_size_bytes(repo_info) > 0


@router.get("/cached-gguf")
async def list_cached_gguf(
    current_subject: str = Depends(get_current_subject),
):
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
                    total_size = _repo_gguf_size_bytes(repo_info)
                    if total_size == 0:
                        continue
                    key = repo_id.lower()
                    existing = seen_lower.get(key)
                    if existing is None or total_size > existing["size_bytes"]:
                        seen_lower[key] = {
                            "repo_id": repo_id,
                            "size_bytes": total_size,
                            "cache_path": str(repo_info.repo_path),
                        }
                except Exception as e:
                    repo_label = getattr(repo_info, "repo_id", "<unknown>")
                    logger.warning(f"Skipping cached GGUF repo {repo_label}: {e}")
                    continue
        cached = sorted(seen_lower.values(), key = lambda c: c["repo_id"])
        return {"cached": cached}
    except Exception as e:
        logger.error(f"Error listing cached GGUF repos: {e}", exc_info = True)
        return {"cached": []}


@router.get("/cached-models")
async def list_cached_models(
    current_subject: str = Depends(get_current_subject),
):
    """List non-GGUF model repos downloaded to HF cache, legacy Unsloth cache, and HF default cache."""
    _WEIGHT_EXTENSIONS = (".safetensors", ".bin")

    try:
        cache_scans = _all_hf_cache_scans()

        seen_lower: dict[str, dict] = {}
        for hf_cache in cache_scans:
            for repo_info in hf_cache.repos:
                try:
                    if repo_info.repo_type != "model":
                        continue
                    repo_id = repo_info.repo_id
                    if _repo_has_gguf_files(repo_info):
                        continue
                    total_size = sum(
                        (f.size_on_disk or 0)
                        for rev in repo_info.revisions
                        for f in rev.files
                    )
                    if total_size == 0:
                        continue
                    has_weights = any(
                        f.file_name.endswith(_WEIGHT_EXTENSIONS)
                        for rev in repo_info.revisions
                        for f in rev.files
                    )
                    if not has_weights:
                        continue
                    key = repo_id.lower()
                    existing = seen_lower.get(key)
                    if existing is None or total_size > existing["size_bytes"]:
                        seen_lower[key] = {
                            "repo_id": repo_id,
                            "size_bytes": total_size,
                        }
                except Exception as e:
                    repo_label = getattr(repo_info, "repo_id", "<unknown>")
                    logger.warning(f"Skipping cached model repo {repo_label}: {e}")
                    continue
        cached = sorted(seen_lower.values(), key = lambda c: c["repo_id"])
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

    When *variant* is provided, only the GGUF files matching that quant label
    are removed (e.g. ``UD-Q4_K_XL``).  Otherwise the entire repo is deleted.
    Refuses if the model is currently loaded for inference.
    """
    if not _is_valid_repo_id(repo_id):
        raise HTTPException(status_code = 400, detail = "Invalid repo_id format")

    # Check if model is currently loaded
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
                    # Delete the blob (actual data) and the snapshot symlink
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
            detail = f"Failed to delete cached model: {str(e)}",
        )


@router.get("/checkpoints", response_model = CheckpointListResponse)
async def list_checkpoints(
    outputs_dir: str = Query(
        default = str(outputs_root()),
        description = "Directory to scan for checkpoints",
    ),
    current_subject: str = Depends(get_current_subject),
):
    """
    List available checkpoints in the outputs directory.

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
        logger.error(f"Error listing checkpoints: {e}", exc_info = True)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to list checkpoints: {str(e)}",
        )
