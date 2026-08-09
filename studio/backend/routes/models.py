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
import threading
import time
import uuid
import weakref
from pathlib import Path
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query
from pydantic import BaseModel
from typing import List, NamedTuple, Optional
import structlog
from loggers import get_logger

# Dependency-light leaf (PEP 562 package init): no llama.cpp / torch import chain.
from core.inference.model_ids import display_model_name
from utils.utils import canonical_model_repo_id, log_and_http_error

import re as _re

_VALID_REPO_ID = _re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


class CachedModelRepo(BaseModel):
    repo_id: str
    size_bytes: int
    last_modified: Optional[float] = None
    # "text-to-image" for cached diffusers image repos; declared here or response_model drops it.
    task: Optional[str] = None
    # Snapshot incomplete (cancelled/partial download): the picker must not treat it as usable.
    partial: Optional[bool] = None
    # Diffusion-tagged repo with NO top-level model_index.json: needs from_single_file + a filename.
    single_file: Optional[bool] = None
    # True for an sd.cpp companion mirror (VAE / text encoders, no denoiser). Declared here or
    # response_model drops it and the flag never reaches the picker that has to filter on it.
    companion: Optional[bool] = None


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


# Shared with the hub inventory scans; private aliases kept for existing importers.
# ``_HF_REPO_ID_RE`` is the Hub repo id shape ("owner/name"); anything else is a path.
from utils.hidden_models import (
    _HF_REPO_ID_RE,
    _existing_resolved_path,
    _safe_resolve,
    is_hidden_model as _is_hidden_model,
)


def hidden_model_matchers() -> tuple[list[str], list[str], list[str]]:
    """Substring needles, exact repo ids, and exact resolved paths identifying
    infra models (the RAG embedder and the llama.cpp install validation probe)
    that pickers hide. Served by the ``/api/hub/hidden-models`` endpoint. A
    configured HF-repo embedder is published as its exact lowercased repo id
    (mirroring ``utils.hidden_models.is_hidden_model``) and a local-path
    embedder as its exact resolved path only: a generic basename like "model"
    must not substring-hide unrelated chat models."""
    from core.rag import config as rag_config

    needles = [
        # Validation probe repo + exact filename; .gguf so it won't hide unrelated *-GGUF repos.
        "ggml-org/models",
        "stories260k.gguf",
    ]
    exact_ids: list[str] = []
    exact_paths: list[str] = []
    for model in (
        rag_config.effective_embedding_model(),
        rag_config.effective_gguf_repo(),
    ):
        # Resolve a local path before the repo-id regex: "models/embedder" is a path, not a repo id.
        existing_path = _existing_resolved_path(model)
        if existing_path:
            exact_paths.append(existing_path.lower())
        elif _HF_REPO_ID_RE.match(model):
            exact_ids.append(model.lower())
        else:
            resolved = _safe_resolve(Path(model).expanduser())
            if resolved:
                exact_paths.append(resolved.lower())
    return needles, exact_ids, exact_paths


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
        _is_mtp_drafter,
        is_audio_input_type,
    )
    from core.inference import get_inference_backend
    from utils.paths import (
        is_local_path,
        normalize_path,
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
        _is_mtp_drafter,
        is_audio_input_type,
    )
    from core.inference import get_inference_backend
    from utils.paths import (
        is_local_path,
        normalize_path,
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
    from utils.hf_cache_settings import get_hf_cache_paths
    return get_hf_cache_paths().hub_cache


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


# Weight ``.bin`` files the local scanners accept, as opposed to companions like
# ``tokenizer.bin``. Mirrors ``_is_weight_file`` so every weight check agrees.
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


def _local_pipeline_index(d: Path) -> bool:
    """True when *d* is a diffusers PIPELINE root (top-level ``model_index.json``, weights in
    component subdirs), which ``_is_model_directory`` (root config + loose weights) rejects."""
    try:
        return (d / "model_index.json").is_file()
    except OSError:
        return False


def _is_gguf_companion_only_dir(path: Path) -> bool:
    """True for a folder whose entire content is GGUF companions -- a lone mmproj adapter, an
    MTP drafter, or both -- with nothing servable beside them.

    The scanners report ``model_format = None`` for such a folder, because neither companion is a
    primary weight, and that is also what a plain checkpoint reports. The custom-folder scan below
    validates GGUF rows through ``detect_gguf_model`` and waves the rest through, so without this
    the folder is published as a model that no loader can start.
    """
    try:
        if not path.is_dir():
            return False
        if (path / "config.json").exists() or (path / "adapter_config.json").exists():
            return False
        return any(path.glob("*.gguf")) and not _has_non_gguf_weights(path)
    except OSError:
        return False


def _scan_models_dir(models_dir: Path, *, limit: int | None = None) -> List[LocalModelInfo]:
    if not models_dir.exists() or not models_dir.is_dir():
        return []

    # A scan folder can point at a diffusers PIPELINE dir, which _is_model_directory rejects but the load path accepts.
    _is_self_model = _is_model_directory(models_dir) or _local_pipeline_index(models_dir)

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
            gguf_names = [p.name for p in child.glob("*.gguf")]
            has_gguf = bool(gguf_names)
            # mmproj alone is a vision adapter, not servable weights: decides presence, never format.
            has_main_gguf = any(_is_main_gguf_filename(n) for n in gguf_names)
            has_non_gguf_weights = _has_non_gguf_weights(child)
            has_config = (child / "config.json").exists() or (
                child / "adapter_config.json"
            ).exists()
            # A diffusers PIPELINE folder (weights in component subdirs) is missed above but loadable.
            has_pipeline_index = _local_pipeline_index(child)
            has_model_files = has_gguf or has_non_gguf_weights or has_config or has_pipeline_index
        except OSError:
            # Skip unreadable children rather than failing the scan.
            continue
        if not has_model_files:
            continue
        try:
            updated_at = child.stat().st_mtime
        except OSError:
            updated_at = None
        # A folder whose only weights are .gguf is GGUF-format even with a config.json (common for
        # HF GGUF repos, often without a -GGUF suffix), so surface the format for the UI.
        model_format = "gguf" if has_main_gguf and not has_non_gguf_weights else None
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
    if limit is None or len(found) < limit:
        for gguf_file in models_dir.glob("*.gguf"):
            if limit is not None and len(found) >= limit:
                break
            # A standalone mmproj is a vision adapter, not servable weights.
            if gguf_file.is_file() and _is_main_gguf_filename(gguf_file.name):
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

    # A scan folder can also point at a BARE single-file checkpoint dir (one loose .safetensors,
    # no configs): both checks reject it, but resolve_local_single_file loads it.
    if not found and (limit is None or limit > 0) and _has_non_gguf_weights(models_dir):
        try:
            updated_at = models_dir.stat().st_mtime
        except OSError:
            updated_at = None
        found.append(
            LocalModelInfo(
                id = str(models_dir),
                display_name = models_dir.name,
                path = str(models_dir),
                source = "models_dir",
                model_format = _dir_model_format(models_dir),
                updated_at = updated_at,
            ),
        )

    return found


def _scan_hf_cache(
    cache_dir: Path,
    *,
    active_cache: bool = True,
    classify_format: bool = True,
    variant_states = None,
) -> List[LocalModelInfo]:
    if not cache_dir.exists() or not cache_dir.is_dir():
        return []

    from hub.utils import inventory_scan as hf_cache_scan

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

        variant_state = (
            variant_states.for_repo("model", model_id, hub_cache = cache_dir)
            if variant_states is not None
            else None
        )
        partial = hf_cache_scan.is_snapshot_partial(
            "model", model_id, repo_dir, variant_state = variant_state
        )
        partial = partial or hf_cache_scan.is_gguf_repo_partial(
            model_id, repo_dir, variant_state = variant_state
        )

        load_id = model_id
        snapshot = _resolve_hf_cache_realpath(repo_dir)
        if not active_cache:
            load_id = snapshot or str(repo_dir.resolve())
        # Classify from the snapshot's own weights: a GGUF repo without a -GGUF suffix is common,
        # and leaving this unset makes every consumer guess from the name.
        model_format = (
            _dir_model_format(Path(snapshot), recursive = True)
            if snapshot and classify_format
            else None
        )
        found.append(
            LocalModelInfo(
                id = load_id,
                model_id = model_id,
                display_name = model_id.split("/")[-1],
                model_format = model_format,
                path = load_id if not active_cache else str(repo_dir),
                source = "hf_cache",
                active_cache = active_cache,
                partial = partial,
                updated_at = updated_at,
            ),
        )
    return found


def _dir_model_format(path: Path, recursive: bool = False) -> Optional[str]:
    """Return ``"gguf"`` for a directory whose only weights are ``.gguf`` files.

    LM Studio and custom GGUF folders frequently lack a ``-GGUF`` name suffix,
    so the UI relies on this hint to route them through the GGUF load path
    rather than treating them as plain local checkpoints. A directory whose only
    ``.gguf`` is an mmproj vision adapter is not one: the variant selector drops
    mmproj, so that path would find nothing to serve.

    ``recursive`` is for HF cache snapshots, which keep split quants in per-quant
    subdirectories: a flat glob sees no ``.gguf`` there and would report the
    snapshot as non-GGUF, hiding every sharded repo from the GGUF pickers. It looks
    one level down rather than walking the tree, because that is where split quants
    live and ``/api/models/local`` is async: an unbounded ``rglob`` per repo would
    have to exhaust every non-GGUF snapshot before concluding there is no GGUF,
    blocking the event loop on a large cache.
    """
    try:
        found = path.glob("*.gguf")
        if not any(_is_main_gguf_filename(p.name) for p in found):
            if not recursive:
                return None
            if not any(_is_main_gguf_filename(p.name) for p in path.glob("*/*.gguf")):
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

    # lm_dir may itself be a model directory (not a publisher); return it rather than skip it.
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
                if _is_main_gguf_filename(child.name) and child.is_file():
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

            # Surface a model-directory child directly instead of descending into it as a publisher.
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
                    elif _is_main_gguf_filename(model_dir.name) and model_dir.is_file():
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

    # Fallback: namespace by a hash of ollama_dir so two roots don't collide (cache path only).
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

        # Skip if the link already points at the same blob; size checks can reuse stale links.
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
                manifest = json.loads(tag_file.read_text(encoding = "utf-8-sig"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
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
                        cfg = json.loads(config_blob.read_text(encoding = "utf-8-sig"))
                        model_type = cfg.get("model_type", "")
                        file_type = cfg.get("file_type", "")
                    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
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


class _CompatLocalInventorySources(NamedTuple):
    hf_cache_dir: Path
    legacy_hf: Path
    hf_default: Path
    lm_dirs: tuple[Path, ...]
    known_hf_caches: tuple[Path, ...]


def _compat_local_inventory_sources() -> _CompatLocalInventorySources:
    from utils.paths import hf_default_cache_dir, legacy_hf_cache_dir, lmstudio_model_dirs
    from utils.hf_cache_settings import known_hf_hub_caches
    return _CompatLocalInventorySources(
        _resolve_hf_cache_dir(),
        legacy_hf_cache_dir(),
        hf_default_cache_dir(),
        tuple(lmstudio_model_dirs()),
        tuple(known_hf_hub_caches()),
    )


def collect_local_models(
    models_root: Path,
    *,
    custom_folders: Optional[list[dict]] = None,
    sources: Optional[_CompatLocalInventorySources] = None,
) -> List[LocalModelInfo]:
    """Scan ``models_root``, the HF caches, LM Studio dirs, and user scan folders,
    returning a deduplicated, hidden-filtered list of discovered local models.

    Shared by ``GET /models/local`` (the model picker) and the OpenAI-compatible
    catalog (``GET /v1/models``) so the UI and the API never drift. ``models_root``
    must already be validated/trusted by the caller.
    """
    from storage.studio_db import list_scan_folders
    from utils.models.model_config import detect_gguf_model

    sources = sources or _compat_local_inventory_sources()
    hf_cache_dir = sources.hf_cache_dir
    legacy_hf = sources.legacy_hf
    hf_default = sources.hf_default
    lm_dirs = sources.lm_dirs
    if custom_folders is None:
        try:
            custom_folders = list_scan_folders()
        except Exception as e:
            logger.warning("Could not load custom scan folders: %s", e)
            custom_folders = []

    local_models = _scan_models_dir(models_root)
    active_cache_real = _safe_resolve(hf_cache_dir)
    active_cache_key = os.path.normcase(active_cache_real) if active_cache_real else None
    seen_hf: set[str] = set()
    hf_sources: list[tuple[Path, bool]] = []
    for cache_dir in (
        hf_cache_dir,
        *sources.known_hf_caches,
        legacy_hf,
        hf_default,
    ):
        cache_real = _safe_resolve(cache_dir)
        if cache_real is None:
            continue
        cache_key = os.path.normcase(str(cache_real))
        if cache_key in seen_hf:
            continue
        seen_hf.add(cache_key)
        hf_sources.append((cache_dir, cache_key == active_cache_key))

    state_repositories = []
    state_cache_dirs = [cache_dir for cache_dir, _active_cache in hf_sources]
    state_cache_dirs.extend(Path(folder["path"]) for folder in custom_folders)
    for cache_dir in dict.fromkeys(state_cache_dirs):
        try:
            for repo_dir in cache_dir.glob("models--*"):
                repo_name = repo_dir.name[len("models--") :]
                if repo_name and repo_dir.is_dir():
                    state_repositories.append(("model", repo_name.replace("--", "/"), cache_dir))
        except OSError:
            continue
    try:
        from hub.utils import download_manifest
        variant_states = download_manifest.build_variant_state_index(
            state_repositories,
            active_hub_cache = hf_cache_dir,
        )
    except Exception as e:
        logger.warning("Could not build shared legacy Hub-state index: %s", e)
        variant_states = None

    for cache_dir, active_cache in hf_sources:
        local_models += _scan_hf_cache(
            cache_dir,
            active_cache = active_cache,
            variant_states = variant_states,
        )

    for lm_dir in lm_dirs:
        local_models += _scan_lmstudio_dir(lm_dir)

    # Scan user-added custom folders (per-folder cap).
    _MAX_MODELS_PER_FOLDER = 200
    for folder in custom_folders:
        folder_path = Path(folder["path"])
        try:
            # Filter Ollama .studio_links/ from generic scanners: duplicates and internal paths.
            _generic = [
                m
                for m in (
                    _scan_models_dir(folder_path, limit = _MAX_MODELS_PER_FOLDER)
                    + _scan_hf_cache(
                        folder_path,
                        active_cache = False,
                        variant_states = variant_states,
                    )
                    + _scan_lmstudio_dir(folder_path)
                )
                if not any(p in (".studio_links", "ollama_links") for p in Path(m.path).parts)
            ]
            custom_models = []
            for model in _generic:
                path = Path(model.path)
                is_gguf_row = model.model_format == "gguf" or _is_gguf_companion_only_dir(path)
                if not is_gguf_row or model.partial:
                    custom_models.append(model)
                    continue
                if path.is_dir():
                    patterns = ("*", "*/*") if model.source == "hf_cache" else ("*",)
                    if any(
                        detect_gguf_model(str(file), model_root = str(folder_path)) is not None
                        for pattern in patterns
                        for file in path.glob(pattern)
                        if not _safe_is_dir(file) and file.suffix.lower() == ".gguf"
                    ):
                        custom_models.append(model)
                elif (
                    detect_gguf_model(
                        model.path,
                        model_root = str(folder_path),
                    )
                    is not None
                ):
                    custom_models.append(model)
            if len(custom_models) < _MAX_MODELS_PER_FOLDER:
                custom_models += _scan_ollama_dir(
                    folder_path,
                    limit = _MAX_MODELS_PER_FOLDER - len(custom_models),
                )
        except OSError as e:
            logger.warning("Skipping unreadable scan folder %s: %s", folder_path, e)
            continue
        local_models += [m.model_copy(update = {"source": "custom"}) for m in custom_models]

    # Deduplicate, but always keep custom folder entries (keyed by (id, source)) so they show
    # in the "Custom Folders" UI section even when the model is also in the HF cache.
    deduped: dict[str, LocalModelInfo] = {}
    for model in local_models:
        semantic_id = model.model_id if model.source == "hf_cache" and model.model_id else model.id
        key = f"{semantic_id}\x00custom" if model.source == "custom" else semantic_id
        existing = deduped.get(key)
        prefer_model = existing is None
        if existing is not None and model.source == existing.source == "hf_cache":
            if model.partial != existing.partial:
                prefer_model = not model.partial
            elif bool(model.active_cache) != bool(existing.active_cache):
                prefer_model = bool(model.active_cache)
            else:
                prefer_model = (model.updated_at or 0) > (existing.updated_at or 0)
        if prefer_model:
            deduped[key] = model

    models = sorted(
        deduped.values(),
        key = lambda item: item.updated_at or 0,
        reverse = True,
    )
    return [m for m in models if not _is_hidden_model(m.id, m.model_id, m.path)]


_CompatLocalInventoryKey = tuple[Path, _CompatLocalInventorySources, tuple[str, ...], int]
_compat_local_inventory_flights: dict[
    tuple[asyncio.AbstractEventLoop, _CompatLocalInventoryKey], asyncio.Task[List[LocalModelInfo]]
] = {}


# Retrying a superseded scan is only worth it while invalidations are occasional;
# past this the endpoint must answer instead of restarting the walk forever.
_COMPAT_LOCAL_INVENTORY_MAX_ATTEMPTS = 8


class _CompatLocalCacheChanged(RuntimeError):
    def __init__(self, models: List[LocalModelInfo]) -> None:
        super().__init__("local inventory sources changed during the scan")
        # Carried so the attempt cap can serve the freshest scan it has instead
        # of looping forever or answering with nothing.
        self.models = models


def _compat_inventory_path_identity(path: object) -> str:
    """Canonical source identity for compatibility inventory flights."""
    raw = str(path)
    try:
        return os.path.normcase(os.path.realpath(os.path.expanduser(raw)))
    except (OSError, UnicodeError, ValueError):
        return os.path.normcase(raw)


async def _shared_compat_local_inventory_scan(
    models_root: Path, sources: Optional[_CompatLocalInventorySources] = None
) -> List[LocalModelInfo]:
    from storage.studio_db import list_scan_folders
    from hub.utils import inventory_scan as hf_cache_scan

    requested_sources = sources

    async def collect(
        expected_epoch: int, custom_folders: List[dict], scan_sources: _CompatLocalInventorySources
    ) -> List[LocalModelInfo]:
        models = await asyncio.to_thread(
            collect_local_models,
            models_root,
            custom_folders = custom_folders,
            sources = scan_sources,
        )
        if hf_cache_scan.hf_cache_scans_epoch() != expected_epoch:
            raise _CompatLocalCacheChanged(models)
        return models

    # Discard obsolete results and retry their waiters against the current cache epoch.
    superseded: Optional[List[LocalModelInfo]] = None
    for _attempt in range(_COMPAT_LOCAL_INVENTORY_MAX_ATTEMPTS):
        # Epoch first: the sources and folders below are read after it, so any
        # change to them lands in a later epoch and the post-scan check sees it.
        # A caller-supplied ``sources`` stays pinned - the /local route validated
        # its models_dir against exactly those roots.
        epoch = hf_cache_scan.hf_cache_scans_epoch()
        scan_sources = requested_sources or _compat_local_inventory_sources()
        try:
            custom_folders = await asyncio.to_thread(list_scan_folders)
        except Exception as e:
            logger.warning("Could not load custom scan folders: %s", e)
            custom_folders = []
        key: _CompatLocalInventoryKey = (
            Path(_compat_inventory_path_identity(models_root)),
            scan_sources,
            tuple(
                _compat_inventory_path_identity(folder.get("path", "")) for folder in custom_folders
            ),
            epoch,
        )
        try:
            return await hf_cache_scan.shared_scan(
                _compat_local_inventory_flights,
                key,
                lambda expected_epoch = epoch, folders = custom_folders, roots = scan_sources: (
                    collect(expected_epoch, folders, roots)
                ),
            )
        except _CompatLocalCacheChanged as changed:
            superseded = changed.models
            continue
    # Invalidations are outpacing the walk, so no scan will ever confirm as
    # current. Answer with the freshest one (the loop only reaches here through
    # the retry path, so there is always one) instead of rescanning forever.
    logger.warning("Compat local inventory kept racing cache invalidations; serving the last scan")
    return superseded


@router.get("/local", response_model = LocalModelListResponse)
async def list_local_models(
    models_dir: str = Query(
        default = "./models", description = "Directory to scan for local model folders"
    ),
    current_subject: str = Depends(get_current_subject),
):
    """List local model candidates from the models dir, HF caches, and LM Studio dirs."""
    # Resolve all scan directories up front.
    sources = _compat_local_inventory_sources()
    hf_cache_dir = sources.hf_cache_dir
    legacy_hf = sources.legacy_hf
    hf_default = sources.hf_default
    lm_dirs = sources.lm_dirs

    # Validate models_dir against an allowlist of trusted dirs. Only the trusted Path objects
    # are used for FS access; the user string is for matching only, never path construction.
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
        models = await _shared_compat_local_inventory_scan(models_root, sources)
        # Tag each model with its task so the Images picker can filter to diffusion.
        models = [m.model_copy(update = {"task": _local_model_task(m)}) for m in models]

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
    # Ollama layout: each manifest is JSON referencing content-addressable blobs. A manifest
    # alone is not enough -- a failed or pruned pull leaves it behind with the model blob
    # missing, so resolve the ``application/vnd.ollama.image.model`` layer to an on-disk blob
    # before counting it (mirrors _scan_ollama_dir), else the chip leads to an empty picker.
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
                    manifest = json.loads(m.read_text(encoding = "utf-8-sig"))
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
    # Generic weights: any GGUF/safetensors in a bounded BFS that skips hidden directories.
    # ``rglob`` walks in arbitrary order and counts every entry, so a large hidden subtree
    # could exhaust the budget before reaching real weights and falsely report "no model".
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
                    # PyTorch checkpoints; gate by name so tokenizer.bin and friends don't count as weights.
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

    try:
        for p in lmstudio_model_dirs():
            _add(p)
    except Exception as e:
        logger.warning("Failed to scan for LM Studio model directories: %s", e)

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


# Max children to stat when checking if a directory "looks like" it holds models.
_BROWSE_MODEL_HINT_PROBE = 64
# Hard cap on subdirectory entries so browsing ``/usr/lib`` can't stat-storm the process.
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
    from utils.paths import external_media
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
        media_roots = [
            *external_media.linux_run_media_mount_roots(),
            *external_media.macos_volume_roots(),
        ]
    if drive_roots is None:
        drive_roots = external_media.windows_drive_roots()
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
            # Bare POSIX root ("/"): equality above is the only match; don't authorize descendants.
            continue
        if drive.startswith(("\\\\", "//")) and not tail:
            # Bare UNC share root (\\server\share): os.path.commonpath raises on it, so authorize
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
        # Zero-component case: the requested path IS an allowlist root (legacy "/" or a drive root).
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


# Sync (def, not async) so FastAPI runs the blocking filesystem I/O in the threadpool: a
# disconnected mapped drive can make the probe wait out its timeout, which on the event
# loop would stall every other request. Matches the hub browse endpoint.
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
    from utils.paths import external_media
    from storage.studio_db import (
        contains_sensitive_path_component,
        is_denied_system_path,
        list_scan_folders,
    )

    # Probe removable-media and Windows drive roots once; allowlist and chips reuse the result.
    media_roots = [
        *external_media.linux_run_media_mount_roots(),
        *external_media.macos_volume_roots(),
    ]
    drive_roots = external_media.windows_drive_roots()
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
            # Bound by *visited*, not *appended*: a cap on len(entries) would never trigger in dirs
            # full of files. Counting visits caps worst-case work at ``_BROWSE_ENTRY_CAP``.
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
            # Hide denied system dirs (C:\Windows, /etc, ...) so they don't render as rows that then
            # 403 on descent. Resolve first so a symlink into a denied dir is hidden too.
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

    # Model-bearing first, then plain, then hidden; case-insensitive within each bucket.
    def _sort_key(e: BrowseEntry) -> tuple[int, str]:
        bucket = 0 if e.has_models else (2 if e.hidden else 1)
        return (bucket, e.name.lower())

    entries.sort(key = _sort_key)

    # Parent is None at the filesystem root and when it would leave the sandbox (else the
    # up-row would 403); users can still hop to other allowed roots via the chips.
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
        # Drop a denied system dir (e.g. a stale scan-folder row) so it never becomes a chip that
        # 403s on click. Drive roots stay: only their system subdirectories are denied.
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
    # Dirs used by other local-LLM tools (LM Studio, Ollama, ~/models); existing paths only.
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
        # Off-loop: building the singleton calls get_device(), which would freeze on the torch import.
        inference_backend = await asyncio.to_thread(get_inference_backend)

        default_models = inference_backend.default_models

        loaded_models = []
        for model_name, model_data in inference_backend.models.items():
            _is_vision = model_data.get("is_vision", False)
            _audio_type = model_data.get("audio_type")
            model_info = ModelDetails(
                id = model_name,
                name = display_model_name(model_name),
                is_vision = _is_vision,
                is_lora = model_data.get("is_lora", False),
                is_mlx = model_data.get("is_mlx", False),
                is_audio = model_data.get("is_audio", False),
                audio_type = _audio_type,
                has_audio_input = model_data.get("has_audio_input", False),
                model_type = derive_model_type(_is_vision, _audio_type),
            )
            loaded_models.append(model_info)

        # Active GGUF model (llama-server), labelled from the display id
        # /api/inference/status publishes; the id stays raw for agents-tab's path filter.
        from routes.inference import _llama_status_model_ids, get_llama_cpp_backend

        llama_backend = get_llama_cpp_backend()
        if llama_backend.is_loaded and llama_backend.model_identifier:
            display_id, _reported_identifier = _llama_status_model_ids(llama_backend)
            loaded_models.append(
                ModelDetails(
                    id = llama_backend.model_identifier,
                    name = display_model_name(display_id or llama_backend.model_identifier),
                    is_gguf = True,
                    is_vision = llama_backend.is_vision,
                    is_audio = getattr(llama_backend, "_is_audio", False),
                    audio_type = getattr(llama_backend, "_audio_type", None),
                )
            )

        # Combine default and loaded; prefer loaded entries for duplicate ids so runtime flags survive.
        all_models = []
        seen_ids = set()
        loaded_by_id = {model_info.id: model_info for model_info in loaded_models}

        for model_id in default_models:
            if model_id not in seen_ids:
                model_info = loaded_by_id.get(model_id) or ModelDetails(
                    id = model_id,
                    name = display_model_name(model_id),
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


_MODEL_WEIGHT_EXTENSIONS = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


def _get_model_size_bytes(model_name: str, hf_token: Optional[str] = None) -> Optional[int]:
    """Total size of model weight files from HF Hub."""
    try:
        from huggingface_hub import HfApi

        api = HfApi(token = hf_token)
        info = api.repo_info(model_name, repo_type = "model", token = hf_token)
        if not info.siblings:
            return None

        total = 0
        for sibling in info.siblings:
            if sibling.rfilename and sibling.rfilename.endswith(_MODEL_WEIGHT_EXTENSIONS):
                if sibling.size is not None:
                    total += sibling.size

        return total if total > 0 else None
    except Exception as e:
        logger.warning(f"Could not get model size for {model_name}: {e}")
        return None


def _get_snapshot_model_size_bytes(snapshot_path: str) -> Optional[int]:
    try:
        snapshot = Path(snapshot_path).resolve(strict = True)
        snapshots_dir = snapshot.parent.resolve(strict = True)
        repo_dir = snapshots_dir.parent.resolve(strict = True)
        if not snapshot.is_dir() or snapshots_dir.name != "snapshots" or not repo_dir.is_dir():
            return None
        blobs_dir = repo_dir / "blobs"
        resolved_blobs_dir = blobs_dir.resolve(strict = True) if blobs_dir.is_dir() else None
    except (OSError, RuntimeError, ValueError):
        return None

    total = 0
    scan_failed = False

    def _record_walk_error(_error: OSError) -> None:
        nonlocal scan_failed
        scan_failed = True

    try:
        for root, _, filenames in os.walk(
            snapshot,
            followlinks = False,
            onerror = _record_walk_error,
        ):
            root_path = Path(root)
            for filename in filenames:
                if not filename.endswith(_MODEL_WEIGHT_EXTENSIONS):
                    continue
                try:
                    candidate = (root_path / filename).resolve(strict = True)
                    if not candidate.is_file():
                        continue
                    if not candidate.is_relative_to(snapshot) and not (
                        resolved_blobs_dir is not None
                        and candidate.is_relative_to(resolved_blobs_dir)
                    ):
                        continue
                    total += candidate.stat().st_size
                except (OSError, RuntimeError, ValueError):
                    scan_failed = True
    except OSError:
        return None
    return total if total > 0 and not scan_failed else None


def _model_config_inspection_target(
    model_name: str, prefer_local_cache: bool, local_path: Optional[str]
) -> str:
    if not prefer_local_cache or is_local_path(model_name):
        return model_name
    from hub.utils.hf_cache_state import (
        latest_snapshot_from_cache_path,
        with_load_subdirs,
    )

    snapshot = latest_snapshot_from_cache_path(
        local_path,
        "model",
        canonical_model_repo_id(model_name),
        with_load_subdirs(model_name, ("config.json", "adapter_config.json")),
    )
    if snapshot is None:
        raise HTTPException(
            status_code = 404,
            detail = "Selected cached model is no longer available.",
        )
    return snapshot


@router.get("/config/{model_name:path}")
async def get_model_config(
    model_name: str,
    hf_token: Optional[str] = Query(None),
    prefer_local_cache: bool = False,
    local_path: Optional[str] = None,
    header_hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    """Get configuration for a specific model (wraps load_model_defaults)."""
    hf_token = _normalize_hf_token(header_hf_token) or _normalize_hf_token(hf_token)
    from core.inference.llama_cpp import _hf_offline_if_unreachable_for

    def _resolve(model_name: str) -> ModelDetails:
        # Each probe below can reach the hub, so the guard wraps the whole handler: offline they
        # must all resolve from the HF cache. Local paths stay on disk and skip the probe.
        with _hf_offline_if_unreachable_for(model_name):
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

            inspection_target = _model_config_inspection_target(
                model_name,
                prefer_local_cache,
                local_path,
            )
            config_dict = load_model_defaults(model_name)

            is_vision = is_vision_model(
                inspection_target,
                hf_token = hf_token,
                local_files_only = prefer_local_cache,
            )
            is_embedding = is_embedding_model(inspection_target, hf_token = hf_token)
            audio_type = detect_audio_type(
                inspection_target,
                hf_token = hf_token,
                local_files_only = prefer_local_cache,
            )

            is_lora = False
            base_model = None
            max_position_embeddings = None
            try:
                model_config = ModelConfig.from_identifier(
                    inspection_target,
                    hf_token = hf_token,
                )
                is_lora = model_config.is_lora
                base_model = model_config.base_model if is_lora else None
                max_position_embeddings = _get_max_position_embeddings(model_config)
            except Exception:
                pass

            # Fallback: raw config.json (declarative fields only) -- must never run a repo's auto_map.
            if max_position_embeddings is None:
                try:
                    from utils.transformers_version import _load_config_json
                    from types import SimpleNamespace

                    _cfg = _load_config_json(inspection_target, hf_token = hf_token)
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
                model_size_bytes = (
                    _get_snapshot_model_size_bytes(inspection_target)
                    if prefer_local_cache
                    else _get_model_size_bytes(model_name, hf_token)
                ),
            )

    try:
        # Off the loop: the guard blocks on DNS + HEAD + TCP, stalling every other request.
        return await asyncio.to_thread(_resolve, model_name)

    except HTTPException:
        raise
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
    prefer_local_cache: bool = Body(False, embed = True),
    model_local_path: Optional[str] = Body(None, embed = True),
    model_snapshot_path: Optional[str] = Body(None, embed = True),
    model_snapshot_repo_id: Optional[str] = Body(None, embed = True),
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
        from utils.security import (
            load_scan_target,
            preflight_remote_code_consent_for_targets,
            security_load_subdirs,
        )

        local_model = is_local_path(model_name)
        if not local_model:
            model_name = resolve_cached_repo_id_case(model_name)
        scan_target = model_name
        exact_snapshot_path = (
            model_snapshot_path.strip()
            if isinstance(model_snapshot_path, str) and model_snapshot_path.strip()
            else None
        )
        exact_snapshot_repo_id = model_name
        if isinstance(model_snapshot_repo_id, str):
            snapshot_repo_id = model_snapshot_repo_id.strip()
            # Namespace-less Hub ids like "gpt2" are valid, so use the shared validator, not the regex.
            from hub.utils.paths import is_valid_repo_id as _shared_is_valid_repo_id

            if snapshot_repo_id and not _shared_is_valid_repo_id(snapshot_repo_id):
                raise HTTPException(
                    status_code = 400,
                    detail = "Invalid model snapshot repository ID.",
                )
            if snapshot_repo_id:
                exact_snapshot_repo_id = snapshot_repo_id
        if local_model:
            normalized_model_name = normalize_path(model_name)
            try:
                scan_target = str(Path(normalized_model_name).expanduser().resolve(strict = False))
            except (OSError, RuntimeError, ValueError):
                scan_target = normalized_model_name
        if exact_snapshot_path and not local_model:
            exact_snapshot_repo_id = resolve_cached_repo_id_case(exact_snapshot_repo_id)
            scan_target = _model_config_inspection_target(
                exact_snapshot_repo_id,
                True,
                normalize_path(exact_snapshot_path),
            )
        elif prefer_local_cache is True and not local_model:
            from core.training.training import _resolve_model_snapshot
            local_path = normalize_path(model_local_path) if model_local_path else None
            scan_target = _resolve_model_snapshot(model_name, local_path) or model_name
        # Scan the adapter AND the base together (a LoRA runs both repos' code), pinned by one
        # combined fingerprint. Snapshot the primary's cache state BEFORE resolving the base: that
        # resolve downloads adapter_config.json, which would hide the adapter from cleanup on decline.
        primary_cache_target, _ = load_scan_target(scan_target, ())
        try:
            _primary_preexisting = is_local_path(primary_cache_target) or _repo_in_any_hf_cache(
                primary_cache_target
            )
        except Exception:
            _primary_preexisting = True
        requested_scan_target = scan_target
        requested_security_targets = [requested_scan_target]
        try:
            from utils.models.model_config import get_base_model_from_lora_identifier

            # Resolve a LOCAL or REMOTE adapter's base so its code/weights are scanned too.
            _base = get_base_model_from_lora_identifier(requested_scan_target, hf_token)
            if _base:
                requested_security_targets.append(_base)
        except Exception:
            pass
        security_targets: list[str] = []
        consent_load_subdirs: dict[str, tuple] = {}
        for _requested_target in dict.fromkeys(requested_security_targets):
            _subdirs = security_load_subdirs(_requested_target, hf_token)
            if _requested_target == requested_scan_target and requested_scan_target != model_name:
                _subdirs = tuple(
                    dict.fromkeys((*_subdirs, *security_load_subdirs(model_name, hf_token)))
                )
            _target, _subdirs = load_scan_target(_requested_target, _subdirs)
            if _target not in consent_load_subdirs:
                security_targets.append(_target)
                consent_load_subdirs[_target] = ()
            _subdirs = tuple(dict.fromkeys((*consent_load_subdirs[_target], *_subdirs)))
            consent_load_subdirs[_target] = _subdirs
        # Record every repo OUR scan is first to pull into the cache (adapter, base, and external
        # auto_map repos), so a decline purges exactly what was downloaded. Computed BEFORE the
        # preflight downloads, against every cache the discard searches, so pre-existing repos stay.
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
                _target,
                preexisting = _primary_preexisting if _target == primary_cache_target else None,
            )
            for _ext in external_auto_map_repos(
                _target,
                hf_token,
                load_subdirs = consent_load_subdirs[_target],
            ):
                external_refs.append(_ext)
                _mark_scan_created(_ext)
        decision = preflight_remote_code_consent_for_targets(
            security_targets,
            hf_token = hf_token,
            subject = current_subject,
            load_subdirs_by_target = consent_load_subdirs,
        )
        payload = decision.response_payload()
        payload["model_name"] = exact_snapshot_repo_id if exact_snapshot_path else model_name
        payload["requires_trust_remote_code"] = decision.has_remote_code
        # Prior approval lets the dialog be skipped; the scan still ran, so this is a real match.
        payload["already_approved"] = (
            decision.has_remote_code
            and not decision.blocked
            and decision.reason == "approved by fingerprint"
        )
        # created_by_scan = primary flag (older clients); scan_created_repos drives cleanup.
        payload["created_by_scan"] = primary_cache_target in scan_created_repos
        payload["scan_created_repos"] = scan_created_repos
        # Provider tag decided here, where locality/scan scope/external refs are known.
        provider_target = exact_snapshot_repo_id if exact_snapshot_path else model_name
        if requested_scan_target == model_name and primary_cache_target != model_name:
            provider_target = primary_cache_target
        payload["provider"] = _consent_provider(provider_target, security_targets, external_refs)

        # Malware gate (metadata-only): HF-flagged unsafe files, orthogonal to remote code.
        from utils.security import evaluate_file_security

        unsafe_files: list = []
        security_blocked = False
        for _target in security_targets:
            _sec = evaluate_file_security(
                _target,
                hf_token = hf_token,
                load_subdirs = consent_load_subdirs[_target],
            )
            security_blocked = security_blocked or _sec.blocked
            unsafe_files.extend(_sec.unsafe_files)
        payload["unsafe_files"] = unsafe_files
        payload["security_blocked"] = security_blocked
        if security_blocked:
            # Non-approvable hard block: hides "Enable and continue" while forcing the dialog open.
            payload["approvable"] = False
            payload["requires_trust_remote_code"] = True
            payload["error_kind"] = "malware_blocked"
        return payload
    except HTTPException:
        raise
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
        from hub.services.models.deletion import _loaded_id_matches_repo
        from routes.inference import get_llama_cpp_backend

        llama_backend = get_llama_cpp_backend()
        if llama_backend.is_loaded and llama_backend.model_identifier:
            if _loaded_id_matches_repo(llama_backend.model_identifier, model_name):
                return {"deleted": False, "reason": "loaded"}
    except Exception:
        pass
    try:
        # Peek, not construct: no orchestrator means no active model, and building one hits get_device().
        from core.inference.orchestrator import peek_inference_backend
        inference_backend = peek_inference_backend()
        if inference_backend is not None and inference_backend.active_model_name:
            if _loaded_id_matches_repo(inference_backend.active_model_name, model_name):
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


@router.get("/diffusion-loras")
async def scan_diffusion_loras(
    family: Optional[str] = Query(
        default = None, description = "Filter to LoRAs compatible with this diffusion family"
    ),
    current_subject: str = Depends(get_current_subject),
):
    """List diffusion image LoRA adapters for the Images workflow.

    Merges the curated catalog with local files in ``<studio_home>/loras/diffusion``,
    optionally filtered to the loaded model's family. Cheap: one directory scan, no network
    (a hub adapter is only downloaded when actually selected for a generation). Distinct from
    ``/loras`` above, which lists trained/exported TEXT adapters.
    """
    from core.inference import diffusion_lora

    entries = diffusion_lora.list_loras(family = family)
    return {
        "loras": [
            {
                "id": e.id,
                "display_name": e.display_name,
                "source": e.source,
                "format": e.fmt,
                "families": list(e.families),
                "size_bytes": e.size_bytes,
                "weight_default": e.weight_default,
            }
            for e in entries
        ],
        "loras_dir": str(diffusion_lora.loras_dir()),
    }


@router.get("/diffusion-controlnets")
async def scan_diffusion_controlnets(
    family: Optional[str] = Query(
        default = None, description = "Filter to ControlNets compatible with this diffusion family"
    ),
    current_subject: str = Depends(get_current_subject),
):
    """List diffusion ControlNet models for the Images workflow.

    Merges the curated, family-tagged catalog with local model folders in
    ``<studio_home>/controlnets/diffusion``, optionally filtered to the loaded model's family.
    Cheap: one directory scan, no network (a hub model is only downloaded when selected).
    """
    from core.inference import diffusion_controlnet

    entries = diffusion_controlnet.list_controlnets(family = family)
    return {
        "controlnets": [
            {
                "id": e.id,
                "display_name": e.display_name,
                "source": e.source,
                "families": list(e.families),
                "control_types": list(e.control_types),
                "is_union": e.is_union,
            }
            for e in entries
        ],
        "control_types": list(diffusion_controlnet.CONTROL_TYPES),
        "controlnets_dir": str(diffusion_controlnet.controlnets_dir()),
    }


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


def _active_diffusion_backend():
    """The live Images engine, or None when this install has no diffusion stack.

    Fails OPEN on import: a chat-only install (no diffusers) must still be able to delete
    its fine-tuned models. Reading the returned backend's state is what fails closed.
    """
    try:
        from core.inference.diffusion_engine_router import get_active_diffusion_engine
        return get_active_diffusion_engine()
    except Exception as e:
        logger.debug(f"Images engine unavailable during delete guard: {e}")
        return None


def _active_video_backend():
    """The live Video backend, or None when this install has no video stack."""
    try:
        from core.inference.video import get_video_backend
        return get_video_backend()
    except Exception as e:
        logger.debug(f"Video backend unavailable during delete guard: {e}")
        return None


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
            # The diffusion (Images) trainer is a second independent run on the same storage root, so
            # checking only the LLM backend let a delete rmtree a live run's output directory.
            from core.training.diffusion_training_service import get_diffusion_training_service

            if get_diffusion_training_service().is_active():
                raise HTTPException(
                    status_code = 409,
                    detail = (
                        "Cannot delete trained models while diffusion (Images) training is running"
                    ),
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
        # Peek: building an orchestrator to learn there is none reaches get_device() (a torch import).
        from core.inference.orchestrator import peek_inference_backend
        inference_backend = peek_inference_backend()
        if inference_backend is not None:
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

    # Every guard above is chat-only, and Images / Video hold their own pipelines: a local model
    # loads by path, so rmtree would pull weights from under a live engine. Cached matches by id.
    for label, get_backend in (
        ("Images", _active_diffusion_backend),
        ("Video", _active_video_backend),
    ):
        backend = get_backend()
        if backend is None:
            continue
        try:
            status = backend.status()
            held = (
                [status.get(key) for key in ("repo_id", "base_repo")]
                if status.get("loaded")
                else []
            )
            held += list(getattr(backend, "loaded_repo_ids", tuple)())
            if any(h and _loaded_model_matches_deleted_path(str(h), target_path) for h in held):
                raise HTTPException(
                    status_code = 400,
                    detail = "Unload the model before deleting",
                )
            if any(
                _loading_model_matches_deleted_path(lid, target_path)
                for lid in getattr(backend, "loading_repo_ids", tuple)()
            ):
                raise HTTPException(
                    status_code = 409,
                    detail = "Cannot delete a model while it is loading",
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning("Could not check the %s model before delete: %s", label, e)
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
    header_hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    """
    Check if a model is a vision model.

    This endpoint wraps the backend is_vision_model function.
    """
    hf_token = _normalize_hf_token(header_hf_token) or _normalize_hf_token(hf_token)
    try:
        logger.info(f"Checking if vision model: {model_name}")
        # Authenticate so a gated/private VLM classifies correctly (else 404 -> non-vision). Offline
        # the guard keeps this on the HF cache; a local path resolves from disk and skips the probe.
        from core.inference.llama_cpp import _hf_offline_if_unreachable_for

        # Off-loop: the probes block and is_vision_model()'s lazy sets can import transformers.
        def _check():
            with _hf_offline_if_unreachable_for(model_name):
                return is_vision_model(model_name, hf_token = hf_token)

        is_vision = await asyncio.to_thread(_check)

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
    header_hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    """
    Check if a model is an embedding model.

    This endpoint wraps the backend is_embedding_model function.
    """
    hf_token = _normalize_hf_token(header_hf_token) or _normalize_hf_token(hf_token)
    try:
        logger.info(f"Checking if embedding model: {model_name}")
        # Same guard as /check-vision: is_embedding_model hits the hub with a 15s timeout.
        from core.inference.llama_cpp import _hf_offline_if_unreachable_for

        def _check():
            with _hf_offline_if_unreachable_for(model_name):
                return is_embedding_model(model_name, hf_token = hf_token)

        is_embedding = await asyncio.to_thread(_check)

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


# Budget for the walk below: a slow volume or large tree can outlast the listing.
_NATIVE_CONTEXT_READ_TIMEOUT_SECONDS = 5.0
# Backstop the walk's own budget cannot cover: a single syscall that never returns. Longer
# than the walk budget, so a responding filesystem always ends the walk itself.
_NATIVE_CONTEXT_HARD_TIMEOUT_SECONDS = 8.0
# Concurrent reads. A read stranded on a hung mount holds its slot, so retries wait.
_NATIVE_CONTEXT_MAX_CONCURRENT_READS = 4
_NATIVE_CONTEXT_SLOTS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _native_context_slots() -> asyncio.Semaphore:
    """Per running loop, since an asyncio primitive cannot be shared across loops."""
    loop = asyncio.get_running_loop()
    slots = _NATIVE_CONTEXT_SLOTS.get(loop)
    if slots is None:
        slots = asyncio.Semaphore(_NATIVE_CONTEXT_MAX_CONCURRENT_READS)
        _NATIVE_CONTEXT_SLOTS[loop] = slots
    return slots


def _settle_native_context(
    slots: asyncio.Semaphore, future: "asyncio.Future", value: Optional[int]
) -> None:
    slots.release()
    if not future.done():
        future.set_result(value)


async def _read_native_context_length_bounded(model: str, is_local: bool) -> Optional[int]:
    """``_read_native_context_length`` off the event loop, with a hard bound.

    Reporting None costs a pre-filled context field; waiting costs the whole variant
    listing, which is what left the picker on "Loading variants…". Runs on a daemon
    thread, not a pool: a stranded read must not join at interpreter exit, which would
    hang shutdown for as long as the mount stays hung. Waiting for a slot is awaited
    rather than skipped, so ordinary concurrent reads queue instead of losing their
    length; the wait and the read share one budget.
    """
    slots = _native_context_slots()
    began = time.monotonic()
    try:
        await asyncio.wait_for(slots.acquire(), timeout = _NATIVE_CONTEXT_HARD_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.debug("native context read for '%s' waited out its slot; reporting none", model)
        return None

    remaining = _NATIVE_CONTEXT_HARD_TIMEOUT_SECONDS - (time.monotonic() - began)
    loop = asyncio.get_running_loop()
    future: "asyncio.Future" = loop.create_future()

    def worker() -> None:
        try:
            value = _read_native_context_length(model, is_local = is_local)
        except Exception:
            value = None
        try:
            loop.call_soon_threadsafe(_settle_native_context, slots, future, value)
        except RuntimeError:
            pass  # loop already closed; nothing is waiting on this

    if remaining <= 0:
        slots.release()
        return None
    try:
        threading.Thread(target = worker, name = "native-ctx", daemon = True).start()
    except RuntimeError:
        slots.release()  # thread never ran, so it will never release
        return None

    try:
        return await asyncio.wait_for(future, timeout = remaining)
    except asyncio.TimeoutError:
        logger.debug("native context read for '%s' did not return; reporting none", model)
        return None


def _read_native_context_length(repo_id: str, is_local: bool) -> Optional[int]:
    """Native max context from a downloaded GGUF for this repo, or None.

    The value is identical across quants, so reading one non-mmproj shard's
    header is enough. Only resolves once a file is on disk. Never raises.

    Bounded by ``_NATIVE_CONTEXT_READ_TIMEOUT_SECONDS``: this only pre-fills a
    context field on an already selectable row, so a dragging walk reports None
    rather than holding the variant listing open. Checked between files, and
    files already read stay cached, so a later request resumes.
    """
    try:
        from utils.models.gguf_metadata import read_gguf_context_length

        # Before cache discovery (also filesystem I/O): started after, a slow enumeration would hand the walk a fresh budget.
        deadline = time.monotonic() + _NATIVE_CONTEXT_READ_TIMEOUT_SECONDS
        if is_local:
            roots = [Path(repo_id)]
        else:
            from hub.utils.hf_cache_state import iter_repo_cache_dirs
            if not _is_valid_repo_id(repo_id):
                return None
            roots = list(iter_repo_cache_dirs("model", repo_id))

        for root in roots:
            if time.monotonic() >= deadline:
                logger.debug("native context read for '%s' out of budget", repo_id)
                return None
            for f in _iter_gguf_paths(root, deadline):
                if time.monotonic() >= deadline:
                    logger.debug("native context read for '%s' out of budget", repo_id)
                    return None
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
        if is_local:
            roots = [Path(repo_id)]
        else:
            from hub.utils.hf_cache_state import iter_repo_cache_dirs

            if not _is_valid_repo_id(repo_id):
                return None, 0
            roots = []
            for entry in iter_repo_cache_dirs("model", repo_id):
                snaps = entry / "snapshots"
                if snaps.is_dir():
                    roots.extend(s for s in snaps.iterdir() if s.is_dir())

        want = _normalized_quant_label(quant)
        best_total = 0
        best_first: Optional[str] = None
        for root in roots:
            matches: list[tuple[str, Path]] = []
            total = 0
            for f in _iter_gguf_paths(root):
                try:
                    rel = f.relative_to(root).as_posix()
                except ValueError:
                    rel = f.name
                q = _main_variant_gguf_label(rel)
                if q is None or _normalized_quant_label(q) != want:
                    continue
                try:
                    total += f.stat().st_size
                except OSError:
                    continue
                matches.append((rel, f))
            # Prefer the most complete snapshot so a partial older revision can't underestimate bytes.
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
    cache_type_kv: Optional[str] = Query(
        None,
        description = "KV cache dtype (e.g. q8_0, q4_0, q5_0, iq4_nl, f32)",
    ),
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
    prefer_local_cache: bool = False,
    offline: bool = False,
    local_path: Optional[str] = None,
    hf_token: Optional[str] = Query(None, description = "HuggingFace token for private repos"),
    hf_token_header: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    """List GGUF quantization variants for a HF repo or local directory."""
    try:
        hf_token = _normalize_hf_token(hf_token_header) or _normalize_hf_token(hf_token)
        from hub.services.models import gguf_variants as hub_gguf_variants

        answer = await hub_gguf_variants.get_gguf_variants_answer(
            repo_id,
            prefer_local_cache = prefer_local_cache,
            offline = offline,
            local_path = local_path,
            hf_token = hf_token,
        )
        response = answer.response
        # The copy the listing answered from, else the pin; both beat a repo-wide walk.
        context_model = (
            answer.context_source
            or hub_gguf_variants.pinned_snapshot_for_request(repo_id, local_path)
            or repo_id
        )
        local = is_local_path(context_model)

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
                    partial = bool(getattr(v, "partial", False)),
                    cleanable = bool(getattr(v, "cleanable", False)),
                )
                for v in response.variants
            ],
            has_vision = response.has_vision,
            default_variant = response.default_variant,
            context_length = await _read_native_context_length_bounded(context_model, local),
            resolved_locally = bool(getattr(response, "resolved_locally", False)),
            loadable_variants = getattr(response, "loadable_variants", None),
            loadable = getattr(response, "loadable", None),
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
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    """Compatibility route backed by the shared multi-cache progress service."""
    from hub.services.models import downloads
    return await downloads.get_gguf_download_progress_response(
        repo_id,
        variant = variant,
        expected_bytes = expected_bytes,
        hf_token = hf_token,
    )


def _resolve_hf_cache_realpath(repo_dir: Path) -> Optional[str]:
    """Most useful on-disk path for a HF cache repo.

    Delegates to the Hub scanner's function of the same name so this route and
    ``/api/hub/local-models`` name one directory: the newest snapshot dir, ties broken by
    ``snapshot_selection_key``.
    """
    from hub.utils import inventory_scan as hf_cache_scan
    return hf_cache_scan.resolve_hf_cache_realpath(repo_dir)


@router.get("/download-progress")
async def get_download_progress(
    repo_id: str = Query(..., description = "HuggingFace repo ID"),
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    """Compatibility route backed by the shared multi-cache progress service."""
    from hub.services.models import downloads
    return await downloads.get_download_progress_response(repo_id, hf_token = hf_token)


def _repo_in_any_hf_cache(model_name: str) -> bool:
    """Whether ``model_name`` already exists in ANY HF cache the discard searches
    (active, legacy, default).

    ``created_by_scan`` must be True only when the scan itself first pulled the repo;
    checking just the active cache (``get_cache_path``) would mark a repo the user
    already had in a legacy/default cache as scan-created, so declining the consent
    would delete a model they did not download via the scan. Mirrors the cache set in
    ``_all_hf_cache_scans`` but only probes for the one repo dir (cheap, no full scan).
    """
    from utils.paths import resolve_cached_repo_id_case

    dirname = f"models--{resolve_cached_repo_id_case(model_name).replace('/', '--')}"
    dirname_lower = dirname.lower()
    from hub.utils.hf_cache_state import hf_cache_roots

    candidates = hf_cache_roots()
    # resolve_cached_repo_id_case only normalizes the ACTIVE cache, but discard deletes
    # case-insensitively across all caches, else a pre-existing case-variant is deleted on decline.
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
    from hub.utils.inventory_scan import all_hf_cache_scans
    return all_hf_cache_scans()


def _is_gguf_filename(name: str) -> bool:
    return name.lower().endswith(".gguf")


def _is_mmproj_filename(name: str) -> bool:
    """Match GGUF vision-adapter (mmproj) files. Consistent with
    ``utils.models.model_config._is_mmproj``."""
    return "mmproj" in name.lower()


def _is_main_gguf_filename(name: str) -> bool:
    """A primary GGUF weight, not an mmproj vision adapter or an MTP drafter. Same rule as
    ``hub.services.models.common``; pass a snapshot-relative path to catch ``MTP/`` copies too."""
    return _is_gguf_filename(name) and not _is_mmproj_filename(name) and not _is_mtp_drafter(name)


def _recovered_repo_is_unusable_by_repo_id(repo_info) -> bool:
    """See hub.utils.inventory_scan; False for anything upstream already returns."""
    from hub.utils.inventory_scan import recovered_repo_is_unusable_by_repo_id as impl
    return impl(repo_info)


def _repo_id_will_not_resolve(repo_cache_dir: Path) -> bool:
    """See hub.utils.inventory_scan; True only in the dangling refs/main window."""
    from hub.utils.inventory_scan import repo_id_will_not_resolve as impl
    return impl(repo_cache_dir)


def _default_ref_offers_no_whole_quant(repo_cache_dir: Path) -> bool:
    """See hub.utils.inventory_scan; True when refs/main resolves onto a torn quant."""
    from hub.utils.inventory_scan import default_ref_offers_no_whole_quant as impl
    return impl(repo_cache_dir)


def _gguf_copy_is_usable(repo_info, load_id: Optional[str]) -> bool:
    """Whether this copy of the repo holds a quant a load can reach.

    A pinned copy names a complete snapshot. An unpinned one is usable when its id resolves onto a
    whole quant, which is exactly what withheld the pin.
    """
    if load_id:
        return True
    try:
        repo_path = Path(repo_info.repo_path)
        return not _repo_id_will_not_resolve(repo_path) and not _default_ref_offers_no_whole_quant(
            repo_path
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _snapshot_has_gguf_projector(snapshot: str) -> bool:
    """See hub.utils.inventory_scan; reads the same walk the variant lister reports from."""
    from hub.utils.inventory_scan import snapshot_has_gguf_projector as impl
    return impl(Path(snapshot))


def _cached_repo_file_name(file_obj) -> str:
    """Snapshot-relative name for a cached file: huggingface_hub records the bare ``file_name``,
    which cannot tell an ``MTP/`` drafter from a quant."""
    from hub.services.models.cache_inventory import _cached_repo_file_name as impl
    return impl(file_obj)


def _main_variant_gguf_label(rel_path: str) -> Optional[str]:
    name = rel_path.rsplit("/", 1)[-1]
    if not _is_main_gguf_filename(name):
        return None
    if _is_mtp_drafter(rel_path):
        return None
    label = _extract_quant_label(rel_path)
    if _is_big_endian_gguf_path(rel_path, label):
        return None
    return label


def _normalized_quant_label(label: str) -> str:
    return label.lower().replace("-", "").replace("_", "")


def _repo_has_mmproj(repo_info) -> bool:
    """True if the repo ships a GGUF vision adapter (mmproj), so it can
    take image inputs. Cheap: scans already-listed file names only."""
    return any(
        _is_mmproj_filename(f.file_name) for revision in repo_info.revisions for f in revision.files
    )


def _cached_gguf_row_has_vision(repo_info, load_id: Optional[str]) -> bool:
    """Whether the copy this row loads ships a projector.

    The loader opens the projector out of the snapshot it loads from, so one in a revision the
    load never reaches is not vision support. A pinned row is judged on its snapshot, an unpinned
    one on the first snapshot the load's own ordering finds a quant in. Judged per snapshot, not
    per file: a split quant can sit in a subdirectory while the projector sits at the root.
    """
    if load_id:
        return _snapshot_has_gguf_projector(load_id)
    # No projector in any revision means none to reach, and saves a cache walk.
    if not _repo_has_mmproj(repo_info):
        return False
    try:
        from hub.utils.gguf import iter_snapshots_preferring_whole, list_local_gguf_variants

        # The row describes this copy; a duplicate in another root is one the load never reaches.
        root = Path(repo_info.repo_path).parent
        for snapshot in iter_snapshots_preferring_whole(repo_info.repo_id, None, root = root):
            variants, has_vision = list_local_gguf_variants(str(snapshot))
            if variants:
                return bool(has_vision)
    except Exception:
        pass
    # Nothing on disk to load, so the row describes the repo rather than a copy of it.
    return True


def _iter_gguf_paths(root: Path, deadline: Optional[float] = None):
    """GGUF files under ``root``. With a ``deadline`` (time.monotonic), gives up mid-walk:
    only .gguf files are yielded, so a large tree can walk for a long time yielding nothing,
    and a caller checking its budget per yield would never get to check it."""
    for path in root.rglob("*"):
        if deadline is not None and time.monotonic() >= deadline:
            return
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
            # Snapshot-relative: only the directory tells an MTP/ drafter from a primary quant.
            name = _cached_repo_file_name(f)
            if _is_main_gguf_filename(name):
                blob_path = getattr(f, "blob_path", None)
                size = f.size_on_disk or 0
                if blob_path:
                    unique_blobs[str(blob_path)] = size
                else:
                    unique_blobs[f"{rev_id}:{name}"] = size
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
            if _is_main_gguf_filename(_cached_repo_file_name(f)):
                latest = max(latest, _blob_mtime(f))
    return latest


# GGUF general.architecture values denoting a diffusion (image) model; lets the Images picker show only image GGUFs.
_DIFFUSION_GGUF_ARCHS = frozenset(
    {
        # ONLY the families the diffusion backend can assemble. Other diffusion archs would pass this filter then 400 in validate_load.
        "flux",  # flux.1
        "flux2",  # flux.2-klein
        "qwen_image",  # qwen-image
        "qwenimage",
        "z_image",  # z-image
        "zimage",
    }
)

# Diffusion / image-video GGUF archs the backend can NOT assemble yet (LlamaCppBackend._DIFFUSION_ARCHES minus the loadable set).
_UNSUPPORTED_DIFFUSION_GGUF_ARCHS = frozenset(
    {
        "sd1",
        "sd3",
        "sdxl",
        "aura",
        "hidream",
        "cosmos",
        "hyvid",
    }
)

# Archs shared by a buildable family and a non-buildable one. Z-Image's DiT is a Lumina2
# derivative, so its GGUFs declare "lumina2"; these resolve from the repo/file name.
_AMBIGUOUS_DIFFUSION_GGUF_ARCHS = frozenset({"lumina2"})

# Video GGUF archs the video backend CAN load (LTX-2.x ships as "ltxv", the Wan community GGUFs as "wan").
_VIDEO_GGUF_ARCHS = frozenset({"ltxv", "wan"})
_VIDEO_GEN_TASK = "text-to-video"

# Task tag for the archs above; mirrored by the frontend NON_CHAT_TASKS gate.
_UNSUPPORTED_DIFFUSION_TASK = "image-diffusion-unsupported"


def _gguf_architecture(path: str) -> Optional[str]:
    """The GGUF ``general.architecture``, or None. Delegates to the shared,
    bounds-checked header reader (cached by path/mtime/size)."""
    from utils.models.gguf_metadata import read_gguf_general_metadata

    arch = (read_gguf_general_metadata(path) or {}).get("general.architecture")
    return arch.strip() if isinstance(arch, str) and arch.strip() else None


def _gguf_family_buildable(name_hints: tuple[Optional[str], ...]) -> bool:
    """Whether an engine on THIS host can build the diffusion family a GGUF belongs to.

    The listing twin of the loader's gate, and the same predicate: ``validate_load_request`` refuses
    a family whose diffusers pipeline class this environment lacks (the newer families ship only in a
    newer diffusers, and packaging still allows an older one on Python 3.9) UNLESS the native sd.cpp
    engine would serve the GGUF, which needs no pipeline class at all. Advertising a row neither
    engine can build is a pick that can only fail; hiding one the native engine loads is the opposite
    mistake, on exactly the CPU/MPS hosts that engine exists for.

    Fails OPEN when no family resolves from the hints or the probe raises: the load path reports a
    real problem properly, and a listing must not hide a usable model over a detection miss."""
    try:
        from core.inference.diffusion_engine_router import family_buildable_here
        from core.inference.diffusion_families import detect_family_for_pick
        for hint in name_hints:
            if not hint:
                continue
            fam = detect_family_for_pick(hint)
            if fam is not None:
                return family_buildable_here(fam, model_kind = "gguf")
    except Exception:  # noqa: BLE001 -- never hide a model over a probe failure
        return True
    return True


def _video_family_buildable(fam) -> bool:
    """Whether the installed diffusers can build this video family's pipeline class.

    The video backend has no native engine, so it is the plain class check its own
    ``validate_load_request`` runs (``video.py`` -> ``assert_pipeline_class_available``): LTX-2 and
    the other newer pipelines exist only in a newer diffusers. Fails OPEN on any probe error."""
    try:
        from core.inference.diffusion_families import family_pipeline_available
        return family_pipeline_available(fam)
    except Exception:  # noqa: BLE001 -- never hide a model over a probe failure
        return True


def _arch_to_task(arch: Optional[str], name_hints: tuple[Optional[str], ...] = ()) -> Optional[str]:
    if arch is None:
        return None
    a = arch.lower()
    if a in _DIFFUSION_GGUF_ARCHS:
        # Third gate, mirroring the cached-repo picker: a family no engine here can build can only fail.
        if not _gguf_family_buildable(name_hints):
            return _UNSUPPORTED_DIFFUSION_TASK
        return "text-to-image"
    if a in _VIDEO_GGUF_ARCHS:
        # Advertise as loadable video only when a VideoFamily resolves. Some archs map straight
        # from the arch (ltxv); bare "wan" is ambiguous, so fall back to repo/file names.
        from core.inference.video_families import detect_video_family

        fam = detect_video_family("", override = a)
        if fam is None:
            for hint in name_hints:
                if hint:
                    fam = detect_video_family(hint)
                    if fam is not None:
                        break
        if fam is not None and not getattr(fam, "is_moe", False) and _video_family_buildable(fam):
            return _VIDEO_GEN_TASK
        return _UNSUPPORTED_DIFFUSION_TASK
    if a in _AMBIGUOUS_DIFFUSION_GGUF_ARCHS:
        # Same as the video branch: the arch is shared, so let the loader's family detection decide.
        from core.inference.diffusion_engine_router import family_buildable_here
        from core.inference.diffusion_families import detect_family_for_pick, family_gguf_loadable

        for hint in name_hints:
            if not hint:
                continue
            fam = detect_family_for_pick(hint)
            if fam is not None:
                # Both gates: a GGUF-assemblable family AND an engine here that can build it.
                loadable = family_gguf_loadable(fam) and family_buildable_here(
                    fam, model_kind = "gguf"
                )
                return "text-to-image" if loadable else _UNSUPPORTED_DIFFUSION_TASK
        return _UNSUPPORTED_DIFFUSION_TASK
    # A diffusion arch the backend cannot assemble: hide from chat and from Images (would 400).
    if a in _UNSUPPORTED_DIFFUSION_GGUF_ARCHS:
        return _UNSUPPORTED_DIFFUSION_TASK
    return "text-generation"


def _repo_gguf_task(repo_info) -> Optional[str]:
    """HF pipeline task of a cached GGUF repo, from its architecture:
    'text-to-image' for a loadable diffusion arch, the non-loadable diffusion tag
    for a recognized-but-unsupported image arch, else 'text-generation' (None if
    unreadable)."""
    repo_id = getattr(repo_info, "repo_id", None)
    try:
        for path in _iter_gguf_paths(Path(repo_info.repo_path)):
            if _is_mmproj_filename(path.name):
                continue
            task = _arch_to_task(_gguf_architecture(str(path)), name_hints = (repo_id, path.name))
            if task is not None:
                return task
    except Exception:
        pass
    return None


def _local_family_needles(model: "LocalModelInfo") -> tuple[str, ...]:
    """Family-detection hints for a local (non-GGUF) checkpoint: model id, display name, leaf dir
    name, and -- for a bare single-file dir -- the sole checkpoint's filename (a generic folder
    holding one ``qwen-image-*.safetensors`` identifies its family only there, and the load route
    resolves it via ``resolve_local_single_file``). Only basenames, so a parent-dir token can't
    match."""
    needles = [model.model_id, model.display_name, Path(model.id).name]
    try:
        from core.inference.diffusion import resolve_local_single_file
        single = resolve_local_single_file(model.path)
        if single:
            needles.append(single)
    except Exception:
        pass
    return tuple(n for n in needles if n)


def _local_model_task(model: "LocalModelInfo") -> Optional[str]:
    """Classify a local model into an HF pipeline task so the Images picker can filter.

    For a GGUF, read its architecture (the path may be the .gguf file itself or a folder
    containing one). For a local non-GGUF image checkpoint (a diffusers pipeline dir or a
    single-file safetensors), fall through to the diffusers detection so on-device image
    models get the 'text-to-image' tag instead of being dropped as task=null; the load
    path accepts these as a local pipeline."""
    path = model.path
    _id_hints = (model.model_id, model.display_name, model.id)
    if model.model_format == "gguf":
        try:
            p = Path(path)
            if p.suffix.lower() == ".gguf" and p.is_file():
                return _arch_to_task(_gguf_architecture(str(p)), name_hints = _id_hints + (p.name,))
            for f in _iter_gguf_paths(p):
                if _is_mmproj_filename(f.name):
                    continue
                task = _arch_to_task(_gguf_architecture(str(f)), name_hints = _id_hints + (f.name,))
                if task is not None:
                    return task
        except Exception:
            pass
        return None
    if _local_is_diffusers(model):
        # A local diffusers pipeline can be a VIDEO family, not just image; tag it text-to-video so it surfaces in the Video picker.
        try:
            from core.inference.video import _is_trusted_video_repo
            from core.inference.video_families import detect_video_family
            for needle in _local_family_needles(model):
                vfam = detect_video_family(needle)
                # Third gate: the video load asserts the family's pipeline class, and newer ones need newer diffusers.
                if vfam is not None and _is_trusted_video_repo(path):
                    return _VIDEO_GEN_TASK if _video_family_buildable(vfam) else None
        except Exception:
            pass
        # The Images load path 400s AFTER eviction when no image family is supported, so tag only when detection succeeds.
        try:
            from core.inference.diffusion_engine_router import family_buildable_here
            from core.inference.diffusion_families import detect_family

            for needle in _local_family_needles(model):
                fam = detect_family(needle)
                if fam is not None:
                    # A local non-GGUF checkpoint always loads through diffusers, so the pipeline class has to exist here.
                    return (
                        "text-to-image"
                        if family_buildable_here(fam, model_kind = "pipeline")
                        else None
                    )
            return None
        except Exception:
            # Detection unavailable: fall back to the prior permissive tag rather than hiding a possibly-loadable pipeline.
            return "text-to-image"
    return None


def _local_is_diffusers(model: "LocalModelInfo") -> bool:
    """True for a local diffusers image checkpoint, mirroring the cached-repo
    ``_repo_is_diffusers`` heuristics: a full pipeline carries a top-level
    ``model_index.json``, while single-file / safetensors image checkpoints ship none, so
    fall back to the model id resolving to a known diffusion family (the same resolver the
    Images backend loads from). Family detection uses _local_family_needles (id / name / sole
    checkpoint filename, not the on-disk path), so a parent-dir keyword can't spuriously match."""
    try:
        p = Path(model.path)
        if p.is_dir() and (p / "model_index.json").is_file():
            return True
    except Exception:
        pass
    try:
        from core.inference.diffusion_families import detect_family
        for needle in _local_family_needles(model):
            if detect_family(needle) is not None:
                return True
    except Exception:
        pass
    # A single-file VIDEO checkpoint (no model_index.json) is missed above but loaded as single_file by the video route, so surface it.
    try:
        from core.inference.video_families import detect_video_family
        for needle in _local_family_needles(model):
            if detect_video_family(needle) is not None:
                return True
    except Exception:
        pass
    return False


def snapshot_variants_all_complete(snapshot: str) -> bool:
    """Re-export; the predicate lives beside the completed-variant walk it uses."""
    from hub.utils import inventory_scan
    return inventory_scan.snapshot_variants_all_complete(snapshot)


def snapshot_has_complete_variants(snapshot: str) -> bool:
    """Re-export of the predicate every load-id pin shares; see above."""
    from hub.utils import inventory_scan
    return inventory_scan.snapshot_has_complete_variants(snapshot)


def _repo_gguf_load_id(repo_info, active_root: Optional[Path]) -> Optional[str]:
    """Snapshot dir holding the newest primary GGUF, for a repo outside the active
    hub cache that does not resolve by id. ``None`` when the id works or no
    snapshot is recorded, since the repo dir itself is not loadable.
    """
    from hub.utils.hf_cache_state import snapshot_selection_key

    repo_path = getattr(repo_info, "repo_path", None)
    if repo_path is None or active_root is None:
        return None
    try:
        # A recovered repo's refs/main names nothing, so its id resolves nowhere and needs a pin.
        if (
            repo_path.parent.resolve(strict = False) == active_root
            and not _repo_id_will_not_resolve(repo_path)
            and not _default_ref_offers_no_whole_quant(repo_path)
        ):
            return None
    except (OSError, RuntimeError, ValueError):
        pass
    # Shared selection key, so this route and the /gguf-variants lister name one snapshot.
    candidates = [
        Path(snapshot)
        for revision in repo_info.revisions
        if (snapshot := getattr(revision, "snapshot_path", None)) is not None
        and any(_is_main_gguf_filename(_cached_repo_file_name(f)) for f in revision.files)
    ]
    candidates.sort(key = snapshot_selection_key, reverse = True)
    # Newest first, skipping any holding no whole quant, else a torn download beats a loadable one.
    for snapshot in candidates:
        if snapshot_has_complete_variants(str(snapshot)):
            return str(snapshot)
    # Nothing complete anywhere: publishing a half-downloaded snapshot would put that path in
    # the copied command and fail on load. Drop the id so the repo id fetches the missing shards.
    return None


def _preferred_gguf_copy(
    rows: dict, ranks: dict, key: str, candidate: tuple[bool, bool], size: int
) -> bool:
    """Whether this copy should replace the one already kept for *key*.

    Same order the Hub inventory deduplicates by: a copy that cannot load loses to one that can,
    whichever cache holds it, then the active cache wins, then the larger download.
    """
    existing = rows.get(key)
    if existing is None:
        return True
    kept = ranks.get(key, (True, True))
    if candidate[0] != kept[0]:
        return candidate[0]
    if candidate[1] != kept[1]:
        return candidate[1]
    return size > int(existing.get("size_bytes") or 0)


@router.get("/cached-gguf")
async def list_cached_gguf(current_subject: str = Depends(get_current_subject)):
    """List GGUF repos downloaded to HF cache, legacy Unsloth cache, and HF default cache."""
    try:
        cache_scans = _all_hf_cache_scans()
        try:
            active_root = _resolve_hf_cache_dir().resolve(strict = False)
        except Exception:
            active_root = None

        seen_lower: dict[str, dict] = {}
        # How each kept row's copy ranks, since the compatibility schema carries neither field.
        seen_rank: dict[str, tuple[bool, bool]] = {}
        for hf_cache in cache_scans:
            for repo_info in hf_cache.repos:
                try:
                    if repo_info.repo_type != "model":
                        continue
                    repo_id = repo_info.repo_id
                    # Pass the snapshot path too so the config check also hides custom Whisper checkpoints.
                    if _is_hidden_model(repo_id, str(repo_info.repo_path)):
                        continue
                    total_size = _repo_gguf_size_bytes(repo_info)
                    if total_size == 0:
                        continue
                    key = repo_id.lower()
                    existing = seen_lower.get(key)
                    last_modified = _repo_gguf_last_modified(repo_info)
                    load_id = _repo_gguf_load_id(repo_info, active_root)
                    rank = (
                        _gguf_copy_is_usable(repo_info, load_id),
                        active_root is not None
                        and Path(repo_info.repo_path).parent.resolve(strict = False) == active_root,
                    )
                    if _preferred_gguf_copy(seen_lower, seen_rank, key, rank, total_size):
                        row = {
                            "repo_id": repo_id,
                            "size_bytes": total_size,
                            "cache_path": str(repo_info.repo_path),
                            "has_vision": _cached_gguf_row_has_vision(repo_info, load_id),
                            "task": _repo_gguf_task(repo_info),
                        }
                        if load_id:
                            row["load_id"] = load_id
                        # Keep the newest timestamp across duplicate caches; absent rows sort as oldest.
                        lm = max(last_modified, (existing or {}).get("last_modified", 0.0))
                        if lm > 0:
                            row["last_modified"] = lm
                        seen_lower[key] = row
                        seen_rank[key] = rank
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


def _repo_has_pipeline_index(repo_info) -> bool:
    """Root-model_index.json check. Shared with the hub inventory scan, which classifies the
    same repos for the same pickers; see :func:`hub.utils.inventory_scan.repo_has_pipeline_index`."""
    from hub.utils import inventory_scan as hf_cache_scan
    return hf_cache_scan.repo_has_pipeline_index(repo_info)


def _repo_is_diffusers(repo_info) -> bool:
    """True for an image-diffusion repo, so the chat picker hides it (it renders
    images, not chat) and the Images picker claims it — mirroring how cached
    diffusion GGUFs are classified by arch.

    Two signals: a full diffusers pipeline carries a top-level model_index.json,
    while single-file / ComfyUI / ControlNet image checkpoints (e.g. an FP8
    Qwen-Image or a z-image .safetensors) ship none. For those, fall back to the
    repo id resolving to a known diffusion family — the same resolver the Images
    backend loads from — so they don't surface as loadable chat models."""
    if _repo_has_pipeline_index(repo_info):
        return True
    try:
        from core.inference.diffusion_families import detect_family
        if detect_family(getattr(repo_info, "repo_id", "") or "") is not None:
            return True
    except Exception:
        pass
    return False


def _repo_pipeline_missing_denoiser(repo_info) -> bool:
    """Companion-only-prefetch check (pipeline manifest present, denoiser weights absent). Shared
    with the hub inventory scan so both listings agree on which rows are really on-device; see
    :func:`hub.utils.inventory_scan.repo_pipeline_missing_denoiser`."""
    from hub.utils import inventory_scan as hf_cache_scan
    return hf_cache_scan.repo_pipeline_missing_denoiser(repo_info)


def _cached_repo_partial(repo_id: str, repo_cache_dir: Optional[Path] = None) -> bool:
    """Whether the cached model snapshot is incomplete (cancelled/partial download).
    Reuses the hub inventory scan's snapshot-partial detector (cancel marker, legacy
    .incomplete blob, manifest walk -- cheapest first). ``repo_cache_dir`` scopes all three
    signals to the specific snapshot being listed: without it the scan spans every HF cache
    root, so a stale .incomplete copy in one root would flag a complete copy in another as
    partial and hide it from the picker (the sibling inventory paths all scope the same way).
    Best-effort: a detection error reports not-partial so a scan glitch never hides a
    genuinely usable repo."""
    try:
        from hub.utils.inventory_scan import is_snapshot_partial
        return bool(is_snapshot_partial("model", repo_id, repo_cache_dir))
    except Exception:  # noqa: BLE001 -- never fail the listing over a partial probe
        return False


def _is_sd_cpp_companion_repo(repo_id: str) -> bool:
    """True for a mirror that holds only sd.cpp companions (VAE / text encoders, no denoiser)."""
    try:
        from core.inference.diffusion_families import sd_cpp_companion_only_repo_ids
        return (repo_id or "").strip().lower() in sd_cpp_companion_only_repo_ids()
    except Exception:  # noqa: BLE001 -- an import failure must not hide a usable repo
        return False


def _cached_repo_task(repo_info) -> Optional[str]:
    """Pipeline task for a cached non-GGUF repo: 'text-to-video' for repos the
    video backend can load as full pipelines (its trust list / family detector),
    else 'text-to-image' for diffusers image repos, else None (chat). Without the
    video tag, cached Lightricks / Wan / Hunyuan pipelines never surfaced in the
    Video picker's On Device list -- everything diffusers was blanket-tagged
    text-to-image."""
    repo_id = getattr(repo_info, "repo_id", "") or ""
    try:
        from core.inference.video import _is_trusted_video_repo
        from core.inference.video_families import detect_video_family

        # Both gates: a detected video family (so image repos don't match) AND the load path's trust
        # rule. Third gate: the video load asserts the family's pipeline class (needs newer diffusers).
        video_fam = detect_video_family(repo_id)
        if video_fam is not None:
            if not _is_trusted_video_repo(repo_id) or not _video_family_buildable(video_fam):
                return None
            return _VIDEO_GEN_TASK
    except Exception:
        pass
    if not _repo_is_diffusers(repo_info):
        return None
    # BOTH gates, mirroring the video branch: trust rule AND a detected image family. A
    # model_index.json only proves it is a diffusers pipeline; newer families need diffusers 0.39.
    try:
        from core.inference.diffusion import _is_trusted_diffusion_repo
        from core.inference.diffusion_families import (
            detect_family,
            family_pipeline_available,
        )

        # An sd.cpp companion repo holds no denoiser, so it is never a pick even though its
        # unsloth/* mirror clears the trust gate below (the third-party ids never did). No task
        # keeps it out of the IMAGE picker; the row's companion flag is what keeps it out of the
        # chat one, since a task of None is what every unclassified chat repo carries.
        if _is_sd_cpp_companion_repo(repo_id):
            return None
        fam = detect_family(repo_id)
        if not _is_trusted_diffusion_repo(repo_id) or fam is None:
            return None
        if not family_pipeline_available(fam):
            return None
        return "text-to-image"
    except Exception:  # noqa: BLE001 -- an import failure must not hide a usable repo
        return "text-to-image"


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
        try:
            active_root = _resolve_hf_cache_dir().resolve(strict = False)
        except Exception:
            active_root = None

        seen_lower: dict[str, dict] = {}
        # Repos whose active-cache copy cannot be loaded by id; this schema carries no path.
        unusable_active: set[str] = set()
        for hf_cache in cache_scans:
            for repo_info in hf_cache.repos:
                try:
                    if repo_info.repo_type != "model":
                        continue
                    repo_id = repo_info.repo_id
                    # Pass the snapshot path too so the config check also hides custom Whisper checkpoints.
                    if _is_hidden_model(repo_id, str(repo_info.repo_path)):
                        continue
                    # No partial or load id here, so a snapshot-path-only repo would read as ready.
                    if _recovered_repo_is_unusable_by_repo_id(repo_info):
                        try:
                            if (
                                active_root is not None
                                and Path(repo_info.repo_path).parent.resolve(strict = False)
                                == active_root
                            ):
                                unusable_active.add(repo_id.lower())
                        except (OSError, RuntimeError, ValueError):
                            pass
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
                    # A companion-only prefetch (manifest + VAE/TE but no transformer shards) is not a loadable pipeline; treat it as partial.
                    is_partial = _cached_repo_partial(
                        repo_id, Path(repo_info.repo_path)
                    ) or _repo_pipeline_missing_denoiser(repo_info)
                    # Prefer the most COMPLETE snapshot, then largest: a partial copy in one cache root must not shadow a complete copy in another.
                    if existing is None or (not is_partial, total_size) > (
                        not bool(existing.get("partial")),
                        existing["size_bytes"],
                    ):
                        row = {
                            "repo_id": repo_id,
                            "size_bytes": total_size,
                            "task": _cached_repo_task(repo_info),
                        }
                        if is_partial:
                            row["partial"] = True
                        # Listed, so tens of GB of companion weights stay visible and deletable,
                        # but flagged, so no picker offers a denoiser-less repo as a load.
                        if _is_sd_cpp_companion_repo(repo_id):
                            row["companion"] = True
                        # Flag diffusion repos with no pipeline index: loadable only via from_single_file, so pickers must not offer a pipeline load.
                        if row["task"] is not None and not _repo_has_pipeline_index(repo_info):
                            row["single_file"] = True
                        # Keep the newest timestamp across duplicate caches; absent rows sort as oldest.
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

        rows = [row for key, row in seen_lower.items() if key not in unusable_active]
        # Local-only list path: update checks are GGUF-only and happen lazily when variants are viewed.
        cached = sorted(
            rows,
            key = lambda c: (-(c.get("last_modified") or 0.0), c["repo_id"].lower()),
        )
        return {"cached": cached}
    except Exception as e:
        logger.error(f"Error listing cached models: {e}", exc_info = True)
        return {"cached": []}


def _loaded_id_matches_repo(loaded_id: str, repo_id: str) -> bool:
    """True when *loaded_id* is *repo_id* or a file within it; ``/``-boundary aware so a
    loaded ``org/model-2512`` does not block deleting the sibling cached ``org/model``."""
    rid = repo_id.lower()
    lid = loaded_id.lower()
    return lid == rid or lid.startswith(f"{rid}/")


@router.delete("/delete-cached")
async def delete_cached_model(
    repo_id: str = Body(...),
    variant: Optional[str] = Body(None),
    cache_path: Optional[str] = Body(None),
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    """Compatibility route backed by the shared multi-cache deletion service."""
    from hub.services.models import deletion
    return await deletion.delete_cached_model_response(repo_id, variant, hf_token, cache_path)


def _resolve_cached_model_path(repo_id: str, variant: Optional[str]) -> Path:
    """Absolute path of a cached repo (newest snapshot dir) or, with *variant*,
    that quant's main GGUF file (first split of a sharded quant). Paths come
    from the HF cache scan only, so callers can't probe arbitrary paths."""
    cache_scans = _all_hf_cache_scans()

    matching_repos = []
    for hf_cache in cache_scans:
        for repo_info in hf_cache.repos:
            if repo_info.repo_type != "model":
                continue
            if repo_info.repo_id.lower() == repo_id.lower():
                matching_repos.append(repo_info)
    if not matching_repos:
        raise HTTPException(status_code = 404, detail = "Model not found in cache")

    if variant:
        want = _normalized_quant_label(variant)
        candidate_revisions = sorted(
            (rev for repo_info in matching_repos for rev in repo_info.revisions),
            key = lambda rev: getattr(rev, "last_modified", 0) or 0,
            reverse = True,
        )
        for rev in candidate_revisions:
            snapshot = getattr(rev, "snapshot_path", None)
            matches = []
            for f in rev.files:
                p = Path(f.file_path)
                rel = f.file_name
                if snapshot:
                    try:
                        rel = p.relative_to(snapshot).as_posix()
                    except ValueError:
                        pass
                label = _main_variant_gguf_label(rel)
                if label is None or _normalized_quant_label(label) != want:
                    continue
                if p.exists() or p.is_symlink():
                    matches.append((rel, p))
            if matches:
                # Path-sorted so a sharded quant deterministically yields its first split.
                return sorted(matches, key = lambda m: m[0].lower())[0][1]
        raise HTTPException(
            status_code = 404,
            detail = f"Variant {variant} not found in cache for {repo_id}",
        )

    def repo_size(repo_info) -> int:
        gguf_size = _repo_gguf_size_bytes(repo_info)
        if gguf_size > 0:
            return gguf_size
        return sum(
            (getattr(f, "size_on_disk", None) or 0)
            for rev in repo_info.revisions
            for f in rev.files
        )

    def repo_last_modified(repo_info) -> float:
        return max(
            (getattr(rev, "last_modified", 0) or 0 for rev in repo_info.revisions),
            default = 0,
        )

    target_repo = max(
        matching_repos,
        key = lambda repo_info: (repo_size(repo_info), repo_last_modified(repo_info)),
    )

    # Whole repo: the newest revision's snapshot dir holds the visible files.
    revisions = sorted(
        (rev for rev in target_repo.revisions if getattr(rev, "snapshot_path", None)),
        key = lambda rev: getattr(rev, "last_modified", 0) or 0,
        reverse = True,
    )
    for rev in revisions:
        p = Path(rev.snapshot_path)
        if p.exists():
            return p
    p = Path(target_repo.repo_path)
    if p.exists():
        return p
    raise HTTPException(status_code = 404, detail = "Cached model path not found")


def _wsl_reveal_in_explorer(path: Path) -> bool:
    import subprocess

    from utils.paths.path_utils import _IS_WSL

    if not _IS_WSL:
        return False
    try:
        windows_path = subprocess.run(
            ["wslpath", "-w", str(path)],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            check = True,
            timeout = 10,
        ).stdout.strip()
        if not windows_path:
            return False
        argument = f"/select,{windows_path}" if path.is_file() else windows_path
        subprocess.Popen(["explorer.exe", argument])
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _reveal_in_file_manager(path: Path) -> None:
    """Open the OS file manager with *path* selected (best effort per platform)."""
    import subprocess

    target = str(path)
    if sys.platform == "darwin":
        cmd = ["open", "-R", target] if path.is_file() else ["open", target]
        subprocess.Popen(cmd)
    elif os.name == "nt":
        if path.is_file():
            subprocess.Popen(["explorer", f"/select,{target}"])
        else:
            os.startfile(target)  # noqa: S606 - local user's own file manager
    elif not _wsl_reveal_in_explorer(path):
        # No cross-desktop "select file" standard on Linux; open the directory.
        directory = target if path.is_dir() else str(path.parent)
        subprocess.Popen(["xdg-open", directory])


class CachedModelPathResponse(BaseModel):
    path: str
    is_dir: bool


@router.get("/cached-model-path", response_model = CachedModelPathResponse)
async def get_cached_model_path(
    repo_id: str = Query(..., description = "HuggingFace repo ID"),
    variant: str = Query("", description = "Quantization variant (empty for whole repo)"),
    current_subject: str = Depends(get_current_subject),
):
    """Absolute on-disk path of a cached repo or one of its GGUF variants."""
    if not _is_valid_repo_id(repo_id):
        raise HTTPException(status_code = 400, detail = "Invalid repo_id format")
    path = await asyncio.to_thread(_resolve_cached_model_path, repo_id, variant.strip() or None)
    return {"path": str(path), "is_dir": path.is_dir()}


@router.post("/reveal-cached-model")
async def reveal_cached_model(
    repo_id: str = Body(...),
    variant: Optional[str] = Body(None),
    current_subject: str = Depends(get_current_subject),
):
    """Reveal a cached repo (or one GGUF variant's file) in the OS file manager."""
    if not _is_valid_repo_id(repo_id):
        raise HTTPException(status_code = 400, detail = "Invalid repo_id format")
    variant = (variant or "").strip() or None
    path = await asyncio.to_thread(_resolve_cached_model_path, repo_id, variant)
    try:
        await asyncio.to_thread(_reveal_in_file_manager, path)
    except Exception as e:
        logger.error(f"Failed to reveal {path}: {e}")
        raise HTTPException(status_code = 500, detail = "Failed to open file manager")
    return {"status": "ok", "path": str(path)}


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


# Successful estimates only, keyed by model id. Failures aren't cached so they can recover.
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
            # Contained lexically; resolve symlinks and re-verify before touching the filesystem.
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

        # A local LoRA adapter is sized via its base model from the adapter config; re-validate that
        # resolved base so a crafted adapter can't redirect the local scan outside the roots.
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
