# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model management API routes."""

import asyncio
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import uuid
import weakref
from pathlib import Path
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel
from typing import List, NamedTuple, Optional
from loggers import get_logger

# Dependency-light leaf (PEP 562 package init): no llama.cpp / torch import chain.
from core.inference.memory_contract import (
    EMPTY_BREAKDOWN,
    build_memory_estimate,
    project_kv_cache_estimate,
)
from core.inference.model_ids import display_model_name
from hub.services.models import catalog_classification as _catalog_classification
from utils import gguf_archs as _gguf_archs
from hub.services.models.catalog_classification import (
    _cached_repo_task,
    _is_sd_cpp_companion_repo,
    _local_model_task,
    _repo_gguf_task,
    _repo_has_pipeline_index,
    _repo_is_diffusers,
)

# Compatibility aliases: these moved to catalog_classification, but callers and tests still
# resolve them from routes.models. Assigned through the module rather than re-imported, since
# an import this module never loads reads as a botched hoist to verify_import_hoist.py.
_AMBIGUOUS_DIFFUSION_GGUF_ARCHS = _catalog_classification._AMBIGUOUS_DIFFUSION_GGUF_ARCHS
_DIFFUSION_GGUF_ARCHS = _catalog_classification._DIFFUSION_GGUF_ARCHS
_TASK_CLASSIFY_WALK_SECONDS = _catalog_classification._TASK_CLASSIFY_WALK_SECONDS
_UNSUPPORTED_DIFFUSION_GGUF_ARCHS = _catalog_classification._UNSUPPORTED_DIFFUSION_GGUF_ARCHS
_UNSUPPORTED_DIFFUSION_TASK = _catalog_classification._UNSUPPORTED_DIFFUSION_TASK
_SPEECH_GGUF_ARCHS = _catalog_classification._SPEECH_GGUF_ARCHS
_SPEECH_TASK = _catalog_classification._SPEECH_TASK
_VIDEO_GEN_TASK = _catalog_classification._VIDEO_GEN_TASK
_VIDEO_GGUF_ARCHS = _catalog_classification._VIDEO_GGUF_ARCHS
_arch_to_task = _catalog_classification._arch_to_task
_gguf_architecture = _catalog_classification._gguf_architecture
_gguf_folder_task = _catalog_classification._gguf_folder_task
_hf_cache_snapshot_repo_id = _catalog_classification._hf_cache_snapshot_repo_id
_is_trailing_split_shard = _catalog_classification._is_trailing_split_shard
_local_family_needles = _catalog_classification._local_family_needles
_local_is_diffusers = _catalog_classification._local_is_diffusers
_local_model_can_chat = _catalog_classification._local_model_can_chat
_task_classify_sort_key = _catalog_classification._task_classify_sort_key
# core.inference.llama_cpp imports this one. Without the alias that import raises, and the
# probe's `except Exception: return True` reads the raise as "the page can build it",
# promising an MoE or familyless GGUF a load that dies in llama-server.
_video_family_buildable = _catalog_classification._video_family_buildable
# The rest of what moved. Nothing in-repo reads these, which is why they went unnoticed --
# the one name that did have a caller surfaced as a wrong message, never an ImportError.
# All were importable from routes.models before and two are public, so a downstream fork or
# an older script may hold one, which this repo cannot see.
_H3_DENOISER_GGUF_PREFIXES = _catalog_classification._H3_DENOISER_GGUF_PREFIXES
_LOADABLE_MEDIA_GGUF_TASKS = _catalog_classification._LOADABLE_MEDIA_GGUF_TASKS
_MAX_TASK_CLASSIFY_GGUFS = _catalog_classification._MAX_TASK_CLASSIFY_GGUFS
_PLACEHOLDER_DIFFUSION_GGUF_ARCHS = _catalog_classification._PLACEHOLDER_DIFFUSION_GGUF_ARCHS
_TASK_CLASSIFY_READ_SECONDS = _catalog_classification._TASK_CLASSIFY_READ_SECONDS
_gguf_family_buildable = _catalog_classification._gguf_family_buildable
_is_h3_bundle_gguf_hint = _catalog_classification._is_h3_bundle_gguf_hint
SPEECH_GGUF_ARCHS = _gguf_archs.SPEECH_GGUF_ARCHS
is_speech_gguf_architecture = _gguf_archs.is_speech_gguf_architecture
from utils.utils import canonical_model_repo_id, log_and_http_error

import re as _re

_VALID_REPO_ID = _re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


class CachedModelRepo(BaseModel):
    repo_id: str
    size_bytes: int
    last_modified: Optional[float] = None
    # "text-to-image" for cached diffusers image repos; declared here or response_model drops it.
    task: Optional[str] = None
    audio_type: Optional[str] = None
    # Snapshot incomplete (cancelled/partial download): the picker must not treat it as usable.
    partial: Optional[bool] = None
    # Diffusion-tagged repo with NO top-level model_index.json: needs from_single_file + a filename.
    single_file: Optional[bool] = None
    # True for an sd.cpp companion mirror (VAE / text encoders, no denoiser). Declared here or
    # response_model drops it and the flag never reaches the picker that has to filter on it.
    companion: Optional[bool] = None
    # Snapshot path for a copy its bare repo id cannot reach (legacy/default cache while
    # another is active); undeclared, response_model drops it and the picker uses the active cache.
    load_id: Optional[str] = None
    # "adapter" for a cached LoRA/PEFT repo; pickers that offer whole models filter on it.
    model_format: Optional[str] = None
    # False for an encoder-only repo (embedding/CLIP/ViT); undeclared, response_model drops it.
    can_chat: Optional[bool] = None
    # True for an image/video diffusion repo. Not the same question as task, which says only
    # whether this backend can load it as a pipeline; an untrusted or unrecognised one has none.
    diffusers: Optional[bool] = None


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


from utils.paths.scan_folder_health import (
    annotate_scan_folders,
    note_scan_folder_scanned,
    record_scan_failure,
    refresh_failed_scan_folders,
)

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
        ModelConfig,
    )
    from utils.models.model_config import (
        _extract_quant_label,
        _is_big_endian_gguf_path,
        _is_imatrix_path,
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
        ModelConfig,
    )
    from utils.models.model_config import (
        _extract_quant_label,
        _is_big_endian_gguf_path,
        _is_imatrix_path,
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
from utils.paths.path_utils import is_appledouble_metadata

router = APIRouter()
logger = get_logger(__name__)

# The shortest context worth pricing, used to separate the part of a footprint
# that shrinks with context from the part that does not. Not zero: zero means
# "the model's native length" to the planner, which is the opposite of what this
# asks. One llama.cpp KV stream pads to 256, so a smaller number would not make
# the cache any smaller and only invites a divide-by-zero somewhere downstream.
_MIN_PRICED_CONTEXT = 256


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
    weight files. Excludes ``mmproj`` GGUFs (vision projectors), calibration
    imatrices and non-weight ``.bin`` files (``tokenizer.bin`` etc.) to avoid
    false positives.
    """

    def _is_weight_file(f: Path) -> bool:
        if is_appledouble_metadata(f):
            return False
        suffix = f.suffix.lower()
        if suffix == ".safetensors":
            return True
        if suffix == ".gguf":
            return "mmproj" not in f.name.lower() and not _is_imatrix_path(f.name)
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
        # Only the safetensors arm needs the check: a weight ".bin" is recognised by its name
        # prefix, which a "._" already fails.
        if any(not is_appledouble_metadata(f) for f in path.glob("*.safetensors")):
            return True
        return any(_is_weight_bin(f.name) for f in path.glob("*.bin"))
    except OSError:
        return False


def _local_pipeline_index(d: Path) -> bool:
    from hub.services.models.common import _is_diffusers_pipeline_dir
    return _is_diffusers_pipeline_dir(d)


def _servable_gguf_names(directory: Path) -> list[str]:
    """The ``.gguf`` names in *directory* that count as a model being present there.

    An imatrix is calibration data, not a model artifact, so it must not make an empty
    folder look like one. mmproj and MTP drafters DO count: they are companions of a real
    model, and presence is all they decide (format still asks _is_main_gguf_filename).
    """
    return [
        p.name
        for p in directory.glob("*.gguf")
        if not is_appledouble_metadata(p) and not _is_imatrix_path(p.name)
    ]


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

            gguf_names = _servable_gguf_names(child)
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
            if (
                gguf_file.is_file()
                and _is_main_gguf_filename(gguf_file.name)
                and not is_appledouble_metadata(gguf_file)
            ):
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

        def _servable(p: Path) -> bool:
            return _is_main_gguf_filename(p.name) and not is_appledouble_metadata(p)

        if not any(_servable(p) for p in path.glob("*.gguf")):
            if not recursive:
                return None
            if not any(_servable(p) for p in path.glob("*/*.gguf")):
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
                if (
                    _is_main_gguf_filename(child.name)
                    and child.is_file()
                    and not is_appledouble_metadata(child)
                ):
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
                            bool(_servable_gguf_names(model_dir))
                            or (model_dir / "config.json").exists()
                            or any(
                                not is_appledouble_metadata(p)
                                for p in model_dir.glob("*.safetensors")
                            )
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
                    elif (
                        _is_main_gguf_filename(model_dir.name)
                        and model_dir.is_file()
                        and not is_appledouble_metadata(model_dir)
                    ):
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
            # Parsed, but the shape is still whatever was on disk. rglob("*") hands us every
            # file under manifests/, so a pruned pull, an editor backup, or any stray JSON can
            # be a list or a string; .get() on one raises AttributeError, which neither this
            # loop's `except OSError` nor the caller's catches, and one such file would 500
            # the whole picker. Mirrors hub/services/models/ollama.py, which already validates
            # each level of the same document.
            if not isinstance(manifest, dict):
                logger.debug("Skipping Ollama manifest %s: top level is not an object", tag_file)
                continue

            config = manifest.get("config")
            config_digest = config.get("digest", "") if isinstance(config, dict) else ""
            if not isinstance(config_digest, str):
                config_digest = ""
            model_type = ""
            file_type = ""
            if config_digest and blobs_dir.is_dir():
                config_blob = blobs_dir / config_digest.replace(":", "-")
                if config_blob.is_file():
                    try:
                        cfg = json.loads(config_blob.read_text(encoding = "utf-8-sig"))
                    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
                        logger.debug(
                            "Could not parse Ollama config blob %s: %s",
                            config_blob,
                            e,
                        )
                        cfg = None
                    if isinstance(cfg, dict):
                        model_type = cfg.get("model_type", "")
                        file_type = cfg.get("file_type", "")

            model_link_dir = links_root / stem_hash

            gguf_link_path: Optional[str] = None
            quant = f"-{file_type}" if file_type else ""
            safe_name = repo_name.replace("/", "-")
            layers = manifest.get("layers") or []
            if not isinstance(layers, list):
                logger.debug("Skipping Ollama manifest %s: layers is not a list", tag_file)
                continue
            for layer in layers:
                if not isinstance(layer, dict):
                    continue
                media = layer.get("mediaType", "")
                digest = layer.get("digest", "")
                if not isinstance(digest, str) or not digest:
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
                    # The frontend groups and labels these rows by this value
                    # (local-model-options.ts, pickers.tsx); "custom" hid them
                    # in the generic folder section (#9986).
                    source = "ollama",
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
    from hub.utils import gguf as gguf_utils
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
            custom_models = gguf_utils.dedupe_custom_gguf_rows(custom_models)
            if len(custom_models) < _MAX_MODELS_PER_FOLDER:
                custom_models += _scan_ollama_dir(
                    folder_path,
                    limit = _MAX_MODELS_PER_FOLDER - len(custom_models),
                )
        except OSError as e:
            logger.warning("Skipping unreadable scan folder %s: %s", folder_path, e)
            # Keep the reason so the folder list can show it instead of nothing.
            record_scan_failure(str(folder.get("path", folder_path)), e)
            continue
        note_scan_folder_scanned(str(folder.get("path", folder_path)), found = bool(custom_models))
        # Keep an already-attributed source: a registered ~/.ollama/models (or a
        # folder shadowing the HF cache) must not re-stamp its rows as generic
        # custom entries. Mirrors _promote_to_custom_source() in
        # hub/services/models/local_inventory.py.
        local_models += [
            m if m.source in ("hf_cache", "ollama") else m.model_copy(update = {"source": "custom"})
            for m in custom_models
        ]

    # Deduplicate, but always keep custom folder entries (keyed by (id, source)) so they show
    # in the "Custom Folders" UI section even when the model is also in the HF cache.
    deduped: dict[str, LocalModelInfo] = {}
    for model in local_models:
        semantic_id = model.model_id if model.source == "hf_cache" and model.model_id else model.id
        if model.source == "custom":
            physical_identity = gguf_utils.local_path_physical_identity(model.path)
            if (
                model.model_id
                and model.model_id.startswith("ollama/")
                and any(
                    part in (".studio_links", "ollama_links") for part in Path(model.path).parts
                )
            ):
                physical_identity = "\x00".join((model.model_id, physical_identity))
            key = "\x00".join((physical_identity, model.model_format or "", "custom"))
        else:
            key = semantic_id
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

    deduped_values = list(deduped.values())
    custom_values = [model for model in deduped_values if model.source == "custom"]
    models = sorted(
        [model for model in deduped_values if model.source != "custom"]
        + gguf_utils.suppress_grouped_gguf_file_rows(custom_values),
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

    def classify(models: List[LocalModelInfo]) -> List[LocalModelInfo]:
        # Tag each model with its task and native-audio type for the model pickers.
        # Inside the shared flight so overlapping callers reuse one classified result
        # instead of each repeating the GGUF header reads.
        classified = []
        for model in models:
            task, audio_type = _catalog_classification._local_model_classification_for_task(
                model, _local_model_task(model)
            )
            classified.append(
                model.model_copy(
                    update = {
                        "task": task,
                        "audio_type": audio_type,
                    }
                )
            )
        return classified

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
        classified = await asyncio.to_thread(classify, models)
        # That hop is an await point of its own, so a mutation can land after the check above.
        if hf_cache_scan.hf_cache_scans_epoch() != expected_epoch:
            raise _CompatLocalCacheChanged(models)
        return classified

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
                lambda expected_epoch = epoch, folders = custom_folders, roots = scan_sources: collect(
                    expected_epoch, folders, roots
                ),
            )
        except _CompatLocalCacheChanged as changed:
            superseded = changed.models
            continue
    # Invalidations are outpacing the walk, so no scan will ever confirm as
    # current. Answer with the freshest one (the loop only reaches here through
    # the retry path, so there is always one) instead of rescanning forever.
    logger.warning("Compat local inventory kept racing cache invalidations; serving the last scan")
    return await asyncio.to_thread(classify, superseded)


async def _invalidate_local_scans() -> None:
    """Retire the cached local scans after something was deleted from disk.

    Every successful deletion branch has to call this. The /v1/models servability scan is
    cached against the resolver generation, so a branch that returns without bumping it
    keeps advertising what was just removed until the catalog TTL expires.

    Off the loop, like the other async invalidation sites in this file: invalidate_index
    takes the resolver lock, and _index() holds that across a full multi-root filesystem
    scan, so calling it inline from an async route would stall unrelated requests and
    in-flight inference streams behind a rebuild.
    """
    from core.inference.local_model_resolver import invalidate_index
    await asyncio.to_thread(invalidate_index)


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

    folders = list_scan_folders()
    # Opening the dialog is how a fixed folder clears, so recheck the bad ones.
    await asyncio.to_thread(refresh_failed_scan_folders, folders)
    return {"folders": annotate_scan_folders(folders)}


@router.post("/scan-folders", response_model = ScanFolderInfo, status_code = 201)
async def add_scan_folder_endpoint(
    body: AddScanFolderRequest, current_subject: str = Depends(get_current_subject)
):
    """Register a new directory to scan for local models."""
    from storage.studio_db import add_scan_folder_with_status

    try:
        folder, inserted = await asyncio.to_thread(add_scan_folder_with_status, body.path)
    except ValueError as e:
        logger.warning("Scan folder rejected: %s (path=%s)", e, body.path)
        # Forward the curated, path-free validation message.
        rejection_message = str(e)
        raise HTTPException(status_code = 400, detail = rejection_message)
    logger.info("Scan folder added: %s", folder.get("path"))
    if inserted:
        from core.inference.local_model_resolver import invalidate_index, warm_index_soon
        await asyncio.to_thread(invalidate_index)
        warm_index_soon()
    return folder


@router.delete("/scan-folders/{folder_id}")
async def remove_scan_folder_endpoint(
    folder_id: int, current_subject: str = Depends(get_current_subject)
):
    """Remove a registered custom scan folder."""
    from storage.studio_db import remove_scan_folder

    removed = await asyncio.to_thread(remove_scan_folder, folder_id)
    if removed:
        logger.info("Scan folder removed: id=%s", folder_id)
        from core.inference.local_model_resolver import invalidate_index, warm_index_soon

        await asyncio.to_thread(invalidate_index)
        warm_index_soon()
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
                # Same shape check as _scan_ollama_dir: a valid-JSON non-object under
                # manifests/ must be skipped, not walked, or the chip probe raises
                # AttributeError past the `except OSError` below.
                if not isinstance(manifest, dict):
                    continue
                layers = manifest.get("layers") or []
                if not isinstance(layers, list):
                    continue
                for layer in layers:
                    if not isinstance(layer, dict):
                        continue
                    if layer.get("mediaType") != "application/vnd.ollama.image.model":
                        continue
                    digest = layer.get("digest", "")
                    if (
                        isinstance(digest, str)
                        and digest
                        and (blobs / digest.replace(":", "-")).is_file()
                    ):
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
                    if is_appledouble_metadata(entry):
                        continue
                    # The scanners no longer surface an imatrix-only folder, so counting
                    # one here would advertise a chip that opens an empty picker.
                    if low.endswith(".gguf") and not _is_imatrix_path(entry.name):
                        return True
                    if low.endswith(".safetensors"):
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
                    if low.endswith((".gguf", ".safetensors")) and not is_appledouble_metadata(
                        child
                    ):
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
    """The window this model was trained for, by the rule a load resolves it with.

    Reading one field name showed a dash for a model spelling it another way -- Kimi
    Linear carries model_max_length alone -- and a number as soon as it loaded.
    """
    from types import SimpleNamespace

    from core.inference.mlx_inference import mlx_native_context_length

    return mlx_native_context_length(SimpleNamespace(config = config))


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
            from utils.models.model_config import detect_audio_type_checked

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
            audio_type, audio_type_definitive = detect_audio_type_checked(
                _audio_probe_target(inspection_target),
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
                f"Model config result for {model_name}: is_vision={is_vision}, is_embedding={is_embedding}, audio_type={audio_type}, audio_type_known={audio_type_definitive}, is_lora={is_lora}, max_position_embeddings={max_position_embeddings}"
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
                audio_type_known = audio_type_definitive,
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
        from core.inference.native_audio import native_audio_security_targets

        try:
            requested_security_targets = native_audio_security_targets(
                requested_scan_target, hf_token = hf_token
            )
        except ValueError as exc:
            raise HTTPException(status_code = 400, detail = str(exc)) from exc
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
        if any(f.file_name.lower().endswith(_WEIGHTS) for f in _cached_files(target_repo)):
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


def _audio_probe_target(inspection_target: str) -> str:
    """Repo to ask about audio capability, resolving a registry alias first.

    A curated entry like "Spark-TTS-0.5B/LLM" names a load subdirectory, not a repo, so
    the probe fetched a repo that does not exist, got a 404 on every path, and read that
    as "definitely not an audio model" rather than "not a repo id". Spark-TTS then looked
    like a text model, and picking it with an audio dataset hit the modality gate. Same
    resolution routes/training.py already uses for the trainer's own preflight.
    """
    if is_local_path(inspection_target):
        return inspection_target
    try:
        from utils.security import load_scan_target
        repo_id, _load_subdirs = load_scan_target(canonical_model_repo_id(inspection_target), ())
        return repo_id or inspection_target
    except Exception:  # noqa: BLE001 - a probe target must never fail the handler
        return inspection_target


def _audio_type_of_checkpoint(
    model_path: str,
    base_model: Optional[str],
    hf_token: Optional[str] = None,
) -> Optional[str]:
    """Codec a trained checkpoint speaks, or None for a text one.

    A scan row carries no modality, so without this every trained audio model reads
    as text: the Audio page filters it out and chat routes it to the GGUF auto-switch,
    which cannot resolve a local adapter directory. Detection reads the checkpoint
    itself first (a merged export has its own tokenizer) and falls back to the base
    repo an adapter names. Cached per model, so the scan stays one pass.
    """
    from utils.models.model_config import detect_audio_type

    for candidate in (model_path, base_model):
        if not candidate:
            continue
        try:
            # local_files_only: this route was a filesystem scan. A trained checkpoint's
            # base is already cached, and a non-definitive miss is deliberately not cached,
            # so a gated or offline base would re-fetch on every poll.
            # hf_token even under local_files_only: a gated base resolves through the same
            # hub helpers, and the capability caches are keyed by token fingerprint, so a
            # token-less probe would both misclassify and poison the cache for the rest.
            audio_type = detect_audio_type(candidate, hf_token = hf_token, local_files_only = True)
        except Exception as exc:  # never let a scan row fail the whole listing
            logger.debug("audio detection failed for %r: %s", candidate, exc)
            continue
        if audio_type:
            return audio_type
    return None


@router.get("/loras")
async def scan_loras(
    outputs_dir: str = Query(
        default = str(outputs_root()), description = "Directory to scan for LoRA adapters"
    ),
    exports_dir: str = Query(
        default = str(exports_root()), description = "Directory to scan for exported models"
    ),
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    """Scan for trained LoRA adapters and exported models.

    Returns training outputs (outputs_dir) and exported models
    (exports_dir) in one list, distinguished by the source field.
    """
    try:
        resolved_outputs_dir = str(resolve_output_dir(outputs_dir))
        resolved_exports_dir = str(resolve_export_dir(exports_dir))
        # Off the event loop: this is a directory walk plus, per checkpoint, a tokenizer
        # read. It was already blocking before the audio probe was added; the probe made
        # the block long enough to delay unrelated requests, streamed tokens included.
        lora_list = await asyncio.to_thread(
            _scan_loras_sync, resolved_outputs_dir, resolved_exports_dir, hf_token
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


def _scan_loras_sync(
    resolved_outputs_dir: str, resolved_exports_dir: str, hf_token: Optional[str]
) -> List[LoRAInfo]:
    """The filesystem half of scan_loras, so it can run in a worker thread."""
    lora_list: List[LoRAInfo] = []

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
                audio_type = _audio_type_of_checkpoint(model_path, base_model, hf_token),
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
                audio_type = _audio_type_of_checkpoint(model_path, base_model, hf_token),
            )
        )

    return lora_list


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


def _variant_names_same_checkpoint(a: Optional[str], b: Optional[str]) -> bool:
    """Whether two variant spellings can name the SAME checkpoint, for the load-state guard.

    Deletion accepts an unambiguous bare quant for a path-qualified key (the shared-container
    layout, ``weights/model-Q4_K_M.gguf``), so a guard comparing the two spellings literally lets
    a model loaded through a legacy bare pin be deleted through its advertised qualified row --
    unlinking the resident model's snapshot and blob. Deliberately loose: a false match only
    refuses a delete, a false miss loses weights.
    """
    from hub.utils.gguf import bare_quant_alias, is_qualified_gguf_variant_key

    left = (a or "").strip().lower()
    right = (b or "").strip().lower()
    if not left or not right:
        return False
    if left == right:
        return True
    for key, bare in ((left, right), (right, left)):
        if (
            is_qualified_gguf_variant_key(key)
            and not is_qualified_gguf_variant_key(bare)
            and bare_quant_alias(key).lower() == bare
        ):
            return True
    return False


def _delete_gguf_variant_files(root: Path, variant: str) -> tuple[int, int]:
    deleted_count = 0
    deleted_bytes = 0
    from hub.utils.gguf import remove_appledouble_sidecar

    for path in root.rglob("*"):
        if not path.is_file() or not _is_main_gguf_filename(path.name):
            continue
        # Counted as a model if left in, so the reported count would follow the walk order; it
        # still goes below, as metadata of the file it belongs to. Proven metadata only: anything
        # else carrying this variant's key is a file the user asked to delete.
        if is_appledouble_metadata(path):
            continue
        # Keyed on the path, not the basename: a repo holding several checkpoints at
        # one quant would otherwise delete every one of them for a single row.
        from utils.models.model_config import _gguf_variant_key

        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.name
        if _gguf_variant_key(relative).lower() != variant.lower():
            continue
        try:
            deleted_bytes += path.stat().st_size
        except OSError:
            pass
        path.unlink()
        # Skipped by the walk above, so this is its only chance to be reclaimed. Its bytes count
        # toward what was freed even though it is not counted as a model.
        deleted_bytes += remove_appledouble_sidecar(path)
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
                # Alias-aware: the delete below accepts a bare quant for a qualified key, so a
                # literal comparison here would wave through the very spelling it then deletes.
                or _variant_names_same_checkpoint(llama_backend.hf_variant, gguf_variant)
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
                or _variant_names_same_checkpoint(llama_backend.hf_variant, gguf_variant)
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
            await _invalidate_local_scans()
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

        await _invalidate_local_scans()
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
        from utils.paths.path_utils import file_contents_available_locally

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
                if _is_mmproj_filename(f.name) or not file_contents_available_locally(f):
                    # Opening a cloud placeholder recalls its data. It keeps its variant row,
                    # but has no context metadata until the file is hydrated.
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
            # A direct file selection names the weights outright: custom, LM
            # Studio and other local inventory entries whose path ends in .gguf
            # never go through variant selection, so there is no quant label to
            # match and nothing to scan for. Answer with the file itself, and
            # with the whole split family's size rather than this shard's, the
            # same way the quant scan below totals its shards.
            direct = Path(repo_id)
            if direct.is_file() and direct.suffix.lower() == ".gguf":
                from core.inference.llama_cpp import LlamaCppBackend
                return str(direct), LlamaCppBackend._get_gguf_size_bytes(str(direct))
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

        want = (quant or "").strip()
        best_total = 0
        best_first: Optional[str] = None
        for root in roots:
            ranked: dict[int, list[tuple[str, Path, int]]] = {0: [], 1: []}
            for f in _iter_gguf_paths(root):
                try:
                    rel = f.relative_to(root).as_posix()
                except ValueError:
                    rel = f.name
                rank = _main_variant_rank(rel, want)
                if rank is None:
                    continue
                try:
                    size = f.stat().st_size
                except OSError:
                    continue
                ranked[rank].append((rel, f, size))
            # Exact keys alone when any exist: summing them with the label matches counts other
            # checkpoints' bytes into this row's estimate and can reveal one of their files.
            # ... and within those, ONE shard family, the same rule group_gguf_variant_files
            # applies: a snapshot holding the same quant twice (QwQ-32B's two BF16 shard sets)
            # would otherwise report double the weights the loader opens, which /kv-cache-estimate
            # turns into a false exceeds-memory warning and which can make a snapshot look
            # "more complete" purely for holding a redundant copy.
            chosen = _one_shard_family_of(ranked[0] or ranked[1])
            matches = [(rel, f) for rel, f, _size in chosen]
            total = sum(size for _rel, _f, size in chosen)
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


def _resolve_mtp_drafter(
    main_gguf_path: str, search_root: Optional[str] = None
) -> tuple[Optional[str], int]:
    """Separate MTP drafter GGUF for a resolved main quant, or (None, 0).

    Some repos ship the drafter as its own file beside the weights (Gemma 4's
    ``mtp-*.gguf``). The main GGUF has no ``nextn_predict_layers`` in that case,
    so the estimator's embedded-head path returns None and the reserve reads as
    zero unless we hand it the drafter.

    Delegates to the two resolvers the LOAD path already uses, rather than
    scanning for a drafter itself: ``_companion_snapshot_sibling`` with the
    loader's own ``_pick_mtp`` for an HF snapshot, and ``detect_mtp_file`` for a
    local folder, which is what ``model_config`` calls when it builds the launch.
    A bespoke scan here is how the estimate ends up pricing a different file from
    the one llama-server opens: ``_pick_mtp`` is root-level and prefix-matched, so
    it cannot be fooled by a directory that happens to be named ``mtp``, it finds
    the snapshot-root companion when the weights sit in a quant subdirectory, it
    sorts on relative strings rather than ``Path`` objects (whose ordering is
    case-folded on Windows and not on POSIX, so two hosts really can disagree),
    and it rejects an incomplete split set. Never raises: a drafter we cannot
    find just costs a segment.
    """
    try:
        from core.inference.llama_cpp import (
            _companion_snapshot_sibling,
            _pick_mtp,
            _snapshot_dir_of,
        )

        if _snapshot_dir_of(main_gguf_path) is not None:
            # An HF snapshot. ``_download_mtp`` resolves through ``_pick_mtp``,
            # which is root-level only, so the ``MTP/`` precision copies are not
            # auto-fetched and must not be priced: charging one would report a
            # reserve for a drafter the load will not open.
            drafter = _companion_snapshot_sibling(main_gguf_path, _pick_mtp)
        else:
            # A local folder, where the load path (model_config) pairs the drafter
            # to the weight by name so a multi-model folder cannot attach a foreign
            # one, and does accept the ``MTP/`` copy when no root drafter exists.
            # No ``accept`` filter: the load path's one enforces a native-lease
            # boundary, which a read-only estimate does not cross.
            from utils.models.model_config import detect_mtp_file
            drafter = detect_mtp_file(main_gguf_path, search_root = search_root)
        if not drafter:
            return None, 0
        # The whole split family, not just the shard llama-server is handed:
        # the load planner sizes the drafter with _get_gguf_size_bytes, and a
        # split companion reserves every shard. Billing shard 1 alone reports a
        # fit for a launch that allocates several times as much.
        from core.inference.llama_cpp import LlamaCppBackend

        return drafter, LlamaCppBackend._get_gguf_size_bytes(drafter)
    except Exception:
        return None, 0


@router.get("/kv-cache-estimate")
async def get_kv_cache_estimate(
    repo_id: str = Query(..., description = "HF repo ID or local path"),
    quant: str = Query(..., description = "Quantization label (e.g. Q4_K_M)"),
    n_ctx: Optional[int] = Query(
        None,
        ge = 1,
        description = "Context length to size the KV cache for; omit for the model's native length",
    ),
    cache_type_kv: Optional[str] = Query(
        None,
        description = "KV cache dtype (e.g. q8_0, q4_0, q5_0, iq4_nl, f32)",
    ),
    n_parallel: Optional[int] = Query(
        None,
        ge = 1,
        description = (
            "--parallel slots; scales the per-slot KV stream padding. Omit to use "
            "the server's own slot count, which is what a default load gets."
        ),
    ),
    speculative_type: Optional[str] = Query(
        None,
        description = "Speculative decoding mode (mtp, ngram, mtp+ngram, dspark, dflash, auto)",
    ),
    spec_draft_n_max: Optional[int] = Query(
        None,
        ge = 0,
        description = (
            "--spec-draft-n-max. A Hybrid Mamba target keeps one recurrent rollback "
            "state per drafted token, so this is the dominant speculative cost there."
        ),
    ),
    spec_draft_cache_type: Optional[str] = Query(
        None,
        description = "Draft KV cache dtype (--spec-draft-type-k/-v), independent of the main cache",
    ),
    ctx_checkpoints: Optional[int] = Query(
        None,
        ge = 0,
        description = "--ctx-checkpoints; each one adds an SWA snapshot per slot",
    ),
    n_batch: Optional[int] = Query(
        None,
        ge = 1,
        description = "--batch-size; the compute buffers scale with it",
    ),
    n_ubatch: Optional[int] = Query(
        None,
        ge = 1,
        description = "--ubatch-size; the dominant term in the flat compute buffer",
    ),
    tensor_parallel: bool = Query(
        False,
        description = "Tensor mode replicates compute buffers on every device in the pool",
    ),
    disable_vision: bool = Query(
        False,
        description = "Load a vision GGUF without its mmproj, freeing the projector's VRAM",
    ),
    request: Request = None,  # type: ignore[assignment]
    current_subject: str = Depends(get_current_subject),
):
    """KV cache, weight and speculative-decoding bytes for a downloaded GGUF.

    Backs the load dialog's "exceeds memory" warning and the picker's memory
    bar, using the same architecture-aware estimator as load. Best-effort: on
    missing metadata it returns nulls and the UI simply shows nothing.

    ``spec_bytes`` is what an MTP draft mode costs on top of ``kv_bytes``. It is
    null for ngram, which drafts from the generated text and costs no VRAM, and
    for models with no drafter -- the caller draws no segment either way.
    """

    # The header read, the HF cache walk in _resolve_quant_gguf, the drafter
    # lookup and the capability probe are all blocking disk work, and this
    # route is called once per visible row. Run it in a worker so a long model
    # list cannot stall the streamed tokens of a chat in the same process.
    # n_ctx and n_parallel are bound as arguments rather than closed over: the
    # body assigns to both (defaulting them), which would otherwise make them
    # locals of this function and raise before either default could be applied.
    def _estimate(n_ctx: Optional[int] = n_ctx, n_parallel: Optional[int] = n_parallel) -> dict:
        null = {
            "kv_bytes": None,
            "weights_bytes": None,
            "native_context": None,
            "spec_bytes": None,
            "n_ctx": None,
            "projector_bytes": None,
            "spec_unpriced": False,
            "kv_checkpoint_bytes": None,
            "spec_fixed_bytes": None,
            "gpu_bytes": None,
            "compute_bytes": None,
            "total_bytes": None,
            "gpu_floor_bytes": None,
            "context_is_pinned": None,
            "inherited_device_pin": None,
        }
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

            # With no pinned context a GGUF loads at its own native length, which
            # only the metadata we just read knows. Defaulting here saves the caller
            # a round trip spent discovering the number it then asks about.
            # Mirror _resolve_parallel_slots: an omitted count means the server's
            # standing slot count, not one slot. The KV estimator scales per-slot
            # padding, so assuming 1 understates a default load.
            if n_parallel is None:
                state = getattr(getattr(request, "app", None), "state", None)
                n_parallel = getattr(state, "llama_parallel_slots", 1) or 1
            # What the launch will actually serve, not what was asked for. A build
            # without --kv-unified splits the window per slot, so load_model falls
            # back to one; pricing a four-slot default against such a build
            # inflates the cache several times over and can warn OOM about a
            # command that would never launch that way. Same resolution
            # /estimate-memory applies, so the two cannot disagree.
            try:
                from routes.inference import _effective_parallel_slots
                n_parallel = _effective_parallel_slots(n_parallel, diffusion_kind = False)
            except Exception as e:
                logger.debug(f"slot clamp unavailable for '{repo_id}' {quant}: {e}")

            # Whether the caller pinned a context, kept before the default below
            # overwrites it. The planner reads the inherited LLAMA_ARG_CTX_SIZE
            # only when its own n_ctx input is zero, so handing it the native
            # length here priced the header's window for a child that will run at
            # the environment's -- and an inherited context LARGER than native is
            # then underpriced while still reading as auto-fitted.
            _inherited_device_pin = False
            try:
                _dev = (os.environ.get("LLAMA_ARG_DEVICE") or "").strip()
                # "none" is a CPU-only launch, which the planner already answers
                # with zero GPU bytes; that path draws no bar on its own.
                _inherited_device_pin = bool(_dev) and _dev.lower() != "none"
            except Exception as e:
                logger.debug(f"inherited device pin unreadable: {e}")

            _ctx_was_omitted = not n_ctx
            # Whether the launch will auto-fit at all. Only a context nobody
            # pinned gets reduced to fit: load_model keeps a positive inherited
            # LLAMA_ARG_CTX_SIZE rather than fitting it, so an inherited window
            # over budget is a real overage the caller must be allowed to warn
            # about. Resolved below once the inherited value is known.
            _context_is_pinned = not _ctx_was_omitted
            # Same precedence the launch uses: an inherited positive context beats
            # the header, since load_model drops it only when it is zero.
            if _ctx_was_omitted:
                try:
                    from routes.inference import _inherited_ctx_size

                    _inherited = _inherited_ctx_size()
                    _context_is_pinned = bool(_inherited)
                    n_ctx = _inherited or be._context_length
                except Exception as e:
                    logger.debug(f"inherited context unavailable: {e}")
                    n_ctx = be._context_length
            if not n_ctx or n_ctx < 1:
                return null

            # The K/V types the launch will really open, including an inherited
            # LLAMA_ARG_CACHE_TYPE_K/V that no structured setting overrode. The
            # planner already resolves these for gpu_bytes; without the same
            # resolution here kv_bytes stayed at f16 and the KV segment, the
            # per-token rate and the readout all contradicted the total beside
            # them. The heavier of the pair, matching the planner's own choice.
            _effective_cache_type = cache_type_kv
            try:
                from core.inference.llama_cpp import (
                    _kv_bytes_per_elem,
                    _planned_main_cache_types,
                )
                _effective_cache_type = max(
                    _planned_main_cache_types(cache_type_kv, None), key = _kv_bytes_per_elem
                )
            except Exception as e:
                logger.debug(f"cache type resolution failed for '{repo_id}': {e}")

            # ctx_checkpoints is not a rounding error: each saved checkpoint is an
            # SWA snapshot per slot, so a 4-slot SWA model at 32k measures 5.82 GiB
            # with none and 11.82 GiB at the llama.cpp default of 32.
            kv = be._estimate_kv_cache_bytes(
                n_ctx,
                _effective_cache_type,
                n_parallel = n_parallel,
                ctx_checkpoints = ctx_checkpoints or 0,
                n_ubatch = n_ubatch,
            )

            # The checkpoint share of that cache, by difference rather than by
            # re-deriving the SWA layer walk -- the snapshots are the only term that
            # separates the two calls, so asking the same function twice cannot drift
            # from it. This is the same derivation the load planner uses at
            # routes/inference.py, and it is reported separately because llama.cpp
            # keeps these snapshots in HOST heap: the planner's GPU figure is
            # kv_bytes - kv_checkpoint_bytes. Folded into the bar's VRAM total they
            # warn OOM over memory that never touches the card.
            kv_checkpoint = 0
            if ctx_checkpoints:
                _kv_without = be._estimate_kv_cache_bytes(
                    n_ctx,
                    _effective_cache_type,
                    n_parallel = n_parallel,
                    ctx_checkpoints = 0,
                    n_ubatch = n_ubatch,
                )
                kv_checkpoint = max(0, int(kv) - int(_kv_without))

            # DSpark and DFlash attach a separate draft GGUF with its own weights
            # and KV context, and Auto promotes to either ahead of MTP. Pricing
            # them means reproducing the loader's whole sidecar precedence, which
            # is how an estimate ends up charging a drafter the launch never
            # opens. A DSpark sidecar alone runs to about 11 GB, so reporting a
            # comfortable fit that omits it is the worst of the options: say the
            # reserve is unpriced and let the caller draw nothing instead.
            _spec_mode = (speculative_type or "").lower()
            spec_unpriced = _spec_mode in ("dspark", "dflash")
            # Auto is not a mode that declines a sidecar: the load planner promotes it
            # to DSpark or DFlash whenever the repo ships one and the binary supports
            # it, ahead of MTP. Reading the explicit modes alone left the largest
            # single allocation this route can miss -- a DSpark sidecar is about 11 GB
            # -- silently absent from an Auto row, which is the case that reports a
            # comfortable fit and then fails to load. Gated on the binary's own
            # capability for the same reason the planner gates on it: charging (or
            # here, abstaining over) a sidecar the launch never opens would blank the
            # bar on hosts whose llama-server cannot run one.
            if not spec_unpriced and _spec_mode == "auto":
                try:
                    # Imported here, not borrowed from _resolve_mtp_drafter: these
                    # live in that function's local scope, so referencing them from
                    # this one raised NameError into the except below and left
                    # spec_unpriced false -- the silent no-op this guard exists to
                    # prevent, and invisible precisely because it failed quietly.
                    from core.inference.llama_cpp import (
                        _companion_snapshot_sibling,
                        _is_dflash_drafter_path,
                        _pick_dspark,
                        _snapshot_dir_of,
                    )

                    def _pick_dflash(candidates: list[str]) -> Optional[str]:
                        hits = sorted(f for f in candidates if _is_dflash_drafter_path(f))
                        return hits[0] if hits else None

                    _caps = be.probe_server_capabilities() or {}
                    if _snapshot_dir_of(path) is not None:
                        _has_dspark = bool(_companion_snapshot_sibling(path, _pick_dspark))
                        _has_dflash = bool(_companion_snapshot_sibling(path, _pick_dflash))
                    else:
                        # A plain local folder resolves its sidecars the way the load
                        # path does, through the same detectors that populate
                        # gguf_dspark_file / gguf_dflash_file. Restricting this to
                        # snapshots left every local model charting a total with the
                        # drafter missing.
                        from utils.models.drafters.dflash import detect_dflash_file
                        from utils.models.model_config import detect_dspark_file

                        _root = repo_id if is_local else None
                        _has_dspark = bool(detect_dspark_file(path, search_root = _root))
                        _has_dflash = bool(detect_dflash_file(path, search_root = _root))
                    if (_caps.get("supports_dspark") and _has_dspark) or (
                        _caps.get("supports_dflash") and _has_dflash
                    ):
                        spec_unpriced = True
                except Exception as e:
                    logger.debug(f"auto sidecar probe failed for '{repo_id}' {quant}: {e}")

            # A vision GGUF launches with its mmproj resident unless the user
            # turned vision off, and the projector is charged at a worst-case
            # multiple of its file size (_MMPROJ_VRAM_SAFETY), not at it. Left
            # out, a vision row shows a comfortable fit for a launch that has to
            # find another gigabyte or push the projector to the CPU.
            projector = None
            if not disable_vision:
                try:
                    from core.inference.llama_cpp import LlamaCppBackend as _Be
                    from utils.models.model_config import detect_mmproj_file

                    # A cached HF layout puts the weights under a quant subdir
                    # (snapshot/Q4_K_M/model.gguf) and the projector at the
                    # snapshot ROOT, so scanning the weights' own directory finds
                    # nothing. Anchored the same way the drafter lookup is, and
                    # the same way the loader's _download_mmproj anchors on
                    # near_path.
                    from core.inference.llama_cpp import (
                        _companion_snapshot_sibling,
                        _pick_mmproj,
                        _snapshot_dir_of,
                    )

                    if _snapshot_dir_of(path) is not None:
                        mmproj = _companion_snapshot_sibling(path, _pick_mmproj)
                    else:
                        mmproj = detect_mmproj_file(path, search_root = repo_id if is_local else None)
                    if mmproj:
                        projector = int(_Be._get_gguf_size_bytes(mmproj) * _Be._MMPROJ_VRAM_SAFETY)
                except Exception as e:
                    logger.debug(f"mmproj estimate failed for '{repo_id}' {quant}: {e}")

            # Only the MTP modes reserve memory; ngram is free. "auto" may or may
            # not resolve to MTP, and the estimator returns None when it doesn't.
            # Guarded separately: the MTP path reads more metadata than the KV path,
            # and a model it can't size should still get its KV bar rather than
            # dropping the whole response to nulls.
            spec = None
            spec_fixed = None
            if (speculative_type or "").lower() in ("mtp", "mtp+ngram", "auto"):
                try:
                    from core.inference.llama_cpp import (
                        _auto_mode_drops_mtp,
                        _extract_model_size_b,
                        _is_mtp_model_name,
                        _mla_mtp_auto_enabled,
                    )

                    drafter_path, drafter_bytes = _resolve_mtp_drafter(
                        path, search_root = repo_id if is_local else None
                    )
                    # Auto declines MTP on a sub-3B embedded head, where the
                    # per-token cost regresses; a separate drafter is exempt. Pricing
                    # a reserve the load will not take would overstate the bar and
                    # could warn OOM on a model that fits.
                    _mode = (speculative_type or "").lower()

                    # Same reason, one level down: llama-server only takes the MTP
                    # path when it advertises a --spec-type mtp token, and the loader
                    # declines on an inconclusive probe too. Both cover the
                    # separate-drafter path, which is emitted behind the same gate.
                    # Probes are cached on (path, mtime), so this stays cheap.
                    _binary_lacks_mtp = not (be.probe_server_capabilities() or {}).get("mtp_token")
                    # Auto also declines an MLA embedded head (GLM/DeepSeek/Kimi):
                    # that path keeps a duplicated full target-KV context and runs
                    # slower than no speculation, so it is off unless opted into.
                    # A separate drafter is unaffected, as is a non-MLA head.
                    _auto_drops_mla = (
                        _mode == "auto"
                        and be._kv_lora_rank is not None
                        and bool(be._nextn_predict_layers)
                        and not drafter_path
                        and not _mla_mtp_auto_enabled()
                    )
                    # The loader's own precondition (is_mtp_model): a model with no
                    # embedded head, no MTP name and no separate drafter cannot run
                    # MTP at all, so llama-server gets --spec-default and reserves
                    # nothing. Without this check _estimate_mtp_overhead_bytes still
                    # charges its target-side terms, and because
                    # mtp_keeps_target_ctx defaults to True -- deliberately, so an
                    # unsure caller over-reserves -- every MLA model was billed a
                    # second full f16 copy of its own KV. That is the whole cache
                    # again, which is the largest way this bar could be wrong, and it
                    # is wrong in the direction that warns OOM on a model that loads.
                    _not_an_mtp_model = not (
                        bool(be._nextn_predict_layers)
                        or _is_mtp_model_name(repo_id, path)
                        or bool(drafter_path)
                    )
                    if (
                        _binary_lacks_mtp
                        or _auto_drops_mla
                        or _not_an_mtp_model
                        or _auto_mode_drops_mtp(
                            _mode,
                            _extract_model_size_b(repo_id),
                            has_separate_drafter = bool(drafter_path),
                        )
                    ):
                        pass
                    else:
                        # The drafter's own weights: resident for as long as the
                        # drafter is open and not reducible by shortening context.
                        # Reported separately so the caller's auto-fit softening,
                        # which exists for the context-linear part of the cache,
                        # cannot swallow a fixed overage no shorter context fixes.
                        spec_fixed = int(drafter_bytes or 0) or None
                        _effective_draft_n_max = spec_draft_n_max
                        if _effective_draft_n_max is None:
                            try:
                                from routes.inference import (
                                    _cached_estimate_config,
                                    _estimate_draft_n_max,
                                )
                                _effective_draft_n_max = _estimate_draft_n_max(
                                    _cached_estimate_config(repo_id, quant, None, False),
                                    drafter_path or "",
                                    requested = None,
                                    extras = [],
                                )
                            except Exception as e:
                                logger.debug(f"draft depth default failed: {e}")
                                _effective_draft_n_max = 0
                        spec = be._estimate_mtp_overhead_bytes(
                            n_ctx,
                            # Draft K/V types are independent of the main cache and
                            # default to f16 at load; leaving them unset keeps this
                            # from underpricing a quantized-main-cache setup.
                            draft_cache_type_k = spec_draft_cache_type,
                            draft_cache_type_v = spec_draft_cache_type,
                            drafter_path = drafter_path,
                            draft_weights_bytes = drafter_bytes,
                            n_parallel = n_parallel,
                            # A Hybrid Mamba target keeps one recurrent rollback
                            # state per drafted token, which dominates everything
                            # else here: on a 4-slot model at 32k the reserve is
                            # 0.125 GiB at the zero default and 6.944 GiB at a
                            # depth of 16, so omitting it is a 55x understatement.
                            # Blank is not zero. _build_speculative_flags emits
                            # its own default when the field is unset (2 with a
                            # GPU, 3 without), and the rollback state is
                            # multiplied by it, so pricing zero dropped the
                            # dominant allocation on a Hybrid Mamba target
                            # outright. An explicit 0 is still honoured, since
                            # that is a real request to draft nothing.
                            spec_draft_n_max = _effective_draft_n_max,
                        )
                except Exception as e:
                    logger.debug(f"mtp overhead estimate failed for '{repo_id}' {quant}: {e}")

            # The load planner's own answer, alongside this route's field-by-field
            # one. It is the authoritative figure: it applies the inherited
            # environment (LLAMA_ARG_CACHE_TYPE_K/V, LLAMA_ARG_SWA_FULL,
            # LLAMA_ARG_CTX_SIZE), derives the companion search roots the loader
            # derives, and includes the compute buffers -- every term this route
            # would otherwise have to reproduce, and has repeatedly reproduced
            # incompletely. gpu_bytes is what lands on the card, with the host-heap
            # checkpoint share already subtracted.
            #
            # Added beside the existing fields rather than replacing them: the
            # planner defines weights_bytes as the weights PLUS whichever projector
            # and drafter the launch opens, while this route's field shipped meaning
            # the quant file alone. Redefining it would move a number an existing
            # caller already reads.
            planner_gpu = None
            planner_compute = None
            planner_total = None
            planner_floor = None
            planner_unsized = False
            # Bound BEFORE the try, not inside it. Every statement below can
            # raise into the surrounding `except`, and a name defined only on the
            # success path then reads as a NameError from the projection after
            # it -- which this same route has already been bitten by once, in a
            # guard that silently never ran because of it.
            _b = None
            try:
                from routes.inference import (
                    _ESTIMATE_NOT_ON_DISK,
                    _cached_estimate_config,
                    _gguf_memory_breakdown,
                    _localized_estimate_config,
                )

                # Tensor mode replicates its compute buffers on every device in
                # the pool, so pricing one device understates the reserve by a
                # factor of however many cards the launch would use. Only consulted
                # in tensor mode: a layer split does not replicate them the same
                # way, and this is the same count the load panel derives.
                # The effective mode, not the request boolean: the planner turns
                # tensor mode on for an inherited LLAMA_ARG_SPLIT_MODE=tensor even
                # when the per-model toggle is off, and it replicates the compute
                # buffers per device when it does. Reading the toggle alone left
                # n_devices at one for exactly that launch.
                _effective_tp = tensor_parallel
                try:
                    from core.inference.llama_server_args import _effective_tensor_parallel
                    _effective_tp = _effective_tensor_parallel(None, bool(tensor_parallel))
                except Exception as e:
                    logger.debug(f"tensor mode resolution failed for '{repo_id}': {e}")
                _planner_devices = 1
                if _effective_tp:
                    from routes.inference import (
                        _cached_inference_devices,
                        _guard_device_count,
                    )
                    _planner_devices = max(
                        1,
                        _guard_device_count(
                            None, _cached_inference_devices(), tensor_parallel = True
                        ),
                    )
                _cfg = _cached_estimate_config(repo_id, quant, None, False)
                if _cfg is not None and _cfg is not _ESTIMATE_NOT_ON_DISK:
                    _cfg = _localized_estimate_config(_cfg, path)
                    _b = _gguf_memory_breakdown(
                        _cfg,
                        path,
                        n_ctx = 0 if _ctx_was_omitted else n_ctx,
                        speculative_type = speculative_type,
                        n_parallel = n_parallel,
                        cache_type_kv = cache_type_kv,
                        ctx_checkpoints = ctx_checkpoints,
                        disable_vision = disable_vision,
                        spec_draft_n_max = spec_draft_n_max,
                        spec_draft_cache_type = spec_draft_cache_type,
                        n_batch = n_batch,
                        n_ubatch = n_ubatch,
                        tensor_parallel = tensor_parallel,
                        n_devices = _planner_devices,
                    )
                    if _b is not None:
                        # `or None` would fold a real zero into "no answer".
                        # Zero is a meaningful result: inherited placement such as
                        # LLAMA_ARG_DEVICE=none makes the launch entirely CPU
                        # resident, and discarding that sent the caller back to
                        # summing segments and drawing VRAM pressure for a load
                        # that touches no card at all.
                        planner_gpu = int(_b.gpu_bytes)
                        planner_compute = int(_b.compute_bytes) or None
                        planner_total = int(_b.total_bytes) or None
                        planner_unsized = bool(_b.drafter_kv_unsized)
                        # The same plan priced at the shortest context worth
                        # asking for. Whatever is still there cannot be reduced by
                        # shortening context: the drafter's weights, the flat
                        # compute buffer, a Hybrid Mamba target's recurrent
                        # rollback state. Taken by difference against the real
                        # plan rather than by naming those terms, because naming
                        # them is how this route kept missing one; the planner
                        # decides what is fixed, and asking it twice cannot drift
                        # from itself.
                        #
                        # This is what an unpinned row's hard verdict must be
                        # drawn against. Auto-fit can shrink the cache, so a
                        # context-driven overage is not a failure, but nothing it
                        # can do touches this floor.
                        _floor = _gguf_memory_breakdown(
                            _cfg,
                            path,
                            n_ctx = _MIN_PRICED_CONTEXT,
                            speculative_type = speculative_type,
                            n_parallel = n_parallel,
                            cache_type_kv = cache_type_kv,
                            ctx_checkpoints = ctx_checkpoints,
                            disable_vision = disable_vision,
                            spec_draft_n_max = spec_draft_n_max,
                            spec_draft_cache_type = spec_draft_cache_type,
                            n_batch = n_batch,
                            n_ubatch = n_ubatch,
                            tensor_parallel = tensor_parallel,
                            n_devices = _planner_devices,
                        )
                        if _floor is not None:
                            planner_floor = min(int(_floor.gpu_bytes), planner_gpu)
            except Exception as e:
                logger.debug(f"planner breakdown failed for '{repo_id}' {quant}: {e}")

            # Shaped through the canonical MemoryEstimate rather than assembled
            # here, so this route and POST /inference/estimate-memory cannot drift
            # apart in vocabulary the way they had. The projection is what keeps
            # this route's own meaning of `weights_bytes` -- the quant file ALONE
            # -- while the panel's projection keeps its aggregate meaning. The two
            # sit side by side in core/inference/memory_contract.py, which is the
            # only place either mapping is written down.
            #
            # The terms this route prices ITSELF (the target cache, the
            # speculative split, the projector, the checkpoint share) are passed
            # in rather than read off the estimate: the planner has its own
            # figures for some of them and they are not interchangeable.
            _estimate = build_memory_estimate(
                _b if _b is not None else EMPTY_BREAKDOWN,
                quant_file_bytes = weights_bytes or 0,
                native_context = be._context_length,
                # What remains on the card at the shortest context, so a caller
                # can tell a context-driven overage from one no context fixes.
                gpu_floor_bytes = planner_floor,
                # False only when the loader is free to shrink the context. A
                # caller that softens its verdict for an auto-fitted row has to
                # stop softening here, or an inherited window over budget reads
                # as a fit for a launch that will OOM.
                context_is_pinned = _context_is_pinned,
                # An inherited LLAMA_ARG_DEVICE confines the child to the cards it
                # names, and an automatic launch preserves it. The caller's budget
                # is an aggregate over the whole visible inventory, which then
                # describes a pool the launch will not open -- a 30 GiB model
                # reads as fitting 2x24 GiB while the child has one card. The
                # caller cannot see the environment, so it is reported here.
                # Any pin at all is enough to say so: the route does not know the
                # host's inventory, and abstaining is the safe direction.
                inherited_device_pin = _inherited_device_pin,
                # The planner saw a drafter whose cache it could not size, so its
                # own total is a floor.
                spec_unpriced = spec_unpriced or planner_unsized,
                # The planner's own figures, kept exactly as this route computed
                # them above: planner_gpu preserves a real zero, the other two do
                # not. Passed in rather than assigned onto the model afterwards,
                # because Pydantic does not validate assignment by default and a
                # post-construction write puts whatever it is handed straight
                # onto the wire.
                gpu_bytes = planner_gpu,
                compute_bytes = planner_compute,
                total_bytes = planner_total,
                n_ctx = int(n_ctx),
            )
            return project_kv_cache_estimate(
                _estimate,
                kv_bytes = int(kv) if kv else None,
                spec_bytes = int(spec) if spec else None,
                # The part of spec_bytes a shorter context cannot reduce.
                spec_fixed_bytes = spec_fixed if spec else None,
                projector_bytes = projector or None,
                # The part of kv_bytes that llama.cpp keeps in host heap rather
                # than on the card, so a VRAM bar can subtract it. Included in
                # kv_bytes, not beside it: the field shipped meaning the whole
                # cache and an existing caller still reads it that way.
                kv_checkpoint_bytes = kv_checkpoint or None,
            )
        except Exception as e:
            logger.debug(f"kv-cache-estimate failed for '{repo_id}' {quant}: {e}")
            return null

    return await asyncio.to_thread(_estimate)


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
                    # A path-qualified key is not a label a picker can show; without this
                    # the row reads as its whole relative path.
                    display_label = getattr(v, "display_label", None),
                    size_bytes = v.size_bytes,
                    shard_count = int(getattr(v, "shard_count", 0) or 0),
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
    """A primary GGUF weight, not an mmproj vision adapter, an MTP drafter or a calibration
    imatrix. Same rule as ``hub.services.models.common``; pass a snapshot-relative path to
    catch ``MTP/`` copies too."""
    return (
        _is_gguf_filename(name)
        and not _is_mmproj_filename(name)
        and not _is_mtp_drafter(name)
        and not _is_imatrix_path(name)
    )


def _recovered_repo_is_unusable_by_repo_id(repo_info) -> bool:
    """See hub.utils.inventory_scan; False for anything upstream already returns."""
    from hub.utils.inventory_scan import recovered_repo_is_unusable_by_repo_id as impl
    return impl(repo_info)


def _repo_id_will_not_resolve(repo_cache_dir: Path) -> bool:
    """See hub.utils.inventory_scan; True only in the dangling refs/main window."""
    from hub.utils.inventory_scan import repo_id_will_not_resolve as impl
    return impl(repo_cache_dir)


def _default_ref_offers_no_whole_quant(repo_cache_dir: Path) -> bool:
    """See hub.utils.inventory_scan; True when refs/main resolves onto a torn quant.

    _gguf_copy_is_usable asks the stronger question now, so this has no caller here. It stays
    because it was importable before, and a name read only from outside looks unused inside.
    """
    from hub.utils.inventory_scan import default_ref_offers_no_whole_quant as impl
    return impl(repo_cache_dir)


def _gguf_copy_is_usable(repo_info, load_id: Optional[str], active_root: Optional[Path]) -> bool:
    """Whether this copy of the repo holds a quant a load can reach.

    A pinned copy names a complete snapshot. An unpinned copy must be in the active cache and have
    an id that resolves onto a whole quant, which is exactly what withheld the pin.
    """
    if load_id:
        return True
    if active_root is None:
        return False
    try:
        repo_path = Path(repo_info.repo_path)
        if repo_path.parent.resolve(strict = False) != active_root:
            return False
        ref_snapshot = _default_ref_snapshot(repo_path)
        return ref_snapshot is not None and snapshot_has_complete_variants(str(ref_snapshot))
    except (OSError, RuntimeError, ValueError):
        return False


def _snapshot_has_gguf_projector(snapshot: str) -> bool:
    """See hub.utils.inventory_scan; reads the same walk the variant lister reports from."""
    from hub.utils.inventory_scan import snapshot_has_gguf_projector as impl
    return impl(Path(snapshot))


def _cached_files(repo_info) -> list:
    return [f for rev in getattr(repo_info, "revisions", ()) or () for f in cached_repo_files(rev)]


def cached_repo_files(revision) -> list:
    from hub.services.models.cache_inventory import cached_repo_files as impl
    return impl(revision)


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


def _one_shard_family_of(entries: list) -> list:
    """*entries* narrowed to the single shard family the loader would open.

    ``(rel, path, size)`` triples. Same rule as ``hub.utils.gguf.group_gguf_variant_files``:
    every shard of one split GGUF shares a family, two files that do not are two checkpoints, and
    the family kept is the one holding the lexicographically first file. A genuinely split GGUF is
    one family and survives whole.
    """
    if len(entries) < 2:
        return list(entries)
    from hub.utils.gguf import gguf_variant_family

    families: dict[str, list] = {}
    for entry in entries:
        families.setdefault(gguf_variant_family(entry[0]), []).append(entry)
    if len(families) < 2:
        return list(entries)
    return min(families.values(), key = lambda group: min(e[0] for e in group))


def _main_variant_rank(rel_path: str, want: str) -> Optional[int]:
    """How well *want* names this file's variant: 0 for its own key, 1 for the legacy
    quant-label spelling, None for neither.

    *want* is the request VERBATIM: the bare-quant folding is applied per comparison, because
    doing it once up front strips a qualified key's own path punctuation and folds ``exp-a/`` into
    ``expa/``. Directory-qualified keys keep their legacy bare spelling, since stored pins predate
    them. Root-level H3 stems do not: a bare quant names both FL2VA and Ref2VA, and picking the
    first file would load a different task. Exact keys are used alone whenever any exist, and the
    label is the fallback for rows with no root-stem identity.
    """
    from hub.utils.gguf import is_qualified_gguf_variant_key
    from utils.models.model_config import _gguf_variant_key

    label = _main_variant_gguf_label(rel_path)
    if label is None:
        return None
    key = _gguf_variant_key(rel_path)
    if _variant_keys_match(key, want):
        return 0
    if is_qualified_gguf_variant_key(key) and "/" not in key.replace("\\", "/"):
        return None
    return 1 if _normalized_quant_label(label) == _normalized_quant_label(want) else None


def _variant_keys_match(key: str, want: str) -> bool:
    """Whether *want* is *key*, for the exact-key test.

    ``_normalized_quant_label`` strips hyphens and underscores, which is right for a bare quant
    (``UD-Q4_K_XL`` and ``udq4kxl`` are the same ask) and wrong for a path: it folds ``exp-a/`` and
    ``expa/`` into one, so two advertised checkpoints both answered to the other's key. A qualified
    key keeps its punctuation and compares case-insensitively; the legacy folding applies to the
    bare aliases it was written for.
    """
    from hub.utils.gguf import is_qualified_gguf_variant_key

    if is_qualified_gguf_variant_key(key) or is_qualified_gguf_variant_key(want):
        return key.strip().lower() == want.strip().lower()
    return _normalized_quant_label(key) == _normalized_quant_label(want)


def _normalized_quant_label(label: str) -> str:
    return label.lower().replace("-", "").replace("_", "")


def _repo_has_mmproj(repo_info) -> bool:
    """True if the repo ships a GGUF vision adapter (mmproj), so it can
    take image inputs. Cheap: scans already-listed file names only."""
    return any(
        _is_mmproj_filename(f.file_name)
        for revision in repo_info.revisions
        for f in cached_repo_files(revision)
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
    from hub.services.models.common import _iter_gguf_paths as iter_paths
    yield from iter_paths(root, deadline)


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
        for f in cached_repo_files(revision):
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
        for f in cached_repo_files(revision):
            if _is_main_gguf_filename(_cached_repo_file_name(f)):
                latest = max(latest, _blob_mtime(f))
    return latest


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
        if repo_path.parent.resolve(strict = False) == active_root:
            ref_snapshot = _default_ref_snapshot(repo_path)
            if ref_snapshot is not None and snapshot_has_complete_variants(str(ref_snapshot)):
                return None
    except (OSError, RuntimeError, ValueError):
        pass
    # Shared selection key, so this route and the /gguf-variants lister name one snapshot.
    candidates = [
        Path(snapshot)
        for revision in repo_info.revisions
        if (snapshot := getattr(revision, "snapshot_path", None)) is not None
        and any(
            _is_main_gguf_filename(_cached_repo_file_name(f)) for f in cached_repo_files(revision)
        )
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
        return {"cached": cached_gguf_rows()}
    except Exception as e:
        logger.error(f"Error listing cached GGUF repos: {e}", exc_info = True)
        return {"cached": []}


def cached_gguf_rows(cache_scans = None) -> list[dict]:
    if cache_scans is None:
        cache_scans = _all_hf_cache_scans()
    try:
        active_root = _resolve_hf_cache_dir().resolve(strict = False)
    except Exception:
        active_root = None

    seen_lower: dict[str, dict] = {}
    # keep active-cache rank beside rows; the compatibility schema only exposes partialness.
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
                selected = (
                    Path(load_id) if load_id else _default_ref_snapshot(Path(repo_info.repo_path))
                )
                rank = (
                    _gguf_copy_is_usable(repo_info, load_id, active_root),
                    active_root is not None
                    and Path(repo_info.repo_path).parent.resolve(strict = False) == active_root,
                )
                if _preferred_gguf_copy(seen_lower, seen_rank, key, rank, total_size):
                    row = {
                        "repo_id": repo_id,
                        "size_bytes": total_size,
                        "cache_path": str(repo_info.repo_path),
                        "has_vision": _cached_gguf_row_has_vision(repo_info, load_id),
                        "task": _repo_gguf_task(repo_info, selected),
                    }
                    if load_id:
                        row["load_id"] = load_id
                    if not rank[0]:
                        row["partial"] = True
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
    return sorted(
        seen_lower.values(),
        key = lambda c: (-(c.get("last_modified") or 0.0), c["repo_id"].lower()),
    )


def _repo_pipeline_missing_denoiser(repo_info, selected: Optional[Path] = None) -> bool:
    """Companion-only-prefetch check for the snapshot this row loads."""
    from hub.utils import inventory_scan as hf_cache_scan

    if selected is not None:
        return hf_cache_scan.snapshot_pipeline_missing_denoiser(selected)
    return hf_cache_scan.repo_pipeline_missing_denoiser(repo_info)


def _cached_repo_partial(
    repo_id: str,
    repo_cache_dir: Optional[Path] = None,
    snapshot_dir: Optional[Path] = None,
) -> bool:
    """Whether the cached model snapshot is incomplete (cancelled/partial download).
    Reuses the hub inventory scan's snapshot-partial detector (cancel marker, legacy
    .incomplete blob, manifest walk -- cheapest first). ``repo_cache_dir`` scopes the cache copy;
    ``snapshot_dir`` attributes repo-wide signals to the selected revision. Without both, an
    interrupted newer revision can flag a complete pinned revision as partial and hide it from the
    picker.
    Best-effort: a detection error reports not-partial so a scan glitch never hides a
    genuinely usable repo."""
    try:
        from hub.utils.inventory_scan import is_snapshot_partial
        return bool(is_snapshot_partial("model", repo_id, repo_cache_dir, snapshot_dir))
    except Exception:  # noqa: BLE001 -- never fail the listing over a partial probe
        return False


@router.get("/cached-models", response_model = CachedModelsResponse)
async def list_cached_models(
    current_subject: str = Depends(get_current_subject),
    hf_token: Optional[str] = Depends(get_hf_token),
):
    """List non-GGUF model repos downloaded to HF cache, legacy Unsloth cache, and HF default cache."""
    try:
        return {"cached": cached_model_rows()}
    except Exception as e:
        logger.error(f"Error listing cached models: {e}", exc_info = True)
        return {"cached": []}


# Row gate only. Broader than the pin's test below on purpose: a legacy diffusers pipeline
# ships diffusion_pytorch_model.bin, which no weight-prefix rule accepts, but Images lists it.
_NON_GGUF_WEIGHT_EXTENSIONS = (".safetensors", ".bin")


def _snapshot_can_serve_a_load(snapshot: Path) -> bool:
    """Whether this snapshot has metadata and a complete weight payload.

    Transformers snapshots use the local scanners' config-plus-weight check. Diffusers
    pipelines instead have a root ``model_index.json`` and component payloads in subdirs,
    so use the same component-completeness check that guards local media loads.

    huggingface_hub keeps one snapshot per commit, so a partial fetch leaves an unloadable
    dir with a newer mtime than the complete one beside it. Both halves happen: metadata
    only (AutoConfig/AutoTokenizer at a newer revision) and weights only (Unsloth's base
    model pre-warm fetches the shards plus index, no config.json).
    """
    if _local_pipeline_index(snapshot):
        from core.inference.media_locality import _pipeline_components_present
        from hub.utils.inventory_scan import snapshot_pipeline_missing_denoiser
        return _pipeline_components_present(snapshot) and not snapshot_pipeline_missing_denoiser(
            snapshot
        )
    return _is_model_directory(snapshot) and not _snapshot_payload_is_torn(snapshot)


def _snapshot_payload_is_torn(snapshot: Optional[Path]) -> bool:
    """See hub.utils.inventory_scan; True when the snapshot proves its payload incomplete."""
    from hub.utils.inventory_scan import _snapshot_cannot_serve_its_payload
    return _snapshot_cannot_serve_its_payload(snapshot)


def _default_ref_snapshot(repo_cache_dir: Path) -> Optional[Path]:
    """See hub.utils.inventory_scan; the snapshot dir ``refs/main`` names, or ``None``."""
    from hub.utils.inventory_scan import default_ref_snapshot
    return default_ref_snapshot(repo_cache_dir)


def _repo_model_snapshots(repo_info) -> list:
    """Snapshot dirs of a cached non-GGUF repo, newest selection order first."""
    from hub.utils.hf_cache_state import snapshot_selection_key

    candidates = [
        Path(snapshot)
        for revision in getattr(repo_info, "revisions", ()) or ()
        if (snapshot := getattr(revision, "snapshot_path", None)) is not None
    ]
    candidates.sort(key = snapshot_selection_key, reverse = True)
    return candidates


def _repo_is_reachable_by_id(repo_path: Path, active_root: Path, loadable: Optional[Path]) -> bool:
    """Whether loading this repo by its bare id lands somewhere that can serve it.

    Only the active cache makes the id a target, and there ``from_pretrained`` follows
    ``refs/main``: ``repo_id_will_not_resolve`` catches a ref naming no directory, but a
    ref naming an EXISTING half-fetched snapshot resolves fine and then fails. This is the
    non-GGUF twin of ``default_ref_offers_no_whole_quant``. An unreadable ref, or a repo
    with no better sibling, keeps the id it had.
    """
    try:
        if repo_path.parent.resolve(strict = False) != active_root:
            return False
        if _repo_id_will_not_resolve(repo_path):
            return False
    except (OSError, RuntimeError, ValueError):
        return False
    if loadable is None:
        return True
    # No refs/main at all is not "fine", it is unresolvable: huggingface_hub writes
    # refs/<revision> only when revision != commit_hash, so a commit-pinned fetch writes
    # none and a tag-pinned one writes refs/<tag>. Offline the bare id then finds nothing.
    ref_snapshot = _default_ref_snapshot(repo_path)
    return ref_snapshot is not None and _snapshot_can_serve_a_load(ref_snapshot)


def _repo_model_selection(
    repo_info, active_root: Optional[Path]
) -> tuple[Optional[Path], Optional[str]]:
    """The snapshot a load of this repo will read, and the load id pinning it.

    One choice of revision, so every field describing the pick answers for the SAME copy.
    Scanning history instead lets an old revision speak for the row: a repo pushed first as
    a LoRA and later merged into the same id (``push_to_hub_merged``) keeps its stale
    ``adapter_config.json``, and a whole-model row read as an adapter is dropped from the
    chat picker. ``load_id`` is ``None`` when the bare id already reaches a serving copy.
    """
    repo_path = getattr(repo_info, "repo_path", None)
    if repo_path is None or active_root is None:
        return None, None
    usable = []
    for snapshot in _repo_model_snapshots(repo_info):
        try:
            if snapshot.is_dir():
                usable.append(snapshot)
        except OSError:
            continue
    # Newest snapshot a load can actually be served from: the row is listed because SOME
    # revision carries weights, but the newest dir can be a half-fetched partial.
    loadable = next((s for s in usable if _snapshot_can_serve_a_load(s)), None)
    if _repo_is_reachable_by_id(Path(repo_path), active_root, loadable):
        # The bare id follows refs/main, so that ref names the revision a load reads.
        return _default_ref_snapshot(Path(repo_path)) or loadable, None
    # no snapshot can prove it serves a load, so keep the previous newest-dir pin rather than
    # drop it.
    selected = loadable or (usable[0] if usable else None)
    return selected, (str(selected) if selected else None)


def _repo_model_load_id(repo_info, active_root: Optional[Path]) -> Optional[str]:
    """Snapshot dir to load a non-GGUF repo by, or ``None`` when the bare id already works.

    The non-GGUF twin of ``_repo_gguf_load_id``: a repo cached only in the legacy or
    default cache otherwise reads as its bare id, which ``ModelConfig`` resolves through
    the active cache instead, so offline the pick cannot load and online it re-downloads.
    """
    return _repo_model_selection(repo_info, active_root)[1]


def _repo_model_format(repo_info, selected: Optional[Path] = None) -> Optional[str]:
    """``"adapter"`` for a cached LoRA/PEFT repo, else ``None``.

    ``cached_model_rows`` set no format, so every consumer filtering out adapters compared
    against a key that never existed. An adapter ships ``adapter_config.json``; a merge does not.
    Judged on the selected revision; only when none is determinable does history stand in.
    """
    for snapshot in [selected] if selected is not None else _repo_model_snapshots(repo_info):
        try:
            if (snapshot / "adapter_config.json").is_file():
                return "adapter"
        except OSError:
            continue
    return None


def _repo_model_can_chat(repo_info, selected: Optional[Path] = None) -> Optional[bool]:
    """``False`` for a cached encoder-only repo (embedding, CLIP, ViT), else ``None``.

    The classification the hub inventory already applies to its own rows. ``task`` is
    ``None`` for everything not diffusion, so a cached BERT or CLIP otherwise read as an
    ordinary chat model. ``None`` when nothing is conclusive, so unknowns are never hidden.

    The selected revision answers first, so capability describes the copy the pin will
    load rather than whichever snapshot sorts newest. Other snapshots stand in only when
    the selection has no readable config, such as a weights-only snapshot.
    """
    from hub.services.models.common import _local_transformers_can_chat, _read_local_json_object

    ordered = _repo_model_snapshots(repo_info)
    if selected is not None:
        verdict = _local_transformers_can_chat(selected)
        if verdict is not None or _read_local_json_object(selected / "config.json"):
            return verdict
        ordered = [s for s in ordered if s != selected]
    for snapshot in ordered:
        verdict = _local_transformers_can_chat(snapshot)
        if verdict is not None:
            return verdict
    return None


def cached_model_rows(cache_scans = None) -> list[dict]:
    _WEIGHT_EXTENSIONS = _NON_GGUF_WEIGHT_EXTENSIONS
    if cache_scans is None:
        cache_scans = _all_hf_cache_scans()
    try:
        active_root = _resolve_hf_cache_dir().resolve(strict = False)
    except Exception:
        active_root = None

    seen_lower: dict[str, dict] = {}
    for hf_cache in cache_scans:
        for repo_info in hf_cache.repos:
            try:
                if repo_info.repo_type != "model":
                    continue
                repo_id = repo_info.repo_id
                # Pass the snapshot path too so the config check also hides custom Whisper checkpoints.
                if _is_hidden_model(repo_id, str(repo_info.repo_path)):
                    continue
                if _repo_has_gguf_files(repo_info):
                    continue
                selected, model_load_id = _repo_model_selection(repo_info, active_root)
                if _recovered_repo_is_unusable_by_repo_id(repo_info):
                    # That guard withheld these because this schema could describe neither a
                    # partial nor a path; it now carries load_id, so a recovery holding a
                    # snapshot that serves a load is listed pinned to it instead of dropped.
                    if not (
                        model_load_id is not None
                        and selected is not None
                        and _snapshot_can_serve_a_load(selected)
                    ):
                        continue
                total_size = sum(
                    (f.size_on_disk or 0) for rev in repo_info.revisions for f in rev.files
                )
                if total_size == 0:
                    continue
                weight_files = [
                    f for f in _cached_files(repo_info) if f.file_name.endswith(_WEIGHT_EXTENSIONS)
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
                    repo_id, Path(repo_info.repo_path), selected
                ) or _repo_pipeline_missing_denoiser(repo_info, selected)
                # Prefer the most COMPLETE snapshot, then largest: a partial copy in one cache root must not shadow a complete copy in another.
                if existing is None or (not is_partial, total_size) > (
                    not bool(existing.get("partial")),
                    existing["size_bytes"],
                ):
                    row_task = _cached_repo_task(repo_info, selected)
                    is_diffusers = _repo_is_diffusers(repo_info, selected)
                    has_pipeline_index = _repo_has_pipeline_index(repo_info, selected)
                    row = {
                        "repo_id": repo_id,
                        "size_bytes": total_size,
                        "task": row_task,
                    }
                    # Pin a copy its bare id cannot reach, so the pick loads the found snapshot.
                    if model_load_id:
                        row["load_id"] = model_load_id
                    model_format = _repo_model_format(repo_info, selected)
                    if model_format:
                        row["model_format"] = model_format
                    # Without this the picker offers an embedding or CLIP repo as a chat model.
                    if _repo_model_can_chat(repo_info, selected) is False:
                        row["can_chat"] = False
                    # task stays None for a diffusion repo this backend cannot load as a
                    # pipeline, and None is what every chat repo carries, so say it plainly.
                    if is_diffusers:
                        row["diffusers"] = True
                    if is_partial:
                        row["partial"] = True
                    # Listed, so tens of GB of companion weights stay visible and deletable,
                    # but flagged, so no picker offers a denoiser-less repo as a load.
                    if _is_sd_cpp_companion_repo(repo_id):
                        row["companion"] = True
                    # Flag diffusion repos with no pipeline index: loadable only via from_single_file, so pickers must not offer a pipeline load.
                    if row["task"] is not None and not has_pipeline_index:
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

    # Local-only list path: update checks are GGUF-only and happen lazily when variants are viewed.
    return sorted(
        seen_lower.values(),
        key = lambda c: (-(c.get("last_modified") or 0.0), c["repo_id"].lower()),
    )


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
        want = (variant or "").strip()
        candidate_revisions = sorted(
            (rev for repo_info in matching_repos for rev in repo_info.revisions),
            key = lambda rev: getattr(rev, "last_modified", 0) or 0,
            reverse = True,
        )
        for rev in candidate_revisions:
            snapshot = getattr(rev, "snapshot_path", None)
            ranked: dict[int, list[tuple[str, Path]]] = {0: [], 1: []}
            for f in rev.files:
                p = Path(f.file_path)
                rel = f.file_name
                if snapshot:
                    try:
                        rel = p.relative_to(snapshot).as_posix()
                    except ValueError:
                        pass
                rank = _main_variant_rank(rel, want)
                if rank is None:
                    continue
                # Listed as a file of the revision, and keyed like the weights beside it.
                if is_appledouble_metadata(p):
                    continue
                if p.exists() or p.is_symlink():
                    ranked[rank].append((rel, p))
            # Exact keys alone when any exist, else the legacy label spelling.
            matches = ranked[0] or ranked[1]
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
    from utils.paths.path_utils import reveal_in_file_manager

    if not _is_valid_repo_id(repo_id):
        raise HTTPException(status_code = 400, detail = "Invalid repo_id format")
    variant = (variant or "").strip() or None
    path = await asyncio.to_thread(_resolve_cached_model_path, repo_id, variant)
    try:
        await asyncio.to_thread(reveal_in_file_manager, path)
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
