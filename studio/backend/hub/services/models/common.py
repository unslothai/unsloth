# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared model inventory helpers for the Hub service layer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Literal, Optional
from urllib.parse import quote

from hub.schemas.inventory import (
    LocalModelCapabilities,
    LocalModelInfo,
    ModelFormat,
    ModelRuntime,
)
from hub.utils.gguf import (
    extract_quant_label,
    is_gguf_filename as _is_gguf_filename,
    is_mmproj_filename as _is_mmproj_filename,
    is_mtp_drafter_path as _is_mtp_drafter_path,
)
from hub.utils.paths import is_valid_repo_id as _is_valid_repo_id

ModelType = Literal["text", "vision", "audio", "embeddings"]
LocalModelSource = Literal["models_dir", "hf_cache", "lmstudio", "ollama", "custom"]


def _safe_is_dir(path) -> bool:
    # Py >= 3.12 propagates PermissionError (EACCES) from is_dir(); folder scans
    # probe root-owned system dirs, so treat un-stat-able paths as not-a-dir.
    try:
        return Path(path).is_dir()
    except OSError:
        return False


_LOCAL_CHECKPOINT_EXTENSIONS = (
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".msgpack",
    ".npz",
)

_LOCAL_BASE_MODEL_PREFIXES = {
    "checkpoint",
    "checkpoints",
    "export",
    "exports",
    "model",
    "models",
    "output",
    "outputs",
    "run",
    "runs",
    "train",
}
_HF_CACHE_MODEL_FILE_PROBE_LIMIT = 2000


def _is_model_directory(d: Path) -> bool:
    """True when *d* has a config plus real weights; excludes mmproj GGUFs and non-weight ``.bin`` files (``tokenizer.bin``) to avoid false positives."""

    def _is_weight_file(f: Path) -> bool:
        suffix = f.suffix.lower()
        if suffix == ".safetensors":
            return True
        if suffix == ".gguf":
            return "mmproj" not in f.name.lower() and not _is_mtp_drafter_path(f.name)
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


def _local_inventory_id(
    source: str,
    model_format: ModelFormat,
    semantic_id: str,
    variant: Optional[str] = None,
) -> str:
    parts = [
        source,
        model_format,
        quote(semantic_id, safe = ""),
    ]
    if variant:
        parts.append(quote(variant, safe = ""))
    return ":".join(parts)


def _runtime_for_format(model_format: ModelFormat) -> ModelRuntime:
    if model_format == "gguf":
        return "llama_cpp"
    if model_format == "adapter":
        return "adapter"
    if model_format in {"safetensors", "checkpoint"}:
        return "transformers"
    return "unknown"


def _capabilities_for_format(
    model_format: ModelFormat,
    source: str,
    *,
    partial: bool = False,
    requires_variant: bool = False,
) -> LocalModelCapabilities:
    is_complete = not partial
    can_chat = model_format in {"gguf", "safetensors", "adapter", "checkpoint"}
    can_train = model_format in {"safetensors", "checkpoint"} and is_complete
    return LocalModelCapabilities(
        can_train = can_train,
        can_chat = can_chat and is_complete,
        can_delete = source == "hf_cache",
        can_download = False,
        requires_variant = requires_variant,
        supports_lora = model_format in {"safetensors", "checkpoint"} and is_complete,
        supports_vision = False,
    )


def _prefer_complete_larger(
    candidate_partial: bool,
    candidate_size_bytes: int,
    existing_partial: bool,
    existing_size_bytes: int,
) -> bool:
    if candidate_partial != existing_partial:
        return not candidate_partial
    return candidate_size_bytes > existing_size_bytes


def _gguf_variant_state_summary(
    repo_id: str, *, hub_cache: Optional[str | Path] = None
) -> tuple[bool, int]:
    """Whether GGUF variant-scoped state exists and its expected size; a cancelled/in-progress variant may have only manifests/markers/`.incomplete` blobs, which inventory needs to avoid a generic fallback row."""
    from hub.utils import download_manifest

    variant_keys: set[str] = set()
    size_by_variant: dict[str, int] = {}
    for variant, _path in download_manifest.iter_variant_manifests(
        "model",
        repo_id,
        hub_cache = hub_cache,
    ):
        key = variant.lower()
        variant_keys.add(key)
        manifest = download_manifest.read_manifest(
            "model",
            repo_id,
            variant,
            hub_cache = hub_cache,
        )
        if manifest is None:
            continue
        size_by_variant[key] = max(
            size_by_variant.get(key, 0),
            sum(max(0, int(file.size or 0)) for file in manifest.expected_files),
        )
    for variant, _path in download_manifest.iter_variant_markers(
        "model",
        repo_id,
        hub_cache = hub_cache,
    ):
        variant_keys.add(variant.lower())
    return bool(variant_keys), sum(size_by_variant.values())


def _apply_format_aware_partial(
    rows: List[LocalModelInfo],
    *,
    snapshot_partial: bool,
    gguf_partial: bool,
    snapshot_partial_transport: Optional[str] = None,
) -> List[LocalModelInfo]:
    """Rewrite each row's partial flag with format-aware predicates so a hybrid (gguf + safetensors) repo's broken format doesn't taint the clean one; capabilities are recomputed from the new flag."""
    rewritten: List[LocalModelInfo] = []
    for row in rows:
        target = gguf_partial if row.model_format == "gguf" else snapshot_partial
        if not target:
            rewritten.append(row)
            continue
        # GGUF row-level transport is ambiguous (variants may differ); per-variant
        # detail lives on GgufVariantDetail.partial_transport via the variants endpoint.
        partial_transport = None if row.model_format == "gguf" else snapshot_partial_transport
        rewritten.append(
            row.model_copy(
                update = {
                    "partial": True,
                    "partial_transport": partial_transport,
                    "capabilities": _capabilities_for_format(
                        row.model_format,
                        row.source,
                        partial = True,
                        requires_variant = row.capabilities.requires_variant,
                    ),
                }
            )
        )
    return rewritten


def _weight_basename(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _is_adapter_weight_name(name: str) -> bool:
    lower = _weight_basename(name)
    return lower.startswith("adapter_model") and lower.endswith((".safetensors", ".bin"))


def _is_transformers_safetensors_weight_name(name: str) -> bool:
    lower = _weight_basename(name)
    return lower.endswith(".safetensors") and lower.startswith(
        ("model", "pytorch_model", "consolidated")
    )


def _is_transformers_bin_weight_name(name: str) -> bool:
    lower = _weight_basename(name)
    if not lower.endswith(".bin"):
        return False
    return lower.startswith(("pytorch_model", "model", "consolidated", "adapter_model"))


def _is_checkpoint_weight_name(name: str) -> bool:
    lower = _weight_basename(name)
    if lower.endswith(".bin"):
        return _is_transformers_bin_weight_name(lower)
    return lower.endswith(_LOCAL_CHECKPOINT_EXTENSIONS)


def _is_adapter_weight_file(path: Path) -> bool:
    return _is_adapter_weight_name(path.name)


def _is_transformers_safetensors_weight_file(path: Path) -> bool:
    return _is_transformers_safetensors_weight_name(path.name)


def _is_transformers_bin_weight_file(path: Path) -> bool:
    return _is_transformers_bin_weight_name(path.name)


def _is_checkpoint_weight_file(path: Path) -> bool:
    return _is_checkpoint_weight_name(path.name)


def _classify_non_gguf_model_format(
    *,
    has_config: bool,
    has_adapter_config: bool,
    has_adapter_weights: bool,
    has_safetensors: bool,
    has_transformers_safetensors: bool,
    has_checkpoint_weights: bool,
    trusted_hf_cache_repo: bool = False,
) -> Optional[ModelFormat]:
    if has_safetensors and (has_config or (trusted_hf_cache_repo and has_transformers_safetensors)):
        return "safetensors"
    if has_adapter_config and has_adapter_weights:
        return "adapter"
    if has_config and has_checkpoint_weights:
        return "checkpoint"
    return None


def _is_main_gguf_filename(name: str) -> bool:
    return (
        _is_gguf_filename(name) and not _is_mmproj_filename(name) and not _is_mtp_drafter_path(name)
    )


def _iter_gguf_paths(root: Path):
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for path in entries:
            try:
                if path.is_dir() and not path.is_symlink():
                    stack.append(path)
                elif path.is_file() and _is_gguf_filename(path.name):
                    yield path
            except OSError:
                continue


def _iter_immediate_files(path: Path, *, include_symlinks: bool = False) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    try:
        return [
            entry
            for entry in path.iterdir()
            if entry.is_file() or (include_symlinks and entry.is_symlink())
        ]
    except OSError:
        return []


def _iter_hf_cache_model_files(path: Path) -> list[Path]:
    files = _iter_immediate_files(path, include_symlinks = True)
    if not path.is_dir():
        return files
    if any(
        _is_main_gguf_filename(entry.name)
        or _is_transformers_safetensors_weight_file(entry)
        or _is_checkpoint_weight_file(entry)
        for entry in files
    ):
        return files
    try:
        bounded: list[Path] = []
        for index, entry in enumerate(path.rglob("*"), start = 1):
            if index > _HF_CACHE_MODEL_FILE_PROBE_LIMIT:
                break
            if entry.is_file() or entry.is_symlink():
                bounded.append(entry)
        return bounded
    except OSError:
        return []


def _file_size_bytes(path: Path) -> int:
    try:
        if path.is_file() or path.is_symlink():
            return path.stat().st_size
    except OSError:
        return 0
    return 0


def _sum_file_sizes(paths) -> int:
    return sum(_file_size_bytes(path) for path in paths)


def _main_gguf_files(path: Path, *, include_symlinks: bool = False) -> list[Path]:
    return [
        entry
        for entry in _iter_immediate_files(path, include_symlinks = include_symlinks)
        if _is_main_gguf_filename(entry.name)
    ]


def _format_label(model_format: ModelFormat) -> str:
    if model_format == "gguf":
        return "GGUF"
    if model_format == "safetensors":
        return "Safetensors"
    if model_format == "adapter":
        return "Adapter"
    if model_format == "checkpoint":
        return "Checkpoint"
    return "Unknown"


def _read_adapter_config(path: Path) -> dict:
    if not path.is_dir():
        return {}
    try:
        with (path / "adapter_config.json").open("r", encoding = "utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _clean_optional_string(value: object) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _base_model_looks_local(value: str) -> bool:
    raw = value.strip()
    normalized = raw.replace("\\", "/")
    if raw.startswith(("/", "./", "../", "~", "\\\\")) or (
        len(raw) >= 3 and raw[1] == ":" and raw[0].isalpha()
    ):
        return True
    first = normalized.split("/", 1)[0].lower()
    return "/" in normalized and first in _LOCAL_BASE_MODEL_PREFIXES


def _base_model_source(value: Optional[str], adapter_dir: Path) -> Optional[str]:
    if not value:
        return None
    candidates = [value, value.replace("\\", "/")]
    for candidate in candidates:
        try:
            expanded = Path(os.path.expanduser(candidate))
            if expanded.exists() or (adapter_dir / candidate).exists():
                return "local"
        except (OSError, ValueError):
            return "unknown"
    if _base_model_looks_local(value):
        return "local"
    if _is_valid_repo_id(value):
        return "huggingface"
    return "unknown"


def _local_model_info(
    *,
    scan_path: Path,
    load_path: Path,
    source: LocalModelSource,
    model_format: ModelFormat,
    display_name: Optional[str] = None,
    model_id: Optional[str] = None,
    updated_at: Optional[float] = None,
    partial: bool = False,
    requires_variant: bool = False,
    format_variant: Optional[str] = None,
    size_bytes: int = 0,
    base_model: Optional[str] = None,
    base_model_source: Optional[str] = None,
    adapter_type: Optional[str] = None,
    training_method: Optional[str] = None,
    active_cache: Optional[bool] = None,
) -> LocalModelInfo:
    load_id = (
        model_id
        if source == "hf_cache" and model_id and active_cache is not False
        else str(load_path)
    )
    semantic_id = model_id or str(load_path)
    return LocalModelInfo(
        id = load_id,
        inventory_id = _local_inventory_id(
            source,
            model_format,
            semantic_id,
            format_variant,
        ),
        load_id = load_id,
        model_id = model_id,
        active_cache = active_cache if source == "hf_cache" else None,
        display_name = display_name or (scan_path.stem if scan_path.is_file() else scan_path.name),
        path = str(load_path),
        size_bytes = max(0, int(size_bytes or 0)),
        source = source,
        base_model = base_model,
        base_model_source = base_model_source,
        adapter_type = adapter_type,
        training_method = training_method,
        updated_at = updated_at,
        partial = partial,
        model_format = model_format,
        runtime = _runtime_for_format(model_format),
        format_variant = format_variant,
        capabilities = _capabilities_for_format(
            model_format,
            source,
            partial = partial,
            requires_variant = requires_variant,
        ),
    )


def _classify_local_path(
    scan_path: Path,
    source: LocalModelSource,
    *,
    load_path: Optional[Path] = None,
    display_name: Optional[str] = None,
    model_id: Optional[str] = None,
    updated_at: Optional[float] = None,
    partial: bool = False,
    active_cache: Optional[bool] = None,
) -> list[LocalModelInfo]:
    load_path = load_path or scan_path
    files = (
        _iter_hf_cache_model_files(scan_path)
        if source == "hf_cache"
        else _iter_immediate_files(scan_path)
    )
    if not files:
        return []

    rows: list[LocalModelInfo] = []
    include_broken_snapshot_symlinks = source == "hf_cache"
    gguf_files = _main_gguf_files(
        scan_path,
        include_symlinks = include_broken_snapshot_symlinks,
    )
    if gguf_files:
        gguf_size_bytes = _sum_file_sizes(gguf_files)
        variant = (
            extract_quant_label(gguf_files[0].name)
            if scan_path.is_file() and len(gguf_files) == 1
            else None
        )
        rows.append(
            _local_model_info(
                scan_path = scan_path,
                load_path = load_path,
                source = source,
                model_format = "gguf",
                display_name = display_name,
                model_id = model_id,
                updated_at = updated_at,
                partial = partial,
                requires_variant = scan_path.is_dir(),
                format_variant = variant,
                size_bytes = gguf_size_bytes,
                active_cache = active_cache,
            )
        )

    has_config = (scan_path / "config.json").is_file() if scan_path.is_dir() else False
    has_adapter_config = (
        (scan_path / "adapter_config.json").is_file() if scan_path.is_dir() else False
    )
    adapter_config = _read_adapter_config(scan_path) if has_adapter_config else {}
    adapter_base_model = _clean_optional_string(adapter_config.get("base_model_name_or_path"))
    adapter_type = _clean_optional_string(adapter_config.get("peft_type"))
    training_method = _clean_optional_string(adapter_config.get("unsloth_training_method"))
    has_adapter_weights = any(_is_adapter_weight_file(f) for f in files)
    has_safetensors = any(
        f.suffix.lower() == ".safetensors" and not _is_adapter_weight_file(f) for f in files
    )
    has_transformers_safetensors = any(
        _is_transformers_safetensors_weight_file(f) and not _is_adapter_weight_file(f)
        for f in files
    )
    has_checkpoint_weights = any(_is_checkpoint_weight_file(f) for f in files)
    trusted_hf_cache_repo = source == "hf_cache" and bool(model_id)

    model_format = _classify_non_gguf_model_format(
        has_config = has_config,
        has_adapter_config = has_adapter_config,
        has_adapter_weights = has_adapter_weights,
        has_safetensors = has_safetensors,
        has_transformers_safetensors = has_transformers_safetensors,
        has_checkpoint_weights = has_checkpoint_weights,
        trusted_hf_cache_repo = trusted_hf_cache_repo,
    )

    if model_format is not None:
        if model_format == "adapter":
            size_bytes = _sum_file_sizes(f for f in files if _is_adapter_weight_file(f))
        elif model_format == "safetensors":
            size_bytes = _sum_file_sizes(
                f
                for f in files
                if f.suffix.lower() == ".safetensors" and not _is_adapter_weight_file(f)
            )
        else:
            size_bytes = _sum_file_sizes(f for f in files if _is_checkpoint_weight_file(f))
        rows.append(
            _local_model_info(
                scan_path = scan_path,
                load_path = load_path,
                source = source,
                model_format = model_format,
                display_name = display_name,
                model_id = model_id,
                updated_at = updated_at,
                partial = partial,
                size_bytes = size_bytes,
                base_model = adapter_base_model if model_format == "adapter" else None,
                base_model_source = (
                    _base_model_source(adapter_base_model, scan_path)
                    if model_format == "adapter"
                    else None
                ),
                adapter_type = adapter_type if model_format == "adapter" else None,
                training_method = training_method if model_format == "adapter" else None,
                active_cache = active_cache,
            )
        )
    elif not rows:
        fallback_format: ModelFormat = (
            "safetensors" if trusted_hf_cache_repo and has_config else "unknown"
        )
        size_bytes = _sum_file_sizes(files)
        rows.append(
            _local_model_info(
                scan_path = scan_path,
                load_path = load_path,
                source = source,
                model_format = fallback_format,
                display_name = display_name,
                model_id = model_id,
                updated_at = updated_at,
                partial = partial or trusted_hf_cache_repo,
                size_bytes = size_bytes,
                active_cache = active_cache,
            )
        )

    if len(rows) > 1:
        rows = [
            row.model_copy(
                update = {
                    "display_name": f"{row.display_name} ({_format_label(row.model_format)})",
                    "inventory_id": _local_inventory_id(
                        row.source,
                        row.model_format,
                        row.model_id or row.path,
                        row.format_variant,
                    ),
                }
            )
            for row in rows
        ]
    return rows
