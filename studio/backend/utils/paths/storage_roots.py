# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import tempfile


def studio_root() -> Path:
    return Path.home() / ".unsloth" / "studio"


def cache_root() -> Path:
    """Central cache directory for all studio downloads (models, datasets, etc.)."""
    return Path.home() / ".unsloth" / "studio" / "cache"


def assets_root() -> Path:
    return studio_root() / "assets"


def datasets_root() -> Path:
    return assets_root() / "datasets"


def dataset_uploads_root() -> Path:
    return datasets_root() / "uploads"


def recipe_datasets_root() -> Path:
    return datasets_root() / "recipes"


def outputs_root() -> Path:
    return studio_root() / "outputs"


def exports_root() -> Path:
    return studio_root() / "exports"


def auth_root() -> Path:
    return studio_root() / "auth"


def auth_db_path() -> Path:
    return auth_root() / "auth.db"


def studio_db_path() -> Path:
    return studio_root() / "studio.db"


def tmp_root() -> Path:
    return Path(tempfile.gettempdir()) / "unsloth-studio"


def seed_uploads_root() -> Path:
    return datasets_root() / "seed-uploads"


def unstructured_seed_cache_root() -> Path:
    return tmp_root() / "unstructured-seed-cache"


def unstructured_uploads_root() -> Path:
    return datasets_root() / "unstructured-uploads"


def oxc_validator_tmp_root() -> Path:
    return tmp_root() / "oxc-validator"


def tensorboard_root() -> Path:
    return studio_root() / "runs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents = True, exist_ok = True)
    return path


def legacy_hf_cache_dir() -> Path:
    """Old Unsloth-specific HF hub cache, kept for backward-compat scanning."""
    return cache_root() / "huggingface" / "hub"


def lmstudio_model_dirs() -> list[Path]:
    """Return LM Studio model directories that exist on disk."""
    dirs: list[Path] = []

    # 1. Check LM Studio settings.json for custom downloads folder
    settings_path = Path.home() / ".lmstudio" / "settings.json"
    if settings_path.is_file():
        try:
            with open(settings_path) as f:
                settings = json.load(f)
            downloads = settings.get("downloadsFolder", "")
            if downloads:
                p = Path(downloads).expanduser()
                if p.is_dir():
                    dirs.append(p)
        except Exception:
            pass

    # 2. Legacy LM Studio cache (Linux/macOS)
    if sys.platform == "win32":
        legacy = Path.home() / ".cache" / "lm-studio" / "models"
    else:
        legacy = Path.home() / ".cache" / "lm-studio" / "models"
    if legacy.is_dir():
        dirs.append(legacy)

    return dirs


def _setup_cache_env() -> None:
    """Set cache environment variables for uv and vLLM.

    HuggingFace cache variables (HF_HOME, HF_HUB_CACHE, HF_XET_CACHE)
    are no longer overridden — HF uses its own defaults unless the user
    has explicitly set them.  The legacy Unsloth HF cache at
    ``~/.unsloth/studio/cache/huggingface/hub`` is still scanned for
    backward compatibility via :func:`legacy_hf_cache_dir`.

    Only sets variables that are not already set by the user, so
    explicit overrides are respected.
    Works on Linux, macOS, and Windows.
    """
    root = cache_root()
    defaults = {
        "UV_CACHE_DIR": str(root / "uv"),
        "VLLM_CACHE_ROOT": str(root / "vllm"),
    }
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            Path(value).mkdir(parents = True, exist_ok = True)


def ensure_studio_directories() -> None:
    """Create all standard studio directories on startup."""
    for dir_fn in (
        studio_root,
        assets_root,
        datasets_root,
        dataset_uploads_root,
        recipe_datasets_root,
        unstructured_uploads_root,
        outputs_root,
        exports_root,
        auth_root,
        tensorboard_root,
    ):
        ensure_dir(dir_fn())
    _setup_cache_env()


def _clean_relative_path(
    path_value: str, *, strip_prefixes: tuple[str, ...] = ()
) -> Path:
    path = Path(path_value).expanduser()
    parts = [part for part in path.parts if part not in ("", ".")]
    while parts and parts[0] in strip_prefixes:
        parts = parts[1:]
    return Path(*parts) if parts else Path()


def resolve_under_root(
    path_value: str | None,
    *,
    root: Path,
    strip_prefixes: tuple[str, ...] = (),
) -> Path:
    if not path_value or not str(path_value).strip():
        return root

    path = Path(str(path_value).strip()).expanduser()
    if path.is_absolute():
        return path

    cleaned = _clean_relative_path(str(path), strip_prefixes = strip_prefixes)
    return root / cleaned


def resolve_output_dir(path_value: str | None = None) -> Path:
    return resolve_under_root(
        path_value,
        root = outputs_root(),
        strip_prefixes = ("outputs",),
    )


def resolve_export_dir(path_value: str | None = None) -> Path:
    return resolve_under_root(
        path_value,
        root = exports_root(),
        strip_prefixes = ("exports",),
    )


def resolve_tensorboard_dir(path_value: str | None = None) -> Path:
    return resolve_under_root(
        path_value,
        root = tensorboard_root(),
        strip_prefixes = ("runs", "tensorboard"),
    )


def resolve_dataset_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path

    parts = [part for part in Path(path_value).parts if part not in ("", ".")]
    if parts[:2] == ["assets", "datasets"]:
        parts = parts[2:]
    if parts and parts[0] == "uploads":
        cleaned = Path(*parts[1:]) if len(parts) > 1 else Path()
        return dataset_uploads_root() / cleaned
    if parts and parts[0] == "recipes":
        cleaned = Path(*parts[1:]) if len(parts) > 1 else Path()
        return recipe_datasets_root() / cleaned

    cleaned = Path(*parts) if parts else Path()
    candidates = [
        dataset_uploads_root() / cleaned,
        recipe_datasets_root() / cleaned,
        datasets_root() / cleaned,
        dataset_uploads_root() / cleaned.name,
        recipe_datasets_root() / cleaned.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
