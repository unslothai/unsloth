# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Offline tokenizer / processor loads open the cached snapshot folder, not the Hub repo id.

transformers 5.x asks for ``<subfolder>/config.json`` first; offline by repo id it ignores the
``.no_exist`` marker and raises, while a local folder just skips the missing file.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any, Optional

_PROBE_FILES = (
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
)


def _offline_requested(local_files_only: bool) -> bool:
    if local_files_only:
        return True
    try:
        from huggingface_hub import constants
        return bool(constants.HF_HUB_OFFLINE)
    except Exception:  # noqa: BLE001 -- no hub library means nothing to redirect
        return False


def offline_snapshot_source(
    repo_id: Any,
    subfolder: Optional[str],
    *,
    local_files_only: bool,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> Any:
    """The cached snapshot folder for ``repo_id`` when this load is offline and ``subfolder`` is
    cached there; ``repo_id`` unchanged otherwise (online, a local path, or nothing cached)."""
    if not isinstance(repo_id, str) or not repo_id or not _offline_requested(local_files_only):
        return repo_id
    try:
        if Path(repo_id).expanduser().is_dir():
            return repo_id
    except (OSError, ValueError):
        return repo_id
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001
        return repo_id
    parts = PurePosixPath((subfolder or "").replace("\\", "/").strip("/")).parts
    for name in _PROBE_FILES:
        rel = "/".join((*parts, name))
        try:
            hit = try_to_load_from_cache(repo_id, rel, cache_dir = cache_dir, revision = revision)
        except Exception:  # noqa: BLE001 -- an invalid id or unreadable cache: keep the caller's source
            continue
        if not isinstance(hit, str):
            continue
        root = Path(hit).parent
        for _ in parts:
            root = root.parent
        if root.is_dir():
            return os.fspath(root)
    return repo_id


def _is_tokenizer_or_processor(type_hint: Any) -> bool:
    if not isinstance(type_hint, type):
        return False
    try:
        import transformers
    except Exception:  # noqa: BLE001
        return False
    bases = tuple(
        cls
        for cls in (
            getattr(transformers, "PreTrainedTokenizerBase", None),
            getattr(transformers, "ProcessorMixin", None),
        )
        if isinstance(cls, type)
    )
    return bool(bases) and issubclass(type_hint, bases)


def offline_component_sources(
    pipe: Any,
    *,
    local_files_only: bool,
    cache_dir: Optional[str] = None,
) -> dict[str, str]:
    """``{component: snapshot folder}`` for the tokenizer / processor specs, as the per-component
    ``load_components(pretrained_model_name_or_path = ...)`` dict."""
    if not _offline_requested(local_files_only):
        return {}
    specs = getattr(pipe, "_component_specs", None)
    if not isinstance(specs, dict):
        return {}
    overrides: dict[str, str] = {}
    for name, spec in specs.items():
        if getattr(spec, "default_creation_method", "from_pretrained") != "from_pretrained":
            continue
        if not _is_tokenizer_or_processor(getattr(spec, "type_hint", None)):
            continue
        repo_id = getattr(spec, "pretrained_model_name_or_path", None)
        source = offline_snapshot_source(
            repo_id,
            getattr(spec, "subfolder", None),
            local_files_only = local_files_only,
            cache_dir = cache_dir,
            revision = getattr(spec, "revision", None),
        )
        if source != repo_id:
            overrides[name] = source
    return overrides
