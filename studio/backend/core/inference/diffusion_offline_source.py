# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Offline tokenizer / processor loads open the cached snapshot folder, not the Hub repo id.

transformers 5.x ``AutoTokenizer.from_pretrained(repo, subfolder = sub)`` always asks for
``<sub>/config.json`` first (it wants the model type), and a ``ProcessorMixin`` reaches the same
call for its tokenizer. Tokenizer and processor subfolders of a diffusers repo never ship that file:

- online, the Hub answers 404, and transformers treats a missing ``config.json`` as "no config";
- offline (``local_files_only`` or ``HF_HUB_OFFLINE``) with a repo id, transformers passes no commit
  hash, so it never consults huggingface_hub's ``.no_exist`` marker, and it raises "We couldn't
  connect to 'https://huggingface.co'" even though every tokenizer file is in the cache;
- with a local folder, a missing ``config.json`` is skipped by name, and the load succeeds.

So a fully downloaded MiniMax-H3 came back from ``load_components`` with ``processor = None``
(the modular loader only WARNS on a failed component), and the Krea 2 tokenizer, retried with its
4.x compat override, failed the same way. Handing the loader the snapshot folder that the cache
already resolves (``refs/main`` -> ``snapshots/<commit>``) removes the network question entirely.
Online loads are left on the repo id, unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any, Optional

# Files that prove a tokenizer / processor subfolder is in the cache. Any one is enough to locate
# the snapshot folder; the loader itself then reports a genuinely missing file by name.
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
    """``{component: snapshot folder}`` for a modular pipeline's tokenizer / processor specs that
    would otherwise be opened offline by repo id. Feed it to
    ``load_components(pretrained_model_name_or_path = ...)``: a dict value is applied per component,
    and components it does not name keep their spec's own source."""
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
