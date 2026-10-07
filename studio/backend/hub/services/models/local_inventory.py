# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local model, HF cache, LM Studio and Ollama inventory services. Ollama logic lives in :mod:`hub.services.models.ollama`; this module orchestrates all on-device sources and exposes the route handlers."""

from __future__ import annotations


import asyncio
import os
import re
import stat
import threading
from collections import OrderedDict
from pathlib import Path
from typing import List, NamedTuple, Optional

from loggers import get_logger

from hub.schemas.inventory import LocalModelInfo, LocalModelListResponse, ModelFormat
from hub.storage.scan_folders import (
    add_scan_folder_with_status,
    list_scan_folders,
    remove_scan_folder,
)
from hub.utils import download_manifest, gguf, inventory_scan as hf_cache_scan
from hub.utils.host_paths import scrub_paths, short_path_for_log
from hub.utils.paths import (
    hermes_model_dirs,
    hf_default_cache_dir,
    legacy_hf_cache_dir,
    lmstudio_model_dirs,
    normalize_path,
    ollama_model_dirs,
    omlx_model_dirs,
    outputs_root,
    path_is_same_or_child,
    studio_root,
)
from hub.services.models import common as model_common
from hub.services.models.hermes import scan_hermes_dir
from hub.services.models.ollama import scan_ollama_dir
from utils.hidden_models import is_hidden_model
from utils.paths.path_utils import is_appledouble_metadata
from utils.paths.scan_folder_health import (
    annotate_scan_folders,
    note_scan_folder_scanned,
    record_scan_failure,
    refresh_failed_scan_folders,
)

logger = get_logger(__name__)
_MAX_MODELS_PER_CUSTOM_FOLDER = 200
_MAX_CUSTOM_FOLDER_ENTRIES = 2000
_MODEL_SIGNAL_PROBE_LIMIT = 200


class _LocalInventorySources(NamedTuple):
    hf_cache_dir: Path
    legacy_hf: Path
    hf_default: Path
    lm_dirs: tuple[Path, ...]
    ollama_dirs: tuple[Path, ...]
    hermes_dirs: tuple[Path, ...]
    known_hf_caches: tuple[Path, ...]
    omlx_dirs: tuple[Path, ...] = ()


_LocalInventoryKey = tuple[str, _LocalInventorySources, tuple[str, ...], int]
_local_inventory_flights: dict[
    tuple[asyncio.AbstractEventLoop, _LocalInventoryKey], asyncio.Task[LocalModelListResponse]
] = {}


# Past this cap, serve the freshest scan instead of retrying.
_LOCAL_INVENTORY_MAX_ATTEMPTS = 8


class _LocalCacheChanged(RuntimeError):
    def __init__(self, response: LocalModelListResponse) -> None:
        super().__init__("local inventory sources changed during the scan")
        self.response = response


# Local aliases keep the extracted code close to the original implementation.
LocalModelSource = model_common.LocalModelSource
_is_model_directory = model_common._is_model_directory
_local_inventory_id = model_common._local_inventory_id
_local_model_info = model_common._local_model_info
_capabilities_for_format = model_common._capabilities_for_format
_apply_format_aware_partial = model_common._apply_format_aware_partial
_classify_local_path = model_common._classify_local_path
_is_main_gguf_filename = model_common._is_main_gguf_filename
_is_transformers_bin_weight_file = model_common._is_transformers_bin_weight_file
_prefer_complete_larger = model_common._prefer_complete_larger
_gguf_variant_state_summary = model_common._gguf_variant_state_summary
_is_diffusers_pipeline_dir = model_common._is_diffusers_pipeline_dir


def _account_access():
    """Imported on use: the CLI reads this inventory without FastAPI, which account_access needs."""
    from hub.services.models import account_access
    return account_access


def _http_error(status_code: int, detail: str):
    from fastapi import HTTPException
    return HTTPException(status_code = status_code, detail = detail)


def _is_immediate_model_weight_file(path: Path) -> bool:
    if is_appledouble_metadata(path):
        return False
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return True
    if suffix == ".gguf":
        return _is_main_gguf_filename(path.name)
    if suffix == ".bin":
        return _is_transformers_bin_weight_file(path)
    return False


def _has_immediate_model_weight(
    path: Path, *, probe_limit: int = _MODEL_SIGNAL_PROBE_LIMIT
) -> bool:
    try:
        for index, entry in enumerate(path.iterdir(), start = 1):
            if index > probe_limit:
                break
            try:
                if entry.is_file() and _is_immediate_model_weight_file(entry):
                    return True
            except OSError:
                continue
    except OSError:
        return False
    return False


def _has_immediate_model_signal(
    path: Path, *, probe_limit: int = _MODEL_SIGNAL_PROBE_LIMIT
) -> bool:
    try:
        if (path / "config.json").exists() or (path / "adapter_config.json").exists():
            return True
    except OSError:
        return False
    if _is_diffusers_pipeline_dir(path):
        return True
    return _has_immediate_model_weight(path, probe_limit = probe_limit)


def _is_model_directory_for_scan(path: Path, *, entry_limit: int | None) -> bool:
    if _is_diffusers_pipeline_dir(path):
        return True
    if entry_limit is None:
        return _is_model_directory(path)
    try:
        has_config = (path / "config.json").exists() or (path / "adapter_config.json").exists()
    except OSError:
        return False
    return has_config and _has_immediate_model_weight(path)


_MAX_NESTED_SCAN_DEPTH = 8
_MAX_NESTED_SCAN_DIRS = 500
_MAX_NESTED_SCAN_ENTRIES = 20000
# blobs: an Ollama store's content-addressed files.
_NESTED_SCAN_SKIP_NAMES = frozenset({"ollama_links", "node_modules", "__pycache__", "blobs"})


def _is_plain_dir_entry(entry: os.DirEntry) -> bool:
    if not entry.is_dir(follow_symlinks = False):
        return False
    # Pre-3.12 junctions look like dirs; OneDrive placeholders are real folders and stay walkable.
    tag = getattr(entry.stat(follow_symlinks = False), "st_reparse_tag", 0)
    return tag != _IO_REPARSE_TAG_MOUNT_POINT


def is_loadable_model_dir(path: Path) -> bool:
    """A folder the scan lists as a loadable model. A config with no weights beside it is not one: its model may sit a level down."""
    return _is_diffusers_pipeline_dir(path) or _has_immediate_model_weight(path)


def nested_scan_roots(folder_path: Path) -> list[Path]:
    """Sub-folders of a recursive scan folder to scan like the folder itself (#6371). Skips listed models, HF cache
    repos, hidden folders, symlinks and junctions, so each model is reached once and the walk stays inside."""
    # Same test _scan_models_dir uses to list the folder as one model.
    if _is_model_directory_for_scan(folder_path, entry_limit = _MAX_CUSTOM_FOLDER_ENTRIES):
        return []
    roots: list[Path] = []
    visited = 0
    stack: list[tuple[Path, int]] = [(folder_path, 0)]
    while stack and len(roots) < _MAX_NESTED_SCAN_DIRS:
        current, depth = stack.pop()
        if depth >= _MAX_NESTED_SCAN_DEPTH:
            continue
        children: list[Path] = []
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    visited += 1
                    if visited > _MAX_NESTED_SCAN_ENTRIES:
                        break
                    name = entry.name
                    if (
                        name.startswith((".", "models--", "datasets--", "spaces--"))
                        or name in _NESTED_SCAN_SKIP_NAMES
                    ):
                        continue
                    try:
                        if not _is_plain_dir_entry(entry):
                            continue
                    except OSError:
                        continue
                    child = Path(entry.path)
                    if not is_loadable_model_dir(child):
                        children.append(child)
        except OSError:
            continue
        children.sort()
        roots.extend(children[: _MAX_NESTED_SCAN_DIRS - len(roots)])
        stack.extend((child, depth + 1) for child in reversed(children))
        if visited > _MAX_NESTED_SCAN_ENTRIES:
            break
    return roots


_SHARD_EVIDENCE_RE = re.compile(r"-\d+-of-\d+\.|\.index\.json$", re.IGNORECASE)
_PAYLOAD_SUFFIXES = (".safetensors", ".gguf", *model_common._LOCAL_CHECKPOINT_EXTENSIONS)
_PAYLOAD_VERDICT_CACHE_MAX = 4096
_payload_verdicts: "OrderedDict[str, tuple[tuple, bool]]" = OrderedDict()
_payload_verdicts_lock = threading.Lock()


def _payload_evidence(scan_path: Path) -> tuple[bool, bool, tuple]:
    """Whether weights / quants could be torn (numbered shard, index, empty weight), plus a fingerprint of every file."""
    weights = quants = False
    fingerprint = []
    for dirpath, _dirnames, filenames in os.walk(scan_path):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path)
                entry = (path, st.st_size, st.st_mtime_ns, st.st_ino)
                empty = st.st_size <= 0
            except OSError:
                entry = (path, -1, -1, -1)
                empty = True
            fingerprint.append(entry)
            lower = name.lower()
            torn = _SHARD_EVIDENCE_RE.search(name) is not None or (
                empty and lower.endswith(_PAYLOAD_SUFFIXES)
            )
            if torn:
                if lower.endswith(".gguf"):
                    quants = True
                else:
                    weights = True
    return weights, quants, tuple(sorted(fingerprint))


def _weights_complete(scan_path: Path, fingerprint: tuple) -> bool:
    # Files are the judge's only input. Quants are not cached: their judge also reads the account's scan folders.
    key = os.path.abspath(scan_path)
    with _payload_verdicts_lock:
        hit = _payload_verdicts.get(key)
        if hit is not None and hit[0] == fingerprint:
            _payload_verdicts.move_to_end(key)
            return hit[1]
    complete = hf_cache_scan.snapshot_holds_a_complete_payload(scan_path, quants = False)
    with _payload_verdicts_lock:
        _payload_verdicts[key] = (fingerprint, complete)
        _payload_verdicts.move_to_end(key)
        while len(_payload_verdicts) > _PAYLOAD_VERDICT_CACHE_MAX:
            _payload_verdicts.popitem(last = False)
    return complete


def _apply_payload_partial(scan_path: Path, rows: List[LocalModelInfo]) -> List[LocalModelInfo]:
    """Local folders carry no downloader markers, so only the payload shows a torn download. ``unknown`` is skipped: a diffusers pipeline's weights live in component subdirs."""
    if not rows:
        return rows
    if scan_path.is_file():
        # A loose quant, split or not: llama-server opens every part, so a missing or empty one fails the load.
        from utils.models.model_config import colocated_split_shards

        candidates = [scan_path]
        try:
            if scan_path.is_symlink():
                # Same fallback as _local_gguf_load_path: a lone link loads from its target's set.
                candidates.append(scan_path.resolve())
        except OSError:
            pass
        for candidate in candidates:
            shards, complete = colocated_split_shards(candidate)
            try:
                if complete and all(shard.stat().st_size > 0 for shard in shards):
                    return rows
            except OSError:
                continue
        return _apply_format_aware_partial(rows, snapshot_partial = False, gguf_partial = True)
    if not scan_path.is_dir():
        return rows
    judged = {row.model_format for row in rows} - {"unknown"}
    if not judged:
        return rows
    weights, quants, fingerprint = _payload_evidence(scan_path)
    snapshot_partial = (
        weights and bool(judged - {"gguf"}) and not _weights_complete(scan_path, fingerprint)
    )
    gguf_partial = (
        quants
        and "gguf" in judged
        and not hf_cache_scan.snapshot_holds_a_complete_payload(scan_path, quants = True)
    )
    if not snapshot_partial and not gguf_partial:
        return rows
    return _apply_format_aware_partial(
        rows, snapshot_partial = snapshot_partial, gguf_partial = gguf_partial
    )


def _resolve_hf_cache_dir() -> Path:
    from utils.hf_cache_settings import get_hf_cache_paths
    return get_hf_cache_paths().hub_cache


def _local_inventory_sources() -> _LocalInventorySources:
    from utils.hf_cache_settings import known_hf_hub_caches
    return _LocalInventorySources(
        _resolve_hf_cache_dir(),
        legacy_hf_cache_dir(),
        hf_default_cache_dir(),
        tuple(lmstudio_model_dirs()),
        tuple(ollama_model_dirs()),
        tuple(hermes_model_dirs()),
        tuple(known_hf_hub_caches()),
        tuple(omlx_model_dirs()),
    )


def _scan_models_dir(
    models_dir: Path,
    *,
    limit: int | None = None,
    entry_limit: int | None = None,
) -> List[LocalModelInfo]:
    if not models_dir.exists() or not models_dir.is_dir():
        return []

    _is_self_model = _is_model_directory_for_scan(
        models_dir,
        entry_limit = entry_limit,
    )

    if _is_self_model:
        try:
            updated_at = models_dir.stat().st_mtime
        except OSError:
            updated_at = None
        rows = _classify_local_path(
            models_dir,
            "models_dir",
            updated_at = updated_at,
        )
        return _apply_payload_partial(models_dir, rows)

    found: List[LocalModelInfo] = []
    visited = 0
    try:
        children = models_dir.iterdir()
    except OSError:
        return found
    for child in children:
        if limit is not None and len(found) >= limit:
            break
        visited += 1
        if entry_limit is not None and visited > entry_limit:
            break
        try:
            is_dir = child.is_dir()
            is_gguf_file = (
                not is_dir
                and child.suffix.lower() == ".gguf"
                and child.is_file()
                and not is_appledouble_metadata(child)
            )
            if not is_dir and not is_gguf_file:
                continue
            has_model_files = is_gguf_file or _has_immediate_model_signal(child)
        except OSError:
            continue
        if not has_model_files:
            continue
        try:
            updated_at = child.stat().st_mtime
        except OSError:
            updated_at = None
        rows = _classify_local_path(
            child,
            "models_dir",
            updated_at = updated_at,
        )
        rows = _apply_payload_partial(child, rows)
        if limit is not None:
            rows = rows[: max(0, limit - len(found))]
        found.extend(rows)

    return found


def _safe_is_dir(path: Path) -> bool:
    """``Path.is_dir()`` treating an unreadable path (``PermissionError`` / ``OSError`` on a restricted ``~/.cache/huggingface/hub``) as "not a directory", so the inventory skips that source instead of 500ing the Hub page."""
    try:
        return path.is_dir()
    except OSError:
        return False


def _hf_repo_dir_has_content(repo_dir: Path) -> bool:
    blobs_dir = repo_dir / "blobs"
    try:
        if blobs_dir.is_dir():
            for entry in blobs_dir.iterdir():
                if entry.is_file() or entry.is_symlink():
                    return True
    except OSError:
        pass
    return _hf_snapshots_hold_files(repo_dir)


def _hf_snapshots_hold_files(repo_dir: Path) -> bool:
    """Whether the newest snapshot (the one ``_scan_hf_cache`` classifies) holds a real file.
    Without symlinks huggingface_hub moves blobs into ``snapshots/<rev>/`` and leaves ``blobs/``
    empty. Walked with ``scandir``, bounded by entries read (``rglob`` lists a whole directory
    before yielding); unreadable entries are skipped."""
    snapshot = hf_cache_scan.latest_snapshot_dir(repo_dir)
    if snapshot is None:
        return False
    walked = 0
    pending = [snapshot]
    while pending:
        try:
            entries = os.scandir(pending.pop())
        except OSError:
            continue
        with entries:
            listing = iter(entries)
            while True:
                try:
                    entry = next(listing)
                except StopIteration:
                    break
                except OSError:
                    break
                walked += 1
                if walked > model_common._HF_CACHE_MODEL_FILE_PROBE_LIMIT:
                    return False
                try:
                    if entry.is_dir(follow_symlinks = False):
                        pending.append(Path(entry.path))
                    elif (
                        entry.is_file()
                        and entry.name not in hf_cache_scan._CACHE_ENTRIES_TO_IGNORE
                        and not is_appledouble_metadata(Path(entry.path))
                    ):
                        return True
                except OSError:
                    continue
    return False


def _discover_hf_cache(
    cache_dir: Path, *, entry_limit: int | None = None
) -> list[tuple[Path, str, Optional[float]]]:
    if not _safe_is_dir(cache_dir):
        return []

    discovered: List[tuple[Path, str, Optional[float]]] = []
    visited = 0
    try:
        entries = cache_dir.iterdir()
    except OSError:
        return []
    for repo_dir in entries:
        visited += 1
        if entry_limit is not None and visited > entry_limit:
            break
        if not repo_dir.name.startswith("models--"):
            continue
        if not repo_dir.is_dir():
            continue
        if not _hf_repo_dir_has_content(repo_dir):
            continue
        repo_name = repo_dir.name[len("models--") :]
        if not repo_name:
            continue
        model_id = repo_name.replace("--", "/")
        try:
            updated_at = repo_dir.stat().st_mtime
        except OSError:
            updated_at = None
        discovered.append((repo_dir, model_id, updated_at))
    return discovered


def _scan_hf_cache(
    cache_dir: Path,
    *,
    entry_limit: int | None = None,
    active_cache: bool = True,
    discovered: Optional[list[tuple[Path, str, Optional[float]]]] = None,
    variant_states: Optional[download_manifest.VariantStateIndex] = None,
    active_hub_cache: Optional[Path] = None,
) -> List[LocalModelInfo]:
    if discovered is None:
        discovered = _discover_hf_cache(cache_dir, entry_limit = entry_limit)
    if not discovered:
        return []
    if variant_states is None:
        # Reached when the guarded build already failed; degrade to per-repo reads.
        try:
            variant_states = download_manifest.build_variant_state_index(
                [("model", model_id, cache_dir) for _repo, model_id, _updated in discovered],
                active_hub_cache = active_hub_cache
                or (cache_dir if active_cache else _resolve_hf_cache_dir()),
            )
        except Exception as e:
            logger.warning(
                "Could not build Hub-state index for %s: %s",
                short_path_for_log(cache_dir),
                scrub_paths(e),
            )
            variant_states = None

    found: list[LocalModelInfo] = []
    for repo_dir, model_id, updated_at in discovered:
        variant_state = (
            variant_states.for_repo("model", model_id, hub_cache = cache_dir)
            if variant_states is not None
            else None
        )
        snapshot_partial = hf_cache_scan.is_snapshot_partial(
            "model",
            model_id,
            repo_dir,
            variant_state = variant_state,
        )
        gguf_partial = hf_cache_scan.is_gguf_repo_partial(
            model_id,
            repo_dir,
            variant_state = variant_state,
        )
        has_gguf_variant_state, gguf_variant_state_size = _gguf_variant_state_summary(
            model_id,
            hub_cache = cache_dir,
            variant_state = variant_state,
        )
        snapshot_partial_transport = (
            hf_cache_scan.partial_transport_for(
                "model",
                model_id,
                repo_cache_dir = repo_dir,
            )
            if snapshot_partial
            else None
        )
        snapshot_partial_resumable = snapshot_partial and hf_cache_scan.partial_resume_available(
            "model",
            model_id,
            repo_cache_dir = repo_dir,
        )
        resolved = hf_cache_scan.resolve_hf_cache_realpath(repo_dir)
        scan_path = Path(resolved) if resolved else repo_dir
        load_path = repo_dir if active_cache else scan_path
        rows = _classify_local_path(
            scan_path,
            "hf_cache",
            load_path = load_path,
            display_name = model_id.split("/")[-1],
            model_id = model_id,
            updated_at = updated_at,
            partial = False,
            active_cache = active_cache,
        )
        if not rows:
            if has_gguf_variant_state and gguf_partial:
                rows = [
                    _local_model_info(
                        scan_path = repo_dir,
                        load_path = load_path,
                        source = "hf_cache",
                        model_format = "gguf",
                        display_name = model_id.split("/")[-1],
                        model_id = model_id,
                        updated_at = updated_at,
                        partial = True,
                        requires_variant = True,
                        size_bytes = gguf_variant_state_size,
                        active_cache = active_cache,
                    )
                ]
            else:
                rows = [
                    _local_model_info(
                        scan_path = repo_dir,
                        load_path = load_path,
                        source = "hf_cache",
                        model_format = "unknown",
                        display_name = model_id.split("/")[-1],
                        model_id = model_id,
                        updated_at = updated_at,
                        partial = snapshot_partial or gguf_partial,
                        active_cache = active_cache,
                    )
                ]
        elif (
            has_gguf_variant_state
            and gguf_partial
            and not any(row.model_format == "gguf" for row in rows)
        ):
            rows.append(
                _local_model_info(
                    scan_path = repo_dir,
                    load_path = load_path,
                    source = "hf_cache",
                    model_format = "gguf",
                    display_name = model_id.split("/")[-1],
                    model_id = model_id,
                    updated_at = updated_at,
                    partial = True,
                    requires_variant = True,
                    size_bytes = gguf_variant_state_size,
                    active_cache = active_cache,
                )
            )
        rows = _apply_format_aware_partial(
            rows,
            snapshot_partial = snapshot_partial,
            gguf_partial = gguf_partial,
            snapshot_partial_transport = snapshot_partial_transport,
            snapshot_partial_resumable = snapshot_partial_resumable,
        )
        # Denoiser-less pipeline: partial so no picker loads it; flagged so Hub offers no Continue.
        if not snapshot_partial and hf_cache_scan.snapshot_pipeline_missing_denoiser(
            hf_cache_scan.latest_snapshot_dir(repo_dir)
        ):
            rows = [
                row
                if row.model_format == "gguf"
                else row.model_copy(update = {"companion_prefetch": True})
                for row in _apply_format_aware_partial(
                    rows, snapshot_partial = True, gguf_partial = gguf_partial
                )
            ]
        found.extend(rows)
    return found


def _scan_lmstudio_dir(
    lm_dir: Path,
    *,
    entry_limit: int | None = None,
    source: LocalModelSource = "lmstudio",
) -> List[LocalModelInfo]:
    """Scan a ``publisher/model-name`` tree (or top-level standalone GGUFs); LM Studio and oMLX
    share this layout, ``source`` names the app."""
    if not lm_dir.exists() or not lm_dir.is_dir():
        return []

    if _is_model_directory(lm_dir) or _is_diffusers_pipeline_dir(lm_dir):
        try:
            updated_at = lm_dir.stat().st_mtime
        except OSError:
            updated_at = None
        rows = _classify_local_path(
            lm_dir,
            source,
            updated_at = updated_at,
        )
        return _apply_payload_partial(lm_dir, rows)

    found: List[LocalModelInfo] = []
    visited = 0
    exhausted = False

    def _consume_visit() -> bool:
        nonlocal visited
        visited += 1
        return entry_limit is not None and visited > entry_limit

    try:
        children = lm_dir.iterdir()
    except OSError:
        return found
    for child in children:
        if _consume_visit():
            break
        try:
            if not child.is_dir():
                if (
                    child.suffix.lower() == ".gguf"
                    and child.is_file()
                    and not is_appledouble_metadata(child)
                ):
                    try:
                        updated_at = child.stat().st_mtime
                    except OSError:
                        updated_at = None
                    rows = _classify_local_path(
                        child,
                        source,
                        updated_at = updated_at,
                    )
                    found.extend(_apply_payload_partial(child, rows))
                continue

            if _is_model_directory(child) or _is_diffusers_pipeline_dir(child):
                try:
                    updated_at = child.stat().st_mtime
                except OSError:
                    updated_at = None
                rows = _classify_local_path(
                    child,
                    source,
                    updated_at = updated_at,
                )
                found.extend(_apply_payload_partial(child, rows))
                continue

            for model_dir in child.iterdir():
                if _consume_visit():
                    exhausted = True
                    break
                try:
                    if model_dir.is_dir():
                        has_model = _has_immediate_model_signal(model_dir)
                        if not has_model:
                            continue
                        model_id = f"{child.name}/{model_dir.name}"
                        try:
                            updated_at = model_dir.stat().st_mtime
                        except OSError:
                            updated_at = None
                        rows = _classify_local_path(
                            model_dir,
                            source,
                            display_name = model_dir.name,
                            model_id = model_id,
                            updated_at = updated_at,
                        )
                        found.extend(_apply_payload_partial(model_dir, rows))
                    elif (
                        model_dir.suffix.lower() == ".gguf"
                        and model_dir.is_file()
                        and not is_appledouble_metadata(model_dir)
                    ):
                        try:
                            updated_at = model_dir.stat().st_mtime
                        except OSError:
                            updated_at = None
                        rows = _classify_local_path(
                            model_dir,
                            source,
                            model_id = f"{child.name}/{model_dir.stem}",
                            updated_at = updated_at,
                        )
                        found.extend(_apply_payload_partial(model_dir, rows))
                except OSError:
                    continue
            if exhausted:
                break
        except OSError:
            continue
    return found


def _resolve_allowed_models_dir(models_dir: str, allowed_roots: list[Path]) -> Path:
    """Resolve a requested model scan directory without widening subpaths."""
    if not models_dir or not models_dir.strip():
        raise ValueError("Directory not allowed")

    requested = Path(os.path.realpath(os.path.expanduser(normalize_path(models_dir.strip()))))
    if any(path_is_same_or_child(requested, root) for root in allowed_roots):
        return requested

    raise ValueError("Directory not allowed")


def _inventory_path_identity(raw_path: str) -> str:
    """Canonical identity for scan roots used in shared-flight keys."""
    raw = raw_path.strip()
    try:
        normalized = normalize_path(raw)
        return os.path.normcase(os.path.realpath(os.path.expanduser(normalized)))
    except (OSError, UnicodeError, ValueError):
        return os.path.normcase(raw)


def _inventory_physical_identity(raw_path: str) -> str:
    """physical identity for an existing discovered path without lossy name folding."""
    return gguf.local_path_physical_identity(raw_path)


_IO_REPARSE_TAG_MOUNT_POINT = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)


def _is_link_component(path: Path) -> bool:
    # is_symlink() is False for a Windows junction; read the reparse tag too.
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISLNK(st.st_mode) or (
        getattr(st, "st_reparse_tag", 0) == _IO_REPARSE_TAG_MOUNT_POINT
    )


def _local_model_path_is_symlink(raw_path: str) -> bool:
    path = Path(raw_path)
    return any(_is_link_component(p) for p in (path, *path.parents))


def _prefer_local_inventory_row(candidate: LocalModelInfo, existing: LocalModelInfo) -> bool:
    if candidate.partial != existing.partial:
        return not candidate.partial
    if (candidate.active_cache is True) != (existing.active_cache is True):
        return candidate.active_cache is True
    candidate_link = _local_model_path_is_symlink(candidate.path)
    existing_link = _local_model_path_is_symlink(existing.path)
    if candidate_link != existing_link:
        return not candidate_link
    return _prefer_complete_larger(
        candidate.partial,
        candidate.size_bytes,
        existing.partial,
        existing.size_bytes,
    )


def _custom_alias_key(model: LocalModelInfo) -> str:
    # Resolve only the scan root: links below it are distinct aliases.
    root = model._scan_root
    if not root:
        return model.path
    try:
        return os.path.join(os.path.realpath(root), os.path.relpath(model.path, root))
    except (OSError, ValueError):
        return model.path


def _dedupe_custom_local_models(custom_models: List[LocalModelInfo]) -> list[LocalModelInfo]:
    """Distinct symlink aliases of one model stay separate rows so each keeps its own settings (#10605)."""
    by_physical: dict[tuple[str, str], list[LocalModelInfo]] = {}
    for model in custom_models:
        physical = _inventory_physical_identity(model.path)
        by_physical.setdefault((physical, model.model_format), []).append(model)

    kept: list[LocalModelInfo] = []
    for group in by_physical.values():
        by_alias_path: dict[str, list[LocalModelInfo]] = {}
        for model in group:
            by_alias_path.setdefault(_custom_alias_key(model), []).append(model)
        unique_rows: list[LocalModelInfo] = []
        for alias_group in by_alias_path.values():
            winner = alias_group[0]
            for candidate in alias_group[1:]:
                if _prefer_local_inventory_row(candidate, winner):
                    winner = candidate
            unique_rows.append(winner)

        symlinks = [m for m in unique_rows if _local_model_path_is_symlink(m.path)]
        non_symlinks = [m for m in unique_rows if not _local_model_path_is_symlink(m.path)]
        if len(symlinks) >= 2 and not non_symlinks:
            kept.extend(unique_rows)
            continue
        winner = unique_rows[0]
        for candidate in unique_rows[1:]:
            if _prefer_local_inventory_row(candidate, winner):
                winner = candidate
        kept.append(winner)
    return kept


def _coerce_scan_folder_path(raw_path: str) -> str:
    """Normalize a scan registration target; the registry stores directories, so a pasted weight-file path is reduced to its parent folder."""
    if not raw_path or not raw_path.strip():
        raise ValueError("Path cannot be empty")
    raw = raw_path.strip()
    if "\x00" in raw:
        raise ValueError("Path cannot contain null bytes")

    def normalize(value: str) -> Path:
        return Path(os.path.realpath(os.path.expanduser(normalize_path(value))))

    try:
        normalized = normalize(raw)
    except (OSError, ValueError) as e:
        raise ValueError(f"Path is not readable: {e}") from e
    try:
        exists = normalized.exists()
        is_dir = normalized.is_dir()
        is_file = normalized.is_file()
    except (OSError, ValueError) as e:
        raise ValueError(f"Path is not readable: {e}") from e

    if not exists and "\\" in raw:
        try:
            slash_normalized = normalize(raw.replace("\\", "/"))
            slash_exists = slash_normalized.exists()
        except (OSError, ValueError) as e:
            raise ValueError(f"Path is not readable: {e}") from e
        if slash_exists:
            normalized = slash_normalized
            try:
                is_dir = normalized.is_dir()
                is_file = normalized.is_file()
            except (OSError, ValueError) as e:
                raise ValueError(f"Path is not readable: {e}") from e
            exists = True

    if not exists:
        return str(normalized)
    if is_dir:
        return str(normalized)
    if is_file:
        suffix = normalized.suffix.lower()
        if suffix not in {".gguf", ".safetensors", ".bin"}:
            raise ValueError("Path must be a folder or model weight file")
        return str(normalized.parent)
    return str(normalized)


async def _scan_source(label: str, scanner, path: Path) -> List[LocalModelInfo]:
    try:
        return await asyncio.to_thread(scanner, path)
    except Exception as e:
        logger.warning(
            "Skipping %s scan for %s: %s", label, short_path_for_log(path), scrub_paths(e)
        )
        return []


async def _collect_models_from_default_sources(
    models_root: Path,
    hf_cache_dir: Path,
    legacy_hf: Path,
    hf_default: Path,
    lm_dirs: tuple[Path, ...],
    ollama_dirs: tuple[Path, ...],
    hermes_dirs: tuple[Path, ...],
    known_hf_caches: tuple[Path, ...],
    custom_folders: list[dict],
    omlx_dirs: tuple[Path, ...] = (),
) -> List[LocalModelInfo]:
    local_models = await _scan_source("models directory", _scan_models_dir, models_root)
    hf_sources = [("HF cache", hf_cache_dir, True)]

    if _safe_is_dir(legacy_hf) and legacy_hf.resolve() != hf_cache_dir.resolve():
        hf_sources.append(("legacy HF cache", legacy_hf, False))

    if (
        _safe_is_dir(hf_default)
        and hf_default.resolve() != hf_cache_dir.resolve()
        and hf_default.resolve() != legacy_hf.resolve()
    ):
        hf_sources.append(("default HF cache", hf_default, False))

    seen_hf = {
        os.path.normcase(str(path.resolve(strict = False)))
        for path in (hf_cache_dir, legacy_hf, hf_default)
    }
    for previous_cache in known_hf_caches:
        key = os.path.normcase(str(previous_cache.resolve(strict = False)))
        if key in seen_hf:
            continue
        seen_hf.add(key)
        hf_sources.append(("previous HF cache", previous_cache, False))
    # oMLX also serves models--* repos kept under its own roots.
    for omlx_dir in omlx_dirs:
        key = os.path.normcase(str(omlx_dir.resolve(strict = False)))
        if key not in seen_hf:
            seen_hf.add(key)
            hf_sources.append(("oMLX HF cache", omlx_dir, False))

    discovered_sources = []
    custom_sources = []
    state_repositories = []
    for label, cache_dir, active_cache in hf_sources:
        discovered = await _scan_source(label, _discover_hf_cache, cache_dir)
        discovered_sources.append((label, cache_dir, active_cache, discovered))
        state_repositories.extend(
            ("model", model_id, cache_dir) for _repo, model_id, _updated in discovered
        )
    for folder in custom_folders:
        folder_path = Path(normalize_path(folder["path"])).expanduser()
        hf_caches = []
        for cache_dir in await asyncio.to_thread(hf_cache_scan.scan_folder_hf_caches, folder_path):
            discovered = await _scan_source(
                "custom HF cache",
                lambda path: _discover_hf_cache(path, entry_limit = _MAX_CUSTOM_FOLDER_ENTRIES),
                cache_dir,
            )
            hf_caches.append((cache_dir, discovered))
            state_repositories.extend(
                ("model", model_id, cache_dir) for _repo, model_id, _updated in discovered
            )
        # Carry the registered path: the status registry is keyed on it, not the normalized Path.
        custom_sources.append(
            (folder_path, hf_caches, str(folder["path"]), bool(folder.get("recursive")))
        )
    try:
        variant_states = await asyncio.to_thread(
            download_manifest.build_variant_state_index,
            state_repositories,
            active_hub_cache = hf_cache_dir,
        )
    except Exception as e:
        logger.warning("Could not build shared Hub-state index: %s", scrub_paths(e))
        variant_states = None
    for label, cache_dir, active_cache, discovered in discovered_sources:
        local_models += await _scan_source(
            label,
            lambda path, rows = discovered, active = active_cache: _scan_hf_cache(
                path,
                active_cache = active,
                discovered = rows,
                variant_states = variant_states,
                active_hub_cache = hf_cache_dir,
            ),
            cache_dir,
        )

    for lm_dir in lm_dirs:
        local_models += await _scan_source("LM Studio", _scan_lmstudio_dir, lm_dir)

    for omlx_dir in omlx_dirs:
        local_models += await _scan_source(
            "oMLX",
            lambda path: _scan_lmstudio_dir(path, source = "omlx"),
            omlx_dir,
        )

    for ollama_dir in ollama_dirs:
        local_models += await _scan_source("Ollama", scan_ollama_dir, ollama_dir)

    for hermes_dir in hermes_dirs:
        local_models += await _scan_source("Hermes", scan_hermes_dir, hermes_dir)

    hermes_identities = {_inventory_physical_identity(str(d)) for d in hermes_dirs}
    for folder_path, hf_caches, row_path, recursive in custom_sources:
        try:
            nested_roots = (
                tuple(await asyncio.to_thread(nested_scan_roots, folder_path)) if recursive else ()
            )
            custom_models = await asyncio.to_thread(
                _scan_custom_folder,
                folder_path,
                hf_caches = hf_caches,
                variant_states = variant_states,
                active_hub_cache = hf_cache_dir,
                nested_roots = nested_roots,
            )
            if _inventory_physical_identity(str(folder_path)) in hermes_identities:
                # Hermes downloads are scanned separately; skip them in a registered ~/.hermes/models.
                staged = {
                    _inventory_physical_identity(row.path)
                    for row in await asyncio.to_thread(scan_hermes_dir, folder_path)
                }
                custom_models = [
                    model
                    for model in custom_models
                    if _inventory_physical_identity(model.path) not in staged
                ]
        except Exception as e:
            logger.warning(
                "Skipping unreadable scan folder %s: %s",
                short_path_for_log(folder_path),
                scrub_paths(e),
            )
            if isinstance(e, OSError):
                record_scan_failure(row_path, e)
            continue
        # Off the loop: scandir on a stalled network mount blocks in the kernel.
        await asyncio.to_thread(note_scan_folder_scanned, row_path, found = bool(custom_models))
        for model in custom_models:
            row = _promote_to_custom_source(model)
            row._scan_root = str(folder_path)
            local_models.append(row)

    return local_models


def _scan_row_key(row: LocalModelInfo) -> tuple[str, str, Optional[str]]:
    return (row.path, row.model_format, row.format_variant)


def _scan_nested_roots(
    nested_roots: tuple[Path, ...],
    *,
    seen: set[tuple[str, str, Optional[str]]],
    active_hub_cache: Optional[Path],
) -> List[LocalModelInfo]:
    """Rows under a recursive folder's nested roots, minus any the folder's own scan already listed."""
    found: List[LocalModelInfo] = []
    for root in nested_roots:
        if len(found) >= _MAX_MODELS_PER_CUSTOM_FOLDER:
            break
        rows = _scan_models_dir(
            root,
            limit = _MAX_MODELS_PER_CUSTOM_FOLDER - len(found),
            entry_limit = _MAX_CUSTOM_FOLDER_ENTRIES,
        ) + _scan_hf_cache(
            root,
            entry_limit = _MAX_CUSTOM_FOLDER_ENTRIES,
            active_cache = False,
            active_hub_cache = active_hub_cache,
        )
        for row in rows:
            key = _scan_row_key(row)
            if key not in seen:
                seen.add(key)
                found.append(row)
    return found


def _scan_custom_folder(
    folder_path: Path,
    *,
    hf_caches: Optional[list[tuple[Path, Optional[list]]]] = None,
    variant_states: Optional[download_manifest.VariantStateIndex] = None,
    active_hub_cache: Optional[Path] = None,
    nested_roots: tuple[Path, ...] = (),
) -> List[LocalModelInfo]:
    from utils.models.model_config import detect_gguf_model

    supported_formats: set[ModelFormat] = {"gguf", "safetensors", "adapter"}
    if hf_caches is None:
        hf_caches = [(path, None) for path in hf_cache_scan.scan_folder_hf_caches(folder_path)]

    def _is_supported(m: LocalModelInfo) -> bool:
        # Diffusers pipeline roots are 'unknown' format; judge by shape.
        if m.model_format in supported_formats:
            return True
        return _is_diffusers_pipeline_dir(Path(m.path))

    generic = [
        m
        for m in (
            _scan_models_dir(
                folder_path,
                limit = _MAX_MODELS_PER_CUSTOM_FOLDER,
                entry_limit = _MAX_CUSTOM_FOLDER_ENTRIES,
            )
            + [
                row
                for cache_dir, discovered in hf_caches
                for row in _scan_hf_cache(
                    cache_dir,
                    entry_limit = _MAX_CUSTOM_FOLDER_ENTRIES,
                    active_cache = False,
                    discovered = discovered,
                    variant_states = variant_states,
                    active_hub_cache = active_hub_cache,
                )
            ]
            + _scan_lmstudio_dir(folder_path, entry_limit = _MAX_CUSTOM_FOLDER_ENTRIES)
        )
        if _is_supported(m)
        if not any(p in (".studio_links", "ollama_links") for p in Path(m.path).parts)
    ]
    if nested_roots:
        generic += [
            m
            for m in _scan_nested_roots(
                nested_roots,
                seen = {_scan_row_key(m) for m in generic},
                active_hub_cache = active_hub_cache,
            )
            if _is_supported(m)
        ]
    selectable = []
    for model in generic:
        if model.model_format != "gguf" or model.partial:
            selectable.append(model)
            continue
        path = Path(model.path)
        if path.is_dir():
            if any(
                detect_gguf_model(str(file), model_root = str(folder_path)) is not None
                for file in path.glob("*")
                if not _safe_is_dir(file) and file.suffix.lower() == ".gguf"
            ):
                selectable.append(model)
        elif detect_gguf_model(model.path, model_root = str(folder_path)) is not None:
            selectable.append(model)

    selectable = gguf.dedupe_custom_gguf_rows(selectable)
    remaining = _MAX_MODELS_PER_CUSTOM_FOLDER - len(selectable)
    if remaining > 0:
        selectable.extend(scan_ollama_dir(folder_path, limit = remaining))
    return selectable[:_MAX_MODELS_PER_CUSTOM_FOLDER]


def _promote_to_custom_source(model: LocalModelInfo) -> LocalModelInfo:
    if model.source in {"hf_cache", "ollama", "hermes"}:
        return model
    return model.model_copy(
        update = {
            "source": "custom",
            "model_id": None,
            "inventory_id": _local_inventory_id(
                "custom",
                model.model_format,
                model.path,
                model.format_variant,
            ),
            "capabilities": _capabilities_for_format(
                model.model_format,
                "custom",
                partial = model.partial,
                requires_variant = model.capabilities.requires_variant,
                # Carry the old can_chat verdict; rebuilding from format alone restored it wrongly.
                can_chat_override = model.capabilities.can_chat,
            ),
        }
    )


async def _load_custom_folders() -> list[dict]:
    try:
        return await asyncio.to_thread(list_scan_folders)
    except Exception as e:
        logger.warning("Could not load custom scan folders: %s", scrub_paths(e))
        return []


def _merge_custom_rows_listed_natively(
    custom_models: List[LocalModelInfo], native_models: List[LocalModelInfo]
) -> tuple[list[LocalModelInfo], list[LocalModelInfo]]:
    """A custom folder overlapping the models dir, LM Studio or oMLX re-lists their models (#9164)."""
    native: dict[tuple[str, str], LocalModelInfo] = {}
    for model in native_models:
        if model.source in ("models_dir", "lmstudio", "omlx"):
            native.setdefault((_inventory_physical_identity(model.path), model.model_format), model)
    if not native:
        return list(native_models), list(custom_models)
    replaced: set[int] = set()
    kept_custom: list[LocalModelInfo] = []
    for model in custom_models:
        twin = native.get((_inventory_physical_identity(model.path), model.model_format))
        # A symlink below the scan root is a deliberate alias (#10605).
        if twin is None or _local_model_path_is_symlink(_custom_alias_key(model)):
            kept_custom.append(model)
        elif twin.source in ("lmstudio", "omlx") and model.capabilities.can_train:
            # The train picker refuses LM Studio and oMLX rows, so the trainable custom row wins.
            replaced.add(id(twin))
            kept_custom.append(model)
    return [m for m in native_models if id(m) not in replaced], kept_custom


def _dedupe_local_models(local_models: List[LocalModelInfo]) -> list[LocalModelInfo]:
    deduped: dict[str, LocalModelInfo] = {}
    custom_models: list[LocalModelInfo] = []
    for model in local_models:
        if model.source == "custom":
            custom_models.append(model)
            continue
        if model.source == "hf_cache" and model.model_id:
            key = "\x00".join(
                (
                    "hf_cache",
                    model.model_id.strip().lower(),
                    model.model_format,
                    model.format_variant or "",
                )
            )
        else:
            row_key = model.inventory_id or model.id
            key = row_key
        existing = deduped.get(key)
        prefer_candidate = existing is None
        if existing is not None:
            if model.partial != existing.partial:
                prefer_candidate = not model.partial
            elif (model.active_cache is True) != (existing.active_cache is True):
                prefer_candidate = model.active_cache is True
            else:
                prefer_candidate = _prefer_complete_larger(
                    model.partial,
                    model.size_bytes,
                    existing.partial,
                    existing.size_bytes,
                )
        if prefer_candidate:
            deduped[key] = model

    native_values, custom_values = _merge_custom_rows_listed_natively(
        _dedupe_custom_local_models(custom_models), list(deduped.values())
    )
    return sorted(
        native_values + gguf.suppress_grouped_gguf_file_rows(custom_values),
        key = lambda item: item.updated_at or 0,
        reverse = True,
    )


def _filter_hidden_models(local_models: List[LocalModelInfo]) -> list[LocalModelInfo]:
    """Remove infrastructure-only models from the shared local inventory."""
    visible: list[LocalModelInfo] = []
    for model in local_models:
        resolved_cache_path = (
            hf_cache_scan.resolve_hf_cache_realpath(Path(model.path))
            if model.source == "hf_cache"
            else None
        )
        if not is_hidden_model(model.id, model.model_id, model.path, resolved_cache_path):
            visible.append(model)
    return visible


def _filter_and_dedupe_local_models(local_models: List[LocalModelInfo]) -> list[LocalModelInfo]:
    return _dedupe_local_models(_filter_hidden_models(local_models))


async def _scan_local_models_response(
    models_dir: str, custom_folders: list[dict], sources: _LocalInventorySources
) -> LocalModelListResponse:
    """List local model candidates from every supported on-device source."""
    (
        hf_cache_dir,
        legacy_hf,
        hf_default,
        lm_dirs,
        ollama_dirs,
        hermes_dirs,
        known_hf_caches,
        omlx_dirs,
    ) = sources

    allowed_roots: list[Path] = [Path("./models").resolve(), hf_cache_dir]
    if _safe_is_dir(legacy_hf):
        allowed_roots.append(legacy_hf)
    if _safe_is_dir(hf_default):
        allowed_roots.append(hf_default)
    allowed_roots.extend([studio_root(), outputs_root()])

    try:
        models_root = _resolve_allowed_models_dir(models_dir, allowed_roots)
    except ValueError:
        raise _http_error(status_code = 403, detail = "Directory not allowed")

    try:
        local_models = await _collect_models_from_default_sources(
            models_root,
            hf_cache_dir,
            legacy_hf,
            hf_default,
            lm_dirs,
            ollama_dirs,
            hermes_dirs,
            known_hf_caches,
            custom_folders,
            omlx_dirs = omlx_dirs,
        )
        models = await asyncio.to_thread(_filter_and_dedupe_local_models, local_models)
        return LocalModelListResponse(
            models_dir = str(models_root),
            hf_cache_dir = str(hf_cache_dir),
            lmstudio_dirs = [str(d) for d in lm_dirs],
            ollama_dirs = [str(d) for d in ollama_dirs],
            hermes_dirs = [str(d) for d in hermes_dirs],
            models = models,
        )
    except Exception as e:
        logger.error("Error listing local models: %s", scrub_paths(e), exc_info = True)
        raise _http_error(
            status_code = 500,
            detail = f"Failed to list local models: {str(e)}",
        )


async def _account_local_response(response):
    if not _account_access().managed_account():
        return response
    models = await asyncio.to_thread(_account_access().filter_model_rows, response.models)
    return response.model_copy(update = {"models": models})


async def list_local_models_response(models_dir: str = "./models") -> LocalModelListResponse:
    """Coalesce overlapping local inventory requests for the same models root."""

    def classify(response: LocalModelListResponse) -> LocalModelListResponse:
        try:
            from hub.services.models import catalog_classification

            models = []
            for model in response.models:
                task, audio_type = catalog_classification._local_model_classification(model)
                workflows = catalog_classification.local_audio_workflows(model, audio_type)
                models.append(
                    model.model_copy(
                        update = {
                            "task": task,
                            "audio_type": audio_type,
                            **({"audio_workflows": workflows} if workflows else {}),
                        }
                    )
                )
            return response.model_copy(update = {"models": models})
        except Exception as e:  # noqa: BLE001 -- classification never breaks the listing
            logger.warning("Could not classify local model tasks: %s", scrub_paths(e))
            return response

    async def scan_and_classify(
        expected_epoch: int, custom_folders: list[dict], sources: _LocalInventorySources
    ) -> LocalModelListResponse:
        response = await _scan_local_models_response(models_dir, custom_folders, sources)
        if hf_cache_scan.hf_cache_scans_epoch() != expected_epoch:
            raise _LocalCacheChanged(response)
        classified = await asyncio.to_thread(classify, response)
        # That hop is an await point; a mutation can land after the check.
        if hf_cache_scan.hf_cache_scans_epoch() != expected_epoch:
            raise _LocalCacheChanged(response)
        return classified

    superseded: Optional[LocalModelListResponse] = None
    for _attempt in range(_LOCAL_INVENTORY_MAX_ATTEMPTS):
        # Epoch first so any later change to sources lands in a later epoch.
        epoch = hf_cache_scan.hf_cache_scans_epoch()
        custom_folders = await _load_custom_folders()
        sources = _local_inventory_sources()
        key: _LocalInventoryKey = (
            _inventory_path_identity(models_dir),
            sources,
            tuple(
                _inventory_path_identity(str(folder.get("path", "")))
                + ("\x00r" if folder.get("recursive") else "")
                for folder in custom_folders
            ),
            epoch,
        )
        try:
            response = await hf_cache_scan.shared_scan(
                _local_inventory_flights,
                key,
                lambda expected_epoch = epoch, folders = custom_folders, roots = sources: (
                    scan_and_classify(expected_epoch, folders, roots)
                ),
            )
            return await _account_local_response(response)
        except _LocalCacheChanged as changed:
            superseded = changed.response
            continue
    logger.warning("Local inventory kept racing cache invalidations; serving the last scan")
    return await _account_local_response(await asyncio.to_thread(classify, superseded))


def get_models_folder_response() -> dict:
    """The directory where downloaded models are stored: the active HF hub cache (honors ``HF_HOME`` / ``HF_HUB_CACHE``), which the desktop app reveals in the OS file manager."""
    path = _resolve_hf_cache_dir()
    # Create so 'Open folder' works before the first download (HF builds the cache lazily).
    try:
        path.mkdir(parents = True, exist_ok = True)
    except OSError as e:
        raise _http_error(
            status_code = 500,
            detail = f"Failed to create models folder: {path}: {e}",
        ) from e
    if not path.is_dir():
        raise _http_error(
            status_code = 500,
            detail = f"Models folder path is not a directory: {path}",
        )
    return {"path": str(path)}


def get_scan_folders_response() -> dict:
    folders = list_scan_folders()
    refresh_failed_scan_folders(folders)
    return {"folders": annotate_scan_folders(folders)}


def add_scan_folder_response(path: str, recursive: Optional[bool] = None) -> dict:
    path = _account_access().private_directory(path, "")
    try:
        folder, inserted = add_scan_folder_with_status(_coerce_scan_folder_path(path), recursive)
    except ValueError as e:
        logger.warning(
            "Scan folder rejected: %s (path=%s)", scrub_paths(e), short_path_for_log(path)
        )
        raise _http_error(status_code = 400, detail = str(e))
    logger.info("Scan folder added: %s", short_path_for_log(folder.get("path")))
    if inserted:
        from core.inference.local_model_resolver import invalidate_index, warm_index_soon
        invalidate_index()
        warm_index_soon()
    return folder


def remove_scan_folder_response(folder_id: int) -> dict:
    removed = remove_scan_folder(folder_id)
    if removed:
        logger.info("Scan folder removed: id=%s", folder_id)
        from core.inference.local_model_resolver import invalidate_index, warm_index_soon

        invalidate_index()
        warm_index_soon()
    return {"ok": True}
