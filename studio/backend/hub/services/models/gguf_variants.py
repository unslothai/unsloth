# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF variant resolution."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple, Optional

from fastapi import HTTPException
from loggers import get_logger

from hub.schemas.inventory import GgufVariantDetail, GgufVariantsResponse
from hub.utils import download_manifest
from hub.utils import download_registry
from hub.utils import inventory_scan as hf_cache_scan
from hub.utils.hf_errors import hf_error_status
from hub.utils.hf_cache_state import (
    INCOMPLETE_SUFFIX,
    iter_destructive_repo_cache_dirs,
    repo_cache_dir_name,
)
from hub.utils.gguf import (
    extract_quant_label,
    iter_hf_cache_snapshots,
    is_big_endian_gguf_path,
    list_empty_gguf_variant_dirs,
    list_gguf_variants,
    list_gguf_variants_from_hf_cache,
    list_local_gguf_variants,
    list_partial_gguf_variants_from_state,
    pick_best_gguf,
)
from hub.utils.paths import (
    is_local_path,
    is_valid_repo_id as _is_valid_repo_id,
)
from hub.services.models.common import (
    _is_mmproj_filename,
    _is_mtp_drafter_path,
    _iter_gguf_paths,
)
from hub.utils.gguf_plan import (
    GgufVariantPlan as _GgufVariantRequirement,
    build_gguf_variant_plans,
    is_main_gguf_variant_path,
)

logger = get_logger(__name__)

_VARIANT_HASH_CACHE: "OrderedDict[tuple[str, str, str, bool], tuple[frozenset[str], float]]" = (
    OrderedDict()
)
_VARIANT_REQUIREMENT_CACHE: "OrderedDict[tuple[str, str, str], tuple[_GgufVariantRequirement, float]]" = OrderedDict()
_VARIANT_REQUIREMENT_NEG_CACHE: "OrderedDict[tuple[str, str], float]" = OrderedDict()
_VARIANT_HASH_MAX = 512
# Blob hashes are derived from the same mutable remote revision metadata as
# variant requirements, so they must not outlive that freshness window.
_VARIANT_HASH_POS_TTL = 60.0
# Refresh resolved variant requirements so a moved repo revision is picked up
# within the session instead of being pinned for the backend's lifetime.
_VARIANT_REQUIREMENT_POS_TTL = 60.0
# Suppress retries on a metadata-fetch failure so a slow/flaky link doesn't
# re-hammer the API on every page refresh.
_VARIANT_REQUIREMENT_NEG_TTL = 60.0
# Fail fast on a slow link so the variant render isn't blocked for seconds.
_GGUF_METADATA_TIMEOUT_SECONDS = 5.0
_VARIANT_HASH_LOCK = threading.Lock()


class VariantIncompleteDeleteResult(NamedTuple):
    deleted: int
    unresolved: bool


def _variant_hash_cache_key(
    repo_id: str, variant: str, hf_token: Optional[str]
) -> tuple[str, str, str]:
    return (
        repo_id.lower(),
        variant.lower(),
        hf_cache_scan.token_fingerprint(hf_token),
    )


def _variant_blob_hash_cache_key(
    repo_id: str, variant: str, hf_token: Optional[str], include_companions: bool
) -> tuple[str, str, str, bool]:
    base = _variant_hash_cache_key(repo_id, variant, hf_token)
    return (*base, include_companions)


def _variant_repo_cache_key(repo_id: str, hf_token: Optional[str]) -> tuple[str, str]:
    return (repo_id.lower(), hf_cache_scan.token_fingerprint(hf_token))


def _variant_requirement_neg_cache_active(key: tuple[str, str]) -> bool:
    with _VARIANT_HASH_LOCK:
        cached_at = _VARIANT_REQUIREMENT_NEG_CACHE.get(key)
        if cached_at is None:
            return False
        if (time.monotonic() - cached_at) < _VARIANT_REQUIREMENT_NEG_TTL:
            _VARIANT_REQUIREMENT_NEG_CACHE.move_to_end(key)
            return True
        _VARIANT_REQUIREMENT_NEG_CACHE.pop(key, None)
        return False


def _variant_requirement_neg_cache_set(key: tuple[str, str]) -> None:
    with _VARIANT_HASH_LOCK:
        _VARIANT_REQUIREMENT_NEG_CACHE[key] = time.monotonic()
        _VARIANT_REQUIREMENT_NEG_CACHE.move_to_end(key)
        while len(_VARIANT_REQUIREMENT_NEG_CACHE) > _VARIANT_HASH_MAX:
            _VARIANT_REQUIREMENT_NEG_CACHE.popitem(last = False)


def _variant_requirement_neg_cache_clear(key: tuple[str, str]) -> None:
    with _VARIANT_HASH_LOCK:
        _VARIANT_REQUIREMENT_NEG_CACHE.pop(key, None)


def _variant_hash_cache_get(key: tuple[str, str, str, bool]) -> Optional[frozenset[str]]:
    with _VARIANT_HASH_LOCK:
        cached = _VARIANT_HASH_CACHE.get(key)
        if cached is None:
            return None
        hashes, ts = cached
        if (time.monotonic() - ts) >= _VARIANT_HASH_POS_TTL:
            _VARIANT_HASH_CACHE.pop(key, None)
            return None
        _VARIANT_HASH_CACHE.move_to_end(key)
        return hashes


def _variant_hash_cache_set(key: tuple[str, str, str, bool], hashes: frozenset[str]) -> None:
    with _VARIANT_HASH_LOCK:
        _VARIANT_HASH_CACHE[key] = (hashes, time.monotonic())
        _VARIANT_HASH_CACHE.move_to_end(key)
        while len(_VARIANT_HASH_CACHE) > _VARIANT_HASH_MAX:
            _VARIANT_HASH_CACHE.popitem(last = False)


def _variant_requirement_cache_get(key: tuple[str, str, str]) -> Optional[_GgufVariantRequirement]:
    with _VARIANT_HASH_LOCK:
        cached = _VARIANT_REQUIREMENT_CACHE.get(key)
        if cached is None:
            return None
        requirement, ts = cached
        if (time.monotonic() - ts) >= _VARIANT_REQUIREMENT_POS_TTL:
            _VARIANT_REQUIREMENT_CACHE.pop(key, None)
            return None
        _VARIANT_REQUIREMENT_CACHE.move_to_end(key)
        return requirement


def _variant_requirement_cache_set_many(
    repo_id: str, hf_token: Optional[str], requirements: dict[str, _GgufVariantRequirement]
) -> None:
    with _VARIANT_HASH_LOCK:
        now = time.monotonic()
        for quant, requirement in requirements.items():
            key = _variant_hash_cache_key(repo_id, quant, hf_token)
            _VARIANT_REQUIREMENT_CACHE[key] = (requirement, now)
            _VARIANT_REQUIREMENT_CACHE.move_to_end(key)
        while len(_VARIANT_REQUIREMENT_CACHE) > _VARIANT_HASH_MAX:
            _VARIANT_REQUIREMENT_CACHE.popitem(last = False)


def _build_gguf_variant_requirements(siblings: list) -> dict[str, _GgufVariantRequirement]:
    return build_gguf_variant_plans(siblings)


def gguf_variant_requirements(
    repo_id: str,
    variant: str,
    hf_token: Optional[str] = None,
) -> Optional[_GgufVariantRequirement]:
    key = _variant_hash_cache_key(repo_id, variant, hf_token)
    cached = _variant_requirement_cache_get(key)
    if cached is not None:
        return cached
    requirements = _fetch_gguf_variant_requirements(repo_id, hf_token)
    return requirements.get(variant.lower())


def _fetch_gguf_variant_requirements(
    repo_id: str,
    hf_token: Optional[str] = None,
    *,
    siblings: Optional[list] = None,
) -> dict[str, _GgufVariantRequirement]:
    repo_key = _variant_repo_cache_key(repo_id, hf_token)
    if siblings is None:
        if _variant_requirement_neg_cache_active(repo_key):
            return {}
        try:
            from huggingface_hub import HfApi
            info = HfApi(token = hf_token).model_info(
                repo_id,
                files_metadata = True,
                timeout = _GGUF_METADATA_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.warning(
                "model_info failed resolving GGUF files for %s: %s",
                repo_id,
                download_registry.scrub_secrets(str(e), hf_token = hf_token),
            )
            _variant_requirement_neg_cache_set(repo_key)
            return {}
        siblings = list(info.siblings)
    requirements = _build_gguf_variant_requirements(siblings)
    if requirements:
        _variant_requirement_cache_set_many(repo_id, hf_token, requirements)
    _variant_requirement_neg_cache_clear(repo_key)
    return requirements


def _gguf_all_variant_requirements(
    repo_id: str,
    hf_token: Optional[str] = None,
    *,
    siblings: Optional[list] = None,
) -> dict[str, _GgufVariantRequirement]:
    return _fetch_gguf_variant_requirements(repo_id, hf_token, siblings = siblings)


def _manifest_variant_blob_hashes(
    repo_id: str,
    variant: str,
    *,
    include_companions: bool = True,
    repo_cache_dir: Optional[Path] = None,
) -> frozenset[str]:
    manifest = download_manifest.read_manifest(
        "model",
        repo_id,
        variant,
        hub_cache = repo_cache_dir.parent if repo_cache_dir is not None else None,
    )
    if manifest is None:
        return frozenset()
    variant_key = variant.lower()
    hashes: set[str] = set()
    for expected in manifest.expected_files:
        if not expected.sha256:
            continue
        if include_companions:
            hashes.add(expected.sha256)
            continue
        if is_main_gguf_variant_path(expected.path, variant_key):
            hashes.add(expected.sha256)
    return frozenset(hashes)


def gguf_variant_blob_hashes(
    repo_id: str,
    variant: str,
    hf_token: Optional[str] = None,
    *,
    include_companions: bool = True,
    allow_remote: bool = True,
    repo_cache_dir: Optional[Path] = None,
) -> frozenset[str]:
    key = _variant_blob_hash_cache_key(
        repo_id,
        variant,
        hf_token,
        include_companions,
    )
    cached = _variant_hash_cache_get(key)
    if cached is not None:
        return cached
    hashes = _manifest_variant_blob_hashes(
        repo_id,
        variant,
        include_companions = include_companions,
        repo_cache_dir = repo_cache_dir,
    )
    if hashes:
        return hashes
    requirement_key = _variant_hash_cache_key(repo_id, variant, hf_token)
    requirement = _variant_requirement_cache_get(requirement_key)
    if requirement is None and allow_remote:
        requirement = gguf_variant_requirements(repo_id, variant, hf_token)
    if requirement is not None:
        hashes = requirement.required_hashes if include_companions else requirement.main_hashes
        if hashes:
            _variant_hash_cache_set(key, hashes)
        return hashes
    return frozenset()


def _partial_transport_for_variant(
    repo_id: str,
    variant: str,
    repo_cache_dir: Optional[Path] = None,
) -> Optional[str]:
    return hf_cache_scan.partial_transport_for(
        "model",
        repo_id,
        variant,
        repo_cache_dir,
    )


def _local_main_gguf_blobs_by_quant(
    repo_id: str, repo_cache_dir: Optional[Path] = None
) -> dict[str, dict[str, set[str]]]:
    """Map quant -> repo-relative expected GGUF filename -> cached blob hashes.

    Shared companions are copied into each main-quant bucket so update checks can
    detect mmproj/MTP-only upstream changes without a separate remote call.
    """
    result: dict[str, dict[str, set[str]]] = {}
    companion_blobs: dict[str, set[str]] = {}
    try:
        from hub.services.models import cache_inventory
        scans = cache_inventory.all_hf_cache_scans()
    except Exception as e:
        logger.warning("Failed to scan local GGUF blobs for %s: %s", repo_id, e)
        return result

    target_lower = repo_id.lower()
    for hf_cache in scans:
        for repo_info in hf_cache.repos:
            if str(getattr(repo_info, "repo_type", "")) != "model":
                continue
            if str(getattr(repo_info, "repo_id", "")).lower() != target_lower:
                continue
            if repo_cache_dir is not None:
                try:
                    if Path(repo_info.repo_path).resolve(strict = False) != repo_cache_dir.resolve(
                        strict = False
                    ):
                        continue
                except (AttributeError, OSError, RuntimeError, ValueError):
                    continue
            for path, hashes in cache_inventory._repo_gguf_blob_map(
                repo_info,
                include_companions = True,
            ).items():
                normalized = str(path).replace("\\", "/")
                if not hashes:
                    continue
                if _is_mmproj_filename(normalized) or _is_mtp_drafter_path(normalized):
                    companion_blobs.setdefault(normalized, set()).update(
                        str(blob) for blob in hashes if blob
                    )
                    continue
                quant = extract_quant_label(normalized).lower()
                if is_big_endian_gguf_path(normalized, quant):
                    continue
                bucket = result.setdefault(quant, {}).setdefault(normalized, set())
                bucket.update(str(blob) for blob in hashes if blob)
    if companion_blobs:
        for local_blobs in result.values():
            for path, hashes in companion_blobs.items():
                local_blobs.setdefault(path, set()).update(hashes)
    return result


def _size_identity_matches(local_set: set[str], remote_size: int) -> bool:
    """Whether a cached file with NO blob hash is current, judged by size.

    A size token only lands in ``local_set`` for a file the cache has no blob for,
    so it never loosens the hash comparison for a normal file. Tradeoff: an
    equal-size requant is missed, versus the status quo where every no-blob GGUF
    shows a phantom update that no re-download clears.
    """
    size = int(remote_size or 0)
    if size <= 0:
        return False
    from hub.services.models import cache_inventory

    return cache_inventory.local_size_identity(size) in local_set


def _variant_update_available_from_requirement(
    local_blobs: dict[str, set[str]], requirement: Optional[_GgufVariantRequirement], variant: str
) -> bool:
    if requirement is None or not local_blobs:
        return False
    local_by_posix = {path.replace("\\", "/"): blobs for path, blobs in local_blobs.items()}
    for expected in requirement.expected_files:
        path = str(expected.path).replace("\\", "/")
        if not (
            is_main_gguf_variant_path(path, variant)
            or _is_mmproj_filename(path)
            or _is_mtp_drafter_path(path)
        ):
            continue
        remote_blob = expected.sha256
        if not remote_blob:
            continue
        local_set = local_by_posix.get(path)
        if not local_set:
            return True
        if remote_blob in local_set:
            continue
        if _size_identity_matches(local_set, expected.size):
            continue
        return True
    return False


def delete_variant_incomplete_blobs_result(
    repo_id: str,
    variant: str,
    hf_token: Optional[str],
    *,
    extra_hashes: frozenset[str] = frozenset(),
    companions: bool = True,
    root: Optional[Path] = None,
) -> VariantIncompleteDeleteResult:
    # With a sibling still downloading, ``companions=False`` keeps a shared mmproj
    # from being unlinked out from under it; the repo's last delete reclaims it.
    target_hashes = (
        gguf_variant_blob_hashes(repo_id, variant, hf_token, include_companions = companions)
        | extra_hashes
    )
    if not target_hashes:
        has_variant_partial_state = hf_cache_scan.is_variant_partial(
            repo_id,
            variant,
            incomplete_blob_hashes = set(),
            variant_blob_hashes = frozenset(),
        )
        has_repo_partials = bool(download_registry.incomplete_blob_hashes("model", repo_id))
        return VariantIncompleteDeleteResult(
            deleted = 0,
            unresolved = has_variant_partial_state and has_repo_partials,
        )
    deleted = 0
    # Destructive iterator: only the exact-case match (or abort if ambiguous),
    # so a case-variant sibling repo's partials are never unlinked. ``root`` scopes
    # the purge to one cache so a delete never touches another cache's partials.
    for entry in iter_destructive_repo_cache_dirs("model", repo_id, root = root):
        blobs_dir = entry / "blobs"
        if not blobs_dir.is_dir():
            continue
        for h in target_hashes:
            incomplete = blobs_dir / f"{h}{INCOMPLETE_SUFFIX}"
            if incomplete.exists():
                try:
                    incomplete.unlink()
                    deleted += 1
                except OSError as e:
                    logger.warning(f"Failed to unlink {incomplete}: {e}")
    return VariantIncompleteDeleteResult(deleted = deleted, unresolved = False)


def _repo_cache_dir_for_request(repo_id: str, local_path: Optional[str]) -> Path:
    """Resolve the one Hub repo cache represented by this variant request."""
    expected_name = repo_cache_dir_name("model", repo_id).lower()
    if local_path:
        try:
            local = Path(local_path).expanduser().resolve(strict = False)
            for candidate in (local, *local.parents):
                if candidate.name.lower() == expected_name:
                    return candidate
        except (OSError, RuntimeError, ValueError):
            pass
    from utils.hf_cache_settings import get_hf_cache_paths

    return get_hf_cache_paths().hub_cache / repo_cache_dir_name("model", repo_id)


def _mark_empty_dir_cleanables(
    repo_id: str,
    response: GgufVariantsResponse,
    repo_cache_dir: Optional[Path] = None,
) -> GgufVariantsResponse:
    """Surface empty leftover ``<quant>/`` folders (interrupted downloads) as
    partial so the UI can delete them -- on local/offline paths too, not just a
    remote listing. A listed quant is flipped to partial; an unlisted one is
    appended as a zero-byte cleanable entry."""
    try:
        empty_labels = (
            list_empty_gguf_variant_dirs(repo_id, root = repo_cache_dir.parent)
            if repo_cache_dir is not None
            else list_empty_gguf_variant_dirs(repo_id)
        )
    except Exception as e:
        logger.warning(f"Failed to scan empty GGUF variant folders for {repo_id}: {e}")
        return response
    if not empty_labels:
        return response
    empty_by_key = {label.lower(): label for label in empty_labels}
    variants = list(response.variants)
    listed = {v.quant.lower() for v in variants}
    for i, v in enumerate(variants):
        if v.quant.lower() in empty_by_key and not v.downloaded and not v.partial:
            variants[i] = v.model_copy(update = {"partial": True})
    for key, label in sorted(empty_by_key.items()):
        if key not in listed:
            variants.append(GgufVariantDetail(filename = f"{label}.gguf", quant = label, partial = True))
    return response.model_copy(update = {"variants": variants})


async def get_gguf_variants_response(
    repo_id: str,
    prefer_local_cache: bool = False,
    offline: bool = False,
    local_path: Optional[str] = None,
    hf_token: Optional[str] = None,
):
    """
    List available GGUF quantization variants for a HuggingFace repo
    or a local directory (e.g. LM Studio model folder).

    Returns all available quantization variants (Q4_K_M, Q8_0, BF16, etc.)
    with file sizes, whether the model supports vision, and the recommended
    default variant.
    """

    def _compute() -> GgufVariantsResponse:
        repo_cache_dir = (
            None if is_local_path(repo_id) else _repo_cache_dir_for_request(repo_id, local_path)
        )
        hub_cache = repo_cache_dir.parent if repo_cache_dir is not None else None

        def _local_response(
            response_repo_id: str, variants, has_vision: bool
        ) -> GgufVariantsResponse:
            filenames = [v.filename for v in variants]
            best = pick_best_gguf(filenames)
            default_variant = extract_quant_label(best) if best else None
            return GgufVariantsResponse(
                repo_id = response_repo_id,
                variants = [
                    GgufVariantDetail(
                        filename = v.filename,
                        quant = v.quant,
                        display_label = v.display_label,
                        size_bytes = v.size_bytes,
                        download_size_bytes = v.size_bytes,
                        downloaded = True,
                    )
                    for v in variants
                ],
                has_vision = has_vision,
                default_variant = default_variant,
            )

        def _partial_local_response(
            response_repo_id: str, variants, has_vision: bool
        ) -> GgufVariantsResponse:
            filenames = [v.filename for v in variants]
            best = pick_best_gguf(filenames)
            default_variant = extract_quant_label(best) if best else None
            return GgufVariantsResponse(
                repo_id = response_repo_id,
                variants = [
                    GgufVariantDetail(
                        filename = v.filename,
                        quant = v.quant,
                        display_label = v.display_label,
                        size_bytes = v.size_bytes,
                        download_size_bytes = v.download_size_bytes or v.size_bytes,
                        downloaded = False,
                        partial = True,
                        partial_transport = _partial_transport_for_variant(
                            response_repo_id,
                            v.quant,
                            repo_cache_dir,
                        ),
                    )
                    for v in variants
                ],
                has_vision = has_vision,
                default_variant = default_variant,
            )

        # Local directory path (e.g. LM Studio models) — scan filesystem
        if is_local_path(repo_id):
            variants, has_vision = list_local_gguf_variants(repo_id)

            return _local_response(repo_id, variants, has_vision)

        # Reject invalid remote repo_ids up front (like download/delete) so a
        # malformed id returns 400 instead of a 500 from the HF client.
        if not _is_valid_repo_id(repo_id):
            raise HTTPException(status_code = 400, detail = f"Invalid repo_id: {repo_id!r}")

        local_only = prefer_local_cache or offline
        if local_only:
            cached = list_gguf_variants_from_hf_cache(
                repo_id,
                hf_token,
                offline = local_only,
                root = hub_cache,
            )
            if cached is not None:
                variants, has_vision = cached
                return _local_response(repo_id, variants, has_vision)
            if local_path and is_local_path(local_path):
                variants, has_vision = list_local_gguf_variants(local_path)
                if variants or has_vision:
                    return _local_response(repo_id, variants, has_vision)
            partial = list_partial_gguf_variants_from_state(repo_id, hub_cache = hub_cache)
            if partial is not None:
                variants, has_vision = partial
                return _partial_local_response(repo_id, variants, has_vision)
            if local_path and offline:
                return GgufVariantsResponse(
                    repo_id = repo_id,
                    variants = [],
                    has_vision = False,
                    default_variant = None,
                )
            if offline:
                raise HTTPException(
                    status_code = 404,
                    detail = "No cached GGUF variants available while offline.",
                )

        try:
            variants, has_vision, siblings = list_gguf_variants(repo_id, hf_token = hf_token)
        except Exception:
            cached = list_gguf_variants_from_hf_cache(
                repo_id,
                hf_token,
                root = hub_cache,
            )
            if cached is not None:
                variants, has_vision = cached
                return _local_response(repo_id, variants, has_vision)
            partial = list_partial_gguf_variants_from_state(repo_id, hub_cache = hub_cache)
            if partial is not None:
                variants, has_vision = partial
                return _partial_local_response(repo_id, variants, has_vision)
            raise

        filenames = [v.filename for v in variants]
        best = pick_best_gguf(filenames)
        default_variant = extract_quant_label(best) if best else None

        # Per-snapshot accounting: a variant counts as present only when one
        # snapshot holds all its files (split GGUFs need every shard together),
        # sizes are max across snapshots so shared blobs aren't double-counted,
        # and keys are lowercased since cache dir casing can differ from repo_id.
        cached_filenames_by_snapshot: list[dict[str, int]] = []
        cached_quant_bytes_by_snapshot: list[dict[str, int]] = []
        if _is_valid_repo_id(repo_id):
            for snap in iter_hf_cache_snapshots(repo_id, root = hub_cache):
                try:
                    gguf_paths = list(_iter_gguf_paths(snap))
                except (OSError, RuntimeError, ValueError) as e:
                    logger.debug("Skipping GGUF cache snapshot %s: %s", snap, e)
                    continue
                by_filename: dict[str, int] = {}
                by_quant: dict[str, int] = {}
                for f in gguf_paths:
                    try:
                        rel = f.relative_to(snap).as_posix()
                        size = f.stat().st_size
                    except (OSError, RuntimeError, ValueError) as e:
                        logger.debug("Skipping GGUF cache file %s: %s", f, e)
                        continue
                    key = rel.lower()
                    by_filename[key] = max(by_filename.get(key, 0), size)
                    if _is_mmproj_filename(f.name) or _is_mtp_drafter_path(rel):
                        continue
                    q = extract_quant_label(rel)
                    if is_big_endian_gguf_path(rel, q):
                        continue
                    q = q.lower()
                    by_quant[q] = by_quant.get(q, 0) + size
                if by_filename:
                    cached_filenames_by_snapshot.append(by_filename)
                if by_quant:
                    cached_quant_bytes_by_snapshot.append(by_quant)

        requirements_by_quant = {
            v.quant.lower(): _variant_requirement_cache_get(
                _variant_hash_cache_key(repo_id, v.quant, hf_token)
            )
            for v in variants
        }
        if any(req is None for req in requirements_by_quant.values()):
            fetched_requirements = _gguf_all_variant_requirements(
                repo_id, hf_token, siblings = siblings
            )
            for v in variants:
                key = v.quant.lower()
                if requirements_by_quant.get(key) is None:
                    requirements_by_quant[key] = fetched_requirements.get(key)

        def _filenames_cached(filenames: frozenset[str], expected_size: int) -> bool:
            if not filenames:
                return False
            wanted = [name.lower() for name in filenames]
            # All files must live in a single snapshot, not spread across several.
            for by_filename in cached_filenames_by_snapshot:
                cached = 0
                for name in wanted:
                    size = by_filename.get(name)
                    if size is None:
                        break
                    cached += size
                else:
                    return expected_size <= 0 or cached >= expected_size * 0.99
            return False

        def _any_mmproj_cached(filenames: frozenset[str]) -> bool:
            if any(
                by_filename.get(name.lower()) is not None
                for by_filename in cached_filenames_by_snapshot
                for name in filenames
            ):
                return True
            return any(
                _is_mmproj_filename(name.rsplit("/", 1)[-1])
                for by_filename in cached_filenames_by_snapshot
                for name in by_filename
            )

        def _quant_bytes_present(quant: str, size_bytes: int) -> bool:
            # Small rounding tolerance for symlinks vs real sizes.
            if size_bytes <= 0:
                return False
            return any(
                by_quant.get(quant, 0) >= size_bytes * 0.99
                for by_quant in cached_quant_bytes_by_snapshot
            )

        def _is_fully_downloaded(variant) -> bool:
            quant = variant.quant.lower()
            requirement = requirements_by_quant.get(quant)
            # Vision repos ship an mmproj adapter; any precision on disk suffices.
            if (
                requirement is not None
                and _filenames_cached(
                    requirement.main_filenames,
                    requirement.main_size_bytes,
                )
                and (
                    not requirement.mmproj_filenames
                    or _any_mmproj_cached(requirement.mmproj_filenames)
                )
            ):
                return True
            # Byte fallback so a present quant isn't demoted by a filename mismatch;
            # vision repos still need an mmproj cached (any precision).
            if not _quant_bytes_present(quant, variant.size_bytes):
                return False
            if (
                requirement is not None
                and requirement.mmproj_filenames
                and not _any_mmproj_cached(requirement.mmproj_filenames)
            ):
                return False
            return True

        partial_quants: set[str] = set()
        partial_quant_transports: dict[str, Optional[str]] = {}
        try:
            incomplete_hashes = download_registry.incomplete_blob_hashes(
                "model",
                repo_id,
                active_only = True,
                root = hub_cache,
            )
        except Exception as e:
            logger.warning(f"Failed to compute partial GGUF variants for {repo_id}: {e}")
            incomplete_hashes = set()
        scan_snapshot_dir = hf_cache_scan.resolve_snapshot_dir_for_scan(
            "model",
            repo_id,
            repo_cache_dir,
        )
        # Manifest + marker + main incomplete-blob check: catches variants whose
        # download was cancelled or whose expected shards are missing/undersized.
        for variant in variants:
            try:
                requirement = requirements_by_quant.get(variant.quant.lower())
                variant_hashes = requirement.main_hashes if requirement is not None else None
                if variant_hashes is None and incomplete_hashes:
                    variant_hashes = gguf_variant_blob_hashes(
                        repo_id,
                        variant.quant,
                        hf_token,
                        include_companions = False,
                        repo_cache_dir = repo_cache_dir,
                    )
                if hf_cache_scan.is_variant_partial(
                    repo_id,
                    variant.quant,
                    scan_snapshot_dir,
                    incomplete_blob_hashes = incomplete_hashes,
                    variant_blob_hashes = variant_hashes,
                    repo_cache_dir = repo_cache_dir,
                ):
                    partial_quants.add(variant.quant)
                    partial_quant_transports[variant.quant] = _partial_transport_for_variant(
                        repo_id,
                        variant.quant,
                        repo_cache_dir,
                    )
            except Exception as e:
                logger.warning(
                    f"Manifest-based partial check failed for {repo_id}/{variant.quant}: {e}"
                )
        if incomplete_hashes:
            for variant in variants:
                requirement = requirements_by_quant.get(variant.quant.lower())
                if requirement is None:
                    continue
                # companion_hashes adds the MTP drafter (mmproj_hashes covers
                # every mmproj precision in the repo, not just the planned one).
                if (
                    (requirement.mmproj_hashes | requirement.companion_hashes) & incomplete_hashes
                ) and _filenames_cached(
                    requirement.main_filenames,
                    requirement.main_size_bytes,
                ):
                    partial_quants.add(variant.quant)
                    partial_quant_transports.setdefault(
                        variant.quant,
                        _partial_transport_for_variant(
                            repo_id,
                            variant.quant,
                            repo_cache_dir,
                        ),
                    )

        local_blobs_by_quant = _local_main_gguf_blobs_by_quant(repo_id, repo_cache_dir)

        def _variant_detail(v) -> GgufVariantDetail:
            is_partial = v.quant in partial_quants
            requirement = requirements_by_quant.get(v.quant.lower())
            downloaded = _is_fully_downloaded(v) and not is_partial
            return GgufVariantDetail(
                filename = v.filename,
                quant = v.quant,
                display_label = v.display_label,
                size_bytes = v.size_bytes,
                download_size_bytes = (
                    requirement.download_size_bytes if requirement is not None else v.size_bytes
                ),
                downloaded = downloaded,
                update_available = downloaded
                and _variant_update_available_from_requirement(
                    local_blobs_by_quant.get(v.quant.lower(), {}),
                    requirement,
                    v.quant,
                ),
                partial = is_partial,
                partial_transport = (partial_quant_transports.get(v.quant) if is_partial else None),
            )

        return GgufVariantsResponse(
            repo_id = repo_id,
            variants = [_variant_detail(v) for v in variants],
            has_vision = has_vision,
            default_variant = default_variant,
        )

    def _compute_with_cleanables() -> GgufVariantsResponse:
        skip = is_local_path(repo_id) or not _is_valid_repo_id(repo_id)
        try:
            response = _compute()
        except Exception:
            # Offline / metadata fetch failed with only an empty leftover
            # <quant>/ folder cached: still surface it so the UI can delete it,
            # otherwise re-raise the original error.
            if skip:
                raise
            enriched = _mark_empty_dir_cleanables(
                repo_id,
                GgufVariantsResponse(repo_id = repo_id, variants = []),
                _repo_cache_dir_for_request(repo_id, local_path),
            )
            if enriched.variants:
                return enriched
            raise
        if skip:
            return response
        return _mark_empty_dir_cleanables(
            repo_id,
            response,
            _repo_cache_dir_for_request(repo_id, local_path),
        )

    try:
        return await asyncio.to_thread(_compute_with_cleanables)
    except HTTPException:
        raise
    except Exception as e:
        scrubbed = download_registry.scrub_secrets(str(e), hf_token = hf_token)
        # Client-side HF error (missing repo, gated, bad token): pass the status through.
        status = hf_error_status(e)
        if status is not None:
            raise HTTPException(status_code = status, detail = scrubbed)
        logger.error("Error listing GGUF variants for %s: %s", repo_id, scrubbed)
        raise HTTPException(
            status_code = 500,
            detail = "Failed to list GGUF variants: " + scrubbed,
        )
