# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF variant resolution."""

from __future__ import annotations

import asyncio
import re
import threading
import time
import weakref
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
    GgufVariantInfo,
    extract_quant_label,
    iter_hf_cache_snapshots,
    is_big_endian_gguf_path,
    list_empty_gguf_variant_dirs,
    list_gguf_variants,
    list_local_gguf_variants,
    list_partial_gguf_variants_from_state,
    pick_best_gguf,
    select_gguf_cache_snapshot,
)
from hub.utils.paths import (
    is_local_path,
    is_valid_repo_id as _is_valid_repo_id,
)

# Loader's normalizer, not the hub's: they disagree on WSL, and only it names what the load opens.
from utils.paths import normalize_path as _loader_normalize_path
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


def _is_scope_key(variant: str) -> bool:
    """Whether a stored variant key is a download SCOPE ("@diffusion"), not a quant.

    A scoped job rides the variant slot with an "@" prefix to keep its state out
    of the quant namespace. Its manifest names the file it fetched, so rebuilding
    quants from download state would list the same .gguf twice, the scope row
    permanently partial, and cost the picker its single-quant collapse.
    """
    return variant.startswith("@")


def _quants_from_state(
    repo_id: str, hub_cache: Optional[Path]
) -> Optional[tuple[list[GgufVariantInfo], bool]]:
    """``list_partial_gguf_variants_from_state`` with download scopes dropped.

    A scope whose payload is gone is dropped by the lister, the only place that
    can tell a recovered digest from a variant truly named like one (see
    _is_state_filename_fallback); left here is the readable "@diffusion" case.
    Scopes alone return None like nothing at all, since a scope naming no .gguf
    reconstructs as ``f"{variant}.gguf"``, a file that never existed, so the
    caller must fall through rather than serve one.
    """
    partial = list_partial_gguf_variants_from_state(repo_id, hub_cache = hub_cache)
    if partial is None:
        return None
    variants, has_vision = partial
    variants = [v for v in variants if not (v.quant and _is_scope_key(v.quant))]
    if not variants:
        return None
    return variants, has_vision


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


def _snapshot_scope_for_request(repo_id: str, local_path: Optional[str]) -> Optional[Path]:
    """The one snapshot *local_path* names, when it names one of *repo_id*'s.

    A row pinned to a snapshot loads out of that directory and nothing else, so readiness has to be
    counted there: a quant sitting in a sibling revision is not one this row can resolve. The
    answer carries the requested repo's identity, so the directory has to be that repo's cache;
    any cache root will do, since the same repo is cached under the same name in each.
    """
    if not local_path:
        return None
    try:
        local = Path(local_path).expanduser().resolve(strict = False)
    except (OSError, RuntimeError, ValueError):
        return None
    if local.parent.name != "snapshots" or not local.is_dir():
        return None
    expected = repo_cache_dir_name("model", repo_id).lower()
    return local if local.parent.parent.name.lower() == expected else None


def pinned_snapshot_for_request(repo_id: str, local_path: Optional[str]) -> Optional[str]:
    """The snapshot *local_path* pins, for callers outside this module that must read that copy."""
    scope = _snapshot_scope_for_request(repo_id, local_path)
    return str(scope) if scope is not None else None


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
            variants.append(
                GgufVariantDetail(
                    filename = f"{label}.gguf", quant = label, partial = True, cleanable = True
                )
            )
    return response.model_copy(update = {"variants": variants})


def _direct_gguf_loads(path: Path) -> bool:
    """Whether the load path takes *path* itself as the model.

    Mirrors ``detect_gguf_model``: refuses companions (mmproj, MTP/dspark
    drafter) and big-endian builds by name+parent, same as the load path.
    """
    # Load extractor, not the hub one: they disagree on F16-be-checkpoint-Q4_K_M shapes.
    from utils.models.model_config import _extract_quant_label

    context = f"{path.parent.name}/{path.name}"
    return not (
        _is_mmproj_filename(path.name)
        or _is_mtp_drafter_path(context)
        or is_big_endian_gguf_path(context, _extract_quant_label(context))
    )


# llama.cpp's split grammar (model_config._GGUF_SPLIT_FILE_RE): five digits exactly, since a
# shorter name loads as an ordinary file, not the cache scan's looser -\d{3,}- resume form.
_DIRECT_SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})$", re.IGNORECASE)


def _direct_gguf_split_is_whole(path: Path) -> bool:
    """Whether *path*'s split set is entirely beside it (True when it is not a split).

    llama.cpp resolves a split's siblings from the main shard's directory (see
    llama_cpp._snapshot_has_all_shards), so a lone shard fails after teardown.
    A symlinked shard follows its target, like _local_gguf_load_path. Unknown
    (unreadable directory, nonsense total) reports whole to keep the row ready.
    """
    match = _DIRECT_SPLIT_RE.match(path.name.rsplit(".", 1)[0])
    if match is None:
        return True
    total = int(match.group("total"))
    if total < 2:
        return True
    sibling = re.compile(
        re.escape(match.group("stem"))
        + r"-(\d{"
        + str(len(match.group("index")))
        + r"})-of-"
        + re.escape(match.group("total"))
        + r"\.gguf$",
        re.IGNORECASE,
    )

    def _indexes_beside(target: Path, pattern) -> set:
        return {
            int(m.group(1))
            for p in target.parent.iterdir()
            if (m := pattern.match(p.name)) and p.is_file() and p.stat().st_size > 0
        }

    def _target_set_is_whole(target: Path) -> bool:
        """Whether the symlink target's own declared set is beside it.

        The target names its own grammar/total (need not match the alias); the
        load launches whatever the target declares.
        """
        m = _DIRECT_SPLIT_RE.match(target.name.rsplit(".", 1)[0])
        if m is None:
            # Split-named link on an ordinary target: loads as-is, nothing missing.
            return True
        target_total = int(m.group("total"))
        pattern = re.compile(
            re.escape(m.group("stem"))
            + r"-(\d{"
            + str(len(m.group("index")))
            + r"})-of-"
            + re.escape(m.group("total"))
            + r"\.gguf$",
            re.IGNORECASE,
        )
        return _indexes_beside(target, pattern) >= set(range(1, target_total + 1))

    try:
        found = _indexes_beside(path, sibling)
        if not found >= set(range(1, total + 1)) and path.is_symlink():
            # _local_gguf_load_path resolves siblings from the TARGET, so a renamed alias
            # still loads the target's real set -- and a torn target stays torn.
            return _target_set_is_whole(path.resolve())
    except OSError:
        return True
    # Declared indexes, not a count: an over-indexed stray must not stand in for a missing
    # shard, and a zero-byte sibling is an interrupted copy.
    return found >= set(range(1, total + 1))


# Cache scan's looser grammar (hub.utils.inventory_scan._GGUF_SPLIT_RE), not the five digits above.
_CACHE_SPLIT_RE = re.compile(r"-(\d{3,})-of-(\d{3,})(?=\.gguf$)", re.IGNORECASE)

# Only enumerates candidates for the resolver to adjudicate; drift costs a candidate, not a verdict.
_KNOWN_QUANT_RE = re.compile(
    r"(UD-)?"
    r"(MXFP[0-9]+(?:_[A-Z0-9]+)*"
    r"|IQ[0-9]+_[A-Z]+(?:_[A-Z0-9]+)?"
    r"|TQ[0-9]+_[0-9]+"
    r"|Q[0-9]+_K_[A-Z]+"
    r"|Q[0-9]+_[0-9]+"
    r"|Q[0-9]+_K"
    r"|BF16|F16|F32)"
    r"(-[0-9]+(?:\.[0-9]+)?bpw)?",
    re.IGNORECASE,
)


def _will_serve(resolved: Optional[str]) -> bool:
    """Whether llama-server can actually open what the resolver chose.

    The resolver is extension-authoritative by design (it must answer inside
    the Windows lock window), so it says yes to an empty copy or a torn split
    too -- the two ways a resolved path still fails after teardown.
    """
    if not resolved:
        return False
    try:
        path = Path(resolved)
        # The resolver answers for nonexistent paths, so absence is caught here: "no such
        # file" is definite, a read error (Windows lock window) stays unknown and serves.
        # stat(), not exists(), which swallows every OSError on 3.14 and would read a
        # sharing violation as absence.
        try:
            size = path.stat().st_size
        except (FileNotFoundError, NotADirectoryError):
            return False
        return size > 0 and _direct_gguf_split_is_whole(path)
    except OSError:
        return True


def _loadable_variants(identifier: str, variants):
    """The advertised quants a load of *identifier* would actually serve.

    Authoritative by construction: asks the same resolver /api/inference/load
    uses, then checks the chosen file as llama-server would find it, so a
    client never has to predict either from filenames. One resolver call per
    row, local answers only. None when the question does not apply.
    """
    from utils.models.model_config import _find_local_gguf_by_variant

    # from_identifier only consults the variant for a DIRECTORY; a direct file loads itself
    # regardless, so leave it unanswered rather than be stricter than the load. stat(), not
    # is_file(): a locked file must stay unanswered (an empty list is authoritative at the
    # gate), which is_file() cannot express -- it raises here, and answers False from 3.14.
    import stat as _stat

    try:
        if _stat.S_ISREG(Path(identifier).expanduser().stat().st_mode):
            return None
    except (FileNotFoundError, NotADirectoryError):
        pass
    except OSError:
        return None

    # The resolver walks the tree per call, so spellings are deduped against `seen` first:
    # ~2 calls per row (36 for 18 quants, 179ms over 144 files), not one per spelling. Each
    # alias is still confirmed, so a token binding a different file is never advertised.
    accepted: list = []
    seen = set()
    for variant in variants:
        quant = getattr(variant, "quant", None)
        if not isinstance(quant, str) or not quant:
            continue
        try:
            bound = _find_local_gguf_by_variant(identifier, quant)
            if not _will_serve(bound):
                continue
        except Exception:
            continue
        key = quant.strip().lower()
        if key not in seen:
            seen.add(key)
            accepted.append(quant)
        # The resolver also takes the snapshot-relative stem (BF16/model) and the basename's
        # own tokens; derive those and let it confirm each. It returns an absolute path, so a
        # relative identifier must be resolved the same way or its alias is lost. Unresolved
        # first: a symlink out of the tree still answers BF16/model there, but not resolved.
        relative = Path(bound).name
        for base_raw, bound_path in (
            (Path(identifier).expanduser(), Path(bound)),
            (Path(identifier).expanduser().resolve(), Path(bound).resolve()),
        ):
            try:
                base = base_raw.parent if base_raw.is_file() else base_raw
                relative = bound_path.relative_to(base).as_posix()
                break
            except (OSError, ValueError):
                continue
        basename = Path(bound).name
        derived = {
            re.sub(r"-\d{3,}-of-\d{3,}$", "", relative.rsplit(".", 1)[0]),
            re.sub(r"-\d{3,}-of-\d{3,}$", "", basename.rsplit(".", 1)[0]),
        }
        for match in _KNOWN_QUANT_RE.finditer(basename.rsplit(".", 1)[0]):
            prefix, core, bpw = match.group(1) or "", match.group(2), match.group(3) or ""
            derived.add(f"{prefix}{core}")
            if bpw:
                derived.add(f"{prefix}{core}{bpw}")
        for spelling in sorted(derived):
            key = spelling.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            try:
                if _will_serve(_find_local_gguf_by_variant(identifier, spelling)):
                    accepted.append(spelling)
            except Exception:
                continue
    return accepted


def _loads_without_variant(identifier: str) -> bool:
    """Whether a variantless load of *identifier* would serve GGUF weights."""
    from utils.models.model_config import detect_gguf_model
    try:
        return _will_serve(detect_gguf_model(identifier))
    except Exception:
        return False


def _complete_quants_under(snapshot: str):
    """Quants whose shards are all present under *snapshot*, or None if unknown.

    None on any error, so every row reports downloaded as before: a scan problem must not mark a
    working folder unusable.
    """
    try:
        complete = hf_cache_scan.complete_snapshot_variants(snapshot)
    except Exception:
        return None
    if complete is None:
        return None
    return complete


def _complete_with_servable(snapshot: str, complete, variants):
    """*complete* plus the quants whose bound file the load would actually serve.

    The scan's looser -\\d{3,}- grammar can read a name the LOAD treats as an
    ordinary file (five-digit splits only) as a torn set. Rather than re-judge
    names here, ask the resolver what it actually chose for the quant.
    """
    if complete is None:
        return None
    from utils.models.model_config import _find_local_gguf_by_variant

    repaired = set(complete)
    for variant in variants:
        quant = getattr(variant, "quant", None)
        if not isinstance(quant, str) or not quant or quant in repaired:
            continue
        try:
            if _will_serve(_find_local_gguf_by_variant(snapshot, quant)):
                repaired.add(quant)
        except Exception:
            continue
    return repaired


# One scan per identical request in flight. Aborting the HTTP request cannot stop the scan
# already running in its thread, so the picker's Retry, and reopening a row, would each start
# another against a filesystem that is not answering: measured 23 retries filling all 20
# default-executor workers, starving unrelated offloaded work and holding up exit. Joining the
# running scan costs at most its own duration in staleness, well inside the client's cache TTL.
_VARIANTS_INFLIGHT: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


async def _shared_variants_scan(key: tuple, compute):
    """*compute* in a thread, shared with any identical request already running."""
    loop = asyncio.get_running_loop()
    inflight = _VARIANTS_INFLIGHT.get(loop)
    if inflight is None:
        inflight = {}
        _VARIANTS_INFLIGHT[loop] = inflight

    task = inflight.get(key)
    if task is None:
        task = asyncio.ensure_future(asyncio.to_thread(compute))

        def _release(finished: asyncio.Future) -> None:
            inflight.pop(key, None)
            if not finished.cancelled():
                finished.exception()  # retrieved, so a caller-less failure stays quiet

        inflight[key] = task
        task.add_done_callback(_release)
    # Shielded: one caller giving up must not cancel the scan the others are waiting on.
    return await asyncio.shield(task)


class VariantsAnswer(NamedTuple):
    """The listing, plus the directory it came from so a caller reading metadata reads the
    same copy. None means no single directory: the repo's caches answered."""

    response: GgufVariantsResponse
    context_source: Optional[str]


async def get_gguf_variants_answer(
    repo_id: str,
    prefer_local_cache: bool = False,
    offline: bool = False,
    local_path: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> VariantsAnswer:
    """
    List available GGUF quantization variants for a HuggingFace repo
    or a local directory (e.g. LM Studio model folder).

    Returns all available quantization variants (Q4_K_M, Q8_0, BF16, etc.)
    with file sizes, whether the model supports vision, and the recommended
    default variant.
    """
    # Set by whichever branch answers, and returned with the listing: the HF cache answers
    # before local_path, so a caller cannot infer the copy from the request alone.
    answered_from: list[Optional[str]] = [None]
    # Set by the existence-first local branch below: a repo-shaped id resolving to a directory
    # is answered by that directory alone, not the HF cache of the same-named repo -- else a
    # GGUF-less directory could evict the resident model via a cleanable row from the cache.
    answered_locally = [False]

    def _compute() -> GgufVariantsResponse:
        repo_cache_dir = (
            None if is_local_path(repo_id) else _repo_cache_dir_for_request(repo_id, local_path)
        )
        hub_cache = repo_cache_dir.parent if repo_cache_dir is not None else None
        snapshot_scope = _snapshot_scope_for_request(repo_id, local_path)

        def _local_response(
            response_repo_id: str,
            variants,
            has_vision: bool,
            complete = None,
        ) -> GgufVariantsResponse:
            """*complete* is the set of quants whose shards are all on disk; None reports every row
            downloaded. A quant short a shard stays listed to resume or delete, but is not offered
            as ready, since the loader would ask llama-server for files that are not there.
            """

            def _downloaded(v) -> bool:
                # An unlabelled quant cannot be judged, so it is kept as ready.
                return complete is None or not v.quant or v.quant in complete

            # The default comes from the ready rows; with none ready every row is the fallback.
            ready = [v for v in variants if _downloaded(v)]
            best = pick_best_gguf([v.filename for v in (ready or variants)])
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
                        downloaded = _downloaded(v),
                        partial = not _downloaded(v),
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

        def _with_state_partials(response: GgufVariantsResponse) -> GgufVariantsResponse:
            """Add quants known only from download state. A sibling cancelled
            before any file landed has no snapshot entry, so a listing built
            from the cache alone reads as if it were never asked for, and the
            row loses its resume."""
            state = _quants_from_state(repo_id, hub_cache)
            if state is None:
                return response
            listed = {v.quant.lower() for v in response.variants if v.quant}
            extra = [
                GgufVariantDetail(
                    filename = v.filename,
                    quant = v.quant,
                    display_label = v.display_label,
                    size_bytes = v.size_bytes,
                    download_size_bytes = v.download_size_bytes or v.size_bytes,
                    downloaded = False,
                    partial = True,
                    partial_transport = _partial_transport_for_variant(
                        repo_id, v.quant, repo_cache_dir
                    ),
                )
                for v in state[0]
                if v.quant and v.quant.lower() not in listed
            ]
            if not extra:
                return response
            return response.model_copy(update = {"variants": [*response.variants, *extra]})

        # Local directory path (e.g. LM Studio models) — scan filesystem. Load-path parity:
        # from_identifier resolves existence-first, so a marker-less relative name that exists
        # here is a local model, not a Hub id, and a direct .gguf file loads without the
        # metadata siblings the directory scan requires. It also normalizes first, so every
        # question below must ask about the same path: under WSL "C:\\models\\qwen" maps to
        # /mnt/c/models/qwen, and probing the raw spelling would report a working model
        # unloadable.
        local_id = _loader_normalize_path(repo_id) if is_local_path(repo_id) else repo_id
        local_target = None
        try:
            probe = Path(local_id).expanduser()
            if is_local_path(repo_id) or probe.exists():
                local_target = probe
        except OSError:
            local_target = None
        if local_target is not None:
            variants, has_vision = list_local_gguf_variants(local_id)
            # The load id is this path, so a scan-torn quant is still ready when the file the
            # resolver binds opens fine.
            complete = _complete_with_servable(local_id, _complete_quants_under(local_id), variants)
            if (
                not variants
                and local_target.is_file()
                and local_target.suffix.lower() == ".gguf"
                and _direct_gguf_loads(local_target)
            ):
                # An unmarked-parent .gguf is skipped by the directory scan but detect_gguf_model
                # still loads it; falling back only here keeps a marked parent's siblings.
                try:
                    size = local_target.stat().st_size
                except OSError:
                    size = 0
                # The load resolver's own extractor over the context it reads, so the quant is
                # what the echoed load resolves (the hub one differs on F16-checkpoint-Q4_K_M).
                from utils.models.model_config import _extract_quant_label

                variants = [
                    GgufVariantInfo(
                        filename = local_target.name,
                        quant = _extract_quant_label(
                            f"{local_target.parent.name}/{local_target.name}"
                        ),
                        size_bytes = size,
                    )
                ]
                # The shard scan resolves a file to its marked parent, so an unmarked one walks
                # a bare file and misreports the row. Ask the file itself: whole unless it is a
                # split missing a sibling or a zero-byte interrupted copy.
                complete = None if size > 0 and _direct_gguf_split_is_whole(local_target) else set()
            answered_from[0] = repo_id
            answered_locally[0] = True
            # Surface the resolution so the CLI gate matches on the local resolver's exact
            # labels rather than the id's shape, and answers its real question -- would this
            # load resolve -- with the loader itself, so no client mirrors its grammar.
            return _local_response(repo_id, variants, has_vision, complete).model_copy(
                update = {
                    "resolved_locally": True,
                    "loadable_variants": _loadable_variants(local_id, variants),
                    "loadable": _loads_without_variant(local_id),
                }
            )

        # Reject invalid remote repo_ids up front (like download/delete) so a
        # malformed id returns 400 instead of a 500 from the HF client.
        if not _is_valid_repo_id(repo_id):
            raise HTTPException(status_code = 400, detail = f"Invalid repo_id: {repo_id!r}")

        def _scoped_local_response():
            """The pinned snapshot's own answer, or None when it holds nothing."""
            if snapshot_scope is None:
                return None
            variants, has_vision = list_local_gguf_variants(str(snapshot_scope))
            if not (variants or has_vision):
                return None
            answered_from[0] = str(snapshot_scope)
            return _with_state_partials(
                _local_response(
                    repo_id, variants, has_vision, _complete_quants_under(str(snapshot_scope))
                )
            )

        local_only = prefer_local_cache or offline
        if local_only:
            scoped_response = _scoped_local_response()
            if scoped_response is not None:
                return scoped_response
            cached = select_gguf_cache_snapshot(repo_id, root = hub_cache)
            if cached is not None:
                variants, has_vision, complete, snapshot = cached
                # Name the answering snapshot: a repo-wide walk could read a different cache's
                # context length, and a repo-dir walk a sibling revision's.
                answered_from[0] = str(snapshot)
                # The lister leaves torn quants in: they stay listed for management, but not ready.
                return _with_state_partials(
                    _local_response(repo_id, variants, has_vision, complete)
                )
            if local_path and is_local_path(local_path):
                variants, has_vision = list_local_gguf_variants(local_path)
                if variants or has_vision:
                    answered_from[0] = local_path
                    # Same reason as the is_local_path branch above.
                    return _local_response(
                        repo_id, variants, has_vision, _complete_quants_under(local_path)
                    )
            partial = _quants_from_state(repo_id, hub_cache)
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

        def _cache_fallback_response():
            """Cached answer scoped to the cache this request names, or None.

            The lister's own cache read is repo-wide, so redoing it here pins the listing and,
            through ``answered_from``, its context metadata to the named copy.
            """
            scoped_response = _scoped_local_response()
            if scoped_response is not None:
                return scoped_response
            cached = select_gguf_cache_snapshot(repo_id, root = hub_cache)
            if cached is not None:
                variants, has_vision, complete, snapshot = cached
                answered_from[0] = str(snapshot)
                # Same reason as the local_only branch above, state partials included:
                # an unreachable Hub is exactly when a resume has nowhere else to surface.
                return _with_state_partials(
                    _local_response(repo_id, variants, has_vision, complete)
                )
            partial = _quants_from_state(repo_id, hub_cache)
            if partial is not None:
                variants, has_vision = partial
                return _partial_local_response(repo_id, variants, has_vision)
            return None

        try:
            variants, has_vision, siblings = list_gguf_variants(repo_id, hf_token = hf_token)
        except Exception:
            fallback = _cache_fallback_response()
            if fallback is not None:
                return fallback
            raise

        # siblings is None only when the lister answered from its own repo-wide cache. Falling
        # through is deliberate: readiness below still counts against this request's own cache.
        if siblings is None:
            fallback = _cache_fallback_response()
            if fallback is not None:
                return fallback

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
            # A pinned row resolves inside one directory, so nothing else counts as downloaded.
            scoped_snapshots = (
                [snapshot_scope]
                if snapshot_scope is not None
                else iter_hf_cache_snapshots(repo_id, root = hub_cache)
            )
            for snap in scoped_snapshots:
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
        scan_snapshot_dir = snapshot_scope or hf_cache_scan.resolve_snapshot_dir_for_scan(
            "model",
            repo_id,
            repo_cache_dir,
        )
        # A marker or manifest carries no revision, so attribute it like the inventory row.
        repo_signal_applies = hf_cache_scan.repo_signal_applies_to_snapshot(
            repo_cache_dir, scan_snapshot_dir
        )
        # The excuse is that this snapshot holds the quant whole, so a quant it lacks stays the
        # cancelled download and keeps its resume and delete affordances.
        excused_quants = (
            frozenset()
            if repo_signal_applies or scan_snapshot_dir is None
            else frozenset(
                q.lower() for q in (_complete_quants_under(str(scan_snapshot_dir)) or ())
            )
        )

        def _repo_signals_apply_to(quant: str) -> bool:
            return repo_signal_applies or quant.lower() not in excused_quants

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
                    repo_signal_applies = _repo_signals_apply_to(variant.quant),
                ):
                    partial_quants.add(variant.quant)
                    partial_quant_transports[variant.quant] = _partial_transport_for_variant(
                        repo_id,
                        variant.quant,
                        repo_cache_dir,
                    )
            except Exception as e:
                logger.warning(
                    f"Manifest-based partial check failed for " f"{repo_id}/{variant.quant}: {e}"
                )
        # Same attribution as above: a pinned snapshot is not judged by a newer attempt's blobs.
        if incomplete_hashes:
            for variant in variants:
                requirement = requirements_by_quant.get(variant.quant.lower())
                if requirement is None or not _repo_signals_apply_to(variant.quant):
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

    def _compute_with_cleanables() -> VariantsAnswer:
        # Returned with the answer, not read from the closure afterwards: coalesced callers
        # share one computation and must all see the copy it actually answered from.
        return VariantsAnswer(_compute_response(), answered_from[0])

    def _compute_response() -> GgufVariantsResponse:
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
        if skip or answered_locally[0]:
            return response
        return _mark_empty_dir_cleanables(
            repo_id,
            response,
            _repo_cache_dir_for_request(repo_id, local_path),
        )

    from utils.hf_cache_settings import configured_cache_key

    inflight_key = (
        repo_id,
        bool(prefer_local_cache),
        bool(offline),
        local_path or "",
        hf_cache_scan.token_fingerprint(hf_token),
        # Switching cache storage must start a fresh scan rather than join one that is
        # stuck on the old volume.
        configured_cache_key(),
    )
    try:
        return await _shared_variants_scan(inflight_key, _compute_with_cleanables)
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


async def get_gguf_variants_response(
    repo_id: str,
    prefer_local_cache: bool = False,
    offline: bool = False,
    local_path: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> GgufVariantsResponse:
    """The listing alone, for callers that do not read metadata off the same copy."""
    answer = await get_gguf_variants_answer(
        repo_id,
        prefer_local_cache = prefer_local_cache,
        offline = offline,
        local_path = local_path,
        hf_token = hf_token,
    )
    return answer.response
