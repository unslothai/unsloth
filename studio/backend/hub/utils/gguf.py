# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF filename helpers. Quantization variants are derived from filenames, not parsed from binary GGUF headers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from loggers import get_logger

logger = get_logger(__name__)
_GGUF_MODEL_INFO_TIMEOUT_SECONDS = 5.0


@dataclass
class GgufVariantInfo:
    filename: str
    quant: str
    size_bytes: int
    display_label: Optional[str] = None
    download_size_bytes: int = 0


GGUF_QUANT_PREFERENCE = [
    "UD-Q4_K_XL",
    "UD-Q4_K_L",
    "UD-Q5_K_XL",
    "UD-Q3_K_XL",
    "UD-Q6_K_XL",
    "UD-Q6_K_S",
    "UD-Q8_K_XL",
    "UD-Q2_K_XL",
    "UD-IQ4_NL",
    "UD-IQ4_XS",
    "UD-IQ3_S",
    "UD-IQ3_XXS",
    "UD-IQ2_M",
    "UD-IQ2_XXS",
    "UD-IQ1_M",
    "UD-IQ1_S",
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q8_0",
    "Q3_K_M",
    "Q3_K_L",
    "Q3_K_S",
    "Q2_K",
    "Q2_K_L",
    "IQ4_NL",
    "IQ4_XS",
    "IQ3_M",
    "IQ3_XXS",
    "IQ2_M",
    "IQ1_M",
    "F16",
    "BF16",
    "F32",
]

_GGUF_SPLIT_SUFFIX_RE = re.compile(r"-\d{3,}-of-\d{3,}", re.IGNORECASE)
_GGUF_QUANT_RE = re.compile(
    r"(UD-)?"
    r"(MXFP[0-9]+(?:_[A-Z0-9]+)*"
    r"|IQ[0-9]+_[A-Z]+(?:_[A-Z0-9]+)?"
    r"|TQ[0-9]+_[0-9]+"
    r"|Q[0-9]+_K_[A-Z]+"
    r"|Q[0-9]+_[0-9]+"
    r"|Q[0-9]+_K"
    r"|BF16|F16|F32)",
    re.IGNORECASE,
)


def is_mmproj_filename(filename: str) -> bool:
    return "mmproj" in filename.lower()


# dspark and dflash are the same DeepSeek V4 Flash drafter: the folder it ships
# in and the architecture it reports.
_DRAFTER_KINDS = ("mtp", "dspark", "dflash")

# Narrower than the prefix rule, since a directory name can be the user's: only
# ``mtp/`` and ``dspark/`` are ever a publisher's companion folder, while
# ``dflash/`` is a family name a user picks for real weights. DFlash drafters
# still match the prefix (ggml-org/Qwen3.6-27B-GGUF: dflash-Qwen3.6-27B-BF16.gguf).
_DRAFTER_DIR_KINDS = ("mtp", "dspark")


def is_mtp_drafter_path(path: str) -> bool:
    """True for a separate-file speculative-decoding drafter, a companion to the
    main model rather than a selectable quant: repo-root ``mtp-*.gguf`` (the Q8_0
    copy unsloth ships for llama.cpp ``-hf`` auto-discovery), the ``MTP/`` subdir
    copies (Gemma 4) and the ``dspark/`` drafters (DeepSeek V4 Flash). Repos that
    bake the head into the main GGUF (Qwen) have no such file, so this is False
    for them. Must be excluded from main-model selection everywhere mmproj is.

    Matched by basename prefix, or by an exact parent directory for the kinds in
    ``_DRAFTER_DIR_KINDS``; never a substring, since the kind names double as
    family names, so ``Qwen3.6-27B-MTP-Q4_K_M.gguf`` and
    ``Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf`` ARE the model.

    CANONICAL COPY, with two mirrors that must change in lockstep:
    utils/models/model_config.py ``_is_mtp_drafter`` (utils cannot import hub)
    and core/inference/llama_cpp.py ``_is_companion_gguf_path`` (core avoids hub
    imports; bundles the mmproj check).
    """
    p = path.replace("\\", "/").lower()
    if not p.endswith(".gguf"):
        return False
    parts = [segment for segment in p.split("/") if segment]
    name, parents = parts[-1], parts[:-1]
    return any(name.startswith(f"{kind}-") for kind in _DRAFTER_KINDS) or any(
        kind in parents for kind in _DRAFTER_DIR_KINDS
    )


def is_gguf_filename(filename: str) -> bool:
    return filename.lower().endswith(".gguf")


def is_drafter_dir_path(path: str) -> bool:
    """True when *path* sits under a drafter directory, the half of the rule the
    publisher controls. Always a companion, unlike the basename prefix."""
    p = path.replace("\\", "/").lower()
    if not p.endswith(".gguf"):
        return False
    return any(kind in [s for s in p.split("/")[:-1] if s] for kind in _DRAFTER_DIR_KINDS)


def drafter_paths_in(paths: Iterable[str]) -> frozenset[str]:
    """The drafter GGUFs among *paths*. A companion is only a companion when it
    has something to accompany, so a PREFIX-named drafter with no main model
    beside it is kept: a repo can be named after the drafter family and carry
    the prefix on every real quant (mradermacher/DFlash-Qwen3.5-27B-Uncensored-
    GGUF), and drafter-only repos exist. Where a main model IS present the
    prefix still wins, or ggml-org/Qwen3.6-27B-GGUF's 3 GB
    ``dflash-Qwen3.6-27B-BF16.gguf`` would merge into the real 54 GB one.

    A DIRECTORY-named drafter (``mtp/``, ``dspark/``) is never reprieved: the
    publisher laid it out as a companion, and a snapshot holding only that is a
    half-downloaded repo, not a model.
    """
    listing = [path for path in paths if is_gguf_filename(path)]
    drafters = {path for path in listing if is_mtp_drafter_path(path)}
    if any(path not in drafters and not is_mmproj_filename(path) for path in listing):
        return frozenset(drafters)
    return frozenset(path for path in drafters if is_drafter_dir_path(path))


_BIG_ENDIAN_GGUF_FILENAME_RE = re.compile(r"(^|[-_])be(?:[._-]|$)", re.IGNORECASE)


def is_big_endian_gguf_path(path: str, quant: str = "") -> bool:
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0].lower()
    quant_key = quant.strip().lower()
    quant_index = stem.find(quant_key) if quant_key else -1
    parent = normalized.rsplit("/", 1)[0].lower() if "/" in normalized else ""
    quant_in_parent_only = (
        bool(parent)
        and quant_index < 0
        and (
            (quant_key and quant_key in parent)
            or (not quant_key and _GGUF_QUANT_RE.search(parent) is not None)
        )
    )
    for match in _BIG_ENDIAN_GGUF_FILENAME_RE.finditer(stem):
        if quant_index >= 0 and quant_index < match.start():
            return True
        tail = stem[match.end() :].lstrip("._-")
        if not tail or _GGUF_QUANT_RE.search(tail) is None:
            return not quant_in_parent_only
    return False


# Cap recursive walks so a huge or system path cannot run unbounded.
_MAX_LOCAL_SCAN_ENTRIES = 100_000


def iter_gguf_files(directory: Path, recursive: bool = False):
    if not directory.is_dir():
        return
    if recursive:
        seen = 0
        # os.walk skips unreadable subdirs instead of raising (e.g. /proc).
        for dirpath, dirnames, filenames in os.walk(directory, onerror = lambda _e: None):
            for name in filenames:
                if is_gguf_filename(name):
                    yield Path(dirpath) / name
            seen += len(dirnames) + len(filenames)
            if seen > _MAX_LOCAL_SCAN_ENTRIES:
                return
        return
    try:
        entries = list(directory.iterdir())
    except OSError:
        return
    for file in entries:
        try:
            if file.is_file() and is_gguf_filename(file.name):
                yield file
        except OSError:
            continue


def pick_best_gguf(filenames: list[str]) -> Optional[str]:
    drafters = drafter_paths_in(filenames)
    gguf_files = [
        name
        for name in filenames
        if is_gguf_filename(name)
        and not is_mmproj_filename(name)
        and name not in drafters
        and not is_big_endian_gguf_path(name, extract_quant_label(name))
    ]
    if not gguf_files:
        return None
    by_quant: dict[str, str] = {}
    for name in gguf_files:
        by_quant.setdefault(extract_quant_label(name).upper(), name)
    for quant in GGUF_QUANT_PREFERENCE:
        filename = by_quant.get(quant.upper())
        if filename is not None:
            return filename
    return gguf_files[0]


def _gguf_stem(filename: str) -> str:
    basename = filename.rsplit("/", 1)[-1]
    return _GGUF_SPLIT_SUFFIX_RE.sub("", basename.rsplit(".", 1)[0]).strip()


_FLOAT_PRECISION_QUANTS = frozenset({"BF16", "F16", "F32"})


def _select_quant_match(text: str) -> Optional[re.Match]:
    fallback: Optional[re.Match] = None
    for match in _GGUF_QUANT_RE.finditer(text):
        if match.group(2).upper() in _FLOAT_PRECISION_QUANTS:
            if fallback is None:
                fallback = match
            continue
        return match
    return fallback


def extract_quant_token(filename: str) -> Optional[str]:
    stem = _gguf_stem(filename)
    match = _select_quant_match(stem)
    if not match and "/" in filename:
        parents = filename.rsplit("/", 1)[0]
        for segment in reversed(parents.split("/")):
            parent_match = _select_quant_match(segment)
            if parent_match:
                match = parent_match
                break
    if match:
        prefix = match.group(1) or ""
        return f"{prefix}{match.group(2)}"
    return None


def _unknown_gguf_variant_key(filename: str) -> str:
    stem = _gguf_stem(filename)
    if "/" not in filename:
        return stem or "gguf"
    parents = filename.rsplit("/", 1)[0].strip("/")
    return f"{parents}/{stem}" if parents and stem else stem or "gguf"


def extract_quant_label(filename: str) -> str:
    return extract_quant_token(filename) or _unknown_gguf_variant_key(filename)


def _apply_gguf_display_labels(variants: list[GgufVariantInfo]) -> None:
    unknown_variants = [
        variant for variant in variants if extract_quant_token(variant.filename) is None
    ]
    if not unknown_variants:
        return
    ambiguous = len(unknown_variants) > 1
    for variant in unknown_variants:
        variant.display_label = f"GGUF · {variant.filename}" if ambiguous else "GGUF"


def _env_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in (
        "1",
        "true",
        "yes",
    ) or os.environ.get("TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes")


def iter_hf_cache_snapshots(repo_id: str, root: Optional[Path] = None):
    from hub.utils.hf_cache_state import (
        iter_active_repo_cache_dirs,
        iter_repo_cache_dirs,
        snapshot_selection_key,
    )

    snapshots: list[Path] = []
    repo_dirs = (
        iter_active_repo_cache_dirs("model", repo_id, root = root)
        if root is not None
        else iter_repo_cache_dirs("model", repo_id)
    )
    for repo_dir in repo_dirs:
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        try:
            snapshots.extend(snap for snap in snapshots_dir.iterdir() if snap.is_dir())
        except OSError:
            continue

    # Same key the inventory row selects with, so both name one snapshot.
    snapshots.sort(key = snapshot_selection_key, reverse = True)
    yield from snapshots


def list_empty_gguf_variant_dirs(repo_id: str, root: Optional[Path] = None) -> set[str]:
    """Quant labels present only as an EMPTY snapshot ``<quant>/`` folder (an
    interrupted split download); a quant with shards in any snapshot is excluded."""
    empty: dict[str, str] = {}
    nonempty: set[str] = set()
    snapshots = (
        iter_hf_cache_snapshots(repo_id, root = root)
        if root is not None
        else iter_hf_cache_snapshots(repo_id)
    )
    for snapshot in snapshots:
        try:
            entries = list(snapshot.iterdir())
        except OSError:
            continue
        for sub in entries:
            try:
                if sub.is_symlink() or not sub.is_dir():
                    continue
                quant = extract_quant_token(sub.name)
                if not quant:
                    continue
                has_child = any(sub.iterdir())
            except OSError:
                continue
            if has_child:
                nonempty.add(quant.lower())
            else:
                empty.setdefault(quant.lower(), quant)
    return {label for key, label in empty.items() if key not in nonempty}


def list_gguf_variants_from_hf_cache(
    repo_id: str, root: Optional[Path] = None
) -> Optional[tuple[list[GgufVariantInfo], bool, set]]:
    """``(variants, has_vision, complete)`` for the snapshot a load would read.

    Everything in that snapshot is listed, so a torn download stays visible to resume or delete;
    *complete* is the subset whose shards are all present, so the caller marks the rest partial
    rather than ready, as the snapshot-path form of this call does.
    """
    # Local import: inventory_scan imports this module.
    from hub.utils.inventory_scan import complete_snapshot_variants

    snapshots = (
        iter_hf_cache_snapshots(repo_id, root = root)
        if root is not None
        else iter_hf_cache_snapshots(repo_id)
    )
    # Pick the snapshot the inventory row does: newest holding a whole quant, else first non-empty.
    fallback: Optional[tuple[list[GgufVariantInfo], bool, set]] = None
    for snapshot in snapshots:
        variants, has_vision = list_local_gguf_variants(str(snapshot))
        complete = complete_snapshot_variants(str(snapshot)) if variants else set()
        if variants:
            # Selection only: an unlabelled quant cannot be judged, so it counts as usable.
            if any(not v.quant or v.quant in complete for v in variants):
                return variants, has_vision, complete
        if fallback is None and (variants or has_vision):
            fallback = (variants, has_vision, complete)
    return fallback


def list_partial_gguf_variants_from_state(
    repo_id: str, hub_cache: Optional[Path] = None
) -> Optional[tuple[list[GgufVariantInfo], bool]]:
    """Reconstruct GGUF variants from download manifests/markers alone.

    Used when no completed snapshot exists (download cancelled or interrupted)
    and the HF API is unreachable (offline/gated/private). Each variant's
    ``quant`` is the stored variant key so a resume passes the matching
    ``--variant`` back to the worker.
    """
    from hub.utils import download_manifest

    # Variant identity on disk is case-insensitive (_entry_key lowercases it), so
    # dedupe on the lowercased key. Manifests are read first to keep their
    # original-casing label over a lowercased cancel marker for the same variant.
    seen: set[str] = set()
    ordered: list[str] = []
    sources = (
        (
            download_manifest.iter_variant_manifests("model", repo_id),
            download_manifest.iter_variant_markers("model", repo_id),
        )
        if hub_cache is None
        else (
            download_manifest.iter_variant_manifests(
                "model",
                repo_id,
                hub_cache = hub_cache,
            ),
            download_manifest.iter_variant_markers(
                "model",
                repo_id,
                hub_cache = hub_cache,
            ),
        )
    )
    for source in sources:
        for variant, _path in source:
            key = variant.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(variant)
    if not ordered:
        return None

    variants: list[GgufVariantInfo] = []
    has_vision = False
    for variant in ordered:
        manifest = (
            download_manifest.read_manifest("model", repo_id, variant)
            if hub_cache is None
            else download_manifest.read_manifest(
                "model",
                repo_id,
                variant,
                hub_cache = hub_cache,
            )
        )
        main_filename: Optional[str] = None
        size_bytes = 0
        companion_bytes = 0
        if manifest is not None:
            drafters = drafter_paths_in(file.path for file in manifest.expected_files)
            for expected in manifest.expected_files:
                if not is_gguf_filename(expected.path):
                    continue
                if expected.path in drafters:
                    # Downloaded with every variant (like mmproj) but not a
                    # selectable quant; count it so the shown download size
                    # matches what is fetched.
                    companion_bytes += max(0, int(expected.size or 0))
                    continue
                if is_mmproj_filename(expected.path):
                    has_vision = True
                    companion_bytes += max(0, int(expected.size or 0))
                    continue
                if main_filename is None:
                    main_filename = expected.path
                size_bytes += max(0, int(expected.size or 0))
        if main_filename is None:
            main_filename = f"{variant}.gguf"
        variants.append(
            GgufVariantInfo(
                filename = main_filename,
                quant = variant,
                size_bytes = size_bytes,
                download_size_bytes = size_bytes + companion_bytes,
            )
        )

    variants.sort(key = lambda variant: -variant.size_bytes)
    _apply_gguf_display_labels(variants)
    return variants, has_vision


def iter_snapshots_preferring_whole(
    repo_id: str,
    gguf_variant: Optional[str],
    root = None,
):
    """Cache snapshots newest first, but ones holding *gguf_variant* whole ahead of ones short a
    shard. The lister and the load both take the whole copy, so mtime order alone would read
    metadata out of a newer half download nothing will load.
    """
    ordered = list(iter_hf_cache_snapshots(repo_id, root = root))
    if not gguf_variant or len(ordered) < 2:
        return ordered
    from hub.utils.inventory_scan import complete_snapshot_variants

    whole, torn = [], []
    for snapshot in ordered:
        try:
            is_whole = gguf_variant in complete_snapshot_variants(str(snapshot))
        except Exception:
            is_whole = True
        (whole if is_whole else torn).append(snapshot)
    return whole + torn


def resolve_local_gguf_path(repo_id: str, gguf_variant: Optional[str]) -> Optional[str]:
    """Absolute path to the (shard-1) GGUF file for ``repo_id`` + ``gguf_variant``
    if it is already downloaded in the HF cache, else ``None``. Read-only — never
    triggers a download. Lets callers read header metadata before a load."""
    for snapshot in iter_snapshots_preferring_whole(repo_id, gguf_variant):
        variants, _ = list_local_gguf_variants(str(snapshot))
        for variant in variants:
            if gguf_variant is None or variant.quant == gguf_variant:
                candidate = snapshot / variant.filename
                if candidate.is_file():
                    return str(candidate)
    return None


def _ready_cached_variants(cached: tuple) -> tuple[list[GgufVariantInfo], bool, None]:
    """Cache result for a caller with nowhere to put readiness: drop the quants short a shard, but
    keep the whole list when none is complete so the folder still shows up to manage."""
    variants, has_vision, complete = cached
    whole = [v for v in variants if not v.quant or v.quant in complete]
    return whole or variants, has_vision, None


def list_gguf_variants(
    repo_id: str, hf_token: Optional[str] = None
) -> tuple[list[GgufVariantInfo], bool, Optional[list]]:
    from huggingface_hub import HfApi

    if _env_offline():
        cached = list_gguf_variants_from_hf_cache(repo_id)
        if cached is not None:
            return _ready_cached_variants(cached)

    try:
        info = HfApi(token = hf_token).model_info(
            repo_id,
            files_metadata = True,
            timeout = _GGUF_MODEL_INFO_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        if type(exc).__name__ in (
            "RepositoryNotFoundError",
            "GatedRepoError",
            "RevisionNotFoundError",
            "EntryNotFoundError",
        ):
            raise
        cached = list_gguf_variants_from_hf_cache(repo_id)
        if cached is not None:
            logger.warning(
                "HF API unreachable for %s (%s); using local cache snapshot.",
                repo_id,
                exc.__class__.__name__,
            )
            return _ready_cached_variants(cached)
        raise

    variants: list[GgufVariantInfo] = []
    has_vision = False
    quant_totals: dict[str, int] = {}
    quant_first_file: dict[str, str] = {}
    drafters = drafter_paths_in(
        name for s in info.siblings if isinstance(name := getattr(s, "rfilename", None), str)
    )

    for sibling in info.siblings:
        filename = getattr(sibling, "rfilename", None)
        if not isinstance(filename, str) or not is_gguf_filename(filename):
            continue
        if filename in drafters:
            continue
        if is_mmproj_filename(filename):
            has_vision = True
            continue
        quant = extract_quant_label(filename)
        if is_big_endian_gguf_path(filename, quant):
            continue
        quant_totals[quant] = quant_totals.get(quant, 0) + int(getattr(sibling, "size", 0) or 0)
        quant_first_file.setdefault(quant, filename)

    for quant, total_size in quant_totals.items():
        variants.append(
            GgufVariantInfo(
                filename = quant_first_file[quant],
                quant = quant,
                size_bytes = total_size,
            )
        )

    variants.sort(key = lambda variant: -variant.size_bytes)
    _apply_gguf_display_labels(variants)
    return variants, has_vision, list(info.siblings)


def _resolve_gguf_dir(path: Path) -> Optional[Path]:
    if path.is_dir():
        return path
    if path.is_file() and path.suffix.lower() == ".gguf":
        parent = path.parent
        if (
            (parent / "config.json").exists()
            or (parent / "adapter_config.json").exists()
            or (parent / "export_metadata.json").exists()
        ):
            return parent
    return None


def list_local_gguf_variants(
    directory: str, model_root: Optional[str] = None
) -> tuple[list[GgufVariantInfo], bool]:
    root = _resolve_gguf_dir(Path(directory))
    if root is None:
        return [], False
    from utils.models.model_config import (
        _is_local_mtp_drafter,
        _registered_custom_model_root,
    )

    custom_root = (
        Path(os.path.abspath(Path(model_root).expanduser()))
        if model_root is not None
        else _registered_custom_model_root(directory)
    )

    quant_totals: dict[str, int] = {}
    quant_first_file: dict[str, str] = {}
    has_vision = False

    # Same drafter-only reprieve as drafter_paths_in, but keyed on the local
    # predicate, which also resolves paths against a registered model root.
    scanned = [
        (file, file.relative_to(root).as_posix())
        for file in sorted(iter_gguf_files(root, recursive = True))
    ]
    drafters = {rel for file, rel in scanned if _is_local_mtp_drafter(file, custom_root, rel)}
    if not any(rel not in drafters and not is_mmproj_filename(rel) for _file, rel in scanned):
        drafters = {rel for rel in drafters if is_drafter_dir_path(rel)}

    for file, rel in scanned:
        if is_mmproj_filename(file.name):
            # A projector llama.cpp cannot open is not vision support.
            try:
                has_vision = has_vision or file.stat().st_size > 0
            except OSError:
                pass
            continue
        try:
            size = file.stat().st_size
        except OSError:
            size = 0
        if rel in drafters:
            continue
        quant = extract_quant_label(rel)
        if is_big_endian_gguf_path(rel, quant):
            continue
        quant_totals[quant] = quant_totals.get(quant, 0) + size
        quant_first_file.setdefault(quant, rel)

    variants = [
        GgufVariantInfo(
            filename = quant_first_file[quant],
            quant = quant,
            size_bytes = size,
        )
        for quant, size in quant_totals.items()
    ]
    variants.sort(key = lambda variant: -variant.size_bytes)
    _apply_gguf_display_labels(variants)
    return variants, has_vision
