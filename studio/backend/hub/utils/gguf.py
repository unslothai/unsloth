# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF filename helpers. Quantization variants are derived from filenames, not parsed from binary GGUF headers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


def is_mtp_drafter_path(path: str) -> bool:
    """True for a separate-file MTP drafter (speculative head), a companion to
    the main model rather than a selectable quant.

    Covers the repo-root ``mtp-*.gguf`` (the Q8_0 copy unsloth ships for
    llama.cpp ``-hf`` auto-discovery) and the ``MTP/`` subdir copies (Gemma 4).
    Repos that bake the head into the main GGUF (Qwen) have no such file, so
    this is False for them. Must be excluded from main-model selection
    everywhere mmproj is.

    CANONICAL COPY. Layering keeps two mirrors that must change in lockstep:
    utils/models/model_config.py ``_is_mtp_drafter`` (utils cannot import
    hub) and core/inference/llama_cpp.py ``_is_companion_gguf_path`` (core
    avoids hub imports; bundles the mmproj check).
    """
    p = path.lower()
    if not p.endswith(".gguf"):
        return False
    name = p.rsplit("/", 1)[-1]
    return name.startswith("mtp-") or "/mtp/" in f"/{p}"


def is_gguf_filename(filename: str) -> bool:
    return filename.lower().endswith(".gguf")


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
    gguf_files = [
        name
        for name in filenames
        if is_gguf_filename(name)
        and not is_mmproj_filename(name)
        and not is_mtp_drafter_path(name)
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
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ) or os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def iter_hf_cache_snapshots(repo_id: str):
    from hub.utils.hf_cache_state import iter_repo_cache_dirs

    snapshots: list[Path] = []
    for repo_dir in iter_repo_cache_dirs("model", repo_id):
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        try:
            snapshots.extend(snap for snap in snapshots_dir.iterdir() if snap.is_dir())
        except OSError:
            continue

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    snapshots.sort(key = _mtime, reverse = True)
    yield from snapshots


def list_empty_gguf_variant_dirs(repo_id: str) -> set[str]:
    """Quant labels present only as an EMPTY snapshot ``<quant>/`` folder (an
    interrupted split download); a quant with shards in any snapshot is excluded."""
    empty: dict[str, str] = {}
    nonempty: set[str] = set()
    for snapshot in iter_hf_cache_snapshots(repo_id):
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
    repo_id: str,
    hf_token: Optional[str] = None,
    *,
    offline: bool = False,
) -> Optional[tuple[list[GgufVariantInfo], bool]]:
    verify_sizes = not (offline or _env_offline())
    any_vision = False
    for snapshot in iter_hf_cache_snapshots(repo_id):
        variants, has_vision = list_local_gguf_variants(str(snapshot))
        any_vision = any_vision or has_vision
        if variants:
            from core.inference.llama_cpp import _snapshot_dir_of, cached_gguf_for_load
            variants = [
                variant
                for variant in variants
                if (
                    cached_path := cached_gguf_for_load(
                        repo_id,
                        variant.quant,
                        verify_sizes = verify_sizes,
                        hf_token = hf_token,
                    )
                )
                and _snapshot_dir_of(cached_path) == snapshot
            ]
        if variants:
            return variants, any_vision
    if any_vision:
        return [], True
    return None


def list_partial_gguf_variants_from_state(
    repo_id: str,
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
    for source in (
        download_manifest.iter_variant_manifests("model", repo_id),
        download_manifest.iter_variant_markers("model", repo_id),
    ):
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
        manifest = download_manifest.read_manifest("model", repo_id, variant)
        main_filename: Optional[str] = None
        size_bytes = 0
        companion_bytes = 0
        if manifest is not None:
            for expected in manifest.expected_files:
                if not is_gguf_filename(expected.path):
                    continue
                if is_mtp_drafter_path(expected.path):
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


def resolve_local_gguf_path(repo_id: str, gguf_variant: Optional[str]) -> Optional[str]:
    """Absolute path to the (shard-1) GGUF file for ``repo_id`` + ``gguf_variant``
    if it is already downloaded in the HF cache, else ``None``. Read-only — never
    triggers a download. Lets callers read header metadata before a load."""
    for snapshot in iter_hf_cache_snapshots(repo_id):
        variants, _ = list_local_gguf_variants(str(snapshot))
        for variant in variants:
            if gguf_variant is None or variant.quant == gguf_variant:
                candidate = snapshot / variant.filename
                if candidate.is_file():
                    return str(candidate)
    return None


def list_gguf_variants(
    repo_id: str, hf_token: Optional[str] = None
) -> tuple[list[GgufVariantInfo], bool, Optional[list]]:
    from huggingface_hub import HfApi

    if _env_offline():
        cached = list_gguf_variants_from_hf_cache(repo_id, hf_token)
        if cached is not None:
            return (*cached, None)

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
        cached = list_gguf_variants_from_hf_cache(repo_id, hf_token)
        if cached is not None:
            logger.warning(
                "HF API unreachable for %s (%s); using local cache snapshot.",
                repo_id,
                exc.__class__.__name__,
            )
            return (*cached, None)
        raise

    variants: list[GgufVariantInfo] = []
    has_vision = False
    quant_totals: dict[str, int] = {}
    quant_first_file: dict[str, str] = {}

    for sibling in info.siblings:
        filename = getattr(sibling, "rfilename", None)
        if not isinstance(filename, str) or not is_gguf_filename(filename):
            continue
        if is_mtp_drafter_path(filename):
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


def list_local_gguf_variants(directory: str) -> tuple[list[GgufVariantInfo], bool]:
    root = _resolve_gguf_dir(Path(directory))
    if root is None:
        return [], False

    quant_totals: dict[str, int] = {}
    quant_first_file: dict[str, str] = {}
    has_vision = False

    for file in sorted(iter_gguf_files(root, recursive = True)):
        if is_mmproj_filename(file.name):
            has_vision = True
            continue
        try:
            size = file.stat().st_size
        except OSError:
            size = 0
        rel = file.relative_to(root).as_posix()
        if is_mtp_drafter_path(rel):
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
