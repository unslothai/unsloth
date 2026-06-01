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

_GGUF_SPLIT_RE = re.compile(r"-\d{3,}-of-\d{3,}", re.IGNORECASE)
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


def is_gguf_filename(filename: str) -> bool:
    return filename.lower().endswith(".gguf")


def iter_gguf_files(directory: Path, recursive: bool = False):
    if not directory.is_dir():
        return
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    for file in iterator:
        if file.is_file() and is_gguf_filename(file.name):
            yield file


def pick_best_gguf(filenames: list[str]) -> Optional[str]:
    gguf_files = [
        name
        for name in filenames
        if is_gguf_filename(name) and not is_mmproj_filename(name)
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
    return _GGUF_SPLIT_RE.sub("", basename.rsplit(".", 1)[0]).strip()


def extract_quant_token(filename: str) -> Optional[str]:
    stem = _gguf_stem(filename)
    match = _GGUF_QUANT_RE.search(stem)
    if not match and "/" in filename:
        parents = filename.rsplit("/", 1)[0]
        for segment in reversed(parents.split("/")):
            parent_match = _GGUF_QUANT_RE.search(segment)
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


def list_gguf_variants_from_hf_cache(
    repo_id: str,
) -> Optional[tuple[list[GgufVariantInfo], bool]]:
    for snapshot in iter_hf_cache_snapshots(repo_id):
        variants, has_vision = list_local_gguf_variants(str(snapshot))
        if variants or has_vision:
            return variants, has_vision
    return None


def list_partial_gguf_variants_from_state(
    repo_id: str,
) -> Optional[tuple[list[GgufVariantInfo], bool]]:
    """Reconstruct GGUF variants from download manifests/markers alone.

    Used when no completed snapshot exists (download cancelled or interrupted)
    and the HF API is unreachable (offline/gated/private). Each variant keyed
    on disk maps to one entry whose ``quant`` is the stored variant key so a
    resume passes the matching ``--variant`` back to the worker.
    """
    from hub.utils import download_manifest

    # Variant identity on disk is case-insensitive (state_dir._entry_key lowercases
    # it), so dedupe on the lowercased key. Manifests are read first, keeping their
    # original-casing label when a lowercased cancel marker names the same variant.
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
        if manifest is not None:
            for expected in manifest.expected_files:
                if not is_gguf_filename(expected.path):
                    continue
                if is_mmproj_filename(expected.path):
                    has_vision = True
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
            )
        )

    variants.sort(key = lambda variant: -variant.size_bytes)
    _apply_gguf_display_labels(variants)
    return variants, has_vision


def list_gguf_variants(
    repo_id: str,
    hf_token: Optional[str] = None,
) -> tuple[list[GgufVariantInfo], bool]:
    from huggingface_hub import HfApi

    if _env_offline():
        cached = list_gguf_variants_from_hf_cache(repo_id)
        if cached is not None:
            return cached

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
            return cached
        raise

    variants: list[GgufVariantInfo] = []
    has_vision = False
    quant_totals: dict[str, int] = {}
    quant_first_file: dict[str, str] = {}

    for sibling in info.siblings:
        filename = getattr(sibling, "rfilename", None)
        if not isinstance(filename, str) or not is_gguf_filename(filename):
            continue
        if is_mmproj_filename(filename):
            has_vision = True
            continue
        quant = extract_quant_label(filename)
        quant_totals[quant] = quant_totals.get(quant, 0) + int(
            getattr(sibling, "size", 0) or 0
        )
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
    return variants, has_vision


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
    directory: str,
) -> tuple[list[GgufVariantInfo], bool]:
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
        quant = extract_quant_label(rel)
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
