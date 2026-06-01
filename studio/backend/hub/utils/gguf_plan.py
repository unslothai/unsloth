# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from hub.utils.download_manifest import ExpectedFile
from hub.utils.gguf import extract_quant_label, is_gguf_filename, is_mmproj_filename


@dataclass(frozen = True)
class GgufVariantPlan:
    main_filenames: frozenset[str]
    target_filenames: tuple[str, ...]
    main_hashes: frozenset[str]
    required_hashes: frozenset[str]
    companion_hashes: frozenset[str]
    mmproj_filenames: frozenset[str]
    mmproj_hashes: frozenset[str]
    expected_files: tuple[ExpectedFile, ...]
    main_size_bytes: int
    download_size_bytes: int


def sibling_sha256(sibling) -> Optional[str]:
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, dict):
        value = lfs.get("sha256")
    else:
        value = getattr(lfs, "sha256", None)
    return value if isinstance(value, str) and value else None


def sibling_size(sibling) -> int:
    size = getattr(sibling, "size", 0) or 0
    try:
        return int(size)
    except (TypeError, ValueError):
        return 0


def expected_file_from_sibling(sibling) -> Optional[ExpectedFile]:
    name = getattr(sibling, "rfilename", None)
    if not isinstance(name, str):
        return None
    return ExpectedFile(
        path = name,
        size = sibling_size(sibling),
        sha256 = sibling_sha256(sibling),
    )


def is_companion_gguf_path(path: str) -> bool:
    return is_gguf_filename(path) and is_mmproj_filename(path)


def is_main_gguf_variant_path(path: str, variant: str) -> bool:
    return (
        is_gguf_filename(path)
        and not is_mmproj_filename(path)
        and extract_quant_label(path).lower() == variant.lower()
    )


def mmproj_siblings(siblings: Sequence) -> list:
    return [
        s
        for s in siblings
        if isinstance(getattr(s, "rfilename", None), str)
        and is_companion_gguf_path(getattr(s, "rfilename"))
    ]


def preferred_mmproj_sibling(siblings: Sequence) -> Optional[object]:
    candidates = mmproj_siblings(siblings)
    if not candidates:
        return None
    return next(
        (
            s for s in candidates
            if extract_quant_label(getattr(s, "rfilename")).upper() == "F16"
        ),
        candidates[0],
    )


def build_gguf_variant_plans(siblings: Sequence) -> dict[str, GgufVariantPlan]:
    main: dict[str, list] = {}
    all_mmproj = mmproj_siblings(siblings)
    all_mmproj_filenames = frozenset(
        getattr(s, "rfilename") for s in all_mmproj
        if isinstance(getattr(s, "rfilename", None), str)
    )
    all_mmproj_hashes = frozenset(
        h for h in (sibling_sha256(s) for s in all_mmproj) if h
    )
    companion = preferred_mmproj_sibling(siblings)
    companion_expected = (
        expected_file_from_sibling(companion) if companion is not None else None
    )

    for sibling in siblings:
        name = getattr(sibling, "rfilename", None)
        if not isinstance(name, str) or not is_gguf_filename(name):
            continue
        if is_mmproj_filename(name):
            continue
        quant = extract_quant_label(name).lower()
        main.setdefault(quant, []).append(sibling)

    plans: dict[str, GgufVariantPlan] = {}
    for quant, target_main_siblings in main.items():
        main_expected = tuple(
            file for sibling in target_main_siblings
            if (file := expected_file_from_sibling(sibling)) is not None
        )
        expected_files = (
            (*main_expected, companion_expected)
            if companion_expected is not None
            else main_expected
        )
        plans[quant] = plan_from_expected_files(
            quant,
            expected_files,
            all_mmproj_filenames = all_mmproj_filenames,
            all_mmproj_hashes = all_mmproj_hashes,
        )
    return plans


def plan_from_expected_files(
    variant: str,
    expected_files: Sequence[ExpectedFile],
    *,
    all_mmproj_filenames: frozenset[str] | None = None,
    all_mmproj_hashes: frozenset[str] | None = None,
) -> GgufVariantPlan:
    expected = tuple(expected_files)
    main_files = tuple(
        file for file in expected
        if is_main_gguf_variant_path(file.path, variant)
    )
    companion_files = tuple(
        file for file in expected
        if is_companion_gguf_path(file.path)
    )
    main_hashes = frozenset(file.sha256 for file in main_files if file.sha256)
    companion_hashes = frozenset(
        file.sha256 for file in companion_files if file.sha256
    )
    required_hashes = frozenset(file.sha256 for file in expected if file.sha256)
    main_size = sum(max(0, int(file.size or 0)) for file in main_files)
    download_size = sum(max(0, int(file.size or 0)) for file in expected)
    return GgufVariantPlan(
        main_filenames = frozenset(file.path for file in main_files),
        target_filenames = tuple(file.path for file in expected),
        main_hashes = main_hashes,
        required_hashes = required_hashes,
        companion_hashes = companion_hashes,
        mmproj_filenames = (
            all_mmproj_filenames
            if all_mmproj_filenames is not None
            else frozenset(file.path for file in companion_files)
        ),
        mmproj_hashes = (
            all_mmproj_hashes
            if all_mmproj_hashes is not None
            else companion_hashes
        ),
        expected_files = expected,
        main_size_bytes = main_size,
        download_size_bytes = download_size,
    )
