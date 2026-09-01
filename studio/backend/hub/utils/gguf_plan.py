# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from hub.utils.download_manifest import ExpectedFile
from hub.utils.gguf import (
    bare_quant_alias,
    extract_quant_label,
    gguf_variant_family,
    gguf_variant_key,
    is_big_endian_gguf_path,
    drop_shadowed_appledouble_siblings,
    is_gguf_filename,
    is_imatrix_filename,
    is_mmproj_filename,
    is_mtp_drafter_path,
)


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
    if isinstance(value, str) and value:
        return value
    blob_id = getattr(sibling, "blob_id", None)
    return blob_id if isinstance(blob_id, str) and blob_id else None


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
    """Companion (non-main) GGUF downloaded alongside a variant: the vision
    mmproj or the separate MTP drafter (Gemma 4)."""
    return is_gguf_filename(path) and (is_mmproj_filename(path) or is_mtp_drafter_path(path))


def is_main_gguf_variant_path(path: str, variant: str) -> bool:
    """Whether *path* is one of *variant*'s own weight files.

    Keyed on :func:`gguf_variant_key`, in lockstep with the listers: a row built under
    one identity and matched under another produces a variant that can be shown but
    not downloaded.
    """
    return (
        is_gguf_filename(path)
        and not is_mmproj_filename(path)
        and not is_mtp_drafter_path(path)
        and not is_imatrix_filename(path)
        # The endian predicate reads a quant TOKEN, so hand it the label: given the qualified key it
        # cannot see a parent-only quant and drops the file, leaving the plan with no main files.
        and not is_big_endian_gguf_path(path, extract_quant_label(path))
        and gguf_variant_key(path).lower() == variant.lower()
    )


def _gguf_rfilename(sibling) -> Optional[str]:
    """The sibling's rfilename when it is a GGUF, else None."""
    name = getattr(sibling, "rfilename", None)
    if isinstance(name, str) and is_gguf_filename(name):
        return name
    return None


def mmproj_siblings(siblings: Sequence) -> list:
    return [s for s in siblings if (name := _gguf_rfilename(s)) and is_mmproj_filename(name)]


def preferred_mmproj_sibling(siblings: Sequence) -> Optional[object]:
    candidates = mmproj_siblings(siblings)
    if not candidates:
        return None
    return next(
        (s for s in candidates if extract_quant_label(getattr(s, "rfilename")).upper() == "F16"),
        candidates[0],
    )


def preferred_mtp_sibling(siblings: Sequence) -> Optional[object]:
    """The separate MTP drafter to fetch with every variant: the repo-root
    ``mtp-*.gguf`` copy unsloth ships for llama.cpp ``-hf`` auto-discovery
    (Gemma 4). Same pick as the loader's drafter resolution (root-level
    ``mtp-`` prefix, first in sort order) so download and load resolve the same
    file; the higher-precision ``MTP/`` subdir copies are for explicit
    selection and are not auto-fetched. None for repos with the head baked into
    the main GGUF (Qwen)."""
    # Root-level only: the MTP/ subdir copies now share the mtp- prefix too.
    candidates = sorted(
        (
            s
            for s in siblings
            if (name := _gguf_rfilename(s)) and "/" not in name and name.lower().startswith("mtp-")
        ),
        key = lambda s: getattr(s, "rfilename"),
    )
    return candidates[0] if candidates else None


def preferred_dflash_sibling(
    siblings: Sequence,
    weight_name: Optional[str] = None,
    other_weight_names: Sequence[str] = (),
) -> Optional[object]:
    """The DFlash sidecar to fetch alongside ``weight_name``.

    Root level only, like preferred_mtp_sibling: detect_dflash_file never offers a
    nested ``quants/dflash-*.gguf``, and a listing cannot read a header, so matching
    the basename would plan a whole ordinary weight nothing could reject in time.

    Ordered by dflash_repo_preference_key, as the download, snapshot reuse and offline
    cache are, so the manifest promises the file the loader launches.
    """
    from utils.models.drafters import dflash_repo_preference_key

    candidates = [
        s
        for s in siblings
        if (name := _gguf_rfilename(s)) and "/" not in name and name.lower().startswith("dflash-")
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key = lambda s: dflash_repo_preference_key(
            getattr(s, "rfilename"), weight_name, other_weight_names
        ),
    )


def dflash_plan_files(
    siblings: Sequence,
    weight_name: Optional[str] = None,
    other_weight_names: Sequence[str] = (),
    *,
    max_bytes: int = 0,
) -> tuple[ExpectedFile, ...]:
    """Every shard of the DFlash sidecar to plan alongside ``weight_name``, or ().

    Whole shard family, not the ranked file alone: the loader refuses an incomplete
    split set, so planning shard 1 reports the variant complete and then loses DFlash.
    A half-published family is dropped for the same reason.

    Bounded by ``max_bytes``, the variant's own weights. ``dflash-`` is a prefix real
    weights carry (Lucebox/Qwen3.6-27B-DFlash-GGUF) and a listing cannot read the
    ``general.architecture`` the loader rejects them by, but a drafter is a few layers
    of its target and cannot outweigh it. An unknown size stays out.

    Both rules filter BEFORE the ranking, so an oversized or half-published name at the
    top steps aside for a usable sidecar behind it.
    """
    from utils.models.drafters import dflash_repo_preference_key, split_listing_is_complete

    families: dict[str, list[ExpectedFile]] = {}
    for sibling in siblings:
        name = _gguf_rfilename(sibling)
        if not name or "/" in name or not name.lower().startswith("dflash-"):
            continue
        file = expected_file_from_sibling(sibling)
        if file is not None:
            families.setdefault(gguf_variant_family(name), []).append(file)

    eligible: dict[str, tuple[ExpectedFile, ...]] = {}
    for family, files in families.items():
        shards = tuple(sorted(files, key = lambda file: file.path))
        if not split_listing_is_complete([f.path for f in shards], shards[0].path):
            continue
        total = sum(max(0, int(file.size or 0)) for file in shards)
        if not total or max_bytes <= 0 or total >= max_bytes:
            continue
        eligible[family] = shards
    if not eligible:
        return ()
    best = min(
        eligible,
        key = lambda family: dflash_repo_preference_key(
            eligible[family][0].path, weight_name, other_weight_names
        ),
    )
    return eligible[best]


def build_gguf_variant_plans(siblings: Sequence) -> dict[str, GgufVariantPlan]:
    # Family grouping keeps the family holding the lexicographically first name, which is the "._"
    # one, so the plan fetched the sidecar and marked the variant complete, leaving header-based local
    # discovery no main GGUF to load.
    siblings = drop_shadowed_appledouble_siblings(list(siblings))
    main: dict[str, list] = {}
    all_mmproj = mmproj_siblings(siblings)
    all_mmproj_filenames = frozenset(
        getattr(s, "rfilename")
        for s in all_mmproj
        if isinstance(getattr(s, "rfilename", None), str)
    )
    all_mmproj_hashes = frozenset(h for h in (sibling_sha256(s) for s in all_mmproj) if h)
    companion = preferred_mmproj_sibling(siblings)
    companion_expected = expected_file_from_sibling(companion) if companion is not None else None
    mtp_sibling = preferred_mtp_sibling(siblings)
    mtp_expected = expected_file_from_sibling(mtp_sibling) if mtp_sibling is not None else None
    companions_expected = tuple(
        file for file in (companion_expected, mtp_expected) if file is not None
    )

    for sibling in siblings:
        name = _gguf_rfilename(sibling)
        if name is None:
            continue
        # Keep companions out of the quant grouping so a drafter never lands in a variant's main files: the
        # root mtp-*.gguf carries a quant label.
        # An imatrix leaves entirely rather than joining companions_expected: no variant needs llama-
        # quantize's calibration data downloaded.
        if is_mmproj_filename(name) or is_mtp_drafter_path(name) or is_imatrix_filename(name):
            continue
        quant = gguf_variant_key(name).lower()
        # The endian predicate reads a quant TOKEN, so a qualified key would make it misread the path and
        # drop the file from every plan.
        if is_big_endian_gguf_path(name, extract_quant_label(name)):
            continue
        main.setdefault(quant, []).append(sibling)

    plans: dict[str, GgufVariantPlan] = {}
    # Every weight in the listing, so the ranking can tell a sidecar naming a neighbouring family from
    # one naming this variant's.
    all_weight_names = [
        name.rsplit("/", 1)[-1]
        for quant_siblings in main.values()
        for sibling in quant_siblings
        if (name := _gguf_rfilename(sibling))
    ]
    for quant, target_main_siblings in main.items():
        main_expected = tuple(
            file
            for sibling in target_main_siblings
            if (file := expected_file_from_sibling(sibling)) is not None
        )
        # Per variant, unlike mmproj and the MTP drafter: ranked against the weight being fetched, and
        # against the family plan_from_expected_files KEEPS, or a two-family variant key pairs the wrong
        # sidecar.
        kept_main = _one_shard_family(main_expected)
        target_weight_name = (
            min(file.path for file in kept_main).rsplit("/", 1)[-1] if kept_main else None
        )
        dflash_expected = dflash_plan_files(
            siblings,
            target_weight_name,
            [n for n in all_weight_names if n != target_weight_name],
            max_bytes = sum(max(0, int(file.size or 0)) for file in kept_main),
        )
        expected_files = (*main_expected, *companions_expected, *dflash_expected)
        plans[quant] = plan_from_expected_files(
            quant,
            expected_files,
            all_mmproj_filenames = all_mmproj_filenames,
            all_mmproj_hashes = all_mmproj_hashes,
        )
    return plans


def plan_for_variant(plans: dict[str, GgufVariantPlan], variant: str) -> Optional[GgufVariantPlan]:
    """The plan for *variant*, accepting a bare quant when exactly one plan carries it.

    A repo that files every variant under one shared container (``weights/model-Q4_K_M.gguf``)
    qualifies every key, because the key is a pure function of the path and cannot know that the
    directory disambiguates nothing. Every stored pin and every explicit ``repo:Q4_K_M`` then
    missed the plan map and the worker exited with "No GGUF shards matching variant".

    Resolved at LOOKUP rather than by aliasing the map, so the key stays a pure function of the
    path -- the remote listing and a partial cache scan have to agree on it -- and the advertised
    rows stay one per checkpoint. Only when the bare name is UNAMBIGUOUS: a repo that really does
    hold several checkpoints at one quant gets no fallback, because there the bare name genuinely
    does not name one of them.
    """
    wanted = (variant or "").strip().lower()
    if not wanted:
        return None
    exact = plans.get(wanted)
    if exact is not None:
        return exact
    # PATH-qualified keys only, not is_qualified_gguf_variant_key: an H3 root stem's bare quant names
    # both partitions, and picking either would load a different task.
    matches = [key for key in plans if "/" in key and bare_quant_alias(key).lower() == wanted]
    return plans[matches[0]] if len(matches) == 1 else None


def _one_shard_family(main_files: Sequence[ExpectedFile]) -> tuple[ExpectedFile, ...]:
    """Narrow a variant's weight files to the single shard family a load would read.

    A repo can ship one quant twice under names that share a variant key -- the same
    BF16 as ``QwQ-32B-BF16-*`` and ``QwQ-32B.BF16-*``, or one Q6_K under both ``Q6_K/``
    and ``<model>-Q6_K/``. Fetching both doubles the download and leaves the variant
    permanently short of its expected bytes, because the loader only ever opens one.
    Keep the family holding the lexicographically first file, the shard the lister
    advertises and the loader opens. A genuinely split GGUF is one family, so all of
    its shards survive this untouched.
    """
    if len(main_files) < 2:
        return tuple(main_files)
    families: dict[str, list[ExpectedFile]] = {}
    for file in main_files:
        families.setdefault(gguf_variant_family(file.path), []).append(file)
    if len(families) < 2:
        return tuple(main_files)
    chosen = min(families.values(), key = lambda group: min(file.path for file in group))
    return tuple(chosen)


def plan_from_expected_files(
    variant: str,
    expected_files: Sequence[ExpectedFile],
    *,
    all_mmproj_filenames: frozenset[str] | None = None,
    all_mmproj_hashes: frozenset[str] | None = None,
) -> GgufVariantPlan:
    expected = tuple(expected_files)
    all_main = tuple(file for file in expected if is_main_gguf_variant_path(file.path, variant))
    main_files = _one_shard_family(all_main)
    # A discarded family has to leave the plan ENTIRELY: target_filenames, required_hashes and
    # download_size_bytes are what the worker fetches, so leaving the copy there downloaded it, then
    # reclaim deleted it as not-ours (absent from main_hashes) and the job fetched it again.
    kept = {file.path for file in main_files}
    expected = tuple(file for file in expected if file not in all_main or file.path in kept)
    companion_files = tuple(file for file in expected if is_companion_gguf_path(file.path))
    # companion_files also holds the MTP drafter, so keep an mmproj-only view for the manifest-resume fallback.
    mmproj_files = tuple(file for file in companion_files if is_mmproj_filename(file.path))
    main_hashes = frozenset(file.sha256 for file in main_files if file.sha256)
    companion_hashes = frozenset(file.sha256 for file in companion_files if file.sha256)
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
            else frozenset(file.path for file in mmproj_files)
        ),
        mmproj_hashes = (
            all_mmproj_hashes
            if all_mmproj_hashes is not None
            else frozenset(file.sha256 for file in mmproj_files if file.sha256)
        ),
        expected_files = expected,
        main_size_bytes = main_size,
        download_size_bytes = download_size,
    )
