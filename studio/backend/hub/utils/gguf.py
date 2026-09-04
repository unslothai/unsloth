# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF filename helpers. Quantization variants are derived from filenames, not parsed from binary GGUF headers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from loggers import get_logger
from utils.paths.path_utils import (
    drop_shadowed_appledouble_names as _drop_shadowed_appledouble_names,
    file_contents_available_locally,
    has_appledouble_magic,
    is_appledouble_metadata,
)
from utils.models.gguf_metadata import mmproj_accepts_image

logger = get_logger(__name__)
_GGUF_MODEL_INFO_TIMEOUT_SECONDS = 5.0


@dataclass
class GgufVariantInfo:
    filename: str
    quant: str
    size_bytes: int
    display_label: Optional[str] = None
    download_size_bytes: int = 0
    shard_count: int = 0


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
_GGUF_CANONICAL_SPLIT_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$",
    re.IGNORECASE,
)
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


# Anchored at an END of the stem, never a substring, for the reason is_mtp_drafter_path documents:
# a name that merely contains the word (Qwen3-Imatrix-Tuned-Q4_K_M.gguf) is a real model, while
# every published imatrix leads or closes with it.
_IMATRIX_TOKEN_RE = re.compile(r"^imatrix(?:[._\-]|$)|[._\-]imatrix$", re.IGNORECASE)


def is_imatrix_filename(path: str) -> bool:
    """True for a calibration imatrix: ``imatrix.gguf``, ``imatrix_unsloth.gguf``,
    ``imatrix_unsloth.dat``, ``<model>-imatrix.gguf``, ``<model>.imatrix``.

    It holds activation statistics for llama-quantize, not a model, and llama-server
    never opens one, so it is neither a selectable variant nor a companion to fetch:
    unlike mmproj and the MTP drafter it is excluded outright, everywhere they are.
    Most repos spell it .dat or .gguf_file, which no GGUF listing sees; the ones that
    publish imatrix_unsloth.gguf (unsloth/Qwen3.8-27B-GGUF) put a valid GGUF container
    carrying no model in front of the variant menu, download and loader.

    CANONICAL COPY. Two mirrors must change in lockstep:
    utils/models/model_config.py ``_is_imatrix_path`` (utils cannot import hub) and
    core/inference/llama_cpp.py ``_is_companion_gguf_path`` (core avoids hub imports).
    """
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return bool(_IMATRIX_TOKEN_RE.search(stem)) or name.lower().endswith(".imatrix")


# dspark and dflash are the same DeepSeek V4 Flash drafter: the folder it ships in and the architecture it reports.
_DRAFTER_KINDS = ("mtp", "dspark", "dflash")

# Directories only: mtp/ and dspark/ are always a publisher's companion folder, while dflash/ is a
# family name a user picks for real weights.
_DRAFTER_DIR_KINDS = ("mtp", "dspark")


def is_mtp_drafter_path(path: str) -> bool:
    """True for a separate-file drafter, a companion to the main model rather than
    a selectable quant: the repo-root ``mtp-*.gguf`` (the Q8_0 copy unsloth ships
    for llama.cpp ``-hf`` auto-discovery), the ``MTP/`` subdir copies (Gemma 4)
    and the ``dspark/`` drafters (DeepSeek V4 Flash). Repos that bake the head
    into the main GGUF (Qwen) have no such file, so this is False for them. Must
    be excluded from main-model selection everywhere mmproj is.

    Matched by basename prefix, or by an exact parent dir for
    ``_DRAFTER_DIR_KINDS``; never a substring, since the kind names double as
    family names, so ``Qwen3.6-27B-MTP-Q4_K_M.gguf`` and
    ``Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf`` ARE the model.

    CANONICAL COPY. Two mirrors must change in lockstep:
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


def is_reclaimable_drafter_path(path: str) -> bool:
    """Drafters a repo's last-variant delete may reclaim: MTP, fetched with every
    variant, and DSpark, fetched on opt-in. Both are useless once no main GGUF is
    left, and companion filtering hides them from the variant menu, so leaving one
    behind is an invisible allocation (DSpark is ~11 GB). DFlash is excluded even
    though Auto now launches it: the name doubles as a family a user picks for
    real weights, whole repos publish nothing but root-level ``dflash-*.gguf``
    (Lucebox/Qwen3.6-27B-DFlash-GGUF), and the two outcomes are not symmetric.
    Reclaiming wrongly destroys weights a user chose; not reclaiming leaves
    ~1.5 GiB, an order of magnitude under the DSpark case this rule was written
    for. Locked by test_deleting_the_last_variant_keeps_a_dflash_weight."""
    p = path.replace("\\", "/").lower()
    if not p.endswith(".gguf"):
        return False
    parts = [segment for segment in p.split("/") if segment]
    name, parents = parts[-1], parts[:-1]
    return any(name.startswith(f"{kind}-") for kind in _DRAFTER_DIR_KINDS) or any(
        kind in parents for kind in _DRAFTER_DIR_KINDS
    )


def _appledouble_subject_key(name: str) -> object:
    """What a listing entry has to share with a ``._`` name for it to be that name's subject.

    Only a GGUF has shards to belong to, and a split quant's companions need not all be
    shadowed; every other suffix is its own subject.
    """
    return ("gguf", gguf_variant_family(name)) if is_gguf_filename(name) else ("name", name)


def drop_shadowed_appledouble_names(files: list[str]) -> list[str]:
    """Shadow pairing keyed by GGUF shard family."""
    return _drop_shadowed_appledouble_names(files, subject_key = _appledouble_subject_key)


def drop_shadowed_appledouble_siblings(siblings: Sequence) -> list:
    """Shadow pairing for callers holding sizes and hashes beside the name."""
    names = [getattr(s, "rfilename", None) for s in siblings]
    kept = set(drop_shadowed_appledouble_names([n for n in names if isinstance(n, str)]))
    return [s for s, n in zip(siblings, names) if not isinstance(n, str) or n in kept]


def remove_appledouble_sidecar(path: Path) -> int:
    """Remove the sidecar beside *path* if it really is one; return the bytes freed.

    Now that discovery rejects sidecars, nothing else would ever name one, and metadata would
    outlive the weights it describes. Identified positively because this unlinks a file nobody
    asked for: a user's own ``._model.gguf`` is a real model, and deciding by name would delete
    weights. Best effort -- a sidecar that resists removal is a few KB, never a failed delete.
    """
    try:
        sidecar = Path(path).with_name(f"._{Path(path).name}")
        if not has_appledouble_magic(sidecar):
            return 0
        if sidecar.is_symlink() or sidecar.exists():
            freed = sidecar.stat().st_size
            sidecar.unlink()
            return freed
    except (OSError, ValueError):
        pass
    return 0


def is_gguf_filename(filename: str) -> bool:
    return filename.lower().endswith(".gguf")


# Every repo that bundles H3's denoisers with its companion models: the Unsloth mirror carries the
# Qwen3-VL encoder quants beside the denoisers, so listing one repo would aggregate encoder GGUFs
# as selectable transformer quants.
_H3_BUNDLE_REPOS = frozenset({"leejet/minimax-h3-gguf", "unsloth/minimax-h3-gguf"})


def is_h3_bundle_repo(repo_id: str) -> bool:
    return repo_id.strip().lower() in _H3_BUNDLE_REPOS


# Both released denoiser partitions are valid picks and which one is picked IS the task. Kept in
# step with validate_h3_transformer_filename in core/inference/video_minimax_h3.py, which the load
# enforces: listing only FL2VA hid every published Ref2VA quant from the picker.
_H3_DENOISER_PARTITIONS = ("minimax_h3_fl2va", "minimax_h3_ref2va")


def _is_selectable_repo_gguf(repo_id: str, filename: str) -> bool:
    """Hide auxiliary GGUFs from repos that bundle several model roles."""
    if is_h3_bundle_repo(repo_id):
        name = Path(filename).name.lower()
        return name.startswith(_H3_DENOISER_PARTITIONS) and name.endswith(".gguf")
    return True


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
                    path = Path(dirpath) / name
                    if is_appledouble_metadata(path):
                        continue
                    yield path
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
                if is_appledouble_metadata(file):
                    continue
                yield file
        except OSError:
            continue


def pick_best_gguf(filenames: list[str]) -> Optional[str]:
    gguf_files = [
        name
        # The preference loop returns the first matching name, and "._" sorts ahead of it.
        for name in drop_shadowed_appledouble_names(list(filenames))
        if is_gguf_filename(name)
        and not is_mmproj_filename(name)
        and not is_mtp_drafter_path(name)
        and not is_imatrix_filename(name)
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


def _quant_search_stem(filename: str) -> str:
    """The text a quant token is looked for in: the basename, less any shard suffix.

    Unlike :func:`_gguf_stem` this keeps everything after the last dot. The extension is not in
    the way, since the token is matched rather than the whole string, while cutting at the dot
    truncates a bare ``IQ4_XS-3.53bpw`` to ``IQ4_XS-3``. Bare names do arrive here: a ``<quant>/``
    folder and a stored variant string both come without ``.gguf``.
    """
    return _GGUF_SPLIT_SUFFIX_RE.sub("", filename.rsplit("/", 1)[-1]).strip()


def _locate_quant_match(filename: str) -> tuple[Optional[re.Match], str]:
    """The quant match naming *filename*, with the text it was found in.

    The basename decides; only when it names no quant do parent directories, nearest first.
    Callers that need what trails the token (the bpw modifier) need that text too, so this is
    the one place the search order lives.
    """
    stem = _quant_search_stem(filename)
    match = _select_quant_match(stem)
    if match:
        return match, stem
    if "/" in filename:
        for segment in reversed(filename.rsplit("/", 1)[0].split("/")):
            parent_match = _select_quant_match(segment)
            if parent_match:
                return parent_match, segment
    return None, ""


def extract_quant_token(filename: str) -> Optional[str]:
    match, _ = _locate_quant_match(filename)
    if match:
        prefix = match.group(1) or ""
        return f"{prefix}{match.group(2)}"
    return None


# Two builds of one base quant at different bpw are two checkpoints (byteshape ships IQ4_XS at
# 3.53, 3.97 and 4.19), so the variant KEY has to keep the modifier: without it the advertised
# name 404s.
# match, never search: only a modifier IMMEDIATELY after the quant qualifies it, so
# flux1-dev-Q8_0-fp32-08.577bpw keeps the bare Q8_0 rather than borrowing a number describing
# something else.
_GGUF_BPW_SUFFIX_RE = re.compile(r"-[0-9]+(?:\.[0-9]+)?bpw", re.IGNORECASE)

# The same modifier ending a name of its own, used only for the basename under a quant DIRECTORY
# (Q6_K/model-3.5bpw.gguf), where the token is in the parent and the number is all the file has to
# identify the build.
_GGUF_BPW_TRAILING_RE = re.compile(r"-[0-9]+(?:\.[0-9]+)?bpw(?=\.[A-Za-z0-9]+$|$)", re.IGNORECASE)


def quant_token_with_bpw(filename: str) -> Optional[str]:
    """:func:`extract_quant_token` with the bpw modifier that trails it, when there is one.

    The spelling that identifies a checkpoint. The bare token does not: a repo publishing one
    base quant at several bit widths gives every one of them the same token, and the modifier is
    the only thing telling them apart.

    Where the modifier is found follows where the token was. Named by the basename, only a
    modifier IMMEDIATELY after it counts, so ``flux1-dev-Q8_0-fp32-08.577bpw`` keeps the bare
    ``Q8_0`` rather than borrowing a number describing something else. Named by a parent
    directory, the modifier may sit against the token there (``IQ4_XS-3.53bpw/model.gguf``) or
    end the basename instead (``Q6_K/model-3.5bpw.gguf``), and both are this build's own.

    Returns the bare token unchanged for every repo without one, which is nearly all of them,
    and ``None`` when nothing in the path names a quant.
    """
    path = filename.replace("\\", "/")
    match, text = _locate_quant_match(path)
    if match is None:
        return None
    token = f"{match.group(1) or ''}{match.group(2)}"
    adjacent = _GGUF_BPW_SUFFIX_RE.match(text[match.end() :])
    if adjacent:
        return f"{token}{adjacent.group(0)}"
    stem = _quant_search_stem(path)
    if text != stem:
        trailing = _GGUF_BPW_TRAILING_RE.search(stem)
        if trailing:
            return f"{token}{trailing.group(0)}"
    return token


def _unknown_gguf_variant_key(filename: str) -> str:
    stem = _gguf_stem(filename)
    if "/" not in filename:
        return stem or "gguf"
    parents = filename.rsplit("/", 1)[0].strip("/")
    return f"{parents}/{stem}" if parents and stem else stem or "gguf"


def gguf_variant_family(filename: str) -> str:
    """The shard family *filename* belongs to: its directory plus its shard-stripped name.

    The unit a row and a download plan describe. Every shard of one split GGUF shares
    a family, which is why summing sizes within a family is right; two files that do
    NOT share one are two different checkpoints, so summing across them is not.
    """
    return _unknown_gguf_variant_key(filename)


def complete_gguf_shard_count(filenames: Sequence[str], first_filename: str) -> int:
    """Return a complete canonical split's part count, otherwise zero."""
    normalized_first = first_filename.replace("\\", "/")
    first_match = _GGUF_CANONICAL_SPLIT_RE.match(normalized_first.rsplit("/", 1)[-1])
    if first_match is None or int(first_match.group("index")) != 1:
        return 0
    total = int(first_match.group("total"))
    if total < 2:
        return 0

    family = gguf_variant_family(normalized_first).casefold()
    indices: list[int] = []
    for filename in filenames:
        normalized = filename.replace("\\", "/")
        if gguf_variant_family(normalized).casefold() != family:
            continue
        match = _GGUF_CANONICAL_SPLIT_RE.match(normalized.rsplit("/", 1)[-1])
        if match is None or match.group("total") != first_match.group("total"):
            return 0
        indices.append(int(match.group("index")))
    return total if sorted(indices) == list(range(1, total + 1)) else 0


def gguf_checkpoint_family(filename: str) -> Optional[str]:
    """the exact checkpoint name shared by quant variants, or ``None`` when unnamed."""
    path = filename.replace("\\", "/")
    family_parts = gguf_variant_family(path).split("/")
    basename = family_parts.pop()
    family_parts = [
        segment for segment in family_parts if segment and not _is_quant_directory(segment)
    ]
    quant = quant_token_with_bpw(basename)
    if quant is not None:
        basename = re.sub(re.escape(quant), "", basename, count = 1, flags = re.IGNORECASE)
    family = "/".join((*family_parts, basename))
    return family.strip("-_. /\t\r\n") or None


def local_path_physical_identity(raw_path: str) -> str:
    """Identity for an existing local path without folding valid filename characters."""
    try:
        path = os.path.realpath(os.path.expanduser(raw_path))
        stat = os.stat(path)
        if stat.st_ino:
            return "\x00".join(("stat", str(stat.st_dev), str(stat.st_ino)))
        return f"path\x00{path}"
    except (OSError, UnicodeError, ValueError):
        return f"path\x00{os.path.abspath(os.path.expanduser(raw_path))}"


def should_group_local_gguf_files(filenames: Sequence[str]) -> bool:
    """Whether filenames are variants of one selectable checkpoint."""
    if len(filenames) == 1:
        return True

    families = {gguf_checkpoint_family(filename) for filename in filenames}
    variant_keys = [gguf_variant_key(filename) for filename in filenames]
    exact_families_by_variant: dict[str, set[str]] = {}
    for filename, variant_key in zip(filenames, variant_keys):
        exact_families_by_variant.setdefault(variant_key.casefold(), set()).add(
            gguf_variant_family(filename)
        )
    has_ambiguous_variant = any(
        len(exact_families) > 1 for exact_families in exact_families_by_variant.values()
    )
    return (
        not has_ambiguous_variant
        and (families == {None} or (None not in families and len(families) == 1))
    ) or all(is_h3_denoiser_variant_key(key) for key in variant_keys)


def suppress_grouped_gguf_file_rows(rows: Sequence[object]) -> list:
    """Drop rows already represented by a physical GGUF directory row."""
    from utils.models.model_config import _extract_quant_label as _loader_quant

    def row_path_is_dir(row) -> bool:
        try:
            return Path(getattr(row, "path", "")).is_dir()
        except OSError:
            return False

    grouped_rows = [
        (row, Path(os.path.abspath(os.path.expanduser(getattr(row, "path", "")))))
        for row in rows
        if getattr(row, "model_format", None) == "gguf"
        and not getattr(row, "partial", False)
        and row_path_is_dir(row)
    ]
    if not grouped_rows:
        return list(rows)

    paths_by_group = {
        id(row): list(iter_gguf_files(directory, recursive = True)) for row, directory in grouped_rows
    }
    physical_ancestors_by_group = {
        id(row): {
            local_path_physical_identity(str(parent))
            for parent in Path(os.path.realpath(directory)).parents
        }
        for row, directory in grouped_rows
    }
    families_by_group = {}
    for row, directory in grouped_rows:
        families_by_file = {}
        for file in paths_by_group[id(row)]:
            try:
                filename = str(file.relative_to(directory))
            except ValueError:
                filename = file.name
            if (
                is_mmproj_filename(file.name)
                or is_imatrix_filename(file.name)
                or is_mtp_drafter_path(filename)
                or is_appledouble_metadata(file)
                or is_big_endian_gguf_path(filename, _loader_quant(filename))
            ):
                continue
            file_identity = local_path_physical_identity(str(file))
            families_by_file.setdefault(file_identity, set()).add(gguf_checkpoint_family(filename))
        families_by_group[id(row)] = families_by_file

    nested_groups_to_drop = set()
    for row, directory in grouped_rows:
        families_by_file = families_by_group[id(row)]
        if not families_by_file:
            continue
        physical_ancestors = physical_ancestors_by_group[id(row)]
        for parent_row, parent_directory in grouped_rows:
            if row is parent_row:
                continue
            parent_families_by_file = families_by_group[id(parent_row)]
            parent_identity = local_path_physical_identity(getattr(parent_row, "path", ""))
            try:
                lexically_nested = bool(directory.relative_to(parent_directory).parts)
            except ValueError:
                lexically_nested = False
            matching_families = [
                families & parent_families_by_file.get(file_identity, set())
                for file_identity, families in families_by_file.items()
            ]
            outside_families = set().union(
                *(
                    families
                    for file_identity, families in parent_families_by_file.items()
                    if file_identity not in families_by_file
                )
            )
            if (
                (lexically_nested or parent_identity in physical_ancestors)
                and all(matching_families)
                and (
                    not outside_families
                    or all(families & outside_families for families in matching_families)
                )
            ):
                nested_groups_to_drop.add(id(row))
                break

    filtered = []
    for row in rows:
        if id(row) in nested_groups_to_drop:
            continue
        if getattr(row, "model_format", None) == "gguf" and not row_path_is_dir(row):
            path = getattr(row, "path", "")
            loose_path = Path(os.path.abspath(os.path.expanduser(path)))
            file_identity = local_path_physical_identity(path)
            family = gguf_checkpoint_family(Path(path).name)
            resolved_parent = Path(os.path.realpath(loose_path.parent))
            physical_parent_ancestry = {
                local_path_physical_identity(str(parent))
                for parent in (resolved_parent, *resolved_parent.parents)
            }
            for grouped_row, directory in grouped_rows:
                grouped_families = families_by_group[id(grouped_row)].get(file_identity, set())
                if not grouped_families:
                    continue
                try:
                    lexically_nested = bool(loose_path.relative_to(directory).parts)
                except ValueError:
                    lexically_nested = False
                grouped_identity = local_path_physical_identity(getattr(grouped_row, "path", ""))
                if (
                    not lexically_nested and grouped_identity not in physical_parent_ancestry
                ) or family in grouped_families:
                    break
            else:
                filtered.append(row)
                continue
            continue
        filtered.append(row)
    return filtered


def dedupe_custom_gguf_rows(rows: Sequence[object]) -> list:
    """Remove scanner overlap while preserving independently loadable GGUF checkpoints."""
    from utils.models.model_config import colocated_split_shards

    def row_path_is_dir(row) -> bool:
        try:
            return Path(getattr(row, "path", "")).is_dir()
        except OSError:
            return False

    deduped = list(rows)
    split_groups: dict[tuple[str, ...], list[tuple[object, Path, list[Path]]]] = {}
    for row in deduped:
        path = Path(getattr(row, "path", ""))
        if (
            getattr(row, "source", None) not in {"models_dir", "lmstudio"}
            or getattr(row, "model_format", None) != "gguf"
            or getattr(row, "partial", False)
            or row_path_is_dir(row)
        ):
            continue
        shards, complete = colocated_split_shards(path)
        if not complete or len(shards) < 2:
            continue
        key = tuple(local_path_physical_identity(str(shard)) for shard in shards)
        split_groups.setdefault(key, []).append((row, path, shards))

    dropped_rows: set[int] = set()
    replacement_rows: dict[int, object] = {}
    for candidates in split_groups.values():
        shards = candidates[0][2]
        first_identity = local_path_physical_identity(str(shards[0]))
        chosen = next(
            (
                row
                for row, path, _shards in candidates
                if local_path_physical_identity(str(path)) == first_identity
            ),
            None,
        )
        if chosen is None:
            continue
        for row, _path, _shards in candidates:
            if row is not chosen:
                dropped_rows.add(id(row))
        if hasattr(chosen, "size_bytes"):
            size_bytes = 0
            for shard in shards:
                try:
                    size_bytes += shard.stat().st_size
                except OSError:
                    pass
            replacement_rows[id(chosen)] = chosen.model_copy(update = {"size_bytes": size_bytes})

    deduped = [replacement_rows.get(id(row), row) for row in deduped if id(row) not in dropped_rows]

    grouped_dirs = {
        local_path_physical_identity(getattr(row, "path", ""))
        for row in deduped
        if getattr(row, "source", None) != "lmstudio"
        and getattr(row, "model_format", None) == "gguf"
        and not getattr(row, "partial", False)
        and row_path_is_dir(row)
    }
    gguf_dirs = [
        Path(getattr(row, "path", ""))
        for row in deduped
        if getattr(row, "model_format", None) == "gguf"
        and not getattr(row, "partial", False)
        and row_path_is_dir(row)
    ]
    files_by_parent: dict[str, list[Path]] = {}
    for row in deduped:
        path = Path(getattr(row, "path", ""))
        if (
            getattr(row, "source", None) == "lmstudio"
            and getattr(row, "model_format", None) == "gguf"
            and not getattr(row, "partial", False)
            and not row_path_is_dir(row)
        ):
            files_by_parent.setdefault(local_path_physical_identity(str(path.parent)), []).append(
                path
            )

    grouped_decisions = {
        parent: should_group_local_gguf_files([path.name for path in files])
        for parent, files in files_by_parent.items()
        if parent in grouped_dirs
    }
    if not grouped_decisions:
        return deduped

    grouped_paths = {
        local_path_physical_identity(str(path)): path
        for path in gguf_dirs
        if local_path_physical_identity(str(path)) in grouped_decisions
    }
    nested_groups_to_drop: set[str] = set()
    for parent_identity, parent_path in grouped_paths.items():
        nested_paths = []
        for candidate in gguf_dirs:
            if candidate == parent_path:
                continue
            try:
                candidate.relative_to(parent_path)
            except ValueError:
                continue
            nested_paths.append(candidate)
        if not nested_paths:
            continue

        filenames = [path.name for path in files_by_parent[parent_identity]]
        nested_identities = set()
        parent_visible_files = {
            local_path_physical_identity(str(file))
            for file in iter_gguf_files(parent_path, recursive = True)
        }
        for nested_path in nested_paths:
            nested_identities.add(local_path_physical_identity(str(nested_path)))
            try:
                relative_parent = nested_path.relative_to(parent_path)
                nested_files = [
                    file
                    for file in nested_path.iterdir()
                    if file.is_file()
                    and is_gguf_filename(file.name)
                    and not is_mmproj_filename(file.name)
                    and not is_imatrix_filename(file.name)
                    and not is_mtp_drafter_path(file.name)
                    and not is_appledouble_metadata(file)
                ]
            except OSError:
                grouped_decisions[parent_identity] = False
                break
            if not nested_files or any(
                local_path_physical_identity(str(file)) not in parent_visible_files
                for file in nested_files
            ):
                grouped_decisions[parent_identity] = False
                break
            filenames.extend(str(relative_parent / file.name) for file in nested_files)
        else:
            grouped_decisions[parent_identity] = should_group_local_gguf_files(filenames)
        if grouped_decisions[parent_identity]:
            nested_groups_to_drop.update(nested_identities)

    filtered = []
    for row in deduped:
        path = Path(getattr(row, "path", ""))
        is_dir = row_path_is_dir(row)
        identity = local_path_physical_identity(str(path if is_dir else path.parent))
        keep_group = grouped_decisions.get(identity)
        if getattr(row, "model_format", None) == "gguf" and (
            (
                keep_group is not None
                and ((is_dir and not keep_group) or (not is_dir and keep_group))
            )
            or (is_dir and local_path_physical_identity(str(path)) in nested_groups_to_drop)
        ):
            continue
        filtered.append(row)
    return filtered


def extract_quant_label(filename: str) -> str:
    return extract_quant_token(filename) or _unknown_gguf_variant_key(filename)


def bare_quant_alias(key: str) -> str:
    """The bare quant spelling a path-qualified *key* also answers to.

    ``weights/model-IQ4_XS-3.53bpw`` -> ``IQ4_XS-3.53bpw``. The bpw modifier is part of the
    identity now, so the three compatibility fallbacks that accept a bare name for a qualified
    key (the plan lookup, auto-download admission, deletion) have to keep it -- comparing on the
    token alone means a persisted ``IQ4_XS-3.53bpw`` resolves nothing.
    """
    basename = (key or "").replace("\\", "/").rsplit("/", 1)[-1]
    token = quant_token_with_bpw(basename)
    if token is not None:
        return token
    # Nothing in the name is a quant, so the alias is the unknown-variant spelling; hand it an extension
    # to strip rather than let it cut at the dot in "ltx-2.3".
    return extract_quant_label(f"{basename}.gguf")


def is_qualified_gguf_variant_key(key: str) -> bool:
    """Whether *key* names more than its bare quantization.

    Usually a directory (``distilled/model-Q6_K``), but H3's root-level partitions use the full
    filename stem (``minimax_h3_ref2va_pruned-Q6_K``). Comparing against the bare alias covers
    both and leaves ordinary keys such as ``Q6_K`` and ``IQ4_XS-3.53bpw`` untouched.
    """
    normalized = (key or "").strip().replace("\\", "/")
    return bool(normalized) and bare_quant_alias(normalized).lower() != normalized.lower()


def is_h3_denoiser_variant_key(key: str) -> bool:
    """Whether *key* is one of H3's root-level denoiser checkpoint identities."""
    normalized = (key or "").strip().replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1].lower()
    return basename.startswith(_H3_DENOISER_PARTITIONS) and is_qualified_gguf_variant_key(
        normalized
    )


def _is_quant_directory(segment: str) -> bool:
    """Whether a path segment names a quant (``Q6_K/``, ``Llama-3.3-70B-Instruct-Q6_K/``).

    Such a directory says only how the file was quantized, which its name already says,
    so it adds nothing to the file's identity. A directory naming something else
    (``distilled/``) is a different checkpoint and does. The basename still wins on
    which quant it is: ``Q8_0/model-Q4_K_M.gguf`` IS the Q4_K_M file.
    """
    return _select_quant_match(segment) is not None


def gguf_variant_key(filename: str) -> str:
    """The persisted identity of a selectable GGUF variant.

    The bare quant token when that token names the file within its path -- the shape
    almost every repo uses, so this is byte-identical to the historical key there and
    every stored pin, manifest and marker keeps resolving. When the token does NOT
    single the file out, because a sibling directory holds another checkpoint at the
    same quant (``distilled/`` and ``distilled-1.1/`` beside the repo root), the key
    is the file's :func:`gguf_variant_family` instead, which is unique within the
    repo and which the loader already accepts as a spelling
    (``model_config._find_local_gguf_by_variant`` matches its shard-stripped relative
    path, as does ``llama_cpp._gguf_files_for_variant``).

    A pure function of the path, deliberately: the remote listing sees a whole repo
    while a cache scan sees whatever was downloaded, and a key that depended on the
    set would disagree between them and strand a finished download as incomplete.
    """
    path = filename.replace("\\", "/")
    quant = quant_token_with_bpw(path)
    if quant is None:
        return _unknown_gguf_variant_key(path)
    # MiniMax H3 bundles two denoiser partitions and may publish full and pruned builds at one quant,
    # so the bare quant cannot identify the row, its download state, or the file to open.
    if path.rsplit("/", 1)[-1].lower().startswith(_H3_DENOISER_PARTITIONS):
        return _unknown_gguf_variant_key(path)
    parents = path.rpartition("/")[0]
    if any(segment and not _is_quant_directory(segment) for segment in parents.split("/")):
        return _unknown_gguf_variant_key(path)
    return quant


def _variant_scope_label(filename: str, *, with_stem: bool = False) -> str:
    """The part of a qualified variant's path that tells it apart from its namesakes.

    The directory alone reads best and is enough for the usual shape, where each checkpoint
    has its own. ``with_stem`` adds the filename back for the case where it is not: two
    checkpoints in ONE directory at one quant are two rows, and a label naming only the
    directory would print the same text on both.
    """
    parents = filename.replace("\\", "/").rpartition("/")[0].strip("/")
    stem = _gguf_stem(filename)
    if not parents:
        return stem
    return f"{parents}/{stem}" if with_stem else parents


def _apply_gguf_display_labels(variants: list[GgufVariantInfo]) -> None:
    unknown_variants = [
        variant for variant in variants if extract_quant_token(variant.filename) is None
    ]
    ambiguous = len(unknown_variants) > 1

    # The bpw modifier reads perfectly well on its own, so a key that is only the token plus its bpw
    # suffix is NOT path-qualified and needs no scope label.
    def _plain_key(variant) -> Optional[str]:
        return quant_token_with_bpw(variant.filename)

    qualified = [
        variant
        for variant in variants
        if (plain := _plain_key(variant)) is not None and variant.quant.lower() != plain.lower()
    ]
    # A scope shared by two rows does not tell them apart, so those rows show the file too.
    scopes: dict[str, int] = {}
    for variant in qualified:
        scope = _variant_scope_label(variant.filename).lower()
        scopes[scope] = scopes.get(scope, 0) + 1
    for variant in variants:
        token = extract_quant_token(variant.filename)
        h3_name = Path(variant.filename).name.lower()
        h3_partition = next(
            (partition for partition in _H3_DENOISER_PARTITIONS if h3_name.startswith(partition)),
            None,
        )
        if h3_partition is not None:
            task = "References" if h3_partition.endswith("ref2va") else "Text & frames"
            build = "Pruned" if "_pruned-" in h3_name else "Full"
            variant.display_label = f"{task} · {token or 'GGUF'} · {build}"
            continue
        if token is None:
            variant.display_label = f"GGUF · {variant.filename}" if ambiguous else "GGUF"
        elif variant.quant.lower() != (_plain_key(variant) or "").lower():
            # A key qualified by path: show the quant, plus what distinguishes it.
            collides = scopes.get(_variant_scope_label(variant.filename).lower(), 0) > 1
            variant.display_label = (
                f"{token} · {_variant_scope_label(variant.filename, with_stem = collides)}"
            )


def group_gguf_variant_files(entries) -> dict[str, tuple[str, int]]:
    """``variant key -> (first filename, size of that variant's shard family)``.

    *entries* is an iterable of ``(path, size)`` for main GGUFs only, already filtered
    of mmproj, drafters and big-endian builds.

    Sizes are summed across the shards of ONE family, never across families. A repo
    that ships the same quant twice (``BF16/QwQ-32B-BF16-*`` beside
    ``BF16/QwQ-32B.BF16-*``) therefore advertises what a load would actually read
    rather than the total of both copies. The family kept is the one holding the
    lexicographically first file, which is the shard the lister and the loader open.
    """
    families: dict[str, dict[str, list[tuple[str, int]]]] = {}
    for path, size in entries:
        families.setdefault(gguf_variant_key(path), {}).setdefault(
            gguf_variant_family(path), []
        ).append((path, int(size or 0)))
    grouped: dict[str, tuple[str, int]] = {}
    for key, by_family in families.items():
        chosen = min(by_family.values(), key = lambda members: min(path for path, _ in members))
        grouped[key] = (min(path for path, _ in chosen), sum(size for _, size in chosen))
    return grouped


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
        # is_dir() ignores only ENOENT/ENOTDIR/EBADF/ELOOP, so an unreadable root raised EACCES up to 3.13
        # (3.14 returns False, gh-101357); skip it instead of 500ing.
        try:
            if not snapshots_dir.is_dir():
                continue
            snapshots.extend(snap for snap in snapshots_dir.iterdir() if snap.is_dir())
        except OSError as e:
            logger.debug("Skipping unreadable cache snapshots dir %s: %s", snapshots_dir, e)
            continue

    # Same key the inventory row selects with, so both name one snapshot.
    snapshots.sort(key = snapshot_selection_key, reverse = True)
    yield from snapshots


def list_empty_gguf_variant_dirs(repo_id: str, root: Optional[Path] = None) -> set[str]:
    """Quant labels present only as an EMPTY snapshot ``<quant>/`` folder (an
    interrupted split download); a quant with shards in any snapshot is excluded.

    Labelled the way a row is keyed, bpw modifier included, or an empty ``IQ4_XS-3.53bpw/``
    folder would be reported against a bare ``IQ4_XS`` row that does not exist while the row
    it belongs to reads as absent."""
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
                quant = quant_token_with_bpw(sub.name)
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


def select_gguf_cache_snapshot(
    repo_id: str, root: Optional[Path] = None
) -> Optional[tuple[list[GgufVariantInfo], bool, set, Path]]:
    """``list_gguf_variants_from_hf_cache`` plus the snapshot it answered from.

    A repo dir holds every revision, so a caller that then reads metadata from wherever this
    listing came from needs the snapshot, not the dir: the dir includes revisions it skipped.
    """
    # Local import: inventory_scan imports this module.
    from hub.utils.inventory_scan import complete_snapshot_variants

    snapshots = (
        iter_hf_cache_snapshots(repo_id, root = root)
        if root is not None
        else iter_hf_cache_snapshots(repo_id)
    )
    # Pick the snapshot the inventory row does: newest holding a whole quant, else first non-empty.
    fallback: Optional[tuple[list[GgufVariantInfo], bool, set, Path]] = None
    for snapshot in snapshots:
        variants, has_vision = list_local_gguf_variants(str(snapshot))
        complete = complete_snapshot_variants(str(snapshot)) if variants else set()
        if variants:
            # Selection only: an unlabelled quant cannot be judged, so it counts as usable.
            if any(not v.quant or v.quant in complete for v in variants):
                return variants, has_vision, complete, snapshot
        if fallback is None and (variants or has_vision):
            fallback = (variants, has_vision, complete, snapshot)
    return fallback


def merge_sibling_snapshot_variants(
    repo_id: str,
    selected: tuple[list[GgufVariantInfo], bool, set, Path],
    root: Optional[Path] = None,
) -> tuple[list[GgufVariantInfo], bool, set, Path]:
    """*selected* widened with the quants the repo dir's other revisions hold.

    The selector names one directory, so an inventory built on it drops what an upstream re-upload
    left elsewhere. Only safe for a row loading by repo id, which the caller establishes. One repo
    dir only: a cache can hold two differing in case, and the row owns one.
    """
    from hub.utils.hf_cache_state import same_existing_path
    from hub.utils.inventory_scan import complete_snapshot_variants

    variants, has_vision, complete, snapshot = selected
    merged = list(variants)
    merged_complete = set(complete)
    whole = {quant.lower() for quant in merged_complete}
    held: dict[str, int] = {}
    for index, variant in enumerate(merged):
        if variant.quant:
            held.setdefault(variant.quant.lower(), index)
    # Per row, so a projector cannot outlive the row it arrived with.
    merged_vision: dict[str, bool] = {}
    changed = False
    for other in iter_hf_cache_snapshots(repo_id, root = root):
        if other.parent != snapshot.parent:
            continue
        if other == snapshot or same_existing_path(other, snapshot):
            continue
        extra, extra_vision = list_local_gguf_variants(str(other))
        candidates = [v for v in extra if v.quant and v.quant.lower() not in whole]
        if not candidates:
            continue
        other_complete = complete_snapshot_variants(str(other)) or set()
        for variant in candidates:
            key = variant.quant.lower()
            index = held.get(key)
            is_whole = variant.quant in other_complete
            if index is None:
                held[key] = len(merged)
                merged.append(variant)
            elif is_whole:
                # A load skips the torn copy for this one, so the row describes this one.
                merged[index] = variant
            else:
                continue
            changed = True
            merged_vision[key] = extra_vision
            if is_whole:
                merged_complete.add(variant.quant)
                whole.add(key)
    if changed:
        # Labels disambiguate within a revision, so a merged set can hold names only the merge brings together.
        _apply_gguf_display_labels(merged)
    return merged, has_vision or any(merged_vision.values()), merged_complete, snapshot


def list_gguf_variants_from_hf_cache(
    repo_id: str, root: Optional[Path] = None
) -> Optional[tuple[list[GgufVariantInfo], bool, set]]:
    """``(variants, has_vision, complete)`` for the snapshot a load would read.

    Everything in that snapshot is listed, so a torn download stays visible to resume or delete;
    *complete* is the subset whose shards are all present, so the caller marks the rest partial
    rather than ready, as the snapshot-path form of this call does.
    """
    selected = select_gguf_cache_snapshot(repo_id, root = root)
    return selected[:3] if selected is not None else None


def _is_state_filename_fallback(variant: str, path: Path) -> bool:
    """Whether *variant* was read off *path*'s own name rather than out of it.

    An unreadable payload leaves the reader the filename, whose fragment for an
    unspellable variant is a digest. Spelling cannot tell that from a variant
    genuinely called ``sha256-<32 hex>``, but the file can: a real one is stored
    under the hash of itself, never under its own name. A recovered digest names
    nothing -- it cannot be spelled back, and a resume would re-key it again.
    """
    from hub.utils.state_dir import variant_is_hashed_fragment
    return variant_is_hashed_fragment(variant) and path.stem.lower().endswith(
        f"--variant--{variant.strip().lower()}"
    )


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

    # Variant identity on disk is case-insensitive, so dedupe on the lowercased key; manifests are read
    # first to keep their original-casing label over a lowercased cancel marker.
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
        for variant, path in source:
            if _is_state_filename_fallback(variant, path):
                continue
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
        main_filenames: list[str] = []
        size_bytes = 0
        companion_bytes = 0
        imatrix_only = False
        if manifest is not None:
            for expected in manifest.expected_files:
                if not is_gguf_filename(expected.path):
                    continue
                if is_imatrix_filename(expected.path):
                    # A manifest predating this filtering can still name one: it is neither weights nor a companion, so
                    # it counts towards neither size.
                    imatrix_only = True
                    continue
                if is_mtp_drafter_path(expected.path):
                    # Downloaded with every variant like mmproj but not a selectable quant; counted so the shown
                    # download size matches what is fetched.
                    companion_bytes += max(0, int(expected.size or 0))
                    continue
                if is_mmproj_filename(expected.path):
                    has_vision = True
                    companion_bytes += max(0, int(expected.size or 0))
                    continue
                if main_filename is None:
                    main_filename = expected.path
                main_filenames.append(expected.path)
                size_bytes += max(0, int(expected.size or 0))
        if main_filename is None:
            # An older build could download the imatrix as a variant of its own, so naming the synthetic file
            # after the variant would put that interrupted row back in the menu at zero bytes. Only when
            # NOTHING eligible was found.
            if imatrix_only or is_imatrix_filename(variant):
                continue
            main_filename = f"{variant}.gguf"
        variants.append(
            GgufVariantInfo(
                filename = main_filename,
                quant = variant,
                size_bytes = size_bytes,
                download_size_bytes = size_bytes + companion_bytes,
                shard_count = complete_gguf_shard_count(main_filenames, main_filename),
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

    has_vision = False
    main_files: list[tuple[str, int]] = []

    # Sidecars in the listing would otherwise be advertised as the variant, at their own size.
    for sibling in drop_shadowed_appledouble_siblings(list(info.siblings)):
        filename = getattr(sibling, "rfilename", None)
        if not isinstance(filename, str) or not is_gguf_filename(filename):
            continue
        if not _is_selectable_repo_gguf(repo_id, filename):
            continue
        if is_mtp_drafter_path(filename) or is_imatrix_filename(filename):
            continue
        if is_mmproj_filename(filename):
            has_vision = True
            continue
        # The two extractors disagree on F16-be-checkpoint-Q4_K_M shapes; judge with the loader's label so
        # no row is advertised for a file the detector refuses.
        # The two extractors disagree on F16-be-checkpoint-Q4_K_M shapes; judge with the loader's label so no
        # row is listed for a file the local detector refuses.
        from utils.models.model_config import _extract_quant_label as _loader_quant

        if is_big_endian_gguf_path(filename, _loader_quant(filename)):
            continue
        main_files.append((filename, int(getattr(sibling, "size", 0) or 0)))

    variants = [
        GgufVariantInfo(
            filename = filename,
            quant = quant,
            size_bytes = size,
            shard_count = complete_gguf_shard_count([path for path, _size in main_files], filename),
        )
        for quant, (filename, size) in group_gguf_variant_files(main_files).items()
    ]

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

    main_files: list[tuple[str, int]] = []
    has_vision = False
    # Match the cache dir of ANY H3 bundle repo: the aggregation runs over whichever mirror the user
    # actually downloaded.
    root_key = root.as_posix().lower()
    h3_bundle_repo = next(
        (r for r in _H3_BUNDLE_REPOS if f"models--{r.replace('/', '--')}" in root_key), None
    )

    for file in sorted(iter_gguf_files(root, recursive = True)):
        if h3_bundle_repo and not _is_selectable_repo_gguf(h3_bundle_repo, file.name):
            continue
        if is_imatrix_filename(file.name):
            continue
        if is_mmproj_filename(file.name):
            # Header metadata distinguishes vision projectors from audio-only ones, read only when Windows
            # reports the file fully present: opening a cloud placeholder would recall it during discovery.
            try:
                info = file.stat()
                has_vision = has_vision or (
                    info.st_size > 0
                    and file_contents_available_locally(file, info)
                    and mmproj_accepts_image(str(file))
                )
            except OSError:
                pass
            continue
        try:
            size = file.stat().st_size
        except OSError:
            size = 0
        rel = file.relative_to(root).as_posix()
        if _is_local_mtp_drafter(file, custom_root, rel):
            continue
        from utils.models.model_config import _extract_quant_label as _loader_quant

        if is_big_endian_gguf_path(rel, _loader_quant(rel)):
            continue
        main_files.append((rel, size))

    variants = [
        GgufVariantInfo(
            filename = filename,
            quant = quant,
            size_bytes = size,
            shard_count = complete_gguf_shard_count([path for path, _size in main_files], filename),
        )
        for quant, (filename, size) in group_gguf_variant_files(main_files).items()
    ]
    variants.sort(key = lambda variant: -variant.size_bytes)
    _apply_gguf_display_labels(variants)
    return variants, has_vision
