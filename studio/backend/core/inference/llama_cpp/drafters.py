# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MTP, DSpark and DFlash drafter detection."""

import re

from pathlib import Path
from typing import Optional

# Gemma 4 ships MTP as a separate drafter (no "-mtp" in the name). Gemma 3n
# ships no drafter, so it is excluded -- it takes the normal non-MTP path.
_GEMMA_MTP_FAMILY_RE = re.compile(r"gemma[-_]?4[-_]", re.IGNORECASE)


def _is_gemma_mtp_family(name: Optional[str]) -> bool:
    """Match Gemma 4 by name."""
    return bool(name) and bool(_GEMMA_MTP_FAMILY_RE.search(name))


def _is_gemma_mtp_name(model_identifier: Optional[str], gguf_path: Optional[str] = None) -> bool:
    """Match Gemma 4 by id or GGUF filename."""
    return _is_gemma_mtp_family(model_identifier) or _is_gemma_mtp_family(
        Path(gguf_path).name if gguf_path else None
    )


def _is_mtp_model_name(model_identifier: Optional[str], gguf_path: Optional[str] = None) -> bool:
    """Name-based MTP detector. Fallback for the metadata signal."""
    for cand in (model_identifier, Path(gguf_path).name if gguf_path else None):
        if cand and "-mtp" in cand.lower():
            return True
        # Recognise Gemma 4 too, so a failed drafter download surfaces a
        # fallback reason instead of silently defaulting.
        if cand and _is_gemma_mtp_family(cand):
            return True
    return False


# Mirrors hub.utils.gguf._DRAFTER_KINDS / _DRAFTER_DIR_KINDS.
_DRAFTER_KINDS = ("mtp", "dspark", "dflash", "eagle3", "lilicorr")
_DRAFTER_DIR_KINDS = ("mtp", "dspark")

# Human label per resolved spec mode, for launch logging and the UI notice.
_DRAFTER_DISPLAY_LABELS = {"dspark": "DSpark", "dflash": "DFlash", "eagle3": "EAGLE-3"}


def _drafter_path_kind(path: str) -> Optional[str]:
    """Drafter kind naming *path*: basename prefix, or exact parent dir for
    ``_DRAFTER_DIR_KINDS``. Never a substring: the kind names double as family
    names, so ``Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf`` IS the model, as is anything
    in a user's ``dflash/`` folder."""
    p = path.replace("\\", "/").lower()
    if not p.endswith(".gguf"):
        return None
    parts = [segment for segment in p.split("/") if segment]
    name, parents = parts[-1], parts[:-1]
    for kind in _DRAFTER_KINDS:
        if name.startswith(f"{kind}-"):
            return kind
    for kind in _DRAFTER_DIR_KINDS:
        if kind in parents:
            return kind
    return None


_IMATRIX_TOKEN_RE = re.compile(r"^imatrix(?:[._\-]|$)|[._\-]imatrix$", re.IGNORECASE)


def _is_companion_gguf_path(path: str) -> bool:
    """True for a non-main GGUF: vision mmproj, a calibration imatrix, or a separate
    drafter (repo-root ``mtp-*.gguf``, the ``MTP/`` subdir copies for Gemma 4, the ``dspark/``
    drafters for DeepSeek V4 Flash, the ``<model>-dspark-<quant>.gguf`` drafters for
    prism-ml Bonsai, the ``eagle3-*.gguf`` draft heads). Mirrors
    hub.utils.gguf so variant resolution never picks a companion as the main model --
    a Gemma ``Q8_0`` request must not resolve to ``MTP/...-Q8_0-MTP.gguf``, which sorts
    ahead of the real weight.

    EXCLUSION ONLY. Use ``_is_mtp_only_drafter_path`` to pick a drafter to launch.
    """
    p = path.lower()
    if not p.endswith(".gguf"):
        return False
    if "mmproj" in p:
        return True
    # Mirrors hub.utils.gguf.is_imatrix_filename: an imatrix is a valid GGUF container
    # holding calibration statistics, so only the name rules it out.
    if _IMATRIX_TOKEN_RE.search(Path(p).stem):
        return True
    if _DSPARK_BEFORE_QUANT_RE.search(Path(p).name):
        return True
    return _drafter_path_kind(path) is not None


def _is_mtp_only_drafter_path(path: str) -> bool:
    """MTP only, not the other companion GGUFs. Each drafter kind needs its own
    ``--spec-type``, so discovery narrows to one kind rather than reusing the
    broad companion predicate, which exists to EXCLUDE companions from menus."""
    return _drafter_path_kind(path) == "mtp"


def _is_dspark_drafter_path(path: str) -> bool:
    """True for a DSpark sidecar, excluding the separate DFlash method.

    Broader than model_config.detect_dspark_file, which additionally requires a
    published drafter NAME: a repo's ``dspark/`` directory holds only sidecars,
    while a user's local folder may hold anything. Only ever used against repo
    listings and cache snapshots for that reason.
    """
    return _drafter_path_kind(path) == "dspark"


def _is_dflash_drafter_path(path: str) -> bool:
    """True for a DFlash sidecar, excluding the separate DSpark method.

    Prefix only: ``dflash`` doubles as a family name a publisher puts on real
    weights, so ``Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf`` IS the model, and a
    user's ``dflash/`` folder holds whatever they put there (``dflash`` is
    absent from ``_DRAFTER_DIR_KINDS`` for that reason). Only a root-level
    ``dflash-*.gguf`` names a sidecar.

    Broader than model_config.detect_dflash_file, which additionally confirms
    ``general.architecture = dflash`` in the header. Only ever used against repo
    listings and cache snapshots, where a header read is not available.
    """
    return _drafter_path_kind(path) == "dflash"


def _is_root_dflash_drafter_path(path: str) -> bool:
    """A DFlash sidecar as a repo listing or cache snapshot offers one: the
    ``dflash-`` prefix AND the repository root.

    _is_dflash_drafter_path tests the basename, which is all its other callers
    have (a bare filename carries no parent to look at). Remote discovery does
    have the repository-relative path, and needs it: the local contract is root
    level only, so ``quants/dflash-model-Q8_0.gguf`` is an ordinary weight that
    detect_dflash_file would never offer. Going by basename there made it a
    candidate, and since the header can only be read once the bytes are here,
    the whole weight was downloaded before the rejection.

    Checked here rather than inside the basename predicate so the prefix-only
    naming rule stays exactly as it is for the callers that share it.
    """
    normalized = path.replace("\\", "/")
    return "/" not in normalized and _is_dflash_drafter_path(normalized)


_BIG_ENDIAN_GGUF_FILENAME_RE = re.compile(r"(^|[-_])be(?:[._-]|$)", re.IGNORECASE)
_GGUF_KNOWN_QUANT_RE = re.compile(
    r"(UD-)?"
    r"((?:MX|NV)FP[0-9]+(?:_[A-Z0-9]+)*"
    r"|IQ[0-9]+_[A-Z]+(?:_[A-Z0-9]+)?"
    r"|P?TQ[0-9]+_[0-9]+"
    r"|Q[0-9]+_K_[A-Z]+"
    r"|P?Q[0-9]+_(?:[0-9]+(?:_G[0-9]+)?|G[0-9]+)"
    r"|Q[0-9]+_K"
    r"|BF16|F16|F32)",
    re.IGNORECASE,
)
# Mirrors hub.utils.gguf._DSPARK_BEFORE_QUANT_RE, for _is_companion_gguf_path.
_DSPARK_BEFORE_QUANT_RE = re.compile(
    rf"-dspark-(?:{_GGUF_KNOWN_QUANT_RE.pattern})(?:-[0-9]{{5}}-of-[0-9]{{5}})?\.gguf$",
    re.IGNORECASE,
)


def _is_big_endian_gguf_path(path: str, variant_key: str = "") -> bool:
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0].lower()
    variant_key = variant_key.strip().lower()
    variant_index = stem.find(variant_key) if variant_key else -1
    parent = normalized.rsplit("/", 1)[0].lower() if "/" in normalized else ""
    variant_in_parent_only = (
        bool(parent)
        and variant_index < 0
        and (
            (variant_key and variant_key in parent)
            or (not variant_key and _GGUF_KNOWN_QUANT_RE.search(parent) is not None)
        )
    )
    for match in _BIG_ENDIAN_GGUF_FILENAME_RE.finditer(stem):
        if variant_index >= 0 and variant_index < match.start():
            return True
        tail = stem[match.end() :].lstrip("._-")
        if not tail or _GGUF_KNOWN_QUANT_RE.search(tail) is None:
            return not variant_in_parent_only
    return False
