# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Standard-library-only GGUF filename rules shared by discovery and loading."""

from __future__ import annotations

import re
from typing import Optional

# NVFP is the quant only last in the name (Model-NVFP4.gguf, NVFP4/); mid-name it is part of a
# community model name (Qwen3.8-27B-NVFP4-MTP-HIGH). MXFP predates NVFP, so it matches anywhere.
GGUF_QUANT_RE = re.compile(
    r"(UD-)?"
    r"(MXFP[0-9]+(?:_[A-Z0-9]+)*"
    r"|NVFP[0-9]+(?:_[A-Z0-9]+)*"
    r"(?=(?:-[0-9]+(?:[.][0-9]+)?bpw)?(?:-[0-9]{3,}-of-[0-9]{3,})?(?:[.]gguf)?$)"
    r"|IQ[0-9]+_[A-Z]+(?:_[A-Z0-9]+)?"
    r"|P?TQ[0-9]+_[0-9]+"
    r"|Q[0-9]+_K_[A-Z]+"
    r"|P?Q[0-9]+_(?:[0-9]+(?:_G[0-9]+)?|G[0-9]+)"
    r"|Q[0-9]+_K"
    r"|BF16|F16|F32)",
    re.IGNORECASE,
)


def is_mmproj_filename(filename: str) -> bool:
    return "mmproj" in filename.lower()


IMATRIX_TOKEN_RE = re.compile(r"^imatrix(?:[._\-]|$)|[._\-]imatrix$", re.IGNORECASE)
# A quant label before the trailing token marks imatrix-weighted weights. Float labels and
# `.imatrix` do not count: mradermacher names its calibration file `<model>.imatrix.gguf`.
IMATRIX_WEIGHTED_RE = re.compile(
    r"[._\-](?:I?Q[0-9]+_|P?TQ[0-9]+_|(?:MX|NV)FP[0-9])[^/]*[_\-]imatrix$", re.IGNORECASE
)


def is_imatrix_filename(path: str) -> bool:
    """Exclude calibration files, but retain imatrix-weighted models."""
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    if IMATRIX_TOKEN_RE.search(stem) and not IMATRIX_WEIGHTED_RE.search(stem):
        return True
    return name.lower().endswith(".imatrix")


DRAFTER_KINDS = ("mtp", "dspark", "dflash", "eagle3", "lilicorr")

# mtp/ and dspark/ are always a companion folder; dflash/ is a family name users pick for weights.
DRAFTER_DIR_KINDS = ("mtp", "dspark")

# Names continuing beyond the quant may identify real models.
DSPARK_BEFORE_QUANT_RE = re.compile(
    rf"-dspark-(?:{GGUF_QUANT_RE.pattern})(?:-[0-9]{{5}}-of-[0-9]{{5}})?\.gguf$",
    re.IGNORECASE,
)


def drafter_path_kind(path: str) -> Optional[str]:
    """Match exact directories or basename prefixes, never family-name substrings."""
    p = path.replace("\\", "/").lower()
    if not p.endswith(".gguf"):
        return None
    parts = [segment for segment in p.split("/") if segment]
    name, parents = parts[-1], parts[:-1]
    for kind in DRAFTER_KINDS:
        if name.startswith(f"{kind}-"):
            return kind
    for kind in DRAFTER_DIR_KINDS:
        if kind in parents:
            return kind
    return None


def is_mtp_drafter_path(path: str) -> bool:
    if drafter_path_kind(path) is not None:
        return True
    p = path.replace("\\", "/").lower()
    return p.endswith(".gguf") and DSPARK_BEFORE_QUANT_RE.search(p.rsplit("/", 1)[-1]) is not None
