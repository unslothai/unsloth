# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF filename rules: quant labels, projectors, imatrix files and separate drafters.

The single owner for hub, utils.models and core.inference, so it imports only the standard library.
"""

from __future__ import annotations

import re
from typing import Optional

GGUF_QUANT_RE = re.compile(
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


def is_mmproj_filename(filename: str) -> bool:
    return "mmproj" in filename.lower()


# Anchored at an end of the stem: Qwen3-Imatrix-Tuned-Q4_K_M.gguf is a real model, while every
# published imatrix leads or closes with the token.
IMATRIX_TOKEN_RE = re.compile(r"^imatrix(?:[._\-]|$)|[._\-]imatrix$", re.IGNORECASE)
# A quant label before the trailing token marks imatrix-weighted weights. Float labels and
# `.imatrix` do not count: mradermacher names its calibration file `<model>.imatrix.gguf`.
IMATRIX_WEIGHTED_RE = re.compile(
    r"[._\-](?:I?Q[0-9]+_|P?TQ[0-9]+_|(?:MX|NV)FP[0-9])[^/]*[_\-]imatrix$", re.IGNORECASE
)


def is_imatrix_filename(path: str) -> bool:
    """True for a calibration imatrix (``imatrix_unsloth.gguf``, ``<model>-imatrix.gguf``,
    ``<model>.imatrix``), never for imatrix-weighted ``<model>-Q4_K_M-imatrix.gguf``. A GGUF one
    is a valid container, so a size-ranked scan would otherwise hand it to llama-server."""
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    if IMATRIX_TOKEN_RE.search(stem) and not IMATRIX_WEIGHTED_RE.search(stem):
        return True
    return name.lower().endswith(".imatrix")


# Separate-file drafters, by basename prefix. dspark and dflash are the same DeepSeek V4 Flash
# drafter: the folder it ships in and the architecture it reports.
DRAFTER_KINDS = ("mtp", "dspark", "dflash", "eagle3", "lilicorr")

# mtp/ and dspark/ are always a companion folder; dflash/ is a family name users pick for weights.
DRAFTER_DIR_KINDS = ("mtp", "dspark")

# <model>-dspark-<quant>.gguf drafters (prism-ml Bonsai, third-party DSpark GGUFs); a name going on
# past the quant (DeepSeek-V4-Flash-Dspark-Abliterated-MXFP4-BF16.gguf) is the model.
DSPARK_BEFORE_QUANT_RE = re.compile(
    rf"-dspark-(?:{GGUF_QUANT_RE.pattern})(?:-[0-9]{{5}}-of-[0-9]{{5}})?\.gguf$",
    re.IGNORECASE,
)


def drafter_path_kind(path: str) -> Optional[str]:
    """The drafter kind ``path`` names by basename prefix or exact parent directory, or None.

    Never a substring: kind names are also family names (``Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf``)."""
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
    """True for a separate-file drafter rather than a selectable quant: ``mtp-*.gguf``, the
    ``MTP/`` and ``dspark/`` copies, ``<model>-dspark-<quant>.gguf``, ``eagle3-*.gguf`` and
    ``lilicorr-*.gguf``. Excluded wherever mmproj is, or a ``Q8_0`` request can resolve to
    ``MTP/...-Q8_0-MTP.gguf``."""
    if drafter_path_kind(path) is not None:
        return True
    p = path.replace("\\", "/").lower()
    return p.endswith(".gguf") and DSPARK_BEFORE_QUANT_RE.search(p.rsplit("/", 1)[-1]) is not None
