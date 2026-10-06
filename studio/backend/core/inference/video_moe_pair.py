# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pair the two expert files of a dual-expert (Wan2.2-A14B) video DiT by name.

Single-file distributions ship each expert as its own checkpoint, differing only in a noise
token: ComfyUI's ``wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors`` / ``..._low_noise_...``,
or ``HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf`` / ``LowNoise/...-LowNoise-...``. Picking
either one names both: the partner is the same path with every high token swapped for low (or
the reverse), case kept. The high-noise expert is the pipeline's ``transformer`` (it runs the
early, high-noise steps), the low-noise one its ``transformer_2``, as in the diffusers layout and
Studio's hosted per-expert checkpoints.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

# "high_noise", "high-noise", "high noise", "highnoise" / "HighNoise": the noise word follows directly.
_EXPERT_RE = re.compile(r"(high|low)([_\- ]?noise)", re.IGNORECASE)


def _swap_case(word: str, like: str) -> str:
    if like.isupper():
        return word.upper()
    if like[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def moe_expert_of(filename: Optional[str]) -> Optional[str]:
    """``"high"`` or ``"low"`` for an expert file name, else None (no token, or both kinds)."""
    if not filename:
        return None
    kinds = {m.group(1).lower() for m in _EXPERT_RE.finditer(str(filename))}
    return kinds.pop() if len(kinds) == 1 else None


def moe_partner_filename(filename: Optional[str]) -> Optional[str]:
    """The other expert's path (folder components included), or None when the name pairs nothing."""
    expert = moe_expert_of(filename)
    if expert is None:
        return None
    other = "low" if expert == "high" else "high"
    return _EXPERT_RE.sub(lambda m: _swap_case(other, m.group(1)) + m.group(2), str(filename))


def moe_expert_pair(filename: Optional[str]) -> Optional[tuple[str, str]]:
    """``(high, low)`` file names for either expert's name, or None."""
    partner = moe_partner_filename(filename)
    if partner is None:
        return None
    return (
        (str(filename), partner) if moe_expert_of(filename) == "high" else (partner, str(filename))
    )


def moe_pick_pairs(fam: Any, *names: Optional[str]) -> bool:
    """Whether a single-file / GGUF pick of ``fam`` is loadable: any single-DiT family, or a dual-expert
    one whose name pairs its partner expert."""
    if not getattr(fam, "is_moe", False):
        return True
    return any(moe_partner_filename(Path(str(n)).name if n else None) for n in names if n)
